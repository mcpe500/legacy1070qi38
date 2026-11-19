#include "engine.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include <cuda_runtime.h>

#include "cpu_ops.hpp"
#include "gpu_layers.hpp"
#include "ternary_cpu.hpp"
#include "dn_layer.hpp"  // model dimension constants only (kHidden, kFFN, ...)

namespace {

constexpr int kLayers = 64;
constexpr int kNH = 24, kNKV = 4, kHD = 256, kRope = 64;
constexpr int kQW = kNH * kHD;       // 6144 values of q per token
constexpr int kKVW = kNKV * kHD;     // 1024 values of k (and of v) per token
// ctx_max hard caps. Ternary Bonsai (all weights 1.6-2 bit, every layer can sit in VRAM) reaches 128K positions with the
// iso3 KV (12.5 KiB per position over 16 attention layers) and the tiled attention kernel above 8192. The Qwen K-quant
// path keeps 8192: its CPU attention layers hold FP32 KV (8 MiB per position over 8 CPU attention layers at 32 GPU layers).
constexpr int kMaxCtxTernary = 131072;
constexpr int kMaxCtxQwen = 8192;
// Host RAM budget of the brief: 14.2 GB (decimal, = 13.22 GiB). Checked at load against the arena, the resident CPU-layer
// weights, the mlocked LM head / MTP block and a fixed margin for the process (tokenizer, generator buffers, CUDA pinned).
constexpr double kHostBudgetBytes = 14.2e9;
constexpr double kHostMarginBytes = 512.0 * 1024 * 1024;
constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
constexpr int kQG = kNH / kNKV;  // 6 query heads share one K/V group: their score and V passes read each row once

uint64_t u64(const GgufIndex & ix, const char * k) {
    auto it = ix.u_kv.find(k);
    if (it == ix.u_kv.end()) throw std::runtime_error(std::string("gguf: missing key ") + k);
    return it->second;
}

void expect(const GgufIndex & ix, const char * k, uint64_t v) {
    const uint64_t got = u64(ix, k);
    if (got != v) throw std::runtime_error(std::string("gguf: ") + k + " = " + std::to_string(got) +
                                           ", engine is locked to " + std::to_string(v));
}

const float * f32p(const GgufIndex & ix, const std::string & name, int64_t n) {
    const GgufTensor & t = ix.get(name);
    if (t.type != 0 /* F32 */ || t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3] != n)
        throw std::runtime_error("gguf: expected F32 tensor with " + std::to_string(n) + " values: " + name);
    return reinterpret_cast<const float *>(ix.data(t));
}

// Reads MemAvailable without touching the heap (raw read into a stack buffer), so the token loop stays allocation free.
double mem_available_gib() {
    char buf[4096];
    const int fd = ::open("/proc/meminfo", O_RDONLY);
    if (fd < 0) return 1e9;
    const ssize_t n = ::read(fd, buf, sizeof(buf) - 1);
    ::close(fd);
    if (n <= 0) return 1e9;
    buf[n] = 0;
    const char * p = std::strstr(buf, "MemAvailable:");
    if (!p) return 1e9;
    return std::strtod(p + 13, nullptr) * 1024.0 / kGiB;
}

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
inline float sigm(float x) { return 1.0f / (1.0f + std::exp(-x)); }

}  // namespace

Engine::Engine(const EngineConfig & cfg) : cfg_(cfg) {
    if (cfg_.ctx_max < 16 || cfg_.ctx_max > kMaxCtxTernary)
        throw std::runtime_error("ctx_max must be in 16 .. " + std::to_string(kMaxCtxTernary));
    qmat_init();
    ix.open(cfg_.gguf);

    // The engine is locked to the qwen35 architecture: 65 blocks on Qwen3.8-27B (64 layers + the MTP head), 64 on
    // Ternary Bonsai 2 27B (no MTP block). A missing blk.64 disables the draft head automatically; --pld is unaffected.
    const uint64_t nblocks = u64(ix, "qwen35.block_count");
    if (nblocks != 65 && nblocks != 64) throw std::runtime_error("gguf: qwen35.block_count = " + std::to_string(nblocks) + ", engine supports 64 (Bonsai, no MTP) or 65 (Qwen)");
    if (nblocks == 64 && cfg_.mtp_steps > 0)
        std::fprintf(stderr, "mtp: blk.64 absent (Ternary Bonsai), draft head disabled; --pld stays available\n");
    expect(ix, "qwen35.embedding_length", kHidden);
    expect(ix, "qwen35.feed_forward_length", kFFN);
    expect(ix, "qwen35.attention.head_count", kNH);
    expect(ix, "qwen35.attention.head_count_kv", kNKV);
    expect(ix, "qwen35.attention.key_length", kHD);
    expect(ix, "qwen35.rope.dimension_count", kRope);
    expect(ix, "qwen35.ssm.conv_kernel", kConvK);
    expect(ix, "qwen35.ssm.state_size", kHeadDim);
    expect(ix, "qwen35.ssm.group_count", kQKHeads);
    expect(ix, "qwen35.ssm.time_step_rank", kVHeads);
    expect(ix, "qwen35.ssm.inner_size", kInner);
    expect(ix, "qwen35.full_attention_interval", 4);
    eps_ = ix.f32_kv.at("qwen35.attention.layer_norm_rms_epsilon");
    if (ix.f32_kv.at("qwen35.rope.freq_base") != 10000000.0f) throw std::runtime_error("rope freq_base is not 1e7");

    tok.init(ix);
    n_vocab = static_cast<int64_t>(tok.n_vocab());
    pool_ = std::make_unique<Pool>(cfg_.threads);

    tok_embd_ = make_qmat(ix, "token_embd.weight");
    lm_head_ = make_qmat(ix, "output.weight");
    if (lm_head_.out != n_vocab) throw std::runtime_error("output.weight rows differ from the vocabulary size");
    out_norm_ = f32p(ix, "output_norm.weight", kHidden);

    // Ternary Bonsai auto-detect: any tensor of ggml type 42 (Q2_0) or 143 (PTQ1_0) switches the engine into the
    // rotated ternary mode, where every projected input is Hadamard-rotated before the GEMV.
    for (const auto & kv : ix.tensors) {
        if (qmat_is_ternary(kv.second.type)) { ternary_mode_ = true; break; }
    }
    if (ternary_mode_) {
        if (!ix.arr_kv.count("prism.hadamard.sign_widths"))
            throw std::runtime_error("ternary model without prism.hadamard.sign_widths");
        if (!ix.f32_arr_kv.count("prism.hadamard.sign_values") && !ix.i32_arr_kv.count("prism.hadamard.sign_values"))
            throw std::runtime_error("ternary model without prism.hadamard.sign_values");
        const std::vector<int64_t> & w = ix.arr_kv.at("prism.hadamard.sign_widths");
        if (w.size() != 3 || w[0] != kHidden || w[1] != kInner || w[2] != kFFN)
            throw std::runtime_error("ternary: unexpected prism.hadamard.sign_widths");
        if (ix.f32_arr_kv.count("prism.hadamard.sign_values")) {
            signs_ = ix.f32_arr_kv.at("prism.hadamard.sign_values");
        } else {  // stored as int32 +-1: widen to float
            const std::vector<int32_t> & sv = ix.i32_arr_kv.at("prism.hadamard.sign_values");
            signs_.resize(sv.size());
            for (size_t i = 0; i < sv.size(); ++i) signs_[i] = static_cast<float>(sv[i]);
        }
        if (signs_.size() != static_cast<size_t>(kHidden + kInner + kFFN))
            throw std::runtime_error("ternary: sign_values length does not match the widths");
        signs_5120_ = signs_.data();
        signs_6144_ = signs_.data() + kHidden;
        signs_17408_ = signs_.data() + kHidden + kInner;
        gdn_v_grouped_ = ix.u_kv.count("prism.hadamard.gdn_v_grouped") ? ix.u_kv.at("prism.hadamard.gdn_v_grouped") != 0 : false;
        std::fprintf(stderr, "ternary: Hadamard basis (block 1024, explicit signs), gdn_v_grouped=%d\n", gdn_v_grouped_ ? 1 : 0);
    }
    if (!ternary_mode_ && cfg_.ctx_max > kMaxCtxQwen)
        throw std::runtime_error("ctx_max above " + std::to_string(kMaxCtxQwen) + " is only supported on the Ternary Bonsai path");

    layers_.resize(kLayers);
    for (int il = 0; il < kLayers; ++il) {
        LayerW & L = layers_[il];
        const std::string b = "blk." + std::to_string(il) + ".";
        L.is_attn = (il % 4 == 3);
        L.attn_norm = f32p(ix, b + "attn_norm.weight", kHidden);
        L.post_norm = f32p(ix, b + "post_attention_norm.weight", kHidden);
        L.ffn_gate = make_qmat(ix, b + "ffn_gate.weight");
        L.ffn_up = make_qmat(ix, b + "ffn_up.weight");
        L.ffn_down = make_qmat(ix, b + "ffn_down.weight");
        if (L.is_attn) {
            L.wq = make_qmat(ix, b + "attn_q.weight");
            L.wk = make_qmat(ix, b + "attn_k.weight");
            L.wv = make_qmat(ix, b + "attn_v.weight");
            L.wo = make_qmat(ix, b + "attn_output.weight");
            L.q_norm = f32p(ix, b + "attn_q_norm.weight", kHD);
            L.k_norm = f32p(ix, b + "attn_k_norm.weight", kHD);
            if (L.wq.out != 2 * kQW || L.wk.out != kKVW || L.wv.out != kKVW || L.wo.in != kQW)
                throw std::runtime_error("unexpected attention shapes in " + b);
        } else {
            L.qkv = make_qmat(ix, b + "attn_qkv.weight");
            L.gate_z = make_qmat(ix, b + "attn_gate.weight");
            L.alpha = make_qmat(ix, b + "ssm_alpha.weight");
            L.beta = make_qmat(ix, b + "ssm_beta.weight");
            L.ssm_out = make_qmat(ix, b + "ssm_out.weight");
            L.ssm_norm = f32p(ix, b + "ssm_norm.weight", kHeadDim);
            L.ssm_a = f32p(ix, b + "ssm_a", kVHeads);
            L.ssm_dt = f32p(ix, b + "ssm_dt.bias", kVHeads);
            L.conv_w = f32p(ix, b + "ssm_conv1d.weight", kConvCh * kConvK);
            if (L.qkv.out != kConvCh || L.gate_z.out != kInner || L.alpha.out != kVHeads || L.ssm_out.out != kHidden)
                throw std::runtime_error("unexpected DeltaNet shapes in " + b);
        }
    }

    // MTP head (block 64): a full attention + FFN block plus the nextn projection. Optional: absent => no speculative decoding.
    if (cfg_.mtp_steps > 0 && ix.tensors.count("blk.64.nextn.eh_proj.weight")) {
        const std::string b = "blk.64.";
        LayerW & L = mtp_L_;
        L.is_attn = true;
        L.attn_norm = f32p(ix, b + "attn_norm.weight", kHidden);
        L.post_norm = f32p(ix, b + "post_attention_norm.weight", kHidden);
        L.ffn_gate = make_qmat(ix, b + "ffn_gate.weight");
        L.ffn_up = make_qmat(ix, b + "ffn_up.weight");
        L.ffn_down = make_qmat(ix, b + "ffn_down.weight");
        L.wq = make_qmat(ix, b + "attn_q.weight");
        L.wk = make_qmat(ix, b + "attn_k.weight");
        L.wv = make_qmat(ix, b + "attn_v.weight");
        L.wo = make_qmat(ix, b + "attn_output.weight");
        L.q_norm = f32p(ix, b + "attn_q_norm.weight", kHD);
        L.k_norm = f32p(ix, b + "attn_k_norm.weight", kHD);
        mtp_eh_ = make_qmat(ix, b + "nextn.eh_proj.weight");
        mtp_enorm_ = f32p(ix, b + "nextn.enorm.weight", kHidden);
        mtp_hnorm_ = f32p(ix, b + "nextn.hnorm.weight", kHidden);
        mtp_head_norm_ = f32p(ix, b + "nextn.shared_head_norm.weight", kHidden);
        if (mtp_eh_.in != 2 * kHidden || mtp_eh_.out != kHidden || L.wq.out != 2 * kQW || L.wk.out != kKVW)
            throw std::runtime_error("unexpected MTP block shapes");
        mtp_ok_ = true;
    }

    // File span of every layer, page aligned, for the residency policy.
    const uint64_t page = 4096;
    std::vector<uint64_t> lo(kLayers, UINT64_MAX), hi(kLayers, 0);
    for (const auto & kv : ix.tensors) {
        const std::string & n = kv.first;
        if (n.compare(0, 4, "blk.") != 0) continue;
        const int il = std::atoi(n.c_str() + 4);
        if (il < 0 || il >= kLayers) continue;
        const uint64_t a = ix.data_start + kv.second.offset;
        const uint64_t e = a + kv.second.nbytes;
        if (a < lo[il]) lo[il] = a;
        if (e > hi[il]) hi[il] = e;
    }
    // ---- GPU layer count and prefill streaming, checked against the VRAM budget before anything is allocated.
    // --gpu-layers auto (-1) takes the largest contiguous prefix 0 .. n-1 whose plan fits. Prefill streaming needs a
    // staging buffer plus one FP32 KV layer (ctx_max * 8 KiB: 1 GiB at 128K); when it does not fit, it is turned off.
    const bool roll = cfg_.mtp_steps > 0 && mtp_ok_;
    auto gpu_config = [&](int n, bool stream, bool head, bool mtp = false) {
        GpuConfig gc;
        gc.n_layers = n;
        gc.ctx_max = cfg_.ctx_max;
        gc.t_max = cfg_.t_max;
        gc.kv_iso3 = cfg_.kv_iso3;
        gc.dp4a_mask = cfg_.dp4a_mask;
        gc.stream_prefill = stream;
        gc.roll_t = roll ? kRollT : 0;
        gc.use_graph = cfg_.cuda_graph;
        gc.pin_stream = cfg_.pin_stream;
        gc.vram_budget_mib = cfg_.vram_budget_mib;
        gc.head = head;
        gc.mtp = mtp;
        if (ternary_mode_) {  // rotated basis: the GPU layers need the sign vectors and the grouped-V flag
            gc.signs_5120 = signs_5120_;
            gc.signs_6144 = signs_6144_;
            gc.signs_17408 = signs_17408_;
            gc.gdn_v_grouped = gdn_v_grouped_;
        }
        return gc;
    };
    bool stream_on = false;
    if (cfg_.gpu_layers != 0) {
        const bool want_stream = cfg_.stream_min_t > 0;
        auto fits = [&](int n, bool stream) { return GpuLayers::projected_mib(ix, gpu_config(n, stream, false)) <= cfg_.vram_budget_mib; };
        if (cfg_.gpu_layers < 0) {
            int pick = 0;
            for (int n = kLayers; n >= 1 && pick == 0; --n) {
                if (want_stream && n < kLayers && fits(n, true)) { pick = n; stream_on = true; }
                else if (fits(n, false)) { pick = n; stream_on = false; }
            }
            if (pick == 0) throw std::runtime_error("--gpu-layers auto: not even one layer fits the VRAM budget");
            std::fprintf(stderr, "gpu-layers auto: %d layers fit %.0f MiB at ctx %d (prefill streaming %s)\n", pick,
                         cfg_.vram_budget_mib, cfg_.ctx_max, stream_on ? "on" : "off");
            cfg_.gpu_layers = pick;
        } else {
            if (cfg_.gpu_layers > kLayers) throw std::runtime_error("--gpu-layers above 64");
            stream_on = want_stream && cfg_.gpu_layers < kLayers;
            if (stream_on && !fits(cfg_.gpu_layers, true) && fits(cfg_.gpu_layers, false)) {
                std::fprintf(stderr, "stream: staging for ctx %d does not fit the VRAM budget, prefill streaming off\n", cfg_.ctx_max);
                stream_on = false;
            }
        }
    }
    const int g0 = cfg_.gpu_layers;
    // LM head on the GPU: only with every layer there (the stream is already in VRAM), when it fits after the layers.
    // Auto (-1) puts it there if the plan stays under the budget; 1 requires it; 0 keeps it on the CPU.
    if (g0 == kLayers && cfg_.gpu_head != 0) {
        bool fit = GpuLayers::projected_mib(ix, gpu_config(g0, false, true)) <= cfg_.vram_budget_mib;
        // At long context the iso3 KV leaves no room for the head next to the prefill scratch (about 270 MiB at t_max 512).
        // Smaller prefill chunks free it: the batched GEMVs tile at most 16 tokens, so prefill speed barely changes.
        for (int tm : {256, 128, 64}) {
            if (fit || cfg_.t_max <= tm) break;
            const int keep = cfg_.t_max;
            cfg_.t_max = tm;
            fit = GpuLayers::projected_mib(ix, gpu_config(g0, false, true)) <= cfg_.vram_budget_mib;
            if (fit) std::fprintf(stderr, "LM head: prefill chunk lowered from %d to %d tokens so the head fits in VRAM\n", keep, tm);
            else cfg_.t_max = keep;
        }
        if (!fit && cfg_.gpu_head == 1) throw std::runtime_error("--gpu-head 1: the LM head does not fit the VRAM budget at this context");
        head_on_ = fit;
        std::fprintf(stderr, "LM head: %s\n", fit ? "GPU" : "CPU (does not fit the VRAM budget next to 64 layers)");
    } else if (cfg_.gpu_head == 1) {
        throw std::runtime_error("--gpu-head 1 needs every layer on the GPU (--gpu-layers 64)");
    }
    // MTP draft head on the GPU: when the GGUF has blk.64, every layer and the LM head are on the GPU, and the block plus
    // its KV fit next to them (about 335 MiB of weights for the Qwen-format head, 0.8 KiB of iso3 KV per position).
    if (mtp_ok_ && head_on_ && cfg_.gpu_mtp != 0) {
        const bool fit = GpuLayers::projected_mib(ix, gpu_config(g0, false, true, true)) <= cfg_.vram_budget_mib;
        if (!fit && cfg_.gpu_mtp == 1) throw std::runtime_error("--gpu-mtp 1: the MTP block does not fit the VRAM budget at this context");
        mtp_gpu_ = fit;
        std::fprintf(stderr, "MTP block: %s\n", fit ? "GPU" : "CPU (does not fit the VRAM budget)");
    }

    // ---- Host arena, sized from the configuration (measuring pass with the same allocation sequence as the real one).
    {
        Arena probe{Arena::Measure{}};
        carve_buffers(probe, g0, roll);
        arena_bytes_need_ = probe.used();
    }
    size_t locked_bytes = 0;   // LM head + MTP block, mlocked below
    {
        const GgufTensor & t = ix.get("output.weight");
        locked_bytes += t.nbytes;
        if (mtp_ok_)
            for (const auto & kv : ix.tensors)
                if (kv.first.compare(0, 7, "blk.64.") == 0) locked_bytes += kv.second.nbytes;
    }
    const double host_fixed = static_cast<double>(arena_bytes_need_) + static_cast<double>(locked_bytes) + kHostMarginBytes;
    if (host_fixed > kHostBudgetBytes) {
        char msg[256];
        std::snprintf(msg, sizeof(msg), "host RAM budget: arena %.0f MiB + locked %.0f MiB + margin exceed 14.2 GB "
                      "(ctx %d, %d CPU layers): put more layers on the GPU or lower --ctx",
                      static_cast<double>(arena_bytes_need_) / 1048576.0, static_cast<double>(locked_bytes) / 1048576.0,
                      cfg_.ctx_max, kLayers - g0);
        throw std::runtime_error(msg);
    }
    // Resident CPU-layer weights fill what is left of the budget (the rest is streamed through the page cache and dropped).
    const double resident_cap = std::min(cfg_.resident_gib, (kHostBudgetBytes - host_fixed) / kGiB);

    double cum = 0;
    n_resident_layers = 0;
    for (int il = 0; il < kLayers; ++il) {
        layers_[il].span_begin = lo[il] / page * page;
        layers_[il].span_end = (hi[il] + page - 1) / page * page;
        const double gib = static_cast<double>(layers_[il].span_end - layers_[il].span_begin) / kGiB;
        if (il < g0) {  // lives in VRAM after the upload; its file pages are dropped from the page cache below
            layers_[il].resident = false;
            continue;
        }
        layers_[il].resident = (cum + gib <= resident_cap);
        if (layers_[il].resident) { cum += gib; ++n_resident_layers; }
        else streamed_gib += gib;
    }
    resident_gib_actual = cum;

    // Lock only the LM head (about 1 GiB, under the 1.94 GiB memlock limit) so it never leaves RAM.
    {
        const GgufTensor & t = ix.get("output.weight");
        const uint64_t a = (ix.data_start + t.offset) / page * page;
        const uint64_t e = (ix.data_start + t.offset + t.nbytes + page - 1) / page * page;
        if (mlock(ix.base + a, e - a) != 0) std::fprintf(stderr, "warning: mlock of the LM head failed (continuing)\n");
    }

    if (mtp_ok_) {  // the MTP block (about 350 MiB) is read on every draft step: keep it in RAM (memlock limit 1.94 GiB)
        uint64_t lo = UINT64_MAX, hi = 0;
        for (const auto & kv : ix.tensors) {
            if (kv.first.compare(0, 7, "blk.64.") != 0) continue;
            const uint64_t a = ix.data_start + kv.second.offset;
            lo = std::min<uint64_t>(lo, a);
            hi = std::max<uint64_t>(hi, a + kv.second.nbytes);
        }
        const uint64_t a = lo / page * page, e = (hi + page - 1) / page * page;
        if (mlock(ix.base + a, e - a) != 0) std::fprintf(stderr, "warning: mlock of the MTP block failed (continuing)\n");
    }

    // Static arena: every buffer is carved out here, then the arena is frozen.
    arena_ = std::make_unique<Arena>((arena_bytes_need_ + 4095) / 4096 * 4096);
    carve_buffers(*arena_, g0, roll);
    {
        // Same theta chain as the old per-call rope lambda (theta = pos, then repeated theta_scale multiplies),
        // so the table values are bit-identical to what std::cos/std::sin produced inline.
        const float theta_scale = std::pow(10000000.0f, -2.0f / static_cast<float>(kRope));
        for (int pos = 0; pos < cfg_.ctx_max; ++pos) {
            float theta = static_cast<float>(pos);
            for (int p = 0; p < kRope / 2; ++p) {
                rope_cos_[static_cast<size_t>(pos) * (kRope / 2) + p] = std::cos(theta);
                rope_sin_[static_cast<size_t>(pos) * (kRope / 2) + p] = std::sin(theta);
                theta *= theta_scale;
            }
        }
    }
    reset();
    arena_->freeze();

    if (g0 > 0) {
        const GpuConfig gc = gpu_config(g0, stream_on, head_on_, mtp_gpu_);
        gpu_ = std::make_unique<GpuLayers>(ix, gc, eps_);
        for (int il = 0; il < g0; ++il) hint_layer(il, false);  // weights are in VRAM now: drop their file pages
        const double used = vram_used_mib();
        if (used > cfg_.vram_budget_mib) {
            char msg[160];
            std::snprintf(msg, sizeof(msg), "device memory in use %.0f MiB after load exceeds the budget of %.0f MiB", used, cfg_.vram_budget_mib);
            throw std::runtime_error(msg);
        }
    }
}

void Engine::clear_kv() {
    if (gpu_) gpu_->clear_kv();
    for (size_t i = 0; i < kcache_.size(); ++i) {
        if (kcache_[i]) std::memset(kcache_[i], 0, sizeof(float) * static_cast<size_t>(cfg_.ctx_max) * kKVW);
        if (vcache_[i]) std::memset(vcache_[i], 0, sizeof(float) * static_cast<size_t>(cfg_.ctx_max) * kKVW);
    }
}

size_t Engine::state_bytes(int n_pos) const {
    size_t b = gpu_ ? gpu_->state_bytes(n_pos) : 0;
    for (int il = std::max(cfg_.gpu_layers, 0); il < kLayers; ++il) {
        if (layers_[il].is_attn) b += 2 * sizeof(float) * static_cast<size_t>(n_pos) * kKVW;
        else b += sizeof(float) * (static_cast<size_t>(kConvCh) * (kConvK - 1) + static_cast<size_t>(kVHeads) * kHeadDim * kHeadDim);
    }
    if (mtp_ok_ && !mtp_gpu_) b += 2 * sizeof(float) * static_cast<size_t>(n_pos) * kKVW;
    return b;
}

void Engine::save_state(int n_pos, const std::function<void(const void *, size_t)> & io) const {
    if (n_pos < 0 || n_pos > cfg_.ctx_max) throw std::runtime_error("save_state: n_pos out of range");
    if (gpu_) gpu_->save_state(n_pos, io);
    const size_t kv = sizeof(float) * static_cast<size_t>(n_pos) * kKVW;
    for (int il = std::max(cfg_.gpu_layers, 0); il < kLayers; ++il) {
        if (layers_[il].is_attn) {
            io(kcache_[il / 4], kv);
            io(vcache_[il / 4], kv);
        } else {
            const int di = (il / 4) * 3 + il % 4;
            io(conv_state_[di], sizeof(float) * kConvCh * (kConvK - 1));
            io(ssm_state_[di], sizeof(float) * kVHeads * kHeadDim * kHeadDim);
        }
    }
    if (mtp_ok_ && !mtp_gpu_) {
        io(kcache_[16], kv);
        io(vcache_[16], kv);
    }
}

void Engine::load_state(int n_pos, const std::function<void(void *, size_t)> & io) {
    if (n_pos < 0 || n_pos > cfg_.ctx_max) throw std::runtime_error("load_state: n_pos out of range");
    if (gpu_) gpu_->load_state(n_pos, io);
    const size_t kv = sizeof(float) * static_cast<size_t>(n_pos) * kKVW;
    for (int il = std::max(cfg_.gpu_layers, 0); il < kLayers; ++il) {
        if (layers_[il].is_attn) {
            io(kcache_[il / 4], kv);
            io(vcache_[il / 4], kv);
        } else {
            const int di = (il / 4) * 3 + il % 4;
            io(conv_state_[di], sizeof(float) * kConvCh * (kConvK - 1));
            io(ssm_state_[di], sizeof(float) * kVHeads * kHeadDim * kHeadDim);
        }
    }
    if (mtp_ok_ && !mtp_gpu_) {
        io(kcache_[16], kv);
        io(vcache_[16], kv);
    }
}

uint64_t Engine::state_fingerprint() const {
    // FNV-1a over the tensor table (name, type, offset, size) and the state layout.
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](const void * p, size_t n) {
        const uint8_t * b = static_cast<const uint8_t *>(p);
        for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    };
    std::vector<std::string> names;
    names.reserve(ix.tensors.size());
    for (const auto & kv : ix.tensors) names.push_back(kv.first);
    std::sort(names.begin(), names.end());
    for (const std::string & n : names) {
        const GgufTensor & t = ix.tensors.at(n);
        mix(n.data(), n.size());
        const int64_t v[3] = {static_cast<int64_t>(t.type), static_cast<int64_t>(t.offset), static_cast<int64_t>(t.nbytes)};
        mix(v, sizeof(v));
    }
    const int32_t layout[5] = {cfg_.gpu_layers, cfg_.kv_iso3 ? 1 : 0, mtp_ok_ ? 1 : 0, ternary_mode_ ? 1 : 0, mtp_gpu_ ? 1 : 0};
    mix(layout, sizeof(layout));
    return h;
}

// Carves every host buffer out of `a` (also run on a measuring arena to size the real one). Clears the per-layer vectors first.
void Engine::carve_buffers(Arena & a, int g0, bool roll) {
    kcache_.clear(); vcache_.clear(); conv_state_.clear(); ssm_state_.clear();
    snap_conv_.clear(); snap_ssm_.clear(); sv_qkv_.clear(); sv_z_.clear(); sv_gate_.clear(); sv_beta_.clear();
    const size_t Tm = static_cast<size_t>(cfg_.t_max);
    x_ = a.alloc_n<float>(Tm * kHidden);
    xn_ = a.alloc_n<float>(Tm * kHidden);
    proj_ = a.alloc_n<float>(Tm * kHidden);
    bufA_ = a.alloc_n<float>(Tm * 2 * kQW);          // qkv [T][10240] or q+gate [T][12288]
    bufB_ = a.alloc_n<float>(Tm * kInner);           // z
    if (ternary_mode_) {  // Hadamard-rotated inputs for the ternary GEMVs, one buffer per input width
        rot5120_ = a.alloc_n<float>(Tm * kHidden);
        rot6144_ = a.alloc_n<float>(Tm * kInner);
        rot17408_ = a.alloc_n<float>(Tm * kFFN);
    }
    sc_ = a.alloc_n<float>(static_cast<size_t>(pool_->size()) * kQG * cfg_.ctx_max);  // attention scores
    conv_silu_ = a.alloc_n<float>(Tm * kConvCh);
    attn_o_ = a.alloc_n<float>(Tm * kInner);
    alpha_ = a.alloc_n<float>(Tm * kVHeads);
    betar_ = a.alloc_n<float>(Tm * kVHeads);
    gate_ = a.alloc_n<float>(Tm * kVHeads);
    beta_ = a.alloc_n<float>(Tm * kVHeads);
    fg_ = a.alloc_n<float>(Tm * kFFN);
    fu_ = a.alloc_n<float>(Tm * kFFN);
    qbuf_ = a.alloc_n<float>(Tm * kQW);
    kbuf_ = a.alloc_n<float>(Tm * kKVW);
    vbuf_ = a.alloc_n<float>(Tm * kKVW);
    hlast_ = a.alloc_n<float>(kHidden);
    hn_ = a.alloc_n<float>(8 * static_cast<size_t>(kHidden));
    act_ = static_cast<uint8_t *>(a.alloc(qmat_act_bytes(12 /* Q4_K -> Q8_K */, kFFN, cfg_.t_max)));
    rope_cos_ = a.alloc_n<float>(static_cast<size_t>(cfg_.ctx_max) * (kRope / 2));
    rope_sin_ = a.alloc_n<float>(static_cast<size_t>(cfg_.ctx_max) * (kRope / 2));
    for (int il = 0; il < kLayers; ++il) {
        // Host state only for the layers that run on the CPU: a GPU layer keeps its KV / DeltaNet state in VRAM, and an
        // FP32 KV layer costs ctx_max * 8 KiB here (1 GiB at 128K). Null keeps the per-layer indexing (ai = il / 4).
        const bool cpu_layer = il >= g0;
        if (layers_[il].is_attn) {
            kcache_.push_back(cpu_layer ? a.alloc_n<float>(static_cast<size_t>(cfg_.ctx_max) * kKVW) : nullptr);
            vcache_.push_back(cpu_layer ? a.alloc_n<float>(static_cast<size_t>(cfg_.ctx_max) * kKVW) : nullptr);
        } else {
            conv_state_.push_back(cpu_layer ? a.alloc_n<float>(kConvCh * (kConvK - 1)) : nullptr);
            ssm_state_.push_back(cpu_layer ? a.alloc_n<float>(kVHeads * kHeadDim * kHeadDim) : nullptr);
            const size_t R = kRollT;
            snap_conv_.push_back(roll && cpu_layer ? a.alloc_n<float>(kConvCh * (kConvK - 1)) : nullptr);
            snap_ssm_.push_back(roll && cpu_layer ? a.alloc_n<float>(kVHeads * kHeadDim * kHeadDim) : nullptr);
            sv_qkv_.push_back(roll && cpu_layer ? a.alloc_n<float>(R * kConvCh) : nullptr);
            sv_z_.push_back(roll && cpu_layer ? a.alloc_n<float>(R * kInner) : nullptr);
            sv_gate_.push_back(roll && cpu_layer ? a.alloc_n<float>(R * kVHeads) : nullptr);
            sv_beta_.push_back(roll && cpu_layer ? a.alloc_n<float>(R * kVHeads) : nullptr);
        }
    }
    if (mtp_ok_) {  // KV of the MTP attention layer = cache index 16 (host copy only when the MTP block runs on the CPU)
        kcache_.push_back(mtp_gpu_ ? nullptr : a.alloc_n<float>(static_cast<size_t>(cfg_.ctx_max) * kKVW));
        vcache_.push_back(mtp_gpu_ ? nullptr : a.alloc_n<float>(static_cast<size_t>(cfg_.ctx_max) * kKVW));
    }
}

Engine::~Engine() = default;

size_t Engine::gpu_arena_bytes() const { return gpu_ ? gpu_->vram_bytes() : 0; }
size_t Engine::gpu_weight_bytes() const { return gpu_ ? gpu_->weight_bytes() : 0; }
size_t Engine::gpu_kv_bytes() const { return gpu_ ? gpu_->kv_bytes() : 0; }
double Engine::vram_used_mib() const {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return -1.0;
    return static_cast<double>(total_b - free_b) / (1024.0 * 1024.0);
}

void Engine::reset() {
    if (gpu_) gpu_->reset();
    for (float * p : conv_state_) if (p) std::memset(p, 0, sizeof(float) * kConvCh * (kConvK - 1));
    for (float * p : ssm_state_) if (p) std::memset(p, 0, sizeof(float) * kVHeads * kHeadDim * kHeadDim);
}

void Engine::hint_layer(int il, bool willneed) {
    if (il < 0 || il >= kLayers) return;
    const LayerW & L = layers_[il];
    uint8_t * p = const_cast<uint8_t *>(ix.base) + L.span_begin;
    const size_t len = L.span_end - L.span_begin;
    if (willneed) {
        madvise(p, len, MADV_WILLNEED);
    } else if (!L.resident) {
        madvise(p, len, MADV_DONTNEED);
        posix_fadvise(ix.fd(), static_cast<off_t>(L.span_begin), static_cast<off_t>(len), POSIX_FADV_DONTNEED);
    }
}

void Engine::check_memory(int il) const {
    if (il % 8 != 0) return;
    const double avail = mem_available_gib();
    if (avail < 0.5) {
        throw std::runtime_error("MemAvailable fell to " + std::to_string(avail) + " GiB at layer " +
                                 std::to_string(il) + ", stopping before the OOM killer acts");
    }
}

// ssm_out input permutation (gdn_v_grouped=1): the reference permutes the DeltaNet output from tiled head order
// [hd, nk, rep] to grouped [hd, rep, nk] before signs and rotation. With hd = 128, nk = 16 groups, rep = 3 value
// heads per group, the flat v index v = nk + 16*rep becomes v' = 3*nk + rep.
void Engine::gdn_v_permute(const float * src, float * dst, int64_t T) {
    for (int64_t t = 0; t < T; ++t) {
        const float * s = src + t * kInner;
        float * d = dst + t * kInner;
        for (int v = 0; v < kVHeads; ++v) {
            const int nk = v % kQKHeads, rep = v / kQKHeads;
            std::memcpy(d + (3 * nk + rep) * kHeadDim, s + v * kHeadDim, sizeof(float) * kHeadDim);
        }
    }
}

// y = W . hadamard(x) for a ternary (PTQ1_0 / Q2_0) weight, else y = W . x. The rotation cost is O(width log 1024)
// per token and the rotated buffer is reused for every matrix of the same input width in the layer.
void Engine::tmatmul(const QMat & W, const float * x, int64_t T, float * y, int width, uint8_t * act) {
    if (!ternary_mode_ || !qmat_is_ternary(W.type)) {
        qmatmul(*pool_, W, x, T, y, act);
        return;
    }
    const float * signs = width == kHidden ? signs_5120_ : (width == kInner ? signs_6144_ : signs_17408_);
    float * rot = width == kHidden ? rot5120_ : (width == kInner ? rot6144_ : rot17408_);
    for (int64_t t = 0; t < T; ++t) std::memcpy(rot + t * width, x + t * width, sizeof(float) * static_cast<size_t>(width));
    hadamard_rotate(rot, signs, width, T);
    qmatmul(*pool_, W, rot, T, y, nullptr);
}

void Engine::ffn(const LayerW & L, int T) {
    for (int t = 0; t < T; ++t) rmsnorm(x_ + t * kHidden, kHidden, L.post_norm, eps_, xn_ + t * kHidden);
    tmatmul(L.ffn_gate, xn_, T, fg_, kHidden, act_);
    tmatmul(L.ffn_up, xn_, T, fu_, kHidden, act_);
    const int64_t n = static_cast<int64_t>(T) * kFFN;
    pool_->parallel_for(n, 4096, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) fg_[i] = silu(fg_[i]) * fu_[i];
    });
    tmatmul(L.ffn_down, fg_, T, proj_, kFFN, act_);
    for (int64_t i = 0; i < static_cast<int64_t>(T) * kHidden; ++i) x_[i] += proj_[i];
}

void Engine::dn_conv_rec(int di, const LayerW & L, const float * qkv, const float * z, const float * gate, const float * beta,
                         int T, float * out) {
    float * cs = conv_state_[di];
    float * ss = ssm_state_[di];
    constexpr int64_t D = kHeadDim, Hk = kQKHeads, Hv = kVHeads;

    // Causal depthwise conv (kernel 4) with the 3-value history in the layer state, then SiLU.
    pool_->parallel_for(kConvCh, 64, [&](int64_t b, int64_t e) {
        for (int64_t c = b; c < e; ++c) {
            float h0 = cs[c * 3], h1 = cs[c * 3 + 1], h2 = cs[c * 3 + 2];
            const float * w = L.conv_w + c * kConvK;
            for (int t = 0; t < T; ++t) {
                const float xin = qkv[static_cast<int64_t>(t) * kConvCh + c];
                float s = 0.0f;
                s += h0 * w[0];
                s += h1 * w[1];
                s += h2 * w[2];
                s += xin * w[3];
                conv_silu_[static_cast<int64_t>(t) * kConvCh + c] = silu(s);
                h0 = h1; h1 = h2; h2 = xin;
            }
            cs[c * 3] = h0; cs[c * 3 + 1] = h1; cs[c * 3 + 2] = h2;
        }
    });

    // Gated delta rule per V head (K=1), then the gated RMSNorm: RMSNorm(out) * ssm_norm * SiLU(z). `out` is null when the
    // call only replays the state update (rollback).
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    pool_->parallel_for(Hv, 1, [&](int64_t hb, int64_t he) {
        float qn[kHeadDim], kn[kHeadDim], o[kHeadDim], dl[kHeadDim];
        for (int64_t h = hb; h < he; ++h) {
            float * M = ss + h * D * D;
            const int64_t hk = h % Hk;
            for (int t = 0; t < T; ++t) {
                const float * row = conv_silu_ + static_cast<int64_t>(t) * kConvCh;
                const float * q = row + hk * D;
                const float * k = row + Hk * D + hk * D;
                const float * v = row + 2 * Hk * D + h * D;
                double sq = 0, sk = 0;
                for (int64_t i = 0; i < D; ++i) { sq += static_cast<double>(q[i]) * q[i]; sk += static_cast<double>(k[i]) * k[i]; }
                const float iq = 1.0f / std::fmax(std::sqrt(static_cast<float>(sq)), eps_);
                const float ik = 1.0f / std::fmax(std::sqrt(static_cast<float>(sk)), eps_);
                for (int64_t i = 0; i < D; ++i) { qn[i] = q[i] * iq; kn[i] = k[i] * ik; }

                const float b = beta[t * Hv + h];
                const float ex = std::exp(gate[t * Hv + h]);
                for (int64_t i = 0; i < D * D; ++i) M[i] *= ex;
                for (int64_t j = 0; j < D; ++j) dl[j] = (v[j] - dotf(M + j * D, kn, D)) * b;
                for (int64_t j = 0; j < D; ++j) {
                    float * Mj = M + j * D;
                    const float dj = dl[j];
                    for (int64_t i = 0; i < D; ++i) Mj[i] += kn[i] * dj;
                }
                if (!out) continue;
                double ms = 0;
                for (int64_t j = 0; j < D; ++j) { o[j] = dotf(M + j * D, qn, D) * scale; ms += static_cast<double>(o[j]) * o[j]; }
                const float rs = 1.0f / std::sqrt(static_cast<float>(ms / static_cast<double>(D)) + eps_);
                const float * zz = z + static_cast<int64_t>(t) * kInner + h * D;
                float * dst = out + static_cast<int64_t>(t) * kInner + h * D;
                for (int64_t j = 0; j < D; ++j) dst[j] = o[j] * rs * L.ssm_norm[j] * silu(zz[j]);
            }
        }
    });
}

void Engine::dn_layer(int il, const LayerW & L, int T) {
    const int di = (il / 4) * 3 + il % 4;
    constexpr int64_t Hv = kVHeads;

    for (int t = 0; t < T; ++t) rmsnorm(x_ + t * kHidden, kHidden, L.attn_norm, eps_, xn_ + t * kHidden);
    tmatmul(L.qkv, xn_, T, bufA_, kHidden, act_);      // [T][10240]
    tmatmul(L.gate_z, xn_, T, bufB_, kHidden, act_);   // [T][6144]
    qmatmul(*pool_, L.alpha, xn_, T, alpha_, act_);    // [T][48]  (not in prism.hadamard.weight_names)
    qmatmul(*pool_, L.beta, xn_, T, betar_, act_);     // [T][48]
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < Hv; ++h) {
            const int k = t * static_cast<int>(Hv) + h;
            beta_[k] = sigm(betar_[k]);
            gate_[k] = softplus_(alpha_[k] + L.ssm_dt[h]) * L.ssm_a[h];
        }
    }
    if (dn_save_ && snap_ssm_[di]) {  // state before this batch and the inputs of the recurrence, for dn_rollback()
        std::memcpy(snap_conv_[di], conv_state_[di], sizeof(float) * kConvCh * (kConvK - 1));
        std::memcpy(snap_ssm_[di], ssm_state_[di], sizeof(float) * kVHeads * kHeadDim * kHeadDim);
        std::memcpy(sv_qkv_[di], bufA_, sizeof(float) * static_cast<size_t>(T) * kConvCh);
        std::memcpy(sv_z_[di], bufB_, sizeof(float) * static_cast<size_t>(T) * kInner);
        std::memcpy(sv_gate_[di], gate_, sizeof(float) * static_cast<size_t>(T) * kVHeads);
        std::memcpy(sv_beta_[di], beta_, sizeof(float) * static_cast<size_t>(T) * kVHeads);
    }
    dn_conv_rec(di, L, bufA_, bufB_, gate_, beta_, T, attn_o_);

    if (ternary_mode_ && gdn_v_grouped_) {
        // value heads arrive grouped for the folded ssm_out weight: permute, rotate, GEMV
        gdn_v_permute(attn_o_, rot6144_, T);
        hadamard_rotate(rot6144_, signs_6144_, kInner, T);
        qmatmul(*pool_, L.ssm_out, rot6144_, T, proj_, nullptr);
    } else {
        tmatmul(L.ssm_out, attn_o_, T, proj_, kInner, act_);
    }
    for (int64_t i = 0; i < static_cast<int64_t>(T) * kHidden; ++i) x_[i] += proj_[i];
}

void Engine::dn_snapshot(bool on) {
    if (on && !(cfg_.mtp_steps > 0 && mtp_ok_)) throw std::runtime_error("dn_snapshot: rollback buffers were not allocated");
    dn_save_ = on;
    if (gpu_) gpu_->set_snapshot(on);
}

void Engine::dn_rollback(int keep) {
    if (keep < 1 || keep > kRollT) throw std::runtime_error("dn_rollback: keep out of range");
    if (gpu_) gpu_->rollback(keep);
    for (int il = std::max(cfg_.gpu_layers, 0); il < kLayers; ++il) {
        if (layers_[il].is_attn) continue;
        const int di = (il / 4) * 3 + il % 4;
        if (!snap_ssm_[di]) continue;
        std::memcpy(conv_state_[di], snap_conv_[di], sizeof(float) * kConvCh * (kConvK - 1));
        std::memcpy(ssm_state_[di], snap_ssm_[di], sizeof(float) * kVHeads * kHeadDim * kHeadDim);
        dn_conv_rec(di, layers_[il], sv_qkv_[di], sv_z_[di], sv_gate_[di], sv_beta_[di], keep, nullptr);
    }
}

void Engine::attn_layer(int il, const LayerW & L, int T, int pos0) {
    const int ai = il / 4;
    float * kc = kcache_[ai];
    float * vc = vcache_[ai];

    for (int t = 0; t < T; ++t) rmsnorm(x_ + t * kHidden, kHidden, L.attn_norm, eps_, xn_ + t * kHidden);
    tmatmul(L.wq, xn_, T, bufA_, kHidden, act_);   // per head: [q(256) | gate(256)]
    tmatmul(L.wk, xn_, T, kbuf_, kHidden, act_);
    tmatmul(L.wv, xn_, T, vbuf_, kHidden, act_);

    // RoPE angles come from the precomputed tables (generated at load with the identical accumulation chain),
    // so the values are bit-identical to the old inline chain; the 28 heads of a token share one lookup.
    auto rope = [&](float * v, int pos) {
        const float * cs = rope_cos_ + static_cast<size_t>(pos) * (kRope / 2);
        const float * sn = rope_sin_ + static_cast<size_t>(pos) * (kRope / 2);
        for (int p = 0; p < kRope / 2; ++p) {
            const float c = cs[p], s = sn[p];
            const float x0 = v[p], x1 = v[p + kRope / 2];
            v[p] = x0 * c - x1 * s;
            v[p + kRope / 2] = x0 * s + x1 * c;
        }
    };

    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < kNH; ++h) {
            const float * q = bufA_ + static_cast<int64_t>(t) * 2 * kQW + h * 2 * kHD;
            float * qo = qbuf_ + static_cast<int64_t>(t) * kQW + h * kHD;
            rmsnorm(q, kHD, L.q_norm, eps_, qo);
            rope(qo, pos0 + t);
        }
        for (int g = 0; g < kNKV; ++g) {
            float * kk = kbuf_ + static_cast<int64_t>(t) * kKVW + g * kHD;
            rmsnorm(kk, kHD, L.k_norm, eps_, kk);
            rope(kk, pos0 + t);
            std::memcpy(kc + static_cast<int64_t>(pos0 + t) * kKVW + g * kHD, kk, sizeof(float) * kHD);
            std::memcpy(vc + static_cast<int64_t>(pos0 + t) * kKVW + g * kHD, vbuf_ + static_cast<int64_t>(t) * kKVW + g * kHD,
                        sizeof(float) * kHD);
        }
    }

    const float kq_scale = 1.0f / std::sqrt(static_cast<float>(kHD));
    pool_->parallel_for(static_cast<int64_t>(T) * kNKV, 1, [&](int64_t b, int64_t e) {
        // Score slab of this pool slot: [kQG][ctx_max]. The stride is ctx_max, the size it was allocated with.
        const size_t C = static_cast<size_t>(cfg_.ctx_max);
        float * const sc0 = sc_ + static_cast<size_t>(pool_->slot()) * kQG * C;
        auto sc = [sc0, C](int hh) { return sc0 + static_cast<size_t>(hh) * C; };
        for (int64_t idx = b; idx < e; ++idx) {
            const int t = static_cast<int>(idx / kNKV), g = static_cast<int>(idx % kNKV);
            const int n = pos0 + t + 1;  // causal: positions j0 .. pos0+t (j0 > 0 only for the MTP layer with a prefill window)
            const int j0 = il == 64 ? std::min(mtp_kv_start_, pos0 + t) : 0;
            const float * q[kQG];
            float mx[kQG], sum[kQG];
            for (int hh = 0; hh < kQG; ++hh) {
                q[hh] = qbuf_ + static_cast<int64_t>(t) * kQW + (g * kQG + hh) * kHD;
                mx[hh] = -INFINITY;
                sum[hh] = 0.0f;
            }
            // Score pass: every K row is read once for all six query heads of the group.
            for (int j = 0; j < j0; ++j)
                for (int hh = 0; hh < kQG; ++hh) sc(hh)[j] = 0.0f;
            for (int j = j0; j < n; ++j) {
                const float * krow = kc + static_cast<int64_t>(j) * kKVW + g * kHD;
                for (int hh = 0; hh < kQG; ++hh) {
                    const float s = dotf(q[hh], krow, kHD) * kq_scale;
                    sc(hh)[j] = s;
                    if (s > mx[hh]) mx[hh] = s;
                }
            }
            for (int hh = 0; hh < kQG; ++hh)
                for (int j = j0; j < n; ++j) { sc(hh)[j] = std::exp(sc(hh)[j] - mx[hh]); sum[hh] += sc(hh)[j]; }
            float o[kQG][kHD];
            for (int hh = 0; hh < kQG; ++hh) {
                for (int d = 0; d < kHD; ++d) o[hh][d] = 0.0f;
            }
            // Value pass: every V row is read once for all six query heads of the group.
            for (int j = j0; j < n; ++j) {
                const float * v = vc + static_cast<int64_t>(j) * kKVW + g * kHD;
                for (int hh = 0; hh < kQG; ++hh) {
                    const float pj = sc(hh)[j] * (1.0f / sum[hh]);
                    for (int d = 0; d < kHD; ++d) o[hh][d] += pj * v[d];
                }
            }
            for (int hh = 0; hh < kQG; ++hh) {
                const int h = g * kQG + hh;
                const float * gate = bufA_ + static_cast<int64_t>(t) * 2 * kQW + h * 2 * kHD + kHD;
                float * dst = attn_o_ + static_cast<int64_t>(t) * kInner + h * kHD;
                for (int d = 0; d < kHD; ++d) dst[d] = o[hh][d] * sigm(gate[d]);
            }
        }
    });

    tmatmul(L.wo, attn_o_, T, proj_, kInner, act_);
    for (int64_t i = 0; i < static_cast<int64_t>(T) * kHidden; ++i) x_[i] += proj_[i];
}

void Engine::forward(const int32_t * tokens, int T, int pos0, float * logits) { forward_ex(tokens, T, pos0, logits, logits ? 1 : 0, nullptr); }

// Runs layers cfg_.gpu_layers .. 63 on the GPU for a large first batch (prefill). x_ holds the stream after the resident GPU
// layers; K, V and the DeltaNet state of each layer come back in the CPU layouts so decoding continues on the CPU layers.
void Engine::forward_stream(int T, int pos0) {
    const int first_cpu = cfg_.gpu_layers;
    gpu_->stream_begin(x_, T);
    hint_layer(first_cpu, true);
    hint_layer(first_cpu + 1, true);
    for (int il = first_cpu; il < kLayers; ++il) {
        hint_layer(il + 2, true);
        check_memory(il);
        if (layers_[il].is_attn) {
            const int ai = il / 4;
            gpu_->stream_layer(il, T, pos0, nullptr, nullptr, kcache_[ai], vcache_[ai]);
        } else {
            const int di = (il / 4) * 3 + il % 4;
            gpu_->stream_layer(il, T, pos0, conv_state_[di], ssm_state_[di], nullptr, nullptr);
        }
        if (tap) { gpu_->stream_peek(x_, T); tap(il, x_, T); }
        hint_layer(il, false);
    }
    gpu_->stream_end(x_, T);
}

void Engine::forward_ex(const int32_t * tokens, int T, int pos0, float * logits, int n_out, float * h_rows) {
    if (T < 1 || T > cfg_.t_max || pos0 < 0 || pos0 + T > cfg_.ctx_max)
        throw std::runtime_error("forward: T or position out of range");
    if (n_out < 0 || n_out > T || (!h_rows && n_out > 8) || (n_out > 0 && !logits)) throw std::runtime_error("forward_ex: bad output request");
    for (int t = 0; t < T; ++t) {
        if (tokens[t] < 0 || tokens[t] >= n_vocab) throw std::runtime_error("forward: token id out of range");
        qmat_row_to_f32(tok_embd_, tokens[t], x_ + static_cast<int64_t>(t) * kHidden);
        if (ternary_mode_) hadamard_unrotate(x_ + static_cast<int64_t>(t) * kHidden, signs_5120_, kHidden, 1);
    }
    using Clk = std::chrono::steady_clock;
    const int first_cpu = cfg_.gpu_layers;
    const auto c0 = Clk::now();
    if (gpu_) gpu_->forward(x_, T, pos0, tap ? &tap : nullptr);   // layers 0 .. first_cpu-1 on the GPU
    const auto c1 = Clk::now();
    if (gpu_ && gpu_->stream_ready() && T >= cfg_.stream_min_t && T >= 1 && !dn_save_) {
        forward_stream(T, pos0);
    } else {
        hint_layer(first_cpu, true);
        hint_layer(first_cpu + 1, true);
        for (int il = first_cpu; il < kLayers; ++il) {
            hint_layer(il + 2, true);
            check_memory(il);
            const LayerW & L = layers_[il];
            if (L.is_attn) attn_layer(il, L, T, pos0);
            else dn_layer(il, L, T);
            ffn(L, T);
            if (tap) tap(il, x_, T);
            hint_layer(il, false);
        }
    }
    const auto c2 = Clk::now();
    if ((h_rows || n_out > 0) && head_on_ && n_out <= 8) {
        gpu_->head(T, n_out, logits, h_rows);   // the stream after layer 63 is still on the device
    } else if (h_rows || n_out > 0) {
        if (!h_rows && n_out == 1) {
            rmsnorm(x_ + static_cast<int64_t>(T - 1) * kHidden, kHidden, out_norm_, eps_, hlast_);
            if (ternary_mode_) {
                std::memcpy(rot5120_, hlast_, sizeof(float) * kHidden);
                hadamard_rotate(rot5120_, signs_5120_, kHidden, 1);
                qmatmul(*pool_, lm_head_, rot5120_, 1, logits, nullptr);
            } else {
                qmatmul(*pool_, lm_head_, hlast_, 1, logits, act_);
            }
        } else {
            float * hn = h_rows ? h_rows : hn_;
            const int r0 = h_rows ? 0 : T - n_out;
            for (int t = r0; t < T; ++t)
                rmsnorm(x_ + static_cast<int64_t>(t) * kHidden, kHidden, out_norm_, eps_, hn + static_cast<int64_t>(t - r0) * kHidden);
            if (n_out > 0) {
                const float * src = hn + static_cast<int64_t>(T - n_out - r0) * kHidden;
                if (ternary_mode_) {
                    for (int r = 0; r < n_out; ++r) std::memcpy(rot5120_ + r * kHidden, src + r * kHidden, sizeof(float) * kHidden);
                    hadamard_rotate(rot5120_, signs_5120_, kHidden, n_out);
                    qmatmul(*pool_, lm_head_, rot5120_, n_out, logits, nullptr);
                } else {
                    qmatmul(*pool_, lm_head_, src, n_out, logits, act_);
                }
            }
        }
    }
    const auto c3 = Clk::now();
    last_gpu_s = std::chrono::duration<double>(c1 - c0).count();
    last_cpu_s = std::chrono::duration<double>(c2 - c1).count();
    last_head_s = std::chrono::duration<double>(c3 - c2).count();
    if (T > 8 && std::getenv("QWEN_STREAM_DEBUG"))
        std::fprintf(stderr, "forward T=%d pos0=%d: GPU layers %.2f s, CPU/streamed layers %.2f s, head %.2f s\n", T, pos0, last_gpu_s, last_cpu_s, last_head_s);
}

void Engine::mtp_prefill(const int32_t * tokens, const float * h_in, int T, int pos0) {
    if (mtp_gpu_) { mtp_forward(tokens, h_in, T, pos0, nullptr, nullptr); return; }
    if (!(gpu_ && gpu_->stream_ready() && cfg_.mtp_gpu_prefill && T >= cfg_.stream_min_t)) { mtp_forward(tokens, h_in, T, pos0, nullptr, nullptr); return; }
    if (T < 1 || T > cfg_.t_max || pos0 < 0 || pos0 + T > cfg_.ctx_max) throw std::runtime_error("mtp_prefill: T or position out of range");
    for (int t = 0; t < T; ++t) {
        if (tokens[t] < 0 || tokens[t] >= n_vocab) throw std::runtime_error("mtp_prefill: token id out of range");
        qmat_row_to_f32(tok_embd_, tokens[t], proj_);
        if (ternary_mode_) hadamard_unrotate(proj_, signs_5120_, kHidden, 1);
        float * dst = bufA_ + static_cast<int64_t>(t) * 2 * kHidden;
        rmsnorm(proj_, kHidden, mtp_enorm_, eps_, dst);
        rmsnorm(h_in + static_cast<int64_t>(t) * kHidden, kHidden, mtp_hnorm_, eps_, dst + kHidden);
    }
    gpu_->mtp_kv(bufA_, T, pos0, kcache_[16], vcache_[16]);
}

void Engine::mtp_forward(const int32_t * tokens, const float * h_in, int T, int pos0, float * h_out, float * logits) {
    if (!mtp_ok_) throw std::runtime_error("mtp_forward: no MTP block loaded");
    if (T < 1 || T > cfg_.t_max || pos0 < 0 || pos0 + T > cfg_.ctx_max) throw std::runtime_error("mtp_forward: T or position out of range");
    // concat(rmsnorm(embedding, enorm), rmsnorm(h, hnorm)) -> eh_proj, then an ordinary attention + FFN block with its own KV.
    for (int t = 0; t < T; ++t) {
        if (tokens[t] < 0 || tokens[t] >= n_vocab) throw std::runtime_error("mtp_forward: token id out of range");
        qmat_row_to_f32(tok_embd_, tokens[t], proj_);
        if (ternary_mode_) hadamard_unrotate(proj_, signs_5120_, kHidden, 1);
        float * dst = bufA_ + static_cast<int64_t>(t) * 2 * kHidden;
        rmsnorm(proj_, kHidden, mtp_enorm_, eps_, dst);
        rmsnorm(h_in + static_cast<int64_t>(t) * kHidden, kHidden, mtp_hnorm_, eps_, dst + kHidden);
    }
    if (mtp_gpu_) { gpu_->mtp_forward(bufA_, T, pos0, h_out, logits); return; }
    qmatmul(*pool_, mtp_eh_, bufA_, T, x_, act_);
    attn_layer(64, mtp_L_, T, pos0);   // cache index 64 / 4 = 16
    ffn(mtp_L_, T);
    if (h_out || logits) {
        rmsnorm(x_ + static_cast<int64_t>(T - 1) * kHidden, kHidden, mtp_head_norm_, eps_, hlast_);
        if (h_out) std::memcpy(h_out, hlast_, sizeof(float) * kHidden);
        if (logits) {
            if (ternary_mode_) {   // the ternary LM head reads its input in the Hadamard basis
                std::memcpy(rot5120_, hlast_, sizeof(float) * kHidden);
                hadamard_rotate(rot5120_, signs_5120_, kHidden, 1);
                qmatmul(*pool_, lm_head_, rot5120_, 1, logits, nullptr);
            } else {
                qmatmul(*pool_, lm_head_, hlast_, 1, logits, act_);
            }
        }
    }
}
