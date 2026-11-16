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
#include "dn_layer.hpp"  // model dimension constants only (kHidden, kFFN, ...)

namespace {

constexpr int kLayers = 64;
constexpr int kNH = 24, kNKV = 4, kHD = 256, kRope = 64;
constexpr int kQW = kNH * kHD;       // 6144 values of q per token
constexpr int kKVW = kNKV * kHD;     // 1024 values of k (and of v) per token
constexpr int kMaxCtxStack = 4096;   // attention score buffer lives on the stack
constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

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
    if (cfg_.ctx_max > kMaxCtxStack) throw std::runtime_error("ctx_max above 4096 is not supported");
    qmat_init();
    ix.open(cfg_.gguf);

    // The engine is locked to Qwen3.8-27B: verify the numbers it hardcodes.
    expect(ix, "qwen35.block_count", 65);
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
    arena_ = std::make_unique<Arena>(static_cast<size_t>(2.0 * kGiB));
    pool_ = std::make_unique<Pool>(cfg_.threads);

    tok_embd_ = make_qmat(ix, "token_embd.weight");
    lm_head_ = make_qmat(ix, "output.weight");
    if (lm_head_.out != n_vocab) throw std::runtime_error("output.weight rows differ from the vocabulary size");
    out_norm_ = f32p(ix, "output_norm.weight", kHidden);

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
    double cum = 0;
    n_resident_layers = 0;
    const int g0 = cfg_.gpu_layers;
    for (int il = 0; il < kLayers; ++il) {
        layers_[il].span_begin = lo[il] / page * page;
        layers_[il].span_end = (hi[il] + page - 1) / page * page;
        const double gib = static_cast<double>(layers_[il].span_end - layers_[il].span_begin) / kGiB;
        if (il < g0) {  // lives in VRAM after the upload; its file pages are dropped from the page cache below
            layers_[il].resident = false;
            continue;
        }
        layers_[il].resident = (cum + gib <= cfg_.resident_gib);
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
    const size_t Tm = static_cast<size_t>(cfg_.t_max);
    x_ = arena_->alloc_n<float>(Tm * kHidden);
    xn_ = arena_->alloc_n<float>(Tm * kHidden);
    proj_ = arena_->alloc_n<float>(Tm * kHidden);
    bufA_ = arena_->alloc_n<float>(Tm * 2 * kQW);          // qkv [T][10240] or q+gate [T][12288]
    bufB_ = arena_->alloc_n<float>(Tm * kInner);           // z
    conv_silu_ = arena_->alloc_n<float>(Tm * kConvCh);
    attn_o_ = arena_->alloc_n<float>(Tm * kInner);
    alpha_ = arena_->alloc_n<float>(Tm * kVHeads);
    betar_ = arena_->alloc_n<float>(Tm * kVHeads);
    gate_ = arena_->alloc_n<float>(Tm * kVHeads);
    beta_ = arena_->alloc_n<float>(Tm * kVHeads);
    fg_ = arena_->alloc_n<float>(Tm * kFFN);
    fu_ = arena_->alloc_n<float>(Tm * kFFN);
    qbuf_ = arena_->alloc_n<float>(Tm * kQW);
    kbuf_ = arena_->alloc_n<float>(Tm * kKVW);
    vbuf_ = arena_->alloc_n<float>(Tm * kKVW);
    hlast_ = arena_->alloc_n<float>(kHidden);
    hn_ = arena_->alloc_n<float>(8 * static_cast<size_t>(kHidden));
    act_ = static_cast<uint8_t *>(arena_->alloc(qmat_act_bytes(12 /* Q4_K -> Q8_K */, kFFN, cfg_.t_max)));
    rope_cos_ = arena_->alloc_n<float>(static_cast<size_t>(cfg_.ctx_max) * (kRope / 2));
    rope_sin_ = arena_->alloc_n<float>(static_cast<size_t>(cfg_.ctx_max) * (kRope / 2));
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
    const bool roll = cfg_.mtp_steps > 0 && mtp_ok_;
    for (int il = 0; il < kLayers; ++il) {
        if (layers_[il].is_attn) {
            kcache_.push_back(arena_->alloc_n<float>(static_cast<size_t>(cfg_.ctx_max) * kKVW));
            vcache_.push_back(arena_->alloc_n<float>(static_cast<size_t>(cfg_.ctx_max) * kKVW));
        } else {
            conv_state_.push_back(arena_->alloc_n<float>(kConvCh * (kConvK - 1)));
            ssm_state_.push_back(arena_->alloc_n<float>(kVHeads * kHeadDim * kHeadDim));
            const bool cpu_layer = il >= g0;  // the GPU layers keep their snapshots in VRAM
            const size_t R = kRollT;
            snap_conv_.push_back(roll && cpu_layer ? arena_->alloc_n<float>(kConvCh * (kConvK - 1)) : nullptr);
            snap_ssm_.push_back(roll && cpu_layer ? arena_->alloc_n<float>(kVHeads * kHeadDim * kHeadDim) : nullptr);
            sv_qkv_.push_back(roll && cpu_layer ? arena_->alloc_n<float>(R * kConvCh) : nullptr);
            sv_z_.push_back(roll && cpu_layer ? arena_->alloc_n<float>(R * kInner) : nullptr);
            sv_gate_.push_back(roll && cpu_layer ? arena_->alloc_n<float>(R * kVHeads) : nullptr);
            sv_beta_.push_back(roll && cpu_layer ? arena_->alloc_n<float>(R * kVHeads) : nullptr);
        }
    }
    if (mtp_ok_) {  // KV of the MTP attention layer = cache index 16
        kcache_.push_back(arena_->alloc_n<float>(static_cast<size_t>(cfg_.ctx_max) * kKVW));
        vcache_.push_back(arena_->alloc_n<float>(static_cast<size_t>(cfg_.ctx_max) * kKVW));
    }
    reset();
    arena_->freeze();

    if (g0 > 0) {
        GpuConfig gc;
        gc.n_layers = g0;
        gc.ctx_max = cfg_.ctx_max;
        gc.t_max = cfg_.t_max;
        gc.kv_iso3 = cfg_.kv_iso3;
        gc.dp4a_mask = cfg_.dp4a_mask;
        gc.stream_prefill = cfg_.stream_min_t > 0 && g0 < kLayers;
        gc.roll_t = roll ? kRollT : 0;
        gc.use_graph = cfg_.cuda_graph;
        gc.pin_stream = cfg_.pin_stream;
        gpu_ = std::make_unique<GpuLayers>(ix, gc, eps_);
        for (int il = 0; il < g0; ++il) hint_layer(il, false);  // weights are in VRAM now: drop their file pages
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
    for (float * p : conv_state_) std::memset(p, 0, sizeof(float) * kConvCh * (kConvK - 1));
    for (float * p : ssm_state_) std::memset(p, 0, sizeof(float) * kVHeads * kHeadDim * kHeadDim);
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

void Engine::ffn(const LayerW & L, int T) {
    for (int t = 0; t < T; ++t) rmsnorm(x_ + t * kHidden, kHidden, L.post_norm, eps_, xn_ + t * kHidden);
    qmatmul(*pool_, L.ffn_gate, xn_, T, fg_, act_);
    qmatmul(*pool_, L.ffn_up, xn_, T, fu_, act_);
    const int64_t n = static_cast<int64_t>(T) * kFFN;
    pool_->parallel_for(n, 4096, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) fg_[i] = silu(fg_[i]) * fu_[i];
    });
    qmatmul(*pool_, L.ffn_down, fg_, T, proj_, act_);
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
    qmatmul(*pool_, L.qkv, xn_, T, bufA_, act_);       // [T][10240]
    qmatmul(*pool_, L.gate_z, xn_, T, bufB_, act_);    // [T][6144]
    qmatmul(*pool_, L.alpha, xn_, T, alpha_, act_);    // [T][48]
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

    qmatmul(*pool_, L.ssm_out, attn_o_, T, proj_, act_);
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
    qmatmul(*pool_, L.wq, xn_, T, bufA_, act_);   // per head: [q(256) | gate(256)]
    qmatmul(*pool_, L.wk, xn_, T, kbuf_, act_);
    qmatmul(*pool_, L.wv, xn_, T, vbuf_, act_);

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
    constexpr int kQG = kNH / kNKV;  // 6 query heads share one K/V group: their score and V passes read each row once
    pool_->parallel_for(static_cast<int64_t>(T) * kNKV, 1, [&](int64_t b, int64_t e) {
        float sc[kQG][kMaxCtxStack];
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
                for (int hh = 0; hh < kQG; ++hh) sc[hh][j] = 0.0f;
            for (int j = j0; j < n; ++j) {
                const float * krow = kc + static_cast<int64_t>(j) * kKVW + g * kHD;
                for (int hh = 0; hh < kQG; ++hh) {
                    const float s = dotf(q[hh], krow, kHD) * kq_scale;
                    sc[hh][j] = s;
                    if (s > mx[hh]) mx[hh] = s;
                }
            }
            for (int hh = 0; hh < kQG; ++hh)
                for (int j = j0; j < n; ++j) { sc[hh][j] = std::exp(sc[hh][j] - mx[hh]); sum[hh] += sc[hh][j]; }
            float o[kQG][kHD];
            for (int hh = 0; hh < kQG; ++hh) {
                for (int d = 0; d < kHD; ++d) o[hh][d] = 0.0f;
            }
            // Value pass: every V row is read once for all six query heads of the group.
            for (int j = j0; j < n; ++j) {
                const float * v = vc + static_cast<int64_t>(j) * kKVW + g * kHD;
                for (int hh = 0; hh < kQG; ++hh) {
                    const float pj = sc[hh][j] * (1.0f / sum[hh]);
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

    qmatmul(*pool_, L.wo, attn_o_, T, proj_, act_);
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
    if (h_rows || n_out > 0) {
        if (!h_rows && n_out == 1) {
            rmsnorm(x_ + static_cast<int64_t>(T - 1) * kHidden, kHidden, out_norm_, eps_, hlast_);
            qmatmul(*pool_, lm_head_, hlast_, 1, logits, act_);
        } else {
            float * hn = h_rows ? h_rows : hn_;
            const int r0 = h_rows ? 0 : T - n_out;
            for (int t = r0; t < T; ++t)
                rmsnorm(x_ + static_cast<int64_t>(t) * kHidden, kHidden, out_norm_, eps_, hn + static_cast<int64_t>(t - r0) * kHidden);
            if (n_out > 0) qmatmul(*pool_, lm_head_, hn + static_cast<int64_t>(T - n_out - r0) * kHidden, n_out, logits, act_);
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
    if (!(gpu_ && gpu_->stream_ready() && cfg_.mtp_gpu_prefill && T >= cfg_.stream_min_t)) { mtp_forward(tokens, h_in, T, pos0, nullptr, nullptr); return; }
    if (T < 1 || T > cfg_.t_max || pos0 < 0 || pos0 + T > cfg_.ctx_max) throw std::runtime_error("mtp_prefill: T or position out of range");
    for (int t = 0; t < T; ++t) {
        if (tokens[t] < 0 || tokens[t] >= n_vocab) throw std::runtime_error("mtp_prefill: token id out of range");
        qmat_row_to_f32(tok_embd_, tokens[t], proj_);
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
        float * dst = bufA_ + static_cast<int64_t>(t) * 2 * kHidden;
        rmsnorm(proj_, kHidden, mtp_enorm_, eps_, dst);
        rmsnorm(h_in + static_cast<int64_t>(t) * kHidden, kHidden, mtp_hnorm_, eps_, dst + kHidden);
    }
    qmatmul(*pool_, mtp_eh_, bufA_, T, x_, act_);
    attn_layer(64, mtp_L_, T, pos0);   // cache index 64 / 4 = 16
    ffn(mtp_L_, T);
    if (h_out || logits) {
        rmsnorm(x_ + static_cast<int64_t>(T - 1) * kHidden, kHidden, mtp_head_norm_, eps_, hlast_);
        if (h_out) std::memcpy(h_out, hlast_, sizeof(float) * kHidden);
        if (logits) qmatmul(*pool_, lm_head_, hlast_, 1, logits, act_);
    }
}
