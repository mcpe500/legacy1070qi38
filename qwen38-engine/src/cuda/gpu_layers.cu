#include "gpu_layers.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

#include "dp4a_gemv.cuh"
#include "gemv.cuh"
#include "ops.cuh"
#include "q8_gemv.cuh"
#include "ternary_gemv.cuh"
#include "ternary_spec.hpp"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) \
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e_) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); } while (0)

namespace {
double g_wait_s = 0, g_pread_s = 0, g_up_s = 0, g_comp_s = 0, g_fetch_s = 0;
double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

constexpr int kHidden = 5120, kFFN = 17408, kNH = 24, kNKV = 4, kHD = 256, kRope = 64;
constexpr int kConvCh = 10240, kConvK = 4, kVH = 48, kDH = 128, kInner = 6144;
constexpr int kQFull = kNH * 2 * kHD;   // 12288
constexpr int kQW = kNH * kHD;          // 6144
constexpr int kKVW = kNKV * kHD;        // 1024

struct WMat {
    const uint8_t * p = nullptr;
    int type = 0;
    int64_t in = 0, out = 0;
    const GgufTensor * src = nullptr;  // streamed layer: the matrix is uploaded into the staging buffer right before its GEMV
};

struct LayerD {
    bool is_attn = false;
    const float *attn_norm = nullptr, *post_norm = nullptr;
    WMat ffn_gate, ffn_up, ffn_down;
    WMat qkv, gate_z, alpha, beta, ssm_out;
    const float *ssm_norm = nullptr, *ssm_a = nullptr, *ssm_dt = nullptr, *conv_w = nullptr;
    WMat wq, wk, wv, wo;
    const float *q_norm = nullptr, *k_norm = nullptr;
    float *conv_state = nullptr, *ssm_state = nullptr;  // DeltaNet layers
    uint8_t *kcache = nullptr, *vcache = nullptr;        // attention layers
    // DeltaNet rollback buffers (cfg.roll_t > 0): state before the batch and the inputs of the recurrence.
    float *snap_conv = nullptr, *snap_ssm = nullptr, *sv_qkv = nullptr, *sv_gate = nullptr, *sv_beta = nullptr, *sv_z = nullptr;
};

// Two-pass planner: pass one only counts bytes, pass two (base != null) hands out pointers into the single allocation.
struct Planner {
    size_t off = 0;
    uint8_t * base = nullptr;
    template <class T> T * take(size_t n) {
        const size_t o = (off + 255) & ~static_cast<size_t>(255);
        off = o + n * sizeof(T);
        return base ? reinterpret_cast<T *>(base + o) : nullptr;
    }
};

}  // namespace

struct GpuLayers::Impl {
    const GgufIndex & ix;
    GpuConfig cfg;
    float eps;
    uint8_t * arena = nullptr;
    size_t arena_bytes = 0, weights = 0, kv = 0;
    Q8Scratch q8;
    ActQ8 act;  // Q8_1 activations of the current GEMV input
    const float * act_src = nullptr;  // buffer `act` was quantized from (rotated or plain)
    bool tmode = false;               // any weight of the GPU layers is ternary (PTQ1_0 / Q2_0)
    const float * sgn[3] = {nullptr, nullptr, nullptr};   // device sign vectors for widths 5120 / 6144 / 17408
    float * xr5120 = nullptr, *xr6144 = nullptr, *xr17408 = nullptr;  // rotated activations [t_max][width]
    float * xr_of(int sidx) const { return sidx == 0 ? xr5120 : (sidx == 1 ? xr6144 : xr17408); }
    float * h_stage = nullptr;  // pinned staging buffer
    // GPU LM head (cfg.head)
    WMat hd_w;
    const float * hd_norm = nullptr;
    float * hd_lg = nullptr;     // device logits [8][V]
    float * attn_part = nullptr; // split-K attention partials (ctx_max > 8192)
    // MTP block (cfg.mtp)
    LayerD mtpL;
    WMat eh_w;
    const float * mtp_hn = nullptr;
    float * h_lg = nullptr;      // pinned host logits [8][V]
    int64_t hd_v = 0;
    size_t hd_bytes = 0;
    std::vector<LayerD> L;
    // scratch
    float *x = nullptr, *xn = nullptr, *proj = nullptr, *bufA = nullptr, *bufB = nullptr, *conv = nullptr, *attn_o = nullptr;
    float *alpha = nullptr, *betar = nullptr, *gate = nullptr, *beta = nullptr, *fg = nullptr, *fu = nullptr;
    float *qbuf = nullptr, *kproj = nullptr, *kout = nullptr, *vproj = nullptr;
    float theta_scale = 0.0f;
    float * rope_cos = nullptr, * rope_sin = nullptr;   // [ctx_max][32] angle tables, generated once on the device
    // Prefill streaming: one staging buffer for the weights of the layer being run, plus one layer of state and FP32 KV.
    uint8_t * stage = nullptr;   // one matrix at a time
    uint8_t * stage2 = nullptr;  // second staging buffer: the next matrix uploads while the current one is multiplied
    // Overlapped streaming (needs the CPU-layer part of the mmap registered as pinned, read-only host memory): the copy stream `cp`
    // uploads matrix k+1 into the other staging buffer while `st` multiplies matrix k. seq is the order of the GEMV calls.
    cudaStream_t cp = nullptr;
    cudaEvent_t up_done[2] = {}, comp_done[2] = {};
    std::vector<const GgufTensor *> seq;
    size_t cur = 0, issued = 0;
    bool registered = false;
    void * reg_ptr = nullptr;
    uint8_t * svec = nullptr;    // the small vectors of one layer
    size_t stage_bytes = 0;
    float *s_conv = nullptr, *s_ssm = nullptr, *s_kc = nullptr, *s_vc = nullptr;
    bool saving = false;
    // Work stream (blocking, so it stays ordered with the legacy-stream cudaMemcpy calls) and CUDA graphs of the whole GPU layer stack
    // for decode and verification batches (T = 1..4, with and without snapshots). Replays read the position from pos_dev.
    cudaStream_t st = nullptr;
    int * pos_dev = nullptr;
    int * h_pos = nullptr;
    bool in_graph = false;
    cudaGraphExec_t gexec[10] = {};   // T=1..5 × (saving off/on) = 10 entries
    int n_graphs = 0;

    Impl(const GgufIndex & i, const GpuConfig & c, float e) : ix(i), cfg(c), eps(e) {}

    // Pinned double buffer: pread() from the page cache into pinned memory (no per-page mmap faults), then an async DMA copy.
    static constexpr size_t kChunk = 32u << 20;
    uint8_t * pin[2] = {nullptr, nullptr};
    cudaEvent_t pin_done[2] = {nullptr, nullptr};
    void pin_alloc() {
        for (int i = 0; i < 2; ++i) {
            CK(cudaHostAlloc(reinterpret_cast<void **>(&pin[i]), kChunk, cudaHostAllocDefault));
            CK(cudaEventCreateWithFlags(&pin_done[i], cudaEventDisableTiming));
            CK(cudaEventRecord(pin_done[i], st));
        }
    }
    void pin_free() {
        for (int i = 0; i < 2; ++i) {
            if (pin_done[i]) cudaEventDestroy(pin_done[i]);
            if (pin[i]) cudaFreeHost(pin[i]);
            pin[i] = nullptr;
            pin_done[i] = nullptr;
        }
    }
    void upload_tensor(void * dst, const GgufTensor & t) {
        if (!pin[0]) { CK(cudaMemcpy(dst, ix.data(t), t.nbytes, cudaMemcpyHostToDevice)); return; }
        const off_t base = static_cast<off_t>(ix.data_start + t.offset);
        size_t done = 0;
        int b = 0;
        while (done < t.nbytes) {
            const size_t n = std::min(kChunk, static_cast<size_t>(t.nbytes) - done);
            const double w0 = now_s();
            CK(cudaEventSynchronize(pin_done[b]));   // the previous copy out of this buffer has finished
            const double w1 = now_s();
            g_wait_s += w1 - w0;
            size_t got = 0;
            while (got < n) {
                const ssize_t r = pread(ix.fd(), pin[b] + got, n - got, base + static_cast<off_t>(done + got));
                if (r <= 0) throw std::runtime_error("pread failed while uploading " + std::to_string(t.nbytes) + " bytes");
                got += static_cast<size_t>(r);
            }
            g_pread_s += now_s() - w1;
            CK(cudaMemcpyAsync(static_cast<uint8_t *>(dst) + done, pin[b], n, cudaMemcpyHostToDevice, st));
            CK(cudaEventRecord(pin_done[b], st));
            done += n;
            b ^= 1;
        }
    }

    WMat mat(Planner & pl, const std::string & name, bool upload) {
        const GgufTensor & t = ix.get(name);
        if (t.ne[2] != 1 || t.ne[3] != 1) throw std::runtime_error("not a matrix: " + name);
        const int type = static_cast<int>(t.type);
        if (ternary::is_ternary(type)) {  // PTQ1_0 / Q2_0: decoded by the ternary dp4a kernels, no ggml row size
            const size_t rb = type == ternary::kTypePtq1_0
                                  ? static_cast<size_t>(t.ne[0] / ternary::kPtqGroup) * ternary::kPtqGroupBytes
                                  : static_cast<size_t>(t.ne[0] / ternary::kQ2_0Block) * ternary::kQ2_0BlockBytes;
            if (static_cast<int64_t>(t.nbytes) != static_cast<int64_t>(rb) * t.ne[1]) throw std::runtime_error("byte size mismatch in " + name);
            if (lazy_plan) { lazy_max = std::max(lazy_max, static_cast<size_t>(t.nbytes)); return WMat{nullptr, type, t.ne[0], t.ne[1], &t}; }
            uint8_t * d = pl.take<uint8_t>(t.nbytes);
            if (upload) upload_tensor(d, t);
            weights += t.nbytes;
            return WMat{d, type, t.ne[0], t.ne[1], nullptr};
        }
        if (!gemv_fp32_supported(type)) throw std::runtime_error("unsupported quantization in " + name + ": type " + std::to_string(type));
        if (static_cast<int64_t>(t.nbytes) != gemv_row_bytes(type, t.ne[0]) * t.ne[1]) throw std::runtime_error("byte size mismatch in " + name);
        if (lazy_plan) {
            lazy_max = std::max(lazy_max, static_cast<size_t>(t.nbytes));
            return WMat{nullptr, type, t.ne[0], t.ne[1], &t};
        }
        uint8_t * d = pl.take<uint8_t>(t.nbytes);
        if (upload) upload_tensor(d, t);
        weights += t.nbytes;
        return WMat{d, type, t.ne[0], t.ne[1], nullptr};
    }

    const float * vec(Planner & pl, const std::string & name, int64_t n, bool upload) {
        const GgufTensor & t = ix.get(name);
        if (t.type != 0 || t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3] != n) throw std::runtime_error("expected F32 vector of " + std::to_string(n) + ": " + name);
        float * d = pl.take<float>(static_cast<size_t>(n));
        if (upload) upload_tensor(d, t);
        if (!lazy_plan) weights += static_cast<size_t>(n) * 4;
        return d;
    }

    // Streamed layer: the small vectors are uploaded at once into the planner's region, the matrices stay lazy (see WMat::src).
    bool lazy_plan = false;
    size_t lazy_max = 0;
    void plan_stream_layer(int il, Planner & pl, LayerD & l, bool upload) {
        lazy_plan = true;
        const std::string b = "blk." + std::to_string(il) + ".";
        l.is_attn = (il % 4 == 3);
        l.attn_norm = vec(pl, b + "attn_norm.weight", kHidden, upload);
        l.post_norm = vec(pl, b + "post_attention_norm.weight", kHidden, upload);
        l.ffn_gate = mat(pl, b + "ffn_gate.weight", upload);
        l.ffn_up = mat(pl, b + "ffn_up.weight", upload);
        l.ffn_down = mat(pl, b + "ffn_down.weight", upload);
        if (l.is_attn) {
            l.wq = mat(pl, b + "attn_q.weight", upload);
            l.wk = mat(pl, b + "attn_k.weight", upload);
            l.wv = mat(pl, b + "attn_v.weight", upload);
            l.wo = mat(pl, b + "attn_output.weight", upload);
            l.q_norm = vec(pl, b + "attn_q_norm.weight", kHD, upload);
            l.k_norm = vec(pl, b + "attn_k_norm.weight", kHD, upload);
        } else {
            l.qkv = mat(pl, b + "attn_qkv.weight", upload);
            l.gate_z = mat(pl, b + "attn_gate.weight", upload);
            l.alpha = mat(pl, b + "ssm_alpha.weight", upload);
            l.beta = mat(pl, b + "ssm_beta.weight", upload);
            l.ssm_out = mat(pl, b + "ssm_out.weight", upload);
            l.ssm_norm = vec(pl, b + "ssm_norm.weight", kDH, upload);
            l.ssm_a = vec(pl, b + "ssm_a", kVH, upload);
            l.ssm_dt = vec(pl, b + "ssm_dt.bias", kVH, upload);
            l.conv_w = vec(pl, b + "ssm_conv1d.weight", static_cast<int64_t>(kConvCh) * kConvK, upload);
        }
        lazy_plan = false;
    }

    // Runs once with base == nullptr to size the arena, then again with the real base to assign pointers and upload.
    void plan(Planner & pl, bool upload) {
        weights = kv = 0;
        L.assign(static_cast<size_t>(cfg.n_layers), LayerD{});
        for (int il = 0; il < cfg.n_layers; ++il) {
            LayerD & l = L[static_cast<size_t>(il)];
            const std::string b = "blk." + std::to_string(il) + ".";
            l.is_attn = (il % 4 == 3);
            l.attn_norm = vec(pl, b + "attn_norm.weight", kHidden, upload);
            l.post_norm = vec(pl, b + "post_attention_norm.weight", kHidden, upload);
            l.ffn_gate = mat(pl, b + "ffn_gate.weight", upload);
            l.ffn_up = mat(pl, b + "ffn_up.weight", upload);
            l.ffn_down = mat(pl, b + "ffn_down.weight", upload);
            if (l.is_attn) {
                l.wq = mat(pl, b + "attn_q.weight", upload);
                l.wk = mat(pl, b + "attn_k.weight", upload);
                l.wv = mat(pl, b + "attn_v.weight", upload);
                l.wo = mat(pl, b + "attn_output.weight", upload);
                l.q_norm = vec(pl, b + "attn_q_norm.weight", kHD, upload);
                l.k_norm = vec(pl, b + "attn_k_norm.weight", kHD, upload);
                const size_t unit = cfg.kv_iso3 ? 400 : static_cast<size_t>(kKVW) * 4;  // bytes per position and tensor
                l.kcache = pl.take<uint8_t>(static_cast<size_t>(cfg.ctx_max) * unit);
                l.vcache = pl.take<uint8_t>(static_cast<size_t>(cfg.ctx_max) * unit);
                kv += 2 * static_cast<size_t>(cfg.ctx_max) * unit;
            } else {
                l.qkv = mat(pl, b + "attn_qkv.weight", upload);
                l.gate_z = mat(pl, b + "attn_gate.weight", upload);
                l.alpha = mat(pl, b + "ssm_alpha.weight", upload);
                l.beta = mat(pl, b + "ssm_beta.weight", upload);
                l.ssm_out = mat(pl, b + "ssm_out.weight", upload);
                l.ssm_norm = vec(pl, b + "ssm_norm.weight", kDH, upload);
                l.ssm_a = vec(pl, b + "ssm_a", kVH, upload);
                l.ssm_dt = vec(pl, b + "ssm_dt.bias", kVH, upload);
                l.conv_w = vec(pl, b + "ssm_conv1d.weight", static_cast<int64_t>(kConvCh) * kConvK, upload);
                l.conv_state = pl.take<float>(static_cast<size_t>(kConvCh) * (kConvK - 1));
                l.ssm_state = pl.take<float>(static_cast<size_t>(kVH) * kDH * kDH);
                if (cfg.roll_t > 0) {
                    const size_t R = static_cast<size_t>(cfg.roll_t);
                    l.snap_conv = pl.take<float>(static_cast<size_t>(kConvCh) * (kConvK - 1));
                    l.snap_ssm = pl.take<float>(static_cast<size_t>(kVH) * kDH * kDH);
                    l.sv_qkv = pl.take<float>(R * kConvCh);
                    l.sv_gate = pl.take<float>(R * kVH);
                    l.sv_beta = pl.take<float>(R * kVH);
                    l.sv_z = pl.take<float>(R * kInner);
                }
            }
        }
        stage_bytes = 0;
        if (cfg.stream_prefill && cfg.n_layers < 64) {
            size_t vec_bytes = 0;
            lazy_max = 0;
            const size_t w0 = weights;
            for (int il = cfg.n_layers; il < 64; ++il) {
                Planner c;
                LayerD tmp;
                plan_stream_layer(il, c, tmp, false);
                vec_bytes = std::max(vec_bytes, c.off);
            }
            weights = w0;
            stage_bytes = lazy_max;
            svec = pl.take<uint8_t>(vec_bytes);
            stage = pl.take<uint8_t>(stage_bytes);
            stage2 = pl.take<uint8_t>(stage_bytes);
            s_conv = pl.take<float>(static_cast<size_t>(kConvCh) * (kConvK - 1));
            s_ssm = pl.take<float>(static_cast<size_t>(kVH) * kDH * kDH);
            s_kc = pl.take<float>(static_cast<size_t>(cfg.ctx_max) * kKVW);
            s_vc = pl.take<float>(static_cast<size_t>(cfg.ctx_max) * kKVW);
        }
        const size_t Tm = static_cast<size_t>(cfg.t_max);
        x = pl.take<float>(Tm * kHidden);
        xn = pl.take<float>(Tm * kHidden);
        proj = pl.take<float>(Tm * kHidden);
        bufA = pl.take<float>(Tm * kQFull);
        bufB = pl.take<float>(Tm * kInner);
        conv = pl.take<float>(Tm * kConvCh);
        attn_o = pl.take<float>(Tm * kInner);
        alpha = pl.take<float>(Tm * kVH);
        betar = pl.take<float>(Tm * kVH);
        gate = pl.take<float>(Tm * kVH);
        beta = pl.take<float>(Tm * kVH);
        fg = pl.take<float>(Tm * kFFN);
        fu = pl.take<float>(Tm * kFFN);
        qbuf = pl.take<float>(Tm * kQW);
        kproj = pl.take<float>(Tm * kKVW);
        kout = pl.take<float>(Tm * kKVW);
        vproj = pl.take<float>(Tm * kKVW);
        if (cfg.signs_5120) {  // Ternary Bonsai mode: sign vectors plus one rotated activation buffer per width
            tmode = true;
            if (upload) ternary_ptq1_init();  // constant-memory upload, never inside a graph capture
            const size_t Tm2 = static_cast<size_t>(cfg.t_max);
            xr5120 = pl.take<float>(Tm2 * kHidden);
            xr6144 = pl.take<float>(Tm2 * kInner);
            xr17408 = pl.take<float>(Tm2 * kFFN);
            sgn[0] = pl.take<float>(kHidden);
            sgn[1] = pl.take<float>(kInner);
            sgn[2] = pl.take<float>(kFFN);
            if (upload) {
                CK(cudaMemcpy(const_cast<float *>(sgn[0]), cfg.signs_5120, sizeof(float) * kHidden, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(const_cast<float *>(sgn[1]), cfg.signs_6144, sizeof(float) * kInner, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(const_cast<float *>(sgn[2]), cfg.signs_17408, sizeof(float) * kFFN, cudaMemcpyHostToDevice));
                weights += (static_cast<size_t>(kHidden) + kInner + kFFN) * sizeof(float);
            }
        }
        if (cfg.mtp) {
            const std::string b = "blk.64.";
            mtpL = LayerD{};
            mtpL.is_attn = true;
            mtpL.attn_norm = vec(pl, b + "attn_norm.weight", kHidden, upload);
            mtpL.post_norm = vec(pl, b + "post_attention_norm.weight", kHidden, upload);
            mtpL.ffn_gate = mat(pl, b + "ffn_gate.weight", upload);
            mtpL.ffn_up = mat(pl, b + "ffn_up.weight", upload);
            mtpL.ffn_down = mat(pl, b + "ffn_down.weight", upload);
            mtpL.wq = mat(pl, b + "attn_q.weight", upload);
            mtpL.wk = mat(pl, b + "attn_k.weight", upload);
            mtpL.wv = mat(pl, b + "attn_v.weight", upload);
            mtpL.wo = mat(pl, b + "attn_output.weight", upload);
            mtpL.q_norm = vec(pl, b + "attn_q_norm.weight", kHD, upload);
            mtpL.k_norm = vec(pl, b + "attn_k_norm.weight", kHD, upload);
            const size_t unit = cfg.kv_iso3 ? 400 : static_cast<size_t>(kKVW) * 4;
            mtpL.kcache = pl.take<uint8_t>(static_cast<size_t>(cfg.ctx_max) * unit);
            mtpL.vcache = pl.take<uint8_t>(static_cast<size_t>(cfg.ctx_max) * unit);
            kv += 2 * static_cast<size_t>(cfg.ctx_max) * unit;
            eh_w = mat(pl, b + "nextn.eh_proj.weight", upload);
            mtp_hn = vec(pl, b + "nextn.shared_head_norm.weight", kHidden, upload);
        }
        if (cfg.head) {
            const size_t w0 = weights;
            hd_norm = vec(pl, "output_norm.weight", kHidden, upload);
            hd_w = mat(pl, "output.weight", upload);
            hd_v = hd_w.out;
            hd_lg = pl.take<float>(static_cast<size_t>(8) * static_cast<size_t>(hd_v));
            hd_bytes = weights - w0;
        }
        // split-K decode attention: above 8192 positions, and at every context on the ternary path (where the shared-memory
        // kernel's serial value pass costs about 1.5 ms per layer at 4K positions)
        attn_part = (cfg.ctx_max > 8192 || cfg.signs_5120) ? pl.take<float>(op_attention_part_floats()) : nullptr;
        rope_cos = pl.take<float>(static_cast<size_t>(cfg.ctx_max) * kRope / 2);
        rope_sin = pl.take<float>(static_cast<size_t>(cfg.ctx_max) * kRope / 2);
        pos_dev = pl.take<int>(64);
    }

    bool use_dp4a(const WMat & W) const { return W.type != GT_Q8_0 && gemv_dp4a_supported(W.type) && ((cfg.dp4a_mask >> W.type) & 1u); }

    // Quantizes `x` once for all the listed matrices that take the dp4a path (nothing happens when none does).
    void quant(const float * x, int64_t in, int T, std::initializer_list<const WMat *> mats) {
        for (const WMat * w : mats) {
            if (use_dp4a(*w)) { CK(actq8_quantize(x, in, T, act, st)); act_src = x; return; }
        }
    }

    // Ternary mode: rotate `x` into the Hadamard basis of width slot sidx (0/1/2 for 5120/6144/17408) and
    // quantize the rotated vector for the listed matrices. Unrotated plain-basis matrices keep using quant().
    void quant_rot(std::initializer_list<const WMat *> mats, const float * x, int64_t in, int T, int sidx) {
        bool any = false;
        for (const WMat * w : mats) if (ternary::is_ternary(w->type)) any = true;
        if (!any || !sgn[sidx]) { quant(x, in, T, mats); return; }
        CK(ternary_rotate_q8(x, sgn[sidx], xr_of(sidx), static_cast<int>(in), T, act, st));   // rotate + Q8_1 in one launch
        act_src = xr_of(sidx);
    }

    // The caller must have run quant() for the same `in` buffer when use_dp4a(W) is true.
    void gemv(const WMat & W0, const float * in, int T, float * out) {
        WMat W = W0;
        int b = -1;
        if (W.src && registered) {   // overlapped: the matrix was (or is being) uploaded by the copy stream
            if (cur >= seq.size() || seq[cur] != W.src) throw std::runtime_error("streamed GEMV order differs from the upload order");
            b = static_cast<int>(cur & 1);
            CK(cudaStreamWaitEvent(st, up_done[b], 0));
            W.p = b ? stage2 : stage;
        } else if (W.src) {  // fallback: synchronous upload into the staging buffer
            upload_tensor(stage, *W.src);
            W.p = stage;
        }
        if (ternary::is_ternary(W.type)) {
            if (W.type == ternary::kTypePtq1_0) CK(gemv_ptq1_0_dp4a(W.p, W.in, W.out, act, T, out, st));
            else CK(gemv_q2_0_dp4a(W.p, W.in, W.out, act, T, out, st));
        } else if (W.type == GT_Q8_0) CK(q8_0_gemv_dp4a(W.p, W.in, W.out, in, T, out, q8, st));
        else if (use_dp4a(W)) {
            if (act_src != in) {  // plain-basis matrix after a rotated quant (alpha/beta of a ternary layer)
                CK(actq8_quantize(in, W.in, T, act, st));
                act_src = in;
            }
            CK(gemv_dp4a(W.type, W.p, W.in, W.out, act, T, out, st));
        }
        else CK(gemv_fp32(W.type, W.p, W.in, W.out, in, T, out, st));
        if (b >= 0) {
            CK(cudaEventRecord(comp_done[b], st));
            ++cur;
            issue_uploads();
        }
    }

    // Keeps up to two uploads ahead of the multiplications. Upload k reuses buffer k & 1 once GEMV k - 2 has finished.
    void issue_uploads() {
        while (issued < seq.size() && issued < cur + 2) {
            const int b = static_cast<int>(issued & 1);
            CK(cudaStreamWaitEvent(cp, comp_done[b], 0));
            const GgufTensor & t = *seq[issued];
            CK(cudaMemcpyAsync(b ? stage2 : stage, ix.data(t), t.nbytes, cudaMemcpyHostToDevice, cp));
            CK(cudaEventRecord(up_done[b], cp));
            ++issued;
        }
    }

    void build_seq() {
        seq.clear();
        for (int il = cfg.n_layers; il < 64; ++il) {
            const std::string b = "blk." + std::to_string(il) + ".";
            const char * attn[] = {"attn_q", "attn_k", "attn_v", "attn_output"};
            const char * dn[] = {"attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta", "ssm_out"};
            if (il % 4 == 3) for (const char * n : attn) seq.push_back(&ix.get(b + n + ".weight"));
            else for (const char * n : dn) seq.push_back(&ix.get(b + n + ".weight"));
            for (const char * n : {"ffn_gate", "ffn_up", "ffn_down"}) seq.push_back(&ix.get(b + n + ".weight"));
        }
    }

    void run_layers(int T, int pos0) {
        for (int il = 0; il < cfg.n_layers; ++il) {
            const LayerD & l = L[static_cast<size_t>(il)];
            if (l.is_attn) attention(l, T, pos0, cfg.kv_iso3);
            else deltanet(l, T);
            ffn(l, T);
        }
    }

    void ffn(const LayerD & l, int T) {
        CK(op_rmsnorm_rows(x, l.post_norm, xn, kHidden, T, eps, st));
        quant_rot({&l.ffn_gate, &l.ffn_up}, xn, kHidden, T, 0);
        gemv(l.ffn_gate, xn, T, fg);
        gemv(l.ffn_up, xn, T, fu);
        CK(op_silu_mul(fg, fu, static_cast<int64_t>(T) * kFFN, st));
        quant_rot({&l.ffn_down}, fg, kFFN, T, 2);
        gemv(l.ffn_down, fg, T, proj);
        CK(op_add_inplace(x, proj, static_cast<int64_t>(T) * kHidden, st));
    }

    void deltanet(const LayerD & l, int T) {
        if (saving) {
            CK(cudaMemcpyAsync(l.snap_conv, l.conv_state, sizeof(float) * kConvCh * (kConvK - 1), cudaMemcpyDeviceToDevice, st));
            CK(cudaMemcpyAsync(l.snap_ssm, l.ssm_state, sizeof(float) * kVH * kDH * kDH, cudaMemcpyDeviceToDevice, st));
        }
        CK(op_rmsnorm_rows(x, l.attn_norm, xn, kHidden, T, eps, st));
        quant_rot({&l.qkv, &l.gate_z}, xn, kHidden, T, 0);
        gemv(l.qkv, xn, T, bufA);
        gemv(l.gate_z, xn, T, bufB);
        if (l.alpha.type == GT_BF16 && l.beta.type == GT_BF16 && l.alpha.out == l.beta.out && l.alpha.in == l.beta.in &&
            !l.alpha.src && !l.beta.src) {
            // plain basis (ssm_alpha/ssm_beta are not Hadamard-folded): both 48-row BF16 GEMVs in one launch
            CK(gemv_fp32_pair(GT_BF16, l.alpha.p, l.beta.p, l.alpha.in, l.alpha.out, xn, T, alpha, betar, st));
        } else {
            gemv(l.alpha, xn, T, alpha);
            gemv(l.beta, xn, T, betar);
        }
        CK(op_dn_prep(alpha, betar, l.ssm_dt, l.ssm_a, gate, beta, T, st));
        if (saving) {
            CK(cudaMemcpyAsync(l.sv_qkv, bufA, sizeof(float) * T * kConvCh, cudaMemcpyDeviceToDevice, st));
            CK(cudaMemcpyAsync(l.sv_gate, gate, sizeof(float) * T * kVH, cudaMemcpyDeviceToDevice, st));
            CK(cudaMemcpyAsync(l.sv_beta, beta, sizeof(float) * T * kVH, cudaMemcpyDeviceToDevice, st));
            CK(cudaMemcpyAsync(l.sv_z, bufB, sizeof(float) * T * kInner, cudaMemcpyDeviceToDevice, st));
        }
        CK(op_dn_conv(bufA, l.conv_w, l.conv_state, conv, T, st));
        CK(op_dn_rec(conv, gate, beta, bufB, l.ssm_norm, l.ssm_state, attn_o, T, eps, st));
        if (tmode && cfg.gdn_v_grouped) {  // ssm_out reads its value heads in grouped order (prism metadata)
            CK(op_gdn_v_permute(attn_o, xr6144, T, st));
            CK(ternary_rotate_q8(xr6144, sgn[1], xr6144, kInner, T, act, st));
            act_src = xr6144;
        } else {
            quant_rot({&l.ssm_out}, attn_o, kInner, T, 1);
        }
        gemv(l.ssm_out, attn_o, T, proj);
        CK(op_add_inplace(x, proj, static_cast<int64_t>(T) * kHidden, st));
    }

    void attention(const LayerD & l, int T, int pos0, bool iso3) {
        CK(op_rmsnorm_rows(x, l.attn_norm, xn, kHidden, T, eps, st));
        quant_rot({&l.wq, &l.wk, &l.wv}, xn, kHidden, T, 0);
        gemv(l.wq, xn, T, bufA);
        gemv(l.wk, xn, T, kproj);
        gemv(l.wv, xn, T, vproj);
        const int * pd = in_graph ? pos_dev : nullptr;
        CK(op_qk_prep(bufA, kproj, l.q_norm, l.k_norm, qbuf, kout, T, pos0, rope_cos, rope_sin, eps, st, pd));
        CK(op_kv_store(kout, l.kcache, T, pos0, iso3, st, pd));
        CK(op_kv_store(vproj, l.vcache, T, pos0, iso3, st, pd));
        CK(op_attention(qbuf, bufA, l.kcache, l.vcache, attn_o, T, pos0, iso3, cfg.ctx_max, st, pd, attn_part, tmode));
        quant_rot({&l.wo}, attn_o, kInner, T, 1);
        gemv(l.wo, attn_o, T, proj);
        CK(op_add_inplace(x, proj, static_cast<int64_t>(T) * kHidden, st));
    }
};

GpuLayers::GpuLayers(const GgufIndex & ix, const GpuConfig & cfg, float rms_eps) : p_(new Impl(ix, cfg, rms_eps)) {
    Impl & m = *p_;
    if (cfg.n_layers < 1 || cfg.n_layers > 64) throw std::runtime_error("n_layers out of range");
    // Up to 8192 positions the attention kernel keeps its scores in dynamic shared memory (32 KiB, under the 48 KiB
    // sm_61 limit). Above that the tiled online-softmax kernel runs, so ctx_max can reach 131072 (Bonsai path).
    if (cfg.ctx_max > 131072) throw std::runtime_error("ctx_max above 131072 is not supported");
    m.theta_scale = std::pow(10000000.0f, -2.0f / static_cast<float>(kRope));

    CK(cudaStreamCreate(&m.st));
    CK(cudaHostAlloc(reinterpret_cast<void **>(&m.h_pos), sizeof(int), cudaHostAllocDefault));
    Planner count;
    m.plan(count, false);
    m.arena_bytes = count.off;
    {
        // VRAM budget (process total, CUDA context included): refuse the plan before the big allocation.
        const double proj = projected_mib(ix, cfg);
        std::fprintf(stderr, "VRAM plan: arena %.1f MiB, projected process total %.1f MiB (budget %.0f)\n",
                     static_cast<double>(m.arena_bytes) / 1048576.0, proj, cfg.vram_budget_mib);
        if (proj > cfg.vram_budget_mib) {
            char msg[256];
            std::snprintf(msg, sizeof(msg), "GPU plan needs %.0f MiB of device memory (arena %.0f MiB, %d layers, ctx %d), budget %.0f MiB: "
                          "use fewer --gpu-layers, a smaller --ctx, or --gpu-layers auto",
                          proj, static_cast<double>(m.arena_bytes) / 1048576.0, cfg.n_layers, cfg.ctx_max, cfg.vram_budget_mib);
            cudaFreeHost(m.h_pos);
            cudaStreamDestroy(m.st);
            m.h_pos = nullptr;
            m.st = nullptr;
            throw std::runtime_error(msg);
        }
    }
    CK(cudaMalloc(&m.arena, m.arena_bytes));
    Planner real;
    real.base = m.arena;
    m.pin_alloc();
    m.plan(real, true);
    CK(cudaDeviceSynchronize());
    if (!m.stage) m.pin_free();   // kept for the streamed prefill uploads otherwise
    if (m.stage) {
        CK(cudaStreamCreateWithFlags(&m.cp, cudaStreamNonBlocking));
        for (int i = 0; i < 2; ++i) {
            CK(cudaEventCreateWithFlags(&m.up_done[i], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&m.comp_done[i], cudaEventDisableTiming));
        }
        // Register the file pages of the streamed layers as pinned read-only host memory, so the DMA reads the page cache directly.
        uintptr_t lo = UINTPTR_MAX, hi = 0;
        for (const auto & kv : ix.tensors) {
            if (kv.first.compare(0, 4, "blk.") != 0) continue;
            const int il = std::atoi(kv.first.c_str() + 4);
            if (il < cfg.n_layers || il >= 64) continue;
            const uintptr_t a = reinterpret_cast<uintptr_t>(ix.data(kv.second));
            lo = std::min(lo, a);
            hi = std::max(hi, a + static_cast<uintptr_t>(kv.second.nbytes));
        }
        const uintptr_t pg = 4096;
        lo = lo / pg * pg;
        hi = (hi + pg - 1) / pg * pg;
        const bool want = cfg.pin_stream;
        if (want && hi > lo && cudaHostRegister(reinterpret_cast<void *>(lo), hi - lo, cudaHostRegisterReadOnly) == cudaSuccess) {
            m.registered = true;
            m.reg_ptr = reinterpret_cast<void *>(lo);
            std::fprintf(stderr, "stream: %.2f GiB of CPU-layer weights registered as pinned memory (overlapped uploads)\n",
                         static_cast<double>(hi - lo) / (1024.0 * 1024.0 * 1024.0));
        } else {
            if (want) std::fprintf(stderr, "stream: cudaHostRegister failed (%s), synchronous uploads\n", cudaGetErrorString(cudaGetLastError()));
            cudaGetLastError();
        }
    }
    CK(q8_scratch_alloc(m.q8, cfg.t_max, kHidden));
    CK(actq8_alloc(m.act, cfg.t_max, kFFN));
    CK(op_rope_table(m.rope_cos, m.rope_sin, cfg.ctx_max, m.theta_scale, m.st));
    if (cfg.head) CK(cudaHostAlloc(reinterpret_cast<void **>(&m.h_lg), sizeof(float) * 8 * static_cast<size_t>(m.hd_v), cudaHostAllocDefault));
    CK(cudaHostAlloc(reinterpret_cast<void **>(&m.h_stage), std::max(static_cast<size_t>(cfg.t_max) * kHidden, static_cast<size_t>(kVH) * kDH * kDH) * sizeof(float), cudaHostAllocDefault));
    reset();
    CK(cudaDeviceSynchronize());
}

GpuLayers::~GpuLayers() {
    if (!p_) return;
    p_->pin_free();
    for (cudaGraphExec_t g : p_->gexec) if (g) cudaGraphExecDestroy(g);
    if (p_->registered) cudaHostUnregister(p_->reg_ptr);
    for (int i = 0; i < 2; ++i) {
        if (p_->up_done[i]) cudaEventDestroy(p_->up_done[i]);
        if (p_->comp_done[i]) cudaEventDestroy(p_->comp_done[i]);
    }
    if (p_->cp) cudaStreamDestroy(p_->cp);
    if (p_->h_pos) cudaFreeHost(p_->h_pos);
    if (p_->st) cudaStreamDestroy(p_->st);
    q8_scratch_free(p_->q8);
    actq8_free(p_->act);
    if (p_->h_stage) cudaFreeHost(p_->h_stage);
    if (p_->h_lg) cudaFreeHost(p_->h_lg);
    if (p_->arena) cudaFree(p_->arena);
}

void GpuLayers::reset() {
    for (LayerD & l : p_->L) {
        if (l.conv_state) CK(cudaMemset(l.conv_state, 0, sizeof(float) * kConvCh * (kConvK - 1)));
        if (l.ssm_state) CK(cudaMemset(l.ssm_state, 0, sizeof(float) * kVH * kDH * kDH));
    }
}

void GpuLayers::forward(float * x_host, int T, int pos0, const std::function<void(int, const float *, int)> * tap) {
    Impl & m = *p_;
    if (T < 1 || T > m.cfg.t_max || pos0 < 0 || pos0 + T > m.cfg.ctx_max) throw std::runtime_error("GpuLayers::forward: T or position out of range");
    if (m.saving && T > m.cfg.roll_t) throw std::runtime_error("GpuLayers::forward: batch larger than roll_t with snapshots on");
    const size_t bytes = static_cast<size_t>(T) * kHidden * sizeof(float);
    CK(cudaMemcpy(m.x, x_host, bytes, cudaMemcpyHostToDevice));
    if (m.cfg.use_graph && !tap && T <= 5) {
        // One CUDA graph per (T, snapshots on/off) holds every kernel launch of the layer stack; replaying it costs one launch.
        const int key = (T - 1) * 2 + (m.saving ? 1 : 0);
        *m.h_pos = pos0;
        CK(cudaMemcpyAsync(m.pos_dev, m.h_pos, sizeof(int), cudaMemcpyHostToDevice, m.st));
        if (!m.gexec[key]) {
            cudaGraph_t g = nullptr;
            CK(cudaStreamBeginCapture(m.st, cudaStreamCaptureModeThreadLocal));
            m.in_graph = true;
            try { m.run_layers(T, pos0); } catch (...) { m.in_graph = false; cudaGraph_t dead; cudaStreamEndCapture(m.st, &dead); throw; }
            m.in_graph = false;
            CK(cudaStreamEndCapture(m.st, &g));
            CK(cudaGraphInstantiate(&m.gexec[key], g, 0));
            CK(cudaGraphDestroy(g));
        }
        CK(cudaGraphLaunch(m.gexec[key], m.st));
        CK(cudaMemcpy(x_host, m.x, bytes, cudaMemcpyDeviceToHost));
        return;
    }
    for (int il = 0; il < m.cfg.n_layers; ++il) {
        const LayerD & l = m.L[static_cast<size_t>(il)];
        if (l.is_attn) m.attention(l, T, pos0, m.cfg.kv_iso3);
        else m.deltanet(l, T);
        m.ffn(l, T);
        if (tap) {
            CK(cudaMemcpy(m.h_stage, m.x, bytes, cudaMemcpyDeviceToHost));
            (*tap)(il, m.h_stage, T);
        }
    }
    CK(cudaMemcpy(x_host, m.x, bytes, cudaMemcpyDeviceToHost));
}

bool GpuLayers::has_head() const { return p_->cfg.head; }
bool GpuLayers::has_mtp() const { return p_->cfg.mtp; }

void GpuLayers::mtp_forward(const float * concat_host, int T, int pos0, float * h_out, float * logits) {
    Impl & m = *p_;
    if (!m.cfg.mtp) throw std::runtime_error("GpuLayers::mtp_forward: no MTP block on the GPU");
    if (T < 1 || T > m.cfg.t_max || pos0 < 0 || pos0 + T > m.cfg.ctx_max) throw std::runtime_error("GpuLayers::mtp_forward: T or position out of range");
    CK(cudaMemcpy(m.conv, concat_host, static_cast<size_t>(T) * 2 * kHidden * sizeof(float), cudaMemcpyHostToDevice));
    m.quant(m.conv, 2 * kHidden, T, {&m.eh_w});
    m.gemv(m.eh_w, m.conv, T, m.x);
    m.attention(m.mtpL, T, pos0, m.cfg.kv_iso3);
    m.ffn(m.mtpL, T);
    if (!h_out && !logits) return;
    CK(op_rmsnorm_rows(m.x + static_cast<size_t>(T - 1) * kHidden, m.mtp_hn, m.xn, kHidden, 1, m.eps, m.st));
    if (h_out) CK(cudaMemcpy(h_out, m.xn, sizeof(float) * kHidden, cudaMemcpyDeviceToHost));
    if (logits) {
        if (!m.cfg.head) throw std::runtime_error("GpuLayers::mtp_forward: logits need the GPU LM head");
        m.quant_rot({&m.hd_w}, m.xn, kHidden, 1, 0);
        m.gemv(m.hd_w, m.xn, 1, m.hd_lg);
        const size_t nb = sizeof(float) * static_cast<size_t>(m.hd_v);
        CK(cudaMemcpy(m.h_lg, m.hd_lg, nb, cudaMemcpyDeviceToHost));
        std::memcpy(logits, m.h_lg, nb);
    }
}
size_t GpuLayers::head_bytes() const { return p_->hd_bytes; }

void GpuLayers::head(int T, int n_out, float * logits, float * h_rows) {
    Impl & m = *p_;
    if (!m.cfg.head) throw std::runtime_error("GpuLayers::head: no head on the GPU");
    if (T < 1 || T > m.cfg.t_max || n_out < 0 || n_out > 8 || n_out > T) throw std::runtime_error("GpuLayers::head: bad rows");
    CK(op_rmsnorm_rows(m.x, m.hd_norm, m.xn, kHidden, T, m.eps, m.st));
    if (h_rows) CK(cudaMemcpy(h_rows, m.xn, sizeof(float) * kHidden * static_cast<size_t>(T), cudaMemcpyDeviceToHost));
    if (n_out == 0) return;
    const float * src = m.xn + static_cast<size_t>(T - n_out) * kHidden;
    m.quant_rot({&m.hd_w}, src, kHidden, n_out, 0);
    m.gemv(m.hd_w, src, n_out, m.hd_lg);
    const size_t nb = sizeof(float) * static_cast<size_t>(n_out) * static_cast<size_t>(m.hd_v);
    CK(cudaMemcpy(m.h_lg, m.hd_lg, nb, cudaMemcpyDeviceToHost));   // pinned: no driver staging
    std::memcpy(logits, m.h_lg, nb);
}

void GpuLayers::set_snapshot(bool on) {
    if (on && p_->cfg.roll_t <= 0) throw std::runtime_error("snapshots need cfg.roll_t > 0");
    p_->saving = on;
}

void GpuLayers::rollback(int keep) {
    Impl & m = *p_;
    if (m.cfg.roll_t <= 0 || keep < 1 || keep > m.cfg.roll_t) throw std::runtime_error("GpuLayers::rollback: bad keep or no snapshot buffers");
    for (const LayerD & l : m.L) {
        if (l.is_attn) continue;
        CK(cudaMemcpyAsync(l.conv_state, l.snap_conv, sizeof(float) * kConvCh * (kConvK - 1), cudaMemcpyDeviceToDevice, m.st));
        CK(cudaMemcpyAsync(l.ssm_state, l.snap_ssm, sizeof(float) * kVH * kDH * kDH, cudaMemcpyDeviceToDevice, m.st));
        CK(op_dn_conv(l.sv_qkv, l.conv_w, l.conv_state, m.conv, keep, m.st));
        CK(op_dn_rec(m.conv, l.sv_gate, l.sv_beta, l.sv_z, l.ssm_norm, l.ssm_state, m.attn_o, keep, m.eps, m.st));
    }
    CK(cudaDeviceSynchronize());
}

bool GpuLayers::stream_ready() const { return p_->stage != nullptr; }

void GpuLayers::stream_begin(const float * x_host, int T) {
    Impl & m = *p_;
    if (!m.stage) throw std::runtime_error("prefill streaming is not enabled");
    g_up_s = g_comp_s = g_fetch_s = g_pread_s = g_wait_s = 0;
    if (T < 1 || T > m.cfg.t_max) throw std::runtime_error("stream_begin: T out of range");
    CK(cudaMemcpy(m.x, x_host, static_cast<size_t>(T) * kHidden * sizeof(float), cudaMemcpyHostToDevice));
    if (m.registered) {
        m.build_seq();
        m.cur = m.issued = 0;
        for (int i = 0; i < 2; ++i) CK(cudaEventRecord(m.comp_done[i], m.st));
        m.issue_uploads();
    }
}

void GpuLayers::stream_layer(int il, int T, int pos0, float * conv_state_host, float * ssm_state_host, float * k_host, float * v_host) {
    Impl & m = *p_;
    if (!m.stage || il < m.cfg.n_layers || il >= 64) throw std::runtime_error("stream_layer: layer not streamable");
    const double t0 = now_s();
    LayerD l;
    {
        Planner pl;
        pl.base = m.svec;
        m.plan_stream_layer(il, pl, l, true);
    }
    if (pos0 < 0 || pos0 + T > m.cfg.ctx_max) throw std::runtime_error("stream_layer: position out of range");
    if (l.is_attn) {
        l.kcache = reinterpret_cast<uint8_t *>(m.s_kc);
        l.vcache = reinterpret_cast<uint8_t *>(m.s_vc);
        if (pos0 > 0) {  // earlier positions of this layer come from the host cache
            CK(cudaMemcpy(m.s_kc, k_host, static_cast<size_t>(pos0) * kKVW * sizeof(float), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(m.s_vc, v_host, static_cast<size_t>(pos0) * kKVW * sizeof(float), cudaMemcpyHostToDevice));
        }
        m.attention(l, T, pos0, false);
    } else {
        l.conv_state = m.s_conv;
        l.ssm_state = m.s_ssm;
        if (pos0 == 0) {
            CK(cudaMemset(m.s_conv, 0, sizeof(float) * kConvCh * (kConvK - 1)));
            CK(cudaMemset(m.s_ssm, 0, sizeof(float) * kVH * kDH * kDH));
        } else {
            CK(cudaMemcpy(m.s_conv, conv_state_host, sizeof(float) * kConvCh * (kConvK - 1), cudaMemcpyHostToDevice));
            for (int h = 0; h < kVH; ++h) {  // host M[j][i] -> device Mt[i][j]
                const float * src = ssm_state_host + static_cast<size_t>(h) * kDH * kDH;
                float * dst = m.h_stage + static_cast<size_t>(h) * kDH * kDH;
                for (int j = 0; j < kDH; ++j)
                    for (int i = 0; i < kDH; ++i) dst[i * kDH + j] = src[j * kDH + i];
            }
            CK(cudaMemcpy(m.s_ssm, m.h_stage, sizeof(float) * kVH * kDH * kDH, cudaMemcpyHostToDevice));
        }
        m.deltanet(l, T);
    }
    m.ffn(l, T);
    CK(cudaStreamSynchronize(m.st));   // the copy stream keeps uploading the next layer's matrices meanwhile
    g_comp_s += now_s() - t0;
    if (l.is_attn) {
        CK(cudaMemcpy(k_host + static_cast<size_t>(pos0) * kKVW, m.s_kc + static_cast<size_t>(pos0) * kKVW,
                      static_cast<size_t>(T) * kKVW * sizeof(float), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(v_host + static_cast<size_t>(pos0) * kKVW, m.s_vc + static_cast<size_t>(pos0) * kKVW,
                      static_cast<size_t>(T) * kKVW * sizeof(float), cudaMemcpyDeviceToHost));
    } else {
        CK(cudaMemcpy(conv_state_host, m.s_conv, sizeof(float) * kConvCh * (kConvK - 1), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(m.h_stage, m.s_ssm, sizeof(float) * kVH * kDH * kDH, cudaMemcpyDeviceToHost));
        for (int h = 0; h < kVH; ++h) {  // device Mt[i][j] -> host M[j][i]
            const float * src = m.h_stage + static_cast<size_t>(h) * kDH * kDH;
            float * dst = ssm_state_host + static_cast<size_t>(h) * kDH * kDH;
            for (int i = 0; i < kDH; ++i)
                for (int j = 0; j < kDH; ++j) dst[j * kDH + i] = src[i * kDH + j];
        }
    }
}

void GpuLayers::mtp_kv(const float * concat_host, int T, int pos0, float * k_host, float * v_host) {
    Impl & m = *p_;
    if (!m.stage) throw std::runtime_error("mtp_kv needs prefill streaming buffers");
    if (T < 1 || T > m.cfg.t_max || pos0 < 0 || pos0 + T > m.cfg.ctx_max) throw std::runtime_error("mtp_kv: T or position out of range");
    LayerD l;
    WMat eh;
    {
        Planner pl;
        pl.base = m.svec;   // small vectors
        l.attn_norm = m.vec(pl, "blk.64.attn_norm.weight", kHidden, true);
        l.q_norm = m.vec(pl, "blk.64.attn_q_norm.weight", kHD, true);
        l.k_norm = m.vec(pl, "blk.64.attn_k_norm.weight", kHD, true);
    }
    auto load = [&](const char * name, uint8_t * dst) {
        const GgufTensor & t = m.ix.get(name);
        if (static_cast<size_t>(t.nbytes) > m.stage_bytes) throw std::runtime_error(std::string("staging buffer too small for ") + name);
        m.upload_tensor(dst, t);   // legacy stream: ordered after the earlier work on the blocking stream st
        return WMat{dst, static_cast<int>(t.type), t.ne[0], t.ne[1], nullptr};
    };
    CK(cudaMemcpy(m.conv, concat_host, static_cast<size_t>(T) * 2 * kHidden * sizeof(float), cudaMemcpyHostToDevice));
    eh = load("blk.64.nextn.eh_proj.weight", m.stage);
    m.quant(m.conv, 2 * kHidden, T, {&eh});
    m.gemv(eh, m.conv, T, m.x);
    CK(op_rmsnorm_rows(m.x, l.attn_norm, m.xn, kHidden, T, m.eps, m.st));
    const WMat wk = load("blk.64.attn_k.weight", m.stage2);
    const WMat wv = load("blk.64.attn_v.weight", m.stage);   // after eh_proj finished (legacy-stream copy waits for st)
    m.quant(m.xn, kHidden, T, {&wk, &wv});
    m.gemv(wk, m.xn, T, m.kproj);
    m.gemv(wv, m.xn, T, m.vproj);
    // q is not needed; qk_prep also writes qbuf from bufA (unused here).
    CK(op_qk_prep(m.bufA, m.kproj, l.q_norm, l.k_norm, m.qbuf, m.kout, T, pos0, m.rope_cos, m.rope_sin, m.eps, m.st));
    CK(cudaStreamSynchronize(m.st));
    CK(cudaMemcpy(k_host + static_cast<size_t>(pos0) * kKVW, m.kout, static_cast<size_t>(T) * kKVW * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(v_host + static_cast<size_t>(pos0) * kKVW, m.vproj, static_cast<size_t>(T) * kKVW * sizeof(float), cudaMemcpyDeviceToHost));
}

void GpuLayers::stream_peek(float * x_host, int T) {
    CK(cudaMemcpy(x_host, p_->x, static_cast<size_t>(T) * kHidden * sizeof(float), cudaMemcpyDeviceToHost));
}

void GpuLayers::stream_end(float * x_host, int T) {
    stream_peek(x_host, T);
    if (p_->registered) {
        if (p_->cur != p_->seq.size()) throw std::runtime_error("streamed prefill ended before every matrix was used");
        CK(cudaStreamSynchronize(p_->cp));
    }
    if (std::getenv("QWEN_STREAM_DEBUG")) std::fprintf(stderr, "stream: layers %.2f s (pread %.2f, dma wait %.2f)\n", g_comp_s, g_pread_s, g_wait_s);
    g_up_s = g_comp_s = g_fetch_s = g_pread_s = g_wait_s = 0;
}

namespace {
// Device bytes allocated after the arena (Q8 scratch, Q8_1 activations) plus what the driver adds lazily (module loads,
// CUDA graph instantiation), on top of the memory in use when the plan is checked (CUDA context, about 106 MiB).
// Calibrated on PTQ1_0, 64 layers + GPU head, ctx 131072, t_max 64: arena 7226.8 MiB, projection without margin 7333 MiB,
// nvidia-smi peak over the whole run 7349 MiB (+16). 32 MiB keeps 16 MiB of headroom over that.
constexpr double kScratchMarginMib = 32.0;
}  // namespace

size_t GpuLayers::plan_bytes(const GgufIndex & ix, const GpuConfig & cfg) {
    Impl m(ix, cfg, 0.0f);
    Planner count;
    m.plan(count, false);
    return count.off;
}

double GpuLayers::projected_mib(const GgufIndex & ix, const GpuConfig & cfg) {
    CK(cudaFree(nullptr));   // creates the context, so its memory is part of "in use"
    size_t free_b = 0, total_b = 0;
    CK(cudaMemGetInfo(&free_b, &total_b));
    const double used = static_cast<double>(total_b - free_b) / 1048576.0;
    return used + static_cast<double>(plan_bytes(ix, cfg)) / 1048576.0 + kScratchMarginMib;
}

void GpuLayers::clear_kv() {
    const size_t unit = p_->cfg.kv_iso3 ? 400 : static_cast<size_t>(kKVW) * 4;
    for (LayerD & l : p_->L) {
        if (!l.is_attn) continue;
        CK(cudaMemset(l.kcache, 0, unit * static_cast<size_t>(p_->cfg.ctx_max)));
        CK(cudaMemset(l.vcache, 0, unit * static_cast<size_t>(p_->cfg.ctx_max)));
    }
    if (p_->cfg.mtp) {
        CK(cudaMemset(p_->mtpL.kcache, 0, unit * static_cast<size_t>(p_->cfg.ctx_max)));
        CK(cudaMemset(p_->mtpL.vcache, 0, unit * static_cast<size_t>(p_->cfg.ctx_max)));
    }
    CK(cudaDeviceSynchronize());
}

size_t GpuLayers::state_bytes(int n_pos) const {
    const size_t unit = p_->cfg.kv_iso3 ? 400 : static_cast<size_t>(kKVW) * 4;
    size_t b = 0;
    for (const LayerD & l : p_->L) {
        if (l.is_attn) b += 2 * static_cast<size_t>(n_pos) * unit;
        else b += sizeof(float) * (static_cast<size_t>(kConvCh) * (kConvK - 1) + static_cast<size_t>(kVH) * kDH * kDH);
    }
    if (p_->cfg.mtp) b += 2 * static_cast<size_t>(n_pos) * unit;
    return b;
}

namespace {
// The state file I/O bounces through the pinned h_stage buffer (allocated at load, >= 3 MiB): no pageable cudaMemcpy of
// large blocks, which made the driver's own staging show up as transient device memory.
size_t stage_capacity(int t_max) {
    return std::max(static_cast<size_t>(t_max) * 5120, static_cast<size_t>(48) * 128 * 128) * sizeof(float);
}
}  // namespace

void GpuLayers::save_state(int n_pos, const std::function<void(const void *, size_t)> & io) const {
    if (n_pos < 0 || n_pos > p_->cfg.ctx_max) throw std::runtime_error("save_state: n_pos out of range");
    CK(cudaDeviceSynchronize());
    const size_t chunk = stage_capacity(p_->cfg.t_max);
    uint8_t * buf = reinterpret_cast<uint8_t *>(p_->h_stage);
    auto put = [&](const void * dev, size_t bytes) {
        for (size_t o = 0; o < bytes; o += chunk) {
            const size_t n = std::min(chunk, bytes - o);
            CK(cudaMemcpy(buf, static_cast<const uint8_t *>(dev) + o, n, cudaMemcpyDeviceToHost));
            io(buf, n);
        }
    };
    const size_t unit = p_->cfg.kv_iso3 ? 400 : static_cast<size_t>(kKVW) * 4;
    for (const LayerD & l : p_->L) {
        if (l.is_attn) {
            put(l.kcache, static_cast<size_t>(n_pos) * unit);
            put(l.vcache, static_cast<size_t>(n_pos) * unit);
        } else {
            put(l.conv_state, sizeof(float) * kConvCh * (kConvK - 1));
            put(l.ssm_state, sizeof(float) * kVH * kDH * kDH);
        }
    }
    if (p_->cfg.mtp) {
        put(p_->mtpL.kcache, static_cast<size_t>(n_pos) * unit);
        put(p_->mtpL.vcache, static_cast<size_t>(n_pos) * unit);
    }
}

void GpuLayers::load_state(int n_pos, const std::function<void(void *, size_t)> & io) {
    if (n_pos < 0 || n_pos > p_->cfg.ctx_max) throw std::runtime_error("load_state: n_pos out of range");
    CK(cudaDeviceSynchronize());
    const size_t chunk = stage_capacity(p_->cfg.t_max);
    uint8_t * buf = reinterpret_cast<uint8_t *>(p_->h_stage);
    auto get = [&](void * dev, size_t bytes) {
        for (size_t o = 0; o < bytes; o += chunk) {
            const size_t n = std::min(chunk, bytes - o);
            io(buf, n);
            CK(cudaMemcpy(static_cast<uint8_t *>(dev) + o, buf, n, cudaMemcpyHostToDevice));
        }
    };
    const size_t unit = p_->cfg.kv_iso3 ? 400 : static_cast<size_t>(kKVW) * 4;
    for (LayerD & l : p_->L) {
        if (l.is_attn) {
            get(l.kcache, static_cast<size_t>(n_pos) * unit);
            get(l.vcache, static_cast<size_t>(n_pos) * unit);
        } else {
            get(l.conv_state, sizeof(float) * kConvCh * (kConvK - 1));
            get(l.ssm_state, sizeof(float) * kVH * kDH * kDH);
        }
    }
    if (p_->cfg.mtp) {
        get(p_->mtpL.kcache, static_cast<size_t>(n_pos) * unit);
        get(p_->mtpL.vcache, static_cast<size_t>(n_pos) * unit);
    }
    CK(cudaDeviceSynchronize());
}

size_t GpuLayers::vram_bytes() const {
    return p_->arena_bytes + static_cast<size_t>(p_->q8.T_max) * p_->q8.in_max +
           static_cast<size_t>(p_->q8.T_max) * (p_->q8.in_max / 32) * sizeof(float) +
           static_cast<size_t>(p_->act.T_max) * p_->act.in_max + 2 * static_cast<size_t>(p_->act.T_max) * (p_->act.in_max / 32) * sizeof(float);
}
size_t GpuLayers::weight_bytes() const { return p_->weights; }
size_t GpuLayers::kv_bytes() const { return p_->kv; }
int GpuLayers::n_layers() const { return p_->cfg.n_layers; }

uint32_t parse_dp4a_mask(const char * list) {
    const std::string s = list ? list : "";
    if (s.empty() || s == "none") return 0;
    struct Name { const char * n; int id; };
    static const Name names[] = {{"q3_k", 11}, {"q4_k", 12}, {"q5_k", 13}, {"q6_k", 14}, {"iq4_nl", 20}, {"iq3_s", 21}, {"iq4_xs", 23}};
    uint32_t m = 0;
    if (s == "all") {
        for (const Name & n : names) m |= 1u << n.id;
        return m;
    }
    size_t b = 0;
    while (b <= s.size()) {
        size_t e = s.find(',', b);
        if (e == std::string::npos) e = s.size();
        const std::string tok = s.substr(b, e - b);
        bool found = false;
        for (const Name & n : names) {
            if (tok == n.n || tok == std::to_string(n.id)) { m |= 1u << n.id; found = true; }
        }
        if (!found && !tok.empty()) throw std::runtime_error("unknown dp4a type: " + tok);
        b = e + 1;
    }
    return m;
}
