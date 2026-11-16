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
    float * h_stage = nullptr;  // pinned staging buffer
    std::vector<LayerD> L;
    // scratch
    float *x = nullptr, *xn = nullptr, *proj = nullptr, *bufA = nullptr, *bufB = nullptr, *conv = nullptr, *attn_o = nullptr;
    float *alpha = nullptr, *betar = nullptr, *gate = nullptr, *beta = nullptr, *fg = nullptr, *fu = nullptr;
    float *qbuf = nullptr, *kproj = nullptr, *kout = nullptr, *vproj = nullptr;
    float theta_scale = 0.0f;
    // Prefill streaming: one staging buffer for the weights of the layer being run, plus one layer of state and FP32 KV.
    uint8_t * stage = nullptr;   // one matrix at a time
    uint8_t * svec = nullptr;    // the small vectors of one layer
    size_t stage_bytes = 0;
    float *s_conv = nullptr, *s_ssm = nullptr, *s_kc = nullptr, *s_vc = nullptr;
    bool saving = false;

    Impl(const GgufIndex & i, const GpuConfig & c, float e) : ix(i), cfg(c), eps(e) {}

    // Pinned double buffer: pread() from the page cache into pinned memory (no per-page mmap faults), then an async DMA copy.
    static constexpr size_t kChunk = 32u << 20;
    uint8_t * pin[2] = {nullptr, nullptr};
    cudaEvent_t pin_done[2] = {nullptr, nullptr};
    void pin_alloc() {
        for (int i = 0; i < 2; ++i) {
            CK(cudaHostAlloc(reinterpret_cast<void **>(&pin[i]), kChunk, cudaHostAllocDefault));
            CK(cudaEventCreateWithFlags(&pin_done[i], cudaEventDisableTiming));
            CK(cudaEventRecord(pin_done[i], 0));
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
            CK(cudaMemcpyAsync(static_cast<uint8_t *>(dst) + done, pin[b], n, cudaMemcpyHostToDevice, 0));
            CK(cudaEventRecord(pin_done[b], 0));
            done += n;
            b ^= 1;
        }
    }

    WMat mat(Planner & pl, const std::string & name, bool upload) {
        const GgufTensor & t = ix.get(name);
        if (t.ne[2] != 1 || t.ne[3] != 1) throw std::runtime_error("not a matrix: " + name);
        const int type = static_cast<int>(t.type);
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
    }

    bool use_dp4a(const WMat & W) const { return W.type != GT_Q8_0 && gemv_dp4a_supported(W.type) && ((cfg.dp4a_mask >> W.type) & 1u); }

    // Quantizes `x` once for all the listed matrices that take the dp4a path (nothing happens when none does).
    void quant(const float * x, int64_t in, int T, std::initializer_list<const WMat *> mats) {
        for (const WMat * w : mats) {
            if (use_dp4a(*w)) { CK(actq8_quantize(x, in, T, act, 0)); return; }
        }
    }

    // The caller must have run quant() for the same `in` buffer when use_dp4a(W) is true.
    void gemv(const WMat & W0, const float * in, int T, float * out) {
        WMat W = W0;
        if (W.src) {  // streamed layer: bring this matrix into the staging buffer (stream order keeps the previous GEMV safe)
            upload_tensor(stage, *W.src);
            W.p = stage;
        }
        if (W.type == GT_Q8_0) CK(q8_0_gemv_dp4a(W.p, W.in, W.out, in, T, out, q8, 0));
        else if (use_dp4a(W)) CK(gemv_dp4a(W.type, W.p, W.in, W.out, act, T, out, 0));
        else CK(gemv_fp32(W.type, W.p, W.in, W.out, in, T, out, 0));
    }

    void ffn(const LayerD & l, int T) {
        CK(op_rmsnorm_rows(x, l.post_norm, xn, kHidden, T, eps, 0));
        quant(xn, kHidden, T, {&l.ffn_gate, &l.ffn_up});
        gemv(l.ffn_gate, xn, T, fg);
        gemv(l.ffn_up, xn, T, fu);
        CK(op_silu_mul(fg, fu, static_cast<int64_t>(T) * kFFN, 0));
        quant(fg, kFFN, T, {&l.ffn_down});
        gemv(l.ffn_down, fg, T, proj);
        CK(op_add_inplace(x, proj, static_cast<int64_t>(T) * kHidden, 0));
    }

    void deltanet(const LayerD & l, int T) {
        if (saving) {
            CK(cudaMemcpyAsync(l.snap_conv, l.conv_state, sizeof(float) * kConvCh * (kConvK - 1), cudaMemcpyDeviceToDevice, 0));
            CK(cudaMemcpyAsync(l.snap_ssm, l.ssm_state, sizeof(float) * kVH * kDH * kDH, cudaMemcpyDeviceToDevice, 0));
        }
        CK(op_rmsnorm_rows(x, l.attn_norm, xn, kHidden, T, eps, 0));
        quant(xn, kHidden, T, {&l.qkv, &l.gate_z});
        gemv(l.qkv, xn, T, bufA);
        gemv(l.gate_z, xn, T, bufB);
        gemv(l.alpha, xn, T, alpha);
        gemv(l.beta, xn, T, betar);
        CK(op_dn_prep(alpha, betar, l.ssm_dt, l.ssm_a, gate, beta, T, 0));
        if (saving) {
            CK(cudaMemcpyAsync(l.sv_qkv, bufA, sizeof(float) * T * kConvCh, cudaMemcpyDeviceToDevice, 0));
            CK(cudaMemcpyAsync(l.sv_gate, gate, sizeof(float) * T * kVH, cudaMemcpyDeviceToDevice, 0));
            CK(cudaMemcpyAsync(l.sv_beta, beta, sizeof(float) * T * kVH, cudaMemcpyDeviceToDevice, 0));
            CK(cudaMemcpyAsync(l.sv_z, bufB, sizeof(float) * T * kInner, cudaMemcpyDeviceToDevice, 0));
        }
        CK(op_dn_conv(bufA, l.conv_w, l.conv_state, conv, T, 0));
        CK(op_dn_rec(conv, gate, beta, bufB, l.ssm_norm, l.ssm_state, attn_o, T, eps, 0));
        quant(attn_o, kInner, T, {&l.ssm_out});
        gemv(l.ssm_out, attn_o, T, proj);
        CK(op_add_inplace(x, proj, static_cast<int64_t>(T) * kHidden, 0));
    }

    void attention(const LayerD & l, int T, int pos0, bool iso3) {
        CK(op_rmsnorm_rows(x, l.attn_norm, xn, kHidden, T, eps, 0));
        quant(xn, kHidden, T, {&l.wq, &l.wk, &l.wv});
        gemv(l.wq, xn, T, bufA);
        gemv(l.wk, xn, T, kproj);
        gemv(l.wv, xn, T, vproj);
        CK(op_qk_prep(bufA, kproj, l.q_norm, l.k_norm, qbuf, kout, T, pos0, theta_scale, eps, 0));
        CK(op_kv_store(kout, l.kcache, T, pos0, iso3, 0));
        CK(op_kv_store(vproj, l.vcache, T, pos0, iso3, 0));
        CK(op_attention(qbuf, bufA, l.kcache, l.vcache, attn_o, T, pos0, iso3, cfg.ctx_max, 0));
        quant(attn_o, kInner, T, {&l.wo});
        gemv(l.wo, attn_o, T, proj);
        CK(op_add_inplace(x, proj, static_cast<int64_t>(T) * kHidden, 0));
    }
};

GpuLayers::GpuLayers(const GgufIndex & ix, const GpuConfig & cfg, float rms_eps) : p_(new Impl(ix, cfg, rms_eps)) {
    Impl & m = *p_;
    if (cfg.n_layers < 1 || cfg.n_layers > 64) throw std::runtime_error("n_layers out of range");
    if (cfg.ctx_max > 4096) throw std::runtime_error("ctx_max above 4096 is not supported");
    m.theta_scale = std::pow(10000000.0f, -2.0f / static_cast<float>(kRope));

    Planner count;
    m.plan(count, false);
    m.arena_bytes = count.off;
    CK(cudaMalloc(&m.arena, m.arena_bytes));
    Planner real;
    real.base = m.arena;
    m.pin_alloc();
    m.plan(real, true);
    CK(cudaDeviceSynchronize());
    if (!m.stage) m.pin_free();   // kept for the streamed prefill uploads otherwise
    CK(q8_scratch_alloc(m.q8, cfg.t_max, kHidden));
    CK(actq8_alloc(m.act, cfg.t_max, kFFN));
    CK(cudaHostAlloc(reinterpret_cast<void **>(&m.h_stage), std::max(static_cast<size_t>(cfg.t_max) * kHidden, static_cast<size_t>(kVH) * kDH * kDH) * sizeof(float), cudaHostAllocDefault));
    reset();
    CK(cudaDeviceSynchronize());
}

GpuLayers::~GpuLayers() {
    if (!p_) return;
    p_->pin_free();
    q8_scratch_free(p_->q8);
    actq8_free(p_->act);
    if (p_->h_stage) cudaFreeHost(p_->h_stage);
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

void GpuLayers::set_snapshot(bool on) {
    if (on && p_->cfg.roll_t <= 0) throw std::runtime_error("snapshots need cfg.roll_t > 0");
    p_->saving = on;
}

void GpuLayers::rollback(int keep) {
    Impl & m = *p_;
    if (m.cfg.roll_t <= 0 || keep < 1 || keep > m.cfg.roll_t) throw std::runtime_error("GpuLayers::rollback: bad keep or no snapshot buffers");
    for (const LayerD & l : m.L) {
        if (l.is_attn) continue;
        CK(cudaMemcpyAsync(l.conv_state, l.snap_conv, sizeof(float) * kConvCh * (kConvK - 1), cudaMemcpyDeviceToDevice, 0));
        CK(cudaMemcpyAsync(l.ssm_state, l.snap_ssm, sizeof(float) * kVH * kDH * kDH, cudaMemcpyDeviceToDevice, 0));
        CK(op_dn_conv(l.sv_qkv, l.conv_w, l.conv_state, m.conv, keep, 0));
        CK(op_dn_rec(m.conv, l.sv_gate, l.sv_beta, l.sv_z, l.ssm_norm, l.ssm_state, m.attn_o, keep, m.eps, 0));
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
    CK(cudaDeviceSynchronize());
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

void GpuLayers::stream_peek(float * x_host, int T) {
    CK(cudaMemcpy(x_host, p_->x, static_cast<size_t>(T) * kHidden * sizeof(float), cudaMemcpyDeviceToHost));
}

void GpuLayers::stream_end(float * x_host, int T) {
    stream_peek(x_host, T);
    if (std::getenv("QWEN_STREAM_DEBUG")) std::fprintf(stderr, "stream: layers %.2f s (pread %.2f, dma wait %.2f)\n", g_comp_s, g_pread_s, g_wait_s);
    g_up_s = g_comp_s = g_fetch_s = g_pread_s = g_wait_s = 0;
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
