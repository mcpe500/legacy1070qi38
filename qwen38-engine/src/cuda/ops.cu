#include <algorithm>
#include <cstdlib>

#include "f16bits.hpp"
#include "ops.cuh"

#define QT_BEGIN(type, name, size) static __device__ const type name[size] = {
#define QT_END() };
#include "tables.inc"
#undef QT_BEGIN
#undef QT_END

namespace {

constexpr int NH = 24, NKV = 4, HD = 256, ROPE_DIM = 64;
constexpr int DH = 128, VH = 48, QKH = 16, CONV_CH = 10240, CONV_K = 4;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}
// Block-wide sum for blockDim.x a multiple of 32 and at most 256. red must hold 8 floats. Safe to call repeatedly.
__device__ __forceinline__ float block_sum(float v, float * red) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    float r = 0.0f;
    for (int i = 0; i < nw; ++i) r += red[i];
    return r;
}
__device__ __forceinline__ float block_max(float v, float * red) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    float r = red[0];
    for (int i = 1; i < nw; ++i) r = fmaxf(r, red[i]);
    return r;
}
__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float sigm_f(float x) { return 1.0f / (1.0f + expf(-x)); }

// ---------------------------------------------------------------------------------------------- basic operators

__global__ void rmsnorm_rows_kernel(const float * x, const float * w, float * y, int n, float eps) {
    __shared__ float red[8];
    const float * xr = x + static_cast<int64_t>(blockIdx.x) * n;
    float * yr = y + static_cast<int64_t>(blockIdx.x) * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, red);
    const float scale = 1.0f / sqrtf(ss / static_cast<float>(n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * scale * (w ? w[i] : 1.0f);
}

__global__ void add_kernel(float * x, const float * y, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

__global__ void silu_mul_kernel(float * g, const float * u, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) g[i] = silu_f(g[i]) * u[i];
}

// ------------------------------------------------------------------------------------------------ Gated DeltaNet

__global__ void dn_prep_kernel(const float * alpha, const float * beta_raw, const float * dt, const float * a, float * gate,
                               float * beta, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int h = i % VH;
    beta[i] = sigm_f(beta_raw[i]);
    const float x = alpha[i] + dt[h];
    const float sp = x > 20.0f ? x : log1pf(expf(x));
    gate[i] = sp * a[h];
}

// One thread per channel; the 3-value history lives in conv_state[c][0..2] (oldest first).
__global__ void dn_conv_kernel(const float * qkv, const float * w, float * state, float * out, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= CONV_CH) return;
    float h0 = state[c * 3], h1 = state[c * 3 + 1], h2 = state[c * 3 + 2];
    const float w0 = w[c * CONV_K], w1 = w[c * CONV_K + 1], w2 = w[c * CONV_K + 2], w3 = w[c * CONV_K + 3];
    for (int t = 0; t < T; ++t) {
        const float xin = qkv[static_cast<int64_t>(t) * CONV_CH + c];
        float s = 0.0f;
        s += h0 * w0;
        s += h1 * w1;
        s += h2 * w2;
        s += xin * w3;
        out[static_cast<int64_t>(t) * CONV_CH + c] = silu_f(s);
        h0 = h1; h1 = h2; h2 = xin;
    }
    state[c * 3] = h0; state[c * 3 + 1] = h1; state[c * 3 + 2] = h2;
}

// Gated delta rule for one value head per block (thread j owns column j of Mt, i.e. value index j).
//   M *= exp(g);  delta_j = (v_j - sum_i M[j][i] k_i) * beta;  M[j][i] += k_i * delta_j;  o_j = scale * sum_i M[j][i] q_i
// followed by the gated RMSNorm: o -> o / rms(o) * ssm_norm * silu(z).
// Same recurrence as dn_rec_kernel_ref below, with the 128-value state column of the thread kept in registers for the whole
// call: the state is read once and written once (the reference reads and writes it twice per token, 4 x 64 KiB per head).
// Every multiply and add happens in the same order as in the reference, so the result is bit-identical.
__global__ void __launch_bounds__(DH) dn_rec_kernel(const float * conv_out, const float * gate, const float * beta, const float * z,
                                                    const float * ssm_norm, float * state_t, float * attn_o, int T, float eps) {
    __shared__ float qn[DH], kn[DH], red[8];
    const int h = blockIdx.x, j = threadIdx.x, hk = h % QKH;
    float * Mt = state_t + static_cast<int64_t>(h) * DH * DH;
    const float scale = 1.0f / sqrtf(static_cast<float>(DH));
    float col[DH];
#pragma unroll
    for (int i = 0; i < DH; ++i) col[i] = Mt[i * DH + j];
    for (int t = 0; t < T; ++t) {
        const float * row = conv_out + static_cast<int64_t>(t) * CONV_CH;
        const float qv = row[hk * DH + j], kv = row[QKH * DH + hk * DH + j], vj = row[2 * QKH * DH + h * DH + j];
        const float sq = block_sum(qv * qv, red);
        const float sk = block_sum(kv * kv, red);
        const float iq = 1.0f / fmaxf(sqrtf(sq), eps);
        const float ik = 1.0f / fmaxf(sqrtf(sk), eps);
        qn[j] = qv * iq;
        kn[j] = kv * ik;
        __syncthreads();

        const float ex = expf(gate[t * VH + h]);
        const float b = beta[t * VH + h];
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < DH; ++i) {
            const float m = col[i] * ex;
            col[i] = m;
            dot += m * kn[i];
        }
        const float delta = (vj - dot) * b;
        float o = 0.0f;
#pragma unroll
        for (int i = 0; i < DH; ++i) {
            const float m = col[i] + kn[i] * delta;
            col[i] = m;
            o += m * qn[i];
        }
        o *= scale;
        const float ms = block_sum(o * o, red);
        const float rs = 1.0f / sqrtf(ms / static_cast<float>(DH) + eps);
        const float zz = z[static_cast<int64_t>(t) * (VH * DH) + h * DH + j];
        attn_o[static_cast<int64_t>(t) * (VH * DH) + h * DH + j] = o * rs * ssm_norm[j] * silu_f(zz);
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < DH; ++i) Mt[i * DH + j] = col[i];
}

// Reference (state read and written in global memory twice per token), kept for tests/cuda/dn_rec_ab_test.cu.
__global__ void dn_rec_kernel_ref(const float * conv_out, const float * gate, const float * beta, const float * z,
                              const float * ssm_norm, float * state_t, float * attn_o, int T, float eps) {
    __shared__ float qn[DH], kn[DH], red[8];
    const int h = blockIdx.x, j = threadIdx.x, hk = h % QKH;
    float * Mt = state_t + static_cast<int64_t>(h) * DH * DH;
    const float scale = 1.0f / sqrtf(static_cast<float>(DH));
    for (int t = 0; t < T; ++t) {
        const float * row = conv_out + static_cast<int64_t>(t) * CONV_CH;
        const float qv = row[hk * DH + j], kv = row[QKH * DH + hk * DH + j], vj = row[2 * QKH * DH + h * DH + j];
        const float sq = block_sum(qv * qv, red);
        const float sk = block_sum(kv * kv, red);
        const float iq = 1.0f / fmaxf(sqrtf(sq), eps);
        const float ik = 1.0f / fmaxf(sqrtf(sk), eps);
        qn[j] = qv * iq;
        kn[j] = kv * ik;
        __syncthreads();

        const float ex = expf(gate[t * VH + h]);
        const float b = beta[t * VH + h];
        float dot = 0.0f;
        for (int i = 0; i < DH; ++i) {
            const float m = Mt[i * DH + j] * ex;
            Mt[i * DH + j] = m;
            dot += m * kn[i];
        }
        const float delta = (vj - dot) * b;
        float o = 0.0f;
        for (int i = 0; i < DH; ++i) {
            const float m = Mt[i * DH + j] + kn[i] * delta;
            Mt[i * DH + j] = m;
            o += m * qn[i];
        }
        o *= scale;
        const float ms = block_sum(o * o, red);
        const float rs = 1.0f / sqrtf(ms / static_cast<float>(DH) + eps);
        const float zz = z[static_cast<int64_t>(t) * (VH * DH) + h * DH + j];
        attn_o[static_cast<int64_t>(t) * (VH * DH) + h * DH + j] = o * rs * ssm_norm[j] * silu_f(zz);
        __syncthreads();
    }
}

// ----------------------------------------------------------------------------------------------- Gated Attention

// One thread per (position, pair): the theta chain and cosf/sinf are identical to the inline computation this table
// replaces, so every table value is bit-identical to what qk_prep used to compute per token and per head.
__global__ void rope_table_kernel(float * cos_tab, float * sin_tab, int n_pos, float theta_scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pos * (ROPE_DIM / 2)) return;
    const int pos = i / (ROPE_DIM / 2), p = i % (ROPE_DIM / 2);
    float theta = static_cast<float>(pos);
    for (int j = 0; j < p; ++j) theta *= theta_scale;
    cos_tab[i] = cosf(theta);
    sin_tab[i] = sinf(theta);
}

// grid (T, 28), block 256 (one thread per dimension). Blocks 0..23 prepare a query head, blocks 24..27 a key head.
__global__ void qk_prep_kernel(const float * qfull, const float * kproj, const float * qnorm, const float * knorm, float * qbuf,
                               float * kout, int pos0, const int * pos_dev, const float * rope_cos, const float * rope_sin, float eps) {
    if (pos_dev) pos0 = *pos_dev;   // CUDA graph replay: the position lives in device memory
    __shared__ float yv[HD], red[8];
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const float * src;
    const float * nw;
    float * dst;
    if (h < NH) {
        src = qfull + static_cast<int64_t>(t) * (NH * 2 * HD) + h * 2 * HD;
        nw = qnorm;
        dst = qbuf + static_cast<int64_t>(t) * (NH * HD) + h * HD;
    } else {
        const int g = h - NH;
        src = kproj + static_cast<int64_t>(t) * (NKV * HD) + g * HD;
        nw = knorm;
        dst = kout + static_cast<int64_t>(t) * (NKV * HD) + g * HD;
    }
    const float v = src[d];
    const float ss = block_sum(v * v, red);
    const float scale = 1.0f / sqrtf(ss / static_cast<float>(HD) + eps);
    yv[d] = v * scale * nw[d];
    __syncthreads();
    float outv = yv[d];
    if (d < ROPE_DIM) {
        const int p = d < ROPE_DIM / 2 ? d : d - ROPE_DIM / 2;  // rotation pair (p, p + 32)
        const int rp = (pos0 + t) * (ROPE_DIM / 2) + p;
        const float c = rope_cos[rp], s = rope_sin[rp];
        const float x0 = yv[p], x1 = yv[p + ROPE_DIM / 2];
        outv = d < ROPE_DIM / 2 ? x0 * c - x1 * s : x0 * s + x1 * c;
    }
    dst[d] = outv;
}

// Decodes one group of 4 values of an iso3 block (inverse quaternion rotation), unscaled by the block norm.
__device__ __forceinline__ void iso3_group(const uint8_t * blk, int gi, float * r) {
    const uint32_t qsb = blk[2 + gi];
    const uint32_t sg = (blk[34 + (gi >> 1)] >> (4 * (gi & 1))) & 0xF;
    float q[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) q[c] = iso3_codebook[((qsb >> (2 * c)) & 3) | (((sg >> c) & 1) << 2)];
    const float w = iso3_quat_w[gi], x = -iso3_quat_x[gi], y = -iso3_quat_y[gi], zq = -iso3_quat_z[gi];
    r[0] = w * q[0] - x * q[1] - y * q[2] - zq * q[3];
    r[1] = w * q[1] + x * q[0] + y * q[3] - zq * q[2];
    r[2] = w * q[2] - x * q[3] + y * q[0] + zq * q[1];
    r[3] = w * q[3] + x * q[2] - y * q[1] + zq * q[0];
}
__device__ __forceinline__ float iso3_norm(const uint8_t * blk) {
    return f16_to_f32(static_cast<uint32_t>(blk[0]) | (static_cast<uint32_t>(blk[1]) << 8));
}

// grid T * 24 (one block per token and query head), block 256. Scores live in dynamic shared memory (ctx_max floats).
template <bool ISO3>
__global__ void attention_kernel(const float * qbuf, const float * qfull, const uint8_t * kc, const uint8_t * vc, float * attn_o,
                                 int pos0, const int * pos_dev) {
    if (pos_dev) pos0 = *pos_dev;
    extern __shared__ float sc[];
    __shared__ float qs[HD], red[8];
    const int t = blockIdx.x / NH, h = blockIdx.x % NH, g = h / (NH / NKV);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int n = pos0 + t + 1;
    qs[tid] = qbuf[static_cast<int64_t>(t) * (NH * HD) + h * HD + tid];
    __syncthreads();
    const float kq_scale = 1.0f / sqrtf(static_cast<float>(HD));

    for (int j = warp; j < n; j += 8) {
        float acc = 0.0f;
        if (!ISO3) {
            const float4 * k4 = reinterpret_cast<const float4 *>(reinterpret_cast<const float *>(kc) +
                                                                  static_cast<int64_t>(j) * (NKV * HD) + g * HD + lane * 8);
            const float4 a = k4[0], b = k4[1];
            acc = qs[lane * 8 + 0] * a.x + qs[lane * 8 + 1] * a.y + qs[lane * 8 + 2] * a.z + qs[lane * 8 + 3] * a.w +
                  qs[lane * 8 + 4] * b.x + qs[lane * 8 + 5] * b.y + qs[lane * 8 + 6] * b.z + qs[lane * 8 + 7] * b.w;
        } else {
            const uint8_t * kb = kc + (static_cast<int64_t>(j) * 8 + g * 2) * 50;
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
                const uint8_t * b = kb + hf * 50;
                float r[4];
                iso3_group(b, lane, r);
                const float nm = iso3_norm(b);
                const int base = hf * 128 + 4 * lane;
                acc += qs[base] * (r[0] * nm) + qs[base + 1] * (r[1] * nm) + qs[base + 2] * (r[2] * nm) + qs[base + 3] * (r[3] * nm);
            }
        }
        acc = warp_sum(acc);
        if (lane == 0) sc[j] = acc * kq_scale;
    }
    __syncthreads();

    float mx = -INFINITY;
    for (int j = tid; j < n; j += 256) mx = fmaxf(mx, sc[j]);
    mx = block_max(mx, red);
    float sum = 0.0f;
    for (int j = tid; j < n; j += 256) {
        const float e = expf(sc[j] - mx);
        sc[j] = e;
        sum += e;
    }
    sum = block_sum(sum, red);
    const float inv = 1.0f / sum;
    __syncthreads();

    float o = 0.0f;
    if (!ISO3) {
        const float * vf = reinterpret_cast<const float *>(vc);
        for (int j = 0; j < n; ++j) o += sc[j] * inv * vf[static_cast<int64_t>(j) * (NKV * HD) + g * HD + tid];
    } else {
        const int grp = tid >> 2, c = tid & 3, hf = grp >> 5, gi = grp & 31;
        for (int j = 0; j < n; ++j) {
            const uint8_t * b = vc + (static_cast<int64_t>(j) * 8 + g * 2 + hf) * 50;
            float r[4];
            iso3_group(b, gi, r);
            o += sc[j] * inv * (r[c] * iso3_norm(b));
        }
    }
    const float gate = qfull[static_cast<int64_t>(t) * (NH * 2 * HD) + h * 2 * HD + HD + tid];
    attn_o[static_cast<int64_t>(t) * (NH * HD) + h * HD + tid] = o * sigm_f(gate);
}

constexpr int kFastCtxMax = 8192;   // largest ctx_max of the shared-memory score kernel

// Long-context attention (ctx_max > 8192): the scores no longer fit in shared memory (131072 floats = 512 KiB),
// so K/V is streamed in tiles of kTile positions and the softmax runs online (running max, running sum,
// accumulator rescaled per tile). Shared memory stays at a few KiB, far under the 48 KiB sm_61 limit. The
// accumulator is normalized once at the end. Same result as the fast path up to float summation order.
constexpr int kTile = 256;   // positions per tile: 8 warps score one position each, 1 KiB of scores

template <bool ISO3>
__global__ void attention_kernel_tiled(const float * qbuf, const float * qfull, const uint8_t * kc, const uint8_t * vc,
                                       float * attn_o, int pos0, const int * pos_dev) {
    if (pos_dev) pos0 = *pos_dev;
    __shared__ float qs[HD], red[8], tsc[kTile];
    const int t = blockIdx.x / NH, h = blockIdx.x % NH, g = h / (NH / NKV);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int n = pos0 + t + 1;
    qs[tid] = qbuf[static_cast<int64_t>(t) * (NH * HD) + h * HD + tid];
    __syncthreads();
    const float kq_scale = 1.0f / sqrtf(static_cast<float>(HD));

    float m = -INFINITY, l = 0.0f, o = 0.0f;
    for (int j0 = 0; j0 < n; j0 += kTile) {
        const int nt = min(kTile, n - j0);
        for (int w = warp; w < nt; w += 8) {   // one warp scores one position of the tile
            const int j = j0 + w;
            float acc = 0.0f;
            if (!ISO3) {
                const float4 * k4 = reinterpret_cast<const float4 *>(reinterpret_cast<const float *>(kc) +
                                                                      static_cast<int64_t>(j) * (NKV * HD) + g * HD + lane * 8);
                const float4 a = k4[0], b = k4[1];
                acc = qs[lane * 8 + 0] * a.x + qs[lane * 8 + 1] * a.y + qs[lane * 8 + 2] * a.z + qs[lane * 8 + 3] * a.w +
                      qs[lane * 8 + 4] * b.x + qs[lane * 8 + 5] * b.y + qs[lane * 8 + 6] * b.z + qs[lane * 8 + 7] * b.w;
            } else {
                const uint8_t * kb = kc + (static_cast<int64_t>(j) * 8 + g * 2) * 50;
#pragma unroll
                for (int hf = 0; hf < 2; ++hf) {
                    const uint8_t * b = kb + hf * 50;
                    float r[4];
                    iso3_group(b, lane, r);
                    const float nm = iso3_norm(b);
                    const int base = hf * 128 + 4 * lane;
                    acc += qs[base] * (r[0] * nm) + qs[base + 1] * (r[1] * nm) + qs[base + 2] * (r[2] * nm) + qs[base + 3] * (r[3] * nm);
                }
            }
            acc = warp_sum(acc);
            if (lane == 0) tsc[w] = acc * kq_scale;
        }
        __syncthreads();
        float tmx = -INFINITY;
        for (int j = tid; j < nt; j += 256) tmx = fmaxf(tmx, tsc[j]);
        tmx = block_max(tmx, red);
        const float mn = fmaxf(m, tmx);
        float ts = 0.0f;
        for (int j = tid; j < nt; j += 256) {
            const float e = expf(tsc[j] - mn);
            tsc[j] = e;
            ts += e;
        }
        ts = block_sum(ts, red);
        __syncthreads();   // every tsc[j] is final before any thread reads the whole tile
        const float corr = (m == -INFINITY) ? 0.0f : expf(m - mn);
        l = l * corr + ts;
        o *= corr;
        m = mn;
        if (!ISO3) {
            const float * vf = reinterpret_cast<const float *>(vc);
            for (int j = 0; j < nt; ++j) o += tsc[j] * vf[static_cast<int64_t>(j0 + j) * (NKV * HD) + g * HD + tid];
        } else {
            const int grp = tid >> 2, c = tid & 3, hf = grp >> 5, gi = grp & 31;
            for (int j = 0; j < nt; ++j) {
                const uint8_t * b = vc + (static_cast<int64_t>(j0 + j) * 8 + g * 2 + hf) * 50;
                float r[4];
                iso3_group(b, gi, r);
                o += tsc[j] * (r[c] * iso3_norm(b));
            }
        }
        __syncthreads();   // tsc is rewritten by the next tile
    }
    const float gate = qfull[static_cast<int64_t>(t) * (NH * 2 * HD) + h * 2 * HD + HD + tid];
    attn_o[static_cast<int64_t>(t) * (NH * HD) + h * HD + tid] = o * (1.0f / l) * sigm_f(gate);
}

// Split-K decode attention (flash-decoding) for ctx_max > 8192 and small batches (T <= kSplitTMax): the attention of
// one (token, head) pair is spread over kSplits blocks, each running the online softmax of the tiled kernel over its
// own slice of the positions and writing (m, l, o[256]) to `part`; attention_combine merges the slices. Without it a
// decode step at 128K positions runs only 24 blocks (one per head) over the whole cache.
constexpr int kSplits = 64;
constexpr int kSplitTMax = 8;

template <bool ISO3>
__global__ void attention_kernel_split(const float * qbuf, const uint8_t * kc, const uint8_t * vc, float * part, int pos0,
                                       const int * pos_dev) {
    // One block per (token, kv group, slice): the 6 query heads of the group share every K / V row, so each row is read
    // and dequantized once instead of six times (GQA 24 / 4).
    constexpr int QG = NH / NKV;
    if (pos_dev) pos0 = *pos_dev;
    __shared__ float qs[QG][HD], tsc[QG][kTile], sm[QG], sl[QG], scorr[QG];
    const int t = blockIdx.x / NKV, g = blockIdx.x % NKV, sp = blockIdx.y;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int n = pos0 + t + 1;
    const int chunk = (n + kSplits - 1) / kSplits;
    const int js = min(n, sp * chunk), je = min(n, js + chunk);
    for (int i = tid; i < QG * HD; i += 256) qs[i / HD][i % HD] = qbuf[static_cast<int64_t>(t) * (NH * HD) + (g * QG) * HD + i];
    if (tid < QG) { sm[tid] = -INFINITY; sl[tid] = 0.0f; }
    __syncthreads();
    const float kq_scale = 1.0f / sqrtf(static_cast<float>(HD));
    float o[QG];
#pragma unroll
    for (int hh = 0; hh < QG; ++hh) o[hh] = 0.0f;
    for (int j0 = js; j0 < je; j0 += kTile) {
        const int nt = min(kTile, je - j0);
        for (int w = warp; w < nt; w += 8) {   // one warp scores one position for the 6 heads
            const int j = j0 + w;
            float kv[8];
            int d0, d1;   // the lane's two 4-value (iso3) or one 8-value (f32) slices of the 256 dimensions
            if (!ISO3) {
                const float4 * k4 = reinterpret_cast<const float4 *>(reinterpret_cast<const float *>(kc) +
                                                                      static_cast<int64_t>(j) * (NKV * HD) + g * HD + lane * 8);
                const float4 a = k4[0], b = k4[1];
                kv[0] = a.x; kv[1] = a.y; kv[2] = a.z; kv[3] = a.w; kv[4] = b.x; kv[5] = b.y; kv[6] = b.z; kv[7] = b.w;
                d0 = lane * 8; d1 = lane * 8 + 4;
            } else {
                const uint8_t * kb = kc + (static_cast<int64_t>(j) * 8 + g * 2) * 50;
#pragma unroll
                for (int hf = 0; hf < 2; ++hf) {
                    const uint8_t * b = kb + hf * 50;
                    float r[4];
                    iso3_group(b, lane, r);
                    const float nm = iso3_norm(b);
#pragma unroll
                    for (int c = 0; c < 4; ++c) kv[hf * 4 + c] = r[c] * nm;
                }
                d0 = 4 * lane; d1 = 128 + 4 * lane;
            }
#pragma unroll
            for (int hh = 0; hh < QG; ++hh) {
                float acc = qs[hh][d0] * kv[0] + qs[hh][d0 + 1] * kv[1] + qs[hh][d0 + 2] * kv[2] + qs[hh][d0 + 3] * kv[3] +
                            qs[hh][d1] * kv[4] + qs[hh][d1 + 1] * kv[5] + qs[hh][d1 + 2] * kv[6] + qs[hh][d1 + 3] * kv[7];
                acc = warp_sum(acc);
                if (lane == 0) tsc[hh][w] = acc * kq_scale;
            }
        }
        __syncthreads();
        if (warp < QG) {   // warp hh: online-softmax update of head hh over this tile
            const int hh = warp;
            float tmx = -INFINITY;
            for (int jj = lane; jj < nt; jj += 32) tmx = fmaxf(tmx, tsc[hh][jj]);
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) tmx = fmaxf(tmx, __shfl_xor_sync(0xffffffffu, tmx, off));
            const float mo = sm[hh];
            const float mn = fmaxf(mo, tmx);
            float ts = 0.0f;
            for (int jj = lane; jj < nt; jj += 32) {
                const float e = expf(tsc[hh][jj] - mn);
                tsc[hh][jj] = e;
                ts += e;
            }
            ts = warp_sum(ts);
            if (lane == 0) {
                const float corr = (mo == -INFINITY) ? 0.0f : expf(mo - mn);
                sl[hh] = sl[hh] * corr + ts;
                sm[hh] = mn;
                scorr[hh] = corr;
            }
        }
        __syncthreads();
#pragma unroll
        for (int hh = 0; hh < QG; ++hh) o[hh] *= scorr[hh];
        if (!ISO3) {
            const float * vf = reinterpret_cast<const float *>(vc);
            for (int jj = 0; jj < nt; ++jj) {
                const float v = vf[static_cast<int64_t>(j0 + jj) * (NKV * HD) + g * HD + tid];
#pragma unroll
                for (int hh = 0; hh < QG; ++hh) o[hh] += tsc[hh][jj] * v;
            }
        } else {
            const int grp = tid >> 2, c = tid & 3, hf = grp >> 5, gi = grp & 31;
            for (int jj = 0; jj < nt; ++jj) {
                const uint8_t * b = vc + (static_cast<int64_t>(j0 + jj) * 8 + g * 2 + hf) * 50;
                float r[4];
                iso3_group(b, gi, r);
                const float v = r[c] * iso3_norm(b);
#pragma unroll
                for (int hh = 0; hh < QG; ++hh) o[hh] += tsc[hh][jj] * v;
            }
        }
        __syncthreads();   // tsc / scorr are rewritten by the next tile
    }
#pragma unroll
    for (int hh = 0; hh < QG; ++hh) {
        float * pr = part + ((static_cast<int64_t>(t) * NH + g * QG + hh) * kSplits + sp) * (HD + 2);
        pr[2 + tid] = o[hh];
        if (tid == 0) { pr[0] = sm[hh]; pr[1] = sl[hh]; }
    }
}

// Merges the kSplits slices of one (token, head): out = sum_s o_s e^(m_s - M) / sum_s l_s e^(m_s - M), then the gate.
__global__ void attention_combine(const float * part, const float * qfull, float * attn_o) {
    const int t = blockIdx.x / NH, h = blockIdx.x % NH, tid = threadIdx.x;
    const float * pr = part + static_cast<int64_t>(blockIdx.x) * kSplits * (HD + 2);
    float M = -INFINITY;
    for (int s = 0; s < kSplits; ++s) M = fmaxf(M, pr[s * (HD + 2)]);
    float L = 0.0f, O = 0.0f;
    for (int s = 0; s < kSplits; ++s) {
        const float ms = pr[s * (HD + 2)];
        if (ms == -INFINITY) continue;   // empty slice (short context)
        const float w = expf(ms - M);
        L += pr[s * (HD + 2) + 1] * w;
        O += pr[s * (HD + 2) + 2 + tid] * w;
    }
    const float gate = qfull[static_cast<int64_t>(t) * (NH * 2 * HD) + h * 2 * HD + HD + tid];
    attn_o[static_cast<int64_t>(t) * (NH * HD) + h * HD + tid] = O * (1.0f / L) * sigm_f(gate);
}

// ---- Prefill attention with K/V reuse (ctx_max > 8192, T > kSplitTMax).
// The tiled kernel above runs one block per (token, head) and streams the whole K/V of its group for every one of the
// 24 heads of every query token. Here one block owns one KV group and kPQ consecutive query tokens: warp w handles token
// t0 + w and the 6 query heads of the group (48 (token, head) rows per block). Each K/V tile of kPP positions is read
// from global memory and dequantized (iso3 -> FP32) once into shared memory and then used by all 48 rows, so the K/V
// traffic and the dequantization work drop by 48x against the tiled kernel (6 heads x 8 tokens).
// Lane layout: lane L holds dims 8L .. 8L+7 of q and of the output accumulator for each of the 6 rows (96 registers).
// Scores: per row, 8 positions at a time are reduced with a reduce-scatter over the warp (9 shuffles for 8 dot products
// instead of 40); lane L then owns the score of position (L >> 2) & 7. Online softmax per row (m, l, o) in FP32.
// Split over positions (blockIdx.z, ns slices) keeps the GPU busy for short prefill chunks (T = 64 at 128K): every slice
// writes (m, l, o) partials that attention_combine_n merges. ns == 1 writes the gated output directly.
constexpr int kPQ = 8;                 // query tokens per block (= warps)
constexpr int kPP = 16;                // positions per K/V tile
constexpr int kQG = NH / NKV;          // 6 query heads per KV group
constexpr int kPrefillPartRows = kSplitTMax * kSplits;   // (token x slice) rows of the shared partial buffer per head

// v[0..7] (one value per position of a group of 8, partial over this lane's dims) -> sum over the 32 lanes of
// v[(lane >> 2) & 7], returned in every lane.
__device__ __forceinline__ float reduce_scatter8(float (&v)[8], int lane) {
    const bool b4 = lane & 16, b3 = lane & 8, b2 = lane & 4;
    float a[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float send = b4 ? v[i] : v[i + 4], keep = b4 ? v[i + 4] : v[i];
        a[i] = keep + __shfl_xor_sync(0xffffffffu, send, 16);
    }
    float c[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const float send = b3 ? a[i] : a[i + 2], keep = b3 ? a[i + 2] : a[i];
        c[i] = keep + __shfl_xor_sync(0xffffffffu, send, 8);
    }
    float e;
    {
        const float send = b2 ? c[0] : c[1], keep = b2 ? c[1] : c[0];
        e = keep + __shfl_xor_sync(0xffffffffu, send, 4);
    }
    e += __shfl_xor_sync(0xffffffffu, e, 1);
    e += __shfl_xor_sync(0xffffffffu, e, 2);
    return e;
}

template <bool ISO3, int HPW>
__global__ void __launch_bounds__(kPQ * (kQG / HPW) * 32) attention_kernel_prefill(const float * __restrict__ qbuf, const float * __restrict__ qfull,
                                                                const uint8_t * __restrict__ kc, const uint8_t * __restrict__ vc,
                                                                float * __restrict__ attn_o, float * __restrict__ part, int T,
                                                                int pos0, int ns) {
    __shared__ float4 ks[kPP][HD / 4], vs[kPP][HD / 4];   // 2 x 16 KiB
    // iso3: the raw bytes of the tile (100 contiguous bytes per position and group, for K and V) are staged with
    // coalesced 32-bit loads, then dequantized from shared memory; codebook in shared memory, the quaternion of the
    // thread's group index (always tid & 31) in registers.
    __shared__ uint32_t raw[ISO3 ? 2 : 1][ISO3 ? kPP : 1][25];
    __shared__ float s_cb[8];
    constexpr int WPT = kQG / HPW;               // warps per query token
    constexpr int NT = kPQ * WPT * 32;           // threads per block
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int t0 = blockIdx.x * kPQ, g = blockIdx.y, sp = blockIdx.z;
    const int t = t0 + warp / WPT, hb = (warp % WPT) * HPW;   // token and first head (within the group) of this warp
    const bool live = t < T;
    const int qpos = pos0 + (live ? t : T - 1);                  // last position this row may see
    const int last = pos0 + min(t0 + kPQ, T) - 1;                // last position any row of the block sees
    const int chunk = ((last + 1 + ns - 1) / ns + kPP - 1) / kPP * kPP;
    const int js = sp * chunk, je = min(last + 1, js + chunk);
    const float kq_scale = 1.0f / sqrtf(static_cast<float>(HD));
    float qw = 0.f, qxq = 0.f, qyq = 0.f, qzq = 0.f;
    if (ISO3) {
        if (tid < 8) s_cb[tid] = iso3_codebook[tid];
        const int gi = tid & 31;
        qw = iso3_quat_w[gi]; qxq = -iso3_quat_x[gi]; qyq = -iso3_quat_y[gi]; qzq = -iso3_quat_z[gi];
    }

    float q[HPW][8], o[HPW][8], m[HPW], l[HPW];
#pragma unroll
    for (int r = 0; r < HPW; ++r) {
        const float4 * qp = reinterpret_cast<const float4 *>(qbuf + (static_cast<int64_t>(live ? t : 0) * NH + g * kQG + hb + r) * HD + lane * 8);
        const float4 a = qp[0], b = qp[1];
        q[r][0] = a.x * kq_scale; q[r][1] = a.y * kq_scale; q[r][2] = a.z * kq_scale; q[r][3] = a.w * kq_scale;
        q[r][4] = b.x * kq_scale; q[r][5] = b.y * kq_scale; q[r][6] = b.z * kq_scale; q[r][7] = b.w * kq_scale;
        m[r] = -INFINITY;
        l[r] = 0.0f;
#pragma unroll
        for (int k = 0; k < 8; ++k) o[r][k] = 0.0f;
    }

    // iso3: the raw bytes of the next tile are loaded into registers while the current tile is computed (one block per SM:
    // without this the global-load latency of every tile was exposed between two barriers). Same bytes, same result.
    constexpr int RPT = (2 * kPP * 25 + NT - 1) / NT;   // raw words per thread and tile
    uint32_t pre[RPT];
    auto load_raw = [&](int jt, int npt) {
#pragma unroll
        for (int k = 0; k < RPT; ++k) {
            const int i = tid + NT * k;
            const int kv = i / (kPP * 25), rem = i % (kPP * 25), p = rem / 25, w = rem % 25;
            pre[k] = (i < 2 * kPP * 25 && p < npt)
                         ? __ldg(reinterpret_cast<const uint32_t *>((kv ? vc : kc) + static_cast<int64_t>(jt + p) * 400 + g * 100) + w)
                         : 0u;
        }
    };
    if (ISO3 && js < je) load_raw(js, min(kPP, je - js));

    for (int j0 = js; j0 < je; j0 += kPP) {
        const int np = min(kPP, je - j0);
        __syncthreads();   // previous tile fully consumed
        if (ISO3) {
#pragma unroll
            for (int k = 0; k < RPT; ++k) {
                const int i = tid + NT * k;
                if (i < 2 * kPP * 25) {
                    const int kv = i / (kPP * 25), rem = i % (kPP * 25), p = rem / 25, w = rem % 25;
                    raw[kv][p][w] = pre[k];
                }
            }
            __syncthreads();
            if (j0 + kPP < je) load_raw(j0 + kPP, min(kPP, je - j0 - kPP));   // in flight during the dequant and the compute
            // 2 tensors x 16 positions x 2 halves x 32 groups of 4 values = 2048 groups; the group index is tid & 31
#pragma unroll
            for (int i = 0; i < 2048 / NT; ++i) {
                const int idx = tid + NT * i;
                const int kv = idx >> 10, rem = idx & 1023, p = rem >> 6, hf = (rem >> 5) & 1, gi = rem & 31;
                const uint8_t * blk = reinterpret_cast<const uint8_t *>(raw[kv][p]) + hf * 50;
                const uint32_t qsb = blk[2 + gi];
                const uint32_t sg = (blk[34 + (gi >> 1)] >> (4 * (gi & 1))) & 0xF;
                float qv[4];
#pragma unroll
                for (int c = 0; c < 4; ++c) qv[c] = s_cb[((qsb >> (2 * c)) & 3) | (((sg >> c) & 1) << 2)];
                const float r0 = qw * qv[0] - qxq * qv[1] - qyq * qv[2] - qzq * qv[3];
                const float r1 = qw * qv[1] + qxq * qv[0] + qyq * qv[3] - qzq * qv[2];
                const float r2 = qw * qv[2] - qxq * qv[3] + qyq * qv[0] + qzq * qv[1];
                const float r3 = qw * qv[3] + qxq * qv[2] - qyq * qv[1] + qzq * qv[0];
                const float nm = f16_to_f32(static_cast<uint32_t>(blk[0]) | (static_cast<uint32_t>(blk[1]) << 8));
                (kv ? vs : ks)[p][hf * 32 + gi] = p < np ? make_float4(r0 * nm, r1 * nm, r2 * nm, r3 * nm) : make_float4(0.f, 0.f, 0.f, 0.f);
            }
        } else {
#pragma unroll
            for (int i = 0; i < 2048 / NT; ++i) {
                const int idx = tid + NT * i;
                const int kv = idx >> 10, rem = idx & 1023, p = rem >> 6, c = rem & 63;
                float4 val = make_float4(0.f, 0.f, 0.f, 0.f);
                if (p < np)
                    val = reinterpret_cast<const float4 *>(reinterpret_cast<const float *>(kv ? vc : kc) +
                                                           static_cast<int64_t>(j0 + p) * (NKV * HD) + g * HD)[c];
                (kv ? vs : ks)[p][c] = val;
            }
        }
        __syncthreads();
        const int nv = min(np, qpos - j0 + 1);   // causal: positions j0 .. j0 + nv - 1 are visible to this row
        if (!live || nv <= 0) continue;          // warp-uniform

        // scores: two groups of 8 positions; lane holds the score of position (lane >> 2) & 7 of each group
        float s[2][HPW];
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            float pv[HPW][8];
#pragma unroll
            for (int pp = 0; pp < 8; ++pp) {
                const float4 a = ks[c * 8 + pp][lane * 2], b = ks[c * 8 + pp][lane * 2 + 1];
#pragma unroll
                for (int r = 0; r < HPW; ++r)
                    pv[r][pp] = q[r][0] * a.x + q[r][1] * a.y + q[r][2] * a.z + q[r][3] * a.w + q[r][4] * b.x + q[r][5] * b.y +
                                q[r][6] * b.z + q[r][7] * b.w;
            }
            const int pos_rel = c * 8 + ((lane >> 2) & 7);
#pragma unroll
            for (int r = 0; r < HPW; ++r) {
                const float v = reduce_scatter8(pv[r], lane);
                s[c][r] = pos_rel < nv ? v : -INFINITY;
            }
        }
        // online softmax per row; every value appears in 4 lanes (bits 0 and 1 of the lane)
#pragma unroll
        for (int r = 0; r < HPW; ++r) {
            float tm = fmaxf(s[0][r], s[1][r]);
            tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, 4));
            tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, 8));
            tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, 16));
            const float mn = fmaxf(m[r], tm);   // finite: position j0 is visible (nv >= 1)
            s[0][r] = expf(s[0][r] - mn);
            s[1][r] = expf(s[1][r] - mn);
            float ts = s[0][r] + s[1][r];
            ts += __shfl_xor_sync(0xffffffffu, ts, 4);
            ts += __shfl_xor_sync(0xffffffffu, ts, 8);
            ts += __shfl_xor_sync(0xffffffffu, ts, 16);
            const float corr = expf(m[r] - mn);   // m = -inf on the first tile: corr = 0, o and l are 0 anyway
            l[r] = l[r] * corr + ts;
            m[r] = mn;
#pragma unroll
            for (int k = 0; k < 8; ++k) o[r][k] *= corr;
        }
        // o += p * V
        for (int p = 0; p < nv; ++p) {
            const float4 a = vs[p][lane * 2], b = vs[p][lane * 2 + 1];
            const int src = (p & 7) * 4;
#pragma unroll
            for (int r = 0; r < HPW; ++r) {
                const float w = __shfl_sync(0xffffffffu, p < 8 ? s[0][r] : s[1][r], src);
                o[r][0] += w * a.x; o[r][1] += w * a.y; o[r][2] += w * a.z; o[r][3] += w * a.w;
                o[r][4] += w * b.x; o[r][5] += w * b.y; o[r][6] += w * b.z; o[r][7] += w * b.w;
            }
        }
    }
    if (!live) return;
#pragma unroll
    for (int r = 0; r < HPW; ++r) {
        const int h = g * kQG + hb + r;
        if (ns == 1) {
            const float inv = 1.0f / l[r];
            const float * gp = qfull + static_cast<int64_t>(t) * (NH * 2 * HD) + h * 2 * HD + HD + lane * 8;
            float * dst = attn_o + static_cast<int64_t>(t) * (NH * HD) + h * HD + lane * 8;
            float4 a, b;
            a.x = o[r][0] * inv * sigm_f(gp[0]); a.y = o[r][1] * inv * sigm_f(gp[1]);
            a.z = o[r][2] * inv * sigm_f(gp[2]); a.w = o[r][3] * inv * sigm_f(gp[3]);
            b.x = o[r][4] * inv * sigm_f(gp[4]); b.y = o[r][5] * inv * sigm_f(gp[5]);
            b.z = o[r][6] * inv * sigm_f(gp[6]); b.w = o[r][7] * inv * sigm_f(gp[7]);
            reinterpret_cast<float4 *>(dst)[0] = a;
            reinterpret_cast<float4 *>(dst)[1] = b;
        } else {
            float * pr = part + ((static_cast<int64_t>(t) * NH + h) * ns + sp) * (HD + 2);
#pragma unroll
            for (int k = 0; k < 8; ++k) pr[2 + lane * 8 + k] = o[r][k];
            if (lane == 0) { pr[0] = m[r]; pr[1] = l[r]; }
        }
    }
}

// attention_combine for `ns` slices per (token, head).
__global__ void attention_combine_n(const float * part, const float * qfull, float * attn_o, int ns) {
    const int t = blockIdx.x / NH, h = blockIdx.x % NH, tid = threadIdx.x;
    const float * pr = part + static_cast<int64_t>(blockIdx.x) * ns * (HD + 2);
    float M = -INFINITY;
    for (int s = 0; s < ns; ++s) M = fmaxf(M, pr[s * (HD + 2)]);
    float L = 0.0f, O = 0.0f;
    for (int s = 0; s < ns; ++s) {
        const float ms = pr[s * (HD + 2)];
        if (ms == -INFINITY) continue;   // slice with no visible position for this row
        const float w = expf(ms - M);
        L += pr[s * (HD + 2) + 1] * w;
        O += pr[s * (HD + 2) + 2 + tid] * w;
    }
    const float gate = qfull[static_cast<int64_t>(t) * (NH * 2 * HD) + h * 2 * HD + HD + tid];
    attn_o[static_cast<int64_t>(t) * (NH * HD) + h * HD + tid] = O * (1.0f / L) * sigm_f(gate);
}


// ---- iso3 decode attention v2 (issue #55 M2, T <= kSplitTMax, ctx_max > 8192 or the Bonsai path). The split-K kernel above
// spent one warp per position with six full warp reductions (30 shuffles), a block-wide barrier per 256-position tile and
// a 4x redundant iso3 decode of V: about 12 GB/s of K/V traffic at 128K. Here every warp owns one slice of the positions
// and the 6 query heads of the KV group, with the lane layout of the prefill kernel (lane L: dims 8L .. 8L+7 = iso3 groups
// 2L, 2L+1 of half L/16) and no barrier in the loop. The quaternion rotation of iso3 is moved out of the loop: with
// k = nm * M c (M = the group's rotation, c = codebook values) the score is nm * (M^T q) . c, so q is rotated once, and
// V is accumulated as o' = sum p * nm * c and rotated once at the end (o = M o'). Scores of 8 positions per reduce-scatter,
// online softmax per 16 positions. Each warp writes its own (m, l, o) slice; attention_combine_n merges the ns slices.
constexpr int kDecWarps = 4;   // warps (slices) per block

__device__ __forceinline__ void iso3_codes8(const uint8_t * blk, int gi0, const float * cb, float (&c)[8]) {
    // codebook values of groups gi0 and gi0 + 1 (gi0 even) of one 50-byte iso3 half block (2-byte aligned: one 16-bit load)
    const uint32_t qs = __ldg(reinterpret_cast<const unsigned short *>(blk + 2 + gi0));
    const uint32_t sg = __ldg(blk + 34 + (gi0 >> 1));
#pragma unroll
    for (int k = 0; k < 8; ++k) c[k] = cb[((qs >> (2 * k)) & 3) | (((sg >> k) & 1) << 2)];
}

template <int NSL_MAX>
__global__ void __launch_bounds__(kDecWarps * 32, 3) attention_kernel_decode_iso3(const float * __restrict__ qbuf, const uint8_t * __restrict__ kc,
                                                                                  const uint8_t * __restrict__ vc, float * __restrict__ part,
                                                                                  int pos0, const int * pos_dev, int nsl) {
    constexpr int QG = NH / NKV;
    __shared__ float s_cb[8];
    if (pos_dev) pos0 = *pos_dev;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    if (tid < 8) s_cb[tid] = iso3_codebook[tid];
    __syncthreads();
    const int t = blockIdx.x / NKV, g = blockIdx.x % NKV;
    const int sl = blockIdx.y * kDecWarps + warp;           // slice of this warp
    const int n = pos0 + t + 1;
    const int chunk = ((n + nsl - 1) / nsl + 15) / 16 * 16;
    const int js = min(n, sl * chunk), je = min(n, js + chunk);
    const int hf = lane >> 4, gi0 = (2 * lane) & 31;
    const float kq_scale = 1.0f / sqrtf(static_cast<float>(HD));
    // rotation of the lane's two groups: M = L(p), p = (w, -x, -y, -z) as in iso3_group
    float pw[2], px[2], py[2], pz[2];
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        pw[k] = iso3_quat_w[gi0 + k]; px[k] = -iso3_quat_x[gi0 + k]; py[k] = -iso3_quat_y[gi0 + k]; pz[k] = -iso3_quat_z[gi0 + k];
    }
    float q[QG][8], o[QG][8], m[QG], l[QG];
#pragma unroll
    for (int r = 0; r < QG; ++r) {
        const float4 * qp = reinterpret_cast<const float4 *>(qbuf + (static_cast<int64_t>(t) * NH + g * QG + r) * HD + lane * 8);
        const float4 a = qp[0], b = qp[1];
        const float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
        for (int k = 0; k < 2; ++k) {   // q' = M^T q (scaled)
            const float q0 = v[4 * k] * kq_scale, q1 = v[4 * k + 1] * kq_scale, q2 = v[4 * k + 2] * kq_scale, q3 = v[4 * k + 3] * kq_scale;
            q[r][4 * k + 0] = pw[k] * q0 + px[k] * q1 + py[k] * q2 + pz[k] * q3;
            q[r][4 * k + 1] = -px[k] * q0 + pw[k] * q1 + pz[k] * q2 - py[k] * q3;
            q[r][4 * k + 2] = -py[k] * q0 - pz[k] * q1 + pw[k] * q2 + px[k] * q3;
            q[r][4 * k + 3] = -pz[k] * q0 + py[k] * q1 - px[k] * q2 + pw[k] * q3;
        }
        m[r] = -INFINITY;
        l[r] = 0.0f;
#pragma unroll
        for (int k = 0; k < 8; ++k) o[r][k] = 0.0f;
    }
    for (int j0 = js; j0 < je; j0 += 16) {
        float s[2][QG];
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            float pv[QG][8];
#pragma unroll
            for (int pp = 0; pp < 8; ++pp) {
                const int j = j0 + c * 8 + pp;
                float cv[8];
                float nm = 0.0f;
                if (j < je) {
                    const uint8_t * kb = kc + (static_cast<int64_t>(j) * 8 + g * 2 + hf) * 50;
                    iso3_codes8(kb, gi0, s_cb, cv);
                    nm = f16_to_f32(__ldg(reinterpret_cast<const unsigned short *>(kb)));
                } else {
#pragma unroll
                    for (int k = 0; k < 8; ++k) cv[k] = 0.0f;
                }
#pragma unroll
                for (int r = 0; r < QG; ++r) {
                    float d = q[r][0] * cv[0];
#pragma unroll
                    for (int k = 1; k < 8; ++k) d += q[r][k] * cv[k];
                    pv[r][pp] = d * nm;
                }
            }
            const bool ok = j0 + c * 8 + ((lane >> 2) & 7) < je;
#pragma unroll
            for (int r = 0; r < QG; ++r) {
                const float v = reduce_scatter8(pv[r], lane);
                s[c][r] = ok ? v : -INFINITY;
            }
        }
        float pr[2][QG];
#pragma unroll
        for (int r = 0; r < QG; ++r) {
            float tm = fmaxf(s[0][r], s[1][r]);
            tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, 4));
            tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, 8));
            tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, 16));
            const float mn = fmaxf(m[r], tm);   // finite: position j0 < je is valid
            pr[0][r] = expf(s[0][r] - mn);
            pr[1][r] = expf(s[1][r] - mn);
            float ts = pr[0][r] + pr[1][r];
            ts += __shfl_xor_sync(0xffffffffu, ts, 4);
            ts += __shfl_xor_sync(0xffffffffu, ts, 8);
            ts += __shfl_xor_sync(0xffffffffu, ts, 16);
            const float corr = expf(m[r] - mn);
            l[r] = l[r] * corr + ts;
            m[r] = mn;
#pragma unroll
            for (int k = 0; k < 8; ++k) o[r][k] *= corr;
        }
        const int nv = min(16, je - j0);
        for (int p = 0; p < nv; ++p) {
            const uint8_t * vb = vc + (static_cast<int64_t>(j0 + p) * 8 + g * 2 + hf) * 50;
            float cv[8];
            iso3_codes8(vb, gi0, s_cb, cv);
            const float nm = f16_to_f32(__ldg(reinterpret_cast<const unsigned short *>(vb)));
            const int src = (p & 7) * 4;
#pragma unroll
            for (int r = 0; r < QG; ++r) {
                const float w = __shfl_sync(0xffffffffu, p < 8 ? pr[0][r] : pr[1][r], src) * nm;
#pragma unroll
                for (int k = 0; k < 8; ++k) o[r][k] += w * cv[k];
            }
        }
    }
#pragma unroll
    for (int r = 0; r < QG; ++r) {
        float * dst = part + ((static_cast<int64_t>(t) * NH + g * QG + r) * nsl + sl) * (HD + 2);
        float out[8];
#pragma unroll
        for (int k = 0; k < 2; ++k) {   // o = M o'
            const float c0 = o[r][4 * k], c1 = o[r][4 * k + 1], c2 = o[r][4 * k + 2], c3 = o[r][4 * k + 3];
            out[4 * k + 0] = pw[k] * c0 - px[k] * c1 - py[k] * c2 - pz[k] * c3;
            out[4 * k + 1] = pw[k] * c1 + px[k] * c0 + py[k] * c3 - pz[k] * c2;
            out[4 * k + 2] = pw[k] * c2 - px[k] * c3 + py[k] * c0 + pz[k] * c1;
            out[4 * k + 3] = pw[k] * c3 + px[k] * c2 - py[k] * c1 + pz[k] * c0;
        }
        float * d8 = dst + 2 + lane * 8;   // slice rows are HD + 2 floats: not 16-byte aligned
#pragma unroll
        for (int k = 0; k < 8; ++k) d8[k] = out[k];
        if (lane == 0) { dst[0] = js < je ? m[r] : -INFINITY; dst[1] = l[r]; }
    }
}

}  // namespace

bool g_attn_tiled_ref = false;
// iso3 decode v2 (QWEN_ATTN_DEC=1, default) or the split-K kernel (0); op_attention_set_dec2 for tests
bool g_attn_dec2 = [] { const char * e = std::getenv("QWEN_ATTN_DEC"); return e ? std::atoi(e) != 0 : true; }();
void op_attention_set_dec2(bool on) { g_attn_dec2 = on; }
void op_attention_set_tiled_ref(bool on) { g_attn_tiled_ref = on; }
size_t op_attention_part_floats() { return static_cast<size_t>(kSplitTMax) * NH * kSplits * (HD + 2); }

cudaError_t op_rmsnorm_rows(const float * x, const float * w, float * y, int n, int rows, float eps, cudaStream_t s) {
    rmsnorm_rows_kernel<<<rows, 256, 0, s>>>(x, w, y, n, eps);
    return cudaGetLastError();
}
cudaError_t op_add_inplace(float * x, const float * y, int64_t n, cudaStream_t s) {
    add_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, s>>>(x, y, n);
    return cudaGetLastError();
}
cudaError_t op_silu_mul(float * g, const float * u, int64_t n, cudaStream_t s) {
    silu_mul_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, s>>>(g, u, n);
    return cudaGetLastError();
}
namespace {
// One token row: the 48 value heads of 128 values move from v to 3*(v%16) + v/16.
__global__ void gdn_v_permute_kernel(const float * __restrict__ src, float * __restrict__ dst, int T) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<int64_t>(T) * 48 * 128) return;
    const int64_t t = i / (48 * 128), r = i % (48 * 128);
    const int v = static_cast<int>(r / 128), d = static_cast<int>(r % 128);
    dst[static_cast<int64_t>(t) * 48 * 128 + static_cast<int64_t>(3 * (v % 16) + v / 16) * 128 + d] = src[i];
}
}  // namespace

cudaError_t op_gdn_v_permute(const float * src, float * dst, int T, cudaStream_t s) {
    const int64_t n = static_cast<int64_t>(T) * 48 * 128;
    gdn_v_permute_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, s>>>(src, dst, T);
    return cudaGetLastError();
}

cudaError_t op_dn_prep(const float * alpha, const float * beta_raw, const float * dt_bias, const float * ssm_a, float * gate,
                       float * beta, int T, cudaStream_t s) {
    const int n = T * VH;
    dn_prep_kernel<<<(n + 127) / 128, 128, 0, s>>>(alpha, beta_raw, dt_bias, ssm_a, gate, beta, n);
    return cudaGetLastError();
}
cudaError_t op_dn_conv(const float * qkv, const float * conv_w, float * conv_state, float * out, int T, cudaStream_t s) {
    dn_conv_kernel<<<(CONV_CH + 127) / 128, 128, 0, s>>>(qkv, conv_w, conv_state, out, T);
    return cudaGetLastError();
}
cudaError_t op_dn_rec(const float * conv_out, const float * gate, const float * beta, const float * z, const float * ssm_norm,
                      float * state_t, float * attn_o, int T, float eps, cudaStream_t s) {
    dn_rec_kernel<<<VH, DH, 0, s>>>(conv_out, gate, beta, z, ssm_norm, state_t, attn_o, T, eps);
    return cudaGetLastError();
}
cudaError_t op_dn_rec_ref(const float * conv_out, const float * gate, const float * beta, const float * z, const float * ssm_norm,
                          float * state_t, float * attn_o, int T, float eps, cudaStream_t s) {
    dn_rec_kernel_ref<<<VH, DH, 0, s>>>(conv_out, gate, beta, z, ssm_norm, state_t, attn_o, T, eps);
    return cudaGetLastError();
}
cudaError_t op_rope_table(float * cos_tab, float * sin_tab, int n_pos, float theta_scale, cudaStream_t s) {
    const int n = n_pos * (ROPE_DIM / 2);
    rope_table_kernel<<<(n + 255) / 256, 256, 0, s>>>(cos_tab, sin_tab, n_pos, theta_scale);
    return cudaGetLastError();
}
cudaError_t op_qk_prep(const float * qfull, const float * kproj, const float * qnorm, const float * knorm, float * qbuf,
                       float * kout, int T, int pos0, const float * rope_cos, const float * rope_sin, float eps, cudaStream_t s,
                       const int * pos_dev) {
    qk_prep_kernel<<<dim3(T, NH + NKV), HD, 0, s>>>(qfull, kproj, qnorm, knorm, qbuf, kout, pos0, pos_dev, rope_cos, rope_sin, eps);
    return cudaGetLastError();
}
cudaError_t op_attention(const float * qbuf, const float * qfull, const void * kcache, const void * vcache, float * attn_o,
                         int T, int pos0, bool iso3, int ctx_max, cudaStream_t s, const int * pos_dev, float * part,
                         bool long_kernels) {
    const dim3 grid(T * NH);
    // long_kernels (Ternary Bonsai): the split-K decode and K/V-reuse prefill kernels at every ctx_max. Without it (Qwen) the
    // shared-memory kernel below runs up to 8192 positions, unchanged and bit-identical to the stage 4/5 results.
    const bool longk = ctx_max > kFastCtxMax || long_kernels;
    if (longk && part && T <= kSplitTMax && iso3 && g_attn_dec2) {
        // decode v2: slices fixed by ctx_max (the CUDA graph replays the grid; the slice length follows the position on the
        // device), about 256 positions per warp at the full context, at most kSplitTMax * kSplits / T slices (part buffer)
        int nsl = std::min(ctx_max / 256, kSplitTMax * kSplits / T);
        nsl = std::max(kDecWarps, nsl / kDecWarps * kDecWarps);
        const dim3 g2(T * NKV, nsl / kDecWarps);
        attention_kernel_decode_iso3<0><<<g2, kDecWarps * 32, 0, s>>>(qbuf, static_cast<const uint8_t *>(kcache), static_cast<const uint8_t *>(vcache),
                                                                        part, pos0, pos_dev, nsl);
        attention_combine_n<<<grid, 256, 0, s>>>(part, qfull, attn_o, nsl);
        return cudaGetLastError();
    }
    if (longk && part && T <= kSplitTMax) {
        const dim3 g2(T * NKV, kSplits);
        if (iso3) attention_kernel_split<true><<<g2, 256, 0, s>>>(qbuf, static_cast<const uint8_t *>(kcache), static_cast<const uint8_t *>(vcache), part, pos0, pos_dev);
        else attention_kernel_split<false><<<g2, 256, 0, s>>>(qbuf, static_cast<const uint8_t *>(kcache), static_cast<const uint8_t *>(vcache), part, pos0, pos_dev);
        attention_combine<<<grid, 256, 0, s>>>(part, qfull, attn_o);
        return cudaGetLastError();
    }
    // Up to 8192 positions the scores fit in dynamic shared memory (32 KiB of the 48 KiB sm_61 limit) and the original
    // kernel runs unchanged (the Qwen path is bit-identical to before). Above that the tiled online-softmax kernel takes
    // over (ctx_max up to 131072, Bonsai path).
    if (longk && !g_attn_tiled_ref) {
        // prefill: query-tiled K/V-reuse kernel; slices over the positions while the grid is small and the partial
        // buffer (sized for kSplitTMax x kSplits rows per head) has room
        const int qt = (T + kPQ - 1) / kPQ;
        int ns = 1;
        if (part) {
            const int want = (2 * 19 * 8 + qt * NKV - 1) / (qt * NKV);   // about 2 waves of 8 blocks on the 19 SMs
            ns = std::max(1, std::min(want, kPrefillPartRows / T));
        }
        const dim3 gp(qt, NKV, ns);
        static int hpw = -1;
        if (hpw < 0) { const char * e = std::getenv("QWEN_ATTN_HPW"); hpw = e ? std::atoi(e) : 6; }
        const uint8_t * k8 = static_cast<const uint8_t *>(kcache), * v8 = static_cast<const uint8_t *>(vcache);
        if (hpw == 6) {   // 8 warps, 6 heads each (178 registers: one block per SM)
            if (iso3) attention_kernel_prefill<true, 6><<<gp, 256, 0, s>>>(qbuf, qfull, k8, v8, attn_o, part, T, pos0, ns);
            else attention_kernel_prefill<false, 6><<<gp, 256, 0, s>>>(qbuf, qfull, k8, v8, attn_o, part, T, pos0, ns);
        } else if (hpw == 2) {   // 24 warps, 2 heads each
            if (iso3) attention_kernel_prefill<true, 2><<<gp, 768, 0, s>>>(qbuf, qfull, k8, v8, attn_o, part, T, pos0, ns);
            else attention_kernel_prefill<false, 2><<<gp, 768, 0, s>>>(qbuf, qfull, k8, v8, attn_o, part, T, pos0, ns);
        } else {   // 16 warps, 3 heads each
            if (iso3) attention_kernel_prefill<true, 3><<<gp, 512, 0, s>>>(qbuf, qfull, k8, v8, attn_o, part, T, pos0, ns);
            else attention_kernel_prefill<false, 3><<<gp, 512, 0, s>>>(qbuf, qfull, k8, v8, attn_o, part, T, pos0, ns);
        }
        if (ns > 1) attention_combine_n<<<grid, 256, 0, s>>>(part, qfull, attn_o, ns);
        return cudaGetLastError();
    }
    if (ctx_max > kFastCtxMax) {   // reference tiled kernel (op_attention_set_tiled_ref, tests and benchmarks only)
        if (iso3) {
            attention_kernel_tiled<true><<<grid, 256, 0, s>>>(qbuf, qfull, static_cast<const uint8_t *>(kcache),
                                                              static_cast<const uint8_t *>(vcache), attn_o, pos0, pos_dev);
        } else {
            attention_kernel_tiled<false><<<grid, 256, 0, s>>>(qbuf, qfull, static_cast<const uint8_t *>(kcache),
                                                               static_cast<const uint8_t *>(vcache), attn_o, pos0, pos_dev);
        }
        return cudaGetLastError();
    }
    const size_t smem = static_cast<size_t>(ctx_max) * sizeof(float);
    if (iso3) {
        attention_kernel<true><<<grid, 256, smem, s>>>(qbuf, qfull, static_cast<const uint8_t *>(kcache),
                                                       static_cast<const uint8_t *>(vcache), attn_o, pos0, pos_dev);
    } else {
        attention_kernel<false><<<grid, 256, smem, s>>>(qbuf, qfull, static_cast<const uint8_t *>(kcache),
                                                        static_cast<const uint8_t *>(vcache), attn_o, pos0, pos_dev);
    }
    return cudaGetLastError();
}
