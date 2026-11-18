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
__global__ void dn_rec_kernel(const float * conv_out, const float * gate, const float * beta, const float * z,
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

}  // namespace

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
                         int T, int pos0, bool iso3, int ctx_max, cudaStream_t s, const int * pos_dev) {
    const size_t smem = static_cast<size_t>(ctx_max) * sizeof(float);
    const dim3 grid(T * NH);
    if (iso3) {
        attention_kernel<true><<<grid, 256, smem, s>>>(qbuf, qfull, static_cast<const uint8_t *>(kcache),
                                                       static_cast<const uint8_t *>(vcache), attn_o, pos0, pos_dev);
    } else {
        attention_kernel<false><<<grid, 256, smem, s>>>(qbuf, qfull, static_cast<const uint8_t *>(kcache),
                                                        static_cast<const uint8_t *>(vcache), attn_o, pos0, pos_dev);
    }
    return cudaGetLastError();
}
