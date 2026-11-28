// Isolated prototype: iso3 decode attention with an int8 (__dp4a) K path. Does NOT touch src/cuda/ops.cu.
// Proposal (E31): decode the 3-bit iso3 codes to int8 words (one 4096-entry shared-memory table: 8 code bits + 4 sign bits ->
// 4 int8 values), quantize q' = M^T q to int8 once (one scale per head), and dot 6 heads with 2 dp4a per head and position
// (12 dp4a instead of 48 FFMA). The cross-lane reduction runs on int32 inside each 16-lane half, the float conversion, the K
// block norm and the head scale are applied once per 8 positions, then the two halves are added.
// V stays on the FP32 pair-table path (VM = 0); VM = 1 adds the int8 V path (see the kernel).
// Reports: time (best/mean) of kernel+combine for the int8 path and of the current FP32 op_attention, the error of both against
// an FP64 reference built from the dequantized iso3 bytes and from the ORIGINAL float K/V (shows how much int8 adds on top of
// the 3-bit KV error), register/spill/occupancy, FNV hash of the output bits.
// Usage: attn_int8_bench [max_n=102400] [reps=20]   (tests n = 100, 4097 and max_n as prefixes of one cache)
// Build: see bench/README / the log entry (links libgpu_cuda.a and libiso3_ref.a). FP32/INT32 only, no half arithmetic.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <type_traits>
#include <vector>

#include "f16bits.hpp"
#include "iso3_ref.hpp"
#include "ops.cuh"

#define QT_BEGIN(type, name, size) static __device__ const type name[size] = {
#define QT_END() };
#include "tables.inc"
#undef QT_BEGIN
#undef QT_END

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

#ifndef MINB
#define MINB 3
#endif
// codebook scale of the int8 table: T = round(cb / cbmax * A). A = 124.75 gives T = {14, 43, 77, 125}: rms relative rounding error
// 0.18 % (A = 127 gives {14, 44, 78, 127}: 1.1 %, found by an offline scan, log E32).
#ifndef CB_A
#define CB_A 124.75f
#endif
constexpr int NH = 24, NKV = 4, HD = 256, KVW = NKV * HD, kDecWarps = 4, kNsl = 56;

// ------------------------------------------------------------------------------------------------------------- device
__device__ __forceinline__ float f16f(unsigned short h) { return f16_to_f32(h); }

// 8 positions' int32 partials (this lane's 8 dims) -> sum over the 16 lanes of the half for position (lane >> 1) & 7, every lane
// of the half ends with it (8 shuffles).
__device__ __forceinline__ int reduce_scatter8_half(const int (&v)[8], int lane) {
    const bool b3 = lane & 8, b2 = lane & 4, b1 = lane & 2;
    int a[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int send = b3 ? v[i] : v[i + 4], keep = b3 ? v[i + 4] : v[i];
        a[i] = keep + __shfl_xor_sync(0xffffffffu, send, 8);
    }
    int c[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int send = b2 ? a[i] : a[i + 2], keep = b2 ? a[i + 2] : a[i];
        c[i] = keep + __shfl_xor_sync(0xffffffffu, send, 4);
    }
    const int send = b1 ? c[0] : c[1], keep = b1 ? c[1] : c[0];
    int e = keep + __shfl_xor_sync(0xffffffffu, send, 2);
    e += __shfl_xor_sync(0xffffffffu, e, 1);
    return e;
}

__device__ __forceinline__ void iso3_codes8(const uint8_t * blk, int gi0, const float2 * cb2, float (&c)[8]) {
    const uint32_t qs = __ldg(reinterpret_cast<const unsigned short *>(blk + 2 + gi0));
    const uint32_t sg = __ldg(blk + 34 + (gi0 >> 1));
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const float2 v = cb2[((qs >> (4 * p)) & 15u) | (((sg >> (2 * p)) & 3u) << 4)];
        c[2 * p] = v.x; c[2 * p + 1] = v.y;
    }
}

template <int VM>
__global__ void __launch_bounds__(kDecWarps * 32, MINB) attn_i8(const float * __restrict__ qbuf, const uint8_t * __restrict__ kc,
                                                                 const uint8_t * __restrict__ vc, float * __restrict__ part, int pos0, int nsl) {
    constexpr int QG = NH / NKV;
    __shared__ float2 s_cb[64];
    __shared__ unsigned s_c8[4096];   // (8 code bits | 4 sign bits << 8) -> 4 int8 codebook values (byte k = element k)
    __shared__ int s_t8[8];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const float cbmax = iso3_codebook[7];
    if (tid < 64) {
        const int e = tid;
        s_cb[e] = make_float2(iso3_codebook[(e & 3) | (((e >> 4) & 1) << 2)], iso3_codebook[((e >> 2) & 3) | (((e >> 5) & 1) << 2)]);
    }
    if (tid < 8) s_t8[tid] = __float2int_rn(iso3_codebook[tid] * (CB_A / cbmax));
    __syncthreads();
    for (int e = tid; e < 4096; e += kDecWarps * 32) {
        unsigned w = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) w |= (static_cast<unsigned>(s_t8[((e >> (2 * k)) & 3) | (((e >> (8 + k)) & 1) << 2)]) & 0xFFu) << (8 * k);
        s_c8[e] = w;
    }
    __syncthreads();
    const int t = blockIdx.x / NKV, g = blockIdx.x % NKV;
    const int sl = blockIdx.y * kDecWarps + warp;
    const int n = pos0 + t + 1;
    const int chunk = ((n + nsl - 1) / nsl + 15) / 16 * 16;
    const int js = min(n, sl * chunk), je = min(n, js + chunk);
    const int hf = lane >> 4, gi0 = (2 * lane) & 31;
    const int P = (lane >> 1) & 7;   // position of this lane after the int reduce-scatter
    const float kq_scale = 1.0f / sqrtf(static_cast<float>(HD));
    float pw[2], px[2], py[2], pz[2];
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        pw[k] = iso3_quat_w[gi0 + k]; px[k] = -iso3_quat_x[gi0 + k]; py[k] = -iso3_quat_y[gi0 + k]; pz[k] = -iso3_quat_z[gi0 + k];
    }
    int q8lo[QG], q8hi[QG];
#ifdef QRES
    int q8blo[QG], q8bhi[QG];
#endif
    float ksc[QG], m[QG], l[QG], o[QG][8];
    float pmx[QG];
#pragma unroll
    for (int r = 0; r < QG; ++r) {
        const float4 * qp = reinterpret_cast<const float4 *>(qbuf + (static_cast<int64_t>(t) * NH + g * QG + r) * HD + lane * 8);
        const float4 a = qp[0], b = qp[1];
        const float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        float qf[8];
#pragma unroll
        for (int k = 0; k < 2; ++k) {   // q' = M^T q (scaled)
            const float q0 = v[4 * k] * kq_scale, q1 = v[4 * k + 1] * kq_scale, q2 = v[4 * k + 2] * kq_scale, q3 = v[4 * k + 3] * kq_scale;
            qf[4 * k + 0] = pw[k] * q0 + px[k] * q1 + py[k] * q2 + pz[k] * q3;
            qf[4 * k + 1] = -px[k] * q0 + pw[k] * q1 + pz[k] * q2 - py[k] * q3;
            qf[4 * k + 2] = -py[k] * q0 - pz[k] * q1 + pw[k] * q2 + px[k] * q3;
            qf[4 * k + 3] = -pz[k] * q0 + py[k] * q1 - px[k] * q2 + pw[k] * q3;
        }
        float amax = 0.0f;
#pragma unroll
        for (int k = 0; k < 8; ++k) amax = fmaxf(amax, fabsf(qf[k]));
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));   // head-wide scale
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        int qi[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) qi[k] = __float2int_rn(qf[k] * inv);
#ifdef QRES
        int qb[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) qb[k] = max(-127, min(127, __float2int_rn((qf[k] * inv - static_cast<float>(qi[k])) * 256.0f)));
        q8blo[r] = (qb[0] & 0xFF) | ((qb[1] & 0xFF) << 8) | ((qb[2] & 0xFF) << 16) | (qb[3] << 24);
        q8bhi[r] = (qb[4] & 0xFF) | ((qb[5] & 0xFF) << 8) | ((qb[6] & 0xFF) << 16) | (qb[7] << 24);
#endif
        q8lo[r] = (qi[0] & 0xFF) | ((qi[1] & 0xFF) << 8) | ((qi[2] & 0xFF) << 16) | (qi[3] << 24);
        q8hi[r] = (qi[4] & 0xFF) | ((qi[5] & 0xFF) << 8) | ((qi[6] & 0xFF) << 16) | (qi[7] << 24);
#ifdef QRES
        ksc[r] = amax * cbmax * (1.0f / (127.0f * 256.0f * CB_A));
#else
        ksc[r] = amax * cbmax * (1.0f / (127.0f * CB_A));
#endif   // q' ~ q8 * amax / 127, c ~ T * cbmax / A
        m[r] = -INFINITY;
        l[r] = 0.0f;
#pragma unroll
        for (int k = 0; k < 8; ++k) o[r][k] = 0.0f;
    }
    auto tile = [&](const int j0, auto full_tag) {
        constexpr bool FULL = decltype(full_tag)::value;
        float s[2][QG];
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            int pv[QG][8];
#pragma unroll
            for (int pp = 0; pp < 8; ++pp) {
                const int j = j0 + c * 8 + pp;
                unsigned wlo = 0, whi = 0;
                if (FULL || j < je) {
                    const uint8_t * kb = kc + (static_cast<int64_t>(j) * 8 + g * 2 + hf) * 50;
                    const uint32_t qs = __ldg(reinterpret_cast<const unsigned short *>(kb + 2 + gi0));
                    const uint32_t sg = __ldg(kb + 34 + (gi0 >> 1));
                    wlo = s_c8[(qs & 0xFFu) | ((sg & 0xFu) << 8)];
                    whi = s_c8[((qs >> 8) & 0xFFu) | ((sg >> 4) << 8)];
                }
#pragma unroll
#ifdef QRES
                for (int r = 0; r < QG; ++r) {
                    const int d1 = __dp4a(static_cast<int>(whi), q8hi[r], __dp4a(static_cast<int>(wlo), q8lo[r], 0));
                    const int d2 = __dp4a(static_cast<int>(whi), q8bhi[r], __dp4a(static_cast<int>(wlo), q8blo[r], 0));
                    pv[r][pp] = (d1 << 8) + d2;
                }
#else
                for (int r = 0; r < QG; ++r) pv[r][pp] = __dp4a(static_cast<int>(whi), q8hi[r], __dp4a(static_cast<int>(wlo), q8lo[r], 0));
#endif
            }
            const int jP = j0 + c * 8 + P;
            const bool ok = FULL || jP < je;
            const float nmP = f16f(__ldg(reinterpret_cast<const unsigned short *>(kc + (static_cast<int64_t>(min(jP, je - 1)) * 8 + g * 2 + hf) * 50)));
#pragma unroll
            for (int r = 0; r < QG; ++r) {
                const int e = reduce_scatter8_half(pv[r], lane);
                float f = static_cast<float>(e) * (nmP * ksc[r]);
                f += __shfl_xor_sync(0xffffffffu, f, 16);   // add the two 128-dim halves
                s[c][r] = ok ? f : -INFINITY;
            }
        }
        float pr[2][QG];
#pragma unroll
        for (int r = 0; r < QG; ++r) {
            float tm = fmaxf(s[0][r], s[1][r]);
            tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, 2));
            tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, 4));
            tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, 8));
            const float mn = fmaxf(m[r], tm);
            pmx[r] = expf(tm - mn);   // largest probability of this tile (<= 1): scale of the int8 weights
            pr[0][r] = expf(s[0][r] - mn);
            pr[1][r] = expf(s[1][r] - mn);
            float ts = pr[0][r] + pr[1][r];
            ts += __shfl_xor_sync(0xffffffffu, ts, 2);
            ts += __shfl_xor_sync(0xffffffffu, ts, 4);
            ts += __shfl_xor_sync(0xffffffffu, ts, 8);
            const float corr = expf(m[r] - mn);
            l[r] = l[r] * corr + ts;
            m[r] = mn;
#pragma unroll
            for (int k = 0; k < 8; ++k) o[r][k] *= corr;
        }
        auto vstep = [&](const int p) {
            const uint8_t * vb = vc + (static_cast<int64_t>(j0 + p) * 8 + g * 2 + hf) * 50;
            float cv[8];
            iso3_codes8(vb, gi0, s_cb, cv);
            const float nm = f16f(__ldg(reinterpret_cast<const unsigned short *>(vb)));
            const int src = (p & 7) * 2;
#pragma unroll
            for (int r = 0; r < QG; ++r) {
                const float w = __shfl_sync(0xffffffffu, p < 8 ? pr[0][r] : pr[1][r], src) * nm;
#pragma unroll
                for (int k = 0; k < 8; ++k) o[r][k] += w * cv[k];
            }
        };
        if constexpr (VM == 1) {
            // ---- int8 V: weights w = p * nm_v quantized to int8 (scale = tile max p * tile max nm_v), codes transposed over 4 positions
            const int jn = min(j0 + (lane & 15), je - 1);
            float nmreg = f16f(__ldg(reinterpret_cast<const unsigned short *>(vc + (static_cast<int64_t>(jn) * 8 + g * 2 + hf) * 50)));
            nmreg = (FULL || j0 + (lane & 15) < je) ? nmreg : 0.0f;
            float nmmax = nmreg;
            nmmax = fmaxf(nmmax, __shfl_xor_sync(0xffffffffu, nmmax, 1));
            nmmax = fmaxf(nmmax, __shfl_xor_sync(0xffffffffu, nmmax, 2));
            nmmax = fmaxf(nmmax, __shfl_xor_sync(0xffffffffu, nmmax, 4));
            nmmax = fmaxf(nmmax, __shfl_xor_sync(0xffffffffu, nmmax, 8));
            float invsw[QG], sc[QG];
#pragma unroll
            for (int r = 0; r < QG; ++r) {
                const float wmax = pmx[r] * nmmax;
                invsw[r] = wmax > 0.0f ? 127.0f / wmax : 0.0f;
                sc[r] = wmax * (1.0f / 127.0f) * (cbmax / CB_A);
            }
            int acc[QG][8];
#pragma unroll
            for (int r = 0; r < QG; ++r)
#pragma unroll
                for (int d = 0; d < 8; ++d) acc[r][d] = 0;
            const int ngrp = FULL ? 4 : (min(16, je - j0) + 3) >> 2;
#pragma unroll
            for (int pg = 0; pg < 4; ++pg) {
                if (!FULL && pg >= ngrp) break;
                unsigned alo[4], ahi[4];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int jj = min(j0 + pg * 4 + k, je - 1);
                    const uint8_t * vb = vc + (static_cast<int64_t>(jj) * 8 + g * 2 + hf) * 50;
                    const uint32_t qs = __ldg(reinterpret_cast<const unsigned short *>(vb + 2 + gi0));
                    const uint32_t sg = __ldg(vb + 34 + (gi0 >> 1));
                    alo[k] = s_c8[(qs & 0xFFu) | ((sg & 0xFu) << 8)];
                    ahi[k] = s_c8[((qs >> 8) & 0xFFu) | ((sg >> 4) << 8)];
                }
                unsigned tl[4], th[4];   // 4x4 byte transposes: tl[d] = bytes (alo[0].d, alo[1].d, alo[2].d, alo[3].d)
                {
                    const unsigned u0 = __byte_perm(alo[0], alo[1], 0x5140), u1 = __byte_perm(alo[0], alo[1], 0x7362);
                    const unsigned u2 = __byte_perm(alo[2], alo[3], 0x5140), u3 = __byte_perm(alo[2], alo[3], 0x7362);
                    tl[0] = __byte_perm(u0, u2, 0x5410); tl[1] = __byte_perm(u0, u2, 0x7632);
                    tl[2] = __byte_perm(u1, u3, 0x5410); tl[3] = __byte_perm(u1, u3, 0x7632);
                    const unsigned w0 = __byte_perm(ahi[0], ahi[1], 0x5140), w1 = __byte_perm(ahi[0], ahi[1], 0x7362);
                    const unsigned w2 = __byte_perm(ahi[2], ahi[3], 0x5140), w3 = __byte_perm(ahi[2], ahi[3], 0x7362);
                    th[0] = __byte_perm(w0, w2, 0x5410); th[1] = __byte_perm(w0, w2, 0x7632);
                    th[2] = __byte_perm(w1, w3, 0x5410); th[3] = __byte_perm(w1, w3, 0x7632);
                }
                float nmp[4];
#pragma unroll
                for (int k = 0; k < 4; ++k) nmp[k] = __shfl_sync(0xffffffffu, nmreg, (lane & 16) | (pg * 4 + k));
#pragma unroll
                for (int r = 0; r < QG; ++r) {
                    unsigned u[4];
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const int p = pg * 4 + k;
                        const float wf = __shfl_sync(0xffffffffu, p < 8 ? pr[0][r] : pr[1][r], (p & 7) * 2) * nmp[k];
                        u[k] = __float_as_uint(fmaf(wf, invsw[r], 12582912.0f));   // low byte = round(wf * invsw) as int8
                    }
                    const unsigned wq = __byte_perm(__byte_perm(u[0], u[1], 0x0040), __byte_perm(u[2], u[3], 0x0040), 0x5410);
#pragma unroll
                    for (int d = 0; d < 4; ++d) { acc[r][d] = __dp4a(static_cast<int>(tl[d]), static_cast<int>(wq), acc[r][d]);
                                                  acc[r][d + 4] = __dp4a(static_cast<int>(th[d]), static_cast<int>(wq), acc[r][d + 4]); }
                }
            }
#pragma unroll
            for (int r = 0; r < QG; ++r)
#pragma unroll
                for (int d = 0; d < 8; ++d) o[r][d] += static_cast<float>(acc[r][d]) * sc[r];
        } else if constexpr (FULL) {
#pragma unroll 4
            for (int p = 0; p < 16; ++p) vstep(p);
        } else {
            const int nv = min(16, je - j0);
            for (int p = 0; p < nv; ++p) vstep(p);
        }
    };
    int j0 = js;
    for (; j0 + 16 <= je; j0 += 16) tile(j0, std::true_type{});
    if (j0 < je) tile(j0, std::false_type{});
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
        float * d8 = dst + 2 + lane * 8;
#pragma unroll
        for (int k = 0; k < 8; ++k) d8[k] = out[k];
        if (lane == 0) { dst[0] = js < je ? m[r] : -INFINITY; dst[1] = l[r]; }
    }
}

__global__ void combine_n(const float * part, const float * qfull, float * attn_o, int ns) {
    const int t = blockIdx.x / NH, h = blockIdx.x % NH, tid = threadIdx.x;
    const float * pr = part + static_cast<int64_t>(blockIdx.x) * ns * (HD + 2);
    float M = -INFINITY;
    for (int s = 0; s < ns; ++s) M = fmaxf(M, pr[s * (HD + 2)]);
    float L = 0.0f, O = 0.0f;
    for (int s = 0; s < ns; ++s) {
        const float ms = pr[s * (HD + 2)];
        if (ms == -INFINITY) continue;
        const float w = expf(ms - M);
        L += pr[s * (HD + 2) + 1] * w;
        O += pr[s * (HD + 2) + 2 + tid] * w;
    }
    const float gate = qfull[static_cast<int64_t>(t) * (NH * 2 * HD) + h * 2 * HD + HD + tid];
    attn_o[static_cast<int64_t>(t) * (NH * HD) + h * HD + tid] = O * (1.0f / L) * (1.0f / (1.0f + expf(-gate)));
}

// ------------------------------------------------------------------------------------------------------------- host
struct Err { double max_abs, rms_ref, rel; };
static Err compare(const std::vector<float> & a, const std::vector<double> & b) {
    double mx = 0, ss = 0;
    for (size_t i = 0; i < a.size(); ++i) { mx = std::max(mx, std::fabs(a[i] - b[i])); ss += b[i] * b[i]; }
    const double rms = std::sqrt(ss / a.size());
    return {mx, rms, mx / std::max(rms, 1e-30)};
}
static Err compare32(const std::vector<float> & a, const std::vector<float> & b) {
    std::vector<double> bd(b.begin(), b.end());
    return compare(a, bd);
}

// double reference of one query row (T = 1, positions 0..n-1) for K/V given as float rows [n][KVW]
static void reference(const std::vector<float> & q, const std::vector<float> & qfull, const std::vector<float> & K, const std::vector<float> & V,
                      int n, std::vector<double> & out) {
    out.assign(NH * HD, 0.0);
    const double scale = 1.0 / std::sqrt(static_cast<double>(HD));
    std::vector<double> sc(n);
    for (int h = 0; h < NH; ++h) {
        const int g = h / (NH / NKV);
        const float * qh = q.data() + static_cast<size_t>(h) * HD;
        double mx = -1e300;
        for (int j = 0; j < n; ++j) {
            const float * k = K.data() + static_cast<size_t>(j) * KVW + g * HD;
            double s = 0;
            for (int d = 0; d < HD; ++d) s += static_cast<double>(qh[d]) * k[d];
            sc[j] = s * scale; mx = std::max(mx, sc[j]);
        }
        double sum = 0;
        for (int j = 0; j < n; ++j) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
        double * o = out.data() + static_cast<size_t>(h) * HD;
        for (int j = 0; j < n; ++j) {
            const float * v = V.data() + static_cast<size_t>(j) * KVW + g * HD;
            const double w = sc[j] / sum;
            for (int d = 0; d < HD; ++d) o[d] += w * v[d];
        }
        const float * gate = qfull.data() + static_cast<size_t>(h) * 2 * HD + HD;
        for (int d = 0; d < HD; ++d) o[d] *= 1.0 / (1.0 + std::exp(-static_cast<double>(gate[d])));
    }
}

static uint64_t fnv(const std::vector<float> & o) {
    uint64_t h = 1469598103934665603ull;
    for (float v : o) { uint32_t u; std::memcpy(&u, &v, 4); h = (h ^ u) * 1099511628211ull; }
    return h;
}

int main(int argc, char ** argv) {
    const int max_n = argc > 1 ? std::atoi(argv[1]) : 102400;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 20;
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> q(NH * HD), qfull(NH * 2 * HD);
    for (float & v : q) v = nd(rng);
    for (float & v : qfull) v = nd(rng);
    std::vector<float> K(static_cast<size_t>(max_n) * KVW), V(K.size());
    for (float & v : K) v = nd(rng);
    for (float & v : V) v = nd(rng);

    float * dq, * dqf, * dout32, * dout8, * dsrc, * dpart32, * dpart8;
    void * kc, * vc;
    CK(cudaMalloc(&dq, q.size() * 4)); CK(cudaMalloc(&dqf, qfull.size() * 4)); CK(cudaMalloc(&dout32, NH * HD * 4)); CK(cudaMalloc(&dout8, NH * HD * 4));
    CK(cudaMalloc(&dsrc, sizeof(float) * 512 * KVW)); CK(cudaMalloc(&dpart32, sizeof(float) * op_attention_part_floats()));
    CK(cudaMalloc(&dpart8, sizeof(float) * NH * kNsl * (HD + 2)));
    CK(cudaMalloc(&kc, static_cast<size_t>(max_n) * 400)); CK(cudaMalloc(&vc, static_cast<size_t>(max_n) * 400));
    CK(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dqf, qfull.data(), qfull.size() * 4, cudaMemcpyHostToDevice));
    for (int which = 0; which < 2; ++which) {
        const std::vector<float> & src = which ? V : K;
        for (int p = 0; p < max_n; p += 512) {
            const int m = std::min(512, max_n - p);
            CK(cudaMemcpy(dsrc, src.data() + static_cast<size_t>(p) * KVW, sizeof(float) * m * KVW, cudaMemcpyHostToDevice));
            CK(op_kv_store(dsrc, which ? vc : kc, m, p, true, 0));
        }
    }
    CK(cudaDeviceSynchronize());
    // host copy of the dequantized cache (what the GPU kernels actually read)
    std::vector<float> Kh(K.size()), Vh(K.size());
    {
        std::vector<uint8_t> b(static_cast<size_t>(max_n) * 400);
        for (int which = 0; which < 2; ++which) {
            CK(cudaMemcpy(b.data(), which ? vc : kc, b.size(), cudaMemcpyDeviceToHost));
            std::vector<float> & dst = which ? Vh : Kh;
            for (size_t blk = 0; blk < static_cast<size_t>(max_n) * 8; ++blk) iso3_dequantize_block_ref(b.data() + blk * 50, dst.data() + blk * 128);
        }
    }
    for (int vm = 0; vm < 2; ++vm) {
        cudaFuncAttributes fa;
        CK(cudaFuncGetAttributes(&fa, vm ? attn_i8<1> : attn_i8<0>));
        int occ = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, vm ? attn_i8<1> : attn_i8<0>, kDecWarps * 32, 0));
        std::printf("kernel %s: %d registers, %zu B local (stack/spill), %zu B static smem, %d blocks/SM (launch_bounds min %d)\n",
                    vm ? "int8 K + int8 V" : "int8 K + FP32 V", fa.numRegs, fa.localSizeBytes, fa.sharedSizeBytes, occ, MINB);
    }

    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    int g_vm = 0;
    auto run8 = [&](int n) {
        if (g_vm) attn_i8<1><<<dim3(NKV, kNsl / kDecWarps), kDecWarps * 32>>>(dq, static_cast<const uint8_t *>(kc), static_cast<const uint8_t *>(vc), dpart8, n - 1, kNsl);
        else attn_i8<0><<<dim3(NKV, kNsl / kDecWarps), kDecWarps * 32>>>(dq, static_cast<const uint8_t *>(kc), static_cast<const uint8_t *>(vc), dpart8, n - 1, kNsl);
        combine_n<<<NH, HD>>>(dpart8, dqf, dout8, kNsl);
        CK(cudaGetLastError());
    };
    auto run32 = [&](int n) { CK(op_attention(dq, dqf, kc, vc, dout32, 1, n - 1, true, 131072, 0, nullptr, dpart32, true)); };
    auto timeit = [&](auto f) {
        f(); CK(cudaDeviceSynchronize());
        float best = 1e30f, tot = 0;
        for (int i = 0; i < reps; ++i) {
            CK(cudaEventRecord(e0)); f(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); best = std::min(best, ms); tot += ms;
        }
        return std::make_pair(best, tot / reps);
    };
    std::printf("%-8s | %-26s | %-26s | %-34s | %-34s | %s\n", "n", "FP32 best/mean ms", "int8 best/mean ms", "FP32 vs FP64(deq): maxabs rel", "int8 vs FP64(deq): maxabs rel", "int8 vs FP32 maxabs rel | hashes");
    for (g_vm = 0; g_vm < 2; ++g_vm) {
    std::printf("---- %s ----\n", g_vm ? "int8 K + int8 V" : "int8 K + FP32 V");
    for (int n : {100, 4097, max_n}) {
        if (n > max_n) continue;
        std::vector<double> refd, refo;
        reference(q, qfull, Kh, Vh, n, refd);
        std::vector<float> Ksub(K.begin(), K.begin() + static_cast<size_t>(n) * KVW), Vsub(V.begin(), V.begin() + static_cast<size_t>(n) * KVW);
        reference(q, qfull, Ksub, Vsub, n, refo);
        std::vector<float> o32(NH * HD), o8(NH * HD);
        run32(n); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(o32.data(), dout32, o32.size() * 4, cudaMemcpyDeviceToHost));
        run8(n); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(o8.data(), dout8, o8.size() * 4, cudaMemcpyDeviceToHost));
        const auto t32 = timeit([&] { run32(n); });
        const auto t8 = timeit([&] { run8(n); });
        const Err e32 = compare(o32, refd), e8 = compare(o8, refd), e88 = compare32(o8, o32);
        const Err e32o = compare(o32, refo), e8o = compare(o8, refo);
        std::printf("%-8d | %8.3f / %8.3f      | %8.3f / %8.3f      | %.3e %.3e              | %.3e %.3e              | %.3e %.3e | %016llx %016llx\n", n,
                    t32.first, t32.second, t8.first, t8.second, e32.max_abs, e32.rel, e8.max_abs, e8.rel, e88.max_abs, e88.rel,
                    static_cast<unsigned long long>(fnv(o32)), static_cast<unsigned long long>(fnv(o8)));
        std::printf("         vs FP64 on the ORIGINAL float K/V (iso3's own error included): FP32 kernel rel %.3e, int8 kernel rel %.3e (rms ref %.3e)\n", e32o.rel, e8o.rel, e8o.rms_ref);
    }
    }
    return 0;
}
