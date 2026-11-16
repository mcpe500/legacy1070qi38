#include "gemv.cuh"

#include "f16bits.hpp"

#define QT_BEGIN(type, name, size) static __device__ const type name[size] = {
#define QT_END() };
#include "tables.inc"
#undef QT_BEGIN
#undef QT_END

namespace {

constexpr int TT = 8;          // tokens per thread block (weights are read once per tile)
constexpr int WARPS = 4;       // output rows per thread block

__device__ __forceinline__ uint32_t rd16(const uint8_t * p) { return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8); }
__device__ __forceinline__ uint32_t rd32(const uint8_t * p) { return rd16(p) | (rd16(p + 2) << 16); }
__device__ __forceinline__ float half_at(const uint8_t * p) { return f16_to_f32(rd16(p)); }

__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t * q, int & d, int & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Each deq32 writes the 32 FP32 weights of unit `u` of one row. A unit is 32 consecutive values of the row, so it lines
// up with 32 activations. For the 256-value types unit u lives in super-block u / 8, sub-block u % 8.

template <int T> struct Deq;

template <> struct Deq<GT_Q4_K> {  // block 144 bytes: d, dmin, scales[12], qs[128]
    static constexpr int BLOCK_BYTES = 144, UPB = 8;
    static __device__ __forceinline__ void run(const uint8_t * blk, int s, float * w) {
        const float d = half_at(blk), dmin = half_at(blk + 2);
        int sc, m;
        get_scale_min_k4(s, blk + 4, sc, m);
        const float d1 = d * sc, m1 = dmin * m;
        const uint8_t * q = blk + 16 + 32 * (s >> 1);
        const int sh = (s & 1) * 4;
#pragma unroll
        for (int l = 0; l < 32; ++l) w[l] = d1 * static_cast<float>((q[l] >> sh) & 0xF) - m1;
    }
};

template <> struct Deq<GT_Q5_K> {  // block 176 bytes: d, dmin, scales[12], qh[32], qs[128]
    static constexpr int BLOCK_BYTES = 176, UPB = 8;
    static __device__ __forceinline__ void run(const uint8_t * blk, int s, float * w) {
        const float d = half_at(blk), dmin = half_at(blk + 2);
        int sc, m;
        get_scale_min_k4(s, blk + 4, sc, m);
        const float d1 = d * sc, m1 = dmin * m;
        const uint8_t * qh = blk + 16;
        const uint8_t * q = blk + 48 + 32 * (s >> 1);
        const int sh = (s & 1) * 4;
#pragma unroll
        for (int l = 0; l < 32; ++l) {
            const int v = ((q[l] >> sh) & 0xF) + (((qh[l] >> s) & 1) << 4);
            w[l] = d1 * static_cast<float>(v) - m1;
        }
    }
};

template <> struct Deq<GT_Q6_K> {  // block 210 bytes: ql[128], qh[64], scales[16] (int8), d
    static constexpr int BLOCK_BYTES = 210, UPB = 8;
    static __device__ __forceinline__ void run(const uint8_t * blk, int s, float * w) {
        const float d = half_at(blk + 208);
        const int n = s >> 2, qd = s & 3;
        const uint8_t * ql = blk + 64 * n + 32 * (qd & 1);
        const uint8_t * qh = blk + 128 + 32 * n;
        const int8_t * sc = reinterpret_cast<const int8_t *>(blk + 192) + 8 * n + 2 * qd;
        const int nsh = qd >= 2 ? 4 : 0;
#pragma unroll
        for (int l = 0; l < 32; ++l) {
            const int q = ((ql[l] >> nsh) & 0xF) | (((qh[l] >> (2 * qd)) & 3) << 4);
            w[l] = d * static_cast<float>(sc[l >> 4]) * static_cast<float>(q - 32);
        }
    }
};

template <> struct Deq<GT_Q3_K> {  // block 110 bytes: hmask[32], qs[64], scales[12], d
    static constexpr int BLOCK_BYTES = 110, UPB = 8;
    static __device__ __forceinline__ void run(const uint8_t * blk, int s, float * w) {
        const float d_all = half_at(blk + 108);
        const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
        uint32_t a0 = rd32(blk + 96), a1 = rd32(blk + 100), tmp = rd32(blk + 104);
        const uint32_t a2 = ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        const uint32_t a3 = ((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        a0 = (a0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
        a1 = (a1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int is = 2 * s;  // scale index of the first 16-value group of this unit
        const uint32_t word = (is < 4) ? a0 : (is < 8) ? a1 : (is < 12) ? a2 : a3;
        const int sc0 = static_cast<int>((word >> (8 * (is & 3))) & 0xFF) - 32;
        const uint32_t word1 = ((is + 1) < 4) ? a0 : ((is + 1) < 8) ? a1 : ((is + 1) < 12) ? a2 : a3;
        const int sc1 = static_cast<int>((word1 >> (8 * ((is + 1) & 3))) & 0xFF) - 32;
        const int n = s >> 2, j = s & 3;
        const uint8_t * hm = blk;
        const uint8_t * q = blk + 32 + 32 * n;
        const int shift = 2 * j;
        const uint32_t m = 1u << (4 * n + j);
        const float dl0 = d_all * static_cast<float>(sc0), dl1 = d_all * static_cast<float>(sc1);
#pragma unroll
        for (int l = 0; l < 16; ++l) {
            w[l] = dl0 * static_cast<float>(static_cast<int>((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
            w[16 + l] = dl1 * static_cast<float>(static_cast<int>((q[16 + l] >> shift) & 3) - ((hm[16 + l] & m) ? 0 : 4));
        }
    }
};

template <> struct Deq<GT_IQ4_XS> {  // block 136 bytes: d, scales_h (u16), scales_l[4], qs[128]
    static constexpr int BLOCK_BYTES = 136, UPB = 8;
    static __device__ __forceinline__ void run(const uint8_t * blk, int ib, float * w) {
        const float d = half_at(blk);
        const uint32_t sh = rd16(blk + 2);
        const int ls = ((blk[4 + (ib >> 1)] >> (4 * (ib & 1))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
        const float dl = d * static_cast<float>(ls - 32);
        const uint8_t * qs = blk + 8 + 16 * ib;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            w[j] = dl * static_cast<float>(kvalues_iq4nl[qs[j] & 0xF]);
            w[16 + j] = dl * static_cast<float>(kvalues_iq4nl[qs[j] >> 4]);
        }
    }
};

template <> struct Deq<GT_IQ4_NL> {  // block 18 bytes: d, qs[16]; one block is one unit
    static constexpr int BLOCK_BYTES = 18, UPB = 1;
    static __device__ __forceinline__ void run(const uint8_t * blk, int, float * w) {
        const float d = half_at(blk);
        const uint8_t * qs = blk + 2;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            w[j] = d * static_cast<float>(kvalues_iq4nl[qs[j] & 0xF]);
            w[16 + j] = d * static_cast<float>(kvalues_iq4nl[qs[j] >> 4]);
        }
    }
};

template <> struct Deq<GT_IQ3_S> {  // block 110 bytes: d, qs[64], qh[8], signs[32], scales[4]
    static constexpr int BLOCK_BYTES = 110, UPB = 8;
    static __device__ __forceinline__ void run(const uint8_t * blk, int ib32, float * w) {
        const float d = half_at(blk);
        const uint8_t * qs = blk + 2 + 8 * ib32;
        const uint32_t qh = blk[66 + ib32];
        const uint8_t * sg = blk + 74 + 4 * ib32;
        const float db = d * static_cast<float>(1 + 2 * ((blk[106 + (ib32 >> 1)] >> (4 * (ib32 & 1))) & 0xF));
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t g1 = iq3s_grid[qs[2 * l + 0] | ((qh << (8 - 2 * l)) & 256)];
            const uint32_t g2 = iq3s_grid[qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256)];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                w[8 * l + j] = db * static_cast<float>((g1 >> (8 * j)) & 0xFF) * ((sg[l] & (1 << j)) ? -1.0f : 1.0f);
                w[8 * l + 4 + j] = db * static_cast<float>((g2 >> (8 * j)) & 0xFF) * ((sg[l] & (1 << (j + 4))) ? -1.0f : 1.0f);
            }
        }
    }
};

template <> struct Deq<GT_Q8_0> {  // block 34 bytes: d, qs[32] (reference path for tests; production uses dp4a)
    static constexpr int BLOCK_BYTES = 34, UPB = 1;
    static __device__ __forceinline__ void run(const uint8_t * blk, int, float * w) {
        const float d = half_at(blk);
#pragma unroll
        for (int l = 0; l < 32; ++l) w[l] = d * static_cast<float>(static_cast<int8_t>(blk[2 + l]));
    }
};

template <int TYPE>
__global__ void gemv_fp32_kernel(const uint8_t * __restrict__ W, int64_t in, int64_t out, const float * __restrict__ x,
                                 int T, float * __restrict__ y) {
    using D = Deq<TYPE>;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t o = static_cast<int64_t>(blockIdx.x) * WARPS + warp;
    if (o >= out) return;
    const int t0 = blockIdx.y * TT;
    const int nt = min(TT, T - t0);
    const int64_t nblk = in / (D::UPB * 32);
    const uint8_t * row = W + o * nblk * D::BLOCK_BYTES;
    const int64_t nunits = nblk * D::UPB;

    float acc[TT];
#pragma unroll
    for (int t = 0; t < TT; ++t) acc[t] = 0.0f;

    for (int64_t u = lane; u < nunits; u += 32) {
        float w[32];
        D::run(row + (u / D::UPB) * D::BLOCK_BYTES, static_cast<int>(u % D::UPB), w);
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            if (t < nt) {
                const float4 * xp = reinterpret_cast<const float4 *>(x + static_cast<int64_t>(t0 + t) * in + u * 32);
                float a = 0.0f;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const float4 v = xp[k];
                    a = fmaf(w[4 * k + 0], v.x, a);
                    a = fmaf(w[4 * k + 1], v.y, a);
                    a = fmaf(w[4 * k + 2], v.z, a);
                    a = fmaf(w[4 * k + 3], v.w, a);
                }
                acc[t] += a;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < TT; ++t) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], off);
    }
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            if (t < nt) y[static_cast<int64_t>(t0 + t) * out + o] = acc[t];
        }
    }
}

template <int TYPE>
cudaError_t launch(const uint8_t * W, int64_t in, int64_t out, const float * x, int64_t T, float * y, cudaStream_t s) {
    dim3 grid(static_cast<unsigned>((out + WARPS - 1) / WARPS), static_cast<unsigned>((T + TT - 1) / TT));
    gemv_fp32_kernel<TYPE><<<grid, WARPS * 32, 0, s>>>(W, in, out, x, static_cast<int>(T), y);
    return cudaGetLastError();
}

}  // namespace

bool gemv_fp32_supported(int type) { return gemv_row_bytes(type, 256) != 0; }

int64_t gemv_row_bytes(int type, int64_t in) {
    switch (type) {
        case GT_Q4_K: return in / 256 * 144;
        case GT_Q5_K: return in / 256 * 176;
        case GT_Q6_K: return in / 256 * 210;
        case GT_Q3_K: return in / 256 * 110;
        case GT_IQ4_XS: return in / 256 * 136;
        case GT_IQ4_NL: return in / 32 * 18;
        case GT_IQ3_S: return in / 256 * 110;
        case GT_Q8_0: return in / 32 * 34;
        default: return 0;
    }
}

cudaError_t gemv_fp32(int type, const uint8_t * W, int64_t in, int64_t out, const float * x, int64_t T, float * y,
                      cudaStream_t s) {
    switch (type) {
        case GT_Q4_K: return launch<GT_Q4_K>(W, in, out, x, T, y, s);
        case GT_Q5_K: return launch<GT_Q5_K>(W, in, out, x, T, y, s);
        case GT_Q6_K: return launch<GT_Q6_K>(W, in, out, x, T, y, s);
        case GT_Q3_K: return launch<GT_Q3_K>(W, in, out, x, T, y, s);
        case GT_IQ4_XS: return launch<GT_IQ4_XS>(W, in, out, x, T, y, s);
        case GT_IQ4_NL: return launch<GT_IQ4_NL>(W, in, out, x, T, y, s);
        case GT_IQ3_S: return launch<GT_IQ3_S>(W, in, out, x, T, y, s);
        case GT_Q8_0: return launch<GT_Q8_0>(W, in, out, x, T, y, s);
        default: return cudaErrorInvalidValue;
    }
}
