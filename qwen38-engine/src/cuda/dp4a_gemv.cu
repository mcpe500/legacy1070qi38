#include "dp4a_gemv.cuh"

#include "f16bits.hpp"
#include "gemv.cuh"  // GemvType ids

#define QT_BEGIN(type, name, size) static __device__ const type name[size] = {
#define QT_END() };
#include "tables.inc"
#undef QT_BEGIN
#undef QT_END

namespace {

constexpr int WARPS = 4;   // output rows per thread block

__device__ __forceinline__ uint32_t rd16(const uint8_t * p) { return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8); }
__device__ __forceinline__ uint32_t rd32(const uint8_t * p) { return rd16(p) | (rd16(p + 2) << 16); }  // 2-byte alignment is enough
__device__ __forceinline__ float half_at(const uint8_t * p) { return f16_to_f32(rd16(p)); }

// 128-bit and 64-bit vector loads (the caller guarantees 16-byte / 8-byte alignment: Q4_K and Q5_K blocks are 144 / 176 bytes,
// IQ4_XS blocks 136 bytes, and every tensor starts on a 256-byte boundary of the VRAM arena).
__device__ __forceinline__ uint4 ld128(const uint8_t * p) { return *reinterpret_cast<const uint4 *>(p); }
__device__ __forceinline__ uint2 ld64(const uint8_t * p) { return *reinterpret_cast<const uint2 *>(p); }

// PTX lop3.b32: any bitwise function of three inputs in one instruction (immLut computed for a = 0xF0, b = 0xCC, c = 0xAA).
__device__ __forceinline__ uint32_t lop3_and(uint32_t a, uint32_t m) {            // a & m                     (0xF0 & 0xCC = 0xC0)
    uint32_t r;
    asm("lop3.b32 %0, %1, %2, 0, 0xC0;" : "=r"(r) : "r"(a), "r"(m));
    return r;
}
__device__ __forceinline__ uint32_t lop3_and_or(uint32_t a, uint32_t m, uint32_t c) {  // (a & m) | c    ((0xF0 & 0xCC) | 0xAA = 0xEA)
    uint32_t r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(m), "r"(c));
    return r;
}

__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t * q, int & d, int & m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// kvalues_iq4nl as four little-endian words (entries 0-3, 4-7, 8-11, 12-15). Checked against libggml by gemv_parity.
constexpr uint32_t KV0 = 0xBFAD9881u, KV1 = 0xF6EADDCFu, KV2 = 0x26190D01u, KV3 = 0x71594535u;

// Looks up 8 four-bit indices packed in q4. Returns {values of the 4 low nibbles, values of the 4 high nibbles} as packed int8x4.
// __byte_perm only selects with 3 bits, so the fourth bit picks between the lookups of the low and the high half of the table.
__device__ __forceinline__ int2 table16(uint32_t q4) {
    const uint32_t sel_hi = 0x32103210u | ((q4 & 0x88888888u) >> 1);
    uint32_t tmp[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t shift = 16u * i;
        const uint32_t idx = (q4 >> shift) & 0x7777u;
        const uint32_t low = __byte_perm(KV0, KV1, idx);
        const uint32_t high = __byte_perm(KV2, KV3, idx);
        tmp[i] = __byte_perm(low, high, sel_hi >> shift);
    }
    return make_int2(static_cast<int>(__byte_perm(tmp[0], tmp[1], 0x6420)), static_cast<int>(__byte_perm(tmp[0], tmp[1], 0x7531)));
}

// 0xFF in the byte i of the result when bit i of the 4-bit value b is set.
__device__ __forceinline__ uint32_t sign_mask4(uint32_t b) { return __vcmpne4((b * 0x01010101u) & 0x08040201u, 0u); }

struct UnitW {
    uint32_t w[8];     // 32 weights as int8, positions 0..31 in order
    float s0, s1;      // scale of positions 0..15 and 16..31
    float mn;          // minimum coefficient (multiplies the activation block sum)
};

template <int T> struct Prep;

template <> struct Prep<GT_Q4_K> {
    static constexpr int BLOCK_BYTES = 144, UPB = 8;
    static __device__ __forceinline__ void run(const uint8_t * blk, int s, UnitW & u) {
        const float d = half_at(blk), dmin = half_at(blk + 2);
        int sc, m;
        get_scale_min_k4(s, blk + 4, sc, m);
        u.s0 = u.s1 = d * static_cast<float>(sc);
        u.mn = dmin * static_cast<float>(m);
        const uint8_t * q = blk + 16 + 32 * (s >> 1);
        const int sh = (s & 1) * 4;
        const uint4 a = ld128(q), b = ld128(q + 16);
        const uint32_t qq[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
        for (int g = 0; g < 8; ++g) u.w[g] = lop3_and(qq[g] >> sh, 0x0F0F0F0Fu);
    }
};

template <> struct Prep<GT_Q5_K> {
    static constexpr int BLOCK_BYTES = 176, UPB = 8;
    static __device__ __forceinline__ void run(const uint8_t * blk, int s, UnitW & u) {
        const float d = half_at(blk), dmin = half_at(blk + 2);
        int sc, m;
        get_scale_min_k4(s, blk + 4, sc, m);
        u.s0 = u.s1 = d * static_cast<float>(sc);
        u.mn = dmin * static_cast<float>(m);
        const uint8_t * qh = blk + 16;
        const uint8_t * q = blk + 48 + 32 * (s >> 1);
        const int sh = (s & 1) * 4;
        const uint4 a = ld128(q), b = ld128(q + 16), ha = ld128(qh), hb = ld128(qh + 16);
        const uint32_t qq[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        const uint32_t hh[8] = {ha.x, ha.y, ha.z, ha.w, hb.x, hb.y, hb.z, hb.w};
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const uint32_t hi = ((hh[g] >> s) & 0x01010101u) << 4;
            u.w[g] = lop3_and_or(qq[g] >> sh, 0x0F0F0F0Fu, hi);   // (low nibble) | (fifth bit)
        }
    }
};

template <> struct Prep<GT_Q6_K> {
    static constexpr int BLOCK_BYTES = 210, UPB = 8;
    static __device__ __forceinline__ void run(const uint8_t * blk, int s, UnitW & u) {
        const float d = half_at(blk + 208);
        const int n = s >> 2, qd = s & 3;
        const uint8_t * ql = blk + 64 * n + 32 * (qd & 1);
        const uint8_t * qh = blk + 128 + 32 * n;
        const int8_t * sc = reinterpret_cast<const int8_t *>(blk + 192) + 8 * n + 2 * qd;
        u.s0 = d * static_cast<float>(sc[0]);
        u.s1 = d * static_cast<float>(sc[1]);
        u.mn = 0.0f;
        const int nsh = qd >= 2 ? 4 : 0;
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const uint32_t lo = (rd32(ql + 4 * g) >> nsh) & 0x0F0F0F0Fu;
            const uint32_t hi = ((rd32(qh + 4 * g) >> (2 * qd)) & 0x03030303u) << 4;
            u.w[g] = __vsub4(lo | hi, 0x20202020u);   // q - 32 per byte
        }
    }
};

template <> struct Prep<GT_Q3_K> {
    static constexpr int BLOCK_BYTES = 110, UPB = 8;
    static __device__ __forceinline__ void run(const uint8_t * blk, int s, UnitW & u) {
        const float d_all = half_at(blk + 108);
        const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
        uint32_t a0 = rd32(blk + 96), a1 = rd32(blk + 100);
        const uint32_t tmp = rd32(blk + 104);
        const uint32_t a2 = ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        const uint32_t a3 = ((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        a0 = (a0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
        a1 = (a1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int is = 2 * s;
        auto scale_at = [&](int i) {
            const uint32_t word = (i < 4) ? a0 : (i < 8) ? a1 : (i < 12) ? a2 : a3;
            return static_cast<int>((word >> (8 * (i & 3))) & 0xFF) - 32;
        };
        u.s0 = d_all * static_cast<float>(scale_at(is));
        u.s1 = d_all * static_cast<float>(scale_at(is + 1));
        u.mn = 0.0f;
        const int n = s >> 2, j = s & 3, shift = 2 * j;
        const uint32_t mm = (1u << (4 * n + j)) * 0x01010101u;
        const uint8_t * q = blk + 32 + 32 * n;
#pragma unroll
        for (int g = 0; g < 8; ++g) {            // g 0..3: positions 0..15 (hmask 0..15), g 4..7: positions 16..31
            const uint32_t qv = rd32(q + 4 * g);
            const uint32_t hv = rd32(blk + 4 * g);
            const uint32_t low2 = (qv >> shift) & 0x03030303u;
            const uint32_t sub = __vcmpeq4(hv & mm, 0u) & 0x04040404u;   // 4 where the high bit is clear
            u.w[g] = __vsub4(low2, sub);
        }
    }
};

template <> struct Prep<GT_IQ4_XS> {
    static constexpr int BLOCK_BYTES = 136, UPB = 8;
    static __device__ __forceinline__ void run(const uint8_t * blk, int ib, UnitW & u) {
        const float d = half_at(blk);
        const uint32_t sh = rd16(blk + 2);
        const int ls = ((blk[4 + (ib >> 1)] >> (4 * (ib & 1))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
        u.s0 = u.s1 = d * static_cast<float>(ls - 32);
        u.mn = 0.0f;
        const uint8_t * qs = blk + 8 + 16 * ib;
        const uint2 a = ld64(qs), b = ld64(qs + 8);
        const uint32_t qq[4] = {a.x, a.y, b.x, b.y};
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            const int2 t = table16(qq[g]);
            u.w[g] = static_cast<uint32_t>(t.x);        // low nibbles -> positions 4g .. 4g+3
            u.w[4 + g] = static_cast<uint32_t>(t.y);    // high nibbles -> positions 16+4g ..
        }
    }
};

template <> struct Prep<GT_IQ4_NL> {
    static constexpr int BLOCK_BYTES = 18, UPB = 1;
    static __device__ __forceinline__ void run(const uint8_t * blk, int, UnitW & u) {
        u.s0 = u.s1 = half_at(blk);
        u.mn = 0.0f;
        const uint8_t * qs = blk + 2;
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            const int2 t = table16(rd32(qs + 4 * g));
            u.w[g] = static_cast<uint32_t>(t.x);
            u.w[4 + g] = static_cast<uint32_t>(t.y);
        }
    }
};

template <> struct Prep<GT_IQ3_S> {
    static constexpr int BLOCK_BYTES = 110, UPB = 8;
    static __device__ __forceinline__ void run(const uint8_t * blk, int ib32, UnitW & u) {
        const float d = half_at(blk);
        const uint8_t * qs = blk + 2 + 8 * ib32;
        const uint32_t qh = blk[66 + ib32];
        const uint8_t * sg = blk + 74 + 4 * ib32;
        u.s0 = u.s1 = d * static_cast<float>(1 + 2 * ((blk[106 + (ib32 >> 1)] >> (4 * (ib32 & 1))) & 0xF));
        u.mn = 0.0f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t g1 = iq3s_grid[qs[2 * l + 0] | ((qh << (8 - 2 * l)) & 256)];
            const uint32_t g2 = iq3s_grid[qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256)];
            const uint32_t m1 = sign_mask4(sg[l] & 0xF), m2 = sign_mask4(sg[l] >> 4);
            u.w[2 * l] = __vsub4(g1 ^ m1, m1);          // negate the bytes whose sign bit is set
            u.w[2 * l + 1] = __vsub4(g2 ^ m2, m2);
        }
    }
};

// ------------------------------------------------------------------------------------------- activation quantization

__global__ void actq8_kernel(const float * __restrict__ x, int64_t in, int8_t * __restrict__ qx, float * __restrict__ dx,
                             float * __restrict__ sx) {
    const int64_t t = blockIdx.y, b = blockIdx.x;
    const int lane = threadIdx.x;
    const int64_t nb32 = in / 32;
    const float v = x[t * in + b * 32 + lane];
    float amax = fabsf(v);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    const float d = amax / 127.0f;
    const float inv = d > 0.0f ? 1.0f / d : 0.0f;
    int q = __float2int_rn(v * inv);
    q = max(-127, min(127, q));
    qx[t * in + b * 32 + lane] = static_cast<int8_t>(q);
    int qsum = q;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) qsum += __shfl_xor_sync(0xffffffffu, qsum, off);
    if (lane == 0) {
        dx[t * nb32 + b] = d;
        sx[t * nb32 + b] = d * static_cast<float>(qsum);
    }
}

// ---------------------------------------------------------------------------------------------------------- GEMV

// TT = tokens per thread block (accumulators per thread). Each token has its own accumulator and the same summation order,
// so the result of a token does not depend on TT or on the batch size. Small tiles keep decode (T = 1) fast, large tiles
// amortize the weight unpacking in prefill.
template <int TYPE, int TT>
__global__ void gemv_dp4a_kernel(const uint8_t * __restrict__ W, int64_t in, int64_t out, const int8_t * __restrict__ qx,
                                 const float * __restrict__ dx, const float * __restrict__ sx, int T, float * __restrict__ y) {
    using P = Prep<TYPE>;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t o = static_cast<int64_t>(blockIdx.x) * WARPS + warp;
    if (o >= out) return;
    const int t0 = blockIdx.y * TT;
    const int nt = min(TT, T - t0);
    const int64_t nblk = in / (P::UPB * 32), nb32 = in / 32;
    const uint8_t * row = W + o * nblk * P::BLOCK_BYTES;
    const int64_t nunits = nblk * P::UPB;

    float acc[TT];
#pragma unroll
    for (int t = 0; t < TT; ++t) acc[t] = 0.0f;

    for (int64_t u = lane; u < nunits; u += 32) {
        UnitW uw;
        P::run(row + (u / P::UPB) * P::BLOCK_BYTES, static_cast<int>(u % P::UPB), uw);
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            if (t < nt) {
                const int4 * x16 = reinterpret_cast<const int4 *>(qx + static_cast<int64_t>(t0 + t) * in + u * 32);   // 32-byte aligned
                const int4 xa = x16[0], xb = x16[1];
                int i0 = 0, i1 = 0;
                i0 = __dp4a(static_cast<int>(uw.w[0]), xa.x, i0);
                i0 = __dp4a(static_cast<int>(uw.w[1]), xa.y, i0);
                i0 = __dp4a(static_cast<int>(uw.w[2]), xa.z, i0);
                i0 = __dp4a(static_cast<int>(uw.w[3]), xa.w, i0);
                i1 = __dp4a(static_cast<int>(uw.w[4]), xb.x, i1);
                i1 = __dp4a(static_cast<int>(uw.w[5]), xb.y, i1);
                i1 = __dp4a(static_cast<int>(uw.w[6]), xb.z, i1);
                i1 = __dp4a(static_cast<int>(uw.w[7]), xb.w, i1);
                const float dxv = dx[static_cast<int64_t>(t0 + t) * nb32 + u];
                const float sxv = sx[static_cast<int64_t>(t0 + t) * nb32 + u];
                acc[t] += dxv * (uw.s0 * static_cast<float>(i0) + uw.s1 * static_cast<float>(i1)) - uw.mn * sxv;
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

template <int TYPE, int TT>
cudaError_t launch_tt(const uint8_t * W, int64_t in, int64_t out, const ActQ8 & a, int64_t T, float * y, cudaStream_t s) {
    dim3 grid(static_cast<unsigned>((out + WARPS - 1) / WARPS), static_cast<unsigned>((T + TT - 1) / TT));
    gemv_dp4a_kernel<TYPE, TT><<<grid, WARPS * 32, 0, s>>>(W, in, out, a.qx, a.dx, a.sx, static_cast<int>(T), y);
    return cudaGetLastError();
}

template <int TYPE>
cudaError_t launch(const uint8_t * W, int64_t in, int64_t out, const ActQ8 & a, int64_t T, float * y, cudaStream_t s) {
    if (T <= 1) return launch_tt<TYPE, 1>(W, in, out, a, T, y, s);
    if (T <= 4) return launch_tt<TYPE, 4>(W, in, out, a, T, y, s);
    if (T <= 8) return launch_tt<TYPE, 8>(W, in, out, a, T, y, s);
    return launch_tt<TYPE, 16>(W, in, out, a, T, y, s);   // 32 and 64 were measured slower (509-token prefill: 7.3 s, 12.8 s, 17.5 s)
}

}  // namespace

cudaError_t actq8_alloc(ActQ8 & a, int64_t T_max, int64_t in_max) {
    cudaError_t e = cudaMalloc(&a.qx, static_cast<size_t>(T_max * in_max));
    if (e != cudaSuccess) return e;
    e = cudaMalloc(&a.dx, sizeof(float) * static_cast<size_t>(T_max * in_max / 32));
    if (e != cudaSuccess) return e;
    e = cudaMalloc(&a.sx, sizeof(float) * static_cast<size_t>(T_max * in_max / 32));
    if (e != cudaSuccess) return e;
    a.T_max = T_max;
    a.in_max = in_max;
    return cudaSuccess;
}

void actq8_free(ActQ8 & a) {
    if (a.qx) cudaFree(a.qx);
    if (a.dx) cudaFree(a.dx);
    if (a.sx) cudaFree(a.sx);
    a = ActQ8{};
}

cudaError_t actq8_quantize(const float * x, int64_t in, int64_t T, const ActQ8 & a, cudaStream_t s) {
    if (in % 32 != 0 || T > a.T_max || in > a.in_max) return cudaErrorInvalidValue;
    actq8_kernel<<<dim3(static_cast<unsigned>(in / 32), static_cast<unsigned>(T)), 32, 0, s>>>(x, in, a.qx, a.dx, a.sx);
    return cudaGetLastError();
}

bool gemv_dp4a_supported(int type) {
    switch (type) {
        case GT_Q4_K: case GT_Q5_K: case GT_Q6_K: case GT_Q3_K: case GT_IQ4_NL: case GT_IQ3_S: case GT_IQ4_XS: return true;
        default: return false;
    }
}

cudaError_t gemv_dp4a(int type, const uint8_t * W, int64_t in, int64_t out, const ActQ8 & a, int64_t T, float * y,
                      cudaStream_t s) {
    if (T > a.T_max || in > a.in_max) return cudaErrorInvalidValue;
    switch (type) {
        case GT_Q4_K: return launch<GT_Q4_K>(W, in, out, a, T, y, s);
        case GT_Q5_K: return launch<GT_Q5_K>(W, in, out, a, T, y, s);
        case GT_Q6_K: return launch<GT_Q6_K>(W, in, out, a, T, y, s);
        case GT_Q3_K: return launch<GT_Q3_K>(W, in, out, a, T, y, s);
        case GT_IQ4_XS: return launch<GT_IQ4_XS>(W, in, out, a, T, y, s);
        case GT_IQ4_NL: return launch<GT_IQ4_NL>(W, in, out, a, T, y, s);
        case GT_IQ3_S: return launch<GT_IQ3_S>(W, in, out, a, T, y, s);
        default: return cudaErrorInvalidValue;
    }
}
