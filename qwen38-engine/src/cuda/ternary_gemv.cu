#include "ternary_gemv.cuh"

#include <cstdlib>

#include "f16bits.hpp"
#include "ternary_spec.hpp"

namespace {

constexpr int WARPS = 4;   // output rows per thread block, one warp per row

__device__ __forceinline__ uint32_t rd16(const uint8_t * p) { return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8); }
__device__ __forceinline__ uint32_t rd32h(const uint8_t * p) { return rd16(p) | (rd16(p + 2) << 16); }   // 2-byte aligned

__device__ __forceinline__ uint32_t lop3_and(uint32_t a, uint32_t m) {   // a & m  (immLut 0xC0)
    uint32_t r;
    asm("lop3.b32 %0, %1, %2, 0, 0xC0;" : "=r"(r) : "r"(a), "r"(m));
    return r;
}

// 16 two-bit codes (one 32-bit word, element j at bits 2j) -> four words E0..E3, lane k of Em = code of element 4k + m.
__device__ __forceinline__ void unpack16(uint32_t q, uint32_t & e0, uint32_t & e1, uint32_t & e2, uint32_t & e3) {
    const uint32_t a = lop3_and(q, 0x33333333u);         // per byte: element 4k (bits 0-1), element 4k+2 (bits 4-5)
    const uint32_t b = lop3_and(q >> 2, 0x33333333u);    // per byte: element 4k+1, element 4k+3
    e0 = lop3_and(a, 0x03030303u);
    e2 = lop3_and(a >> 4, 0x03030303u);
    e1 = lop3_and(b, 0x03030303u);
    e3 = lop3_and(b >> 4, 0x03030303u);
}

// 16 int8 activations x0..x15 in four words -> four words, lane k of Am = x_{4k+m} (4x4 byte transpose with prmt.b32).
__device__ __forceinline__ void gather16(int4 x, int & a0, int & a1, int & a2, int & a3) {
    const uint32_t X0 = static_cast<uint32_t>(x.x), X1 = static_cast<uint32_t>(x.y), X2 = static_cast<uint32_t>(x.z),
                   X3 = static_cast<uint32_t>(x.w);
    const uint32_t P = __byte_perm(X0, X1, 0x5140), Q = __byte_perm(X2, X3, 0x5140);    // (x0,x4,x1,x5) (x8,x12,x9,x13)
    const uint32_t R = __byte_perm(X0, X1, 0x7362), S = __byte_perm(X2, X3, 0x7362);    // (x2,x6,x3,x7) (x10,x14,x11,x15)
    a0 = static_cast<int>(__byte_perm(P, Q, 0x5410));
    a1 = static_cast<int>(__byte_perm(P, Q, 0x7632));
    a2 = static_cast<int>(__byte_perm(R, S, 0x5410));
    a3 = static_cast<int>(__byte_perm(R, S, 0x7632));
}

template <int TT>
__global__ void gemv_q2_0_kernel(const uint8_t * __restrict__ W, int64_t in, int64_t out, const int8_t * __restrict__ qx,
                                 const float * __restrict__ dx, const float * __restrict__ sx, int T, float * __restrict__ y) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t o = static_cast<int64_t>(blockIdx.x) * WARPS + warp;
    if (o >= out) return;
    const int t0 = blockIdx.y * TT;
    const int nt = min(TT, T - t0);
    const int64_t nblk = in / 64, nb32 = in / 32;
    const uint8_t * row = W + o * nblk * kQ2_0BlockBytes;
    float acc[TT];
#pragma unroll
    for (int t = 0; t < TT; ++t) acc[t] = 0.0f;

    for (int64_t u = lane; u < nblk; u += 32) {
        const uint8_t * blk = row + u * kQ2_0BlockBytes;
        const float d = f16_to_f32(rd16(blk));
        uint32_t e[4][4];   // [group of 16 weights][m]
#pragma unroll
        for (int g = 0; g < 4; ++g) unpack16(rd32h(blk + 2 + 4 * g), e[g][0], e[g][1], e[g][2], e[g][3]);
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            if (t < nt) {
                const int64_t tt = t0 + t;
                const int4 * xq = reinterpret_cast<const int4 *>(qx + tt * in + u * 64);   // 64-byte aligned (in % 64 == 0)
                float s = 0.0f;
#pragma unroll
                for (int h = 0; h < 2; ++h) {   // two Q8_1 activation blocks of 32
                    int idot = 0;
#pragma unroll
                    for (int g2 = 0; g2 < 2; ++g2) {
                        const int g = 2 * h + g2;
                        int a0, a1, a2, a3;
                        gather16(xq[g], a0, a1, a2, a3);
                        idot = __dp4a(static_cast<int>(e[g][0]), a0, idot);
                        idot = __dp4a(static_cast<int>(e[g][1]), a1, idot);
                        idot = __dp4a(static_cast<int>(e[g][2]), a2, idot);
                        idot = __dp4a(static_cast<int>(e[g][3]), a3, idot);
                    }
                    const int64_t b = tt * nb32 + 2 * u + h;
                    s += dx[b] * static_cast<float>(idot) - sx[b];
                }
                acc[t] += d * s;
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
        for (int t = 0; t < TT; ++t)
            if (t < nt) y[static_cast<int64_t>(t0 + t) * out + o] = acc[t];
    }
}

template <int TT>
cudaError_t launch(const uint8_t * W, int64_t in, int64_t out, const ActQ8 & a, int64_t T, float * y, cudaStream_t s) {
    const dim3 grid(static_cast<unsigned>((out + WARPS - 1) / WARPS), static_cast<unsigned>((T + TT - 1) / TT));
    gemv_q2_0_kernel<TT><<<grid, WARPS * 32, 0, s>>>(W, in, out, a.qx, a.dx, a.sx, static_cast<int>(T), y);
    return cudaGetLastError();
}

// One block of 256 threads rotates one 1024-value block: 4 values per thread, in-place butterflies in shared memory.
__global__ void rotate_kernel(const float * __restrict__ x, const float * __restrict__ signs, float * __restrict__ y, int width) {
    __shared__ float v[1024];
    const int t = blockIdx.y, blk = blockIdx.x, tid = threadIdx.x;
    const int64_t base = static_cast<int64_t>(t) * width + static_cast<int64_t>(blk) * 1024;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int k = tid + 256 * i;
        v[k] = x[base + k] * signs[blk * 1024 + k];
    }
    __syncthreads();
    for (int h = 1; h < 1024; h <<= 1) {   // 10 stages; each thread owns 2 butterflies per stage
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int p = tid + 256 * i;                  // butterfly index 0..511
            const int lo = (p / h) * 2 * h + (p % h), hi = lo + h;
            const float a = v[lo], b = v[hi];
            v[lo] = a + b;
            v[hi] = a - b;
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int k = tid + 256 * i;
        y[base + k] = v[k] * 0.03125f;   // 1/sqrt(1024)
    }
}

// ---- PTQ1_0 (type 143): base-3 digits of a 128-value group of 28 bytes (26 payload + fp16 scale at +26).
// Constant-memory digit tables, uploaded once by ternary_ptq1_init() from the empirical kPtq1Code table.
__constant__ int8_t c_ptq1_digits5[256 * 5];
__constant__ int8_t c_ptq1_digits4[256 * 4];
template <int TT>
__global__ void gemv_ptq1_0_kernel(const uint8_t * __restrict__ W, int64_t in, int64_t out, const int8_t * __restrict__ qx,
                                   const float * __restrict__ dx, const float * __restrict__ sx, int T, float * __restrict__ y) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t o = static_cast<int64_t>(blockIdx.x) * WARPS + warp;
    const int t0 = blockIdx.y * TT;
    const int nt = min(TT, T - t0);
    const int64_t ngroups = in / 128, nb32 = in / 32;
    const uint8_t * row = W + o * ngroups * ternary::kPtqGroupBytes;
    __shared__ int8_t codes[WARPS][128];   // per-warp scratch: one set of 128 ternary codes per output row
    // Digit tables copied from constant memory into shared memory once per block. The phase-1 lookups index the
    // table with a different payload byte in every lane; constant memory serializes such divergent reads (one
    // address per cycle), shared memory serves them in parallel. Same values, so the result is bit-identical.
    __shared__ int8_t s_d5[256 * 5], s_d4[256 * 4];
    for (int i = threadIdx.x; i < 256 * 5; i += blockDim.x) s_d5[i] = c_ptq1_digits5[i];
    for (int i = threadIdx.x; i < 256 * 4; i += blockDim.x) s_d4[i] = c_ptq1_digits4[i];
    __syncthreads();
    if (o >= out) return;   // after the block-wide table copy
    float acc[TT];
#pragma unroll
    for (int t = 0; t < TT; ++t) acc[t] = 0.0f;

    for (int64_t g = 0; g < ngroups; ++g) {
        const uint8_t * blk = row + g * ternary::kPtqGroupBytes;
        const float d = f16_to_f32(rd16(blk + 26));
        // Phase 1: lanes 0..25 decode their payload byte into the group positions it covers. Each warp writes
        // its own 128-code set (WARPS output rows share the block, so the scratch is per warp).
        int8_t * cw = codes[warp];
        if (lane < 26) {
            const int b = blk[lane];
            if (lane < 16) {   // stage 1: byte j -> positions {j, j+16, j+32, j+48, j+64}
                const int8_t * dg = s_d5 + b * 5;
#pragma unroll
                for (int s = 0; s < 5; ++s) cw[lane + 16 * s] = static_cast<int8_t>(dg[s] - 1);
            } else if (lane < 24) {  // stage 2: byte 16+k -> positions {80+k, 88+k, 96+k, 104+k, 112+k}
                const int k = lane - 16;
                const int8_t * dg = s_d5 + b * 5;
#pragma unroll
                for (int s = 0; s < 5; ++s) cw[80 + k + 8 * s] = static_cast<int8_t>(dg[s] - 1);
            } else {  // tail: byte 24 -> {120,122,124,126}, byte 25 -> {121,123,125,127}
                const int8_t * dg = s_d4 + b * 4;
                const int base = 120 + (lane - 24);
#pragma unroll
                for (int s = 0; s < 4; ++s) cw[base + 2 * s] = static_cast<int8_t>(dg[s] - 1);
            }
        }
        __syncwarp();
        // Phase 2: lane handles the four codes at positions 4*lane .. 4*lane+3 (one Q8_1 activation block each).
        // Four activation bytes are read as one uint32 (4-byte aligned: in % 64 == 0, 128 % 64 == 0, p % 4 == 0).
        const int p = lane * 4;
        const int32_t c4 = *reinterpret_cast<const int32_t *>(cw + p);
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            if (t < nt) {
                const int64_t tt = t0 + t;
                const uint32_t a4 = *reinterpret_cast<const uint32_t *>(qx + tt * in + g * 128 + p);
                const int64_t b32 = tt * nb32 + g * 4 + (lane >> 3);
                // codes are already (digit - 1) in {-1,0,1}: no +1 baseline to remove, so no -sx term (Q2_0 needs one,
                // because its 2-bit codes are 0..2 and sx cancels the +1)
                acc[t] += d * dx[b32] * static_cast<float>(__dp4a(c4, static_cast<int>(a4), 0));
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
        for (int t = 0; t < TT; ++t)
            if (t < nt) y[static_cast<int64_t>(t0 + t) * out + o] = acc[t];
    }
}

template <int TT>
cudaError_t launch_ptq1(const uint8_t * W, int64_t in, int64_t out, const ActQ8 & a, int64_t T, float * y, cudaStream_t s) {
    const dim3 grid(static_cast<unsigned>((out + WARPS - 1) / WARPS), static_cast<unsigned>((T + TT - 1) / TT));
    gemv_ptq1_0_kernel<TT><<<grid, WARPS * 32, 0, s>>>(W, in, out, a.qx, a.dx, a.sx, static_cast<int>(T), y);
    return cudaGetLastError();
}

// ================================================================================================================
// Optimized kernels (exp/bonsai-128k-context). Same lane -> data mapping, same integer dot products and the same FP32
// operation order as the reference kernels above, so every output is bit-identical (tests/cuda/ternary_ab_test.cu).
// ================================================================================================================

// fp16 -> fp32 with the hardware conversion (F2F.F32.F16: exact, a conversion and not half arithmetic, so the SASS gate
// for HFMA2 / HMUL2 / HADD2 is unaffected). Same value as f16_to_f32 for every finite input.
__device__ __forceinline__ float f16_cvt(uint32_t h) {
    float r;
    const unsigned short hs = static_cast<unsigned short>(h & 0xFFFFu);
    asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(hs));
    return r;
}

// int -> float for |x| < 2^22 without I2F (a quarter-rate instruction on sm_61): 0x4B400000 is 1.5 * 2^23, so the bit
// pattern 0x4B400000 + x is exactly the float 12582912 + x, and subtracting 12582912 is exact. Same value as
// static_cast<float>(x) for every |x| < 2^22 (the dp4a sums here are at most 4064 in magnitude), so results stay bit-identical.
__device__ __forceinline__ float i2f_small(int x) { return __int_as_float(0x4B400000 + x) - 12582912.0f; }
// float(dot4(c, a)) with the magic constant as the dp4a accumulator (saves the integer add of i2f_small). Exact for every
// int8 x {-1, 0, 1} dot of 4 terms.
__device__ __forceinline__ float dp4a_f(int c, int a) { return __int_as_float(__dp4a(c, a, 0x4B400000)) - 12582912.0f; }

// ---- Q2_0: byte k of a 16-code word holds elements 4k..4k+3 (2 bits each). Spreading it to one int8 per element gives a
// word that pairs with four consecutive activation bytes, so the activations are loaded as they are (no prmt transpose
// per token and weight word, which the reference kernel needs because unpack16 interleaves the elements).
__device__ __forceinline__ uint32_t spread4(uint32_t b) {   // b = one byte -> codes of 4 consecutive elements, one per byte
    return lop3_and(b | (b << 6) | (b << 12) | (b << 18), 0x03030303u);
}

template <int TT>
__global__ void gemv_q2_0_kernel_v2(const uint8_t * __restrict__ W, int64_t in, int64_t out, const int8_t * __restrict__ qx,
                                    const float * __restrict__ dx, const float * __restrict__ sx, int T, float * __restrict__ y) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t o = static_cast<int64_t>(blockIdx.x) * WARPS + warp;
    if (o >= out) return;
    const int t0 = blockIdx.y * TT;
    const int nt = min(TT, T - t0);
    const int64_t nblk = in / 64, nb32 = in / 32;
    const uint8_t * row = W + o * nblk * kQ2_0BlockBytes;
    float acc[TT];
#pragma unroll
    for (int t = 0; t < TT; ++t) acc[t] = 0.0f;

    for (int64_t u = lane; u < nblk; u += 32) {
        const uint8_t * blk = row + u * kQ2_0BlockBytes;
        const float d = f16_to_f32(rd16(blk));
        uint32_t w[16];   // w[4g + k] = codes of elements 16g + 4k .. 16g + 4k + 3
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            const uint32_t q = rd32h(blk + 2 + 4 * g);
#pragma unroll
            for (int k = 0; k < 4; ++k) w[4 * g + k] = spread4((q >> (8 * k)) & 0xFFu);
        }
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            if (t < nt) {
                const int64_t tt = t0 + t;
                const int4 * xq = reinterpret_cast<const int4 *>(qx + tt * in + u * 64);
                float s = 0.0f;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    int idot = 0;
#pragma unroll
                    for (int g2 = 0; g2 < 2; ++g2) {
                        const int g = 2 * h + g2;
                        const int4 x = xq[g];
                        idot = __dp4a(static_cast<int>(w[4 * g + 0]), x.x, idot);
                        idot = __dp4a(static_cast<int>(w[4 * g + 1]), x.y, idot);
                        idot = __dp4a(static_cast<int>(w[4 * g + 2]), x.z, idot);
                        idot = __dp4a(static_cast<int>(w[4 * g + 3]), x.w, idot);
                    }
                    const int64_t b = tt * nb32 + 2 * u + h;
                    s += dx[b] * i2f_small(idot) - sx[b];
                }
                acc[t] += d * s;
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
        for (int t = 0; t < TT; ++t)
            if (t < nt) y[static_cast<int64_t>(t0 + t) * out + o] = acc[t];
    }
}

// ---- PTQ1_0 v2. Lane L still owns positions 4L .. 4L+3 of every 128-value group (as the reference), but it decodes its
// own four codes straight from the payload instead of going through a per-warp shared-memory code buffer and __syncwarp:
//   L <  20: positions 4L..4L+3 = digit L/4 of payload bytes 4(L%4) .. 4(L%4)+3   (= payload word L%4)
//   L <  30: positions 80+4m..  = digit m/2 of payload bytes 16+4(m%2) .. +3        (= payload word 4 + m%2), m = L-20
//   L = 30 : (b24 d0, b25 d0, b24 d1, b25 d1);  L = 31: (b24 d2, b25 d2, b24 d3, b25 d3)   (word 6, low half)
// The 28-byte groups are read as 7 aligned words; KG = 4 groups (112 bytes, lanes 0..27) are loaded per step and the next
// step is prefetched, so a warp keeps 2 x 112 bytes in flight instead of 26 single-byte loads per group.
constexpr int PW = 8;    // output rows (warps) per block: halves the per-block table copies of the reference
constexpr int KG = 4;    // groups per load step (every PTQ1_0 width: 5120 / 6144 / 17408 = 40 / 48 / 136 groups)

// Decode / verification kernel (T = 1..5): every warp works on R output rows at once (independent load / decode / dot
// chains in flight) and on all TT tokens of the batch, so each decoded code word serves TT tokens. Per row and
// token the accumulator and the group order are those of the reference: bit-identical.
template <int TT, int R>
__global__ void __launch_bounds__(PW * 32) gemv_ptq1_0_kernel_v2(const uint8_t * __restrict__ W, int64_t in, int64_t out,
                                                                 const int8_t * __restrict__ qx, const float * __restrict__ dx,
                                                                 int T, float * __restrict__ y) {
    __shared__ uint8_t s_t5[8 * 256], s_t4[8 * 256];
    for (int i = threadIdx.x; i < 256 * 5; i += blockDim.x) s_t5[(i / 5) * 8 + i % 5] = static_cast<uint8_t>(c_ptq1_digits5[i] - 1);
    for (int i = threadIdx.x; i < 256 * 4; i += blockDim.x) s_t4[(i / 4) * 8 + i % 4] = static_cast<uint8_t>(c_ptq1_digits4[i] - 1);
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const bool tail = lane >= 30;
    const int widx = lane < 20 ? (lane & 3) : (!tail ? 4 + ((lane - 20) & 1) : 6);
    const int dig = lane < 20 ? (lane >> 2) : (!tail ? ((lane - 20) >> 1) : 2 * (lane - 30));
    const int ld_g = lane / 7, ld_w = lane % 7;
    const uint8_t * tab = (tail ? s_t4 : s_t5) + dig;
    const uint32_t bsel = tail ? 0x1010u : 0x3210u;
    const int off2 = tail ? 1 : 0;
    const int ngroups = static_cast<int>(in / 128), nb32 = static_cast<int>(in / 32), in32 = static_cast<int>(in);
    const int8_t * qx0 = qx + lane * 4;
    const float * dx0 = dx + (lane >> 3);
    const int64_t step = static_cast<int64_t>(gridDim.x) * PW * R;
    for (int64_t ob = static_cast<int64_t>(blockIdx.x) * PW * R + warp; ob < out; ob += step) {
        int64_t o[R];
        bool live[R];
        const uint32_t * row[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            o[r] = ob + r * PW;
            live[r] = o[r] < out;
            row[r] = reinterpret_cast<const uint32_t *>(W + (live[r] ? o[r] : ob) * ngroups * ternary::kPtqGroupBytes);
        }
        float acc[R][TT];
        uint32_t nxt[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int t = 0; t < TT; ++t) acc[r][t] = 0.0f;
            nxt[r] = lane < 7 * KG ? __ldg(row[r] + ld_g * 7 + ld_w) : 0u;
        }
        for (int g0 = 0; g0 < ngroups; g0 += KG) {
            uint32_t cur[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                cur[r] = nxt[r];
                if (g0 + KG < ngroups && lane < 7 * KG) nxt[r] = __ldg(row[r] + (g0 + KG + ld_g) * 7 + ld_w);
            }
#pragma unroll
            for (int k = 0; k < KG; ++k) {
                const int g = g0 + k;
                int a4[TT];
                float dxv[TT];
#pragma unroll
                for (int t = 0; t < TT; ++t) {
                    const int tt = t < T ? t : 0;   // T == TT at dispatch; guard anyway
                    a4[t] = static_cast<int>(__ldg(reinterpret_cast<const uint32_t *>(qx0 + tt * in32 + g * 128)));
                    dxv[t] = __ldg(dx0 + tt * nb32 + g * 4);
                }
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const uint32_t wv = __byte_perm(__shfl_sync(0xffffffffu, cur[r], k * 7 + widx), 0u, bsel);
                    const uint32_t w6 = __shfl_sync(0xffffffffu, cur[r], k * 7 + 6);
                    const float d = f16_cvt(w6 >> 16);
                    const uint32_t c0 = tab[8 * (wv & 0xFFu)], c1 = tab[8 * ((wv >> 8) & 0xFFu)];
                    const uint32_t c2 = tab[8 * ((wv >> 16) & 0xFFu) + off2], c3 = tab[8 * (wv >> 24) + off2];
                    const int c4 = static_cast<int>(__byte_perm(__byte_perm(c0, c1, 0x0040), __byte_perm(c2, c3, 0x0040), 0x5410));
#pragma unroll
                    for (int t = 0; t < TT; ++t) acc[r][t] += d * dxv[t] * dp4a_f(c4, a4[t]);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int t = 0; t < TT; ++t) {
                float v = acc[r][t];
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
                if (lane == 0 && live[r] && t < T) y[static_cast<int64_t>(t) * out + o[r]] = v;
            }
        }
    }
}

// ---- PTQ1_0 v2, batched (T > 1): the 8 warps of a block (8 output rows) share the activations of the step. qx and dx of
// KG groups x TT tokens are staged in shared memory once per block, so each activation byte is read from global memory
// once per 8 rows instead of once per row, and the per-token loads in the inner loop become conflict-free LDS. The lane
// mapping, the integer dot products and the FP32 order are those of the reference: bit-identical outputs.
template <int TT>
__global__ void __launch_bounds__(PW * 32) gemv_ptq1_0_kernel_tile(const uint8_t * __restrict__ W, int64_t in, int64_t out,
                                                                   const int8_t * __restrict__ qx, const float * __restrict__ dx,
                                                                   int T, float * __restrict__ y) {
    __shared__ uint8_t s_t5[8 * 256], s_t4[8 * 256];
    __shared__ uint32_t s_x[TT][KG * 32];   // TT tokens x KG groups x 128 int8
    __shared__ float s_dx[TT][KG * 4];
    for (int i = threadIdx.x; i < 256 * 5; i += blockDim.x) s_t5[(i / 5) * 8 + i % 5] = static_cast<uint8_t>(c_ptq1_digits5[i] - 1);
    for (int i = threadIdx.x; i < 256 * 4; i += blockDim.x) s_t4[(i / 4) * 8 + i % 4] = static_cast<uint8_t>(c_ptq1_digits4[i] - 1);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const bool tail = lane >= 30;
    const int widx = lane < 20 ? (lane & 3) : (!tail ? 4 + ((lane - 20) & 1) : 6);
    const int dig = lane < 20 ? (lane >> 2) : (!tail ? ((lane - 20) >> 1) : 2 * (lane - 30));
    const int ld_g = lane / 7, ld_w = lane % 7;
    const uint8_t * tab = (tail ? s_t4 : s_t5) + dig;
    const uint32_t bsel = tail ? 0x1010u : 0x3210u;
    const int off2 = tail ? 1 : 0;
    const int t0 = blockIdx.y * TT;
    const int nt = min(TT, T - t0);
    const int ngroups = static_cast<int>(in / 128), nb32 = static_cast<int>(in / 32);
    const int64_t o = static_cast<int64_t>(blockIdx.x) * PW + warp;
    const bool live = o < out;   // dead warps still help staging and reach every barrier
    const uint32_t * row = reinterpret_cast<const uint32_t *>(W + (live ? o : 0) * ngroups * ternary::kPtqGroupBytes);
    const uint32_t * qxw = reinterpret_cast<const uint32_t *>(qx);
    const int inw = static_cast<int>(in / 4);
    float acc[TT];
#pragma unroll
    for (int t = 0; t < TT; ++t) acc[t] = 0.0f;
    uint32_t nxt = (live && lane < 7 * KG) ? __ldg(row + ld_g * 7 + ld_w) : 0u;
    for (int g0 = 0; g0 < ngroups; g0 += KG) {
        __syncthreads();   // previous step done with s_x / s_dx (and the tables are built before the first step)
        for (int i = threadIdx.x; i < TT * KG * 32; i += blockDim.x) {
            const int t = i / (KG * 32), w = i % (KG * 32);
            s_x[t][w] = t < nt ? __ldg(qxw + static_cast<int64_t>(t0 + t) * inw + g0 * 32 + w) : 0u;
        }
        for (int i = threadIdx.x; i < TT * KG * 4; i += blockDim.x) {
            const int t = i / (KG * 4), b = i % (KG * 4);
            s_dx[t][b] = t < nt ? __ldg(dx + static_cast<int64_t>(t0 + t) * nb32 + g0 * 4 + b) : 0.0f;
        }
        __syncthreads();
        const uint32_t cur = nxt;
        if (live && g0 + KG < ngroups && lane < 7 * KG) nxt = __ldg(row + (g0 + KG + ld_g) * 7 + ld_w);
        if (live) {
#pragma unroll
            for (int k = 0; k < KG; ++k) {
                const uint32_t wv = __byte_perm(__shfl_sync(0xffffffffu, cur, k * 7 + widx), 0u, bsel);
                const uint32_t w6 = __shfl_sync(0xffffffffu, cur, k * 7 + 6);
                const float d = f16_cvt(w6 >> 16);
                const uint32_t c0 = tab[8 * (wv & 0xFFu)], c1 = tab[8 * ((wv >> 8) & 0xFFu)];
                const uint32_t c2 = tab[8 * ((wv >> 16) & 0xFFu) + off2], c3 = tab[8 * (wv >> 24) + off2];
                const int c4 = static_cast<int>(__byte_perm(__byte_perm(c0, c1, 0x0040), __byte_perm(c2, c3, 0x0040), 0x5410));
#pragma unroll
                for (int t = 0; t < TT; ++t) {
                    if (t < nt) {
                        const int a4 = static_cast<int>(s_x[t][k * 32 + lane]);
                        acc[t] += d * s_dx[t][k * 4 + (lane >> 3)] * i2f_small(__dp4a(c4, a4, 0));
                    }
                }
            }
        }
    }
    if (!live) return;
#pragma unroll
    for (int t = 0; t < TT; ++t) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], off);
    }
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < TT; ++t)
            if (t < nt) y[static_cast<int64_t>(t0 + t) * out + o] = acc[t];
    }
}

template <int TT>
cudaError_t launch_ptq1_tile(const uint8_t * W, int64_t in, int64_t out, const ActQ8 & a, int64_t T, float * y, cudaStream_t s) {
    const dim3 grid(static_cast<unsigned>((out + PW - 1) / PW), static_cast<unsigned>((T + TT - 1) / TT));
    gemv_ptq1_0_kernel_tile<TT><<<grid, PW * 32, 0, s>>>(W, in, out, a.qx, a.dx, static_cast<int>(T), y);
    return cudaGetLastError();
}


// ---- PTQ1_0 prefill kernel v3 (T > 5): each warp owns R output rows and TT tokens. Per group the TT activation words
// and scales are read from shared memory once into registers and reused by the R rows (the v2 tile kernel reads them
// once per row: two shared loads per dp4a made it shared-memory bound). Shared layout [word][TT + 4]: one lane reads its
// TT consecutive words as 128-bit loads, and the +4 padding keeps the 8 lanes of a 128-byte phase on distinct banks.
// Per row and token the accumulation is the reference one (acc += d * dx * dot, groups in order): bit-identical.
template <int TT, int R>
__global__ void __launch_bounds__(PW * 32) gemv_ptq1_0_kernel_tile2(const uint8_t * __restrict__ W, int64_t in, int64_t out,
                                                                    const int8_t * __restrict__ qx, const float * __restrict__ dx,
                                                                    int T, float * __restrict__ y) {
    constexpr int TS = TT + 4;
    constexpr int NW = KG * 32;                    // activation words per token and step
    constexpr int XPT = TT * NW / (PW * 32);       // words staged per thread and step
    constexpr int DPT = (TT * KG * 4 + PW * 32 - 1) / (PW * 32);
    static_assert(TT * NW % (PW * 32) == 0, "staging layout");
    __shared__ uint8_t s_t5[8 * 256], s_t4[8 * 256];
    __shared__ __align__(16) uint32_t s_x[2][NW][TS];   // double buffer: one barrier per step
    __shared__ __align__(16) float s_dx[2][KG * 4][TS];
    for (int i = threadIdx.x; i < 256 * 5; i += blockDim.x) s_t5[(i / 5) * 8 + i % 5] = static_cast<uint8_t>(c_ptq1_digits5[i] - 1);
    for (int i = threadIdx.x; i < 256 * 4; i += blockDim.x) s_t4[(i / 4) * 8 + i % 4] = static_cast<uint8_t>(c_ptq1_digits4[i] - 1);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const bool tail = lane >= 30;
    const int widx = lane < 20 ? (lane & 3) : (!tail ? 4 + ((lane - 20) & 1) : 6);
    const int dig = lane < 20 ? (lane >> 2) : (!tail ? ((lane - 20) >> 1) : 2 * (lane - 30));
    const int ld_g = lane / 7, ld_w = lane % 7;
    const uint8_t * tab = (tail ? s_t4 : s_t5) + dig;
    const uint32_t bsel = tail ? 0x1010u : 0x3210u;
    const int off2 = tail ? 1 : 0;
    const int t0 = blockIdx.y * TT;
    const int nt = min(TT, T - t0);
    const int ngroups = static_cast<int>(in / 128), nb32 = static_cast<int>(in / 32);
    // staging: thread copies word sw of tokens st + j * (PW * 32 / NW); one token row is a contiguous run of NW words
    constexpr int TSTEP = PW * 32 / NW;
    const int sw = tid % NW, st = tid / NW;
    const uint32_t * xsrc[XPT];
#pragma unroll
    for (int jx = 0; jx < XPT; ++jx) {
        const int t = st + jx * TSTEP;
        xsrc[jx] = t < nt ? reinterpret_cast<const uint32_t *>(qx + static_cast<int64_t>(t0 + t) * in) + sw : nullptr;
    }
    const int dt = tid / (KG * 4), db = tid % (KG * 4);
    const float * dsrc = (DPT == 1 && dt < TT && dt < nt) ? dx + static_cast<int64_t>(t0 + dt) * nb32 + db : nullptr;
    int64_t o[R];
    bool live[R];
    const uint32_t * row[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        o[r] = (static_cast<int64_t>(blockIdx.x) * R + r) * PW + warp;
        live[r] = o[r] < out;
        row[r] = reinterpret_cast<const uint32_t *>(W + (live[r] ? o[r] : 0) * ngroups * ternary::kPtqGroupBytes);
    }
    float acc[R][TT];
    uint32_t nxt[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < TT; ++t) acc[r][t] = 0.0f;
        nxt[r] = lane < 7 * KG ? __ldg(row[r] + ld_g * 7 + ld_w) : 0u;
    }
    uint32_t xr[XPT];
    float dr = 0.0f;
    auto load_step = [&](int g0) {
#pragma unroll
        for (int jx = 0; jx < XPT; ++jx) xr[jx] = xsrc[jx] ? __ldg(xsrc[jx] + g0 * 32) : 0u;
        if (DPT == 1 && dt < TT) dr = dsrc ? __ldg(dsrc + g0 * 4) : 0.0f;
    };
    auto store_step = [&](int b) {
#pragma unroll
        for (int jx = 0; jx < XPT; ++jx) s_x[b][sw][st + jx * TSTEP] = xr[jx];
        if (DPT == 1 && dt < TT) s_dx[b][db][dt] = dr;
    };
    load_step(0);
    store_step(0);
    __syncthreads();   // tables and the first step are in shared memory
    int buf = 0;
    for (int g0 = 0; g0 < ngroups; g0 += KG, buf ^= 1) {
        const bool more = g0 + KG < ngroups;
        if (more) load_step(g0 + KG);
        uint32_t cur[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            cur[r] = nxt[r];
            if (more && lane < 7 * KG) nxt[r] = __ldg(row[r] + (g0 + KG + ld_g) * 7 + ld_w);
        }
#pragma unroll
        for (int k = 0; k < KG; ++k) {
            int a4[TT];
            float dxv[TT];
#pragma unroll
            for (int t4 = 0; t4 < TT; t4 += 4) {
                const uint4 xa = *reinterpret_cast<const uint4 *>(&s_x[buf][k * 32 + lane][t4]);
                const float4 da = *reinterpret_cast<const float4 *>(&s_dx[buf][k * 4 + (lane >> 3)][t4]);
                a4[t4] = static_cast<int>(xa.x); a4[t4 + 1] = static_cast<int>(xa.y);
                a4[t4 + 2] = static_cast<int>(xa.z); a4[t4 + 3] = static_cast<int>(xa.w);
                dxv[t4] = da.x; dxv[t4 + 1] = da.y; dxv[t4 + 2] = da.z; dxv[t4 + 3] = da.w;
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint32_t wv = __byte_perm(__shfl_sync(0xffffffffu, cur[r], k * 7 + widx), 0u, bsel);
                const uint32_t w6 = __shfl_sync(0xffffffffu, cur[r], k * 7 + 6);
                const float d = f16_cvt(w6 >> 16);
                const uint32_t c0 = tab[8 * (wv & 0xFFu)], c1 = tab[8 * ((wv >> 8) & 0xFFu)];
                const uint32_t c2 = tab[8 * ((wv >> 16) & 0xFFu) + off2], c3 = tab[8 * (wv >> 24) + off2];
                const int c4 = static_cast<int>(__byte_perm(__byte_perm(c0, c1, 0x0040), __byte_perm(c2, c3, 0x0040), 0x5410));
#pragma unroll
                for (int t = 0; t < TT; ++t) acc[r][t] += d * dxv[t] * dp4a_f(c4, a4[t]);
            }
        }
        if (more) store_step(buf ^ 1);
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            float v = acc[r][t];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
            if (lane == 0 && live[r] && t < nt) y[static_cast<int64_t>(t0 + t) * out + o[r]] = v;
        }
    }
}

template <int TT, int R>
cudaError_t launch_ptq1_tile2(const uint8_t * W, int64_t in, int64_t out, const ActQ8 & a, int64_t T, float * y, cudaStream_t s) {
    const dim3 grid(static_cast<unsigned>((out + PW * R - 1) / (PW * R)), static_cast<unsigned>((T + TT - 1) / TT));
    gemv_ptq1_0_kernel_tile2<TT, R><<<grid, PW * 32, 0, s>>>(W, in, out, a.qx, a.dx, static_cast<int>(T), y);
    return cudaGetLastError();
}

// ---- Fused Hadamard rotation + Q8_1 quantization: rotate_kernel followed by the arithmetic of actq8_kernel on the
// rotated values while they are still in shared memory (one launch and one global round trip less per GEMV input).
// Warp w of the block handles positions 256 i + 32 w + lane, i.e. Q8_1 block 8 i + w of the 1024-value block, with the
// same shuffles and the same rounding as actq8_kernel, so qx / dx / sx are bit-identical to rotate -> quantize.
__global__ void rotate_q8_kernel(const float * __restrict__ x, const float * __restrict__ signs, float * __restrict__ y, int width,
                                 int8_t * __restrict__ qx, float * __restrict__ dx, float * __restrict__ sx) {
    __shared__ float v[1024];
    const int t = blockIdx.y, blk = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int64_t base = static_cast<int64_t>(t) * width + static_cast<int64_t>(blk) * 1024;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int k = tid + 256 * i;
        v[k] = x[base + k] * signs[blk * 1024 + k];
    }
    __syncthreads();
    for (int h = 1; h < 1024; h <<= 1) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int p = tid + 256 * i;
            const int lo = (p / h) * 2 * h + (p % h), hi = lo + h;
            const float a = v[lo], b = v[hi];
            v[lo] = a + b;
            v[hi] = a - b;
        }
        __syncthreads();
    }
    const int64_t nb32 = width / 32;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int k = tid + 256 * i;
        const float r = v[k] * 0.03125f;
        y[base + k] = r;
        float amax = fabsf(r);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
        const float dq = amax / 127.0f;
        const float inv = dq > 0.0f ? 1.0f / dq : 0.0f;
        int q = __float2int_rn(r * inv);
        q = max(-127, min(127, q));
        qx[base + k] = static_cast<int8_t>(q);
        int qsum = q;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) qsum += __shfl_xor_sync(0xffffffffu, qsum, off);
        if (lane == 0) {
            const int64_t b = static_cast<int64_t>(t) * nb32 + static_cast<int64_t>(blk) * 32 + 8 * i + warp;
            dx[b] = dq;
            sx[b] = dq * static_cast<float>(qsum);
        }
    }
}

template <int TT>
cudaError_t launch_q2_v2(const uint8_t * W, int64_t in, int64_t out, const ActQ8 & a, int64_t T, float * y, cudaStream_t s) {
    const dim3 grid(static_cast<unsigned>((out + WARPS - 1) / WARPS), static_cast<unsigned>((T + TT - 1) / TT));
    gemv_q2_0_kernel_v2<TT><<<grid, WARPS * 32, 0, s>>>(W, in, out, a.qx, a.dx, a.sx, static_cast<int>(T), y);
    return cudaGetLastError();
}
template <int TT, int R>
cudaError_t launch_ptq1_v2(const uint8_t * W, int64_t in, int64_t out, const ActQ8 & a, int64_t T, float * y, cudaStream_t s) {
    static int n_sm = 0;
    if (!n_sm) { cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, 0); if (n_sm < 1) n_sm = 19; }
    // persistent: at most 8 blocks of 256 threads per SM, each block loops over groups of PW * R rows
    // persistent blocks per SM (QWEN_PTQ1_BPS, tuning). 48 registers x 256 threads: 5 blocks fit an SM, so 5 is one wave;
    // 8 left a partial second wave (ffn_up T=1: 229 -> 208 us, LM head 3244 -> 2778 us).
    static int bps = -1;
    if (bps < 0) { const char * e = std::getenv("QWEN_PTQ1_BPS"); bps = e ? std::atoi(e) : 5; }
    const int64_t rows = (out + PW * R - 1) / (PW * R), cap = static_cast<int64_t>(n_sm) * bps;
    gemv_ptq1_0_kernel_v2<TT, R><<<dim3(static_cast<unsigned>(rows < cap ? rows : cap)), PW * 32, 0, s>>>(W, in, out, a.qx, a.dx, static_cast<int>(T), y);
    return cudaGetLastError();
}

}  // namespace

cudaError_t ternary_rotate(const float * x, const float * signs, float * y, int width, int T, cudaStream_t s) {
    if (width % 1024 != 0 || T < 1) return cudaErrorInvalidValue;
    rotate_kernel<<<dim3(static_cast<unsigned>(width / 1024), static_cast<unsigned>(T)), 256, 0, s>>>(x, signs, y, width);
    return cudaGetLastError();
}

cudaError_t gemv_q2_0_dp4a_ref(const uint8_t * W, int64_t in, int64_t out, const ActQ8 & a, int64_t T, float * y, cudaStream_t s) {
    if (in % 64 != 0) return cudaErrorInvalidValue;
    if (T <= 1) return launch<1>(W, in, out, a, T, y, s);
    if (T <= 4) return launch<4>(W, in, out, a, T, y, s);
    if (T <= 8) return launch<8>(W, in, out, a, T, y, s);
    return launch<16>(W, in, out, a, T, y, s);
}

cudaError_t gemv_ptq1_0_dp4a_ref(const uint8_t * W, int64_t in, int64_t out, const ActQ8 & a, int64_t T, float * y, cudaStream_t s) {
    if (in % 128 != 0) return cudaErrorInvalidValue;
    ternary_ptq1_init();
    if (T <= 1) return launch_ptq1<1>(W, in, out, a, T, y, s);
    if (T <= 4) return launch_ptq1<4>(W, in, out, a, T, y, s);
    if (T <= 8) return launch_ptq1<8>(W, in, out, a, T, y, s);
    return launch_ptq1<16>(W, in, out, a, T, y, s);
}

cudaError_t ternary_rotate_q8(const float * x, const float * signs, float * y, int width, int T, const ActQ8 & a, cudaStream_t s) {
    if (width % 1024 != 0 || T < 1 || T > a.T_max || width > a.in_max) return cudaErrorInvalidValue;
    rotate_q8_kernel<<<dim3(static_cast<unsigned>(width / 1024), static_cast<unsigned>(T)), 256, 0, s>>>(x, signs, y, width, a.qx, a.dx, a.sx);
    return cudaGetLastError();
}

cudaError_t gemv_q2_0_dp4a(const uint8_t * W, int64_t in, int64_t out, const ActQ8 & a, int64_t T, float * y, cudaStream_t s) {
    if (in % 64 != 0) return cudaErrorInvalidValue;
    if (T <= 1) return launch_q2_v2<1>(W, in, out, a, T, y, s);
    if (T <= 4) return launch_q2_v2<4>(W, in, out, a, T, y, s);
    if (T <= 8) return launch_q2_v2<8>(W, in, out, a, T, y, s);
    return launch_q2_v2<16>(W, in, out, a, T, y, s);
}

namespace {
// T > 5. QWEN_PTQ1_TILE (tuning only): 0 = v2 tile kernel, 1..n = tile2 variants below.
cudaError_t ptq1_prefill_dispatch(const uint8_t * W, int64_t in, int64_t out, const ActQ8 & a, int64_t T, float * y, cudaStream_t s) {
    static int mode = -1;
    // Measured (tests/cuda/ternary_bench.cu, ffn_up 5120x17408, T = 16): v2 tile 1741 us, tile2 <16,4> 1158 us,
    // <16,1> 1275 us, <16,2> 762 us (the default; 123 registers, two blocks per SM).
    if (mode < 0) { const char * e = std::getenv("QWEN_PTQ1_TILE"); mode = e ? std::atoi(e) : 1; }
    switch (mode) {
        case 0: return T <= 8 ? launch_ptq1_tile<8>(W, in, out, a, T, y, s) : launch_ptq1_tile<16>(W, in, out, a, T, y, s);
        case 2: return T <= 8 ? launch_ptq1_tile2<8, 4>(W, in, out, a, T, y, s) : launch_ptq1_tile2<16, 4>(W, in, out, a, T, y, s);
        case 4: return T <= 8 ? launch_ptq1_tile2<8, 2>(W, in, out, a, T, y, s) : launch_ptq1_tile2<16, 1>(W, in, out, a, T, y, s);
        case 5: return T <= 8 ? launch_ptq1_tile2<8, 4>(W, in, out, a, T, y, s)
                              : (T <= 16 ? launch_ptq1_tile2<16, 2>(W, in, out, a, T, y, s) : launch_ptq1_tile2<32, 1>(W, in, out, a, T, y, s));
        case 6: return T <= 8 ? launch_ptq1_tile2<8, 4>(W, in, out, a, T, y, s)
                              : (T <= 16 ? launch_ptq1_tile2<16, 2>(W, in, out, a, T, y, s) : launch_ptq1_tile2<32, 2>(W, in, out, a, T, y, s));
        default: return T <= 8 ? launch_ptq1_tile2<8, 4>(W, in, out, a, T, y, s) : launch_ptq1_tile2<16, 2>(W, in, out, a, T, y, s);
    }
}
}  // namespace

cudaError_t gemv_ptq1_0_dp4a(const uint8_t * W, int64_t in, int64_t out, const ActQ8 & a, int64_t T, float * y, cudaStream_t s) {
    if (in % (128 * KG) != 0) return cudaErrorInvalidValue;
    ternary_ptq1_init();
    if (T <= 1) {
        static int r1 = -1;   // rows per warp of the decode kernel (QWEN_PTQ1_R1, tuning)
        if (r1 < 0) { const char * e = std::getenv("QWEN_PTQ1_R1"); r1 = e ? std::atoi(e) : 4; }
        if (r1 == 2) return launch_ptq1_v2<1, 2>(W, in, out, a, T, y, s);
        if (r1 == 3) return launch_ptq1_v2<1, 3>(W, in, out, a, T, y, s);
        if (r1 == 6) return launch_ptq1_v2<1, 6>(W, in, out, a, T, y, s);
        return launch_ptq1_v2<1, 4>(W, in, out, a, T, y, s);
    }
    if (T == 2) return launch_ptq1_v2<2, 2>(W, in, out, a, T, y, s);
    if (T == 3) return launch_ptq1_v2<3, 2>(W, in, out, a, T, y, s);
    if (T == 4) return launch_ptq1_v2<4, 2>(W, in, out, a, T, y, s);
    if (T == 5) return launch_ptq1_v2<5, 2>(W, in, out, a, T, y, s);
    return ptq1_prefill_dispatch(W, in, out, a, T, y, s);
}

void ternary_ptq1_init() {
    static bool done = false;
    if (done) return;
    int8_t d5[256 * 5], d4[256 * 4];
    for (int b = 0; b < 256; ++b) {
        const int c = ternary::kPtq1Code[b];
        const int t3 = (c > 242) ? -1 : c;
        int v = t3 < 0 ? 0 : t3;
        for (int i = 4; i >= 0; --i) { d5[b * 5 + i] = static_cast<int8_t>(v % 3); v /= 3; }
        const int c4 = (t3 < 0 || t3 % 3 != 0) ? -1 : t3 / 3;
        v = c4 < 0 ? 0 : c4;
        for (int i = 3; i >= 0; --i) { d4[b * 4 + i] = static_cast<int8_t>(v % 3); v /= 3; }
    }
    cudaMemcpyToSymbol(c_ptq1_digits5, d5, sizeof(d5));
    cudaMemcpyToSymbol(c_ptq1_digits4, d4, sizeof(d4));
    done = true;
}

