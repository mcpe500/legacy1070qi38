#include "ternary_gemv.cuh"

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
    if (o >= out) return;
    const int t0 = blockIdx.y * TT;
    const int nt = min(TT, T - t0);
    const int64_t ngroups = in / 128, nb32 = in / 32;
    const uint8_t * row = W + o * ngroups * ternary::kPtqGroupBytes;
    __shared__ int8_t codes[WARPS][128];   // per-warp scratch: one set of 128 ternary codes per output row
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
                const int8_t * dg = c_ptq1_digits5 + b * 5;
#pragma unroll
                for (int s = 0; s < 5; ++s) cw[lane + 16 * s] = static_cast<int8_t>(dg[s] - 1);
            } else if (lane < 24) {  // stage 2: byte 16+k -> positions {80+k, 88+k, 96+k, 104+k, 112+k}
                const int k = lane - 16;
                const int8_t * dg = c_ptq1_digits5 + b * 5;
#pragma unroll
                for (int s = 0; s < 5; ++s) cw[80 + k + 8 * s] = static_cast<int8_t>(dg[s] - 1);
            } else {  // tail: byte 24 -> {120,122,124,126}, byte 25 -> {121,123,125,127}
                const int8_t * dg = c_ptq1_digits4 + b * 4;
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

}  // namespace

cudaError_t ternary_rotate(const float * x, const float * signs, float * y, int width, int T, cudaStream_t s) {
    if (width % 1024 != 0 || T < 1) return cudaErrorInvalidValue;
    rotate_kernel<<<dim3(static_cast<unsigned>(width / 1024), static_cast<unsigned>(T)), 256, 0, s>>>(x, signs, y, width);
    return cudaGetLastError();
}

cudaError_t gemv_q2_0_dp4a(const uint8_t * W, int64_t in, int64_t out, const ActQ8 & a, int64_t T, float * y, cudaStream_t s) {
    if (in % 64 != 0) return cudaErrorInvalidValue;
    if (T <= 1) return launch<1>(W, in, out, a, T, y, s);
    if (T <= 4) return launch<4>(W, in, out, a, T, y, s);
    if (T <= 8) return launch<8>(W, in, out, a, T, y, s);
    return launch<16>(W, in, out, a, T, y, s);
}

cudaError_t gemv_ptq1_0_dp4a(const uint8_t * W, int64_t in, int64_t out, const ActQ8 & a, int64_t T, float * y, cudaStream_t s) {
    if (in % 128 != 0) return cudaErrorInvalidValue;
    ternary_ptq1_init();
    if (T <= 1) return launch_ptq1<1>(W, in, out, a, T, y, s);
    if (T <= 4) return launch_ptq1<4>(W, in, out, a, T, y, s);
    if (T <= 8) return launch_ptq1<8>(W, in, out, a, T, y, s);
    return launch_ptq1<16>(W, in, out, a, T, y, s);
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

