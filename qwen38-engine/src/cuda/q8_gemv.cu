#include "q8_gemv.cuh"

#include <cstdlib>

#include "f16bits.hpp"

// Pure FP32 path: fp16 scales are decoded with the integer helper from f16bits.hpp (tested against F16C for all bit patterns).
__device__ __forceinline__ float f16_bits_to_f32(uint32_t h) { return f16_to_f32(h); }

// Quantize activations: one warp per 32-value block. Grid (in/32, T), block 32.
__global__ void quantize_q8_1_kernel(const float * __restrict__ x, int64_t in, int8_t * __restrict__ qx,
                                     float * __restrict__ dx) {
    const int64_t t = blockIdx.y;
    const int64_t b = blockIdx.x;
    const int lane = threadIdx.x;
    const float v = x[t * in + b * 32 + lane];
    float amax = fabsf(v);
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    const float d = amax / 127.0f;
    const float inv = d > 0.0f ? 1.0f / d : 0.0f;
    int q = __float2int_rn(v * inv);
    q = max(-127, min(127, q));
    qx[(t * in) + b * 32 + lane] = (int8_t)q;
    if (lane == 0) dx[t * (in / 32) + b] = d;
}

// One warp per output row o for token t. Lane l handles blocks l, l+32, ...
// Block layout: [d: 2 bytes fp16][qs: 32 int8]. qs starts at byte offset 2 (not 4-byte aligned),
// so the four int8 values are assembled byte by byte before __dp4a.
__global__ void q8_0_gemv_dp4a_kernel(const uint8_t * __restrict__ W, int64_t in, int64_t out,
                                      const int8_t * __restrict__ qx, const float * __restrict__ dx,
                                      float * __restrict__ y, int64_t T) {
    const int64_t o = blockIdx.x;
    const int64_t t = blockIdx.y;
    const int lane = threadIdx.x;
    const int64_t nb = in / 32;
    const uint8_t * row = W + o * nb * 34;
    float acc = 0.0f;
    for (int64_t b = lane; b < nb; b += 32) {
        const uint8_t * blk = row + b * 34;
        const uint32_t dbits = (uint32_t)blk[0] | ((uint32_t)blk[1] << 8);
        const float dw = f16_bits_to_f32(dbits);
        const int8_t * xq = qx + t * in + b * 32;
        int isum = 0;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            const uint8_t * wq = blk + 2 + 4 * k;
            const int wv = (int)(uint8_t)wq[0] | ((int)(uint8_t)wq[1] << 8) | ((int)(uint8_t)wq[2] << 16) |
                           ((int)(uint8_t)wq[3] << 24);
            const int xv = *reinterpret_cast<const int *>(xq + 4 * k);  // activations are 4-byte aligned
            isum = __dp4a(wv, xv, isum);
        }
        acc += dw * dx[t * nb + b] * (float)isum;
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) y[t * out + o] = acc;
}

// Batched variant (issue #55): 8 output rows per block (one warp each) and TT tokens per pass. Each lane decodes the 8 weight
// words of its block once (16-bit loads: the quants start at byte 2) and reuses them for the TT tokens, so the weights are read
// once per TT tokens instead of once per token. Per (row, token) the lane order, the integer dot and the FP32 expression
// acc += dw * dx * isum are those of q8_0_gemv_dp4a_kernel: bit-identical results.
template <int TT>
__global__ void __launch_bounds__(256) q8_0_gemv_dp4a_kernel_bt(const uint8_t * __restrict__ W, int64_t in, int64_t out,
                                                               const int8_t * __restrict__ qx, const float * __restrict__ dx,
                                                               float * __restrict__ y, int64_t T) {
    const int lane = threadIdx.x & 31;
    const int64_t o = static_cast<int64_t>(blockIdx.x) * 8 + (threadIdx.x >> 5);
    if (o >= out) return;
    const int64_t t0 = static_cast<int64_t>(blockIdx.y) * TT;
    const int nt = static_cast<int>(min(static_cast<int64_t>(TT), T - t0));
    const int64_t nb = in / 32;
    const uint8_t * row = W + o * nb * 34;
    float acc[TT];
#pragma unroll
    for (int t = 0; t < TT; ++t) acc[t] = 0.0f;
    for (int64_t b = lane; b < nb; b += 32) {
        const unsigned short * blk = reinterpret_cast<const unsigned short *>(row + b * 34);   // 2-byte aligned
        const float dw = f16_bits_to_f32(__ldg(blk));
        int wv[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) wv[k] = static_cast<int>(static_cast<uint32_t>(__ldg(blk + 1 + 2 * k)) | (static_cast<uint32_t>(__ldg(blk + 2 + 2 * k)) << 16));
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            if (t < nt) {
                const int4 * xq = reinterpret_cast<const int4 *>(qx + (t0 + t) * in + b * 32);
                const int4 x0 = __ldg(xq), x1 = __ldg(xq + 1);
                int isum = 0;
                isum = __dp4a(wv[0], x0.x, isum); isum = __dp4a(wv[1], x0.y, isum);
                isum = __dp4a(wv[2], x0.z, isum); isum = __dp4a(wv[3], x0.w, isum);
                isum = __dp4a(wv[4], x1.x, isum); isum = __dp4a(wv[5], x1.y, isum);
                isum = __dp4a(wv[6], x1.z, isum); isum = __dp4a(wv[7], x1.w, isum);
                acc[t] += dw * dx[(t0 + t) * nb + b] * (float)isum;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < TT; ++t) {
        float v = acc[t];
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
        if (lane == 0 && t < nt) y[(t0 + t) * out + o] = v;
    }
}

cudaError_t q8_scratch_alloc(Q8Scratch & s, int64_t T_max, int64_t in_max) {
    cudaError_t e = cudaMalloc(&s.qx, sizeof(int8_t) * T_max * in_max);
    if (e != cudaSuccess) return e;
    e = cudaMalloc(&s.dx, sizeof(float) * T_max * (in_max / 32));
    if (e != cudaSuccess) return e;
    s.T_max = T_max;
    s.in_max = in_max;
    return cudaSuccess;
}

void q8_scratch_free(Q8Scratch & s) {
    if (s.qx) cudaFree(s.qx);
    if (s.dx) cudaFree(s.dx);
    s = Q8Scratch{};
}

cudaError_t q8_0_gemv_dp4a(const uint8_t * d_W, int64_t in, int64_t out, const float * d_x, int64_t T,
                           float * d_y, const Q8Scratch & s, cudaStream_t stream) {
    if (in % 32 != 0 || T > s.T_max || in > s.in_max) return cudaErrorInvalidValue;
    dim3 gq((unsigned)(in / 32), (unsigned)T);
    quantize_q8_1_kernel<<<gq, 32, 0, stream>>>(d_x, in, s.qx, s.dx);
    static int bt = -1;   // QWEN_Q8_BT=0: the one-warp-per-(row, token) reference kernel
    if (bt < 0) { const char * e = std::getenv("QWEN_Q8_BT"); bt = e ? std::atoi(e) : 1; }
    if (!bt) {
        dim3 gg((unsigned)out, (unsigned)T);
        q8_0_gemv_dp4a_kernel<<<gg, 32, 0, stream>>>(d_W, in, out, s.qx, s.dx, d_y, T);
        return cudaGetLastError();
    }
    const unsigned gx = static_cast<unsigned>((out + 7) / 8);
    if (T <= 1) q8_0_gemv_dp4a_kernel_bt<1><<<dim3(gx, 1), 256, 0, stream>>>(d_W, in, out, s.qx, s.dx, d_y, T);
    else if (T <= 4) q8_0_gemv_dp4a_kernel_bt<4><<<dim3(gx, 1), 256, 0, stream>>>(d_W, in, out, s.qx, s.dx, d_y, T);
    else q8_0_gemv_dp4a_kernel_bt<8><<<dim3(gx, static_cast<unsigned>((T + 7) / 8)), 256, 0, stream>>>(d_W, in, out, s.qx, s.dx, d_y, T);
    return cudaGetLastError();
}

cudaError_t q8_0_gemv_dp4a_ref(const uint8_t * d_W, int64_t in, int64_t out, const float * d_x, int64_t T,
                               float * d_y, const Q8Scratch & s, cudaStream_t stream) {
    if (in % 32 != 0 || T > s.T_max || in > s.in_max) return cudaErrorInvalidValue;
    dim3 gq((unsigned)(in / 32), (unsigned)T);
    quantize_q8_1_kernel<<<gq, 32, 0, stream>>>(d_x, in, s.qx, s.dx);
    dim3 gg((unsigned)out, (unsigned)T);
    q8_0_gemv_dp4a_kernel<<<gg, 32, 0, stream>>>(d_W, in, out, s.qx, s.dx, d_y, T);
    return cudaGetLastError();
}
