#include "q8_gemv.cuh"

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
    dim3 gg((unsigned)out, (unsigned)T);
    q8_0_gemv_dp4a_kernel<<<gg, 32, 0, stream>>>(d_W, in, out, s.qx, s.dx, d_y, T);
    return cudaGetLastError();
}
