// Quantized GEMV on sm_61 with Q8_1-style activations and integer dot products (__dp4a), FP32 accumulation.
// The activation vector is quantized on the fly in blocks of 32 values: int8 values q, a float scale d, and the float
// sum s = d * sum(q) (needed by the types that store a minimum, Q4_K and Q5_K). Each weight unit of 32 values is
// unpacked to eight packed int8x4 words plus scale coefficients, then dotted with the activation block:
//     contribution = d_x * (s0 * sum_{g<4} dp4a(w[g], x[g]) + s1 * sum_{g>=4} dp4a(w[g], x[g])) - mn * s_x
// No half-precision arithmetic: fp16 scales are decoded with integer bit operations.
#pragma once
#include <cuda_runtime.h>

#include <cstdint>

struct ActQ8 {
    int8_t * qx = nullptr;  // [T][in] int8
    float * dx = nullptr;   // [T][in/32] block scales
    float * sx = nullptr;   // [T][in/32] d * sum(q)
    int64_t T_max = 0, in_max = 0;
};

cudaError_t actq8_alloc(ActQ8 & a, int64_t T_max, int64_t in_max);
void actq8_free(ActQ8 & a);
// Quantizes x[T][in] (in a multiple of 32) into `a`.
cudaError_t actq8_quantize(const float * x, int64_t in, int64_t T, const ActQ8 & a, cudaStream_t s);

// ggml type ids with a __dp4a kernel (Q4_K, Q5_K, Q6_K, Q3_K, IQ4_NL, IQ3_S, IQ4_XS). Q8_0 keeps its own kernel in q8_gemv.cu.
bool gemv_dp4a_supported(int type);
// y[t][o] = sum_i W[o][i] * x[t][i] with x taken from the quantized activations in `a`.
cudaError_t gemv_dp4a(int type, const uint8_t * W, int64_t in, int64_t out, const ActQ8 & a, int64_t T, float * y,
                      cudaStream_t stream);
