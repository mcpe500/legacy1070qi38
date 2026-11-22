// Quantized GEMV on sm_61, option A: the weight stays in its GGUF block format in VRAM and each warp dequantizes
// 32 weights at a time to FP32 registers, then multiplies with the FP32 activations (FP32 FMA, FP32 accumulator).
// No half-precision arithmetic: fp16 scales are decoded with integer bit operations (src/common/f16bits.hpp).
// Q8_0 uses the __dp4a path in q8_gemv.cu; the others are upgraded to Q8_1 + __dp4a one by one (see dp4a_gemv.cuh).
#pragma once
#include <cuda_runtime.h>

#include <cstdint>

// ggml type ids of the quantizations found in layers 0-23 of Qwen3.8-27B-UD-Q4_K_M, plus BF16 (the
// ssm_alpha/ssm_beta packing of Ternary Bonsai).
enum GemvType : int {
    GT_Q8_0 = 8,
    GT_Q2_K = 10,
    GT_BF16 = 30,
    GT_Q3_K = 11,
    GT_Q4_K = 12,
    GT_Q5_K = 13,
    GT_Q6_K = 14,
    GT_IQ4_NL = 20,
    GT_IQ3_S = 21,
    GT_IQ4_XS = 23,
};

bool gemv_fp32_supported(int type);
int64_t gemv_row_bytes(int type, int64_t in);  // bytes of one weight row; 0 if the type is unsupported

// y[t][o] = sum_i W[o][i] * x[t][i], t < T, o < out. W: device pointer to `out` rows of gemv_row_bytes(type, in) bytes.
// x: device [T][in] FP32. y: device [T][out] FP32. `in` must be a multiple of 256 (32 for IQ4_NL and Q8_0).
cudaError_t gemv_fp32(int type, const uint8_t * W, int64_t in, int64_t out, const float * x, int64_t T, float * y,
                      cudaStream_t stream);
// Two GEMVs with the same input and shape in one launch (BF16 only: ssm_alpha / ssm_beta of Ternary Bonsai). Per row the
// arithmetic is that of gemv_fp32, so y1 and y2 are bit-identical to two separate calls.
cudaError_t gemv_fp32_pair(int type, const uint8_t * W1, const uint8_t * W2, int64_t in, int64_t out, const float * x, int64_t T,
                           float * y1, float * y2, cudaStream_t s);
