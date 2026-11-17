// Ternary path for Ternary Bonsai 2 27B on sm_61 (prepared, not yet wired into the engine; see docs/AFK_STATUS.md).
//
// Weights: GGUF Q2_0 (ggml type 42), blocks of 64 weights = fp16 scale d + 16 bytes of 2-bit codes (element j at bits 2j of the
// block; value = (code - 1) * d, codes 0..2 for ternary {-1, 0, +1}). Rows are 18-byte blocks, so loads are 2-byte aligned.
// Activations: the weights are stored in a Hadamard-rotated basis (prism.hadamard.*), so every input vector is first rotated:
//   x' = H_1024 (s * x) blockwise over the input dimension (normalized Sylvester-Walsh-Hadamard, block 1024, explicit signs s).
// GEMV: x' is quantized to Q8_1 (ActQ8, blocks of 32) and dotted with __dp4a. The 2-bit codes are unpacked with lop3.b32 into
// four int8x4 words whose lanes hold weights {k, k+4, k+8, k+12}; the matching activation lanes are gathered with prmt.b32
// (__byte_perm), so no table lookup and no per-lane shift is needed. Per activation block b of 32 values:
//   sum_j (code_j - 1) d q_j dx_b = d * (dx_b * sum_j code_j q_j - sx_b),  sx_b = dx_b * sum_j q_j.
// FP32 accumulation, fp16 scales decoded with integer bit operations: no half-precision arithmetic (SASS gate).
#pragma once
#include <cuda_runtime.h>

#include <cstdint>

#include "dp4a_gemv.cuh"   // ActQ8

constexpr int kGtQ2_0 = 42;          // ggml type id of Q2_0
constexpr int kQ2_0BlockBytes = 18;  // fp16 + 64 x 2 bit

// y[t][i] = (1/32) * H_1024 (signs * x[t])  over each 1024-value block of a row of `width` values (width % 1024 == 0).
// signs: device [width] of +1/-1 (FP32). x may alias y.
cudaError_t ternary_rotate(const float * x, const float * signs, float * y, int width, int T, cudaStream_t s);

// y[t][o] = sum_i W[o][i] * x'[t][i] with x' quantized in `a` (in % 64 == 0).
cudaError_t gemv_q2_0_dp4a(const uint8_t * W, int64_t in, int64_t out, const ActQ8 & a, int64_t T, float * y, cudaStream_t s);
