// FP16 Tensor Core GEMM backend for PTQ1_0 ternary weights (--dump-targets fast path on sm_75+).
// The weights stay resident in their quantized VRAM format; one matrix at a time is dequantized to a small
// FP16 scratch (~170-335 MiB, in the same VRAM) and multiplied with the rotated FP16 activations through
// cublasGemmEx (FP16 in, CUBLAS_COMPUTE_32F accumulate). No BF16 anywhere; huge matrices (the 248320-row
// LM head) are processed in 32768-row chunks accumulating into the FP32 output.
// Every kernel body is compiled only for __CUDA_ARCH__ >= 750: the sm_61 build (and its SASS gate of zero
// HFMA2/HMUL2) never sees this code, and the host wrappers refuse to launch on an older device.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

struct TernaryF16;

// Runtime capability: FP16 Tensor Core GEMM wants sm_75+ (Turing T4 / Ampere / Ada). False on sm_61 Pascal.
bool ternary_f16_supported();

// Allocates the scratch (weight chunk + FP16 activations) and the cuBLAS handle. w16_bytes = the largest
// chunk payload (32768 * 5120 * 2 for this model, which also covers 17408 x 5120 whole-matrix dequants),
// xf_bytes = t_max * 17408 * 2 (the largest rotated activation). Returns null when unsupported/OOM.
TernaryF16 * ternary_f16_init(int64_t w16_bytes, int64_t xf_bytes);
void ternary_f16_free(TernaryF16 * f);

// x [T][in] FP32 plain-basis activations -> FP16 Hadamard-rotated activations (the same transform as the
// ternary_rotate_q8 kernel, minus the Q8 quantization). One launch per call.
cudaError_t ternary_f16_rotate(TernaryF16 * f, const float * x, const float * signs, int64_t in, int64_t T,
                               cudaStream_t s);

// y [T][out] FP32 = xf . W^T for a resident PTQ1_0 weight W ([out][in], ternary::kTypePtq1_0 byte layout).
// Q2_0 is not supported here (the caller falls back to the dp4a path). Chunks W by 32768 rows.
cudaError_t ternary_f16_gemm(TernaryF16 * f, const uint8_t * W, int64_t in, int64_t out, int64_t T, float * y,
                             cudaStream_t s);
