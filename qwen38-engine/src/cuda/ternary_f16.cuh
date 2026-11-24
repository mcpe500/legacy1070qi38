// FP16 Tensor Core GEMM backend for PTQ1_0 ternary weights (--dump-targets fast path on sm_75+).
// The weights stay resident in their quantized VRAM format; one matrix at a time is dequantized to a small
// FP16 scratch (~170-335 MiB, in the same VRAM) and multiplied with the rotated FP16 activations through
// cublasGemmEx (FP16 in, CUBLAS_COMPUTE_32F accumulate). No BF16 anywhere; huge matrices (the 248320-row
// LM head) are processed in 32768-row chunks that OVERWRITE disjoint slices of the FP32 output (beta = 0
// on every chunk — the output is never a K-reduction across chunks, so accumulating there corrupts rows
// 32768..out-1 with stale buffer contents).
// Every kernel body is compiled only for __CUDA_ARCH__ >= 750: the sm_61 build (and its SASS gate of zero
// HFMA2/HMUL2/HADD2) never sees this code, and the full-path wrappers refuse to run on an older device.
// The pure-cuBLAS calls (ternary_f16_gemm_pre with host-dequantized weights) are device-agnostic and power
// the sm_61 regression test.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

struct TernaryF16;

// Runtime capability for the FULL path (dequant + rotate kernels): sm_75+ exactly (Turing T4 / Ampere /
// Ada). False on sm_61 Pascal and on sm_70 Volta (the kernel bodies are compiled only for >= 750, so a
// Volta device would run empty kernels and produce garbage).
bool ternary_f16_supported();

// Allocates the scratch (weight chunk + FP16 activations) and the cuBLAS handle. w16_bytes = the largest
// chunk payload (32768 * 5120 * 2 for this model, which also covers 17408 x 5120 whole-matrix dequants),
// xf_bytes = t_max * 17408 * 2 (the largest rotated activation). Returns null when unsupported/OOM.
TernaryF16 * ternary_f16_init(int64_t w16_bytes, int64_t xf_bytes);

// Same allocation on ANY CUDA device (>= 6.0): only cublas runs through it (ternary_f16_gemm_pre +
// ternary_f16_set_activations), so the sm_61 regression test can exercise the chunking/beta logic without
// a Turing card. Returns null on OOM/context failure.
TernaryF16 * ternary_f16_init_pre(int64_t w16_bytes, int64_t xf_bytes);
void ternary_f16_free(TernaryF16 * f);

// Loads FP16 activations [T][in] (as raw bits, little-endian) into the internal scratch: the test path of
// ternary_f16_gemm_pre (the engine uses ternary_f16_rotate instead). in must match the xf capacity shape.
cudaError_t ternary_f16_set_activations(TernaryF16 * f, const uint16_t * xf_host, int64_t T, int64_t in);

// x [T][in] FP32 plain-basis activations -> FP16 Hadamard-rotated activations (the same transform as the
// ternary_rotate_q8 kernel, minus the Q8 quantization). One launch per call. Full path: sm_75+ only.
cudaError_t ternary_f16_rotate(TernaryF16 * f, const float * x, const float * signs, int64_t in, int64_t T,
                               cudaStream_t s);

// Dequantizes a whole PTQ1_0 matrix W ([out][in] quantized) into dst16 ([out][in] FP16, row-major): the
// one-time fill of the LM head FP16 cache. One kernel launch (sm_75+).
cudaError_t ternary_f16_dequant(TernaryF16 * f, const uint8_t * W, int64_t in, int64_t out, uint16_t * dst16,
                                cudaStream_t s);

// y [T][out] FP32 = xf . W^T for a resident PTQ1_0 weight W ([out][in], ternary::kTypePtq1_0 byte layout).
// Dequantizes kChunkRows of W into the scratch before each cublas call. Full path: sm_75+ only.
cudaError_t ternary_f16_gemm(TernaryF16 * f, const uint8_t * W, int64_t in, int64_t out, int64_t T, float * y,
                             cudaStream_t s);

// Same GEMM from an ALREADY dequantized FP16 weight w16 ([out][in] row-major, e.g. the cached LM head).
// No >= 750-only kernel is touched: runs on any device through cublas. Every 32768-row chunk writes its
// disjoint y[r0..] slice with beta = 0.
cudaError_t ternary_f16_gemm_pre(TernaryF16 * f, const uint16_t * w16, int64_t in, int64_t out, int64_t T,
                                 float * y, cudaStream_t s);
