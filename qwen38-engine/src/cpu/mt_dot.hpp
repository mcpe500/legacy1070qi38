// Multi-token W4A8 dot products (AVX2, decision A in docs/AFK_STATUS.md). ggml's vec_dot re-reads and re-unpacks
// every weight block once per token, so an MTP verification batch of T tokens costs about T times a single decode
// step even though the weights are already in cache. Here each quantized weight block is unpacked once into int8
// (or int16) lanes and multiplied against all T activation rows; scales are applied once per unit in a float
// epilogue. The integer dot products are exact, so the only difference from vec_dot is the float accumulation
// order of the scale application (parity bar for this path: cosine >= 0.995 plus greedy token checks).
#pragma once
#include <cstdint>

#include "qmat.hpp"

// True when `type` has a multi-token kernel (Q4_K, Q5_K, Q6_K, IQ4_XS, IQ4_NL, Q8_0) and AVX2 is available.
bool mt_dot_supported(int wtype);

// True when qmatmul should take the multi-token path for this batch (env QWEN_MT_DOT=0 forces the old path).
bool mt_dot_use(int wtype, int64_t T);

// Same contract as the vec_dot loop in qmatmul: act holds the libggml activation quantization of the T rows
// (as produced by from_float of the type's vec_dot_type, T rows of `arow` bytes), y is [T][out].
void mt_dot(Pool & pool, const QMat & W, const uint8_t * act, int64_t arow, int64_t T, float * y, int64_t grain);
