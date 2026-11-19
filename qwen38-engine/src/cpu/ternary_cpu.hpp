// CPU path for the Ternary Bonsai weight formats (Q2_0 type 42, PTQ1_0 type 143) and the Hadamard
// activation rotation of the prism.hadamard.* metadata (block 1024, explicit signs, normalized
// Sylvester-Walsh-Hadamard: x' = H_1024 (s * x) / 32 over each 1024-value block of the input row).
// No weight is dequantized to FP32: the base-3 / 2-bit codes are decoded to int8 in registers and dotted
// in FP32 against the activation (the accumulator that leaves a function is FP32; skills pascal-sm61-cuda
// rule 1 keeps the arithmetic itself in FP32).
#pragma once
#include <cstdint>
#include <cstddef>

#include "pool.hpp"
#include "ternary_spec.hpp"
#include "qmat.hpp"

// y[t][o] = sum_i W[o][i] * x[t][i] with x already rotated into the weight's basis (caller rotates).
// Parallel over output rows like qmatmul. Throws for an unsupported type.
void ternary_matmul(Pool & pool, const QMat & W, const float * x, int64_t T, float * y);

// Rotates x in place: x' = H_1024 (signs * x) / 32 over each 1024-value block. width % 1024 == 0.
// signs has `width` entries of +1/-1. Same transform as the GPU ternary_rotate kernel (bit-identical order).
// This is the forward transform of the prism.hadamard.* metadata (signs first, then the butterflies).
void hadamard_rotate(float * x, const float * signs, int64_t width, int64_t T);

// Inverse of hadamard_rotate (H first, then signs), the transform of a Hadamard-latent lookup table such as
// token_embd: the reference applies it after the row lookup to restore the primal basis.
void hadamard_unrotate(float * x, const float * signs, int64_t width, int64_t T);

// Decodes one row of a ternary weight into out[W.in] (embedding lookup of token_embd, stored in the
// rotated basis: the caller applies hadamard_rotate to get the plain embedding).
void ternary_row_to_f32(const QMat & W, int64_t row, float * out);