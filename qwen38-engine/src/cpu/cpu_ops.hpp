// FP32 CPU operators for the reference path. Every activation and state is FP32.
// Weight dequantization uses ggml's to_float (see docs/AFK_STATUS.md, decision D1).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "gguf_index.hpp"

// Dequantizes a whole GGUF tensor to FP32. Row-major over ne[1]; each row has ne[0] values.
std::vector<float> dequant_tensor(const GgufIndex & ix, const std::string & name);

// y[o] = sum_i W[o*in + i] * x[i], for o in [0, out).
void matvec(const float * W, int64_t in, int64_t out, const float * x, float * y);

// y = x / sqrt(mean(x^2) + eps) * w (w may be null for an unweighted norm).
void rmsnorm(const float * x, int64_t n, const float * w, float eps, float * y);

// y = x / max(||x||_2, eps), in place over n values.
void l2norm_inplace(float * x, int64_t n, float eps);

float sigmoidf_(float x);
float silu_(float x);
float softplus_(float x);  // matches ggml_compute_softplus_f32 (x > 20 returns x)

// Dot product of two FP32 vectors with AVX2 + FMA (4 accumulators of 8 lanes). n must be a multiple of 32.
float dotf(const float * a, const float * b, int64_t n);
