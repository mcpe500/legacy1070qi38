#include <cstring>

#include "cpu_ops.hpp"

#include <immintrin.h>

#include <cmath>
#include <stdexcept>

#include "ggml.h"

std::vector<float> dequant_tensor(const GgufIndex & ix, const std::string & name) {
    const GgufTensor & t = ix.get(name);
    const auto gt = static_cast<enum ggml_type>(t.type);
    int64_t n = 1;
    for (int64_t d : t.ne) n *= d;
    std::vector<float> out(static_cast<size_t>(n));
    if (t.type == GGML_TYPE_F32) {
        std::memcpy(out.data(), ix.data(t), static_cast<size_t>(n) * sizeof(float));
        return out;
    }
    const ggml_type_traits * tr = ggml_get_type_traits(gt);
    if (!tr || !tr->to_float) throw std::runtime_error("dequant: no to_float for " + name);
    tr->to_float(ix.data(t), out.data(), n);
    return out;
}

void matvec(const float * W, int64_t in, int64_t out, const float * x, float * y) {
    for (int64_t o = 0; o < out; ++o) {
        const float * row = W + o * in;
        float acc = 0.0f;
        for (int64_t i = 0; i < in; ++i) acc += row[i] * x[i];
        y[o] = acc;
    }
}

void rmsnorm(const float * x, int64_t n, const float * w, float eps, float * y) {
    double ss = 0.0;
    for (int64_t i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
    const float scale = 1.0f / std::sqrt(static_cast<float>(ss / static_cast<double>(n)) + eps);
    for (int64_t i = 0; i < n; ++i) y[i] = x[i] * scale * (w ? w[i] : 1.0f);
}

void l2norm_inplace(float * x, int64_t n, float eps) {
    double ss = 0.0;
    for (int64_t i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
    const float scale = 1.0f / std::fmax(std::sqrt(static_cast<float>(ss)), eps);
    for (int64_t i = 0; i < n; ++i) x[i] *= scale;
}

float sigmoidf_(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float silu_(float x) { return x / (1.0f + std::exp(-x)); }
float softplus_(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

float dotf(const float * a, const float * b, int64_t n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps(), s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
    for (int64_t i = 0; i < n; i += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
    }
    s0 = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(s0), _mm256_extractf128_ps(s0, 1));
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}
