// Test of the prepared ternary path (src/cuda/ternary_gemv.cu) with synthetic Q2_0 weights:
//  1. our decode of a Q2_0 row equals libggml's to_float for type 42 (bit-exact),
//  2. ternary_rotate equals a double-precision Sylvester-Hadamard reference H_1024 (s * x) / 32,
//  3. gemv_q2_0_dp4a equals a double reference that uses the same quantized activations (rounding-level error), and is close
//     to the FP32-activation reference (the error of quantizing the activation to 8 bit),
//  4. results do not depend on the batch size T (each token uses the same order of operations).
// Usage: ternary_test [seed]
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "dp4a_gemv.cuh"
#include "f16bits.hpp"
#include "ggml.h"
#include "ternary_gemv.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s at %d\n", cudaGetErrorString(e_), __LINE__); std::exit(2); } } while (0)

namespace {

int fails = 0;
void check(const char * what, bool ok, double v, double lim) {
    std::printf("%s %-58s %.3e (limit %.1e)\n", ok ? "PASS" : "FAIL", what, v, lim);
    if (!ok) ++fails;
}

double rel_rms(const std::vector<double> & ref, const std::vector<float> & got) {
    double e = 0, n = 0;
    for (size_t i = 0; i < ref.size(); ++i) { e += (got[i] - ref[i]) * (got[i] - ref[i]); n += ref[i] * ref[i]; }
    return std::sqrt(e / std::max(n, 1e-300));
}

void fwht_ref(std::vector<double> & v, size_t off) {   // in-place, unnormalized, natural (Sylvester) order
    for (size_t h = 1; h < 1024; h <<= 1)
        for (size_t i = 0; i < 1024; i += 2 * h)
            for (size_t j = i; j < i + h; ++j) {
                const double a = v[off + j], b = v[off + j + h];
                v[off + j] = a + b;
                v[off + j + h] = a - b;
            }
}

}  // namespace

int main(int argc, char ** argv) {
    const unsigned seed = argc > 1 ? static_cast<unsigned>(std::atoi(argv[1])) : 7u;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> code(0, 2);
    std::uniform_real_distribution<float> sc(0.004f, 0.03f);

    struct Shape { int in, out; };
    const Shape shapes[] = {{5120, 1024}, {6144, 512}, {17408, 256}};
    for (const Shape & sh : shapes) {
        const int in = sh.in, out = sh.out, T = 11;
        const int nblk = in / 64;
        // ---- weights
        std::vector<uint8_t> W(static_cast<size_t>(out) * nblk * kQ2_0BlockBytes);
        std::vector<float> Wf(static_cast<size_t>(out) * in);
        for (int o = 0; o < out; ++o) {
            for (int b = 0; b < nblk; ++b) {
                uint8_t * blk = W.data() + (static_cast<size_t>(o) * nblk + b) * kQ2_0BlockBytes;
                const uint16_t h = static_cast<uint16_t>(f32_to_f16(sc(rng)));
                std::memcpy(blk, &h, 2);
                std::memset(blk + 2, 0, 16);
                for (int j = 0; j < 64; ++j) blk[2 + j / 4] |= static_cast<uint8_t>(code(rng) << (2 * (j % 4)));
            }
        }
        // 1. decode == libggml to_float (type 42)
        const ggml_type_traits * tr = ggml_get_type_traits(static_cast<ggml_type>(kGtQ2_0));
        size_t mism = 0;
        for (int o = 0; o < out; ++o) {
            tr->to_float(W.data() + static_cast<size_t>(o) * nblk * kQ2_0BlockBytes, Wf.data() + static_cast<size_t>(o) * in, in);
            for (int b = 0; b < nblk; ++b) {
                const uint8_t * blk = W.data() + (static_cast<size_t>(o) * nblk + b) * kQ2_0BlockBytes;
                const float d = f16_to_f32(static_cast<uint32_t>(blk[0]) | (static_cast<uint32_t>(blk[1]) << 8));
                for (int j = 0; j < 64; ++j) {
                    const int c = (blk[2 + j / 4] >> (2 * (j % 4))) & 3;
                    if ((c - 1) * d != Wf[static_cast<size_t>(o) * in + b * 64 + j]) ++mism;
                }
            }
        }
        std::printf("--- in %d out %d T %d\n", in, out, T);
        check("Q2_0 decode vs libggml to_float (mismatching values)", mism == 0, static_cast<double>(mism), 0.0);

        // ---- activations and signs
        std::vector<float> x(static_cast<size_t>(T) * in), s(in);
        for (auto & v : x) v = nd(rng);
        for (auto & v : s) v = (rng() & 1) ? 1.0f : -1.0f;
        std::vector<double> xr(x.size());
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < in; ++i) xr[static_cast<size_t>(t) * in + i] = static_cast<double>(x[static_cast<size_t>(t) * in + i]) * s[i];
        for (int t = 0; t < T; ++t)
            for (int b = 0; b < in / 1024; ++b) fwht_ref(xr, static_cast<size_t>(t) * in + b * 1024);
        for (auto & v : xr) v /= 32.0;

        float *dx, *ds, *dy;
        uint8_t * dW;
        CK(cudaMalloc(&dx, x.size() * 4));
        CK(cudaMalloc(&ds, s.size() * 4));
        CK(cudaMalloc(&dy, static_cast<size_t>(T) * out * 4));
        CK(cudaMalloc(&dW, W.size()));
        CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(ds, s.data(), s.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dW, W.data(), W.size(), cudaMemcpyHostToDevice));

        // 2. rotation (in place)
        CK(ternary_rotate(dx, ds, dx, in, T, 0));
        std::vector<float> xg(x.size());
        CK(cudaMemcpy(xg.data(), dx, x.size() * 4, cudaMemcpyDeviceToHost));
        check("ternary_rotate vs double Sylvester-Hadamard (rel rms)", rel_rms(xr, xg) < 1e-6, rel_rms(xr, xg), 1e-6);

        // 3. GEMV
        ActQ8 a;
        CK(actq8_alloc(a, T, in));
        CK(actq8_quantize(dx, in, T, a, 0));
        CK(gemv_q2_0_dp4a(dW, in, out, a, T, dy, 0));
        std::vector<float> yg(static_cast<size_t>(T) * out);
        CK(cudaMemcpy(yg.data(), dy, yg.size() * 4, cudaMemcpyDeviceToHost));
        std::vector<int8_t> q(static_cast<size_t>(T) * in);
        std::vector<float> dq(static_cast<size_t>(T) * in / 32);
        CK(cudaMemcpy(q.data(), a.qx, q.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(dq.data(), a.dx, dq.size() * 4, cudaMemcpyDeviceToHost));
        std::vector<double> ysame(yg.size()), yf32(yg.size());
        for (int t = 0; t < T; ++t)
            for (int o = 0; o < out; ++o) {
                double s1 = 0, s2 = 0;
                const float * wr = Wf.data() + static_cast<size_t>(o) * in;
                for (int i = 0; i < in; ++i) {
                    s1 += static_cast<double>(wr[i]) * q[static_cast<size_t>(t) * in + i] * dq[(static_cast<size_t>(t) * in + i) / 32];
                    s2 += static_cast<double>(wr[i]) * xr[static_cast<size_t>(t) * in + i];
                }
                ysame[static_cast<size_t>(t) * out + o] = s1;
                yf32[static_cast<size_t>(t) * out + o] = s2;
            }
        check("gemv_q2_0_dp4a vs same-activation double reference", rel_rms(ysame, yg) < 2e-6, rel_rms(ysame, yg), 2e-6);
        check("gemv_q2_0_dp4a vs FP32-activation reference (8-bit act)", rel_rms(yf32, yg) < 1.5e-2, rel_rms(yf32, yg), 1.5e-2);

        // 4. batch independence: T = 1 for token 5 equals row 5 of the T = 11 run
        ActQ8 a1;
        CK(actq8_alloc(a1, 1, in));
        CK(actq8_quantize(dx + static_cast<size_t>(5) * in, in, 1, a1, 0));
        CK(gemv_q2_0_dp4a(dW, in, out, a1, 1, dy, 0));
        std::vector<float> y1(out);
        CK(cudaMemcpy(y1.data(), dy, out * 4, cudaMemcpyDeviceToHost));
        const bool same = std::memcmp(y1.data(), yg.data() + static_cast<size_t>(5) * out, out * 4) == 0;
        check("T=1 result bit-identical to the same row of T=11", same, same ? 0.0 : 1.0, 0.0);
        actq8_free(a);
        actq8_free(a1);
        cudaFree(dx); cudaFree(ds); cudaFree(dy); cudaFree(dW);
    }
    std::printf("ternary_test: %s (%d failed)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
