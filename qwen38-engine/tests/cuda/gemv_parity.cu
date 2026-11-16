// GPU GEMV parity on real tensors from layers 0-23 of Qwen3.8-27B-UD-Q4_K_M, one tensor per quantization type.
// Reference: weights dequantized by libggml (to_float) and multiplied in double precision with the same FP32 input.
// This isolates the correctness of the GPU dequantization from the effect of any activation quantization.
// Also reports the cosine against the CPU path (ggml vec_dot with Q8_K activations) as an informational number.
// Usage: gemv_parity <GGUF> [only-type-id]
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "ggml.h"
#include "dp4a_gemv.cuh"
#include "gemv.cuh"
#include "gguf_index.hpp"
#include "pool.hpp"
#include "qmat.hpp"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

struct Case { const char * name; int type; double tol_rel_rms; };

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <GGUF> [type-id]\n", argv[0]); return 2; }
    const int only = argc > 2 ? std::atoi(argv[2]) : -1;
    qmat_init();
    GgufIndex ix;
    ix.open(argv[1]);
    Pool pool(4);

    const Case cases[] = {
        {"blk.1.attn_qkv.weight", GT_Q4_K, 1e-5},
        {"blk.0.attn_qkv.weight", GT_Q5_K, 1e-5},
        {"blk.1.ssm_out.weight", GT_Q6_K, 1e-5},
        {"blk.0.ffn_up.weight", GT_Q3_K, 1e-5},
        {"blk.0.ffn_gate.weight", GT_IQ4_XS, 1e-5},
        {"blk.1.ffn_down.weight", GT_IQ4_NL, 1e-5},
        {"blk.14.ffn_up.weight", GT_IQ3_S, 1e-5},
        {"blk.14.ffn_down.weight", GT_IQ3_S, 1e-5},
        {"blk.0.ssm_alpha.weight", GT_Q8_0, 1e-5},
    };
    const int T = 11;  // one full tile of 8 tokens plus a partial tile of 3
    int fails = 0;
    std::printf("%-26s %-8s %7s %7s  %10s %10s  %s   %s\n", "tensor", "type", "in", "out", "rel_rms", "max_rel", "cos_vs_cpu_vecdot",
                "dp4a: rel_rms vs same-activation ref | vs FP32-activation ref");
    for (const Case & c : cases) {
        if (only >= 0 && c.type != only) continue;
        const GgufTensor & t = ix.get(c.name);
        if (static_cast<int>(t.type) != c.type) { std::printf("%s: type %u, expected %d\n", c.name, t.type, c.type); ++fails; continue; }
        const int64_t in = t.ne[0], out = t.ne[1];
        const int64_t rowb = gemv_row_bytes(c.type, in);
        if (rowb != static_cast<int64_t>(ggml_row_size(static_cast<ggml_type>(c.type), in))) { std::printf("%s: row bytes mismatch\n", c.name); ++fails; continue; }

        std::mt19937 rng(100 + c.type);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> x(static_cast<size_t>(T * in));
        for (float & v : x) v = nd(rng);

        // Reference: libggml dequantization, double precision products.
        std::vector<float> wf(static_cast<size_t>(in));
        const ggml_type_traits * tr = ggml_get_type_traits(static_cast<ggml_type>(c.type));
        std::vector<double> ref(static_cast<size_t>(T * out), 0.0);
        const uint8_t * wp = ix.data(t);
        for (int64_t o = 0; o < out; ++o) {
            tr->to_float(wp + o * rowb, wf.data(), in);
            for (int tk = 0; tk < T; ++tk) {
                double a = 0;
                const float * xr = &x[static_cast<size_t>(tk * in)];
                for (int64_t i = 0; i < in; ++i) a += static_cast<double>(wf[i]) * xr[i];
                ref[static_cast<size_t>(tk * out + o)] = a;
            }
        }

        uint8_t * dW; float * dx; float * dy;
        CK(cudaMalloc(&dW, static_cast<size_t>(out * rowb)));
        CK(cudaMalloc(&dx, sizeof(float) * x.size()));
        CK(cudaMalloc(&dy, sizeof(float) * static_cast<size_t>(T * out)));
        CK(cudaMemcpy(dW, wp, static_cast<size_t>(out * rowb), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dx, x.data(), sizeof(float) * x.size(), cudaMemcpyHostToDevice));
        CK(gemv_fp32(c.type, dW, in, out, dx, T, dy, 0));
        CK(cudaDeviceSynchronize());
        std::vector<float> y(static_cast<size_t>(T * out));
        CK(cudaMemcpy(y.data(), dy, sizeof(float) * y.size(), cudaMemcpyDeviceToHost));

        double e2 = 0, r2 = 0, maxrel = 0;
        for (size_t i = 0; i < y.size(); ++i) {
            const double d = y[i] - ref[i];
            e2 += d * d;
            r2 += ref[i] * ref[i];
        }
        const double rms = std::sqrt(r2 / static_cast<double>(y.size()));
        for (size_t i = 0; i < y.size(); ++i) maxrel = std::fmax(maxrel, std::fabs(y[i] - ref[i]) / (rms + 1e-30));
        const double rel_rms = std::sqrt(e2 / r2);

        // Informational: CPU path (ggml vec_dot with quantized activations).
        QMat qm = make_qmat(ix, c.name);
        std::vector<float> ycpu(static_cast<size_t>(T * out));
        std::vector<uint8_t> act(qmat_act_bytes(c.type, in, T));
        qmatmul(pool, qm, x.data(), T, ycpu.data(), act.data());
        double dot = 0, na = 0, nb = 0;
        for (size_t i = 0; i < y.size(); ++i) { dot += static_cast<double>(y[i]) * ycpu[i]; na += static_cast<double>(y[i]) * y[i]; nb += static_cast<double>(ycpu[i]) * ycpu[i]; }

        const bool ok = rel_rms < c.tol_rel_rms;
        if (!ok) ++fails;
        std::printf("%-26s %-8s %7lld %7lld  %10.3e %10.3e  %.6f %s\n", c.name, ggml_type_name(static_cast<ggml_type>(c.type)),
                    static_cast<long long>(in), static_cast<long long>(out), rel_rms, maxrel, dot / std::sqrt(na * nb),
                    ok ? "" : "<-- FAIL");
        // dp4a path (Q8_1 activations). Reference 1 quantizes the activations with the same rule on the host and multiplies
        // in double precision, so it checks the integer kernel exactly; reference 2 is the FP32-activation result above.
        if (gemv_dp4a_supported(c.type)) {
            ActQ8 act;
            CK(actq8_alloc(act, T, in));
            CK(actq8_quantize(dx, in, T, act, 0));
            float * dy2;
            CK(cudaMalloc(&dy2, sizeof(float) * static_cast<size_t>(T * out)));
            CK(gemv_dp4a(c.type, dW, in, out, act, T, dy2, 0));
            CK(cudaDeviceSynchronize());
            std::vector<float> y2(static_cast<size_t>(T * out));
            CK(cudaMemcpy(y2.data(), dy2, sizeof(float) * y2.size(), cudaMemcpyDeviceToHost));
            std::vector<double> xq(x.size());
            for (int tk = 0; tk < T; ++tk) {
                for (int64_t b = 0; b < in / 32; ++b) {
                    const float * xb = &x[static_cast<size_t>(tk * in + b * 32)];
                    float amax = 0.0f;
                    for (int l = 0; l < 32; ++l) amax = std::fmax(amax, std::fabs(xb[l]));
                    const float d = amax / 127.0f, inv = d > 0.0f ? 1.0f / d : 0.0f;
                    for (int l = 0; l < 32; ++l) {
                        int q = static_cast<int>(std::nearbyint(xb[l] * inv));
                        q = q < -127 ? -127 : (q > 127 ? 127 : q);
                        xq[static_cast<size_t>(tk * in + b * 32 + l)] = static_cast<double>(q) * static_cast<double>(d);
                    }
                }
            }
            double e_same = 0, r_same = 0, e_fp = 0, r_fp = 0;
            for (int64_t o = 0; o < out; ++o) {
                tr->to_float(wp + o * rowb, wf.data(), in);
                for (int tk = 0; tk < T; ++tk) {
                    double a = 0;
                    for (int64_t i = 0; i < in; ++i) a += static_cast<double>(wf[i]) * xq[static_cast<size_t>(tk * in + i)];
                    const double g = y2[static_cast<size_t>(tk * out + o)];
                    e_same += (g - a) * (g - a);
                    r_same += a * a;
                    const double rf = ref[static_cast<size_t>(tk * out + o)];
                    e_fp += (g - rf) * (g - rf);
                    r_fp += rf * rf;
                }
            }
            const double rel_same = std::sqrt(e_same / r_same), rel_fp = std::sqrt(e_fp / r_fp);
            const bool ok2 = rel_same < 1e-5;
            if (!ok2) ++fails;
            std::printf("%-26s %-8s %7s %7s  %56.3e | %.3e %s\n", "", "dp4a", "", "", rel_same, rel_fp, ok2 ? "" : "<-- FAIL");
            cudaFree(dy2);
            actq8_free(act);
        }
        cudaFree(dW); cudaFree(dx); cudaFree(dy);
    }
    std::printf("result: %s (%d failing)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
