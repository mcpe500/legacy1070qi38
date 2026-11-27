// Gate of the PTQ1_0 lane-per-group decode kernel (gemv_ptq1_0_dp4a at T <= 5, QWEN_PTQ1_DEC=4, the default):
//   1. against an FP64 reference built from the decoded ternary digits (kPtq1Code) and the same quantized activations:
//      rel_rms < 1e-6 and the same argmax per token;
//   2. lossless across batch sizes: row t of a T-token call is bit-identical to the T = 1 call on token t (MTP verify
//      rows must equal decode rows; covers the shared-memory and the global-memory activation paths of in = 17408);
// for every Bonsai input width (5120, 6144, 17408), output counts that are not multiples of the 32 rows per block, and
// T = 1..5. Weights are random canonical PTQ1_0 bytes (inverse of kPtq1Code; tail bytes carry 3 * c4). Exit 0 only if all pass.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "dp4a_gemv.cuh"
#include "ternary_gemv.cuh"
#include "ternary_spec.hpp"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

int main() {
    setenv("QWEN_PTQ1_DEC", "4", 1);
    ternary_ptq1_init();
    uint8_t inv[243];
    for (int b = 0; b < 256; ++b) if (ternary::kPtq1Code[b] != 255) inv[ternary::kPtq1Code[b]] = static_cast<uint8_t>(b);
    std::mt19937 rng(17);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> dig(0, 2);
    const int widths[] = {5120, 6144, 17408}, outs[] = {1027, 4093, 17408};
    ActQ8 a;
    CK(actq8_alloc(a, 5, 17408));
    int fails = 0, cases = 0;
    double worst = 0;
    for (int in : widths) {
        const int G = in / 128;
        for (int out : outs) {
            std::vector<uint8_t> w(static_cast<size_t>(out) * G * 28);
            std::vector<int8_t> dg(static_cast<size_t>(out) * in);
            std::vector<float> dsc(static_cast<size_t>(out) * G);
            for (int o = 0; o < out; ++o)
                for (int g = 0; g < G; ++g) {
                    int8_t * t = &dg[static_cast<size_t>(o) * in + g * 128];
                    for (int i = 0; i < 128; ++i) t[i] = static_cast<int8_t>(dig(rng));
                    uint8_t * p = &w[(static_cast<size_t>(o) * G + g) * 28];
                    for (int j = 0; j < 16; ++j) { int c = 0; for (int s = 0; s < 5; ++s) c = c * 3 + t[j + 16 * s]; p[j] = inv[c]; }
                    for (int k = 0; k < 8; ++k) { int c = 0; for (int s = 0; s < 5; ++s) c = c * 3 + t[80 + k + 8 * s]; p[16 + k] = inv[c]; }
                    for (int k = 0; k < 2; ++k) { int c = 0; for (int s = 0; s < 4; ++s) c = c * 3 + t[120 + k + 2 * s]; p[24 + k] = inv[3 * c]; }
                    const int man = static_cast<int>(rng() & 0x3FF), ex = 7 + static_cast<int>(rng() % 6);   // 2^-8 .. 2^-3
                    const uint16_t h = static_cast<uint16_t>((ex << 10) | man);
                    p[26] = static_cast<uint8_t>(h & 0xFF);
                    p[27] = static_cast<uint8_t>(h >> 8);
                    dsc[static_cast<size_t>(o) * G + g] = std::ldexp(1.0f + man / 1024.0f, ex - 15);
                }
            uint8_t * dw;
            CK(cudaMalloc(&dw, w.size()));
            CK(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
            std::vector<float> x(static_cast<size_t>(5) * in);
            for (float & v : x) v = nd(rng);
            float *dxin, *y1, *yT;
            CK(cudaMalloc(&dxin, x.size() * 4));
            CK(cudaMalloc(&y1, sizeof(float) * 5 * out));
            CK(cudaMalloc(&yT, sizeof(float) * 5 * out));
            CK(cudaMemcpy(dxin, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
            CK(actq8_quantize(dxin, in, 5, a, 0));
            std::vector<int8_t> q(static_cast<size_t>(5) * in);
            std::vector<float> qd(static_cast<size_t>(5) * in / 32);
            CK(cudaMemcpy(q.data(), a.qx, q.size(), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(qd.data(), a.dx, qd.size() * 4, cudaMemcpyDeviceToHost));
            // per-token T = 1 rows: token t as its own one-row activation (a copy of row t)
            std::vector<float> single(static_cast<size_t>(5) * out);
            ActQ8 a1;
            CK(actq8_alloc(a1, 1, 17408));
            for (int t = 0; t < 5; ++t) {
                CK(cudaMemcpy(a1.qx, a.qx + static_cast<size_t>(t) * in, in, cudaMemcpyDeviceToDevice));
                CK(cudaMemcpy(a1.dx, a.dx + static_cast<size_t>(t) * (in / 32), sizeof(float) * (in / 32), cudaMemcpyDeviceToDevice));
                CK(cudaMemcpy(a1.sx, a.sx + static_cast<size_t>(t) * (in / 32), sizeof(float) * (in / 32), cudaMemcpyDeviceToDevice));
                CK(gemv_ptq1_0_dp4a(dw, in, out, a1, 1, y1, 0));
                CK(cudaMemcpy(single.data() + static_cast<size_t>(t) * out, y1, sizeof(float) * out, cudaMemcpyDeviceToHost));
            }
            actq8_free(a1);
            // FP64 reference for every token
            std::vector<double> ref(static_cast<size_t>(5) * out);
            for (int t = 0; t < 5; ++t)
                for (int o = 0; o < out; ++o) {
                    double r = 0;
                    for (int g = 0; g < G; ++g) {
                        double sg = 0;
                        for (int b = 0; b < 4; ++b) {
                            long long si = 0;
                            const int8_t * wd = &dg[static_cast<size_t>(o) * in + g * 128 + b * 32];
                            const int8_t * qq = &q[static_cast<size_t>(t) * in + g * 128 + b * 32];
                            for (int i = 0; i < 32; ++i) si += static_cast<long long>(wd[i] - 1) * qq[i];
                            sg += static_cast<double>(qd[static_cast<size_t>(t) * (in / 32) + g * 4 + b]) * si;
                        }
                        r += dsc[static_cast<size_t>(o) * G + g] * sg;
                    }
                    ref[static_cast<size_t>(t) * out + o] = r;
                }
            for (int T = 1; T <= 5; ++T) {
                CK(gemv_ptq1_0_dp4a(dw, in, out, a, T, yT, 0));
                std::vector<float> h(static_cast<size_t>(T) * out);
                CK(cudaMemcpy(h.data(), yT, h.size() * 4, cudaMemcpyDeviceToHost));
                double e2 = 0, r2 = 0;
                bool arg_ok = true;
                for (int t = 0; t < T; ++t) {
                    int am = 0, amr = 0;
                    for (int o = 0; o < out; ++o) {
                        const double v = h[static_cast<size_t>(t) * out + o], r = ref[static_cast<size_t>(t) * out + o];
                        e2 += (v - r) * (v - r);
                        r2 += r * r;
                        if (v > h[static_cast<size_t>(t) * out + am]) am = o;
                        if (r > ref[static_cast<size_t>(t) * out + amr]) amr = o;
                    }
                    arg_ok = arg_ok && am == amr;
                }
                const double rel = std::sqrt(e2 / r2);
                const bool same = std::memcmp(h.data(), single.data(), h.size() * 4) == 0;
                const bool ok = rel < 1e-6 && arg_ok && same;
                worst = std::fmax(worst, rel);
                ++cases;
                if (!ok) ++fails;
                std::printf("%s in=%5d out=%5d T=%d | rel_rms %.3e | argmax %s | rows == T=1 rows: %s\n", ok ? "PASS" : "FAIL", in, out, T, rel,
                            arg_ok ? "same" : "DIFF", same ? "bit-identical" : "DIFFER");
            }
            CK(cudaFree(dw)); CK(cudaFree(dxin)); CK(cudaFree(y1)); CK(cudaFree(yT));
        }
    }
    actq8_free(a);
    std::printf("ptq1_lpg_test: %d cases, %d fail, worst rel_rms %.3e -> %s\n", cases, fails, worst, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
