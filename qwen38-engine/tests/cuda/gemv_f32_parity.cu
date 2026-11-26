// Synthetic parity test (no model file needed) for F32 and BF16 weight matrices in gemv_fp32 / gemv_fp32_pair, the packings
// of ssm_alpha / ssm_beta in Ternary Bonsai 2.5 (F32) and Bonsai 2 (BF16). Shape 5120 x 48 as in the model.
// Checks: F32 and BF16 single GEMV against a double-precision reference, pair launch bit-identical to two single launches.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "gemv.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return 2; } } while (0)

static uint16_t to_bf16(float f) {  // round to nearest even
    uint32_t b; std::memcpy(&b, &f, 4);
    b += 0x7FFF + ((b >> 16) & 1);
    return static_cast<uint16_t>(b >> 16);
}
static float from_bf16(uint16_t h) { const uint32_t b = static_cast<uint32_t>(h) << 16; float f; std::memcpy(&f, &b, 4); return f; }

int main() {
    const int64_t in = 5120, out = 48;
    int fails = 0;
    for (int type : {static_cast<int>(GT_F32), static_cast<int>(GT_BF16)}) {
        const char * name = type == GT_F32 ? "F32" : "BF16";
        const int64_t rowb = gemv_row_bytes(type, in);
        if (rowb != in * (type == GT_F32 ? 4 : 2) || !gemv_fp32_supported(type)) { std::printf("%s: row bytes / supported mismatch\n", name); ++fails; continue; }
        std::mt19937 rng(7 + type);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> wf[2];
        std::vector<uint8_t> wb[2];
        for (int m = 0; m < 2; ++m) {
            wf[m].resize(static_cast<size_t>(in * out));
            wb[m].resize(static_cast<size_t>(rowb * out));
            for (size_t i = 0; i < wf[m].size(); ++i) {
                float v = nd(rng) * 0.05f;
                if (type == GT_BF16) { const uint16_t h = to_bf16(v); v = from_bf16(h); std::memcpy(&wb[m][2 * i], &h, 2); }
                else std::memcpy(&wb[m][4 * i], &v, 4);
                wf[m][i] = v;
            }
        }
        for (int T : {1, 3, 8, 11, 64}) {
            std::vector<float> x(static_cast<size_t>(T * in));
            for (float & v : x) v = nd(rng);
            uint8_t * dW[2]; float *dx, *dy[2], *dp[2];
            CK(cudaMalloc(&dx, sizeof(float) * x.size()));
            CK(cudaMemcpy(dx, x.data(), sizeof(float) * x.size(), cudaMemcpyHostToDevice));
            for (int m = 0; m < 2; ++m) {
                CK(cudaMalloc(&dW[m], wb[m].size()));
                CK(cudaMemcpy(dW[m], wb[m].data(), wb[m].size(), cudaMemcpyHostToDevice));
                CK(cudaMalloc(&dy[m], sizeof(float) * T * out));
                CK(cudaMalloc(&dp[m], sizeof(float) * T * out));
                CK(gemv_fp32(type, dW[m], in, out, dx, T, dy[m], 0));
            }
            CK(gemv_fp32_pair(type, dW[0], dW[1], in, out, dx, T, dp[0], dp[1], 0));
            CK(cudaDeviceSynchronize());
            for (int m = 0; m < 2; ++m) {
                std::vector<float> y(static_cast<size_t>(T * out)), yp(y.size());
                CK(cudaMemcpy(y.data(), dy[m], sizeof(float) * y.size(), cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(yp.data(), dp[m], sizeof(float) * yp.size(), cudaMemcpyDeviceToHost));
                double e2 = 0, r2 = 0;
                for (int t = 0; t < T; ++t) {
                    for (int64_t o = 0; o < out; ++o) {
                        double a = 0;
                        for (int64_t i = 0; i < in; ++i) a += static_cast<double>(wf[m][o * in + i]) * x[t * in + i];
                        const double d = y[t * out + o] - a;
                        e2 += d * d; r2 += a * a;
                    }
                }
                const double rel = std::sqrt(e2 / r2);
                const bool same = std::memcmp(y.data(), yp.data(), sizeof(float) * y.size()) == 0;
                const bool ok = rel < 1e-5 && same;
                if (!ok) ++fails;
                std::printf("%-4s T=%-3d mat=%d rel_rms=%.3e pair_bit_identical=%d %s\n", name, T, m, rel, same ? 1 : 0, ok ? "" : "<-- FAIL");
            }
            for (int m = 0; m < 2; ++m) { cudaFree(dW[m]); cudaFree(dy[m]); cudaFree(dp[m]); }
            cudaFree(dx);
        }
    }
    std::printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}
