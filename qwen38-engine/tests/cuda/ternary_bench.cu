// Kernel timing of the ternary GEMVs (reference vs v2) on the Bonsai shapes, T = 1 (decode) and T = 16 (prefill tile).
// Random weights, the result is not checked here (tests/cuda/ternary_ab_test.cu does that). Prints us per call and GB/s.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "dp4a_gemv.cuh"
#include "ternary_gemv.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

int main() {
    ternary_ptq1_init();
    struct Shape { const char * name; int in, out; } shapes[] = {
        {"ffn_up 5120x17408", 5120, 17408}, {"ffn_down 17408x5120", 17408, 5120}, {"attn_qkv 5120x10240", 5120, 10240},
        {"attn_k 5120x1024", 5120, 1024}, {"lm_head 5120x248320", 5120, 248320}};
    ActQ8 a;
    CK(actq8_alloc(a, 16, 17408));
    std::mt19937 rng(3);
    std::vector<float> x(16 * 17408);
    std::normal_distribution<float> nd;
    for (float & v : x) v = nd(rng);
    float * dxin;
    CK(cudaMalloc(&dxin, x.size() * 4));
    CK(cudaMemcpy(dxin, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    for (const Shape & sh : shapes) {
        CK(actq8_quantize(dxin, sh.in, 16, a, 0));
        for (int kind = 0; kind < 2; ++kind) {
            const size_t bytes = kind == 0 ? static_cast<size_t>(sh.in / 128) * 28 * sh.out : static_cast<size_t>(sh.in / 64) * 18 * sh.out;
            std::vector<uint8_t> w(bytes);
            for (size_t i = 0; i < bytes; ++i) w[i] = static_cast<uint8_t>(rng());
            if (kind == 0) for (size_t g = 0; g < bytes / 28; ++g) { w[g * 28 + 26] = 0x00; w[g * 28 + 27] = 0x20; }
            else for (size_t g = 0; g < bytes / 18; ++g) { w[g * 18] = 0x00; w[g * 18 + 1] = 0x20; }
            uint8_t * dw;
            float * y;
            CK(cudaMalloc(&dw, bytes));
            CK(cudaMalloc(&y, sizeof(float) * 16 * sh.out));
            CK(cudaMemcpy(dw, w.data(), bytes, cudaMemcpyHostToDevice));
            for (int T : {1, 2, 3, 4, 5, 16}) {
                for (int v = 0; v < 2; ++v) {
                    auto run = [&]() {
                        if (kind == 0) CK(v ? gemv_ptq1_0_dp4a(dw, sh.in, sh.out, a, T, y, 0) : gemv_ptq1_0_dp4a_ref(dw, sh.in, sh.out, a, T, y, 0));
                        else CK(v ? gemv_q2_0_dp4a(dw, sh.in, sh.out, a, T, y, 0) : gemv_q2_0_dp4a_ref(dw, sh.in, sh.out, a, T, y, 0));
                    };
                    run();
                    CK(cudaDeviceSynchronize());
                    const int reps = sh.out > 100000 ? 5 : 30;
                    CK(cudaEventRecord(e0));
                    for (int r = 0; r < reps; ++r) run();
                    CK(cudaEventRecord(e1));
                    CK(cudaEventSynchronize(e1));
                    float ms = 0;
                    CK(cudaEventElapsedTime(&ms, e0, e1));
                    const double us = 1000.0 * ms / reps;
                    std::printf("%-7s %-22s T=%2d %-4s %9.1f us  %6.1f GB/s\n", kind ? "Q2_0" : "PTQ1_0", sh.name, T, v ? "v2" : "ref", us,
                                static_cast<double>(bytes) / (us * 1e3));
                }
            }
            CK(cudaFree(dw));
            CK(cudaFree(y));
        }
    }
    return 0;
}
