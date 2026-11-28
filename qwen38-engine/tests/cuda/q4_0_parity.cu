// Q4_0 GPU GEMV parity (Ternary Bonsai 2.5 built-in MTP head, blk.64, ggml type 2). The test builds a synthetic
// Q4_0 matrix (random nibbles, random small fp16 d) on the host and the device. The reference is libggml's own
// dequantize_row_q4_0 dotted in double precision. Two kernels are checked:
//   dp4a : gemv_dp4a(GT_Q4_0) against the reference with the same Q8_1 activations the kernel quantizes (q * dx),
//   fp32 : gemv_fp32(GT_Q4_0) (the dp4a-mask-off fallback) against the reference with the raw FP32 activations.
// Shapes: in = 5120 (attn / ffn_gate / ffn_up) and in = 10240 (nextn.eh_proj), out not a multiple of 4 rows per block.
// T = 1, 3, 16, 64 cover every tile size of gemv_dp4a. Exit 0 only if every case passes.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "ggml.h"
#include "dp4a_gemv.cuh"
#include "gemv.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

namespace {

uint16_t f16_bits(float f) {   // small positive normal values only
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const int e = static_cast<int>((u >> 23) & 0xFF) - 127 + 15;
    return static_cast<uint16_t>((e << 10) | ((u >> 13) & 0x3FF));
}

double rel_rms(const std::vector<float> & a, const std::vector<double> & b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - b[i];
        num += d * d;
        den += b[i] * b[i];
    }
    return std::sqrt(num / std::max(den, 1e-30));
}

}  // namespace

int main() {
    constexpr int QK = 32, BB = 18;
    const int64_t ins[] = {5120, 10240};
    const int64_t out = 1741;
    const int Ts[] = {1, 3, 16, 64};
    const double kTolDp4a = 3e-6, kTolFp32 = 1e-5;

    const ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_Q4_0);
    if (!tr->to_float) { std::printf("q4_0_parity: libggml has no q4_0 dequantizer\n"); return 2; }
    if (gemv_row_bytes(GT_Q4_0, 5120) != 5120 / QK * BB || !gemv_dp4a_supported(GT_Q4_0) || !gemv_fp32_supported(GT_Q4_0)) {
        std::printf("q4_0_parity: Q4_0 is not registered in the GPU GEMV dispatch\n");
        return 1;
    }

    std::mt19937 rng(2025);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> sc(0.002f, 0.05f);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    int fails = 0, cases = 0;
    double worst = 0.0;

    for (const int64_t in : ins) {
        const int64_t rowb = in / QK * BB;
        std::vector<uint8_t> w(static_cast<size_t>(rowb) * out);
        for (int64_t o = 0; o < out; ++o)
            for (int64_t b = 0; b < in / QK; ++b) {
                uint8_t * blk = w.data() + o * rowb + b * BB;   // ggml block_q4_0: fp16 d, qs[16]
                const uint16_t d = f16_bits(sc(rng));
                blk[0] = d & 0xFF; blk[1] = d >> 8;
                for (int j = 0; j < 16; ++j) blk[2 + j] = static_cast<uint8_t>(byte(rng));
            }
        ActQ8 a;
        CK(actq8_alloc(a, 64, in));
        std::vector<float> x(static_cast<size_t>(64) * in);
        for (float & v : x) v = nd(rng);
        float * dx, * y;
        uint8_t * dw;
        CK(cudaMalloc(&dx, x.size() * 4));
        CK(cudaMalloc(&dw, w.size()));
        CK(cudaMalloc(&y, sizeof(float) * 64 * out));
        CK(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        CK(actq8_quantize(dx, in, 64, a, 0));
        CK(cudaDeviceSynchronize());
        std::vector<int8_t> qh(static_cast<size_t>(64) * in);
        std::vector<float> dq(static_cast<size_t>(64) * (in / 32));
        CK(cudaMemcpy(qh.data(), a.qx, qh.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(dq.data(), a.dx, dq.size() * 4, cudaMemcpyDeviceToHost));

        std::vector<float> wf(in);
        std::vector<double> ref_q(static_cast<size_t>(64) * out), ref_f(static_cast<size_t>(64) * out);
        for (int64_t o = 0; o < out; ++o) {
            for (int64_t b = 0; b < in / QK; ++b) tr->to_float(w.data() + o * rowb + b * BB, wf.data() + b * QK, QK);
            for (int t = 0; t < 64; ++t) {
                double aq = 0, af = 0;
                for (int64_t i = 0; i < in; ++i) {
                    aq += static_cast<double>(wf[i]) * static_cast<double>(qh[static_cast<size_t>(t) * in + i]) *
                          static_cast<double>(dq[static_cast<size_t>(t) * (in / 32) + i / 32]);
                    af += static_cast<double>(wf[i]) * static_cast<double>(x[static_cast<size_t>(t) * in + i]);
                }
                ref_q[static_cast<size_t>(t) * out + o] = aq;
                ref_f[static_cast<size_t>(t) * out + o] = af;
            }
        }

        for (int mode = 0; mode < 2; ++mode) {
            for (int T : Ts) {
                CK(cudaMemset(y, 0xCD, sizeof(float) * 64 * out));
                if (mode == 0) CK(gemv_dp4a(GT_Q4_0, dw, in, out, a, T, y, 0));
                else CK(gemv_fp32(GT_Q4_0, dw, in, out, dx, T, y, 0));
                CK(cudaDeviceSynchronize());
                std::vector<float> h(static_cast<size_t>(T) * out);
                CK(cudaMemcpy(h.data(), y, h.size() * 4, cudaMemcpyDeviceToHost));
                const std::vector<double> & ref = mode == 0 ? ref_q : ref_f;
                std::vector<double> rf(ref.begin(), ref.begin() + static_cast<size_t>(T) * out);
                const double r = rel_rms(h, rf), tol = mode == 0 ? kTolDp4a : kTolFp32;
                const bool ok = r < tol;
                ++cases;
                fails += !ok;
                worst = std::max(worst, r / tol);
                std::printf("[%s] Q4_0 %s in %lld out %lld T %2d: rel_rms vs libggml q4_0 dequant %.3e (bar %.1e)\n",
                            ok ? "PASS" : "FAIL", mode == 0 ? "dp4a" : "fp32", (long long)in, (long long)out, T, r, tol);
            }
        }
        CK(cudaFree(dx)); CK(cudaFree(dw)); CK(cudaFree(y));
        actq8_free(a);
    }
    std::printf("q4_0_parity: %d cases, %d fail, worst r/bar %.3f -> %s\n", cases, fails, worst, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
