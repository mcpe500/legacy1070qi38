// Q2_K GPU GEMV parity (issue #57 M4). Q2_K has no dp4a kernel in the engine before this branch, so the test
// builds a synthetic Q2_K matrix: random payload bytes (codes 0..3), random scale bytes and random small
// fp16 d/dmin, both on the device and on the host. The reference is libggml's own dequantize_row_q2_K
// (GGML_TYPE_Q2_K) dotted in double precision with the same activations the kernel quantizes (q * dx), so
// only the GPU dequantization and the summation order are measured. The check runs for T = 1, 3, 16 and 64
// (the tile sizes gemv_dp4a dispatches to) and for an `in` that is not a multiple of the rows per block.
// Exit 0 only if every case passes.
#include <cuda_runtime.h>

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

uint16_t f16_bits(float f) {   // small positive normal values only (the Q2_K d and dmin)
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
    constexpr int QK = 256;
    const int64_t in = 5120, out = 1741;   // not a multiple of 4 rows per block, like a real ffn_down
    const int Ts[] = {1, 3, 16, 64};
    const double kTol = 3e-6;              // the same-activation bar the other formats use in gemv_parity

    std::mt19937 rng(1022);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> sc(0.002f, 0.05f);
    const int64_t rowb = in / QK * 84;
    std::vector<uint8_t> w(static_cast<size_t>(rowb) * out);
    for (int64_t o = 0; o < out; ++o)
        for (int64_t b = 0; b < in / QK; ++b) {
            uint8_t * blk = w.data() + o * rowb + b * 84;   // ggml block_q2_K: scales, qs, then d and dmin
            for (int j = 0; j < 16; ++j) blk[j] = static_cast<uint8_t>(byte(rng));       // scale/min nibbles
            const uint16_t d = f16_bits(sc(rng)), dm = f16_bits(sc(rng));
            blk[80] = d & 0xFF; blk[81] = d >> 8;
            blk[82] = dm & 0xFF; blk[83] = dm >> 8;
            for (int j = 0; j < 64; ++j) {
                // codes 0..3 fill the two bits of every slot, so every code is valid
                int v = 0;
                for (int c = 0; c < 4; ++c) v |= (byte(rng) & 3) << (2 * c);
                blk[16 + j] = static_cast<uint8_t>(v);
            }
        }

    ActQ8 a;
    CK(actq8_alloc(a, 64, in));
    std::normal_distribution<float> nd(0.0f, 1.0f);
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
    const std::vector<int8_t> q(a.T_max), qh(static_cast<size_t>(64) * in);
    CK(cudaMemcpy(const_cast<int8_t *>(qh.data()), a.qx, qh.size(), cudaMemcpyDeviceToHost));
    const std::vector<float> dq(64 * (in / 32));
    CK(cudaMemcpy(const_cast<float *>(dq.data()), a.dx, dq.size() * 4, cudaMemcpyDeviceToHost));


    const ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_Q2_K);
    if (!tr->to_float) { std::printf("q2_k_parity: libggml has no q2_K dequantizer\n"); return 2; }

    std::vector<float> wf(in);
    std::vector<double> ref(static_cast<size_t>(64) * out);
    for (int64_t o = 0; o < out; ++o) {
        for (int64_t b = 0; b < in / QK; ++b) tr->to_float(w.data() + o * rowb + b * 84, wf.data() + b * QK, QK);
        for (int t = 0; t < 64; ++t) {
            double acc = 0;
            for (int64_t i = 0; i < in; ++i)
                acc += static_cast<double>(wf[i]) * static_cast<double>(qh[static_cast<size_t>(t) * in + i]) *
                       static_cast<double>(dq[static_cast<size_t>(t) * (in / 32) + i / 32]);
            ref[static_cast<size_t>(t) * out + o] = acc;
        }
    }

    int fails = 0, cases = 0;
    double worst = 0.0;
    for (int T : Ts) {
        CK(cudaMemset(y, 0xCD, sizeof(float) * 64 * out));
        CK(gemv_dp4a(GT_Q2_K, dw, in, out, a, T, y, 0));
        CK(cudaDeviceSynchronize());
        std::vector<float> h(static_cast<size_t>(T) * out);
        CK(cudaMemcpy(h.data(), y, h.size() * 4, cudaMemcpyDeviceToHost));
        std::vector<double> rf(ref.begin(), ref.begin() + static_cast<size_t>(T) * out);
        const double r = rel_rms(h, rf);
        const bool ok = r < kTol;
        ++cases;
        fails += !ok;
        worst = std::max(worst, r);
        std::printf("[%s] Q2_K in %lld out %lld T %2d: rel_rms vs libggml q2_K dequant %.3e (bar %.1e)\n",
                    ok ? "PASS" : "FAIL", (long long)in, (long long)out, T, r, kTol);
    }
    std::printf("q2_k_parity: %d cases, %d fail, worst rel_rms %.3e -> %s\n", cases, fails, worst, fails ? "FAIL" : "PASS");
    CK(cudaFree(dx)); CK(cudaFree(dw)); CK(cudaFree(y));
    actq8_free(a);
    return fails ? 1 : 0;
}
