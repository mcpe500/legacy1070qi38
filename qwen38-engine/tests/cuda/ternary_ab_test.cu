// Bit-identity of the v2 ternary kernels against the reference (a89f467) kernels, on random data:
//   rotate -> actq8_quantize        vs  ternary_rotate_q8          (y, qx, dx, sx byte-equal)
//   gemv_q2_0_dp4a_ref               vs  gemv_q2_0_dp4a   (v2)      (y byte-equal)
//   gemv_ptq1_0_dp4a_ref             vs  gemv_ptq1_0_dp4a (v2)      (y byte-equal)
// for every Bonsai input width (5120, 6144, 17408), output counts that are not multiples of the rows per block, and
// T = 1..5 (decode / CUDA-graph batches), 8, 13, 16, 37 (prefill tiles). Exit 0 only if everything matches.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "dp4a_gemv.cuh"
#include "ternary_gemv.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

namespace {
uint16_t f16_bits(float f) {   // small positive normal values only (scales)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const int e = static_cast<int>((u >> 23) & 0xFF) - 127 + 15;
    return static_cast<uint16_t>((e << 10) | ((u >> 13) & 0x3FF));
}

template <class T> std::vector<T> down(const T * d, size_t n) {
    std::vector<T> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}
template <class T> bool same(const T * a, const T * b, size_t n) {
    return std::memcmp(down(a, n).data(), down(b, n).data(), n * sizeof(T)) == 0;
}
}  // namespace

int main() {
    // This test pins the bit-identity of the v3 decode kernel (QWEN_PTQ1_DEC=3) against the reference. The default LPG
    // decode kernel (DEC=4) has its own FP32 order and is gated by tests/cuda/ptq1_lpg_test.cu instead.
    setenv("QWEN_PTQ1_DEC", "3", 0);
    setenv("QWEN_PTQ1_PREFILL_LPG", "0", 0);   // the prefill tiles (T > 5) of the reference order: tile2
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> byte(0, 255), code(0, 2);
    std::uniform_real_distribution<float> sc(0.002f, 0.05f);
    ternary_ptq1_init();
    const int Tmax = 64, widths[] = {5120, 6144, 17408}, outs[] = {1027, 4093};
    const int Ts[] = {1, 2, 3, 4, 5, 6, 8, 13, 16, 31, 37, 64};
    ActQ8 a1, a2;
    CK(actq8_alloc(a1, Tmax, 17408));
    CK(actq8_alloc(a2, Tmax, 17408));
    int fails = 0, cases = 0;
    for (int in : widths) {
        std::vector<float> x(static_cast<size_t>(Tmax) * in), sg(in);
        for (float & v : x) v = nd(rng);
        for (float & v : sg) v = (byte(rng) & 1) ? 1.0f : -1.0f;
        float *dx_in, *dsg, *y1, *y2;
        CK(cudaMalloc(&dx_in, x.size() * 4));
        CK(cudaMalloc(&dsg, sg.size() * 4));
        CK(cudaMalloc(&y1, x.size() * 4));
        CK(cudaMalloc(&y2, x.size() * 4));
        CK(cudaMemcpy(dx_in, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dsg, sg.data(), sg.size() * 4, cudaMemcpyHostToDevice));
        for (int T : Ts) {
            CK(ternary_rotate(dx_in, dsg, y1, in, T, 0));
            CK(actq8_quantize(y1, in, T, a1, 0));
            CK(ternary_rotate_q8(dx_in, dsg, y2, in, T, a2, 0));
            CK(cudaDeviceSynchronize());
            const size_t n = static_cast<size_t>(T) * in, nb = n / 32;
            const bool ok = same(y1, y2, n) && same(a1.qx, a2.qx, n) && same(a1.dx, a2.dx, nb) && same(a1.sx, a2.sx, nb);
            ++cases;
            if (!ok) { ++fails; std::printf("[FAIL] rotate+q8 in=%d T=%d\n", in, T); }
        }
        // GEMVs on the activations of the fused path (identical to the reference ones, checked above)
        CK(ternary_rotate_q8(dx_in, dsg, y2, in, Tmax, a2, 0));
        for (int out : outs) {
            // Q2_0 rows: [fp16 d][16 bytes codes] per 64 values, codes 0..2
            const size_t q2_row = static_cast<size_t>(in / 64) * 18, p1_row = static_cast<size_t>(in / 128) * 28;
            std::vector<uint8_t> wq2(q2_row * out), wp1(p1_row * out);
            for (int o = 0; o < out; ++o)
                for (int b = 0; b < in / 64; ++b) {
                    uint8_t * blk = wq2.data() + o * q2_row + b * 18;
                    const uint16_t h = f16_bits(sc(rng));
                    blk[0] = h & 0xFF; blk[1] = h >> 8;
                    for (int j = 0; j < 16; ++j) {
                        int v = 0;
                        for (int c = 0; c < 4; ++c) v |= code(rng) << (2 * c);
                        blk[2 + j] = static_cast<uint8_t>(v);
                    }
                }
            for (int o = 0; o < out; ++o)
                for (int g = 0; g < in / 128; ++g) {
                    uint8_t * blk = wp1.data() + o * p1_row + g * 28;
                    for (int j = 0; j < 26; ++j) blk[j] = static_cast<uint8_t>(byte(rng));   // any byte, invalid codes too
                    const uint16_t h = f16_bits(sc(rng));
                    blk[26] = h & 0xFF; blk[27] = h >> 8;
                }
            uint8_t *dq2, *dp1, *dp1c;
            unsigned long long * dchg;
            float *o1, *o2;
            CK(cudaMalloc(&dq2, wq2.size()));
            CK(cudaMalloc(&dp1, wp1.size()));
            CK(cudaMalloc(&o1, sizeof(float) * Tmax * out));
            CK(cudaMalloc(&o2, sizeof(float) * Tmax * out));
            CK(cudaMemcpy(dq2, wq2.data(), wq2.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dp1, wp1.data(), wp1.size(), cudaMemcpyHostToDevice));
            // The optimized kernels read the canonical copy (invalid codes rewritten to byte 0, the bytes the reference tables
            // decode them as); the reference reads the raw random bytes, invalid codes included.
            CK(cudaMalloc(&dp1c, wp1.size()));
            CK(cudaMalloc(&dchg, sizeof(unsigned long long)));
            CK(cudaMemset(dchg, 0, sizeof(unsigned long long)));
            CK(cudaMemcpy(dp1c, dp1, wp1.size(), cudaMemcpyDeviceToDevice));
            CK(ternary_ptq1_canonicalize(dp1c, static_cast<int64_t>(out) * (in / 128), dchg, 0));
            unsigned long long nchg = 0;
            CK(cudaMemcpy(&nchg, dchg, sizeof(nchg), cudaMemcpyDeviceToHost));
            if (out == outs[0]) std::printf("canonicalize in=%d out=%d: %llu of %zu payload bytes rewritten\n", in, out, nchg, static_cast<size_t>(out) * (in / 128) * 26);
            for (int T : Ts) {
                for (int kind = 0; kind < 2; ++kind) {
                    CK(cudaMemset(o1, 0xCD, sizeof(float) * Tmax * out));
                    CK(cudaMemset(o2, 0xEF, sizeof(float) * Tmax * out));
                    if (kind == 0) {
                        CK(gemv_q2_0_dp4a_ref(dq2, in, out, a2, T, o1, 0));
                        CK(gemv_q2_0_dp4a(dq2, in, out, a2, T, o2, 0));
                    } else {
                        CK(gemv_ptq1_0_dp4a_ref(dp1, in, out, a2, T, o1, 0));
                        CK(gemv_ptq1_0_dp4a(dp1c, in, out, a2, T, o2, 0));
                    }
                    CK(cudaDeviceSynchronize());
                    const bool ok = same(o1, o2, static_cast<size_t>(T) * out);
                    ++cases;
                    if (!ok) { ++fails; std::printf("[FAIL] %s in=%d out=%d T=%d\n", kind ? "PTQ1_0" : "Q2_0", in, out, T); }
                }
            }
            CK(cudaFree(dq2)); CK(cudaFree(dp1)); CK(cudaFree(dp1c)); CK(cudaFree(dchg)); CK(cudaFree(o1)); CK(cudaFree(o2));
        }
        CK(cudaFree(dx_in)); CK(cudaFree(dsg)); CK(cudaFree(y1)); CK(cudaFree(y2));
    }
    std::printf("ternary_ab_test: %d cases, %d bit-identical, %d differ -> %s\n", cases, cases - fails, fails, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
