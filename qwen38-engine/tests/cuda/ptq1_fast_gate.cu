// Numeric gate for the opt-in PTQ1_0 fast-prefill kernel (gemv_ptq1_0_dp4a with fast = true).
//
// The fast path splits the 128-group dot product into two independent FP32 accumulators (even groups in
// acc, odd groups in acc2) and merges them before the warp reduce, so the FP32 summation order is
// relaxed. That is only allowed when the result stays numerically equivalent to the default path, so
// this test enforces, for every Bonsai input width and every prefill T:
//   1. cosine(fast, FP64 reference) >= 0.99999, with the FP64 reference using the *same* quantized
//      activations and the same canonical weights, so only the summation order is measured;
//   2. cosine(default, FP64 reference) >= 0.99999, the same bar the bit-identical default must clear;
//   3. the greedy token (argmax over the output rows) is identical between fast and default at every
//      position, so the opt-in path cannot change a sampled token;
//   4. fast vs default max abs error is reported, so the drift stays on the record.
//
// The long-context needle (MAGENTA-FALCON-42 at 131072 positions) and the 128K prefill timing are
// engine-level checks; this test is the kernel-level gate that must pass before the flag is wired.
// Exit 0 only if every case passes.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "ternary_gemv.cuh"
#include "ternary_spec.hpp"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

namespace {

constexpr double kCosMin = 0.99999;

uint16_t f16_bits(float f) {   // small positive normal values only (scales)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const int e = static_cast<int>((u >> 23) & 0xFF) - 127 + 15;
    return static_cast<uint16_t>((e << 10) | ((u >> 13) & 0x3FF));
}
// fp16 -> fp32 with the exponent rebased into the fp32 field (a raw h << 16 is not a conversion).
float f16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>((h >> 15) & 1u) << 31, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
    uint32_t u;
    if (exp == 0) {
        if (man == 0) u = sign;
        else {   // subnormal half: man * 2^-24
            int e = -1;
            uint32_t m = man;
            while ((m & 0x400u) == 0) { m <<= 1; --e; }
            m &= 0x3FFu;
            u = sign | (static_cast<uint32_t>(127 - 15 + e) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        u = sign | (0xFFu << 23) | (man << 13);
    } else {
        u = sign | ((exp - 15u + 127u) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// PTQ1_0 packing (src/common/ternary_spec.hpp, kPtqGroupBytes = 28): 26 payload bytes + an fp16 scale
// at offset 26. A payload byte is an index into the empirical kPtq1Code table, whose entry is the code
// word c = sum_s 3^(n-1-s) * (t_s + 1) with t_s in {-1,0,1}; the table holds 5 digits for bytes 0..23
// and 4 digits (of c / 3) for the two tail bytes. Byte j < 16 covers positions {j, j+16, j+32, j+48,
// j+64}, byte 16+k covers {80+k, 88+k, 96+k, 104+k, 112+k}, byte 24 covers {120,122,124,126} and byte
// 25 covers {121,123,125,127}. Decoded weight = (digit - 1) * d.
int g_code_byte[243];   // inverse of kPtq1Code, built once in main

double weight_at(const uint8_t * blk, int p) {
    int byte;
    bool tail;
    int slot;
    if (p < 80) { byte = p % 16; tail = false; slot = p / 16; }
    else if (p < 120) { const int k = (p - 80) % 8; byte = 16 + k; tail = false; slot = (p - 80) / 8; }
    else { byte = 24 + (p & 1); tail = true; slot = (p - 120) / 2; }
    int v = ternary::kPtq1Code[blk[byte]];
    if (v == 255) v = 0;   // unused byte; this test never stores one
    if (tail) v /= 3;      // the tail code word is 3 * c4
    const int n = tail ? 4 : 5;
    int place = 1;
    for (int i = 0; i < n - 1 - slot; ++i) place *= 3;   // slot 0 is the most significant digit
    const int digit = (v / place) % 3;
    return static_cast<double>(digit - 1);
}

double cosine(const std::vector<float> & a, const std::vector<double> & b) {
    double d = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) { d += static_cast<double>(a[i]) * b[i]; na += static_cast<double>(a[i]) * a[i]; nb += b[i] * b[i]; }
    return d / std::sqrt(na * nb);
}

template <class T> std::vector<T> down(const T * d, size_t n) {
    std::vector<T> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

}  // namespace

int main() {
    setenv("QWEN_PTQ1_PREFILL_LPG", "0", 0);   // this gate compares the opt-in fast tile2 path with the tile2 default
    std::mt19937 rng(20261121);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> sc(0.002f, 0.05f);
    ternary_ptq1_init();
    for (int c = 0; c < 243; ++c) g_code_byte[c] = 0;
    for (int b = 0; b < 256; ++b) {
        const int c = ternary::kPtq1Code[b];
        if (c != 255) g_code_byte[c] = b;
    }
    
    const int Tmax = 64, widths[] = {5120, 6144, 17408}, outs[] = {1027, 4093};
    const int Ts[] = {6, 8, 13, 16, 31, 37, 64};   // prefill tiles only: fast is never used at T <= 5
    ActQ8 a;
    CK(actq8_alloc(a, Tmax, 17408));
    int fails = 0, cases = 0;
    double worst_fast = 1.0, worst_def = 1.0, worst_abs = 0.0, worst_ref = 1.0;

    for (int in : widths) {
        std::vector<float> x(static_cast<size_t>(Tmax) * in);
        for (float & v : x) v = nd(rng);
        float * dx;
        CK(cudaMalloc(&dx, x.size() * 4));
        CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        CK(actq8_quantize(dx, in, Tmax, a, 0));   // one quantization reused by every T below
        const std::vector<int8_t> q = down(a.qx, static_cast<size_t>(Tmax) * in);
        const std::vector<float> dq = down(a.dx, static_cast<size_t>(Tmax) * (in / 32));
        for (int out : outs) {
            // PTQ1_0 rows: 28-byte groups, 26 random payload bytes + the fp16 scale. Only canonical codes
            // are stored: an invalid byte decodes differently in the two paths, and the engine always
            // feeds canonicalized weights (ternary_ptq1_canonicalize) to the optimized kernels.
            const size_t p1_row = static_cast<size_t>(in / 128) * 28;
            std::vector<uint8_t> wp(p1_row * out);
            for (int o = 0; o < out; ++o)
                for (int g = 0; g < in / 128; ++g) {
                    uint8_t * blk = wp.data() + o * p1_row + g * 28;
                    for (int j = 0; j < 26; ++j) {
                        // canonical code words only: bytes 0..23 take any of [0,242], the two tail bytes
                        // take multiples of 3 (their code word is 3 * c4)
                        const bool tail = j >= 24;
                        const int c = tail ? 3 * (byte(rng) % 81) : byte(rng) % 243;
                        blk[j] = static_cast<uint8_t>(g_code_byte[c]);
                    }
                    const uint16_t h = f16_bits(sc(rng));
                    blk[26] = h & 0xFF;
                    blk[27] = h >> 8;
                }
            uint8_t * dW;
            float *y_def, *y_fast, *y_ref;
            CK(cudaMalloc(&dW, wp.size()));
            CK(cudaMalloc(&y_def, sizeof(float) * Tmax * out));
            CK(cudaMalloc(&y_fast, sizeof(float) * Tmax * out));
            CK(cudaMalloc(&y_ref, sizeof(float) * Tmax * out));
            CK(cudaMemcpy(dW, wp.data(), wp.size(), cudaMemcpyHostToDevice));
            std::vector<double> ref(static_cast<size_t>(Tmax) * out);
            for (int o = 0; o < out; ++o)
                for (int g = 0; g < in / 128; ++g) {
                    const uint8_t * blk = wp.data() + o * p1_row + g * 28;
                    const double d = static_cast<double>(f16_to_f32(static_cast<uint16_t>(blk[26] | (blk[27] << 8))));
                    for (int t = 0; t < Tmax; ++t) {
                        double s = 0.0;
                        for (int b = 0; b < 4; ++b) {
                            const double dxb = static_cast<double>(dq[static_cast<size_t>(t) * (in / 32) + g * 4 + b]);
                            double dot = 0.0;
                            for (int p = 0; p < 32; ++p)
                                dot += weight_at(blk, b * 32 + p) * static_cast<double>(q[static_cast<size_t>(t) * in + g * 128 + b * 32 + p]);
                            s += dxb * dot;
                        }
                        ref[static_cast<size_t>(t) * out + o] += d * s;
                    }
                }
            for (int T : Ts) {
                CK(gemv_ptq1_0_dp4a(dW, in, out, a, T, y_def, 0, false));
                CK(gemv_ptq1_0_dp4a(dW, in, out, a, T, y_fast, 0, true));
                CK(gemv_ptq1_0_dp4a_ref(dW, in, out, a, T, y_ref, 0));
                CK(cudaDeviceSynchronize());
                const std::vector<float> yd = down(y_def, static_cast<size_t>(T) * out);
                const std::vector<float> yf = down(y_fast, static_cast<size_t>(T) * out);
                const std::vector<float> yr = down(y_ref, static_cast<size_t>(T) * out);
                const std::vector<double> rf(ref.begin(), ref.begin() + static_cast<size_t>(T) * out);
                const double c_def = cosine(yd, rf), c_fast = cosine(yf, rf), c_ref = cosine(yr, rf);
                worst_ref = std::min(worst_ref, c_ref);
                double max_abs = 0.0;
                int greedy_diff = 0;
                for (int t = 0; t < T; ++t) {
                    int bd = 0, bf = 0;
                    for (int o = 1; o < out; ++o) {
                        if (yd[t * out + o] > yd[t * out + bd]) bd = o;
                        if (yf[t * out + o] > yf[t * out + bf]) bf = o;
                    }
                    if (bd != bf) ++greedy_diff;
                }
                for (size_t i = 0; i < yd.size(); ++i) max_abs = std::max(max_abs, std::fabs(static_cast<double>(yd[i]) - static_cast<double>(yf[i])));
                const bool ok = c_fast >= kCosMin && c_def >= kCosMin && greedy_diff == 0;
                ++cases;
                if (!ok) ++fails;
                worst_fast = std::min(worst_fast, c_fast);
                worst_def = std::min(worst_def, c_def);
                worst_abs = std::max(worst_abs, max_abs);
                std::printf("%s in=%5d out=%4d T=%2d | fast cos %.8f | default cos %.8f | ref kernel cos %.8f | max|fast-default| %.3e | greedy diff %d\n",
                            ok ? "PASS" : "FAIL", in, out, T, c_fast, c_def, c_ref, max_abs, greedy_diff);
            }
            CK(cudaFree(dW)); CK(cudaFree(y_def)); CK(cudaFree(y_fast)); CK(cudaFree(y_ref));
        }
        CK(cudaFree(dx));
    }
    std::printf("ptq1_fast_gate: %d cases, %d pass, %d fail | worst fast cos %.8f | worst default cos %.8f | worst ref-kernel cos %.8f | worst abs %.3e -> %s\n",
                cases, cases - fails, fails, worst_fast, worst_def, worst_ref, worst_abs, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}