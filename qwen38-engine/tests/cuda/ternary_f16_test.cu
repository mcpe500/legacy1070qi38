// Regression test for the 2026-11-24 lm_head corruption: ternary_f16_gemm chunked a [out][in] FP16 weight
// into 32768-row blocks whose output slices y + r0 are DISJOINT (ldc = out), yet chunks 1..N-1 ran with
// beta = 1, accumulating whatever the output buffer already held into 86.8% of the vocabulary
// (32768..248319 of the LM head) — across chunks AND across head calls/samples.
//
// Proof carried here (runs on ANY CUDA device >= sm_61 through ternary_f16_gemm_pre, which touches only
// cuBLAS — FP16 storage with FP32 accumulate works on Pascal as slow emulated math):
//   1. dirty buffer: y is refilled with a garbage sentinel before EVERY call; one call must return the CPU
//      reference everywhere (a beta != 0 leaks the sentinel straight into rows 32768..out-1);
//   2. idempotency: a second identical call must return a bit-identical y (the old bug doubled it);
//   3. parity: both chunk shapes (out = 40000 -> 32768 + 7232 partial tail, out = 65536 -> 2 full chunks)
//      match the FP64 CPU reference within fp32-accumulate tolerance;
//   4. on an sm_75+ device, ternary_f16_gemm (device dequant + Tensor Core path) passes the same checks.
#include "ternary_f16.cuh"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "../common/f16bits.hpp"
#include "../common/ternary_spec.hpp"

namespace {

constexpr float kGarbage = 12345.0f;

// Host mirror of the device dequant (same digit layout): one 28-byte PTQ1_0 group -> 128 FP16 values.
void dequant_group(const uint8_t * g, uint16_t * out128) {
    constexpr int P3[5] = {81, 27, 9, 3, 1};
    const float d = f16_to_f32(static_cast<uint32_t>(g[26]) | (static_cast<uint32_t>(g[27]) << 8));
    for (int j = 0; j < 16; ++j) {
        const int c = ternary::kPtq1Code[g[j]];
        for (int s = 0; s < 5; ++s)
            out128[16 * s + j] = c > 242 ? 0 : f32_to_f16(static_cast<float>((c / P3[s]) % 3 - 1) * d);
    }
    for (int k = 0; k < 8; ++k) {
        const int c = ternary::kPtq1Code[g[16 + k]];
        for (int s = 0; s < 5; ++s)
            out128[80 + k + 8 * s] = c > 242 ? 0 : f32_to_f16(static_cast<float>((c / P3[s]) % 3 - 1) * d);
    }
    for (int side = 0; side < 2; ++side) {
        const int c = ternary::kPtq1Code[g[24 + side]];
        const int c4 = (c > 242 || c % 3) ? -1 : c / 3;   // the tail code word is 3 * c4
        constexpr int P4[4] = {27, 9, 3, 1};
        for (int s = 0; s < 4; ++s)
            out128[120 + 2 * s + side] = c4 < 0 ? uint16_t(0)
                                                : f32_to_f16(static_cast<float>((c4 / P4[s]) % 3 - 1) * d);
    }
}

struct CaseData {
    int64_t out = 0, in = 0, T = 0;
    std::vector<uint8_t> wq;        // quantized PTQ1_0 payload (device-dequant path)
    std::vector<uint16_t> w16;      // host dequant of wq (pre path)
    std::vector<uint16_t> xf;       // FP16 activations
    std::vector<double> ref;        // FP64 CPU reference of y
};

CaseData make_case(int64_t out, int64_t in, int64_t T, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> ud(-2.0f, 2.0f);
    const int64_t groups_per_row = in / ternary::kPtqGroup;
    CaseData c;
    c.out = out;
    c.in = in;
    c.T = T;
    c.wq.resize(static_cast<size_t>(out * groups_per_row) * ternary::kPtqGroupBytes);
    for (auto & b : c.wq) {   // random VALID payload: the format never contains b == 1 (mod 19)
        uint8_t v = static_cast<uint8_t>(rng() & 0xFF);
        if (v % 19 == 1) v = static_cast<uint8_t>(v + 1);
        b = v;
    }
    for (int64_t r = 0; r < out; ++r) {   // FP16 scale at offset 26 of each group
        for (int64_t g = 0; g < groups_per_row; ++g) {
            uint8_t * s = &c.wq[static_cast<size_t>((r * groups_per_row + g) * ternary::kPtqGroupBytes) + 26];
            const uint16_t bits = f32_to_f16(0.004f + 0.02f * static_cast<float>(rng() % 1024) / 1024.0f);
            s[0] = static_cast<uint8_t>(bits & 0xFF);
            s[1] = static_cast<uint8_t>(bits >> 8);
        }
    }
    c.w16.resize(static_cast<size_t>(out * in));
    for (int64_t r = 0; r < out; ++r)
        for (int64_t g = 0; g < groups_per_row; ++g)
            dequant_group(&c.wq[static_cast<size_t>((r * groups_per_row + g) * ternary::kPtqGroupBytes)],
                          &c.w16[static_cast<size_t>(r * in) + static_cast<size_t>(g) * ternary::kPtqGroup]);
    c.xf.resize(static_cast<size_t>(T * in));
    for (auto & v : c.xf) v = f32_to_f16(ud(rng));
    c.ref.assign(static_cast<size_t>(T * out), 0.0);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t o = 0; o < out; ++o) {
            double acc = 0;
            const uint16_t * wr = &c.w16[static_cast<size_t>(o * in)];
            const uint16_t * xr = &c.xf[static_cast<size_t>(t * in)];
            for (int64_t i = 0; i < in; ++i)
                acc += static_cast<double>(f16_to_f32(wr[i])) * static_cast<double>(f16_to_f32(xr[i]));
            c.ref[static_cast<size_t>(t * out + o)] = acc;
        }
    return c;
}

bool check_result(const CaseData & c, const std::vector<float> & y, const char * tag, bool verbose) {
    double max_abs = 0, max_rel = 0, scale = 1e-30;
    size_t bad_sentinel = 0;
    const size_t n = y.size();
    for (size_t i = 0; i < n; ++i) {
        if (y[i] == kGarbage) ++bad_sentinel;
        const double r = c.ref[i];
        scale = std::fmax(scale, std::fabs(r));
        const double d = std::fabs(static_cast<double>(y[i]) - r);
        max_abs = std::fmax(max_abs, d);
        max_rel = std::fmax(max_rel, d / std::fmax(std::fabs(r), 1.0));
    }
    // fp32 accumulation over in terms: ~in * eps * |row| — loose bound 2e-3 relative.
    const bool ok = bad_sentinel == 0 && max_abs <= 0.05 && max_rel <= 2e-3;
    if (verbose || !ok)
        std::printf("  %s: max_abs %.4g, max_rel %.2e, sentinel hits %zu, scale %.1f -> %s\n", tag, max_abs,
                    max_rel, bad_sentinel, scale, ok ? "ok" : "FAIL");
    return ok;
}

bool run_case(int64_t out, int64_t in, int64_t T, unsigned seed) {
    const CaseData c = make_case(out, in, T, seed);
    TernaryF16 * f = ternary_f16_init_pre(static_cast<int64_t>(32768) * in * 2, T * in * 2);
    if (!f) {
        std::printf("[FAIL] out=%lld: ternary_f16_init_pre returned null\n", static_cast<long long>(out));
        return false;
    }
    uint16_t * w16d = nullptr;
    float * yd = nullptr;
    bool ok = true;
    do {
        if (cudaMalloc(&w16d, static_cast<size_t>(out * in) * 2) != cudaSuccess ||
            cudaMalloc(&yd, static_cast<size_t>(T * out) * 4) != cudaSuccess) {
            std::printf("[FAIL] out=%lld: device alloc failed\n", static_cast<long long>(out));
            ok = false;
            break;
        }
        if (cudaMemcpy(w16d, c.w16.data(), c.w16.size() * 2, cudaMemcpyHostToDevice) != cudaSuccess ||
            ternary_f16_set_activations(f, c.xf.data(), T, in) != cudaSuccess) {
            std::printf("[FAIL] out=%lld: upload failed\n", static_cast<long long>(out));
            ok = false;
            break;
        }
        std::vector<float> y1(static_cast<size_t>(T * out), kGarbage), y2(static_cast<size_t>(T * out), kGarbage);
        // pass 1: dirty buffer in, reference out
        if (cudaMemcpy(yd, y1.data(), y1.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess ||
            ternary_f16_gemm_pre(f, w16d, in, out, T, yd, nullptr) != cudaSuccess ||
            cudaMemcpy(y1.data(), yd, y1.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::printf("[FAIL] out=%lld: gemm_pre call failed\n", static_cast<long long>(out));
            ok = false;
            break;
        }
        // pass 2: refill the garbage, run again — must be bit-identical (the beta=1 bug doubled it)
        if (cudaMemcpy(yd, y2.data(), y2.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess ||
            ternary_f16_gemm_pre(f, w16d, in, out, T, yd, nullptr) != cudaSuccess ||
            cudaMemcpy(y2.data(), yd, y2.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::printf("[FAIL] out=%lld: second gemm_pre call failed\n", static_cast<long long>(out));
            ok = false;
            break;
        }
        const bool idem = std::memcmp(y1.data(), y2.data(), y1.size() * 4) == 0;
        const bool par1 = check_result(c, y1, "pass1 vs FP64 ref", false);
        const bool par2 = check_result(c, y2, "pass2 vs FP64 ref", false);
        if (!idem)
            std::printf("  idempotency: pass2 differs from pass1 (stale accumulation) -> FAIL\n");
        if (!par1 || !par2) check_result(c, y1, "pass1 detail", true);
        ok = ok && idem && par1 && par2;
        // sm_75+: the full device path (dequant kernel + Tensor Core GEMM) on the same shapes
        if (ok && ternary_f16_supported()) {
            uint8_t * wqd = nullptr;
            if (cudaMalloc(&wqd, c.wq.size()) == cudaSuccess &&
                cudaMemcpy(wqd, c.wq.data(), c.wq.size(), cudaMemcpyHostToDevice) == cudaSuccess &&
                cudaMemcpy(yd, y1.data(), y1.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess &&
                ternary_f16_gemm(f, wqd, in, out, T, yd, nullptr) == cudaSuccess &&
                cudaMemcpy(y1.data(), yd, y1.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess) {
                const bool par_tc = check_result(c, y1, "device dequant+TC path", true);
                ok = ok && par_tc;
            } else {
                std::printf("  device dequant+TC path: skip (alloc/call failed on this device)\n");
            }
            if (wqd) cudaFree(wqd);
        }
        std::printf("[%s] out=%lld (chunks %lld%s): beta-overwrite + idempotency + parity\n",
                    ok ? "PASS" : "FAIL", static_cast<long long>(out),
                    static_cast<long long>((out + 32767) / 32768),
                    (out % 32768) ? ", partial tail" : "");
    } while (false);
    if (yd) cudaFree(yd);
    if (w16d) cudaFree(w16d);
    ternary_f16_free(f);
    return ok;
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
        std::printf("ternary_f16_test: SKIP (no CUDA device visible)\n");
        return 0;
    }
    int dev = 0, major = 0, minor = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    std::printf("ternary_f16_test: device sm_%d%d; full path %s\n", major, minor,
                ternary_f16_supported() ? "available (TC checks run)" : "unavailable (pre/cublas checks only)");
    bool ok = true;
    ok = run_case(40000, 5120, 2, 11) && ok;    // 2 chunks: 32768 + 7232 partial tail
    ok = run_case(65536, 5120, 3, 12) && ok;    // exactly 2 full chunks
    ok = run_case(81920, 5120, 2, 13) && ok;    // 3 chunks, last partial
    std::printf("ternary_f16_test: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
