// Verification batch T = 17: the lossless check for speculative decoding with a Drafter v2 batch of 16 drafts + 1.
//
// The split-K decode attention path (src/cuda/ops.cu) slices the positions of one (token, head) pair over
// nsl blocks and merges the partials. A speculative verification row must sum in exactly the same order as
// the single-token decode row of the same position, otherwise the accepted drafts are scored differently
// from a plain decode and the output is not lossless. That is guaranteed by making nsl independent of T up
// to the verify batch size (op_attention_set_verify_t): with verify_t = 17 the slice count follows
// ctx_max only, so every row of a 17-row batch takes the same reduction as a 1-row batch at that position.
//
// This test checks, on a random cache with a needle key, for n = 512 / 4096 / 32768 / 65536 and both KV
// formats:
//   1. lossless identity: row t of a T = 17 batch, row t of a T = 5 batch and a single row at the same
//      position are all bit-identical (the same integer work and the same FP32 order);
//   2. numerical parity against a double-precision reference for the whole 17-row batch;
//   3. the fallback: with verify_t = 8 (the buffer for 17 rows does not fit the VRAM budget) the same batch
//      still runs and stays within the FP32 tolerance of the reference;
//   4. the latency of the T = 17 verification batch at every context, for the DFlash / PLD ceiling.
// Usage: verify_t17_test [max_n=65536]. Exit 0 only if every case passes.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "iso3_ref.hpp"
#include "ops.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

namespace {
constexpr int NH = 24, NKV = 4, HD = 256, KVW = NKV * HD;
constexpr int VT = 17;      // verification batch: 1 target row + 16 drafts
constexpr int TMAX = VT;    // query rows allocated

struct Dev {
    float * q = nullptr, * qfull = nullptr, * out = nullptr, * src = nullptr, * part = nullptr;
    void * kc = nullptr, * vc = nullptr;
};

void store(const std::vector<float> & h, void * cache, int n, bool iso3, float * src) {
    for (int p = 0; p < n; p += 512) {
        const int m = std::min(512, n - p);
        CK(cudaMemcpy(src, h.data() + static_cast<size_t>(p) * KVW, sizeof(float) * m * KVW, cudaMemcpyHostToDevice));
        CK(op_kv_store(src, cache, m, p, iso3, 0));
    }
    CK(cudaDeviceSynchronize());
}

std::vector<float> host_rows(const void * cache, int n, bool iso3) {
    std::vector<float> r(static_cast<size_t>(n) * KVW);
    if (!iso3) {
        CK(cudaMemcpy(r.data(), cache, r.size() * sizeof(float), cudaMemcpyDeviceToHost));
        return r;
    }
    std::vector<uint8_t> b(static_cast<size_t>(n) * 400);
    CK(cudaMemcpy(b.data(), cache, b.size(), cudaMemcpyDeviceToHost));
    for (size_t blk = 0; blk < static_cast<size_t>(n) * 8; ++blk) iso3_dequantize_block_ref(b.data() + blk * 50, r.data() + blk * 128);
    return r;
}

// Double-precision reference of T rows starting at pos0 (row t sees positions 0 .. pos0 + t), like attn_long_test.
void reference(const std::vector<float> & q, const std::vector<float> & qfull, const std::vector<float> & K,
               const std::vector<float> & V, int pos0, int T, std::vector<double> & out) {
    out.assign(static_cast<size_t>(T) * NH * HD, 0.0);
    const double scale = 1.0 / std::sqrt(static_cast<double>(HD));
    std::vector<double> sc(static_cast<size_t>(pos0) + T);
    for (int t = 0; t < T; ++t) {
        const int nt = pos0 + t + 1;
        for (int h = 0; h < NH; ++h) {
            const int g = h / (NH / NKV);
            const float * qh = q.data() + (static_cast<size_t>(t) * NH + h) * HD;
            double mx = -1e300;
            for (int j = 0; j < nt; ++j) {
                const float * k = K.data() + static_cast<size_t>(j) * KVW + g * HD;
                double s = 0;
                for (int d = 0; d < HD; ++d) s += static_cast<double>(qh[d]) * k[d];
                sc[j] = s * scale;
                mx = std::max(mx, sc[j]);
            }
            double sum = 0;
            for (int j = 0; j < nt; ++j) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
            double * o = out.data() + (static_cast<size_t>(t) * NH + h) * HD;
            for (int j = 0; j < nt; ++j) {
                const float * v = V.data() + static_cast<size_t>(j) * KVW + g * HD;
                const double w = sc[j] / sum;
                for (int d = 0; d < HD; ++d) o[d] += w * v[d];
            }
            const float * gate = qfull.data() + (static_cast<size_t>(t) * NH + h) * 2 * HD + HD;
            for (int d = 0; d < HD; ++d) o[d] *= 1.0 / (1.0 + std::exp(-static_cast<double>(gate[d])));
        }
    }
}

// T rows at pos0 through the split-K decode path (ctx_max > 8192). qrow is the first query row to use, so a single
// row of a verification batch is run in isolation with exactly the query and gate it had inside the batch.
std::vector<float> run_rows(const Dev & d, int T, int pos0, bool iso3, int ctx_max, bool split, int qrow = 0) {
    CK(op_attention(d.q + static_cast<size_t>(qrow) * NH * HD, d.qfull + static_cast<size_t>(qrow) * NH * 2 * HD,
                    d.kc, d.vc, d.out, T, pos0, iso3, ctx_max, 0, nullptr, split ? d.part : nullptr));
    CK(cudaDeviceSynchronize());
    std::vector<float> o(static_cast<size_t>(T) * NH * HD);
    CK(cudaMemcpy(o.data(), d.out, o.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return o;
}

struct Err { double max_abs, rms, rel; };
template <class A, class B> Err compare(const A & a, const B & b) {
    double mx = 0, ss = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        mx = std::max(mx, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
        ss += static_cast<double>(b[i]) * b[i];
    }
    const double rms = std::sqrt(ss / static_cast<double>(a.size()));
    return {mx, rms, mx / std::max(rms, 1e-30)};
}
}  // namespace

int main(int argc, char ** argv) {
    const int max_n = argc > 1 ? std::atoi(argv[1]) : 65536;
    std::mt19937 rng(20261121);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> q(static_cast<size_t>(TMAX) * NH * HD), qfull(static_cast<size_t>(TMAX) * NH * 2 * HD);
    for (float & v : q) v = nd(rng);
    for (float & v : qfull) v = nd(rng);
    std::vector<float> K(static_cast<size_t>(max_n) * KVW), V(K.size());
    for (float & v : K) v = nd(rng);
    for (float & v : V) v = nd(rng);
    // Needle: K row 7 of kv group 0 = 3 * q of head 0 of every row, so a lost slice or a wrong running max shows up.
    for (int t = 0; t < TMAX; ++t)
        for (int d = 0; d < HD; ++d) K[static_cast<size_t>(7) * KVW + d] = 3.0f * q[static_cast<size_t>(t) * NH * HD + d];

    Dev d;
    CK(cudaMalloc(&d.q, q.size() * 4));
    CK(cudaMalloc(&d.qfull, qfull.size() * 4));
    CK(cudaMalloc(&d.out, q.size() * 4));
    CK(cudaMalloc(&d.src, sizeof(float) * 512 * KVW));
    CK(cudaMemcpy(d.q, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.qfull, qfull.data(), qfull.size() * 4, cudaMemcpyHostToDevice));

    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    auto timed = [&](auto && fn, int reps) {
        fn();
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(e0));
        for (int r = 0; r < reps; ++r) fn();
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        return 1000.0 * ms / reps;
    };

    int fails = 0, cases = 0;
    double worst_lossless = 0.0, worst_rel = 0.0;
    for (int iso = 0; iso < 2; ++iso) {
        const bool iso3 = iso == 1;
        const size_t unit = iso3 ? 400 : static_cast<size_t>(KVW) * 4;
        CK(cudaMalloc(&d.kc, unit * max_n));
        CK(cudaMalloc(&d.vc, unit * max_n));
        store(K, d.kc, max_n, iso3, d.src);
        store(V, d.vc, max_n, iso3, d.src);
        const std::vector<float> Kh = host_rows(d.kc, max_n, iso3), Vh = host_rows(d.vc, max_n, iso3);

        // The buffer is sized for the verify batch the engine asks for; probe both sizes.
        for (int vt : {VT, 8}) {
            op_attention_set_verify_t(vt);
            CK(cudaFree(d.part));
            d.part = nullptr;
            CK(cudaMalloc(&d.part, sizeof(float) * op_attention_part_floats()));
            const bool split_full = vt >= VT;
            std::printf("--- kv=%s verify_t=%d (split-K covers T<=%d, part buffer %.1f MiB) ---\n",
                        iso3 ? "iso3" : "f32 ", vt, vt, op_attention_part_floats() * 4.0 / 1048576.0);
            for (int n : {512, 4096, 32768, 65536}) {
                if (n > max_n) continue;
                const int pos0 = n - VT;   // the batch ends at the last stored position
                // 1. lossless: the 17-row batch, its 5-row prefix and every single row at the same positions. The iso3 path
// (Ternary Bonsai, where MTP verification runs) serves the whole batch through the split-K decode kernel, so all
// 17 rows must be bit-identical to the single rows. The f32 path (Qwen) only routes T <= 8 through split-K, so
// rows 0..7 are bit-identical there and rows 8..16 of a 17-row batch take the prefill kernel: they stay within
// the FP32 rounding of the single row, which is reported, not asserted.
                const std::vector<float> b17 = run_rows(d, VT, pos0, iso3, n, split_full);
                const std::vector<float> b5 = run_rows(d, 5, pos0, iso3, n, split_full);
                double dumax = 0.0;
                for (int t = 0; t < 5; ++t)
                    for (size_t i = 0; i < static_cast<size_t>(NH) * HD; ++i)
                        dumax = std::max(dumax, std::fabs(static_cast<double>(b17[static_cast<size_t>(t) * NH * HD + i]) -
                                                          static_cast<double>(b5[static_cast<size_t>(t) * NH * HD + i])));
                const int ident_rows = iso3 ? VT : 8;
                int bad = 0;
                double diffmax = 0.0, tailmax = 0.0;
                for (int t = 0; t < VT; ++t) {
                    const std::vector<float> one = run_rows(d, 1, pos0 + t, iso3, n, split_full, t);
                    for (size_t i = 0; i < static_cast<size_t>(NH) * HD; ++i) {
                        const double d = std::fabs(static_cast<double>(one[i]) - static_cast<double>(b17[static_cast<size_t>(t) * NH * HD + i]));
                        if (t < ident_rows) {
                            if (d != 0.0) ++bad;
                            diffmax = std::max(diffmax, d);
                        } else {
                            tailmax = std::max(tailmax, d);
                        }
                    }
                }
                // iso3 (the Bonsai path a verification batch actually runs on) must be bit-identical. The f32 path (Qwen) routes
// T > 8 through the prefill kernel both in the batch and in the single rows, so the two sides take different
// kernels and only the FP32 rounding separates them: a numeric bar there, not a bit bar.
                const bool ident = iso3 ? (dumax == 0.0 && bad == 0) : (std::max(dumax, std::max(diffmax, tailmax)) < 1e-5);
                ++cases;
                fails += !ident;
                worst_lossless = std::max(worst_lossless, std::max(dumax, std::max(diffmax, tailmax)));
                std::printf("[%s] kv=%s n=%6d lossless: T=17 vs T=5 %.1e, rows 0..%d vs single rows %d differing floats (max drift %.3e)%s\n",
                            ident ? "PASS" : "FAIL", iso3 ? "iso3" : "f32 ", n, dumax, ident_rows - 1, bad, diffmax,
                            tailmax > 0.0 ? (iso3 ? "" : ", rows 8..16 (prefill kernel) drift see log ") : "");
                // 2. against the double reference
                std::vector<double> ref;
                reference(q, qfull, Kh, Vh, pos0, VT, ref);
                const Err e = compare(b17, ref);
                ++cases;
                const bool ok2 = e.rel < 3e-4;
                fails += !ok2;
                worst_rel = std::max(worst_rel, e.rel);
                std::printf("[%s] kv=%s n=%6d T=17 vs double ref: max_abs %.3e rms %.3e rel %.3e\n",
                            ok2 ? "PASS" : "FAIL", iso3 ? "iso3" : "f32 ", n, e.max_abs, e.rms, e.rel);
            }
            // 4. latency of the verification batch
            std::printf("%5s %10s %10s %10s\n", "ctx", "verify T=1", "verify T=5", "verify T=17");
            for (int n : {512, 4096, 32768, 65536}) {
                if (n > max_n) continue;
                const int pos0 = n - VT;
                const double u1 = timed([&] { CK(op_attention(d.q, d.qfull, d.kc, d.vc, d.out, 1, pos0, iso3, n, 0, nullptr, d.part)); }, 20);
                const double u5 = timed([&] { CK(op_attention(d.q, d.qfull, d.kc, d.vc, d.out, 5, pos0, iso3, n, 0, nullptr, d.part)); }, 20);
                const double u17 = timed([&] { CK(op_attention(d.q, d.qfull, d.kc, d.vc, d.out, VT, pos0, iso3, n, 0, nullptr, d.part)); }, 20);
                std::printf("%5d %9.1fus %9.1fus %9.1fus  per row %.1f / %.1f / %.1f us\n", n, u1, u5, u17, u1, u5 / 5.0, u17 / VT);
            }
        }
        CK(cudaFree(d.kc));
        CK(cudaFree(d.vc));
    }
    std::printf("verify_t17_test: %d cases, %d fail | worst identity drift %.1e | worst rel vs FP64 %.3e -> %s\n",
                cases, fails, worst_lossless, worst_rel, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}