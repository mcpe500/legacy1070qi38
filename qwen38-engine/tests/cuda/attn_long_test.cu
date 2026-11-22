// Long-context attention parity (exp/bonsai-128k-context).
//  1. Fast path vs tiled path on the same cache (n <= 8192): op_attention with ctx_max = 8192 (scores in shared memory)
//     against ctx_max = 16384 (tiled online softmax). Both KV formats.
//  3. Split-K decode path (ctx_max > 8192, T <= 8, partials + combine) vs the same double reference.
//  2. Tiled path vs a double-precision CPU reference at n = 4096, 32768, 65536 and 131072 positions, both KV formats
//     (iso3 reference = iso3_dequantize_block_ref of the bytes the GPU stored). A "needle" key at position 7 makes one head
//     attend almost only there, so a tile bookkeeping error (wrong max, lost tile) shows up as a large error.
//  4. Prefill path (ctx_max > 8192, T > 8: query-tiled K/V-reuse kernel, with and without position slices) for T = 64 and
//     T = 37 query rows at pos0 = n - T against the double reference, and the old per-(token, head) tiled kernel as a second
//     reference (bar: rel < 3e-4 and at most 1.5x the old kernel's error). Timing of both kernels for T = 64 at the largest n.
// Usage: attn_long_test [max_n=131072]. Exit 0 only if every case passes.
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
constexpr int TMAX = 64;   // query rows allocated; the decode cases use T = 2 (row 0 sees n-1 positions, row 1 sees n)

struct Dev {
    float * q = nullptr, * qfull = nullptr, * out = nullptr, * src = nullptr, * part = nullptr;
    void * kc = nullptr, * vc = nullptr;
};

// Stores host rows [n][1024] into the device cache (chunks of 512 positions through `src`).
void store(const std::vector<float> & h, void * cache, int n, bool iso3, float * src) {
    for (int p = 0; p < n; p += 512) {
        const int m = std::min(512, n - p);
        CK(cudaMemcpy(src, h.data() + static_cast<size_t>(p) * KVW, sizeof(float) * m * KVW, cudaMemcpyHostToDevice));
        CK(op_kv_store(src, cache, m, p, iso3, 0));
    }
    CK(cudaDeviceSynchronize());
}

// Host copy of the cache as FP32 rows (the iso3 bytes dequantized with the reference decoder).
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

void reference(const std::vector<float> & q, const std::vector<float> & qfull, const std::vector<float> & K,
               const std::vector<float> & V, int n, std::vector<double> & out, int T = 2) {
    out.assign(static_cast<size_t>(T) * NH * HD, 0.0);
    const double scale = 1.0 / std::sqrt(static_cast<double>(HD));
    std::vector<double> sc(static_cast<size_t>(n));
    for (int t = 0; t < T; ++t) {
        const int nt = n - (T - 1) + t;   // positions 0 .. pos0+t, pos0 = n - T
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

std::vector<float> run_gpu(const Dev & d, int n, bool iso3, int ctx_max, bool split = false, int T = 2) {
    CK(op_attention(d.q, d.qfull, d.kc, d.vc, d.out, T, n - T, iso3, ctx_max, 0, nullptr, split ? d.part : nullptr));
    CK(cudaDeviceSynchronize());
    std::vector<float> o(static_cast<size_t>(T) * NH * HD);
    CK(cudaMemcpy(o.data(), d.out, o.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return o;
}

struct Err { double max_abs, rms_ref, rel; };
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
    const int max_n = argc > 1 ? std::atoi(argv[1]) : 131072;
    const bool time_only = argc > 2 && std::string(argv[2]) == "time";   // skip the parity cases (kernel tuning)
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> q(static_cast<size_t>(TMAX) * NH * HD), qfull(static_cast<size_t>(TMAX) * NH * 2 * HD);
    for (float & v : q) v = nd(rng);
    for (float & v : qfull) v = nd(rng);
    std::vector<float> K(static_cast<size_t>(max_n) * KVW), V(K.size());
    for (float & v : K) v = nd(rng);
    for (float & v : V) v = nd(rng);
    // Needle: K row 7 of kv group 0 = 3 * q of head 0, row 1 -> head 0 of the last row puts almost all weight on position 7.
    for (int d = 0; d < HD; ++d) K[static_cast<size_t>(7) * KVW + d] = 3.0f * q[static_cast<size_t>(1) * NH * HD + d];

    Dev d;
    CK(cudaMalloc(&d.q, q.size() * 4));
    CK(cudaMalloc(&d.qfull, qfull.size() * 4));
    CK(cudaMalloc(&d.out, q.size() * 4));
    CK(cudaMalloc(&d.src, sizeof(float) * 512 * KVW));
    CK(cudaMalloc(&d.part, sizeof(float) * op_attention_part_floats()));
    CK(cudaMemcpy(d.q, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.qfull, qfull.data(), qfull.size() * 4, cudaMemcpyHostToDevice));

    int fails = 0;
    for (int iso = 0; iso < 2; ++iso) {
        const bool iso3 = iso == 1;
        const size_t unit = iso3 ? 400 : static_cast<size_t>(KVW) * 4;
        CK(cudaMalloc(&d.kc, unit * max_n));
        CK(cudaMalloc(&d.vc, unit * max_n));
        store(K, d.kc, max_n, iso3, d.src);
        store(V, d.vc, max_n, iso3, d.src);
        const std::vector<float> Kh = host_rows(d.kc, max_n, iso3), Vh = host_rows(d.vc, max_n, iso3);

        // 1. fast vs tiled on the same bytes
        for (int n : {300, 4096, 6000, 8192}) {
            if (time_only) break;
            if (n > max_n) continue;
            const std::vector<float> fast = run_gpu(d, n, iso3, 8192), tiled = run_gpu(d, n, iso3, 16384);
            const Err e = compare(tiled, fast);
            const bool ok = e.rel < 1e-4;
            fails += !ok;
            std::printf("[%s] kv=%s n=%6d fast(ctx 8192) vs tiled(ctx 16384): max_abs %.3e rms %.3e rel %.3e\n",
                        ok ? "PASS" : "FAIL", iso3 ? "iso3" : "f32 ", n, e.max_abs, e.rms_ref, e.rel);
        }
        // 2. tiled vs double reference
        for (int n : {100, 4096, 32768, 65536, 131072}) {
            if (n > max_n || time_only) continue;
            const std::vector<float> tiled = run_gpu(d, n, iso3, 131072);
            std::vector<double> ref;
            reference(q, qfull, Kh, Vh, n, ref);
            {
                const std::vector<float> sp = run_gpu(d, n, iso3, 131072, true);
                const Err es = compare(sp, ref);
                const bool oks = es.rel < 1e-4;
                fails += !oks;
                std::printf("[%s] kv=%s n=%6d split-K(64 slices) vs double ref: max_abs %.3e rms %.3e rel %.3e\n",
                            oks ? "PASS" : "FAIL", iso3 ? "iso3" : "f32 ", n, es.max_abs, es.rms_ref, es.rel);
            }
            const Err e = compare(tiled, ref);
            // needle: head 0 of row 1 vs V row 7 (gated) -- the reference itself says how sharp it is
            const bool ok = e.rel < 1e-4;
            fails += !ok;
            std::printf("[%s] kv=%s n=%6d tiled(ctx 131072) vs double ref: max_abs %.3e rms %.3e rel %.3e (needle out[1][0][0] gpu %.6f ref %.6f)\n",
                        ok ? "PASS" : "FAIL", iso3 ? "iso3" : "f32 ", n, e.max_abs, e.rms_ref, e.rel,
                        tiled[static_cast<size_t>(1) * NH * HD], ref[static_cast<size_t>(1) * NH * HD]);
        }
        // 4. prefill: T query rows at pos0 = n - T (the queries of a chunk see a growing causal window)
        for (int T : {64, 37}) {
            for (int n : {100, 4096, 8193, 32768, 65536, 131072}) {
                if (n > max_n || time_only) continue;
                std::vector<double> ref;
                reference(q, qfull, Kh, Vh, n, ref, T);
                const std::vector<float> one = run_gpu(d, n, iso3, 131072, false, T);   // one slice, direct output
                const std::vector<float> sl = run_gpu(d, n, iso3, 131072, true, T);     // position slices + combine
                op_attention_set_tiled_ref(true);
                const std::vector<float> old = run_gpu(d, n, iso3, 131072, false, T);
                op_attention_set_tiled_ref(false);
                const Err e1 = compare(one, ref), e2 = compare(sl, ref), e3 = compare(old, ref);
                // T = 64 rows of long FP32 sums: the old tiled kernel itself reaches rel 1.7e-4 here (sequential FP32
                // accumulation over 8193 positions), so the bar is 3e-4 and "no worse than 1.5x the old kernel".
                const bool ok = e1.rel < 3e-4 && e2.rel < 3e-4 && e1.rel <= 1.5 * e3.rel + 1e-6 && e2.rel <= 1.5 * e3.rel + 1e-6;
                fails += !ok;
                std::printf("[%s] kv=%s n=%6d prefill T=%d vs double ref: reuse kernel rel %.3e, sliced rel %.3e, old tiled rel %.3e\n",
                            ok ? "PASS" : "FAIL", iso3 ? "iso3" : "f32 ", n, T, e1.rel, e2.rel, e3.rel);
            }
        }
        // timing: T = 64 at the largest n (the 128K prefill chunk)
        {
            const int T = 64, n = max_n;
            cudaEvent_t a, b;
            CK(cudaEventCreate(&a));
            CK(cudaEventCreate(&b));
            float ms[3] = {0, 0, 0};
            for (int k = 0; k < 3; ++k) {
                op_attention_set_tiled_ref(k == 2);
                float * prt = k == 1 ? d.part : nullptr;
                CK(op_attention(d.q, d.qfull, d.kc, d.vc, d.out, T, n - T, iso3, 131072, 0, nullptr, prt));
                CK(cudaEventRecord(a));
                const int reps = k == 2 ? 1 : 3;
                for (int r = 0; r < reps; ++r) CK(op_attention(d.q, d.qfull, d.kc, d.vc, d.out, T, n - T, iso3, 131072, 0, nullptr, prt));
                CK(cudaEventRecord(b));
                CK(cudaEventSynchronize(b));
                CK(cudaEventElapsedTime(&ms[k], a, b));
                ms[k] /= reps;
            }
            op_attention_set_tiled_ref(false);
            std::printf("[TIME] kv=%s n=%d T=%d: reuse kernel %.2f ms, reuse + slices %.2f ms, old tiled %.2f ms (%.1fx)\n",
                        iso3 ? "iso3" : "f32 ", n, T, ms[0], ms[1], ms[2], ms[2] / std::min(ms[0], ms[1]));
        }
        // decode timing: T = 1 and 5 query rows at the largest n, split-K kernel vs the v2 slice kernel (iso3 only)
        if (iso3) {
            for (int T : {1, 5}) {
                const int n = max_n;
                cudaEvent_t a, b;
                CK(cudaEventCreate(&a));
                CK(cudaEventCreate(&b));
                float ms[2] = {0, 0};
                for (int k = 0; k < 2; ++k) {
                    op_attention_set_dec2(k == 1);
                    CK(op_attention(d.q, d.qfull, d.kc, d.vc, d.out, T, n - T, iso3, 131072, 0, nullptr, d.part));
                    CK(cudaEventRecord(a));
                    for (int r = 0; r < 10; ++r) CK(op_attention(d.q, d.qfull, d.kc, d.vc, d.out, T, n - T, iso3, 131072, 0, nullptr, d.part));
                    CK(cudaEventRecord(b));
                    CK(cudaEventSynchronize(b));
                    CK(cudaEventElapsedTime(&ms[k], a, b));
                    ms[k] /= 10;
                }
                op_attention_set_dec2(true);
                std::printf("[TIME] kv=iso3 n=%d decode T=%d: split-K %.3f ms, v2 slices %.3f ms (%.2fx)\n", n, T, ms[0], ms[1], ms[0] / ms[1]);
            }
        }
        CK(cudaFree(d.kc));
        CK(cudaFree(d.vc));
    }
    std::printf("attn_long_test: %s (%d failing cases)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
