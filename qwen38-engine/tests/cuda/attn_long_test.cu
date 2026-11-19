// Long-context attention parity (exp/bonsai-128k-context).
//  1. Fast path vs tiled path on the same cache (n <= 8192): op_attention with ctx_max = 8192 (scores in shared memory)
//     against ctx_max = 16384 (tiled online softmax). Both KV formats.
//  3. Split-K decode path (ctx_max > 8192, T <= 8, partials + combine) vs the same double reference.
//  2. Tiled path vs a double-precision CPU reference at n = 4096, 32768, 65536 and 131072 positions, both KV formats
//     (iso3 reference = iso3_dequantize_block_ref of the bytes the GPU stored). A "needle" key at position 7 makes one head
//     attend almost only there, so a tile bookkeeping error (wrong max, lost tile) shows up as a large error.
// Usage: attn_long_test [max_n=131072]. Exit 0 only if every case passes.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "iso3_ref.hpp"
#include "ops.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

namespace {
constexpr int NH = 24, NKV = 4, HD = 256, KVW = NKV * HD;
constexpr int T = 2;   // two query rows: row 0 sees n-1 positions, row 1 sees n

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
               const std::vector<float> & V, int n, std::vector<double> & out) {
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

std::vector<float> run_gpu(const Dev & d, int n, bool iso3, int ctx_max, bool split = false) {
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
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> q(static_cast<size_t>(T) * NH * HD), qfull(static_cast<size_t>(T) * NH * 2 * HD);
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
            if (n > max_n) continue;
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
        CK(cudaFree(d.kc));
        CK(cudaFree(d.vc));
    }
    std::printf("attn_long_test: %s (%d failing cases)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
