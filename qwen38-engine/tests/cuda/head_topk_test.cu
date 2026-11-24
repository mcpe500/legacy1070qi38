// head_topk kernel vs a CPU reference: per row of FP32 logits, the FP16 logsumexp and the top-K ids/vals
// (descending like torch.topk, ties broken toward the lower vocab id — the same rule the kernel applies).
// Pure FP32/integer kernel: runs on sm_61 too, so the GTX 1070 Ti validates the exact reduction the T4
// dump uses. Includes tie-heavy inputs (clamped duplicates) and a row wider than L2.
#include "head_topk.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "../common/f16bits.hpp"

namespace {

struct Ref {
    std::vector<int32_t> ids;
    std::vector<uint16_t> vals;
    std::vector<uint16_t> lse;
};

Ref cpu_ref(const float * z, int V, int K) {
    Ref r;
    r.ids.resize(K);
    r.vals.resize(K);
    std::vector<int> idx(V);
    for (int i = 0; i < V; ++i) idx[i] = i;
    // stable sort: value descending, ties -> lower index first (the kernel's rule)
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return z[a] > z[b]; });
    for (int k = 0; k < K; ++k) {
        r.ids[k] = idx[k];
        r.vals[k] = f32_to_f16(z[idx[k]]);
    }
    float m = -INFINITY;
    for (int i = 0; i < V; ++i) m = std::fmax(m, z[i]);
    double s = 0;
    for (int i = 0; i < V; ++i) s += std::exp(static_cast<double>(z[i]) - m);    r.lse.push_back(f32_to_f16(m + static_cast<float>(std::log(s))));
    return r;
}

bool run_case(int n, int V, int K, bool ties, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 2.0f);
    std::vector<float> h(static_cast<size_t>(n) * static_cast<size_t>(V));
    for (auto & v : h) v = nd(rng);
    if (ties) {   // clamp 5% of the entries to a few shared values: exact ties in the top-K window
        for (size_t i = 0; i < h.size(); ++i)
            if (rng() % 20 == 0) h[i] = 6.0f + static_cast<float>(rng() % 4);
    }
    float * dg = nullptr;
    int32_t * di = nullptr;
    uint16_t * dv = nullptr;
    uint16_t * dl = nullptr;
    bool ok = true;
    do {
        if (cudaMalloc(&dg, h.size() * 4) != cudaSuccess || cudaMalloc(&di, static_cast<size_t>(n) * K * 4) != cudaSuccess ||
            cudaMalloc(&dv, static_cast<size_t>(n) * K * 2) != cudaSuccess || cudaMalloc(&dl, n * 2) != cudaSuccess) {
            std::printf("[FAIL] V=%d: alloc\n", V);
            ok = false;
            break;
        }
        if (cudaMemcpy(dg, h.data(), h.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess ||
            head_topk(dg, V, n, K, di, dv, dl, nullptr) != cudaSuccess) {
            std::printf("[FAIL] V=%d: kernel launch\n", V);
            ok = false;
            break;
        }
        std::vector<int32_t> ids(static_cast<size_t>(n) * K);
        std::vector<uint16_t> vals(static_cast<size_t>(n) * K), lse(n);
        if (cudaMemcpy(ids.data(), di, ids.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(vals.data(), dv, vals.size() * 2, cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(lse.data(), dl, lse.size() * 2, cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::printf("[FAIL] V=%d: download\n", V);
            ok = false;
            break;
        }
        int bad = 0;
        double worst_lse = 0;
        for (int r = 0; r < n; ++r) {
            const Ref ref = cpu_ref(h.data() + static_cast<size_t>(r) * static_cast<size_t>(V), V, K);
            const size_t o0 = static_cast<size_t>(r) * static_cast<size_t>(K);
            // The kernel and the reference share one tie rule (equal value -> lower vocab id first) and both
            // are deterministic, so the positional ids must match exactly; vals are FP16 roundings of the
            // selected FP32 values (compare bits, not the rounded float against the raw one).
            for (int k = 0; k < K; ++k) {
                const size_t o = o0 + static_cast<size_t>(k);
                const uint16_t got = vals[o];
                const uint16_t want = f32_to_f16(h[static_cast<size_t>(r) * static_cast<size_t>(V) + ids[o]]);
                const bool descending = k == 0 || f16_to_f32(vals[o]) <= f16_to_f32(vals[o - 1]);
                const bool val_ok = got == want && got == ref.vals[static_cast<size_t>(k)];
                if (ids[o] != ref.ids[static_cast<size_t>(k)] || !val_ok || !descending) {
                    if (bad < 4)
                        std::printf("  row %d k %d: id %d (ref %d), val bits %u/%u/%u%s\n", r, k, ids[o],
                                    ref.ids[static_cast<size_t>(k)], static_cast<unsigned>(got),
                                    static_cast<unsigned>(want), static_cast<unsigned>(ref.vals[static_cast<size_t>(k)]),
                                    descending ? "" : " [not descending]");
                    ++bad;
                }
            }
            worst_lse = std::fmax(worst_lse, std::fabs(static_cast<double>(f16_to_f32(lse[static_cast<size_t>(r)])) -
                                                       static_cast<double>(f16_to_f32(ref.lse[0]))));
        }
        if (worst_lse > 0.03) {
            std::printf("  V=%d: |dlse| %.4f > 0.03\n", V, worst_lse);
            ++bad;
        }
        ok = bad == 0;
        std::printf("[%s] n=%d V=%d K=%d ties=%d: rows ok, worst |dlse| %.4f\n", ok ? "PASS" : "FAIL", n, V, K,
                    ties ? 1 : 0, worst_lse);
    } while (false);
    if (dg) cudaFree(dg);
    if (di) cudaFree(di);
    if (dv) cudaFree(dv);
    if (dl) cudaFree(dl);
    return ok;
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
        std::printf("head_topk_test: SKIP (no CUDA device visible)\n");
        return 0;
    }
    bool ok = true;
    ok = run_case(3, 1000, 64, false, 1) && ok;         // tiny
    ok = run_case(4, 65536, 64, true, 2) && ok;         // ties across chunk boundary scale
    ok = run_case(8, 248320, 64, false, 3) && ok;       // real vocab, 8 rows = one head chunk
    ok = run_case(8, 248320, 64, true, 4) && ok;        // real vocab + ties
    ok = run_case(5, 130944, 16, false, 5) && ok;       // other K, odd V
    std::printf("head_topk_test: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
