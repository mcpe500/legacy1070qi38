// Timing of the iso3 decode attention (op_attention, T = 1, long-context decode path) at a given position count.
// Used with ablated copies of ops.cu to find what limits the kernel (no profiler counters on this card).
// Also measures the P0.a sideband overhead (issue #70): the same run with the block-max sideband enabled
// (op_attention_set_blockmax) against the plain run, same build and clocks. The output checksum and bit hash
// must be identical in both modes: the sideband never touches the attention result.
// Usage: attn_dec_bench [n=102400] [ctx_max=131072] [reps=30]
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>

#include "ops.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

int main(int argc, char ** argv) {
    constexpr int NH = 24, NKV = 4, HD = 256, KVW = NKV * HD;
    const int n = argc > 1 ? std::atoi(argv[1]) : 102400;
    const int ctx_max = argc > 2 ? std::atoi(argv[2]) : 131072;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 30;
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> q(NH * HD), qfull(NH * 2 * HD), blk(static_cast<size_t>(512) * KVW);
    for (float & v : q) v = nd(rng);
    for (float & v : qfull) v = nd(rng);
    float * dq, * dqf, * dout, * dsrc, * dpart;
    void * kc, * vc;
    CK(cudaMalloc(&dq, q.size() * 4)); CK(cudaMalloc(&dqf, qfull.size() * 4)); CK(cudaMalloc(&dout, q.size() * 4));
    CK(cudaMalloc(&dsrc, blk.size() * 4)); CK(cudaMalloc(&dpart, sizeof(float) * op_attention_part_floats()));
    CK(cudaMalloc(&kc, static_cast<size_t>(n) * 400)); CK(cudaMalloc(&vc, static_cast<size_t>(n) * 400));
    CK(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dqf, qfull.data(), qfull.size() * 4, cudaMemcpyHostToDevice));
    for (int which = 0; which < 2; ++which)
        for (int p = 0; p < n; p += 512) {
            const int m = std::min(512, n - p);
            for (size_t i = 0; i < static_cast<size_t>(m) * KVW; ++i) blk[i] = nd(rng);
            CK(cudaMemcpy(dsrc, blk.data(), sizeof(float) * m * KVW, cudaMemcpyHostToDevice));
            CK(op_kv_store(dsrc, which ? vc : kc, m, p, true, 0));
        }
    CK(cudaDeviceSynchronize());
    auto run = [&] { CK(op_attention(dq, dqf, kc, vc, dout, 1, n - 1, true, ctx_max, 0, nullptr, dpart, true)); };
    run(); CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    auto time = [&](int r) {
        float best = 1e30f, tot = 0;
        for (int i = 0; i < r; ++i) {
            CK(cudaEventRecord(e0)); run(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); best = std::min(best, ms); tot += ms;
        }
        return std::make_pair(best, tot / r);
    };
    // plain run (sideband off: the pointer is null)
    const auto off = time(reps);
    // sideband run: re-arm the buffer before every rep (outside the timed window) so the CAS retry
    // pattern of a fresh decode step is what gets measured, not an already-saturated maximum
    const int nblk = (n + 63) / 64;
    std::vector<float> bmin(nblk * NKV, -INFINITY);
    float * dbmax;
    CK(cudaMalloc(&dbmax, bmin.size() * 4));
    CK(cudaMemcpy(dbmax, bmin.data(), bmin.size() * 4, cudaMemcpyHostToDevice));
    op_attention_set_blockmax(dbmax);
    float best = 1e30f, tot = 0;
    for (int i = 0; i < reps; ++i) {
        CK(cudaMemcpy(dbmax, bmin.data(), bmin.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaEventRecord(e0)); run(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); best = std::min(best, ms); tot += ms;
    }
    op_attention_set_blockmax(nullptr);
    std::vector<float> o(NH * HD); CK(cudaMemcpy(o.data(), dout, o.size() * 4, cudaMemcpyDeviceToHost));
    double cs = 0; for (float v : o) cs += v;
    unsigned long long h = 1469598103934665603ull;   // FNV-1a of the output bits: bit-identity between builds
    for (float v : o) { unsigned u; std::memcpy(&u, &v, 4); h = (h ^ u) * 1099511628211ull; }
    std::vector<float> bmax(bmin.size()); CK(cudaMemcpy(bmax.data(), dbmax, bmax.size() * 4, cudaMemcpyDeviceToHost));
    int nfin = 0; float gmax = -INFINITY; int gblk = -1;
    for (int b = 0; b < nblk; ++b) {
        float bm = -INFINITY;
        for (int g = 0; g < NKV; ++g) bm = std::max(bm, bmax[b * NKV + g]);
        if (bm > -INFINITY) ++nfin;
        if (bm > gmax) { gmax = bm; gblk = b; }
    }
    std::printf("n=%d ctx_max=%d  decode attention (1 layer, iso3, T=1)\n", n, ctx_max);
    std::printf("  sideband off: best %.3f ms, mean %.3f ms\n", off.first, off.second);
    std::printf("  sideband on : best %.3f ms, mean %.3f ms  (best %+.3f ms, %+.1f %%)\n", best, tot / reps, best - off.first, 100.0 * (best - off.first) / off.first);
    std::printf("  checksum %.6f, bits %016llx  (must match the off mode and the previous build)\n", cs, h);
    std::printf("  sideband: %d/%d blocks written, max %.4f at block %d/%d (blocks of 64 positions, %d KV groups)\n",
                nfin, nblk, gmax, gblk, nblk, NKV);
    return 0;
}
