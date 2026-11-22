// Roofline micro-benchmark of the quantized GEMVs of the engine (issue #55): every weight format with a GPU kernel, at
// T = 1, 4, 16 and 64 activation rows, on the ffn_up shape (5120 x 17408), against two measured ceilings of this card:
//   - DRAM read bandwidth: a coalesced float4 read-and-reduce kernel over 512 MiB (and the nominal 256 GB/s of GDDR5),
//   - INT8 dot throughput: back-to-back independent __dp4a chains (one IDP.4A = 4 MACs), and FP32 FFMA throughput.
// Per format and T it prints the time per call, the effective weight bandwidth (weight bytes / time, % of the measured
// ceiling and of 256 GB/s), and the arithmetic rate (2 * in * out * T ops / time, % of the dp4a ceiling in int8 MACs).
// Weights are random bytes with valid fp16 scales (timing only; the numerics are checked by ternary_ab_test, gemv_parity
// and q8_gemv_parity). Formats without a GPU kernel in the engine (IQ2_XXS / IQ2_XS / IQ2_S, Q2_K) are listed as such.
// Usage: build/quant_roofline_bench [csv_out] [--fast]
//   --fast also measures the opt-in PTQ1_0 prefill path (ternary_gemv.cu fast = true) and runs a paired A/B
//   section for it: the default and the fast kernel are timed alternately in the same process, because the
//   clock of this card drifts several percent between processes and an A/B split across runs proves nothing.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "dp4a_gemv.cuh"
#include "gemv.cuh"
#include "q8_gemv.cuh"
#include "ternary_gemv.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

namespace {

__global__ void read_kernel(const float4 * __restrict__ p, size_t n, float * out) {
    float acc = 0.0f;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const float4 v = __ldg(p + i);
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 12345.678f) *out = acc;   // keeps the loads alive
}

__global__ void dp4a_kernel(int * out, int iters, int seed) {
    int a0 = seed + threadIdx.x, a1 = a0 * 3, a2 = a0 * 5, a3 = a0 * 7, a4 = a0 * 11, a5 = a0 * 13, a6 = a0 * 17, a7 = a0 * 19;
    const int b = 0x01020304 ^ seed;
    for (int i = 0; i < iters; ++i) {
        a0 = __dp4a(a0, b, a0); a1 = __dp4a(a1, b, a1); a2 = __dp4a(a2, b, a2); a3 = __dp4a(a3, b, a3);
        a4 = __dp4a(a4, b, a4); a5 = __dp4a(a5, b, a5); a6 = __dp4a(a6, b, a6); a7 = __dp4a(a7, b, a7);
    }
    const int r = a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7;
    if (r == 0x7fffffff) *out = r;
}

__global__ void ffma_kernel(float * out, int iters, float seed) {
    float a0 = seed + threadIdx.x, a1 = a0 * 1.1f, a2 = a0 * 1.2f, a3 = a0 * 1.3f, a4 = a0 * 1.4f, a5 = a0 * 1.5f, a6 = a0 * 1.6f, a7 = a0 * 1.7f;
    const float b = 0.999f, c = 0.001f;
    for (int i = 0; i < iters; ++i) {
        a0 = fmaf(a0, b, c); a1 = fmaf(a1, b, c); a2 = fmaf(a2, b, c); a3 = fmaf(a3, b, c);
        a4 = fmaf(a4, b, c); a5 = fmaf(a5, b, c); a6 = fmaf(a6, b, c); a7 = fmaf(a7, b, c);
    }
    const float r = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    if (r == 12345.678f) *out = r;
}

struct Fmt {
    const char * name;
    int kind;        // 0 PTQ1_0, 1 Q2_0, 2 dp4a K/IQ family, 3 Q8_0, -1 no GPU kernel
    int type;        // ggml type id
    int blk_vals;    // values per block
    int blk_bytes;   // bytes per block
    int scale_off;   // byte offset of an fp16 scale inside the block (made valid)
};

}  // namespace

int main(int argc, char ** argv) {
    const char * csv_path = nullptr;
    bool fast_mode = false;   // --fast: measures the opt-in PTQ1_0 prefill path (relaxed FP32 group order) too
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--fast") == 0) fast_mode = true;
        else csv_path = argv[i];
    }
    int dev = 0;
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, dev));
    std::printf("device %s, sm_%d%d, %d SMs, %.0f MHz, bus %d bit\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                prop.clockRate / 1000.0, prop.memoryBusWidth);
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
        return 1000.0 * ms / reps;   // us
    };

    // ---- ceilings
    const size_t nbytes = 512ull << 20;
    float4 * big;
    float * sink;
    int * isink;
    CK(cudaMalloc(&big, nbytes));
    CK(cudaMemset(big, 0, nbytes));
    CK(cudaMalloc(&sink, 4));
    CK(cudaMalloc(&isink, 4));
    const int nblk = prop.multiProcessorCount * 8;
    const double us_rd = timed([&] { read_kernel<<<nblk, 256>>>(big, nbytes / 16, sink); }, 10);
    const double bw_read = nbytes / (us_rd * 1e3);
    const int iters = 4096;
    const int blocks = prop.multiProcessorCount * 16;
    const double us_dp = timed([&] { dp4a_kernel<<<blocks, 256>>>(isink, iters, 1); }, 5);
    const double dp4a_gops = static_cast<double>(blocks) * 256 * iters * 8 / (us_dp * 1e3);   // IDP.4A per ns = G instr/s
    const double us_fm = timed([&] { ffma_kernel<<<blocks, 256>>>(sink, iters, 1.0f); }, 5);
    const double ffma_g = static_cast<double>(blocks) * 256 * iters * 8 / (us_fm * 1e3);
    const double int8_macs = dp4a_gops * 4;   // G int8 MAC/s
    std::printf("ceiling: DRAM read %.1f GB/s (%.1f%% of 256 GB/s nominal), dp4a %.0f G instr/s = %.0f G int8 MAC/s, FFMA %.0f G/s\n",
                bw_read, 100.0 * bw_read / 256.0, dp4a_gops, int8_macs, ffma_g);
    CK(cudaFree(big));

    // ---- formats
    const Fmt base_fmts[] = {
        {"PTQ1_0", 0, 143, 128, 28, 26}, {"Q2_0 (PQ2_0)", 1, 42, 64, 18, 0},
        {"Q3_K", 2, 11, 256, 110, 108}, {"Q4_K", 2, 12, 256, 144, 0}, {"Q5_K", 2, 13, 256, 176, 0}, {"Q6_K", 2, 14, 256, 210, 208},
        {"Q8_0", 3, 8, 32, 34, 0}, {"IQ4_NL", 2, 20, 32, 18, 0}, {"IQ4_XS", 2, 23, 256, 136, 0}, {"IQ3_S", 2, 21, 256, 110, 0},
        {"Q2_K", 2, 10, 256, 84, 80},                                  // kernel since exp/bonsai-overnight-v2 (M4)
        {"IQ2_XXS", -1, 16, 256, 66, 0}, {"IQ2_XS", -1, 17, 256, 74, 0}, {"IQ2_S", -1, 22, 256, 82, 0},
    };
    // kind 4 = the same PTQ1_0 weights through the opt-in fast prefill path (fast = true), measured next to kind 0
    const Fmt fast_fmt = {"PTQ1_0 fast", 4, 143, 128, 28, 26};
    std::vector<Fmt> fmts(std::begin(base_fmts), std::end(base_fmts));
    if (fast_mode) fmts.insert(fmts.begin() + 1, fast_fmt);
    const int64_t in = 5120, out = 17408;
    const int Ts[] = {1, 4, 16, 64};
    ternary_ptq1_init();
    ActQ8 act;
    CK(actq8_alloc(act, 64, in));
    Q8Scratch q8s;
    CK(q8_scratch_alloc(q8s, 64, in));
    std::mt19937 rng(5);
    std::vector<float> x(64 * in);
    std::normal_distribution<float> nd;
    for (float & v : x) v = nd(rng);
    float * dx;
    float * y;
    CK(cudaMalloc(&dx, x.size() * 4));
    CK(cudaMalloc(&y, sizeof(float) * 64 * out));
    CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    CK(actq8_quantize(dx, in, 64, act, 0));
    int q8_fail = 0;
    FILE * csv = csv_path ? std::fopen(csv_path, "w") : nullptr;
    if (csv) std::fprintf(csv, "format,bpw,T,us,weight_GBps,pct_measured_bw,pct_256,int8_GMACs,pct_dp4a\n");
    std::printf("\n%-13s %5s %3s %10s %9s %7s %7s %10s %7s\n", "format", "bpw", "T", "us", "GB/s", "%meas", "%256", "GMAC/s", "%dp4a");
    for (const Fmt & f : fmts) {
        const size_t bytes = static_cast<size_t>(in / f.blk_vals) * f.blk_bytes * out;
        const double bpw = 8.0 * f.blk_bytes / f.blk_vals;
        if (f.kind < 0) {
            std::printf("%-13s %5.2f   -  no GPU kernel in the engine (CPU path only)\n", f.name, bpw);
            if (csv) std::fprintf(csv, "%s,%.3f,,,,,,,\n", f.name, bpw);
            continue;
        }
        std::vector<uint8_t> w(bytes);
        for (size_t i = 0; i < bytes; ++i) w[i] = static_cast<uint8_t>(rng());
        // a valid small fp16 scale (0x2000 = 2^-7) at the format's scale offset of every block; for the K/IQ formats with
        // several scales the remaining random bytes only change values, not the speed
        for (size_t b = 0; b < bytes / f.blk_bytes; ++b) { w[b * f.blk_bytes + f.scale_off] = 0x00; w[b * f.blk_bytes + f.scale_off + 1] = 0x20; }
        uint8_t * dw;
        CK(cudaMalloc(&dw, bytes));
        CK(cudaMemcpy(dw, w.data(), bytes, cudaMemcpyHostToDevice));
        if (f.kind == 0 || f.kind == 4) {
            unsigned long long * cnt;
            CK(cudaMalloc(&cnt, 8));
            CK(cudaMemset(cnt, 0, 8));
            CK(ternary_ptq1_canonicalize(dw, static_cast<int64_t>(out) * (in / 128), cnt, 0));
            CK(cudaFree(cnt));
        }
        for (int T : Ts) {
            auto run = [&]() {
                if (f.kind == 0) CK(gemv_ptq1_0_dp4a(dw, in, out, act, T, y, 0, false));
                else if (f.kind == 4) CK(gemv_ptq1_0_dp4a(dw, in, out, act, T, y, 0, true));
                else if (f.kind == 1) CK(gemv_q2_0_dp4a(dw, in, out, act, T, y, 0));
                else if (f.kind == 2) CK(gemv_dp4a(f.type, dw, in, out, act, T, y, 0));
                else CK(q8_0_gemv_dp4a(dw, in, out, dx, T, y, q8s, 0));
            };
            const double us = timed(run, T >= 16 ? 5 : 20);
            if (f.kind == 3) {   // batched Q8_0 kernel vs the per-(row, token) reference: byte-equal outputs
                std::vector<float> y1(static_cast<size_t>(T) * out), y2(y1.size());
                CK(q8_0_gemv_dp4a(dw, in, out, dx, T, y, q8s, 0));
                CK(cudaMemcpy(y1.data(), y, y1.size() * 4, cudaMemcpyDeviceToHost));
                CK(q8_0_gemv_dp4a_ref(dw, in, out, dx, T, y, q8s, 0));
                CK(cudaMemcpy(y2.data(), y, y2.size() * 4, cudaMemcpyDeviceToHost));
                const bool same = std::memcmp(y1.data(), y2.data(), y1.size() * 4) == 0;
                std::printf("  Q8_0 T=%d batched vs reference: %s\n", T, same ? "bit-identical" : "DIFFER");
                if (!same) ++q8_fail;
            }
            const double gbs = bytes / (us * 1e3);
            const double gmac = static_cast<double>(in) * out * T / (us * 1e3);
            std::printf("%-13s %5.2f %3d %10.1f %9.1f %6.1f%% %6.1f%% %10.0f %6.1f%%\n", f.name, bpw, T, us, gbs, 100.0 * gbs / bw_read,
                        100.0 * gbs / 256.0, gmac, 100.0 * gmac / int8_macs);
            if (csv) std::fprintf(csv, "%s,%.3f,%d,%.1f,%.1f,%.1f,%.1f,%.0f,%.1f\n", f.name, bpw, T, us, gbs, 100.0 * gbs / bw_read,
                                  100.0 * gbs / 256.0, gmac, 100.0 * gmac / int8_macs);
        }
        CK(cudaFree(dw));
    }
    if (csv) std::fclose(csv);
    // ---- paired A/B of the default PTQ1_0 prefill path against the opt-in fast one (same process, alternating calls,
    // so both samples see the same clock; the fastest tile for fast is chosen by QWEN_PTQ1_FAST, default <16,2>).
    if (fast_mode) {
        const size_t bytes = static_cast<size_t>(in / 128) * 28 * out;
        std::vector<uint8_t> w(bytes);
        for (size_t i = 0; i < bytes; ++i) w[i] = static_cast<uint8_t>(rng());
        for (size_t b = 0; b < bytes / 28; ++b) { w[b * 28 + 26] = 0x00; w[b * 28 + 27] = 0x20; }
        uint8_t * dw;
        CK(cudaMalloc(&dw, bytes));
        CK(cudaMemcpy(dw, w.data(), bytes, cudaMemcpyHostToDevice));
        unsigned long long * cnt;
        CK(cudaMalloc(&cnt, 8));
        CK(cudaMemset(cnt, 0, 8));
        CK(ternary_ptq1_canonicalize(dw, static_cast<int64_t>(out) * (in / 128), cnt, 0));
        CK(cudaFree(cnt));
        std::printf("\npaired PTQ1_0 prefill A/B on %lldx%lld (default vs fast, alternating, 7 rounds x 5 calls):\n", (long long)in, (long long)out);
        {   // occupancy of the prefill tiles, to say whether the issue rate or the occupancy is the limit
            int b16 = 0, b8 = 0, b32 = 0, b16r1 = 0, regs = 0, shb = 0, mt = 0;
            ptq1_prefill_occupancy(&b16, &b8, &b32, &b16r1, &regs, &shb, &mt);
            std::printf("occupancy (blocks/SM of 19, %d-thread blocks): <16,2> %d, <8,4> %d, <32,1> %d, <16,1> %d | <16,2> regs %d, shared %d B\n",
                        mt, b16, b8, b32, b16r1, regs, shb);
        }
        std::printf("%5s %12s %12s %9s %10s %10s\n", "T", "default us", "fast us", "speedup", "def GB/s", "fast GB/s");
        float * ydv, * yfv;
        CK(cudaMalloc(&ydv, sizeof(float) * 64 * out));
        CK(cudaMalloc(&yfv, sizeof(float) * 64 * out));
        for (int T : Ts) {
            double best_def = 1e30, best_fast = 1e30;
            char dchk[32] = "", fchk[32] = "";
            for (int r = 0; r < 7; ++r) {   // the best round of each side is the least noisy clock sample
                const double ud = timed([&] { CK(gemv_ptq1_0_dp4a(dw, in, out, act, T, ydv, 0, false)); }, 5);
                const double uf = timed([&] { CK(gemv_ptq1_0_dp4a(dw, in, out, act, T, yfv, 0, true)); }, 5);
                float v0 = 0, f0 = 0;
                CK(cudaMemcpy(&v0, ydv, 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(&f0, yfv, 4, cudaMemcpyDeviceToHost));
                if (ud < best_def) { best_def = ud; std::snprintf(dchk, sizeof(dchk), "%.6f", v0); }
                if (uf < best_fast) { best_fast = uf; std::snprintf(fchk, sizeof(fchk), "%.6f", f0); }
            }
            const double same = std::strcmp(dchk, fchk) == 0;
            std::printf("%5d %12.1f %12.1f %8.3fx %10.1f %10.1f  y[0] def %s fast %s %s\n", T, best_def, best_fast, best_def / best_fast,
                        bytes / (best_def * 1e3), bytes / (best_fast * 1e3), dchk, fchk, same ? "(equal)" : "(DIFFER)");
        }
        CK(cudaFree(ydv)); CK(cudaFree(yfv)); CK(cudaFree(dw));
    }
    return q8_fail ? 1 : 0;
}
