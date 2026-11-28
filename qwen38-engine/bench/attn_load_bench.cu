// Load-pattern microbenchmark for the iso3 decode attention (no profiler counters on this card/driver).
// Reproduces only the memory access pattern of attention_kernel_decode_iso3 (src/cuda/ops.cu): grid (4 KV groups, nsl/4),
// 4 warps per block, one slice of positions per warp, per position one 100-byte (g*2+hf) region of the 400-byte row,
// lane L reads a 2-byte code word, a 1-byte sign byte and a 2-byte norm. K and V are both read (78 MiB at n = 102400).
// Variants:
//   v1  as the kernel: loads of one position, consumed immediately (compiler may or may not hoist)
//   v2  same loads, 16 positions issued before any is consumed (explicit memory-level parallelism)
//   v3  same loads, 32 positions issued before any is consumed
//   v4  ceiling for a contiguous group-major layout: each warp streams its slice with 16-byte lane loads (same byte count)
// FP32/integer only. Usage: attn_load_bench [n=102400] [nsl=100]
// Build: nvcc -arch=sm_61 -O3 bench/attn_load_bench.cu -o build/attn_load_bench
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

constexpr int NKV = 4, kWarps = 4;

__device__ __forceinline__ uint32_t one_pos(const uint8_t * kc, int j, int g, int hf, int gi0) {
    const uint8_t * kb = kc + (static_cast<int64_t>(j) * 8 + g * 2 + hf) * 50;
    const uint32_t qs = __ldg(reinterpret_cast<const unsigned short *>(kb + 2 + gi0));
    const uint32_t sg = __ldg(kb + 34 + (gi0 >> 1));
    const uint32_t nm = __ldg(reinterpret_cast<const unsigned short *>(kb));
    return qs * 3u + sg * 5u + nm;
}

template <int B>   // B positions in flight per tensor (0 = plain loop)
__global__ void __launch_bounds__(kWarps * 32, 3) k_pattern(const uint8_t * __restrict__ kc, const uint8_t * __restrict__ vc, uint32_t * out,
                                                            int n, int nsl) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = blockIdx.x, sl = blockIdx.y * kWarps + warp;
    const int chunk = ((n + nsl - 1) / nsl + 15) / 16 * 16;
    const int js = min(n, sl * chunk), je = min(n, js + chunk);
    const int hf = lane >> 4, gi0 = (2 * lane) & 31;
    uint32_t acc = 0;
    if constexpr (B == 0) {
        for (int j = js; j < je; ++j) acc ^= one_pos(kc, j, g, hf, gi0) + one_pos(vc, j, g, hf, gi0);
    } else {
        for (int j0 = js; j0 < je; j0 += B) {
            uint32_t a[B], b[B];
#pragma unroll
            for (int p = 0; p < B; ++p) { const int j = min(j0 + p, je - 1); a[p] = one_pos(kc, j, g, hf, gi0); }
#pragma unroll
            for (int p = 0; p < B; ++p) { const int j = min(j0 + p, je - 1); b[p] = one_pos(vc, j, g, hf, gi0); }
#pragma unroll
            for (int p = 0; p < B; ++p) acc ^= a[p] + b[p];
        }
    }
    if (acc == 0x12345678u) out[0] = acc;   // keep the loads alive
}

// ceiling: contiguous bytes per warp slice, uint4 per lane
__global__ void __launch_bounds__(kWarps * 32, 3) k_stream(const uint4 * __restrict__ kc, const uint4 * __restrict__ vc, uint32_t * out,
                                                           size_t n16_per_slice, int nsl) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int sl = (blockIdx.x * (gridDim.y) + blockIdx.y) * kWarps + warp;   // unique slice id
    if (sl >= nsl * NKV) return;
    const uint4 * kp = kc + sl * n16_per_slice, * vp = vc + sl * n16_per_slice;
    uint32_t acc = 0;
    for (size_t i = lane; i < n16_per_slice; i += 32 * 4) {
        uint4 a[4], b[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) { const size_t k = min(i + 32 * u, n16_per_slice - 1); a[u] = kp[k]; b[u] = vp[k]; }
#pragma unroll
        for (int u = 0; u < 4; ++u) acc ^= a[u].x + a[u].y + a[u].z + a[u].w + b[u].x + b[u].y + b[u].z + b[u].w;
    }
    if (acc == 0x12345678u) out[0] = acc;
}

template <typename F>
float best_ms(F f) {
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    f();   // warm up
    CK(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int i = 0; i < 10; ++i) {
        CK(cudaEventRecord(e0)); f(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); best = std::min(best, ms);
    }
    return best;
}

int main(int argc, char ** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 102400;
    const int nsl = argc > 2 ? std::atoi(argv[2]) : 100;
    const size_t bytes = static_cast<size_t>(n) * 400;
    uint8_t * kc, * vc; uint32_t * out;
    CK(cudaMalloc(&kc, bytes + 4096)); CK(cudaMalloc(&vc, bytes + 4096)); CK(cudaMalloc(&out, 64));
    std::vector<uint8_t> h(bytes);
    for (size_t i = 0; i < bytes; ++i) h[i] = static_cast<uint8_t>(i * 2654435761u >> 13);
    CK(cudaMemcpy(kc, h.data(), bytes, cudaMemcpyHostToDevice)); CK(cudaMemcpy(vc, h.data(), bytes, cudaMemcpyHostToDevice));
    const double useful = 2.0 * n * NKV * 100;   // K and V, 100 bytes per (position, group), as the kernel reads them
    std::printf("n=%d nsl=%d  grid=(%d,%d) x %d warps  useful bytes %.1f MiB\n", n, nsl, NKV, nsl / kWarps, kWarps, useful / 1048576.0);
    const dim3 grid(NKV, nsl / kWarps);
    auto rep = [&](const char * name, float ms) { std::printf("%-34s %8.3f ms  %7.1f GB/s (useful bytes)\n", name, ms, useful / (ms * 1e6)); };
    rep("v1 loop, consume at once", best_ms([&] { k_pattern<0><<<grid, kWarps * 32>>>(kc, vc, out, n, nsl); }));
    rep("v2 16 positions in flight", best_ms([&] { k_pattern<16><<<grid, kWarps * 32>>>(kc, vc, out, n, nsl); }));
    rep("v3 32 positions in flight", best_ms([&] { k_pattern<32><<<grid, kWarps * 32>>>(kc, vc, out, n, nsl); }));
    // ceiling: the same number of useful bytes, contiguous per slice (16-byte aligned), 4 groups x nsl slices
    const size_t per_slice16 = static_cast<size_t>(useful / 2 / (static_cast<double>(NKV) * nsl) / 16);
    const dim3 g4(NKV, nsl / kWarps);   // unique slice id = (x * y + ...) mapping inside the kernel; one slice per warp
    rep("v4 contiguous uint4 (ceiling)", best_ms([&] { k_stream<<<g4, kWarps * 32>>>(reinterpret_cast<const uint4 *>(kc), reinterpret_cast<const uint4 *>(vc), out, per_slice16, nsl); }));
    CK(cudaGetLastError());
    return 0;
}
