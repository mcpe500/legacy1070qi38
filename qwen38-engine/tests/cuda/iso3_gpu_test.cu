// The GPU iso3 KV store must write exactly the bytes of the host reference (which is itself byte-identical to the
// GGML_TYPE_ISO3_0 of the patched libggml, see foundation_test). Also checks the position offset and the FP32 store.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "iso3_ref.hpp"
#include "ops.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

int main() {
    const int T = 37, pos0 = 5, ctx = 64;
    std::mt19937 rng(99);
    int fails = 0;
    long blocks = 0, bad_blocks = 0, bad_bytes = 0;
    const float scales[] = {0.02f, 1.0f, 30.0f};
    for (float sc : scales) {
        std::normal_distribution<float> nd(0.0f, sc);
        std::vector<float> src(static_cast<size_t>(T) * 1024);
        for (float & v : src) v = nd(rng);
        float * dsrc; uint8_t * dcache; float * dcf;
        CK(cudaMalloc(&dsrc, src.size() * 4));
        CK(cudaMalloc(&dcache, static_cast<size_t>(ctx) * 400));
        CK(cudaMalloc(&dcf, static_cast<size_t>(ctx) * 1024 * 4));
        CK(cudaMemset(dcache, 0xAB, static_cast<size_t>(ctx) * 400));
        CK(cudaMemset(dcf, 0, static_cast<size_t>(ctx) * 1024 * 4));
        CK(cudaMemcpy(dsrc, src.data(), src.size() * 4, cudaMemcpyHostToDevice));
        CK(op_kv_store(dsrc, dcache, T, pos0, true, 0));
        CK(op_kv_store(dsrc, dcf, T, pos0, false, 0));
        CK(cudaDeviceSynchronize());
        std::vector<uint8_t> got(static_cast<size_t>(ctx) * 400);
        std::vector<float> gotf(static_cast<size_t>(ctx) * 1024);
        CK(cudaMemcpy(got.data(), dcache, got.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(gotf.data(), dcf, gotf.size() * 4, cudaMemcpyDeviceToHost));
        for (int t = 0; t < T; ++t) {
            for (int b = 0; b < 8; ++b) {
                uint8_t want[ISO3_BYTES];
                iso3_quantize_block_ref(&src[static_cast<size_t>(t) * 1024 + b * 128], want);
                const uint8_t * g = &got[(static_cast<size_t>(pos0 + t) * 8 + b) * 50];
                ++blocks;
                int diff = 0;
                for (int i = 0; i < ISO3_BYTES; ++i) diff += (g[i] != want[i]);
                if (diff) { ++bad_blocks; bad_bytes += diff; }
            }
        }
        // positions outside [pos0, pos0+T) must be untouched
        for (int p = 0; p < pos0; ++p) for (int i = 0; i < 400; ++i) if (got[static_cast<size_t>(p) * 400 + i] != 0xAB) ++fails;
        for (int p = pos0 + T; p < ctx; ++p) for (int i = 0; i < 400; ++i) if (got[static_cast<size_t>(p) * 400 + i] != 0xAB) ++fails;
        for (int t = 0; t < T; ++t) for (int i = 0; i < 1024; ++i)
            if (gotf[static_cast<size_t>(pos0 + t) * 1024 + i] != src[static_cast<size_t>(t) * 1024 + i]) ++fails;
        cudaFree(dsrc); cudaFree(dcache); cudaFree(dcf);
    }
    std::printf("iso3 GPU store: %ld blocks, %ld differ from the host reference (%ld bytes), offset/FP32 failures %d\n", blocks,
                bad_blocks, bad_bytes, fails);
    const bool ok = bad_blocks == 0 && fails == 0;
    std::printf("result: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
