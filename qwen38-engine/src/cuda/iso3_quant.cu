// iso3 (IsoQuant 3-bit) KV store: one thread quantizes one 128-value block with the same sequential arithmetic as
// quantize_row_iso3_0_ref (PR #38). This file is compiled with --fmad=false (see CMakeLists.txt) so that the bytes
// are identical to the CPU reference and to libggml. The fp16 norm is written with integer bit operations.
#include "f16bits.hpp"
#include "ops.cuh"

#define QT_BEGIN(type, name, size) static __device__ const type name[size] = {
#define QT_END() };
#include "tables.inc"
#undef QT_BEGIN
#undef QT_END

namespace {

__device__ void quantize_block(const float * __restrict__ x, uint8_t * __restrict__ y) {
    float sum = 0.0f;
    for (int j = 0; j < 128; ++j) sum += x[j] * x[j];
    const float norm = sqrtf(sum);
    const float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;

    uint8_t qs[32], signs[16];
    for (int j = 0; j < 32; ++j) qs[j] = 0;
    for (int j = 0; j < 16; ++j) signs[j] = 0;
    float recon = 0.0f;
    for (int g = 0; g < 32; ++g) {
        const float w = iso3_quat_w[g], qx = iso3_quat_x[g], qy = iso3_quat_y[g], qz = iso3_quat_z[g];
        const float v0 = x[4 * g + 0] * inv_norm, v1 = x[4 * g + 1] * inv_norm, v2 = x[4 * g + 2] * inv_norm,
                    v3 = x[4 * g + 3] * inv_norm;
        const float r[4] = {
            w * v0 - qx * v1 - qy * v2 - qz * v3,
            w * v1 + qx * v0 + qy * v3 - qz * v2,
            w * v2 - qx * v3 + qy * v0 + qz * v1,
            w * v3 + qx * v2 - qy * v1 + qz * v0,
        };
        for (int c = 0; c < 4; ++c) {
            int idx = 0;
            for (int i = 0; i < 7; ++i) idx += r[c] > 0.5f * (iso3_codebook[i] + iso3_codebook[i + 1]);
            const int j = 4 * g + c;
            qs[j / 4] = static_cast<uint8_t>(qs[j / 4] | ((idx & 3) << (2 * (j % 4))));
            signs[j / 8] = static_cast<uint8_t>(signs[j / 8] | ((idx >> 2) << (j % 8)));
            recon += iso3_codebook[idx] * iso3_codebook[idx];
        }
    }
    const float recon_norm = sqrtf(recon);
    const uint16_t nh = f32_to_f16(recon_norm > 1e-10f ? norm / recon_norm : norm);
    y[0] = static_cast<uint8_t>(nh & 0xFF);
    y[1] = static_cast<uint8_t>(nh >> 8);
    for (int j = 0; j < 32; ++j) y[2 + j] = qs[j];
    for (int j = 0; j < 16; ++j) y[34 + j] = signs[j];
}

// src: [T][1024] floats (4 kv heads x 256). Block b = t*8 + g*2 + half goes to cache position pos0 + t.
__global__ void kv_store_iso3_kernel(const float * __restrict__ src, uint8_t * __restrict__ cache, int T, int pos0, const int * pos_dev) {
    if (pos_dev) pos0 = *pos_dev;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * 8) return;
    const int t = idx >> 3, rem = idx & 7;  // rem = g*2 + half
    quantize_block(src + static_cast<int64_t>(t) * 1024 + rem * 128, cache + (static_cast<int64_t>(pos0 + t) * 8 + rem) * 50);
}

__global__ void kv_store_f32_kernel(const float * __restrict__ src, float * __restrict__ cache, int T, int pos0, const int * pos_dev) {
    if (pos_dev) pos0 = *pos_dev;
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<int64_t>(T) * 1024) return;
    cache[static_cast<int64_t>(pos0) * 1024 + i] = src[i];
}

}  // namespace

cudaError_t op_kv_store(const float * src, void * cache, int T, int pos0, bool iso3, cudaStream_t s, const int * pos_dev) {
    if (iso3) {
        const int n = T * 8;
        kv_store_iso3_kernel<<<(n + 63) / 64, 64, 0, s>>>(src, static_cast<uint8_t *>(cache), T, pos0, pos_dev);
    } else {
        const int64_t n = static_cast<int64_t>(T) * 1024;
        kv_store_f32_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, s>>>(src, static_cast<float *>(cache), T, pos0, pos_dev);
    }
    return cudaGetLastError();
}
