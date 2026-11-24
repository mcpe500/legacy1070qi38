// FP16 Tensor Core GEMM for PTQ1_0 weights on sm_75+ (see ternary_f16.cuh).
// FP32 arithmetic rule: the FWHT rotation runs in FP32 registers; FP16 appears only as a storage format for
// the cublas operands (round-to-nearest-even bit conversion from src/common/f16bits.hpp, no half math), and
// the accumulation is CUBLAS_COMPUTE_32F. The PTQ1_0 scales are FP16 already, so value = digit * scale is
// exact in FP16 and this backend multiplies exactly the same weights the dp4a GEMV path reads.
#include "ternary_f16.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include <cublas_v2.h>

#include "../common/f16bits.hpp"
#include "../common/ternary_spec.hpp"

namespace {

constexpr int64_t kChunkRows = 32768;   // lm_head rows per dequant+GEMM chunk (~320 MiB of FP16 scratch)

#define CKF(expr)                                                                                                       \
    do {                                                                                                                \
        const cudaError_t e_ = (expr);                                                                                  \
        if (e_ != cudaSuccess) return e_;                                                                               \
    } while (0)

#define CKF_HOST(expr)                                                                                                  \
    do {                                                                                                                \
        const cudaError_t e_ = (expr);                                                                                  \
        if (e_ != cudaSuccess) { std::fprintf(stderr, "ternary_f16: %s\n", cudaGetErrorString(e_)); return nullptr; }    \
    } while (0)

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
#define F16_BODY_OK 1
#else
#define F16_BODY_OK 0
#endif

// Same butterfly as ternary_gemv.cu rotate_q8_kernel, FP32 in shared memory, FP16-bit store, no Q8 part.
__global__ void rotate_f16_kernel(const float * __restrict__ x, const float * __restrict__ signs,
                                  uint16_t * __restrict__ y, int width, int64_t x_row0, int64_t y_row0) {
#if F16_BODY_OK
    __shared__ float v[1024];
    const int t = blockIdx.y, blk = blockIdx.x, tid = threadIdx.x;
    const int64_t xb = x_row0 + static_cast<int64_t>(t) * width + static_cast<int64_t>(blk) * 1024;
    const int64_t yb = y_row0 + static_cast<int64_t>(t) * width + static_cast<int64_t>(blk) * 1024;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int k = tid + 256 * i;
        v[k] = x[xb + k] * signs[blk * 1024 + k];
    }
    __syncthreads();
    for (int h = 1; h < 1024; h <<= 1) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int p = tid + 256 * i;
            const int lo = (p / h) * 2 * h + (p % h), hi = lo + h;
            const float a = v[lo], b = v[hi];
            v[lo] = a + b;
            v[hi] = a - b;
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int k = tid + 256 * i;
        y[yb + k] = f32_to_f16(v[k] * 0.03125f);
    }
#endif
}

__constant__ uint8_t c_ptq1_code[256];   // uploaded once by ternary_f16_init from ternary::kPtq1Code

// One PTQ1_0 group (28 bytes -> 128 FP16 weights). Digit layout copied from the CPU reference
// (ternary_cpu.cpp ptq1_0_dot): bytes 0..15 carry 5 digits each for indices 16s+j, bytes 16..23 for
// 80 + k + 8s, bytes 24/25 carry 4 digits each for 120 + 2s / 121 + 2s; value = (digit - 1) * scale.
__global__ void dequant_ptq1_f16_kernel(const uint8_t * __restrict__ W, int64_t groups_per_row, int64_t in,
                                        uint16_t * __restrict__ out) {
#if F16_BODY_OK
    constexpr int P3[5] = {81, 27, 9, 3, 1};
    const int64_t g = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t row = g / groups_per_row, blk = g % groups_per_row;
    const uint8_t * src = W + g * ternary::kPtqGroupBytes;
    const float d = f16_to_f32(static_cast<uint32_t>(src[26]) | (static_cast<uint32_t>(src[27]) << 8));
    uint16_t * dst = out + row * in + blk * ternary::kPtqGroup;
    for (int j = 0; j < 16; ++j) {
        const int c = c_ptq1_code[src[j]];
        if (c > 242) { for (int s = 0; s < 5; ++s) dst[16 * s + j] = 0; continue; }
        for (int s = 0; s < 5; ++s) dst[16 * s + j] = f32_to_f16(static_cast<float>((c / P3[s]) % 3 - 1) * d);
    }
    for (int k = 0; k < 8; ++k) {
        const int c = c_ptq1_code[src[16 + k]];
        if (c > 242) { for (int s = 0; s < 5; ++s) dst[80 + k + 8 * s] = 0; continue; }
        for (int s = 0; s < 5; ++s) dst[80 + k + 8 * s] = f32_to_f16(static_cast<float>((c / P3[s]) % 3 - 1) * d);
    }
    for (int side = 0; side < 2; ++side) {
        const int c = c_ptq1_code[src[24 + side]];
        const int c4 = (c > 242 || c % 3) ? -1 : c / 3;   // the tail code word is 3 * c4
        constexpr int P4[4] = {27, 9, 3, 1};
        for (int s = 0; s < 4; ++s) dst[120 + 2 * s + side] = c4 < 0 ? 0 : f32_to_f16(static_cast<float>((c4 / P4[s]) % 3 - 1) * d);
    }
#endif
}

}  // namespace

struct TernaryF16 {
    uint16_t * w16 = nullptr;    // FP16 weight chunk scratch
    uint16_t * xf = nullptr;     // FP16 rotated activations [t_max][in]
    int64_t w16_cap = 0, xf_cap = 0;
    cublasHandle_t handle = nullptr;
};

bool ternary_f16_supported() {
    int dev = 0, major = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) return false;
    return major >= 7;
}

TernaryF16 * ternary_f16_init(int64_t w16_bytes, int64_t xf_bytes) {
    if (!ternary_f16_supported()) return nullptr;
    CKF_HOST(cudaMemcpyToSymbol(c_ptq1_code, ternary::kPtq1Code, 256));
    TernaryF16 * f = new TernaryF16();
    f->w16_cap = w16_bytes;
    f->xf_cap = xf_bytes;
    if (cudaMalloc(&f->w16, static_cast<size_t>(w16_bytes)) != cudaSuccess ||
        cudaMalloc(&f->xf, static_cast<size_t>(xf_bytes)) != cudaSuccess) {
        ternary_f16_free(f);
        return nullptr;
    }
    if (cublasCreate(&f->handle) != CUBLAS_STATUS_SUCCESS) {
        ternary_f16_free(f);
        return nullptr;
    }
    std::fprintf(stderr, "f16 gemm: sm_75+ path active (w16 scratch %.0f MiB, act scratch %.0f MiB)\n",
                 static_cast<double>(w16_bytes) / 1048576.0, static_cast<double>(xf_bytes) / 1048576.0);
    return f;
}

void ternary_f16_free(TernaryF16 * f) {
    if (!f) return;
    if (f->w16) cudaFree(f->w16);
    if (f->xf) cudaFree(f->xf);
    if (f->handle) cublasDestroy(f->handle);
    delete f;
}

cudaError_t ternary_f16_rotate(TernaryF16 * f, const float * x, const float * signs, int64_t in, int64_t T,
                               cudaStream_t s) {
    if (!f || in % 1024 || static_cast<int64_t>(T) * in > f->xf_cap) return cudaErrorInvalidValue;
    const dim3 grid(static_cast<unsigned>(in / 1024), static_cast<unsigned>(T));
    rotate_f16_kernel<<<grid, 256, 0, s>>>(x, signs, f->xf, static_cast<int>(in), 0, 0);
    return cudaGetLastError();
}

cudaError_t ternary_f16_gemm(TernaryF16 * f, const uint8_t * W, int64_t in, int64_t out, int64_t T, float * y,
                             cudaStream_t s) {
    if (!f || in % ternary::kPtqGroup) return cudaErrorInvalidValue;
    if (static_cast<int64_t>(T) * in > f->xf_cap) return cudaErrorInvalidValue;
    const int64_t groups_per_row = in / ternary::kPtqGroup;
    const int64_t row_bytes = groups_per_row * ternary::kPtqGroupBytes;
    if (cublasSetStream(f->handle, s) != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;
    const float alpha = 1.0f, beta0 = 0.0f;
    float beta_acc = 1.0f;
    for (int64_t r0 = 0; r0 < out; r0 += kChunkRows) {
        const int64_t rows = std::min(kChunkRows, out - r0);
        if (rows * in * 2 > f->w16_cap) return cudaErrorInvalidValue;
        const int64_t groups = rows * groups_per_row;
        const int threads = 256;
        dequant_ptq1_f16_kernel<<<static_cast<unsigned>((groups + threads - 1) / threads), threads, 0, s>>>(
            W + r0 * row_bytes, groups_per_row, in, f->w16);
        CKF(cudaGetLastError());
        // C [rows][T] (column-major view of y[r0..][T] row-major) = W_chunk^T . xf, FP32 accumulate
        const cublasStatus_t st = cublasGemmEx(f->handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(rows),
                                               static_cast<int>(T), static_cast<int>(in), &alpha, f->w16, CUDA_R_16F,
                                               static_cast<int>(in), f->xf, CUDA_R_16F, static_cast<int>(in),
                                               r0 == 0 ? &beta0 : &beta_acc, y + r0, CUDA_R_32F, static_cast<int>(out),
                                               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;
    }
    return cudaSuccess;
}
