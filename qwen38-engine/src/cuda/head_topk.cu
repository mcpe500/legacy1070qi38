// See head_topk.cuh. One block of 256 threads per row.
// Pass 1: block max + sum(exp(z - max)) in FP32 math with a double accumulator -> FP16 lse.
// Pass 2: K sequential block-argmax passes; each winner is masked to -INF in the (scratch) logits row.
// The row (V floats, ~1 MB at V = 248320) stays L2-resident across the K passes, so the reduction reads
// ~K MB per row from L2 rather than DRAM.
#include "head_topk.cuh"

#include "../common/f16bits.hpp"

namespace {

constexpr int kThreads = 256;

__global__ void head_topk_kernel(float * __restrict__ lg, int V, int K, int32_t * __restrict__ ids,
                                 uint16_t * __restrict__ vals, uint16_t * __restrict__ lse_out) {
    const int row = blockIdx.x;
    float * z = lg + static_cast<int64_t>(row) * V;
    const int tid = threadIdx.x;
    __shared__ float red_f[kThreads];
    __shared__ double red_d[kThreads];
    __shared__ float w_v[kThreads / 32];
    __shared__ int w_i[kThreads / 32];

    // 1. row max (block reduce)
    float m = -INFINITY;
    for (int i = tid; i < V; i += kThreads) m = fmaxf(m, z[i]);
    red_f[tid] = m;
    __syncthreads();
    for (int off = kThreads / 2; off > 0; off >>= 1) {
        if (tid < off) red_f[tid] = fmaxf(red_f[tid], red_f[tid + off]);
        __syncthreads();
    }
    const float mx = red_f[0];
    __syncthreads();

    // 2. sum of exp in double accumulation (expf math, double adds: ~1e-6 relative on 248K terms)
    double s = 0.0;
    for (int i = tid; i < V; i += kThreads) s += static_cast<double>(expf(z[i] - mx));
    red_d[tid] = s;
    __syncthreads();
    for (int off = kThreads / 2; off > 0; off >>= 1) {
        if (tid < off) red_d[tid] += red_d[tid + off];
        __syncthreads();
    }
    if (tid == 0) lse_out[row] = f32_to_f16(mx + static_cast<float>(log(red_d[0])));

    // 3. top-K descending: K block-argmax passes, ties -> the lower vocab id (matches a stable CPU sort)
    for (int k = 0; k < K; ++k) {
        float bv = -INFINITY;
        int bi = -1;
        for (int i = tid; i < V; i += kThreads) {
            const float v = z[i];
            if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
        }
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xffffffff, bv, off);
            const int oi = __shfl_down_sync(0xffffffff, bi, off);
            if (ov > bv || (ov == bv && oi >= 0 && (bi < 0 || oi < bi))) { bv = ov; bi = oi; }
        }
        if ((tid & 31) == 0) { w_v[tid / 32] = bv; w_i[tid / 32] = bi; }
        __syncthreads();
        if (tid == 0) {
            float fv = -INFINITY;
            int fi = -1;
            for (int w = 0; w < kThreads / 32; ++w)
                if (w_v[w] > fv || (w_v[w] == fv && w_i[w] >= 0 && (fi < 0 || w_i[w] < fi))) {
                    fv = w_v[w];
                    fi = w_i[w];
                }
            ids[static_cast<int64_t>(row) * K + k] = fi;
            vals[static_cast<int64_t>(row) * K + k] = f32_to_f16(fv);
            z[fi] = -INFINITY;   // mask the winner for the next pass (block-visible after the barrier below)
        }
        __syncthreads();
    }
}

}  // namespace

cudaError_t head_topk(const float * lg, int64_t V, int n, int K, int32_t * ids, uint16_t * vals,
                      uint16_t * lse_out, cudaStream_t s) {
    if (!lg || !ids || !vals || !lse_out) return cudaErrorInvalidValue;
    if (V < 1 || n < 1 || K < 1 || K > 256 || K > V) return cudaErrorInvalidValue;
    head_topk_kernel<<<n, kThreads, 0, s>>>(const_cast<float *>(lg), static_cast<int>(V), K, ids, vals,
                                            lse_out);
    return cudaGetLastError();
}
