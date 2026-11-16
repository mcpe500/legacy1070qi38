// Q8_0 weight x Q8_1 activation GEMV on sm_61 using __dp4a (INT32 accumulate per 32-value block,
// FP32 scaling). Weights are GGUF Q8_0 blocks: fp16 scale d plus 32 int8 values, 34 bytes per block.
// Activations are quantized on the fly per 32 values: int8 q and FP32 scale d (Q8_1 style, no sum term
// because the weights are symmetric).
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

struct Q8Scratch {            // preallocated once (static arena rule); never allocated in the token loop
    int8_t * qx = nullptr;    // [T][in]
    float  * dx = nullptr;    // [T][in/32]
    int64_t  T_max = 0, in_max = 0;
};

// Allocates scratch for up to T_max tokens and in_max input features. Returns cudaSuccess on success.
cudaError_t q8_scratch_alloc(Q8Scratch & s, int64_t T_max, int64_t in_max);
void q8_scratch_free(Q8Scratch & s);

// y[t][o] = sum_i W[o][i] * x[t][i], o in [0, out), t in [0, T).
// d_W: device pointer to out * (in/32) Q8_0 blocks (34 bytes each). d_x: [T][in] FP32. d_y: [T][out] FP32.
cudaError_t q8_0_gemv_dp4a(const uint8_t * d_W, int64_t in, int64_t out, const float * d_x, int64_t T,
                           float * d_y, const Q8Scratch & s, cudaStream_t stream = 0);
