// GPU reduction of the dump-targets head output: per row of full-vocabulary FP32 logits, the stable FP32
// logsumexp and the top-K ids/vals (FP16, descending like torch.topk, ties broken toward the lower vocab
// id). Runs on the head's logits buffer directly in VRAM, so the D2H transfer shrinks from V*4 bytes per
// row to K*6+2 (~16 KB per row at K=64, V=248320) and the 2-vCPU heap of the old CPU path disappears.
// Pure FP32/integer math: compiles and runs on sm_61 too (arch_compile_check covers it).
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// lg: [n][V] FP32 logits (MUTATED: processed entries are masked to -INF, the buffer is engine scratch);
// ids/vals: [n][K]; lse: [n] FP16. K <= 256.
cudaError_t head_topk(const float * lg, int64_t V, int n, int K, int32_t * ids, uint16_t * vals,
                      uint16_t * lse_out, cudaStream_t s);
