// FP32 CUDA operators of the Qwen3.8-27B layers (sm_61). Every activation, norm, gate, RoPE rotation, DeltaNet state
// update and attention score is computed in FP32. No half-precision arithmetic is used anywhere in these kernels.
// All launchers enqueue on the given stream and return cudaGetLastError().
#pragma once
#include <cuda_runtime.h>

#include <cstdint>

// y[r] = x[r] / sqrt(mean(x[r]^2) + eps) * w   (w may be null). Rows are `n` floats; x may alias y.
cudaError_t op_rmsnorm_rows(const float * x, const float * w, float * y, int n, int rows, float eps, cudaStream_t s);
cudaError_t op_add_inplace(float * x, const float * y, int64_t n, cudaStream_t s);          // x += y
cudaError_t op_silu_mul(float * g, const float * u, int64_t n, cudaStream_t s);              // g = silu(g) * u
// ssm_out of a Gated DeltaNet layer (gdn_v_grouped=1): permute the value heads of attn_o [T][6144] from tiled
// order v into grouped order v' = 3 * (v % 16) + v / 16, in place into dst (src may not alias dst).
cudaError_t op_gdn_v_permute(const float * src, float * dst, int T, cudaStream_t s);

// Gated DeltaNet. T tokens, 48 value heads, 16 key heads, head size 128, conv kernel 4.
cudaError_t op_dn_prep(const float * alpha, const float * beta_raw, const float * dt_bias, const float * ssm_a,
                       float * gate, float * beta, int T, cudaStream_t s);
cudaError_t op_dn_conv(const float * qkv, const float * conv_w, float * conv_state, float * out, int T, cudaStream_t s);
// state_t holds, per value head, the transposed matrix Mt[i][j] = M[j][i] (j value index, i key index), 128 x 128.
cudaError_t op_dn_rec(const float * conv_out, const float * gate, const float * beta, const float * z,
                      const float * ssm_norm, float * state_t, float * attn_o, int T, float eps, cudaStream_t s);
// Reference version of op_dn_rec (state kept in global memory), for the A/B test only.
cudaError_t op_dn_rec_ref(const float * conv_out, const float * gate, const float * beta, const float * z,
                      const float * ssm_norm, float * state_t, float * attn_o, int T, float eps, cudaStream_t s);

// Gated Attention. qfull: [T][24 heads][q(256) | gate(256)]. kproj: [T][4][256].
// Writes qbuf [T][24][256] (RMSNorm + RoPE) and kout [T][4][256] (RMSNorm + RoPE). The RoPE angle tables
// [ctx_max][32] are generated once by op_rope_table on the device.
cudaError_t op_rope_table(float * cos_tab, float * sin_tab, int n_pos, float theta_scale, cudaStream_t s);
cudaError_t op_qk_prep(const float * qfull, const float * kproj, const float * qnorm, const float * knorm, float * qbuf,
                       float * kout, int T, int pos0, const float * rope_cos, const float * rope_sin, float eps, cudaStream_t s,
                       const int * pos_dev = nullptr);
// Stores T rows of [4][256] values at cache positions pos0.. . iso3: 8 blocks of 50 bytes per position, else FP32.
// pos_dev (optional): device pointer to the position, read by the kernel instead of pos0 (needed to replay a CUDA graph).
cudaError_t op_kv_store(const float * src, void * cache, int T, int pos0, bool iso3, cudaStream_t s, const int * pos_dev = nullptr);
// Reference iso3 store (one thread per 128-value block), for tests/cuda/iso3_gpu_test.cu.
cudaError_t op_kv_store_ref(const float * src, void * cache, int T, int pos0, cudaStream_t s);
// Causal GQA attention (24 query heads over 4 kv heads) with FP32 dequantization of the iso3 cache inside the kernel.
// ctx_max <= 8192: scores in shared memory; above: tiled online softmax (256 positions per tile), up to 131072.
cudaError_t op_attention(const float * qbuf, const float * qfull, const void * kcache, const void * vcache, float * attn_o,
                         int T, int pos0, bool iso3, int ctx_max, cudaStream_t s, const int * pos_dev = nullptr,
                         float * part = nullptr, bool long_kernels = false);
// Scratch for the split-K decode attention (ctx_max > 8192, T <= 8): floats `part` must hold. Without it the tiled kernel runs.
size_t op_attention_part_floats();
// Prefill (ctx_max > 8192, T > 8) uses the query-tiled K/V-reuse kernel (one block per KV group and 8 query tokens).
// Tests and benchmarks can switch back to the per-(token, head) tiled reference kernel with this flag.
void op_attention_set_tiled_ref(bool on);
// Decode attention (T <= 8, long-context path, iso3): true = the v2 slice kernel (default), false = the split-K kernel.
void op_attention_set_dec2(bool on);
