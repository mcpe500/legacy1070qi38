---
name: qwen38-hybrid-arch
description: "Architecture facts and per-layer computation for Qwen3.8-27B, a hybrid dense model of 16 blocks of 3 Gated DeltaNet layers and 1 Gated Attention layer, with an in-checkpoint MTP head. Read before implementing or changing any layer, weight packer, MTP head, DeltaNet state, or the LM head."
---

# Qwen3.8-27B hybrid architecture

## Facts (model card and the GGUF we use)
| Item | Value | Source |
|---|---|---|
| Layers | 64 = 16 x [3 x (Gated DeltaNet -> FFN), 1 x (Gated Attention -> FFN)] | model card, CLAIMS C-01..C-05 |
| Hidden size | 5120 | model card |
| FFN | SwiGLU, intermediate 17408 | model card |
| Attention | 24 Q heads, 4 KV heads (6:1), head_dim 256, partial RoPE dim 64 | model card |
| DeltaNet | 48 V heads, 16 QK heads, head_dim 128, short causal conv | model card |
| Vocabulary | 248320 (padded) | model card |
| Context | 262144 | model card |
| GGUF blocks | block_count 65 = 64 layers + 1 MTP block (blk.64), nextn_predict_layers 1 | CLAIMS C-52, run-001 |
| Layer types in GGUF | full_attention_interval 4 | run-001 header check |
| output.weight | separate tensor from token_embd | run-001 |

Local files: `/home/ivan/models/27b/Qwen3.8-27B-UD-Q4_K_M.gguf` (16 GB) is the reference quant.
The vision projector is a separate file, `mmproj-BF16.gguf` (931146432 bytes, 0.87 GiB),
and is not part of the LM GGUF.

## Derived sizes (check every number with `gguf inspect` before use)
- FFN parameters per layer: 3 x 5120 x 17408 = 267,386,880. FFN is 69% of layer weights in
  Q4_K_M (9.26 of 13.46 GiB, run-002).
- LM head: 5120 x 248320 = 1,271,398,400 parameters. output.weight is 0.97 GiB in Q4_K_M (run-002).
- KV cache, f16: 16 attention layers x 2 (K, V) x 4 heads x 256 x 2 bytes = 64 KiB per token.
  At 8192 tokens: 512 MiB. With iso3 (50 bytes per 128 values): 100 MiB. See iso3-kv-and-mtp.
- DeltaNet matrix state, f32: 48 layers x 48 V heads x 128 x 128 x 4 bytes = 144 MiB. Add the conv
  history. llama.cpp budget default is 0.146 GiB. The brief's "~154 MB" is consistent with this.

## Decoder block (follow llama.cpp `src/models/qwen35.cpp`, not memory)
Each layer: pre-norm (RMSNorm), then the mixer, residual add, then post-norm, SwiGLU FFN, residual add.

**Gated Attention layer** (every 4th layer, index % 4 == 3):
1. One joint projection produces query and gate (2 x 24 x 256 values). Split them.
2. RMSNorm on Q and on K, per head (`attn_q_norm`, `attn_k_norm`).
3. RoPE on Q and K over the first 64 of 256 dims. llama.cpp uses MRoPE sections. Check the text-only
   case against `build_rope` before porting.
4. GQA attention: 24 query heads over 4 KV heads, causal, scaled by 1/sqrt(256).
5. Multiply the attention output by sigmoid(gate). Then the output projection.

**Gated DeltaNet layer** (the other 3 of every 4):
1. Projections for qkv, beta, alpha (decay input), and the output gate z.
2. beta = sigmoid(beta_raw). alpha = softplus(alpha_raw + dt_bias). Gate g = alpha x (-exp(A_log)).
3. Short causal depthwise conv over the qkv channels (`ssm_conv1d`, kernel `ssm_d_conv`), then SiLU.
4. L2-normalize q and k per head.
5. Recurrence over the state S (128 x 128 per V head). Implement it from the ggml op used by
   llama.cpp (`build_delta_net_base`), and check it against that CPU reference. Do not write the
   update rule from memory.
6. Output: RMSNorm(o) x SiLU(z), then the output projection.

**MTP head** (block 64, drafts the next token):
- Tensors `blk.64.nextn.{eh_proj, enorm, hnorm, shared_head_norm}` (CLAIMS C-52).
- The MTP block's own attention and FFN tensors are created in `qwen35.cpp` under `mtp_flags`.
  Read that code before implementing the MTP forward pass.
- Its KV cache is separate, one attention layer. Its type comes from `-ctkd/-ctvd` in llama.cpp,
  not from `-ctk/-ctv` (PR #38, finding on KV type inheritance).

## Packing rules
- Dense GPU layers: pack each weight tile for coalesced warp reads (skill pascal-sm61-cuda).
- CPU layers: pack for 256-bit AVX2 (8 floats or 32 int8 per register) with znver1 cache lines in mind.
- Keep embedding and LM head on the CPU side unless the memory budget (memory-budget-24gb) says they fit.
- The GGUF tokenizer metadata (`tokenizer.ggml.*`) is the source for the BPE tokenizer. Do not
  re-derive the vocabulary.

## Conflicts with the brief (decide before implementing)
- **Vision stripping saves ~1.2 GB:** not for the GGUF. The vision encoder is a separate
  `mmproj-*.gguf` file (0.87 GiB for BF16) that the LM does not load. The stripper only matters for a
  Hugging Face safetensors source that has `vision_config` weights inside the main checkpoint.
- **"~154 MB" state:** the matrix state is 144 MiB in f32. The remainder is conv history. Confirm
  against llama.cpp's `RS buffer size` log line.

## Checks before merge
- [ ] Every tensor name and shape matches `gguf inspect` on the reference GGUF.
- [ ] A 1-layer CPU run matches llama.cpp logits for the same input (see webui-and-parity-bench).
- [ ] The MTP block is exercised with the same KV type as the main model (see iso3-kv-and-mtp).
