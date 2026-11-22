# Drafter v2 contract (pre-training spec for the RTX PRO 6000 run), issue #55

Any draft head (MTP, EAGLE-3, DFlash) has to meet this contract before it is trained, so that qwen38-engine can run it losslessly on the GTX 1070 Ti next to Ternary Bonsai 2 27B PTQ1_0.
- Measured numbers come from `bench/results/bonsai_local_10h/`.
- Computed numbers are derived from the allocation code (`src/cuda/gpu_layers.cu`, `src/engine/engine.cpp`) and are marked as computed.

## 1. Embedding and hidden-state conventions (audit of the ProCreations "Bonsai MTP embedding fix")

The Bonsai GGUF stores `token_embd` in the Hadamard-rotated basis (latent). The ProCreations patch makes the llama.cpp MTP graph apply the inverse Hadamard transform and the sign vector before `enorm`.

qwen38-engine already does this in `Engine::mtp_forward` and `Engine::mtp_prefill`: `qmat_row_to_f32(tok_embd_)` followed by `hadamard_unrotate(.., signs_5120_)`. The GPU MTP block receives that host-built concat. A drafter therefore sees:

| Item | Contract |
|---|---|
| Token embedding input | **unrotated** (plain basis) row of `token_embd`, i.e. what HF `embed_tokens` returns after `convert_gguf_to_hf_bf16.py` |
| Hidden input | output-normed hidden of the main model (`output_norm(h)`, the vector the LM head multiplies), plain basis |
| Pairing | position p uses the hidden of position p-1 and the token at p; position 0 uses a zero hidden (llama.cpp convention) |
| Norm weights in the GGUF | **with** the +1 of zero-centred norms applied (`inject_mtp_gguf.py` adds it for HF `mtp.*` checkpoints) |
| LM head | shared with the main model (`output.weight`, rotated for the ternary GEMV); the drafter emits `shared_head_norm(h)` in the plain basis |

## 2. Tensor format by drafter type

| Drafter | Tensors the engine loads (GGUF names) | Supported GPU quant types | Status |
|---|---|---|---|
| MTP (Qwen3.8 nextn layout) | `blk.64.{attn_norm,post_attention_norm,attn_q,attn_k,attn_v,attn_output,attn_q_norm,attn_k_norm,ffn_gate,ffn_up,ffn_down}.weight`, `blk.64.nextn.{eh_proj,enorm,hnorm,shared_head_norm}.weight`; norms F32 | Q4_K, Q5_K, Q6_K, Q8_0, IQ4_NL, IQ4_XS, IQ3_S, Q3_K (dp4a kernels); `eh_proj` in = 10240 | **runs today**: as a sidecar with `--mtp-gguf` or inside the main GGUF |
| EAGLE-3 | needs 3 hidden taps (low/mid/high layers) concatenated, an fc 3·5120 → 5120, one decoder layer, a draft vocab map | same | **not implemented**: the engine exposes only the final normed hidden; taps must be added in `forward_ex` (the GPU stream is per layer, so a copy-out per tap layer costs about 20 KiB per token) |
| DFlash (block diffusion drafter, T=16 parallel draft) | drafter weights + a verify batch of 16-17 tokens | same | **not implemented**: needs the T > 5 lossless verify path of section 4 |

Shape rules for every drafter:
- inner sizes are multiples of 256 (K-quants) or 128 (PTQ1_0);
- the attention of a drafter layer uses the GQA layout of the main model (24 query heads, 4 KV heads, head dim 256) so it can reuse the attention kernels and the iso3 KV format;
- a different layout needs new kernels.

## 3. VRAM budget (cap 7373 MiB, process total)

Measured peaks (`nvidia-smi` sampled every 50 ms):

| Configuration | ctx | Peak MiB | Headroom MiB |
|---|---:|---:|---:|
| PTQ1_0, no drafter, GPU LM head | 4096 | 5995 | 1378 |
| + r3 Q4_K_M MTP (GPU), adaptive | 4096 | 6491 | 882 |
| + r3 Q8_0 MTP (GPU), adaptive | 4096 | 6649 | 724 |
| + r3 Q4_K_M MTP, 32K state run, adaptive | 33280 | 6877 | 496 |
| + r3 Q4_K_M MTP, 64K state run, adaptive | 66048 | 7309 | 64 |
| PTQ1_0, no drafter, GPU head, t_max 64 | 131072 | 7349 | 24 |
| + MTP | 131072 | does not fit with the GPU LM head | - (auto mode drops MTP) |

Computed per-item costs, from the allocation code:
- DeltaNet rollback snapshots (needed by every speculative scheme): 48 layers × 3.12 MiB = **149.6 MiB** fixed. Per verify row R add 48 × 64.4 KiB = **3.02 MiB**: R = 5 → 15.1 MiB, R = 17 → 51.3 MiB.
- Draft-block KV in iso3: 800 B per position. That is 25 / 50 / 100 MiB at 32K / 64K / 128K.
- Decode attention partial buffer (lossless T-invariant slicing, 100 slices). Today kSplitTMax 8: 12.1 MiB. For T = 17 verify rows: 17 × 24 × 100 × 258 × 4 B = **40.1 MiB**.
- Draft weights: r3 Q4_K_M 272.9 MiB, Q8_0 430.4 MiB, donor 334.7 MiB.

Consequences for training:
- At 64K, a drafter of up to about 300 MiB of weights fits with 5 verify rows. A T = 17 verify (DFlash / EAGLE-3 tree) adds about 36 MiB of rollback rows and about 28 MiB of partials, which exceeds the 64 MiB headroom. 64K with T = 17 therefore needs a smaller head (Q4_K, at most about 240 MiB) or a lower `t_max`.
- At 128K nothing fits next to the GPU LM head (24 MiB headroom). A drafter at 128K needs one of:
  - the LM head on the CPU (measured 2.46 tok/s with MTP: a loss);
  - a context of about 110K;
  - a head below about 50 MiB, which no 27B-class drafter is.

## 4. Lossless verification rules (measured)

A draft is accepted only if the main model's greedy token at that position equals it. For speculative decoding to give exactly the non-speculative text, a verify row has to compute bit-for-bit what the single-token decode of that position computes:

1. **GEMV:** satisfied for every T. The PTQ1_0 kernels reproduce the reference per-lane accumulation order at all T (`ternary_ab_test` 180/180 at T = 1..64), and so does the batched Q8_0 kernel.
2. **Attention:** single-token decode uses decode attention v2 (T ≤ 8). Its slice count must be the same for every verify size. Before the fix in this branch it was `512 / T` above 26K, and m2 / adaptive diverged from m0 after 16 tokens at 64K (`pre_fix_64k/`). It is now fixed at 100 slices for T ≤ 5. Verified lossless at 32K, at 64K (coding 65586-token context, ids `d7d3417a` for m0 / m2 / adaptive / PLD + adaptive) and at ctx 65536 with a short prompt.
3. **Verify batches of T = 6..17 are not lossless today:**
   - T = 6..8 still use `512 / T` slices.
   - T > 8 goes through the prefill attention kernel, which rounds differently from the decode kernel (measured with the trimmed-history case in `M4-README.md`: chunked vs full prefill differ).
   - A T = 16/17 drafter needs decode attention v2 extended to T ≤ 17 with the same fixed slice count, plus the 40.1 MiB partial buffer of section 3.
4. **DeltaNet rollback:** exact (bit-identical) for keep ≤ 5 (`--rollback-test`, Qwen and Bonsai + r3 sidecar). The rollback replays the saved conv inputs and gates of the kept rows, so its exactness does not depend on R. R = 17 needs `kRollT = 17` and the 51.3 MiB of section 3. It has not been run here; the 128K VRAM budget rules it out at that context.
5. **Prefix and checkpoint reuse:** a reused prefix has to be computed with the same chunk boundaries to give identical text (`x_stats.reuse`). That is a property of the engine, not of the drafter.

## 5. Acceptance targets from M1 (r3 Q4_K_M, this GPU)

Measured, greedy, 256 tokens:

| Setting | Context | Acceptance / speedup |
|---|---|---|
| Adaptive (up to 4 drafts) | 4K | coding 93.8% / 1.34x; chat EN 80.3% / 1.16x; chat ID 88.5% / 1.20x |
| m2 | 4K | repetitive 94.9% / 1.46x |
| Adaptive | 64K | 88.8% / 1.22x |

A new drafter is worth shipping on this card only if it beats these numbers within the VRAM rows above.
