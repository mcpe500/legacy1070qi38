# Smart Precompute + Student-Only Training (T4 16 GB / local GPU) — design and budgets

Goal: keep the MTP/df3 draft-head quality of the live A100 pipeline while removing the 50.1 GiB BF16 teacher from the
training loop, so every iteration after one burst runs on a Colab T4 (16 GB) or the local GTX 1070 Ti.

## 1. Code audit of `train_bonsai_mtp.py` (facts, not estimates)

| Component | What the code does today | Where |
|---|---|---|
| teacher hidden tap | `Teacher.hidden()` runs the model once per batch and returns `H` = **output-normed last hidden state for every position** (a forward hook on `model.norm`) | `Teacher.hidden`, line 287 |
| student input | step 1 feeds `rms(embed(x_{i+1}), enorm)` + `rms(H[i], hnorm)`; steps k>1 feed the student's own feature `f^(k-1)` + `embed(x_{i+k})` | `unrolled_loss`, line 351 |
| loss target | teacher logits are **recomputed per chunk from the target hidden** `H[i+k]` through the shared frozen lm_head, full 248,320-wide softmax KL, plus cosine against `H[i+k]` | `chunk_loss`, line 330 |
| df3 taps | the `df3` drafter is the **same one-layer student** run with K = `--df3-block` (16) unrolled steps; `--df3-layers` is metadata only, a real 2-3 layer head needs M4-style engine work | `unroll_K`, line 344 |
| step cost split (A100 run, PR #60 branch) | 71-76 s/step with the PyTorch conv fallback vs 23-25 s/step with `causal-conv1d` (24.19 s/step user run) → the teacher forward is the dominant term of the step | Colab logs, `bench/results/bonsai_overnight_v2` |

## 2. What the precompute must store per position

| field | bytes/token | why |
|---|---|---|
| token ids (int32) | 4 | the `x_p` sequence itself |
| `H[i]` teacher output-normed hidden, INT4 group-64 | 0.53125 x 5120 = 2720 | step-1 `h` input and the cosine target of every unroll step |
| top-K teacher logits (ids int32 + vals fp16) + logsumexp (fp16) | 64 x 6 + 2 = 386 | exact renormalizable KL target; K ids are gathered from the frozen lm_head rows at training time |

The sparse KL is the **renormalized** teacher distribution: `p_tilde = softmax(topk_vals)` — the teacher's own
conditional distribution restricted to its top-K support — so the loss `D_KL(p_tilde || q_tilde)` is a true KL
divergence and >= 0 in exact arithmetic (fp16 storage rounds it to ~1e-4). Earlier runs printed a slightly
negative KL because they compared the un-renormalized `exp(v_k - lse)` against the renormalized student logits;
that offset is gone. The stored lse is no longer needed by the loss; it stays in the package for tail-mass audits
(`sum exp(v - lse) <= 1`, fp16 rounding of the stored vals makes the measured sum drift by up to ~1% on real
logit scales). `top1_agree` compares the student's winning row mapped back to its vocab id against the teacher's
top-1 id — the engine acceptance proxy.

## 3. Storage budgets (decimal GB, 8.2M token = 8K samples x 1024)

| package | int4-g64 hidden | + top-64 | + ids | total | int8 variant |
|---|---|---|---|---|---|
| 8K x 1024, 1 tap (mtp / 1-layer df3) | 22.3 | 3.1 | 0.03 | **25.4 GB** | 42.0 + 3.1 = 45.1 GB |
| 12K x 1024, 1 tap | 33.4 | 4.7 | 0.05 | **38.2 GB** | 62.9 + 4.7 = 67.6 GB |

The naive 20K x 2048 INT8 package (225.5 GB) is rejected: it fills a 240.3 GB Drive and starves the Drive FUSE read
path. r3 proves 8K samples train a good head; 8-12K x 1024 with 3-5 epochs and random block/anchor offsets per epoch
gives the model far more distinct contexts than its sample count suggests.

Frozen extras (once, per package): `lm_head_embed_frozen.safetensors` = embed_tokens + lm_head + output_norm in FP16 =
2.37 GiB + 2.37 GiB + 10 KiB ≈ **4.7 GiB**.

## 4. VRAM budget on the T4 (15 GiB usable, FP16, no BF16 tensor cores)

| item | mtp 1-layer (424.7M) | df3 2-layer (797M, future) |
|---|---|---|
| student fp16 weights + fp32 master + grads + 8-bit AdamW states (10 B/param) | 3.96 GiB | 7.42 GiB |
| frozen lm_head + embed (FP16, no grad) | 4.74 GiB | 4.74 GiB (or CPU-offloaded) |
| per-batch shard window (B=2, L=1024, H=5120 fp16 + top-64) | ~0.1 GiB | ~0.1 GiB |
| total | **~8.8 GiB** | **~12.3 GiB (borderline; `--frozen-device cpu` drops 4.7 GiB)** |

So: mtp 1-layer fits comfortably; a future 2-layer DFlash student needs `--frozen-device cpu` or a 24 GB card.
Auto rule in the trainer: compute capability < 8.0 → FP16 + `GradScaler` (T4 has no BF16 tensor cores); ≥ 8.0 → BF16.

## 5. Speed budget (the "significantly faster" claim, stated as a roofline)

Per 32,768-token optimizer step, 4-step unroll:

| path | FLOPs | T4 @ ~30 eff TFLOPS | A100 @ ~156 eff TFLOPS |
|---|---|---|---|
| full 248K lm_head (today's live path) | 1334 TFLOPs | 44.5 s | 8.6 s |
| sparse top-64 lm_head gather + student | 170 TFLOPs | **5.6 s** | 1.1 s |

Student decoder work (2.55 GFLOPs/tok fwd+bwd) is unchanged; the lm_head backward (7.63 GFLOPs/tok per unroll step)
disappears with the sparse KL. No 27B teacher forward at all: the per-step cost leaves the 1704 TFLOPs regime entirely.
Claim to verify by measurement: T4 `--from-targets` step time ≈ 2-6 s at 16K-32K tokens/step (vs 52.4 s for a live
teacher on T4 and 24.19 s on the A100 today), i.e. a 10-30x drop.

## 5b. Native PTQ1_0 dump mode (`--dump-targets`, no BF16, no layer streaming)

Phase A no longer needs the 50.1 GiB BF16 conversion or a >= 48 GB session. The engine itself — the same binary
used at home — loads the full PTQ1_0 GGUF (5.95 GB file, ~5.54 GiB in VRAM) with all 64 layers and the LM head
resident (measured 5742 MiB at seq 128 on the GTX 1070 Ti, budget 7373 MiB; T4 uses `--vram-budget-mib 14000`),
and runs the whole precompute per sample:

1. `reset()` (DeltaNet + conv state clean), one full prefill (`forward_ex`) -> the output-normed hidden of every
   position (the exact values the engine's own MTP head consumes — parity by construction, no Hadamard-space
   conversion anywhere).
2. The GPU LM head computes the full logits of 8 rows at a time; top-64 ids/vals (FP16) + FP32 logsumexp (FP16
   out) reduce on the CPU worker threads; the hidden quantizes INT4 group-64 / INT8 with the byte layout of the
   Python quantizer (RNE rounding, low nibble = even element, scale floor 1e-4).
3. Shards are plain `.npy` directories (ids/hq/hs/tk_ids/tk_vals/tk_lse.npy + meta.json) written atomically;
   `manifest.json` + `dims.json` for the trainer; per-shard auto-resume; `--max-gb` quota; tok/s + ETA.
4. `export_frozen_head.py` extracts ONLY token_embd + output_norm + output (unrotated: the Hadamard inverse of
   the stored weights) into `lm_head_embed_frozen.safetensors` (~3 min, 5.09 GB) — the other ~850 tensors are
   never read.

On sm_75+ (T4 / Ampere / Ada) the layer GEMVs and the head go through a dequant-on-the-fly FP16 Tensor Core path
(`src/cuda/ternary_f16.cu`): one matrix at a time dequantized into a ~335 MiB FP16 VRAM scratch, then
`cublasGemmEx` with `CUBLAS_COMPUTE_32F`; the LM head chunks 32768 rows at a time. Every kernel body is compiled
only for `__CUDA_ARCH__ >= 750`, so the sm_61 build keeps its exact SASS (gate: 0 half-arithmetic opcodes) and
runs the FP32/dp4a path unchanged. Measured smoke (4 samples x 128 tok, GTX 1070 Ti, sm_61 fallback path):
52 tok/s dump throughput; parity vs numpy `lm_head @ h`: mean cosine of the gathered rows vs stored vals
**1.000000**, max |lse ref - stored| 0.0177, `train_bonsai_mtp.py --from-targets` consumes the engine shards on
CPU with KL >= 0 descending and top1_agree > 0.94 within 6 steps.

## 6. Files

| file | role |
|---|---|
| `precompute_teacher_targets.py` | one-shot burst: pure teacher forward (`torch.inference_mode`), INT4-g64/INT8 hidden, sparse top-K logits + lse, shard files + manifest (atomic, auto-resume), `--max-gb` quota, tok/s + ETA; `--selftest` covers the quantizer, the shard format and a tiny end-to-end run; its loader accepts the engine's `.npy` shard directories too |
| `train_bonsai_mtp.py` (extended) | `--from-targets DIR --lm-head-file PATH`: no teacher at all; renormalized sparse top-K KL (>= 0) + cosine vs dequantized hidden; `--top-k-kl`, `--hidden-dtype`, `--epochs` with random block/anchor offsets, `--stage-dir`, `--optim adamw8bit|adafactor|adamw`, auto FP16 on sm<80 + GradScaler; live teacher mode untouched |
| `train_bonsai_t4_student_colab.ipynb` | T4-only: cell 2 builds `qwen38-engine` for sm_75 (plus the ggml CPU libs of the `feat/iso3-kv` fork), cell 3 = native Phase A dump + frozen head, cell 4 = Phase B training, cell 5 = sidecar export; generated by `make_colab_notebook.py` and checked by `check_colab_notebook.py` |
| `export_frozen_head.py` | the 3 frozen tensors straight from the PTQ1_0 GGUF (lazy rows, ~3 min on 4 CPU threads), with a `--selftest` for the safetensors writer and the name mapping |
| `tokenize_for_engine.py` | corpus shards -> balanced samples -> `tokens.bin` + `plan.json`, tokenized with the engine's own BPE (`--tokenize-file`), so the C++ dump needs no JSONL/BPE parsing; seed-deterministic selection |
| `src/engine/dump_targets.{hpp,cpp}` + `src/cuda/ternary_f16.cu` | the engine side: `--dump-targets` pass and the sm_75+ FP16 Tensor Core GEMM backend (strict `__CUDA_ARCH__ >= 750` guard) |