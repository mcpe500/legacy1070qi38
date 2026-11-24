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

lse (`logsumexp` of the full row) is stored so the teacher's renormalized top-K distribution `p_k = exp(v_k - lse)`
is exact inside the K set (the dropped tail mass is `1 - sum_k p_k`, small for K=64 on this model).

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
teacher on T4 and 24.19 s on the A100 today), i.e. a 10-30x drop; the one-time precompute burst is 8.2M-12.3M tokens
at the teacher's own forward throughput (5-8K tok/s on the A100) = 20-45 min.

## 6. Files

| file | role |
|---|---|
| `precompute_teacher_targets.py` | one-shot burst: pure teacher forward (`torch.inference_mode`), INT4-g64/INT8 hidden, sparse top-K logits + lse, shard files + manifest (atomic, auto-resume), `--max-gb` quota, tok/s + ETA; `--selftest` covers the quantizer, the shard format and a tiny end-to-end run |
| `train_bonsai_mtp.py` (extended) | `--from-targets DIR --lm-head-file PATH`: no teacher at all; sparse top-K KL + cosine vs dequantized hidden; `--top-k-kl`, `--hidden-dtype`, `--epochs` with random block/anchor offsets, `--stage-dir`, `--optim adamw8bit|adafactor|adamw`, auto FP16 on sm<80 + GradScaler; live teacher mode untouched |
| `train_bonsai_t4_student_colab.ipynb` | Phase A (burst on a big GPU, once) + Phase B (T4 / local GPU, unlimited iterations), generated by `make_colab_notebook.py` and checked by `check_colab_notebook.py` |