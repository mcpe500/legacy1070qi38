# Custom MTP head (blk.64) for Ternary Bonsai 2 27B

Ternary Bonsai 2 27B ships without the multi-token-prediction head that Qwen3.8-27B has, so qwen38-engine decodes it
one token per step. This folder trains such a head on one RTX PRO 6000 (96 GB) and puts it into the Bonsai GGUF. The
engine then drafts with it: every draft is checked by the main model, so the output tokens stay identical to plain
greedy decoding and only the speed changes.

| Script | What it does | Runs on |
|---|---|---|
| `convert_gguf_to_hf_bf16.py` | turns the Bonsai GGUF (PTQ1_0 or Q2_0, Hadamard basis) into the BF16 Hugging Face teacher (`Qwen3_5ForCausalLM`) | CPU, 6 GB RAM, about 35 min |
| `compare_engine_hf.py` | checks a converted checkpoint against the engine's per-layer tap dumps | CPU + engine |
| `prepare_corpus.py` | streams and mixes about 200,000 samples from Hugging Face into ChatML JSONL shards | any machine with network |
| `train_bonsai_mtp.py` | live online distillation: frozen BF16 teacher, one trainable decoder layer + nextn projection, 4-step unrolled self-rollout, full-vocabulary KL + cosine feature alignment | 1x RTX PRO 6000 96 GB |
| `inject_mtp_gguf.py` | quantizes the trained head to Q4_K_M or Q8_0 and writes a new GGUF with `blk.64` and `block_count = 65` | CPU |

## 0. Teacher checkpoint (GGUF -> Hugging Face BF16)

Bonsai is released as GGUF only, so the teacher for training is converted from it:

```bash
python3 convert_gguf_to_hf_bf16.py --gguf Ternary-Bonsai-2-27B-PTQ1_0.gguf --out /models/Ternary-Bonsai-2-27B-bf16
python3 convert_gguf_to_hf_bf16.py --selftest                     # rotation, permutation, bf16 rounding, code table
python3 convert_gguf_to_hf_bf16.py --check --gguf Ternary-Bonsai-2-27B-PTQ1_0.gguf   # real tensors vs the engine formula
```

The converter inverts what llama.cpp (`conversion/qwen.py`, `Qwen3_5TextModel`) and the PrismML Hadamard fold did:

- **Dequantization.** PTQ1_0 decodes base-3 digits through the `kPtq1Code` table of `src/common/ternary_spec.hpp`. Q2_0 decodes 2-bit codes.
- **Rotation.** Every weight in `prism.hadamard.weight_names` is unrotated: `W = W' H diag(s)` is the engine's `hadamard_unrotate` applied to each row. The token embedding in `prism.hadamard.inverse_weight_names` is unrotated the same way.
- **Head order.** The tiled order of the 48 linear-attention value heads is permuted back to grouped order: V rows of `in_proj_qkv`, `in_proj_z`, `in_proj_a/b`, `A_log`, `dt_bias`, and the V channels of `conv1d`. `out_proj` is folded and already grouped.
- **Value fixes.** `1` is subtracted from the zero-centered RMSNorm weights, `A_log = log(-ssm_a)`, and `conv1d` is reshaped to `[C, 1, 4]`.
- **Tokenizer.** It is built from the GGUF vocabulary and merges (`tokenizers` byte-level BPE). Pass `--tokenizer-from` to copy one instead.

Measured on the PTQ1_0 file:

| Check | Result |
|---|---|
| Full conversion | 851 tensors, 26.9B parameters, 50.1 GiB BF16 in 26 shards, 33.5 min, peak RSS 6.0 GB |
| Names and shapes vs `Qwen3_5ForCausalLM` (meta device) | 851 / 851, 0 missing, 0 extra, 0 mismatched |
| `--check` (PTQ1_0 and Q2_0) | plain `W x` = engine `W' H(s x)`, max rel error ≤ 5.5e-7; embedding rows identical to the engine |
| HF forward (`--layers 4 --vocab-limit 32768 --dtype f32`) vs engine tap dumps (`--kv f32`) | layers 0-3: cosine ≥ 0.999998, rel max error ≤ 5.4e-4 (with the iso3 KV the attention layer reaches 0.99985: the 3-bit KV, not the conversion) |
| Tokenizer vs the engine on the 4032-token needle prompt | 4032 / 4032 ids identical |

## 1. Corpus

```bash
pip install datasets huggingface_hub
huggingface-cli login            # bigcode/starcoderdata and uonlp/CulturaX are gated: accept their terms first
python3 prepare_corpus.py --out /data/bonsai_mtp_corpus --total 200000
python3 prepare_corpus.py --selftest   # formatters and quotas, no network
```

Default mix (edit with `--mix mix.json`, a list in the format of `DEFAULT_MIX`):

| Share | Source | Content |
|---|---|---|
| 45 % | `nvidia/OpenCodeInstruct` (15 %), `bigcode/starcoderdata` (30 %: C++, CUDA, Rust, Zig, Python, TypeScript, Go, PHP) | code |
| 15 % | `open-thoughts/OpenThoughts-114k` | reasoning with chain of thought |
| 20 % | `HuggingFaceH4/ultrachat_200k` (`train_sft`) | English dialogue |
| 10 % | `uonlp/CulturaX` `id` (6 %), `wikimedia/wikipedia` `20231101.id` (4 %) | Indonesian |
| 10 % | `uonlp/CulturaX` zh / ja / de / es (6 %), `CohereForAI/aya_collection` `aya_dataset` (4 %) | Chinese, Japanese, German, Spanish |

Every sample is ChatML with the thinking block closed when the source has none, so the token layout matches inference.
Plain documents become an assistant answer to a short "write a text / source file" request. Exact duplicates are
dropped, texts are cut at 24,000 characters, the run is deterministic for a seed and resumable (`manifest.json`).

## 2. Training

```bash
pip install torch transformers safetensors accelerate
python3 train_bonsai_mtp.py --selftest     # tiny random teacher on the CPU: shapes, 4-step loss, backward, GGUF names
# pipeline smoke test on a CPU: a 4-layer converted teacher, random tokens, 2 SGD steps, checkpoint written
python3 convert_gguf_to_hf_bf16.py --gguf Ternary-Bonsai-2-27B-PTQ1_0.gguf --out /tmp/mini --layers 4 --vocab-limit 32768 --dtype f32
python3 train_bonsai_mtp.py --teacher /tmp/mini --smoke --device cpu --optim sgd --out /tmp/smoke --steps 2 \
    --seq-len 32 --batch 1 --grad-accum 1 --onpolicy-frac 0 --log-every 1 --save-every 2
python3 train_bonsai_mtp.py \
    --teacher /models/Ternary-Bonsai-2-27B-bf16 \      # output of convert_gguf_to_hf_bf16.py (step 0)
    --corpus /data/bonsai_mtp_corpus --out runs/mtp1 \
    --init gguf:/models/Qwen3.8-27B-UD-Q4_K_M.gguf \   # warm start from Qwen's own blk.64 (same shapes), or `random`
    --steps 12000 --seq-len 2048 --batch 2 --grad-accum 8 --lr 2e-4
```

What one step does (all live on the GPU, nothing cached on disk):

1. A batch of packed corpus sequences, or with probability `--onpolicy-frac` (0.25) a teacher continuation of a corpus
   prefix (sampled at temperature 0.8, top-p 0.95), so the head also learns on the teacher's own token distribution.
2. Teacher forward (frozen, BF16, no KV cache): the output-normed hidden state `H[i]` of every position, taken from a hook
   on the final norm. This is the same row the engine feeds to the MTP head.
3. Four unrolled steps. Step 1 feeds `(H[i], x[i+1])`. Step k > 1 feeds the student's own feature from step k-1 with
   the token `x[i+k]`, as the engine's drafting loop does (EAGLE-3 training-time test / HASS). Each step is scored on
   the teacher's next position `i+k`:
   `KL(softmax(Z[i+k]/T) || softmax(student/T))` over all 248,320 tokens, plus `0.5 * (1 - cos(f, H[i+k]))`,
   weighted by `0.8^(k-1)`. The teacher logits `Z` come from the frozen teacher LM head applied to `H`, in chunks of
   `--kl-chunk` positions, so the full-vocabulary rows never sit in memory at once.
4. AdamW (betas 0.9 / 0.95, weight decay 0.01), cosine schedule with warmup, gradient clipping at 1.0, BF16 autocast,
   FP32 master weights, gradient checkpointing per unroll step.

The log (`runs/mtp1/train_log.jsonl`) has the KL, the cosine loss and the top-1 agreement with the teacher for each
unroll step. The step-1 agreement approximates the draft acceptance rate the engine reports (`draft_n_accepted / draft_n`).

Memory on 96 GB: teacher 50.1 GiB (the converted BF16 checkpoint), student 424.7M parameters (1.7 GB FP32 + 3.4 GB AdamW + 1.7 GB gradients), plus the
activations of 2 x 2048 tokens.

The student mirrors the engine's blk.64 exactly:
- `e = RMSNorm(embed(x_p), enorm)`, `h = RMSNorm(h_{p-1}, hnorm)`, `z = eh_proj([e | h])`.
- One gated-attention layer: 24 query heads with `[q | gate]` per head, 4 KV heads, `head_dim` 256, per-head RMSNorm on q
  and k, NeoX RoPE on 64 of the 256 dims with theta 1e7, output `sigmoid(gate) * attn`.
- SwiGLU FFN (17,408), then `shared_head_norm`. The teacher LM head is shared.
- RMSNorm multiplies by the weight as stored in the GGUF, with no `1 + w`.

Checkpoints: `runs/mtp1/mtp_step{N}.safetensors` (FP32, GGUF tensor names) and `train_state.pt` (resume with the same command).

## 3. Into the GGUF

```bash
python3 inject_mtp_gguf.py --base Ternary-Bonsai-2-27B-PTQ1_0.gguf \
    --from-checkpoint runs/mtp1/mtp_step12000.safetensors --quant q4_k_m \
    --out Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf        # --in-place replaces the base after a verified write
python3 inject_mtp_gguf.py --selftest                  # Q8_0 / Q4_K round trips against llama.cpp gguf-py
```

- `q4_k_m` uses Q4_K for every matrix except `attn_v` and `ffn_down`, which get Q8_0. Q6_K, which llama.cpp would use for those two, is not implemented here. The result is about 273 MiB.
- `q8_0` uses Q8_0 everywhere, about 450 MiB.
- The tool copies the original header and tensor data byte for byte, sets `qwen35.block_count = 65` and `qwen35.nextn_predict_layers = 1`, appends the 15 `blk.64.*` tensors, and verifies the written bytes before the final rename.

`--from-gguf QWEN.gguf` copies Qwen3.8-27B's own blk.64 raw instead. The engine tests use it (see below).

## 4. In the engine

```bash
taskset -c 0-3 ./build/qwen38-engine -m Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf --gpu-layers auto --ctx 4096 \
    --mtp-adaptive 0 --mtp-steps 2 -p "..." -n 256
```

The loader puts the head on the GPU when every layer and the LM head are there and the block plus its KV fit the
7373 MiB budget (`MTP block: GPU` in the log, `--gpu-mtp 0|1` forces it). Otherwise the head runs on the CPU.

Measured with Qwen3.8-27B's blk.64 copied in (`--from-gguf`, a head trained on Qwen and not on Bonsai), PTQ1_0, ctx 4096,
prompt 509 tokens, 256 generated tokens, both runs with bit-identical output tokens
(`bench/results/bonsai128k/PTQ1_0mtpqwen_p512_*.txt`):

| Mode | tok/s | drafts accepted |
|---|---|---|
| no MTP | 11.4932 | |
| MTP 2 fixed | 17.7078 | 158 / 198 |

These runs are tests of the plumbing: a head distilled on Bonsai itself should accept more drafts.
