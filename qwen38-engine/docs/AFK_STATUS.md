# AFK status: qwen38-engine (Stages 2-5)

Updated: 2026-11-16 (Stages 3 to 5 done: hybrid defaults, MTP + rollback, sampler, chunked prefill, HTTP/SSE server, WebUI, head-to-head benchmark). Optimization round Poin A+B+C landed the same day (multi-token W4A8 CPU kernel, GQA invert, RoPE table; D measured-skipped — see the round section). Single session, no subagents, no `/goal`. Canonical tree locked to this session.
Guardrails followed: `ninja -j4`; `taskset -c 0-3` on every run; one engine at a time (llama-server only between
engine runs, never together); no full 64-layer FP32 load (weights stay quantized in the mmap, layers beyond a byte
budget are streamed); SKILL.md and AGENTS.md stay local, nothing pushed or committed.

## Stage status
| Stage | Status | Evidence |
|---|---|---|
| 1 Reference and baseline | done | docs/stage1-baseline.md, tests/ref_tensors/, tests/ref_full/ |
| 2 Loader, tokenizer, arena, CPU forward, text out | **done (CPU only, slow)** | below |
| 3 CUDA sm_61 + iso3 offload | **done and measured**: hybrid is now the default (32 GPU layers, iso3 KV, dp4a for all types), prefill streams the CPU layers through the GPU. Option A (FP32 dequant) and the dp4a upgrade of the 7 types are below | sections "Stage 3 option A", "Stage 3 step 3", "Stage 3 defaults and prefill streaming" |
| 4 MTP + DeltaNet rollback | **done and measured** (MTP default 2 as requested; it is slower than no MTP on 75% acceptance text, see the table) | section "Stage 4" |
| 5 Sampler, chunked prefill, cpp-httplib server, WebUI, head-to-head bench | **done and measured** | section "Stage 5", `bench/results/final_comparison.md` |

## Stage 2 result: `qwen38-engine` generates coherent text and matches the baseline

Build: `cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release && ninja -C build -j4` (g++ 11.4, `-O3 -march=znver1 -mavx2 -mfma`, no warnings).
Run: `taskset -c 0-3 ./build/qwen38-engine -m ~/models/27b/Qwen3.8-27B-UD-Q4_K_M.gguf -p "The capital of France is" -n 24 --temp 0 --seed 42 --parity-dir tests/ref_full --resident-gib 11.0`

| Check | Result | Log |
|---|---|---|
| Generated text (24 tokens, greedy) | ` Paris.\nThe capital of Germany is Berlin.\nThe capital of Italy is Rome.\nThe capital of Spain is` | bench/results/full_parity_run3.txt |
| Token parity vs baseline llama-server (`-ngl 23`, KV f16, no MTP, temp 0, top_k 1, seed 42) | **24 of 24 identical** | bench/results/stage2/token_parity.txt |
| Token parity, 62-token wikitext prompt, 8 greedy tokens | **8 of 8 identical** | bench/results/stage2/token_parity_p62.txt |
| Per-layer `l_out` cosine vs reference dump (T=5, all 64 layers) | min 0.999421 (layer 63), layers 0-1 = 1.000000, 0 layers below 0.995 | bench/results/stage2/layer_parity.txt |
| Final logits vs reference `result_output` | cosine 0.999620, top-1 match (token 11751), top-5 set identical, values within 0.10 | same |
| Tokenizer vs `llama-tokenize` | **43 of 43 samples identical** (25 ASCII, 5 Latin-1, 13 non-ASCII: CJK, Cyrillic, Greek, Arabic, Hebrew, emoji, symbols, combining marks) | bench/results/tokenizer_parity.txt |
| Heap allocations in the decode `forward()` calls | **0** (192 before the fix, see iteration log) | `heap_allocs_in_decode_forward` in the run JSON |
| Static arena | 597.7 MiB reserved once, frozen after load; any later `alloc()` aborts | engine stderr |

Memory (run3 and the 62-token run):
- Anonymous RSS 263 MB; file-backed (mmap'd page cache) RSS 12.4 GB; peak RSS 12.9 GB includes those mapped file pages.
- `MemAvailable` stayed at about 13 GiB during the whole decode. Swap used changed by -13 MB and +15 MB across runs (no growth trend).
- LM head (`output.weight`, about 1 GiB) is `mlock`ed (under the 1.94 GiB limit). No mlock warning was printed in any run.
- Layer residency (`--resident-gib 11.0`): 53 layers (10.81 GiB) stay in the page cache, 11 layers (2.64 GiB) are prefetched with `MADV_WILLNEED` two layers ahead and dropped with `MADV_DONTNEED` plus `POSIX_FADV_DONTNEED` after use.

Speed (honest): prefill 5 tokens 21 to 29 s; prefill 62 tokens 40 s (1.5 tok/s); **decode 0.060 tok/s (about 16.6 s per token)**.
Decode is disk-bound: 2.64 GiB streamed per token at the measured raw disk speed of 195 MB/s (`dd iflag=direct`) is about 13.5 s.
Baseline llama.cpp gets 2.2 tok/s because 23 layers sit in VRAM. This engine is a correctness reference, not a fast path;
the 24 GPU + 40 CPU split (Stage 3) is what removes the streaming.


## Stage 3 option A: 24 GPU layers + 40 CPU layers, iso3 KV, measured (2026-11-16)

Build: `cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release && ninja -C build -j4` (0 warnings, 0 errors).
Run: `taskset -c 0-3 ./build/qwen38-engine -m ~/models/27b/Qwen3.8-27B-UD-Q4_K_M.gguf -p "The capital of France is" -n 24 --temp 0 --seed 42 --gpu-layers 24 --kv iso3`

**Layer 0-23 audit** (`Qwen3.8-27B-UD-Q4_K_M.gguf`): all eight quantizations occur, so no kernel could be skipped.
| Type | Tensors | MiB | Share of the 4.75 GiB |
|---|---:|---:|---:|
| IQ4_XS | 60 | 2385 | 49% |
| Q4_K | 44 | 1406 | 29% |
| Q5_K | 27 | 553 | 11% |
| IQ4_NL | 5 | 219 | 5% |
| Q6_K | 8 | 135 | 3% |
| Q3_K | 2 | 73 | 1.5% |
| IQ3_S | 2 | 73 | 1.5% |
| Q8_0 | 38 | 20 | 0.4% |

**What runs on the GPU (all FP32, no half arithmetic):** RMSNorm, SwiGLU, residual adds, RoPE, Gated DeltaNet (conv, recurrence with
the state kept transposed for coalesced access, gated norm), GQA attention with the iso3 cache decoded in FP32 inside the kernel,
and the quantized GEMV. Q8_0 uses `__dp4a` with on-the-fly activation quantization. The other seven types dequantize 32 weights
at a time to FP32 inside the kernel (option A) and are being upgraded to `__dp4a` (next section).

| Check | Result | Log |
|---|---|---|
| fp16 conversion (integer bit ops) vs F16C | 0 mismatches over all 63490 patterns and 3989609 random floats | bench/results/foundation_test.txt |
| iso3 host reference vs `GGML_TYPE_ISO3_0` of libggml | byte-identical on 12000 blocks (needs `-ffp-contract=off`, FMA contraction alone caused 2 differing blocks and last-bit differences) | same |
| iso3 GPU store vs host reference | 888 blocks, 0 differ; position offset and FP32 store correct (kernel built with `--fmad=false`) | bench/results/iso3_gpu_test.txt |
| GEMV of 8 types vs libggml dequantization in double precision, T=11 | relative RMS error 1.2e-7 to 1.4e-7 for every type | bench/results/gemv_parity.txt |
| GPU layers 0-23 vs reference `l_out`, KV FP32 | cosine >= 0.99987 at every layer | bench/results/gpu_layers_parity_f32.txt |
| GPU layers 0-23 vs reference `l_out`, KV iso3 | cosine >= 0.9985 at every layer (0.99849 at layer 23) | bench/results/gpu_layers_parity_iso3.txt |
| Full 64-layer hybrid, per-layer cosine | worst 0.998341 (layer 51), 0 layers below 0.995 | bench/results/stage3/hybrid_run1.txt |
| Final logits | cosine 0.999253, top-1 match, top-5 set identical | same |
| Greedy tokens, 5-token prompt, 24 tokens, vs llama-server | **24 of 24 identical** | bench/results/stage3/token_parity_hybrid_iso3.txt |
| Greedy tokens, 62-token prompt, 8 tokens, vs llama-server | **8 of 8 identical** | bench/results/stage3/hybrid_B_p62.txt |
| Heap allocations in the decode `forward()` calls | **0** (GPU calls included) | `heap_allocs_in_decode_forward` |
| SASS gate (`scripts/sass_gate.sh`) | PASS: controls sensitive; 0 HFMA2/HMUL2/HADD2 in every source and in the linked executable; IDP.4A x8 in `q8_gemv` and in the executable | bench/results/stage3/sass_gate_optionA.txt |

Both parity baselines are the same llama-server run as Stage 2 (KV f16, no MTP). The engine uses iso3 for the 6 GPU attention
layers and FP32 for the 10 CPU attention layers.

**Measured numbers** (warm page cache, `taskset -c 0-3`, GPU free, no llama-server):
| Metric | 5-token prompt, 24 generated | 62-token prompt, 8 generated |
|---|---:|---:|
| pp (prompt eval) | 0.50 tok/s (10.1 s) | 2.92 tok/s (21.2 s) |
| **tg (decode)** | **1.97 tok/s** | **1.95 tok/s** |
| Peak VRAM (nvidia-smi, 200 ms) | 5235 MiB | 5235 MiB |
| Peak RSS | 10273 MiB (anonymous 239 MiB, the rest is mmap'd page cache) | 10296 MiB (anonymous 264 MiB) |
| Swap in/out over the run | 12/16 KiB | 20/172 KiB |

VRAM: 5235 MiB peak is under the 7.2 GiB cap (7373 MiB). GPU arena 5122 MiB: weights 4869 MiB, iso3 KV 9.4 MiB (6 layers, ctx 2048;
96 MiB in FP32), the rest scratch. The first run after a cold start loads 4.75 GiB from disk into VRAM (13 s) and the CPU layers into the
page cache; its prefill is slower (25 s for 5 tokens).
DDR4: all 40 CPU layers are resident (8.70 GiB, 0 GiB streamed), the LM head (1 GiB) is mlocked. **No disk streaming per token**:
decode went from 0.059 tok/s (Stage 2, CPU only) to 1.97 tok/s, about 33 times faster. The baseline llama-server reaches 2.2 tok/s with 23 GPU layers.

Limits seen in these numbers:
- Prefill runs the 40 CPU layers on the CPU: about 1.7 s per prompt token at T=5, 0.34 s per token at T=62. llama.cpp offloads batched matmuls
  to the GPU (13 to 28 tok/s in Stage 1). A 512-token prompt would take about 3 minutes here. Uploading CPU layer weights to VRAM for
  large batches is possible within the VRAM headroom (7373 - 5235 MiB) and is not built.
- Decode at 0.5 s per token is bound by reading the 8.7 GiB of CPU-layer weights from DDR4 once per token. More GPU layers would help:
  the cap leaves about 2 GiB, which fits roughly 8 more layers (not tested).

## Stage 3 step 3: Q8_1 + `__dp4a` for the other 7 types (measured)
Types upgraded (`src/cuda/dp4a_gemv.cu`): Q4_K, Q5_K, Q6_K, Q3_K, IQ4_NL, IQ3_S, IQ4_XS. Q8_0 already used dp4a. Enabled per type
with `--dp4a <list|all|none>` (mask `dp4a_mask`, default 0 = FP32 dequant path of option A).

- Per-type GEMV parity (`bench/results/stage3/dp4a_gemv_parity.txt`): 8e-8 to 1.3e-7 relative RMS against a double reference that uses
  the same quantized activation. Against the FP32-activation reference the error is 5.3e-3, which is the cost of quantizing the activation to Q8_1.
- Cumulative steps, none to all (`dp4a_steps.txt`): 24-layer parity worst cosine 0.998383 to 0.998535, every step PASS (bar 0.995).
- Full hybrid, `--dp4a all`, 5-token prompt (`hybrid_dp4a_parity.txt`): worst layer cosine 0.998438 (layer 23), logits cosine 0.999284,
  top-1 match, 24/24 greedy tokens identical to the llama-server baseline, 0 heap allocations in decode.
- 62-token prompt, `--dp4a all` (`dp4a_p62.txt`): 8/8 greedy tokens identical.
- SASS gate (`sass_gate_dp4a.txt`): PASS. `dp4a_gemv` IDP.4A x448, `q8_gemv` x8, linked executable x456; HFMA2/HMUL2/HADD2 x0 in every cubin;
  control probes show the gate detects both IDP.4A and half arithmetic.

| Run (`--gpu-layers 24 --kv iso3`, warm cache) | pp tok/s (prefill) | tg tok/s | GPU ms/token | CPU layers ms/token | LM head ms/token | Peak VRAM | RSS peak / anon |
|---|---:|---:|---:|---:|---:|---:|---|
| option A, 5 tokens (`optA_p5`) | 1.56 (3.2 s) | 1.963 | 78 | 395 | 37 | 5247 MiB | 10268 / 239 MiB |
| dp4a all, 5 tokens (`dp4a_p5`, cold-ish cache) | 0.85 (5.9 s) | 2.126 | 45 | 389 | 36 | 5247 MiB | 10269 MiB |
| dp4a all, 5 tokens (`dp4a_p5_r2`) | 2.165 (2.3 s) | 2.110 | n/a | n/a | n/a | 5247 MiB | 10267 MiB |
| dp4a all, 62 tokens, 8 generated (`dp4a_p62`) | 3.021 (20.5 s) | 2.081 | 49 | 395 | 36 | 5247 MiB | 10292 / 264 MiB |

- The dp4a upgrade cuts the GPU part from 78 to about 45 ms/token. Total decode moves from 1.96 to about 2.1 tok/s because the 40 CPU layers
  (about 390 ms/token, 8.7 GiB read per token) dominate and are bound by DDR4 bandwidth.
- Prefill for 5 tokens varies from 2.3 s to 10 s between runs depending on page cache and swap state (swap out up to 16 MB on `dp4a_p5`). Treat it as noisy.
- VRAM 5247 MiB includes the Q8_1 activation scratch (5235 MiB before it). Cap is 7373 MiB; roughly 2100 MiB headroom.
- Not tested: `--gpu-layers 32`. It should fit the headroom but no parity or speed number exists for it.

## Stage 3 defaults and prefill streaming (2026-11-16)

Default run is now `qwen38-engine -m GGUF -p ...` with no extra flags: 32 GPU layers, `--kv iso3`, `--dp4a all`, prefill streaming
for batches of 40 or more tokens, `--mtp-steps 2`. Flags: `--gpu-layers N`, `--kv iso3|f32`, `--dp4a all|none|LIST`, `--stream-min-t N` (0 = off), `--mtp-steps 0..3`.

**Hardware fact found here:** the GPU sits on a PCIe Gen3 **x1** link. `cudaMemcpy` H2D from pinned memory measures 0.84 GB/s (`/tmp/bw`, 3 runs).
Streaming all 7 GiB of CPU-layer weights through VRAM therefore costs about 9 s per prefill, whatever the prompt length.
(`nvidia-smi` shows gen 1 at idle and gen 3, width 1 under load.)

### GPU layer count sweep (VRAM from `cudaMemGetInfo`, cap 7373 MiB)
| GPU layers | Configuration | VRAM in use | tg tok/s (5-token prompt, 24 tokens) | Result |
|---|---|---:|---:|---|
| 24 | dp4a all, no stream, no MTP | 5247 MiB | 2.11 | earlier default |
| 32 | dp4a all, no stream, no MTP (`gl32_p5`) | 7045 MiB | 2.47 | 24/24 identical |
| 34 | same (`gl34_p5`) | 7467 MiB | 2.54 | **over the cap** |
| 33 | default buffers (stream + MTP rollback), measured at load | 7420 MiB | not measured | **over the cap** |
| **32** | **default: stream buffer + MTP rollback buffers** | **7203 MiB** (7123 MiB with `--mtp-steps 0`) | 2.52 with MTP 0, 2.21 with MTP 2 | **chosen: the largest count under the cap** |

With 32 layers the CPU part reads 6.96 GiB per token instead of 8.7 GiB. `--kv f32` also fits at 32 layers: 7319 MiB (`kvf32_p5`).

### How prefill uses the GPU for all 64 layers
- Layers 0-31 run on the GPU as before (resident weights, batched dp4a GEMV).
- For a first batch (position 0) of at least `stream_min_t` = 40 tokens, layers 32-63 also run on the GPU. Each matrix is read with `pread()` into a
  pinned double buffer (32 MiB chunks) and copied into one staging buffer sized for the largest matrix (about 75 MiB), then multiplied.
  The K, V and DeltaNet state of those layers come back to the CPU layouts (DeltaNet state is transposed on the way back), so decode continues on the CPU layers.
- Streamed layers use FP32 KV (like the CPU layers), the GPU-resident attention layers use iso3.
- Break-even against CPU prefill (0.27 s per token for 32 CPU layers) is near 40 tokens.
- The dp4a GEMV tile (tokens per thread block) is now chosen by batch size (1, 4, 8, 16). Results do not depend on the tile (each token keeps its own accumulator and summation order; `gemv_parity` PASS).
  Tile 1 cut the GPU decode time from 62 to 46 ms/token (24 tokens, same run settings). Tile 16 for large batches gave the 27.2 s prefill of 509 tokens in the table below.

### Prefill measurements (`bench/stage4_run.sh`, `--mtp-steps 0`, GPU free, page cache warm)
| Prompt | CPU layers on the CPU (`--stream-min-t 0`) | Streamed through the GPU (default) |
|---|---|---|
| 62 tokens | 16.5 s, 3.76 tok/s (`p62_cpuprefill`) | 11.6 s, 5.33 tok/s (`final_p62_mtp0`) |
| 509 tokens | 150.5 s, 3.38 tok/s (`p512_cpuprefill`) | 27.2 s, 18.73 tok/s (`final_p512_mtp0`) |

### Long-prompt parity (509 tokens, `bench/prompt512.txt`, references dumped from llama.cpp `-ngl 23 -c 2048`)
| Run | worst layer cosine | logits cosine | greedy tokens vs llama-server (f16 KV, 16 tokens) |
|---|---:|---:|---|
| engine default, iso3 KV (`p512_mtp0_iso3`) | 0.977709 (layer 58); 0.995585 at layer 31 | 0.992587, top-1 match | 16/16 identical |
| engine `--kv f32` (`p512_mtp0_f32`) | 0.999283 (layer 62) | 0.999665, top-1 match | 16/16 identical |
| llama.cpp f16 KV vs llama.cpp iso3 KV (reference run with `-ctk iso3 -ctv iso3 -fa on`) | 0.99456 at layer 62 | 0.992276 | |
| engine iso3 vs the llama.cpp iso3 reference | 0.976631 (layer 58) | 0.988270, top-1 match | |

Reading: the whole pipeline (GPU layers, streamed prefill, CPU layers, dp4a) is accurate at 509 tokens (f32 KV: logits 0.99967). The iso3 3-bit KV cache
costs accuracy at this length: logits cosine 0.9926 against f16 KV, the same size as llama.cpp's own iso3 (0.9923). It is below the 0.995 bar that the short-prompt tests used,
yet the 16 greedy tokens still match. Only 16 tokens were compared, so this is not a proof for long generations. `--kv f32` removes the loss and fits the cap (7319 MiB).
The engine-versus-llama-iso3 comparison is not like for like: llama.cpp quantizes the KV of all 16 attention layers, this engine only the 8 on the GPU.

## Stage 4: MTP draft head and DeltaNet rollback (2026-11-16)

### What was built
- **MTP block 64** (`blk.64.*`, 351 MiB, mostly Q6_K): `eh_proj` over `concat(rmsnorm(embedding, enorm), rmsnorm(h, hnorm))`, then an ordinary gated-attention layer with its own
  FP32 KV (cache index 16) and a dense FFN, then `shared_head_norm` and the shared LM head. It runs on the CPU (no VRAM left) and is mlocked (LM head 0.97 GiB + MTP 0.34 GiB < 1.94 GiB limit).
  Pairing follows `common/speculative.cpp`: at position p the block takes token x_p and the main-model hidden row (after `output_norm`) of position p-1; the first position pairs with zeros.
- **Cycle** (`src/engine/main.cpp`): one MTP batch (catch-up on the accepted drafts of the last cycle plus draft step 0), D-1 single MTP steps, then one verification batch
  `[cur, drafts]` through all 64 layers with logits and hidden rows for every row (`Engine::forward_ex`). The longest matching prefix is kept, the new `cur` is the target token of the first mismatch.
- **DeltaNet rollback, all 48 layers, GPU and CPU.** With snapshots on, each DeltaNet layer stores its conv and SSM state from before the batch plus the inputs of the recurrence
  (qkv, z, gate, beta). `dn_rollback(keep)` restores the state and replays the first `keep` tokens through the same conv and recurrence code. Cost: 83 MiB VRAM (24 GPU DeltaNet layers) and
  82 MiB RAM (24 CPU DeltaNet layers) for batches up to 4 tokens. Attention KV needs no rollback: the position counter moves back and later writes overwrite stale rows.
- `--rollback-test` (in `main.cpp`): fresh state, prefill, snapshot, batch of 4 tokens, rollback to keep = 1, 2, 3, then single steps.

### Results
- `bench/results/stage4/rollback_test.txt`: for keep = 1, 2, 3 the logits rows of the batch and every following single step are **bit-identical** (memcmp) to a run that never drafted. PASS.
- Greedy output with `--mtp-steps 2` equals `--mtp-steps 0` on every run: 24/24 (5-token prompt), 8/8 (62), 16/16 (509), 48/48 (5 and 62 tokens, steps 1, 2, 3).
  All of them also equal the llama-server baseline tokens for the 24, 8 and 16 token runs. The results are bit-identical because every operator gives each token the same numbers whatever the batch size (CPU `vec_dot` per token, GPU per-token accumulators).
- `draft_n_accepted > 0` in every MTP run (table below).
- SASS gate (`bench/results/stage4/sass_gate.txt`): PASS. dp4a_gemv IDP.4A x1624, q8_gemv x8, linked exe x1632, HFMA2/HMUL2/HADD2 x0 everywhere. No new CUDA kernels were added for Stage 4 or for streaming.
- `heap_allocs_in_decode_forward` = 0 in all runs.

### MTP sweep, 48 generated tokens, defaults otherwise (`sweep_*` files)
| Prompt | `--mtp-steps` | tg tok/s | draft_n | draft_n_accepted | accept hist (0..D accepted per cycle) | output |
|---|---:|---:|---:|---:|---|---|
| 5 tokens ("The capital of France is") | 0 | 2.536 | 0 | 0 | | reference |
| | 1 | 2.580 | 25 | 24 (96%) | [1, 24] | identical |
| | **2** | **2.201** | 40 | 30 (75%) | [2, 6, 12] | identical |
| | 3 | 1.971 | 51 | 33 (65%) | [0, 8, 2, 7] | identical |
| 62 tokens (Wikipedia text) | 0 | 2.461 | 0 | 0 | | reference |
| | 1 | 2.658 | 24 | 24 (100%) | [0, 24] | identical |
| | **2** | **2.714** | 32 | 32 (100%) | [0, 0, 16] | identical |
| | 3 | 2.768 | 36 | 36 (100%) | [0, 0, 0, 12] | identical |

### Default MTP 2 on the standard prompts (final runs, `final_*`)
| Prompt | MTP | pp tok/s (prefill s) | tg tok/s | Peak VRAM | Peak RSS (anon) | accepted / drafted |
|---|---|---:|---:|---:|---|---|
| 5 tokens, 24 gen | 0 | 2.94 (1.7) | 2.524 | 7123 MiB | 8554 (240) MiB | |
| | 2 | 2.89 (1.7) | 2.212 | 7203 MiB | 8977 (321) MiB | 15 / 20 |
| 62 tokens, 8 gen | 0 | 5.33 (11.6) | 2.202 | 7123 MiB | 8561 (242) MiB | |
| | 2 | 5.11 (12.1) | 2.274 | 7203 MiB | 9001 (344) MiB | 6 / 6 |
| 509 tokens, 16 gen | 0 | 18.73 (27.2) | 2.286 | 7123 MiB | 8603 (279) MiB | |
| | 2 | 15.83 (32.2) | 2.313 | 7203 MiB | 9195 (523) MiB | 12 / 12 |

With MTP the prefill also fills the MTP KV (4.9 s for 509 tokens).

### Why MTP does not win on this machine (measured, not a guess)
A verification batch of 3 tokens costs 0.82 s on the 32 CPU layers against 0.31 s for a single token: the CPU matmul is compute bound
(`vec_dot` is called once per token per weight row; ggml has no multi-token kernel for K-quants), so verifying more tokens is almost linear in T. GPU layers and LM head barely grow
(GPU 94 ms, head 76 ms per 3-token batch in the first MTP run). A cycle with 2 drafts costs about 1.1 s and a cycle with 1 draft about 0.74 s (from the sweep runs); each yields 1 + accepted tokens. Plain decoding costs about 0.39 s per token.
Break-even from those cycle times is about 88% acceptance per draft for 1 draft and about 92% for 2 drafts, which matches the sweep (75% loses 13%, 96% to 100% wins 2% to 10%).
The default is 2 as requested. `--mtp-steps 0` is faster for text the MTP head predicts less well; `--mtp-steps 1` loses less than 2 or 3 when acceptance is low.

## Stage 5: sampler, long prompts, HTTP/SSE server, WebUI, benchmark against llama.cpp (2026-11-16)

### Run it
```
scripts/serve.sh                       # http://127.0.0.1:8080, WebUI at /, loads in about 45 s (PORT=9000 scripts/serve.sh --mtp-steps 0)
tests/curl_check.sh                    # curl -N checks of every endpoint against a running server
python3 tests/server_test.py [--long]  # end-to-end test, starts its own server on port 8765
python3 tests/webui_test.py            # WebUI logic in QuickJS (pip install quickjs; test-time only)
bench/compare_llamacpp.sh              # sequential head-to-head, writes bench/results/final_comparison.md
```
CLI flags added: `--top-k --top-p --min-p --repeat-penalty --presence-penalty --repeat-last-n --seed --temp` (any temperature), `--server --host --port`.
The server binds to 127.0.0.1 unless `--host 0.0.0.0` is given (no authentication).

### 1. Sampler (`src/engine/sampler.*`)
Order as llama.cpp's default chain: repeat / presence penalty over the last `repeat_last_n` (64) tokens, top_k, top_p, min_p, temperature, seeded draw (`std::mt19937_64`).
Temperature 0 is greedy after the penalties. Without top_k the candidate set is capped at the 4096 largest logits. No heap allocation after construction.
The same sampler runs in the CLI and the server, and it drives the MTP verification: row j of a verification batch is sampled with the normal sampler, and a draft is accepted when it equals the sampled token.
Because the verification rows are bit-identical to single-step logits, **sampled output with a fixed seed is identical with `--mtp-steps 0` and `--mtp-steps 2`**:
CLI runs `s5_samp_mtp0` / `s5_samp_mtp2` (temperature 0.8, top_k 40, top_p 0.95, min_p 0.02, repeat 1.1, presence 0.2, seed 7, 40 tokens) are 40/40 identical;
the server test checks the same with penalties and with seeds. Temperature 0 with MTP 2 still equals the llama-server baseline tokens (24/24).

### 2. Multi-turn and long prompts (`src/engine/generator.*`)
- **Chunked prefill:** prompts of any length up to the context (`ctx_max` default now 4096) are processed in chunks of `t_max` = 512 tokens. Every chunk, also with position > 0, streams the CPU layers through the GPU
  (the K, V and DeltaNet state of those layers are uploaded before and copied back after each layer). The MTP KV is filled chunk by chunk.
- **Cache reuse:** the Generator keeps the tokens whose effect is in the model state. A new prompt that starts with all of them continues from there (only the new tokens are prefilled).
  The recurrent state cannot be rewound, so a prompt that diverges earlier is prefilled from scratch.
  The server builds the prompt with the model's own chat template (system message with reasoning instructions, `<|im_start|>` / `<|im_end|>`, `<think>` blocks of earlier assistant turns kept, as the template's `preserve_thinking` default does)
  and tokenizes client text without special-token parsing. Adjacent markup is merged before tokenizing so that `<think>\n` + empty reasoning + `\n</think>` gives the same tokens the model generated; without that, turn 2 never found its cached prefix.
- **Measured:** chunked prefill against the llama-server baseline, 509 tokens: `--tmax 256` (MTP 2, iso3) 16/16 tokens identical; `--tmax 128 --kv f32` 16/16 identical (`s5_p512_tmax256`, `s5_p512_tmax128_f32`).
  Needle in a haystack (a secret code placed before the filler, question at the end, temperature 0, iso3 KV, `tests/server_test.py --long-only`):

  | Prompt | Result | Time | pp | VRAM |
  |---|---|---:|---:|---:|
  | 2195 tokens (5 chunks) | `PINEAPPLE-7421` found | 158 s | 14.3 tok/s | 7244 MiB |
  | 3889 tokens (8 chunks) | `PINEAPPLE-7421` found | 284 s | 13.9 tok/s | 7244 MiB |

  A prompt that does not fit in the context is rejected with HTTP 400 `context_length_exceeded`.
- VRAM with the 4096 context: 7245 MiB peak (7203 MiB at 2048), under the 7373 MiB cap.
- Multi-turn in the server test: turn 2 reuses 43 of 66 prompt tokens, answers from the first turn ("Zorblax"), and gives the same answer with `cache_prompt: false`.

### 3. Server (`src/engine/server.cpp`, cpp-httplib 0.66 and nlohmann/json 3.12, both vendored in `third_party/`, MIT)
| Endpoint | Behaviour |
|---|---|
| `GET /` | the WebUI (`webui/index.html`, 18 KB), compiled into the binary by `cmake/embed.cmake` |
| `GET /health` | `{"status":"ok","busy":bool}` (never waits for the engine) |
| `GET /v1/models` | OpenAI model list (`qwen3.8-27b`, `n_ctx`, `n_vocab`) |
| `GET /v1/stats` | live VRAM MiB (cudaMemGetInfo), RSS, peak RSS, busy flag, cached tokens, configuration |
| `POST /v1/chat/completions` | OpenAI format, `stream` true or false. Body: `messages`, `temperature` (0.7), `top_k` (20), `top_p` (0.95), `min_p`, `repeat_penalty`, `presence_penalty`, `repeat_last_n`, `seed` (random), `max_tokens` (512), `mtp_steps` 0..3 (server default 2), `enable_thinking` (true), `reasoning_effort` (low, medium, xhigh), `cache_prompt` (true) |
| `POST /v1/cache/clear` | forgets the cached prefix |

- Streaming is SSE: a role chunk, `delta.content` pieces (cut at UTF-8 boundaries), progress chunks during prefill (`x_stats.phase = "prefill"`), a final chunk with `finish_reason`, `usage` and the telemetry, then `data: [DONE]`.
  With thinking on, the stream starts with `<think>\n` (the template puts it into the prompt), so the client sees both tags. Non-stream responses split `reasoning_content` from `content`.
- `x_stats` carries pp tok/s, tg tok/s, TTFT ms, MTP accept %, draft counts, reused tokens, VRAM MiB, RSS MiB. One generation runs at a time (others wait); a client that disconnects aborts its generation.
- Errors: 400 for invalid JSON, empty or invalid messages, parameters out of range, a last message that is not from the user, context exceeded.
- Binary: 2.81 MB (2.38 MB stripped) including cpp-httplib, nlohmann/json and the WebUI, far under the 15 MB limit.

### 4. WebUI (`webui/index.html`, one file, no CDN, no Node)
Dark chat page with streaming text; the `<think>...</think>` block is a collapsible `<details>` that folds when the reasoning ends (setting). The parser holds back any tail that could be the start of a tag, so tags cut between SSE chunks are safe.
Side panel: temperature, top_k, top_p, min_p, repeat_penalty, presence_penalty, seed, max_tokens, mtp_steps (0-3), reasoning effort, thinking, cache reuse.
Bottom bar: pp tok/s, tg tok/s, TTFT ms, MTP accept %, VRAM MiB, RSS MiB, token counts.
Assistant text is inserted with `textContent` (no HTML injection). History is sent back with the raw `<think>` text; the server splits it.
There is no Node or browser on this machine, so the page was **not opened in a browser**. Its logic is tested by running the page's own JavaScript in QuickJS.

### 5. Tests (all pass)
- `tests/webui_test.py`: 46 checks. ThinkParser: seven samples (tags, a lone `<`, unfinished blocks, a second block, multi-byte text) cut at every position into 2 and 3 chunks and one character at a time give the same result as one chunk;
  SSE parser cut at every position, CRLF, comments; request builder (ranges, seed omitted when empty, mtp clamp); the UI script compiles; every element id it uses exists; no external dependency.
- `tests/server_test.py`: 47 checks (`bench/results/stage5/server_test_full.txt`): endpoints, 5 error cases, non-stream and stream format, telemetry fields, multi-turn with reuse,
  seed / top_k 1 / min_p / top_p collapse to greedy, penalties, temperature 2.0, mtp_steps 0 versus 2 identical with `draft_accepted > 0`,
  a real reasoning stream parsed by the page's JavaScript (`</think>` closed within 320 tokens, answer `17 + 25 = **42**`), history with a raw `<think>` block accepted in turn 2.
  With `--long-only` (separate runs): the 2195 and 3889 token needle tests above.
- `tests/curl_check.sh` with `curl -N` against `scripts/serve.sh` (`bench/results/stage5/curl_check.txt`): SSE chunks arrive incrementally, telemetry in the chunks, the final chunk and `[DONE]`;
  non-stream with mtp_steps 0 and 2 gives the same text.
- Regression after the Stage 5 changes: SASS gate PASS (IDP.4A x1632, 0 half arithmetic), rollback test bit-identical, `s5_greedy_mtp2` 24/24 identical to the baseline, `--tmax` 128 / 256 chunk checks.

### 6. Head-to-head benchmark (`bench/compare_llamacpp.sh`, full table in `bench/results/final_comparison.md`)
509-token prompt, 256 generated tokens, temperature 0, seed 42, run one after the other with the GPU free. llama-server: iso3 KV, `draft-mtp` with 2 draft tokens, `-ngl` raised until the whole run would pass 7373 MiB:
-ngl 24 7015 MiB, **-ngl 25 7239 MiB (selected)**, -ngl 26 7473 MiB (over the cap).

| Program | pp tok/s | tg tok/s | MTP accept | Peak VRAM MiB | Peak DDR4 RSS MiB |
|---|---:|---:|---|---:|---:|
| llama-server `-ngl 25`, iso3 KV, draft-mtp 2 | 28.70 (repeat 28.81) | 2.50 (2.51) | 91 % (164 / 181) | 7239 | 14569 |
| qwen38-engine `--gpu-layers 32 --kv iso3 --dp4a all --mtp-steps 2` | 13.42 (13.50) | 2.58 (2.59) | 96 % (169 / 176) | 7245 | 9214 |
| (extra) llama-server `-ngl 25`, no MTP | 29.83 | 2.25 | | 6605 | 14605 |
| (extra) qwen38-engine `--mtp-steps 0` | 16.97 | 2.44 | | 7163 | 8643 |

Reading: decode is 3 % faster in the engine in both rounds, prompt processing is 0.47 times llama.cpp, memory (RSS) is 37 % lower.
The engine's pp is bound by streaming 7 GiB through the PCIe x1 link (about 9 s per prefill) plus 5 s of MTP prefill.
The generated tokens of the two programs agree for the first 12 of 256 tokens, then diverge (different iso3 attention numerics in the two programs); the engine's own output is identical for MTP 0 and 2.

## Experiment branch `exp/bit-lop3-cudagraph-prefill` (2026-11-16)

Baseline commit `f90f3d7` could not be pushed to `main` (protected branch, push declined); it is on GitHub as branch `baseline/stage1-5`.
Full record: `bench/results/experiment_comparison.md` (tables) and `bench/results/experiments/EXPERIMENT-LOG.md` (every attempt with reason, command, output, decision).

A/B on the benchmark prompt (509 + 256, MTP 2, warm cache, alternating binaries, `bench/ab_main_vs_exp.sh`):

| Binary | pp tok/s | tg tok/s | MTP accept | Peak VRAM MiB | Peak RSS MiB | tokens |
|---|---:|---:|---|---:|---:|---|
| baseline | 15.90 / 15.84 | 2.578 / 2.557 | 96 % | 7245 | 9215 | reference |
| experiment | 27.81 / 27.81 | 2.625 / 2.608 | 95 % | 7335 | 9102 | 256/256 identical |

llama-server `-ngl 25` iso3 + draft-mtp 2 in the same session: 27.25 pp, 2.519 tg (28.70 / 2.50 in the Stage 5 round). Prefill now matches llama.cpp; it does not clearly beat it.

Kept: overlapped streaming from pinned page-cache pages (prefill 27.2 s -> 18.1 s), MTP KV prefill on the GPU (5 s -> 0.2 s), 128-bit weight loads + PTX lop3, CUDA graphs (no measurable gain),
CPU matmul grain 128 (298 -> 287 ms/token). Opt-in: adaptive MTP (`--mtp-adaptive 1`), MTP prefill window (`--mtp-window`). Rejected after measurement: CPU-only prefill (3.4 tok/s),
MTP prefill on the last token only (acceptance 96 % -> 51 %), token tiles 32/64, `__ldg`, software prefetch.
Open: the first engine runs right after llama-server are cold (load 73 s, decode about 2.0 tok/s) with the 7 GiB pin; VRAM margin is 38 MiB.

## Decisions
- **D8: dp4a per type behind a mask.** Option A stays the default and the fallback; `--dp4a all` enables the upgraded kernels. Hybrid is still not the default
  (`gpu_layers = 0`, `dp4a_mask = 0` in the config). Needs a user decision whether `--gpu-layers 24 --dp4a all` should become the default.
- **D9: bit-exact iso3.** Host reference built with `-ffp-contract=off`, `iso3_quant.cu` with `--fmad=false` (FMA contraction gave 2 byte mismatches on 12000 blocks).
- **D10: FP16 conversion by integer bit operations** (`src/common/f16bits.hpp`), no `__half`. This fixed a subnormal exponent bug in the old Q8_0 decoder.
- **D11: GPU DeltaNet state stored transposed** for coalesced access; CPU layout unchanged.
- **D12: GPU KV is iso3, CPU attention layers keep FP32 KV.** The parity baseline is a llama-server run with f16 KV; a baseline with iso3 KV was not run.
- **D13: default is the hybrid** (32 GPU layers, iso3, dp4a all, stream prefill from 40 tokens, MTP 2). 32 is the largest layer count whose total stays under 7373 MiB with the stream buffer and the rollback buffers.
- **D14: streaming uploads one matrix at a time** into a 75 MiB staging buffer through a pinned `pread` double buffer. Rejected: whole-layer staging (190 MiB, would have forced 31 GPU layers) and mmap-to-`cudaMemcpy` (7.5 s of page-fault overhead on top of the PCIe time).
- **D15: DeltaNet rollback by snapshot plus replay of the first `keep` tokens.** Rejected: per-token state copies (150 MiB VRAM for T = 3), recomputing the accepted tokens through the full model (one extra forward on partial acceptance), and inverting the recurrence (not bit-exact).
- **D16: MTP runs on the CPU** because the GPU has no spare VRAM (the block is 351 MiB). It shares the LM head and token embedding with the main model (no `shared_head_head` or `embed_tokens` in the GGUF).
- **D17: MTP default 2 follows the user's instruction** although the sweep shows it is slower than 0 when acceptance is about 75%.
- **D18: the server is part of the engine binary** (`--server`), one request at a time, loopback by default. Rejected: a second binary and a request queue with batching (one GPU, one CPU pool, no spare VRAM).
- **D19: sampler order follows llama.cpp** (top_k, top_p, min_p, then temperature) and MTP verification samples each row with the same sampler (sample-and-match), which keeps the seeded output independent of `mtp_steps`.
- **D20: cache reuse by token prefix**, not by text. Recurrent DeltaNet state cannot be rewound, so partial prefix reuse is impossible; a mismatch means a full prefill. The MTP KV of the last few positions of a reused prefix can hold draft-time values (this only affects the acceptance rate, never the output).
- **D21: ctx 4096 default.** The CPU attention score buffer is on the stack (4096 floats), so 4096 is the maximum.
- **D1 (amended): quantized dot products use ggml-cpu `vec_dot` plus its activation quantization (Q8_K or Q8_0)** through
  `ggml_get_type_traits_cpu`, and `to_float` for the embedding row. Reason: 8 quant types appear in the layers
  (Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS, IQ4_NL, IQ3_S). Everything else is our own code: GGUF reader, tokenizer, RMSNorm,
  SwiGLU, RoPE, GQA attention with KV cache, Gated DeltaNet (conv, recurrence, gated norm), LM head loop, arena, thread pool,
  residency policy. This is a dependency on libggml/libggml-cpu and it is the reason numerics track llama.cpp CPU closely.
- **D2 (superseded):** the Stage 2 FP32 dequantizing path (`dn_layer.cpp`) is kept only as an independent oracle for `tests/layer01_parity`.
  The engine itself never dequantizes a weight matrix.
- **D3:** DeltaNet recurrence follows `ggml_compute_forward_gated_delta_net_one_chunk` (K=1, scalar gate, tiled q/k heads).
- **D4: layer residency instead of full residency.** Needed because one pass reads 15.8 GB and RAM is 15.9 GB.
- **D5: RoPE** is NEOX-style over the first 64 dims (pairs i and i+32), theta by repeated multiplication from `pos`, as
  `ggml_mrope_cache_init` does for text-only input where all position components are equal. Verified by logit parity.
- **D6: KV cache is FP32** in this engine (16 attention layers, `ctx_max` 2048 = 268 MiB). iso3 is Stage 3.
- **D7: Unicode classes** for the pre-tokenizer are exact for ASCII and Latin-1 and approximated above that
  (documented in `tokenizer.hpp`). All 13 non-ASCII samples matched, but this is not a proof for all of Unicode.

## Iteration log (limit 5 per sub-issue; none exceeded 2)
- Gated norm (Stage 2 oracle): 1 iteration, capture the unweighted RMS norm.
- Thread pool template: 1 iteration, lambda lvalue deduced `F` as a reference type.
- Reference dump: 2 iterations, `pkill -f` killed its own shell (now `pkill -x`), then `-o` is not a llama.cpp argument (now `DUMP_DIR`).
- Last-layer parity: 1 iteration, the reference `l_out-63` holds only the last token (1x5120).
- Tokenizer parity: 1 iteration, `llama-tokenize` processes escape sequences by default; the harness now passes `--no-escape`.
  The one early mismatch was in the harness, not in the tokenizer.
- Heap allocations: 1 iteration, `check_memory` used `ifstream` (8 allocations per forward); now a raw `read()` into a stack buffer.

- Stage 3 option A: `pkill -f llama-server` killed its own shell, use `pkill -x`. `gguf_index` needed libggml linked for `ggml_row_size`.
  Contracted FMA broke iso3 byte equality (fixed with D9). The first `bench` call of the dp4a run used a 900 ms timeout (unit is ms) and was rerun.

## Stage 3 kernel 1 (from the earlier part of this session)
Q8_0 x Q8_1 `__dp4a` GEMV on `ssm_alpha` and `ssm_beta` of layer 0: cosine 1.000000 against the reference; SASS gate on
`src/cuda/q8_gemv.cu`: IDP.4A x8, HFMA2/HMUL2 x0. Not tuned, no timing reported. See `bench/results/q8_gemv_parity.txt`.

## What is NOT done or NOT verified
- References come from a run with 23 layers on the GPU (CUDA numerics). Differences of about 1e-3 in cosine are expected and were observed.
- The engine was not compared with a pure-CPU llama.cpp run (a pure-CPU `-ngl 0` run was OOM-killed earlier, so it was not repeated).
- Stage 3 is done for the 24 GPU + 40 CPU split. Not built: batched prefill on the GPU (prefill is CPU-bound, 0.34 s/token at T=62), more GPU layers (`--gpu-layers 32` untested),
  an iso3-KV llama.cpp baseline, and prompts longer than 62 tokens on the hybrid path.
- MTP 2 versus MTP 0 output was compared for at most 48 tokens (sweeps) and 16 / 24 tokens (baseline runs); the 256-token benchmark run was not compared with an MTP 0 run token by token. The rollback test is bit-exact for 6 steps after a 4-token batch.
- The iso3 KV loses accuracy at 509 tokens (logits cosine 0.9926 against f16 KV). The needle tests (2195 and 3889 tokens) passed, but they test retrieval of one code, not general quality at 4k.
- **The WebUI was never opened in a browser** (none installed). Its JavaScript logic runs in QuickJS tests; layout, styling and DOM event wiring are checked only by reading the code and by the id and syntax checks.
- One request at a time. A second client waits; there is no queue limit, no authentication, no TLS, no CORS headers (use a reverse proxy if the server is exposed).
- The server does not honour client `stop` strings and has no tool calls, images or `logprobs`.
- Prompt processing of the engine is 0.47 times llama.cpp's on the benchmark prompt (PCIe x1 link plus MTP prefill).
- The benchmark is one 256-token run per program, two rounds for the headline pair, a single prompt; no confidence interval.
- Prefix reuse works when the new prompt starts with the exact token sequence of the previous prompt and reply. Re-tokenizing an assistant reply can differ from the generated tokens; the prompt is then prefilled again.

## Round 2026-11-16: optimization points A + B + C (D measured-skipped)

Three-point optimization round on this branch; every attempt, bug and decision is logged in
`docs/OPT_LOG_2026-11-16.md`, raw files `bench/results/stage4/{pre_*,post_*}`.

| Point | Change | Verification | Verdict |
|---|---|---|---|
| A | multi-token W4A8 dot products, `src/cpu/mt_dot.cpp` + dispatch from `qmatmul`: each weight block is decoded once and multiplied against all T activation rows (T 2..16), AVX2, ggml-shaped integer/float epilog | NOT bit-identical by design. Parity: cosine 0.998853 (bar >= 0.995), KLD(ref\|\|ours) 0.0041, greedy top-1 = reference 11751; generated ids IDENTICAL vs pre-round baselines at both MTP 0 and MTP 2 | kept |
| B | GQA loop inverted in CPU `attn_layer`: six Q heads share one K/V row read | bit-identical, three layers of proof: rollback PASS, parity digits reproduce 0.998678 / 0.083893 exactly, ids 24/24 identical | kept |
| C | RoPE cos/sin precomputed [ctx][32] at load (CPU) + `rope_table_kernel` lookup (GPU `ops.cu`) | bit-identical, same three-layer proof as B | kept |
| D | `sigmoid(-x)` identity for ssm_a | measured and skipped: `tests/ssm_a_scan` over all 48 layers = 2304 values, min -139.42, max -0.0038, zero bitwise -1.0f (`bench/results/stage4/post_ssm_a_scan.txt`) — identity invalid, exact form still needs powf | skipped with evidence |

Verification matrix 6/6 PASS (rollback; digit-exact parity with `QWEN_MT_DOT=0`; postB ids 24/24; Point-A parity
above bar; `post_p5` MTP 2 + mt_dot ids IDENTICAL(24); `post_p512` MTP 0 ids IDENTICAL(16)). SASS gate PASS after
the round (`post_sass_gate.txt`). Battery numbers in `bench/results/experiment_comparison.md`: MTP 0 tg
2.494 -> 2.814 (+13 %, B+C), MTP 2/3 tg up to 3.322-3.373 (+33 % over baseline; MTP now beats no-MTP at 100 %
acceptance — this closes the Stage 5 "Next session" item 3), pp unchanged, VRAM/RSS unchanged. mt_dot micro-bench
at T=4: Q4_K 1.07-1.11x, Q5_K 1.31-1.35x, Q6_K 1.34-1.35x, IQ4_XS 1.21-1.33x.

## Next session
All five stages are done. Open points, in order of value:
1. Open `scripts/serve.sh` in a real browser and check the layout.
2. Faster prefill: overlap the weight upload with compute in streamed prefill (needs a second staging buffer, about 75 MiB), or skip the MTP prefill for short replies.
3. DONE 2026-11-16 (Point A, `src/cpu/mt_dot.cpp`): MTP verify 87.7 -> 75.9 s on the headline 256-token run,
   MTP 2/3 tg up to +33 % over the pre-round baseline and now above no-MTP at 100 % acceptance; see the round
   section above.
4. Decide the MTP default (0 or 1 are faster on low-acceptance text) and whether long contexts should default to `--kv f32` (fits the cap at 7319 MiB).

## Experiment `exp/cascaded-pld-mtp` (2026-11-16)
**Branch**: `exp/cascaded-pld-mtp` from `main` @ 28e9bb2  
**Goal**: Cascaded Speculative Decoding — PLD (3/4-gram suffix match) + Adaptive MTP, T≤8 verify batch.

### Implementation
- `engine.hpp`: `kRollT = 8` (was 5)
- `gpu_layers.cu`: `gexec[16]`, `T <= 8` graph condition
- `generator.cpp`: buffers 8 rows, PLD 3/4-gram open-address hash tables, cascade verify batch = `[cur] + PLD_candidates + MTP_drafts`
- `main.cpp`: `--pld`, `--pld-max`, `--pld-min-len` flags

### Verification Gates
| Gate | Result | Notes |
|------|--------|-------|
| SASS (HFMA2/HMUL2) | ✅ PASS | 0 half arithmetic |
| Greedy parity (free-text, 509→256) | ✅ PASS | 256/256 tokens identical |
| Greedy parity (code/repetitive) | ❌ FAIL | Diverges at ~token 60 |
| Rollback T≤8 bit-identical | ❌ FAIL (both branches) | Pre-existing batch vs sequential numerics issue |

### Benchmarks (509→256 free-text, 524→256 code, MTP 2 adaptive, temp=0 seed=42)

| Metric | Main | Experimental | Δ |
|--------|------|--------------|---|
| **Free-text decode tok/s** | 2.67 | 0.63 | -76% |
| Free-text MTP accept | 96% | ~0% | -96 pp |
| **Code decode tok/s** | 2.75 | 0.68 | -75% |
| Code MTP accept | 100% | ~0% | -100 pp |
| **Free-text non-adaptive tok/s** | 2.52 | 0.91 | -64% |
| Non-adaptive accept | 96% | ~3% | -93 pp |

### Root Cause
MTP drafting logic (copied from original) produces drafts that never match main model's greedy predictions. Acceptance loop / rollback interaction with PLD-inserted verify batch appears to have a subtle bug. PLD candidates (792 generated) also never accepted.

### Status
**Experimental — not ready for merge**. Core infrastructure (T≤8, PLD matcher, cascade framework) works; MTP acceptance regression blocks performance. Needs debug of `hm`/`hm2` pointer state after PLD insertion in verify batch.

### Artifacts
- Full comparison: `bench/results/experiment_comparison.md`
- Branch: `exp/cascaded-pld-mtp` (ready to push)
