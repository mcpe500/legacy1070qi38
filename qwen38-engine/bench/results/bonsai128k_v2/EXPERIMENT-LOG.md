# Experiment log: `exp/bonsai-128k-prefill-opt-v2` (issue #52), 2026-11-19

The branch starts from main `fa4cc1f` (PR #51 merged). Hardware: GTX 1070 Ti (sm_61, 8 GB), Ryzen 5 2400G, 16 GB DDR4.
Budgets: 7373 MiB VRAM for the process and 14.2 GB host RAM.

Rules followed on every step:
- `ninja -j4`; every engine and test run pinned with `taskset -c 0-3`.
- Exactly one process on the GPU. The other agent session (opencode, PIDs 1463 / 1781) was paused with SIGSTOP through `bench/gpu_isolate.sh pause`, and every run started only after `gpu_isolate.sh check` reported a free GPU.
- Peak VRAM is sampled by nvidia-smi every 50 ms (`bench/bonsai_ctx_run.sh`).
- Numbers are copied from the result files in this folder.

## 0. Starting point (open problems from #51)

| Item | Value in #51 |
|---|---|
| 128K prefill (130965 tokens) | > 4 h, interrupted, never finished |
| Prefill attention kernel | `attention_kernel_tiled`, one block per (token, head); 6.59e14 B of K/V traffic for 128K |
| Q2_0 64K peak VRAM | 7417 MiB for about 9 s (> 7373) |
| Needle 32K, state round trips 64K / 128K | not run |
| PTQ1_0 decode | 11.49 tok/s (509-token prompt), 8.15 (4K needle); GEMV 77-91 GB/s |
| BF16 teacher for `train_bonsai_mtp.py` | missing |

## 1. M1: prefill attention with K/V reuse

### 1.1 Kernel design and the reuse factor actually reached

`attention_kernel_prefill<ISO3, HPW>` (src/cuda/ops.cu) uses one block per (8 query tokens, KV group), with 8 warps. Each warp owns one token and HPW of the group's 6 heads; lane L holds dims 8L .. 8L+7 of q and of the accumulator. Per tile of 16 positions:
- The raw iso3 bytes (100 B per position and tensor) are staged with coalesced 32-bit loads, dequantized once into an FP32 shared-memory tile (2 x 16 KiB), and read by every row.
- Scores use a reduce-scatter over the warp: 9 shuffles for 8 dot products, where the plain version needs 40.
- The softmax runs online per row in FP32. Optional position slices write (m, l, o) partials, and `attention_combine_n` merges them. This keeps the GPU busy for the 64-token chunks used at 128K.

The reuse factor is **48x** (6 heads x 8 tokens). The 384x target (64 tokens x 6 heads) would need 384 rows of q and o in FP32 per K/V tile. That is 768 KiB, against 256 KiB of registers and 48 KiB of shared memory per block on sm_61. At 48 rows the kernel already uses 178 registers per thread.

### 1.2 Attempts (T = 64 query rows at n = 131072, one layer, `attn_long_test 131072 time`)

| Variant | iso3 KV | f32 KV | Note |
|---|---|---|---|
| old tiled kernel (reference) | 2178.5 ms | 1715.2 ms | |
| reuse kernel, 6 heads/warp, iso3 dequant from global | 111.3 ms (with slices) | 74.9 ms | first version |
| 3 heads/warp (16 warps, 100 registers) | 105.9 ms | 96.2 ms | more occupancy, more shuffles per head: slower |
| 2 heads/warp (24 warps, 75 registers) | 137.8 ms | 126.5 ms | slower |
| **6 heads/warp + iso3 bytes staged, dequant from shared memory** | **96.9 ms** | **74.1 ms** | kept (22.5x / 23.1x) |
| same with `__launch_bounds__(256, 2)` (128 registers) | 111.4 ms | 80.1 ms | 356 B of spills: reverted |

**Parity.** `attn_long_test` 52/52 PASS. It covers:
- fast vs tiled for n ≤ 8192;
- split-K and the reuse kernel vs a double-precision reference up to n = 131072, f32 and iso3;
- prefill cases T = 64 / 37 at n = 100 / 4096 / 8193 / 32768 / 65536 / 131072, with and without slices.

The first prefill run failed 7 cases at the original 1e-4 bar. The old tiled kernel failed the same cases, with rel up to 1.73e-4 on 64 rows at 8193 positions: these are long sequential FP32 sums. The bar was set to rel < 3e-4 and "at most 1.5x the old kernel"; the worst new case is 1.82e-4. The ≤ 8192 shared-memory kernel is untouched on the Qwen path.

### 1.3 Prefill GEMV (PTQ1_0, T > 5)

| Variant (ffn_up 5120x17408) | T = 16 | T = 64 | Note |
|---|---|---|---|
| v2 tile (#51) | 1741 us | 6941 us | two shared loads per dp4a: shared-memory bound |
| I2F replaced by an exact magic-constant conversion | 1741 us | – | no change: not the bottleneck |
| tile2 <16,4> (R rows per warp reuse activation registers) | 1158 us | 4579 us | 189 registers |
| tile2 <16,1> | 1275 us | 5043 us | |
| tile2 <16,2>, single-buffered | 1182 us | 4679 us | |
| **tile2 <16,2>, double-buffered staging, one barrier per step** | **762 us** | **3089 us** | kept (2.28x / 2.25x) |
| tile2 <32,1> / <32,2> for T > 16 | – | 6009 / 4218 us | worse |

Bit-identity:
- `ternary_ab_test` 180/180 cases, T = 1..64.
- 64-layer tap dump of a real PTQ1_0 prefill (500 tokens) + decode: 0 of 64 dumps differ from main.
- ncu is not usable (ERR_NVGPUCTRPERM without root), so the analysis used SASS instruction counts and timing.

### 1.4 128K prefill

`--state-test`, PTQ1_0, 130965-token prefix, 64 GPU layers, GPU LM head, prefill chunk 64 (auto):

| Run | Prefill | tok/s | Round trip | Peak VRAM |
|---|---|---|---|---|
| #51 | > 4 h, not finished | – | – | 7383 MiB (another process on the GPU) |
| M1 (`PTQ1_0_ctx131072_statetest.txt`) | 3990 s = 66.5 min | 32.82 | PASS | 7349 MiB |
| final, after M2 + M3 (`final_PTQ1_0_ctx131072_statetest.txt`) | **3774.7 s = 62.9 min** | 34.70 | PASS | 7349 MiB |

Model of the cost (from the 32K run and the microbenchmark) is about 17.5 ms per token for GEMV, DeltaNet and the small kernels, plus attention at 1.51 ms per token and layer at depth 128K. Totals for 128K: 1580 s attention + 2290 s rest = 3870 s, which matches the 3990 s measured.

## 2. M2: VRAM cap and long verification

### 2.1 Q2_0 64K, 7417 MiB in #51

| Run | GPU layers | Peak |
|---|---|---|
| #51 queue (other agent session active) | 59 | 7417 MiB (93 samples x 100 ms) |
| r1 / r2 / r3, isolated, 50 ms sampling | 61 (new auto-fit) | 7347 / 7347 / 7347 |
| old configuration `--gpu-layers 59`, isolated | 59 | 7315 |
| M3 depth run | 61 | 7347 |

The excess of +102 MiB matches the CUDA context of a second process: our own process shows 107 MiB right after `cudaFree(0)`. Nothing in the engine allocates device memory after load. The planner's projection for these runs is ≤ 7362.4 MiB.

Root cause: a second GPU client that the #51 queue did not check for. The fix is the isolation helper plus a per-run check.

### 2.2 Auto-fit

`--gpu-layers auto` now tries prefill chunks of 256 / 128 / 64 before it gives a layer up:

| Q2_0 context | GPU layers before | after |
|---|---|---|
| 32K | 62 | 64 (chunk 128) |
| 64K | 59 | 61 (chunk 64) |
| 128K | 53 | 55 (chunk 128) |

### 2.3 Verification runs

| Run | Result |
|---|---|
| PTQ1_0 needle 4K (4032 tokens) | `MAGENTA-FALCON-42`, prefill 42.1 tok/s (this run overlapped with the BF16 conversion on cores 5-7, so its decode is not used) |
| Q2_0 needle 4K | `MAGENTA-FALCON-42`, prefill 24.8 tok/s |
| PTQ1_0 needle 32K (32704) | `MAGENTA-FALCON-42`, prefill 671.7 s (48.7 tok/s), peak 6351 MiB |
| Q2_0 needle 32K | `MAGENTA-FALCON-42`, prefill 1207.8 s (27.1 tok/s), 64 GPU layers, peak 7337 MiB |
| 64K `--state-test` (65429 tokens) | PASS (tokens + first logits bit-identical), prefill 1439.4 s (45.5 tok/s), state 948.3 MiB, save 2.38 s, load 1.46 s |
| 64K fresh `--state-load` | load 5.98 s, 43 new tokens in 1.41 s, `MAGENTA-FALCON-42` |
| 128K fresh `--state-load` | load 2.88 s (1748.3 MiB), 43 new tokens in 1.93 s, `MAGENTA-FALCON-42`, peak 7349 MiB |
| Qwen Q4_K_M greedy | 48/48 ids = `stage4/p5_mtp0.txt` (run twice) |
| Qwen rollback (`QWEN_MT_DOT=0`) | PASS bit-identical, keep = 1..4 |
| Qwen `tests/server_test.py` (`QWEN_MT_DOT=0`) | 69/69 PASS |

### 2.4 DeltaNet recurrence and BF16 pair

- `dn_rec`: the 128-value state column of each thread stays in registers for the whole call. It is read and written once; the reference does it twice per token.
  - Bit-identical (`dn_rec_ab_test`, T = 1..64).
  - T = 1: 62.6 → 42.3 us; T = 64: 2562 → 501 us.
  - The compiler spills 64 B here, which is cheaper than the 4 x 64 KiB of global traffic it replaces.
- `gemv_fp32_pair`: the two 48-row BF16 GEMVs (ssm_alpha, ssm_beta) run in one launch (grid.z = 2), with the same per-row arithmetic. Decode cost went from 2 x 19.7 us to 31.8 us.

## 3. M3: PTQ1_0 decode

### 3.1 Profile (nsys, `--cuda-graph 0`, 64 decode steps, ctx 4096)

| Kernel | before M3 (`m3_decode_profile_before.txt`) | after (`m3_decode_profile_after.txt`) |
|---|---|---|
| `gemv_ptq1_0_kernel_v2<1,4>` | 171.5 us avg, 86.3 % | 162.0 us avg, 86.0 % |
| `dn_rec` | 44.4 us | 44.7 us |
| BF16 alpha+beta pair | 31.3 us | 31.8 us |
| `rmsnorm_rows` | 10.3 us | 10.4 us |
| `rotate_q8` | 4.5 us | 4.6 us |
| iso3 KV store | 25.4 us | **4.9 us** (one warp per block, byte-identical) |
| attention (split-K + combine) | 20.8 + 8.7 us | 21.0 + 8.9 us |

### 3.2 Changes and their effect

| Change | Measurement |
|---|---|
| Bonsai uses split-K decode and K/V-reuse prefill attention at every ctx (`long_kernels`). The shared-memory kernel does a serial value pass, about 1.5 ms per layer at 4K. | depth 4K 8.58 → 11.93 tok/s; prefill 509 tokens 65.9 tok/s |
| PTQ1_0 decode GEMV: 5 persistent blocks per SM (was 8). At 48 registers x 256 threads, 5 blocks fit an SM, and 8 left a partial second wave. | ffn_up 229 → 208 us, LM head 3244 → 2778 us; real decode 12.53 → 13.17 tok/s |
| Rows per warp of the decode GEMV: R = 3 / 4 / 6 swept with 4 / 5 / 6 / 8 blocks per SM | R = 4 with 5 blocks is the best stable choice (bench noise about 10 %) |
| iso3 KV store per warp | 25.4 → 4.9 us; 13.17 → 13.31 tok/s |

### 3.3 Depth probe after M3 (`m3_*_depth.txt`, tok/s)

| Depth | PTQ1_0 #51 | PTQ1_0 now | Q2_0 #51 | Q2_0 now (GPU layers) |
|---|---|---|---|---|
| 512 | – | 12.74 | – | – |
| 4K | 8.53 | 12.68 | 8.32 | 12.00 (64) |
| 16K | – | 11.02 | – | – |
| 32K | 8.79 | 9.29 | 1.65 | 2.42 (64, LM head on the CPU) |
| 64K | 6.76 | 7.06 | 0.96 | 1.34 (61) |
| 128K | 4.59 | 4.72 | 0.47 | 0.65 (55) |

Real generation:

| Prompt | Mode | tok/s | Drafts accepted | Output |
|---|---|---|---|---|
| 509 + 256 tokens | MTP 0 | 13.30 | – | |
| 509 + 256 tokens | MTP 2, Qwen blk.64 injected | 18.48 | 163 / 186 | ids identical to MTP 0 |
| 4032-token needle | MTP 0 / MTP 2 | 12.11 / 11.26 | 6 / 6 | 9 tokens only (stop), MTP gives nothing here |

The Bonsai + MTP rollback test passes, bit-identical.

**15 tok/s without MTP is not reached; the measured value is 13.30 tok/s.**
- The decode GEMV is 86 % of the GPU time, at 93-100 GB/s effective on the large shapes.
- Per 4 weights and row it executes the base-3 digit lookups (4 shared loads, 3 byte permutes, address math), 2 shuffles, dp4a, and the FP32 scale and accumulate. That is ALU-bound, not memory-bound.
- 15 tok/s needs about 115 GB/s.
- A lossless repack to 2 bits does not help: the Q2_0 kernel reaches 107 GB/s on 1.24x more bytes.
- Keeping bit-identity with the reference order rules out regrouping the FP32 accumulation.
- With an MTP head the target is passed: 18.48 tok/s, with a head trained on Qwen, not on Bonsai.

## 4. M4: BF16 teacher converter

| Step | Result |
|---|---|
| `convert_gguf_to_hf_bf16.py --selftest` | PASS (rotation inverse, V-head permutation inverse, bf16 rounding = torch, 243 valid PTQ1_0 bytes) |
| `--check` on PTQ1_0 and Q2_0 | plain `W x` vs engine `W' H(s x)` max rel ≤ 5.5e-7 on output / attn_gate / attn_qkv / attn_q / ffn_down; embedding rows identical. Both packings give the same values: they encode the same ternary weights. |
| First full conversion | killed by the OOM killer: the 248320 x 5120 embedding was concatenated in FP32 (5 GB) next to a 4 GB shard buffer. Fixed by streaming the embedding / LM head to bf16 in 8192-row chunks, with 2 GiB shards. |
| Full conversion (PTQ1_0) | 851 tensors, 26.9B params, 50.10 GiB BF16, 26 shards, 2013 s, peak RSS 6.0 GB |
| Skeleton check (meta device) | 851/851 names and shapes match `Qwen3_5ForCausalLM`, all BF16 |
| HF forward, 4 converted layers (FP32), vs engine taps | engine `--kv f32`: cosine ≥ 0.9999983, rel max ≤ 5.4e-4 on all 4 layers. With the iso3 KV, layer 3 (attention) reaches 0.99985: the 3-bit KV, not the conversion. |
| Tokenizer | 4032 / 4032 ids identical to the engine on the needle prompt |
| `train_bonsai_mtp.py --smoke` on CPU | converted 4-layer teacher (vocab 32768), full-size student (424.7M), 2 SGD steps: finite KL / cosine losses, checkpoint written (`convert/smoke_train_log.jsonl`) |

## 5. Final 128K run (`final_PTQ1_0_ctx131072_statetest.txt`)

| | Value |
|---|---|
| Prefill, 130965 tokens | 3774.7 s = **62.9 min** (34.70 tok/s), against > 4 h in #51 and 66.5 min after M1 |
| State | 1748.3 MiB, save 4.46 s, load 2.75 s |
| Round trip | tokens and first logits row bit-identical (PASS), answer `MAGENTA-FALCON-42` |
| Decode after the 128K prefill | 4.54 tok/s |
| Peak VRAM / RSS | 7349 MiB / 674 MB |

**The < 1 h target is missed by 2.9 min.** By the cost model, attention is about 1580 s and the rest about 2195 s. Getting under 1 h needs about 5 % more, for example a prefill attention kernel with more than 48 rows per K/V tile, which needs fewer registers per row than the 178 used now.
