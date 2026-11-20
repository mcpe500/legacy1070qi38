# Experiment log: `exp/bonsai-local-10h-opt-serving` (issue #55), 2026-11-20

Local GTX 1070 Ti marathon: Ternary Bonsai 2 27B PTQ1_0 with the ProCreations MTP head, decode and prefill kernels, server
hardening. No Colab, no RTX PRO 6000. The branch starts from main `ac0eab4` (PR #53 merged) plus `926e022` (the commit of
the still-open PR #54: `--mtp-only-out`, resume and warm start in `tools/bonsai_mtp_train/`), fast-forwarded in.

Hardware: GTX 1070 Ti (sm_61, 8 GB), Ryzen 5 2400G, 16 GB DDR4. Budgets: 7373 MiB VRAM, 14.2 GB RAM.

Rules on every step:
- `ninja -j4` through `taskset -c 0-3`; every engine and test run pinned with `taskset -c 0-3`; every shell command
  through `rtk`.
- One process on the GPU: the opencode session (PIDs 1463 / 1781) was paused with SIGSTOP (`bench/gpu_isolate.sh pause`)
  for the whole run and resumed with SIGCONT at the end; `bench/bonsai_ctx_run.sh` refuses to start on a busy GPU and
  samples the VRAM every 50 ms.
- Long queues run on a frozen copy of the binary (`ENGINE_BIN`), so a rebuild cannot mix two builds in one table.
- Numbers are copied from the result files in this folder. Failed attempts are listed with their numbers.

One rule was broken once: a GPU unit test (`attn_long_test`, about 1.2 GB) ran while an idle benchmark server held
6.6 GB, and the GPU total reached 7855 MiB (`suite_vram.jsonl`, `suite_512_q8`). No request was running; the engine
stayed under its own cap. From then on GPU tests only ran on an empty GPU (`nvidia-smi --query-compute-apps`).

## M1: ProCreations MTP head as `--mtp-gguf` sidecars, A/B/C/D benchmark

See `M1-README.md` (audit, sidecars, readings), `M1-SUITES-512-4K.md` and `M1-SUITES-32K.md` (all rows).

- Audit: the "Bonsai MTP embedding fix" (inverse Hadamard + signs on the latent `tok_embd` in the llama.cpp MTP graph)
  was already in the engine (`hadamard_unrotate` in `mtp_forward` / `mtp_prefill`). Conventions (output-normed h,
  pairing p with h[p-1], zero-centred norms + 1) match.
- `inject_mtp_gguf.py`: HF `mtp.*` checkpoints, raw donor copy. Sidecars: r3 Q4_K_M 272.9 MiB, r3 Q8_0 430.4 MiB,
  donor 334.7 MiB.
- `--mtp-gguf` (CLI and server): second mmapped index for `blk.64`, used by the CPU block, the GPU upload, the mlock of
  the block and the state fingerprint. The Q8_0 scratch of the GPU is sized for the widest Q8_0 input of the block
  (17408 for the r3 Q8_0 `ffn_down`; the first run failed with `invalid argument` at 5120).
- Placement: CPU MTP now takes its draft logits from the GPU LM head (5.08 -> 15.05 tok/s with `--gpu-mtp 0`); auto mode
  drops MTP when its rollback buffers would push the LM head to the CPU (128K: 2.46 -> 13.52 tok/s on a short prompt).
  The head fit check also tried only the first smaller chunk size (`break` instead of `continue`): fixed.
- Results: r3 Q4_K_M best on all suites, adaptive drafting best or near-best, cascaded PLD + MTP never better than MTP
  alone, every run lossless (identical text / ids to plain greedy).

## M2: decode kernels

See `M2-ATTEMPTS.md` (every attempt with its numbers).

- `gemv_ptq1_0_kernel_v3`: arithmetic TQ1_0 digit decode instead of shared-memory tables; bit-identical (180/180) on
  canonical weights (`ternary_ptq1_canonicalize` at upload, 0 bytes changed on the real file). T = 1: R = 4; T = 2..5:
  R = 3 / 4 / 4 / 3.
- Decode attention v2 for the long-context iso3 path.
- 509-token prompt: 13.30 -> **17.37 tok/s** without MTP (target 15), **26.65 tok/s** with the r3 Q4_K_M head adaptive.
- Depth probe 128K: 4.72 -> **8.91 tok/s**.
- Best MTP at a real 64K context (coding, r3 Q4_K_M adaptive): 14.13 tok/s vs 11.58 without drafts (1.22x), ids
  identical.
- A bug of my own decode attention v2, found by the 64K run: its slice count depended on T, so MTP verify rows did not sum
  like single-token rows at ctx > 26K, and m2 / adaptive diverged from m0. The slice count is now fixed for T <= 5, and
  the 64K runs are lossless again (`pre_fix_64k/` keeps the broken runs).
- Final gates on the final build: SASS PASS (0 HFMA2/HMUL2/HADD2), `ternary_ab_test` 180/180, `attn_long_test` 52/52
  (`attn_long_test_final.txt`). Lossless at ctx 65536: 509-token prompt m0 / m2 / adaptive ids `26a58a94`
  (`m2_fix_ctx65536_p512_*`); real 64K coding context m0 / m2 / adaptive / PLD+adaptive ids `d7d3417a`.
- Depth probe on the final build (`m2_depth3_*`): 32K 14.03, 64K 11.82, 128K 8.90 tok/s (peak 6351 / 6759 / 7349 MiB).

## Quantization roofline and GPU dispatch (second session request)

See `docs/QUANT_MATH_AND_GPU_DISPATCH.md` (formal arithmetic of every format, the identities the kernels rely on, the
`__CUDA_ARCH__` plan) and `bench/quant_roofline_bench.cu` (`roofline_before.*`, `roofline_after.*`).

- Measured ceilings: DRAM read 222.4 GB/s (86.9% of 256 GB/s), dp4a 3.65 T instr/s (14.6 T int8 MAC/s), FFMA 3.53 T/s.
- T=1: Q4_K / Q5_K / IQ4_XS reach 88-91% of the measured bandwidth, PTQ1_0 54% (issue-bound at 1.75 bpw), Q3_K 30%,
  Q6_K 52%. T >= 16: 4-11% of the int8 MAC peak for every format.
- Kept: **batched Q8_0 GEMV** (8 rows per block, weights decoded once per 8 tokens, bit-identical to the old kernel):
  T=4 2126 -> 622 us, T=64 33.8 -> 9.0 ms; `q8_gemv_parity` PASS; Qwen greedy 48/48; r3 Q8_0 MTP head adaptive
  26.96 tok/s on the 509-token prompt (ids `26a58a94`).
- Not kept: Q6_K `q - 32` and Q3_K `low2 - 4*!hbit` with OR/SUB/XOR instead of `__vsub4` / `__vcmpeq4` (same bytes):
  Q3_K T=1 576.6 -> 613.9 us, Q6_K 631.0 -> 683.9 us, reverted. The FWHT + Q8_1 fusion already exists
  (`rotate_q8_kernel`, shared memory); a register-only FWHT was not built (shuffles load the MIO pipe that limits the
  GEMVs). The base-3 decode without div/mod is the M2 v3 kernel. IQ2 formats and Q2_K have no GPU kernel.
- Only sm_61 is built and measured (hardware lock); the dispatch for sm_75..sm_120 is a documented plan.

## M3: 128K prefill

See `M3-ATTEMPTS.md` (8 attempts with their numbers).

- Kept: the raw iso3 bytes of the next K/V tile are prefetched into registers during the dequant and compute of the
  current tile (98.5 -> 85.4 ms per layer at T = 64, n = 131072).
- Full run, 130965 tokens: **3625.4 s = 60.4 min** (PR #53: 62.9 min). Needle `MAGENTA-FALCON-42` found in memory and
  from the state file; the round trip of the 1748 MiB state is bit-identical (save 4.49 s, load 2.73 s). Decode after the
  real 128K context: 4.54 -> 8.66 tok/s. Peak 7349 MiB.
- **The < 50 min target was missed by 10.4 min.** The GEMM part (about 2195 s) is at the balance of FP32 issue and
  shared-memory activation reads for the bit-identical accumulation order; the attention part (about 1430 s) is
  latency-bound with one 8-warp block per SM (229 registers); see the analysis at the end of `M3-ATTEMPTS.md`.

- Second session: one more GEMM attempt (M3-ATTEMPTS #10, more rows per warp with token slices: no gain) and a re-run
  of the full 130965-token prefill on the final build (`m3_final_PTQ1_0_ctx131072_statetest`): **3636.85 s = 60.6 min**,
  needle found in memory and from file, round trip bit-identical (save 4.56 s, load 2.77 s), decode after 128K 8.73 tok/s,
  peak 7349 MiB. K/V reuse above 48x was not reached: 48 rows already need 229 registers per thread, and the variants
  with more or fewer rows per block (#4, #6, #9) were slower. Drafter verify limits (T=5 lossless, T=17 needs, VRAM rows):
  `docs/DRAFTER_V2_CONTRACT.md`.

## M4: server

See `M4-README.md`.

- `--state-save` in the server (`POST /v1/state/save`, save on a clean SIGINT/SIGTERM shutdown), the `--state-load`
  state pinned as a checkpoint, `--mtp-gguf` in the server.
- Multi-turn cache fix: prompt checkpoint (recurrent state after every prefill), so a reply that does not round-trip token
  for token (stop string, `max_tokens` inside an accepted MTP batch, trimmed or re-tokenized history) no longer forces a
  full prefill. `x_stats.reuse` reports the source.
- PLD reported `false` without a draft head (it runs on the MTP verify path).
- Final build re-run: Bonsai stress 57/57 (`m4_bonsai_server_stress_final.log`), `server_test` 69/69
  (`m4_server_test_final.log`), `webui_test` PASS, Qwen rollback bit-identical (`final_qwen_rollback_test`).
- Pre-training contract for MTP / EAGLE-3 / DFlash: `docs/DRAFTER_V2_CONTRACT.md`.
- Regression: Bonsai stress 57/57 (10 turns, VRAM 6545 MiB, RSS 1123 MiB), `server_test` 69/69, `webui_test`,
  `stop_filter_test` 36/36, Qwen greedy 48/48 and rollback, Bonsai MTP rollback, `dn_rec_ab_test`, `iso3_gpu_test`.

## Final numbers

| Metric | PR #53 (main) | this branch |
|---|---:|---:|
| PTQ1_0 decode, 509-token prompt, no MTP | 13.30 tok/s | **17.37 tok/s** |
| PTQ1_0 decode with MTP (509 tokens) | 18.48 (Qwen blk.64, MTP 2) | **26.65** (r3 Q4_K_M, adaptive) |
| Depth probe 32K / 64K / 128K | 9.29 / 7.06 / 4.72 | 13.48 / 11.52 / **8.91** |
| MTP at a real 64K context (coding) | - | 14.13 tok/s adaptive (1.22x over 11.58), lossless |
| 128K prefill (130965 tokens) | 62.9 min | 60.4 min (final build re-run 60.6 min; target 50 missed) |
| Q8_0 GEMV T=4 / T=64 (ffn_up shape) | 2126 us / 33.8 ms | 622 us / 9.0 ms |
| r3 Q8_0 MTP head, adaptive, 509 tokens | - | 26.96 tok/s |
| Decode after a real 128K context | 4.54 | 8.66 |
| GEMV decode kernel ffn_up T=1 / lm_head T=1 | 208.3 / 2778 us | 150.1 / 1997 us |
| Decode attention per layer, n = 131072, T = 1 / 5 | 8.80 / 37.0 ms | 3.48 / 15.5 ms |
| Server checks | `server_test` 69/69 | + Bonsai stress 57/57 |
