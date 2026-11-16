# Experiment branch `exp/bit-lop3-cudagraph-prefill` vs baseline (`main` state, pushed as `baseline/stage1-5`)

Prompt `bench/prompt512.txt` (509 tokens) + 256 generated tokens, temperature 0, seed 42, `--gpu-layers 32 --kv iso3 --dp4a all`.
GTX 1070 Ti (cap 7373 MiB), Ryzen 5 2400G on cores 0-3, 16 GB DDR4. One process at a time. Raw files: `bench/results/experiments/`.
The step-by-step record of every attempt (reason, command, output, decision) is `bench/results/experiments/EXPERIMENT-LOG.md`.

## Headline: baseline vs experiment, same conditions (A/B, alternating, warm page cache, `bench/ab_main_vs_exp.sh`)

| Binary (MTP 2) | Run | pp tok/s (prefill s) | tg tok/s | MTP accept | Peak VRAM MiB | Peak RSS MiB | tokens vs baseline run 1 |
|---|---|---:|---:|---|---:|---:|---|
| baseline (`main`) | 1 | 15.90 (32.0) | 2.578 | 96 % (169/176) | 7245 | 9215 | reference |
| experiment | 1 | **27.81** (18.3) | **2.625** | 95 % (168/176) | 7335 | 9102 | 256/256 identical |
| baseline (`main`) | 2 | 15.84 (32.1) | 2.557 | 96 % (169/176) | 7245 | 9215 | 256/256 identical |
| experiment | 2 | **27.81** (18.3) | **2.608** | 95 % (168/176) | 7335 | 9102 | 256/256 identical |

Prompt processing 1.75x, decode +2 %, RSS -113 MiB, VRAM +90 MiB (second staging buffer and CUDA graphs; 38 MiB under the cap).

Other experiment configurations on the same prompt (warm cache): `--mtp-steps 0` pp 28.22, tg 2.558; `--mtp-adaptive 1` pp 27.88 / 27.81, tg 2.570 / 2.572,
accept 96 % (198/207), peak VRAM 7359 MiB; all 256/256 tokens identical to the baseline.

## Sequential head-to-head against llama.cpp (`bench/compare_llamacpp.sh`, this round)

| Program | pp tok/s | tg tok/s | MTP accept | Peak VRAM MiB | Peak RSS MiB |
|---|---:|---:|---|---:|---:|
| llama-server `-ngl 25` iso3 KV + draft-mtp 2 (largest -ngl under the cap; -ngl 26 = 7473 MiB) | 27.25 | 2.519 | 91 % (164/181) | 7239 | 14569 |
| llama-server `-ngl 25`, no MTP | 29.51 | 2.241 | | 6605 | 14612 |
| experiment `--mtp-steps 2`, warm (A/B runs above) | 27.81 | 2.61-2.63 | 95 % | 7335 | 9102 |
| experiment `--mtp-steps 2`, **first engine run right after llama-server** (cold) | 26.26 | 2.024 | 95 % (168/176) | 7335 | 9087 |
| experiment `--mtp-steps 0`, second run after llama-server | 26.97 | 1.953 | | 7253 | 8660 |
| experiment `--mtp-adaptive 1`, third run after llama-server | 26.69 | 2.582 | 96 % (198/207) | 7359 | 9121 |

Previous round (Stage 5, baseline binary): llama-server -ngl 25 28.70 / 2.50, baseline engine 13.42 / 2.58.

**Reading.**
- Prompt processing is now at llama.cpp's level (27.8 vs 27.3 in this round, 28.7 in the previous round, 29.5 without MTP). It does **not** clearly beat it: the target "above 28.70 tok/s" is not reached.
  What is left is the PCIe Gen3 x1 link (7 GiB of CPU-layer weights at 0.84 GB/s = 8.9 s floor; measured 10.5 s) plus 7.3 s for the 32 GPU-resident layers.
- Decode beats llama-server by about 4 % when the page cache is warm (2.61-2.63 vs 2.52). The first two engine runs directly after a llama-server run were cold: llama-server's
  16 GB mapping had evicted the engine's file pages, the 7 GiB pin at load took 73 s instead of 43 s, and decode fell to 1.95-2.02 tok/s. The third run was back at 2.58.
  The baseline binary in the Stage 5 round did not show this drop; the cold-start sensitivity is new with the pinned registration and is recorded as a regression to fix
  (see the log, entry A3 and "Open problems").

## What each experiment contributed (measured)

| Experiment | Change | Result | Kept |
|---|---|---|---|
| A1 | CPU-only prefill for layers 32-63 (no PCIe streaming), as requested | 150.5 s for 509 tokens (3.38 tok/s), measured in Stage 4 (`p512_cpuprefill`) | no |
| A2 | pinned (cudaHostRegister read-only) mmap pages + two staging buffers, copy stream overlaps compute | prefill of 509 tokens (MTP 0) 27.2 s -> 18.1 s; the streamed half now takes 10.5 s against an 8.9 s PCIe floor | yes |
| A3 | MTP prefill only on the last 1 / 16 / 64 / 128 tokens | prefill -5 s, but acceptance 96 % -> 51-58 % at 256 tokens, tg 2.62 -> 1.82-1.94 | no (opt-in `--mtp-window`) |
| A4 | MTP KV prefill on the GPU (eh_proj, wk, wv only) | MTP prefill 5 s -> 0.2 s; acceptance unchanged (95 %), output identical | yes |
| A5 | dp4a token tile 32 / 64 for large batches | GPU layers 7.3 s -> 12.8 / 17.5 s (slower) | no |
| A6 | `__ldg` loads, 8 rows per block | no change (7.32 / 7.30 s) | no |
| B1 | 128-bit (`uint4`) / 64-bit weight loads, PTX `lop3.b32`, int4 activation loads | GPU decode 46.3 -> 41.8-43.5 ms/token (5-token prompt); `LDG.E.CI.128` x942 in SASS (0 before); LOP3 was already emitted by the compiler for the plain C++ (338 in the baseline kernel) | yes (neutral to slightly positive) |
| B2 | CUDA graph of the 32-layer stack for T = 1..5 | 43.5 vs 44.2 ms/token (no gain: launch overhead is not the bottleneck) | yes (default on, `--cuda-graph 0` off) |
| C1 | adaptive MTP (1..4 drafts, top-1 p >= 0.5 to keep, >= 0.85 to continue) | low-acceptance prompt 2.34 -> 2.56 tok/s (MTP 0: 2.59); high-acceptance prompts equal or -2 % | opt-in `--mtp-adaptive 1` |
| C2 | CPU matmul grain 8 -> 128 rows | CPU layers 298 -> 287 ms/token; DDR4 ceiling measured 33 GB/s (`tests/tools/membw.cpp`) | yes |
| C3 | software prefetch of the next weight rows | CPU layers 298 -> 384-396 ms/token (slower) | no |

Checks after the experiments: SASS gate PASS (IDP.4A x1632, HFMA2/HMUL2/HADD2 x0), rollback test bit-identical, `gemv_parity` PASS, greedy 24/24 identical to the llama-server baseline,
`tests/webui_test.py` 46/46, `tests/server_test.py` 47/47.

## Round 2026-11-16: multi-token W4A8 CPU kernel (A) + GQA invert (B) + RoPE table (C)

Same branch after the round in `docs/OPT_LOG_2026-11-16.md` (every attempt recorded there). Prompt files and token
counts match the Stage 4 pre-round baselines (`bench/results/stage4/pre_p5`, `pre_p512`, same harness, ctx 4096).

### Battery (`bench/results/stage4/run_post_battery.sh`, `taskset -c 0-3`)

| Prompt (gen) | MTP | pp tok/s (prefill s) | tg tok/s | accept | ids vs pre baseline |
|---|---|---:|---:|---|---|
| p5 (24) | 0 | 3.88 (1.3) | 2.811 | | IDENTICAL (24) |
| p5 (24) | 2 | 3.94 (1.3) | 2.527 | 75 % (15/20) | IDENTICAL (24) |
| p5 (24) | 3 | 3.91 (1.3) | 2.519 | 67 % (16/24) | |
| p5 (24) | adaptive | 4.00 (1.2) | 2.918 | 83 % (15/18) | |
| p64 (64) | 0 | 6.20 (10.0) | 2.665 | | |
| p64 (64) | 2 | 6.14 (10.1) | 3.205 | 100 % (42/42) | |
| p64 (64) | 3 | 6.15 (10.1) | **3.373** | 100 % (48/48) | |
| p64 (64) | adaptive | 6.16 (10.1) | 3.305 | 100 % (47/47) | |
| p512 (16) | 0 | 28.20 (18.1) | 2.814 | | IDENTICAL (16) |
| p512 (16) | 2 | 27.88 (18.3) | 3.301 | 100 % (10/10) | |
| p512 (16) | 3 | 27.81 (18.3) | **3.322** | 100 % (12/12) | |
| p512 (16) | adaptive | 27.81 (18.3) | 2.674 | 86 % (12/14) | |
| p512 (256) | 2 | 27.94 (18.2) | **3.013** | 95 % (168/176) | IDENTICAL (256) vs `ab_exp_1` |

Pre-round baselines (same harness, captured before the round): `pre_p5` tg 2.711 (MTP 0), `pre_p512` pp 28.25,
tg 2.494 (MTP 0).

**Reading.**
- MTP 0 decode (multi-token kernel inactive at T=1) improved tg 2.494 -> 2.814 on p512 (+13 %, 16-token sample) and
  2.711 -> 2.811 on p5: that is the CPU-side saving from B (six Q heads share one K/V row read) and C (RoPE cos/sin
  table lookup instead of per-token trig). Prefill unchanged (28.20 vs 28.25), as expected — it is GPU-bound.
- MTP 2/3 now clearly beat MTP 0 on this text (acceptance 100 %): tg 3.21-3.37, +17-27 % over the MTP 0 runs and
  +33 % over the pre-round baseline (2.494). Before the round MTP was slower than no-MTP on 75 % acceptance text; the
  multi-token kernel lowers the acceptance rate at which MTP pays off — this was the open "Next session" item 3 of
  Stage 5, now closed.
- Headline config (p512 + 256 tokens, the A/B section above): tg 2.625/2.608 -> **3.013** (+15 %), ids 256/256
  identical to `ab_exp_1`, MTP verify 87.72 -> 75.85 s (-13.5 %), draft 9.66 -> 8.99 s, acceptance unchanged at
  95 % (168/176), pp 27.94 vs 27.81, peak VRAM 7337 vs 7335 MiB.
- Short 5-token prompt stays MTP-hostile (acceptance 67-75 %, per-cycle overhead dominates 24 tokens); adaptive
  picks 2.918 there.
- Micro-bench of the kernel alone (48 MiB matrix, in 5120 x out 17408, `taskset -c 0-3`): T=4 speedup vs ggml
  vec_dot Q4_K 1.07-1.11x, Q5_K 1.31-1.35x, Q6_K 1.34-1.35x, IQ4_XS 1.21-1.33x; T=3 Q4_K 0.99x (parity),
  others 1.17-1.32x.
- VRAM peak 7255-7361 MiB, RSS peak 8.6-9.1 GiB across the battery: same range as the pre-round runs. No memory
  regression.
- Point D was measured and skipped: all 48 ssm_a tensors, 2304 values, min -139.42, max -0.0038, zero bitwise
  -1.0f (`bench/results/stage4/post_ssm_a_scan.txt`) — the sigmoid(-x) identity is invalid; the exact form still
  needs powf.
