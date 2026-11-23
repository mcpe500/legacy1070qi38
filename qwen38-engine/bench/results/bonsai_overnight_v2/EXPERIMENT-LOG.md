# EXPERIMENT-LOG: marathon v2 (issue #57)

Branch `exp/bonsai-overnight-v2-prefill-t17-drafter`. Six milestones, each committed and pushed at its end.
Everything below was measured on the locked hardware: GTX 1070 Ti (GP104, sm_61, 19 SMs, 8192 MiB,
222.5 GB/s measured DRAM read, 3650 G dp4a instructions/s), Ryzen 5 2400G, 16 GB DDR4, `taskset -c 0-3`,
`ninja -j4`, GPU otherwise free (checked by `scripts/gpu_isolate.sh` before every GPU run).

## Commits

| Commit | Milestone | One line |
|---|---|---|
| `06d6047` | Stage 0 | branch, `gpu_isolate.sh`, `verify_svg_layout.py`, SVG 55 fixed |
| `f37f3fe` | M1 | fast prefill PTQ1_0 implemented, gated, measured, kept out of the engine |
| `96d561c` | M1 | CLAIMS C-89/C-90 |
| `f3db07c` | M2 | verification batch T=17 lossless, buffers sized by the VRAM budget |
| `1c65334` | M2 | CLAIMS C-91/C-92/C-93 |
| `105a273` | M3 | Suffix-PLD fills the 17-row batch, cascaded with r3 MTP, measured |
| `f1b3e6e` | M3 | CLAIMS C-94..C-97 |
| `b253d1f` | M4 | Q2_K `__dp4a` kernel, `__CUDA_ARCH__` guard, multi-arch compile check |
| `ef3ae80` | M4 | CLAIMS C-98..C-101 |
| `ac39e31` | M5 | drafter schedule modes + CPU gate for the Colab pipeline |
| `b2a4cc0` | M5 | CLAIMS C-102..C-104 |
| this | M6 | regression, log, figure 57, PR |

## 1. M1 — the prefill wall (target: 128K prefill under 45 min, NOT met)

Baseline carried in: about 60.6 min for the 130965-token prefill (`M3-ATTEMPTS.md` of
`exp/bonsai-local-10h-opt-serving`), of which about 2195 s is the chunked GEMM and about 1430 s is the
prefill attention.

What was built and then rejected, with the numbers:

| Attempt | Result | Kept |
|---|---|---|
| `fast = true` PTQ1_0 prefill: two independent FP32 group accumulators, `<16,2>` | paired A/B T=16 1113.9 vs 788.3 µs (0.708x), T=64 4349.5 vs 3081.4 µs (0.708x); engine prefill of a 509-token prompt 9.50 s vs 7.17 s (0.76x) | no |
| the same with `<16,1>` / `<32,1>` / `<32,2>` / `<8,2>` / `<16,4>` | best 1.000x, worst 0.708x across all six tiles and both T | no |
| `__launch_bounds__(PW*32, 3)` on the tile2 kernel, to reach 3 blocks/SM | occupancy 2 → 3, registers 127 → 80 (local-memory spill), T=16 788.3 → 946.6 µs (0.83x) | no |

Root cause of the wall, measured rather than assumed: the tile2 kernel is **occupancy-bound, not
FP32-latency-bound**. `127 registers × 256 threads = 32512` of the 65536 registers per SM gives exactly two
blocks, and the extra accumulators of the fast path only steal registers. Per k step a thread issues 32
`IDP.4A` and about 96 FP32 instructions for 256 MACs (about 1.03 issue slots per MAC), and the measured
1850-1870 G int8 MAC/s is about 47% of the lane-slot ceiling of 19 SMs × 4 partitions at 1.683 GHz.

Kept: the kernel (behind `fast = true`), its tile knob, its numeric gate (`ptq1_fast_gate` 42/42, cosine
1.00000000 against an FP64 reference, greedy token identical), and the paired A/B section of the roofline
bench. The `--fast-prefill` CLI flag and the config plumbing were **removed**: the milestone gate required
greedy identity, and the path both fails it at the engine level (tokens diverge at step 16) and is slower.

## 2. M2 — verification batch T=17, lossless (target met)

- Split-K decode attention now serves `T <= verify_t` with a slice count that follows `ctx_max` only
  (`min(ctx_max/256, 100)`, rounded to 4), so a verify row takes the same reduction as the single-token
  decode row of the same position.
- The partial buffer is `verify_t × 100 × 24 × 258 × 4 B` = 40.05 MiB at 17 (was 12.1 MiB at 8 rows × 64
  slices), and the rollback buffers follow the same value.
- The engine walks `17 → 8 → 5` against the projected VRAM of the whole plan, probed with a zero-layer plan
  first and again with the final layer count, and reports the choice in the JSON summary.
- Measured: iso3 rows 0..16 of a T=17 batch, the same rows of a T=5 batch and 17 single-row runs are
  **byte-identical** at ctx 512/4096/32768/65536; the batch stays at rel 2.0e-05 against an FP64 reference;
  `dn_rollback_test` reproduces a keep-row batch byte-identically for keep = 1..17; per-row latency 25.8 µs
  at 512 and 865.8 µs at 64K (against 1463.9 for a T=5 batch), which is the ceiling a DFlash B=16 drafter is
  sized against. VRAM: 6574 MiB at 4K, 7002 at 32K, 7352 at 128K (cap 7373) with the MTP sidecar.

## 3. M3 — Suffix-PLD up to 16 drafts, cascaded (target: 16 drafts, lossless; no speed win)

- The generator builds batches of up to `vcap_ = max(5, min(17, Engine::roll_t()))` rows, so the draft count
  follows the buffer the engine allocated.
- The PLD n-gram index is asked for up to 16 candidates and fills the whole batch; cascade order unchanged
  (PLD first, MTP only on no match).
- Measured at 4K over four workloads: cascade ids are byte-identical to r3-MTP ids on all four, and PLD-only
  ids are byte-identical to non-MTP ids on the Indonesian chat workload. PLD fires 6 cycles out of 4 × 64
  tokens, and its step costs 0.39-0.63 µs against a 200 µs budget.
- No speed win anywhere: the only workload where the match fires often (Indonesian chat) is the one where a
  16-draft list is mostly rejected, 19.76 tok/s at `pld-max` 2 down to 3.29 at 16. The default stays 4.

Failed attempt worth recording: the first cascade runs on the English and Indonesian chat workloads produced
zero tokens, because a raw `User:`/`Assistant:` transcript makes the model emit EOS immediately; the workloads
were rewritten as flowing prose before any number was taken.

## 4. M4 — Q2_K kernel and the arch dispatch (target met for Q2_K)

`Prep<GT_Q2_K>` gives Q2_K the same `__dp4a` path as the other K-quants. Three layout mistakes were made and
fixed before it was right, and each is now recorded in code:

1. The block was assumed to be `[d|dmin|scales|qs]`; ggml's `block_q2_K` is
   `[scales[16]|qs[64]|d fp16 @80|dmin fp16 @82]`.
2. `ld128` on the payload faults: an 84-byte block stride leaves `blk + 16 + 32*half` only 4-byte aligned.
3. The scale byte read as `int8_t` sign-extends the high nibble; ggml shifts it logically (rel_rms 0.62 until
   it was read as `uint8_t`).

Result: rel_rms 8.2e-08 to 8.4e-08 against libggml's own dequantizer at T = 1/3/16/64, and 1071 G int8 MAC/s
at T=64 against Q4_K's 1091 (98%), 1.72x its weight bytes per second. `scripts/arch_compile_check.sh`
compiles all 7 kernels for sm_61, sm_75, sm_86 and sm_89 (28/28), and `ternary_gemv.cu` grew a real
`__CUDA_ARCH__` branch (4 blocks/SM minimum only on sm_90+, where the register file allows it).

Not done: IQ2_XXS/IQ2_XS/IQ2_S stay CPU-only (the value is `LUT[4] × sign`, so the expansion costs about 8
instructions per value against Q2_K's 1 per 4); Q3_K/Q4_K/Q6_K unpack experiments were not re-run because the
M1 finding removes the case for them.

## 5. M5 — Drafter v2 pipeline (CPU-only, reproducible)

`unroll_K(args)` is the single source of the unroll count; `--unroll` is 1..8 for `mtp`; `--drafter df3
--df3-block 16 --df3-layers {2,3}` runs the block schedule whose draft count equals the engine's 17-row
verification batch. `tools/bonsai_mtp_train/verify_m5.sh` gates it on CPU: trainer selftest, the schedule
matrix, the CLI bound, the notebook check, the GGUF name/shape audit against `Engine::load` (15 tensors, none
missing, none extra, `eh_proj (5120, 10240)`), and the ProCreations embedding-fix audit (the converter
unrotates `token_embd` into the plain basis and `Engine::mtp_forward` rebuilds the same row).

Not done: no training run (needs the 54 GB BF16 teacher on the RTX PRO 6000), a real 2-3 layer DFlash head
(the loader knows the one-layer `nextn` layout), and therefore no T=17 verification of a real sidecar.

## 6. M6 — regression, figure, PR

Final gates, all on this branch: `ternary_ab_test` 180/180, `ternary_test`, `ptq1_fast_gate` 42/42,
`dn_rec_ab_test`, `dn_rollback_test` 9/9, `verify_t17_test` 32/32, `q2_k_parity` 4/4, `attn_long_test`
(reuse 98.27 ms vs old tiled 2179.82 ms at n = 131072), `arch_compile_check` 28/28, `verify_m5.sh` 6/6,
SASS gate (IDP.4A ×6205, zero half arithmetic). Figure `57_bonsai_fastprefill_t17_drafter.svg` passes
`verify_svg_layout.py` with 0 errors and 0 warnings.

Not run in this milestone: the HTTP server suites (`server_test.py` 69/69 with `QWEN_MT_DOT=0`,
`bonsai_server_stress.py`), the headless WebUI test, the Qwen greedy 48/48 set, and a fresh 128K needle run.
They are unchanged by this branch's commits (no engine-layer kernel, attention or GEMV path that those suites
exercise was modified), but they were not re-run here, so no claim is made about them.

## 7. PR #53 vs PR #56 vs marathon v2

| Item | PR #53 | PR #56 | marathon v2 (this branch) |
|---|---|---|---|
| prefill 128K (130965 tokens) | 3774.7 s (62.9 min) | 3625.4 s (60.4 min) | not re-measured; unchanged (about 60.4 min) |
| decode PTQ1_0 non-MTP | 13.30 tok/s | 17.37 tok/s | 17.38 tok/s (unchanged) |
| decode with r3 MTP | 18.48 tok/s | 26.65 tok/s (Q4_K_M) | 19.39 tok/s on the refactor workload, 20.19 on Indonesian chat (the sidecar is trained for chat, so the coding number moved) |
| decode after a real 128K | 4.54 tok/s | 8.73 tok/s | 8.7 tok/s (unchanged) |
| verification batch | 5 rows, `512/T` slices above 26K | 5 rows, 100 fixed slices | up to 17 rows, 100 fixed slices, lossless (M2) |
| PLD drafts per cycle | - | up to 4 | up to 16, MTP-cascade, lossless, no speed win (M3) |
| Q2_K GEMV | CPU path | CPU path | GPU `__dp4a` kernel, 1071 G MAC/s at T=64 (M4) |
| multi-arch | sm_61 only | sm_61 only | sm_61 build unchanged, compile-checked for sm_75/86/89 (M4) |
| new gates in the tree | - | - | `ptq1_fast_gate`, `verify_t17_test`, `dn_rollback_test`, `q2_k_parity`, `arch_compile_check`, `verify_m5.sh` |
## 8. MTP sidecar race — user head (2.000-step distillation) vs ProCreations r3 (2026-11-23)

The Colab head trained in milestone 5's pipeline (2.000 steps, 65.54 M tokens, 114.610 samples,
warm start `qwen_blk64`, exported by the user as `mtp_blk64_q4_k_m.gguf` / `q8_0`, sha256-verified
against the training MANIFEST) raced the r3 sidecar on this GPU, same session, same flags
(`--mtp-adaptive 1`, ctx 4096, greedy, 256 tokens, lossless — ids identical to `--gen-mtp-steps 0`
in every run):

| workload | no MTP | r3 | user Q4_K_M | user Q8_0 |
|---|---|---|---|---|
| prompt512 | 17.34 | 27.80 | **28.39** | 27.59 |
| coding (LRU cache) | 17.31 | 27.81 | **28.48** | 27.17 |
| Indonesian (list/tuple/set) | 17.27 | **25.85** | 24.83 | 24.58 |

Verdict (details + logs in `mtp_race/results.md`): adopt the user Q4_K_M sidecar as the default for
coding-agent use (+2.1 % / +2.4 % over r3); r3 keeps Indonesian by 4 % on the one prompt tried.
Adaptive mode never spends a third draft here (`--gen-mtp-steps 3` == `2` everywhere), and the CLI
hard-caps the flag at 3. A 300–500-step polish run (`ONPOLICY_FRAC 0.15`, `LR 3e-5`, resume from the
safetensors) is the optional follow-up if the Indonesian gap matters.
