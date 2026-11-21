# M2: verification batch T = 17 (lossless) + dynamic VRAM buffers

Branch `exp/bonsai-overnight-v2-prefill-t17-drafter`. A Drafter v2 batch is 16 drafts plus 1 target row, so
the verification batch is 17 rows. This milestone makes that batch legal, lossless and affordable.

## What changed

1. `src/cuda/ops.cu`: the split-K decode attention path now serves `T <= verify_t` instead of `T <= 8`, and
   the slice count is `min(ctx_max / 256, 100)` rounded to a multiple of 4, i.e. it follows `ctx_max` only
   and no longer depends on T. A verification row therefore takes exactly the reduction the single-token
   decode row of the same position takes, which is what keeps speculative decoding lossless. The buffer
   that holds the (token x slice) partials is now `verify_t * 100 * 24 * 258` floats
   (`op_attention_part_floats`, 11.8 MiB at 5, 18.8 MiB at 8, 40.0 MiB at 17) instead of `8 * 64 * ...`,
   and the prefill path may slice over the positions with the larger row budget
   (`kPrefillPartRows()`).
   `op_attention_set_verify_t()` / `op_attention_verify_t()` carry the value into the launcher.
2. `GpuConfig::verify_t` / `EngineConfig::verify_t` (default 17): the engine asks for 17 and falls back.
3. `src/engine/engine.cpp`: the fallback chain `17 -> 8 -> 5`. Each candidate is projected against the whole
   VRAM plan (weights + KV + the attention partial buffer + the DeltaNet rollback buffers) and the largest
   one that fits `vram_budget_mib` wins. The chain is probed twice: once with a zero-layer plan to pick the
   value used while the layer count is chosen, and once with the final layer count, so a fixed
   `--gpu-layers 64` at 128K degrades the buffer instead of refusing the run. With MTP off there is no
   verification batch at all, so the buffers stay at 5 and the plan costs the least.
4. `Engine::roll_t()` / `Engine::verify_t()` and the `verify_t` / `verify_t_requested` / `roll_t` fields of
   the JSON summary make the chosen value visible in every run.
5. The DeltaNet rollback buffers are sized by the same value (`roll_t_`), so a 17-row batch can be rolled
   back on the GPU and on every CPU layer.

## Measurements

### Lossless (GTX 1070 Ti, `taskset -c 0-3`, GPU otherwise free)

`tests/cuda/verify_t17_test.cu`, 32/32 PASS
(`bench/results/bonsai_overnight_v2/verify_t17_test.txt`). iso3 KV (the Ternary Bonsai path a verification
batch actually runs on): rows 0..16 of a T = 17 batch, the same rows of a T = 5 batch and 17 single-row
runs at the same positions are **byte-identical** at ctx 512 / 4096 / 32768 / 65536:

```
[PASS] kv=iso3 n=   512 lossless: T=17 vs T=5 0.0e+00, rows 0..16 vs single rows 0 differing floats (max drift 0.000e+00)
[PASS] kv=iso3 n=  4096 lossless: T=17 vs T=5 0.0e+00, rows 0..16 vs single rows 0 differing floats (max drift 0.000e+00)
[PASS] kv=iso3 n= 32768 lossless: T=17 vs T=5 0.0e+00, rows 0..16 vs single rows 0 differing floats (max drift 0.000e+00)
[PASS] kv=iso3 n= 65536 lossless: T=17 vs T=5 0.0e+00, rows 0..16 vs single rows 0 differing floats (max drift 0.000e+00)
```

Against the FP64 reference the whole 17-row batch stays at rel 2.023e-05 at worst (bar 3e-4). The f32 KV
format (Qwen, where no verification batch exceeds 5 rows) routes T > 8 through the prefill kernel, so the
batch and the single rows take different kernels there: the drift is 4.1e-08, which the test reports and
keeps under a 1e-5 bar instead of a bit bar.

`tests/cuda/dn_rollback_test.cu`, 9/9 PASS (`.../dn_rollback_test.txt`): restoring the snapshot and
replaying conv + recurrence for `keep` rows reproduces a `keep`-row batch byte-identically for
keep = 1, 2, 3, 4, 5, 8, 12, 16, and keep = 17 reproduces the full 17-row batch state.

### Latency of the verification batch (iso3, ctx = n)

| ctx | T=1 | T=5 | T=17 | per row T=1 / T=5 / T=17 |
|---|---|---|---|---|
| 512 | 57.8 us | 131.3 us | 438.5 us | 57.8 / 26.3 / 25.8 us |
| 4096 | 450.7 us | 1289.2 us | 3820.6 us | 450.7 / 257.8 / 224.7 us |
| 32768 | 1324.8 us | 3625.3 us | 7302.2 us | 1324.8 / 725.1 / 429.5 us |
| 65536 | 2613.7 us | 7319.7 us | 14719.4 us | 2613.7 / 1463.9 / 865.8 us |

The 17-row batch costs about 29% of a T=5 batch's per-row cost at 512 and about 59% of it at 64K, so a
DFlash-style batch of 16 drafts amortizes the K/V pass far better than five sequential draft cycles. This
is the ceiling the Milestone 3 drafter is sized against: 16 drafts of 865.8 us at 64K.

### The engine's choice (PTQ1_0-mtpqwen, `--mtp-steps 2`, 509-token prompt, 8 generated tokens)

| ctx | verify_t | roll_t | vram used |
|---|---|---|---|
| 4096 | 17 | 17 | 6574.2 MiB |
| 65536 | 17 | 17 | 7002.2 MiB |
| 131072 | 17 | 17 | 7352.2 MiB (cap 7373) |
| 131072, `--vram-budget-mib 7500`, 64 layers | 5 | 5 | 7446.2 MiB |

The last row is the fallback at work (probe output: cand 17 -> 7531.5 MiB, cand 8 -> 7504.3 MiB, cand 5 ->
7474.0 MiB). With `--mtp-steps 0` the same runs report `verify_t 5` and `roll_t 0`.

### Regression

`ternary_ab_test` 180/180 bit-identical, `ternary_test` PASS, `ptq1_fast_gate` 42/42, `dn_rec_ab_test`
PASS, `attn_long_test` PASS (reuse 98.27 ms vs old tiled 2179.82 ms at n = 131072), SASS gate PASS
(IDP.4A present, zero HFMA2/HMUL2).

## What this does NOT do yet

The generator's cascaded path still builds verification batches of at most 5 rows
(`verify_T = vt_idx`, `std::vector<int32_t> vt(5)`, PLD candidates up to 4). Turning on a 17-row batch is
Milestone 3: the PLD matcher must return up to 16 drafts, the MTP head strides must cover 16 draft rows,
and the accept/rollback loop must walk 16 positions. The buffers and the lossless property measured here
are what that work needs, and it is gated on nothing else.
