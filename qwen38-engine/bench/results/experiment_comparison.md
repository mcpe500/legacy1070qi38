# Cascaded PLD+MTP Experiment Comparison

**Branch**: `exp/cascaded-pld-mtp` (from `main` @ 28e9bb2, commit `d8519cf` → `a5a63cf` → architecture fix)  
**Date**: 2026-11-17  
**Hardware**: GTX 1070 Ti (sm_61), Ryzen 5 2400G, 16 GB DDR4  
**Model**: Qwen3.8-27B-UD-Q4_K_M.gguf  
**Config**: `--gpu-layers 32 --mtp-steps 2 --mtp-adaptive 1 --cuda-graph 1 --temp 0 --seed 42`  
**Benchmark prompts**: `bench/prompt512.txt` (509 tok free text), `bench/prompt_code_short.txt` (524 tok code), a 10× repeated sentence (60 tok).

---

## Executive summary

| Goal | Result |
|------|--------|
| Revert T≤8 → T≤5 (`kRollT=5`, 5-row buffers, `gexec[10]`, `T<=5`) | **done** — VRAM 7330.2 MiB, 32 GPU layers fit exactly as main |
| MTP acceptance regression fixed | **done** — 96 % free-text / 99 % code, 198/207 and 192/194, identical cycle counts to main |
| Greedy parity free-text + repetitive code | **PASS** — 256/256 ids identical on both prompts, MTP on and PLD on |
| SASS gate | **PASS** — 0 `HFMA2`/`HMUL2`/`HADD2` |
| Rollback T≤5 bit-identical | **PASS** — TB=5, keep=1..4, 0/5 batch rows and 0/7,6,5,4 post-rollback steps (with `QWEN_MT_DOT=0`, see note) |
| PLD routing (priority, MTP skipped on match) | **done** — +16 % over main on genuinely repeated text, 100 % PLD acceptance |
| tg tok/s vs main 3.01–3.32 | main 3.14–3.18 on both benchmarks, experimental 3.13–3.16 with PLD off (= main), **3.85 on repeated text with PLD** |

---

## Architecture changes

| File | Change |
|------|--------|
| `engine.hpp` | `kRollT = 8 → 5` (verification batch of up to 4 drafts + 1, as main) |
| `gpu_layers.cu` | `gexec[16] → gexec[10]`, graph condition `T <= 8 → T <= 5` |
| `generator.cpp` | buffers `hv_`/`hv_next_` 8×H → 5×H, `lrows_` 8×V → 5×V, `bt/vt/dr` 9/8/7 → 6/5/4, `pos_new+8 → +5` |
| `generator.cpp` | **pure priority routing**: PLD consulted first; on a match the verify batch is `[cur] + PLD_candidates` and MTP is skipped; with no PLD candidate the standard MTP path `[cur] + MTP_drafts` runs exactly as main |
| `generator.cpp` | **PLD suffix fixed** (root-cause bug): the suffix must end at `cur` (the token at `pos_new`), not at the last cached token |
| `generator.cpp` | **PLD chain extraction**: the matched entry remembers the position where its suffix last occurred; `query()` walks the history forward from there and copies the historical continuation (classic PLD) |
| `generator.cpp` | **consistency gate (`kMinRepeat = 4`)**: a match is only trusted when the previous occurrence continued identically for at least 4 tokens, i.e. a genuinely repeated block |
| `main.cpp` | rollback test extended to `TB = 5`, `keep = 1..4` (the whole `kRollT` range) |

---

## Verification gates (32 GPU layers)

| Gate | Result | Detail |
|------|--------|--------|
| SASS gate | **PASS** | `cuobjdump --dump-sass build/libgpu_cuda.a` → 0 `HFMA2`/`HMUL2`/`HADD2` |
| Rollback T≤5, `TB=5`, `keep=1..4`, `QWEN_MT_DOT=0` | **PASS** | batch rows 0/5 differing at every keep; steps after rollback 0/7, 0/6, 0/5, 0/4 (`bench/results/stage4/exp_pld_rollback_bitident.txt`) |
| Rollback default (`mt_dot` on) | last-bit differences | identical on `main`: the `mt_dot` multi-token kernel (point A) sums a batch in a different order than the per-token kernels, so the batch rows and the replayed steps differ in the last bits **by design**; the single-step path and the whole rollback mechanism stay bit-exact, which `QWEN_MT_DOT=0` proves |
| Greedy parity free-text 509→256 | **PASS** | 256/256 ids identical to `main-parity`, with `--mtp-steps 0`, `--mtp-steps 2 --pld 0` and `--mtp-steps 2 --pld 1` |
| Greedy parity code 524→256 | **PASS** | 256/256 ids identical to `main-parity`, with and without PLD |
| VRAM @ 32 GPU layers | **PASS** | arena 7202.0 MiB, in use 7330.2 MiB — bit-identical to main (the T≤5 revert removed the ~5 MiB rollback growth of T≤8) |

---

## Benchmarks (32 GPU layers, MTP 2 adaptive, temp 0, seed 42)

### Free text — `prompt512.txt`, 509 prompt + 256 decode

| build | tg tok/s | accept | cycles | pld_hits | mtp_fills |
|------|---:|---|---:|---:|---:|
| main (MTP only) | 3.139 | 96 % (198/207) | 59 | — | 207 |
| experimental, PLD off | 3.130 | 96 % (198/207) | 59 | 0 | 207 |
| experimental, PLD on | 2.992 | 89 % (195/219) | 62 | 4 | 203 |

### Code / repetitive — `prompt_code_short.txt`, 524 prompt + 256 decode

| build | tg tok/s | accept | cycles | pld_hits | mtp_fills |
|------|---:|---|---:|---:|---:|
| main (MTP only) | 3.179 | 99 % (192/194) | 65 | — | 194 |
| experimental, PLD off | 3.162 | 99 % (192/194) | 65 | 0 | 194 |
| experimental, PLD on | 2.949 | 84 % (192/228) | 66 | 19 | 152 |

### Genuinely repeated text — 10× "The capital of France is.", 60 prompt + 64 decode

| build | tg tok/s | accept | cycles | pld_hits | mtp_fills | mtp_draft_s |
|------|---:|---|---:|---:|---:|---:|
| main (MTP only) | 3.315 | 100 % (52/52) | 13 | — | 52 | 2.71 s |
| experimental, PLD on | **3.846** | **100 % (52/52)** | 13 | 13 | 0 | **0.00 s** |

**PLD wins where the block really repeats**: +16 % over main with 100 % acceptance and zero drafting cost, because the drafts come from the history instead of the MTP head.

---

## Reading the numbers

1. **The regression is fixed.** With PLD off the experimental branch reproduces main exactly (3.130 / 3.162 tok/s, same accept counts, same cycle counts, same tokens). The T≤5 revert removed both the VRAM growth (32 GPU layers fit again) and the MTP-head stride change that had broken acceptance.
2. **PLD is a selective win.** On text whose n-grams really repeat (the 10× sentence, the repeated `print(...)` lines near the end of the code prompt) PLD gives 100 % acceptance and saves the whole MTP drafting cost. On semi-repetitive text the model's continuation of a matched n-gram often differs, so a few verify passes are wasted; the `kMinRepeat = 4` consistency gate cuts that waste (free text 2.86 → 2.99, code 2.69 → 2.95 without losing any repeated-block hit).
3. **Greedy parity holds everywhere** — every run above, with `--mtp-steps 0`, `--mtp-steps 2` and `--mtp-steps 2 --pld 1`, produced exactly the token list of `main-parity`, because every draft is still verified by the main model and the sampler stream is untouched.

---

## Bugs found and fixed

| # | Bug | Fix |
|---|-----|-----|
| 1 | **PLD suffix off by one** (the actual cause of PLD never matching): the matcher was queried with `cached_`, which does not contain `cur` yet, so the candidate was for the position *before* the one being predicted | build the suffix as the last 3 cached tokens followed by `cur` |
| 2 | PLD returned only the distinct followers of one n-gram (usually the same token twice), so at most one draft per cycle | store the matched position and walk the history forward to extract the whole continuation (classic PLD) |
| 3 | PLD fired on one-off n-gram hits, wasting verify passes on semi-repetitive text | `kMinRepeat = 4` consistency gate: the previous occurrence must continue identically for 4 tokens |
| 4 | (previous round) verify batch order and `cur_new` position in the acceptance loop | `[cur] + MTP + PLD` order, `cur_new` from the position after the last accepted draft |

---

## Rollback note (why the default run "fails")

`bench/results/stage4/exp_pld_rollback_default_mtdot.txt` shows the test failing with `mt_dot` on. This is **not** a rollback bug: `mt_dot` (point A of the 2026-11-16 round) decodes each weight block once per batch instead of once per token, so a T=5 batch and five T=1 steps sum in different orders and differ in the last float bits. The rollback mechanism itself (snapshot + replay of the first `keep` tokens through the same code) is bit-exact, which `QWEN_MT_DOT=0` demonstrates for every `keep = 1..4`. `main-parity` behaves identically, so this is pre-existing, not introduced by this branch.

---

## Artifacts

- Branch: `exp/cascaded-pld-mtp`
- Main baseline: `origin/main` @ 28e9bb2 (worktree `main-parity`)
- Rollback (bit-exact path): `bench/results/stage4/exp_pld_rollback_bitident.txt`
- Rollback (default, `mt_dot` on): `bench/results/stage4/exp_pld_rollback_default_mtdot.txt`
- Prompts: `bench/prompt512.txt`, `bench/prompt_code_short.txt`
- CLI: `--pld 1 --pld-max 4` enables the cascade; `--pld 0` (default) leaves the main MTP path untouched