# Cascaded PLD+MTP Experiment Comparison

**Branch**: `exp/cascaded-pld-mtp` (from `main` @ 28e9bb2)  
**Date**: 2026-11-16  
**Hardware**: GTX 1070 Ti (sm_61), Ryzen 5 2400G, 16 GB DDR4  
**Model**: Qwen3.8-27B-UD-Q4_K_M.gguf  
**Config**: `--gpu-layers 24 --mtp-steps 2 --mtp-adaptive 1 --cuda-graph 1 --temp 0 --seed 42`

---

## Summary

| Metric | Main (baseline) | Experimental | Δ |
|--------|-----------------|--------------|---|
| **Free-text (prompt512.txt, 509→256)** | | | |
| decode tok/s | 2.67 | 0.63 | **-76%** |
| MTP acceptance | 96% | ~0% | **-96 pp** |
| cycles | 60 | 243 | +305% |
| **Code/repetitive (prompt_code_short.txt, 524→256)** | | | |
| decode tok/s | 2.75 | 0.68 | **-75%** |
| MTP acceptance | 100% | ~0% | **-100 pp** |
| greedy parity | ✅ | ❌ (diverges at ~token 60) | — |
| **Verification Gates** | | | |
| SASS gate (HFMA2/HMUL2) | 0/0 | 0/0 | ✅ PASS |
| Greedy parity (free-text) | ✅ | ✅ | ✅ PASS |
| Greedy parity (code) | ✅ | ❌ diverges | ❌ FAIL |
| Rollback T≤8 bit-identical | ❌ fails | ❌ fails | — (pre-existing) |

---

## Detailed Results

### Free-text Benchmark (prompt512.txt, 509 prompt + 256 decode)

**Main branch** (adaptive MTP):
```
decode_tok_s: 2.6708
cycles: 60
draft_n: 205, draft_n_accepted: 197 (96%)
accept_hist: [4,4,5,5,42]
mtp_verify_s: 84.6s
```

**Experimental branch** (cascaded PLD+MTP, PLD disabled):
```
decode_tok_s: 0.6259
cycles: 243
draft_n: 901, draft_n_accepted: 12 (1.3%)
accept_hist: [232,10,1,0,0,0,0,0,0]
mtp_verify_s: 361.0s
```

**Experimental branch** (cascaded PLD+MTP, PLD enabled `--pld 1 --pld-max 6`):
```
decode_tok_s: 0.4347
cycles: 255
draft_n: 1737, draft_n_accepted: 0 (0%)
accept_hist: [255,0,0,0,0,0,0,0,0]
pld_hits: 253, pld_candidates: 792
mtp_verify_s: 538.4s
```

### Code/Repetitive Benchmark (prompt_code_short.txt, 524 prompt + 256 decode)

**Main branch** (adaptive MTP):
```
decode_tok_s: 2.7511
cycles: 58
draft_n: 198, draft_n_accepted: 198 (100%)
accept_hist: [3,5,2,3,45]
```

**Experimental branch** (cascaded PLD+MTP, PLD disabled):
```
decode_tok_s: 0.6797
cycles: 243
draft_n: 792, draft_n_accepted: 12 (1.5%)
accept_hist: [232,10,1,0,0,0,0,0,0]
```
**Token divergence**: Experimental output diverges from main at ~token 60 (repetitive loop behavior differs).

### Non-adaptive MTP (prompt512.txt)

**Main branch**:
```
decode_tok_s: 2.5151
cycles: 88, draft_n: 176, accepted: 169 (96%)
accept_hist: [2,3,83,0,0]
```

**Experimental branch**:
```
decode_tok_s: 0.9086
cycles: 242, draft_n: 484, accepted: 13 (2.7%)
accept_hist: [231,9,2,0,0,0,0,0,0]
```

---

## Implementation Status

### ✅ Completed
1. **T≤8 verify batch**: `kRollT = 8` in `engine.hpp`, `gexec[16]` + `T <= 8` in `gpu_layers.cu`
2. **PLD 3/4-gram matcher**: Fixed open-address hash tables in `generator.cpp` (anonymous namespace)
3. **Cascade logic**: PLD candidates → verify batch → MTP fills remaining slots
4. **Buffer sizing**: `hv_`/`hv_next_` 8×H, `lrows_` 8×V, `bt(9)` `vt(8)` `dr(7)`, `accept_hist[9]`
5. **CLI flags**: `--pld`, `--pld-max`, `--pld-min-len` in `main.cpp`
6. **SASS gate**: 0 HFMA2/HMUL2/HADD2 ✅
7. **Greedy parity (free-text)**: 256/256 tokens identical ✅

### ❌ Issues
1. **MTP acceptance near 0%**: Both adaptive and non-adaptive modes reject nearly all drafts
2. **Greedy parity fails on repetitive code**: Tokens diverge after ~60 tokens
3. **Rollback test fails on both branches**: Pre-existing issue with batch vs sequential numerics (not introduced by this experiment)
4. **PLD candidates not accepted**: 792 candidates generated, 0 accepted

### Root Cause Hypothesis
The MTP drafting logic was copied from original but the acceptance/rollback interaction with PLD-inserted verify batch may have a subtle bug:
- Original: verify batch = `[cur, dr[0], dr[1], ...]`
- Experimental: verify batch = `[cur, PLD..., dr[0], dr[1], ...]`
- Acceptance loop checks `vt[j]` but rollback uses `k < n_pld + De`
- The `hm`/`hm2` pointer swap logic for MTP chaining may be disrupted by PLD insertion

---

## VRAM / Memory

| Config | VRAM (MiB) | RSS (MiB) |
|--------|-----------|-----------|
| Main (adaptive) | 5536 | 10652 |
| Experimental (adaptive, PLD off) | 5540 | 11193 |
| Experimental (adaptive, PLD on) | 5548 | 11204 |

VRAM increase: ~12 MiB (within budget)

---

## Next Steps (if continuing)
1. Debug acceptance loop: add logging to trace `vt[]` vs `dr[]` vs sampled tokens
2. Verify `hm`/`hm2` pointer state after PLD insertion in verify batch
3. Consider placing PLD candidates AFTER MTP drafts in verify batch
4. Fix rollback test (pre-existing, may need separate fix)
5. Tune PLD hash table size and eviction policy

---

## Artifacts
- Branch: `exp/cascaded-pld-mtp`
- Main baseline: `origin/main` @ 28e9bb2 (worktree `main-parity`)
- Benchmark prompts: `bench/prompt512.txt`, `bench/prompt_code_short.txt`
- JSON logs: captured in this document