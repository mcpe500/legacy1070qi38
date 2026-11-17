# Cascaded PLD+MTP Experiment Comparison

**Branch**: `exp/cascaded-pld-mtp` (from `main` @ 28e9bb2)  
**Date**: 2026-11-17  
**Hardware**: GTX 1070 Ti (sm_61), Ryzen 5 2400G, 16 GB DDR4  
**Model**: Qwen3.8-27B-UD-Q4_K_M.gguf  
**Config**: `--gpu-layers 16/24 --mtp-steps 1/2 --mtp-adaptive 0/1 --cuda-graph 1 --temp 0 --seed 42`

---

## Summary (Latest: commit a5a63cf)

| Metric | Main (baseline) | Experimental | Δ |
|--------|-----------------|--------------|---|
| **Free-text (prompt512.txt, 509→256, adaptive MTP 2)** | | | |
| decode tok/s | 2.67 | 0.40 | **-85%** |
| MTP acceptance | 96% | ~0% | **-96 pp** |
| **Free-text (16 tok, 16 GPU layers, non-adaptive MTP 1)** | | | |
| decode tok/s | 1.16 | 0.57 | **-51%** |
| MTP acceptance | 100% | 0% | **-100 pp** |
| **Free-text (16 tok, 16 GPU layers, adaptive MTP 2)** | | | |
| decode tok/s | 0.92 | 0.40 | **-57%** |
| MTP acceptance | 75% | 0% | **-75 pp** |
| **Verification Gates** | | | |
| SASS gate (HFMA2/HMUL2) | 0/0 | 0/0 | ✅ PASS |
| Greedy parity (free-text) | ✅ | ✅ | ✅ PASS |
| Greedy parity (code) | ✅ | ❌ diverges | ❌ FAIL |
| Rollback T≤8 bit-identical | ❌ fails | ❌ fails | — (pre-existing) |

---

## Fixes Applied (commit a5a63cf)

### Verify Batch Order Fixed
Changed from `[cur] + PLD + MTP` to `[cur] + MTP + PLD`:
- MTP drafts now occupy contiguous positions 1..De in verify batch
- Preserves `hm`/`hm2` pointer chain for MTP chaining
- PLD candidates appended after MTP drafts

### Acceptance Loop Fixed (matching main branch)
- `cur_new` now comes from position AFTER last accepted draft (`lrows_[k]`)
- Previously incorrectly always used `lrows_[0]`
- Matches main branch logic where accepted drafts advance `cur_new`

### PLD Candidates Appended After MTP
Verify batch structure: `[cur] + dr[0..De-1] + PLD_candidates`
- MTP drafts at fixed positions 1..De
- PLD candidates at positions De+1..De+n_pld
- Acceptance loop handles both MTP and PLD candidates correctly

---

## Detailed Results

### Free-text Benchmark (prompt512.txt, 509 prompt + 256 decode, adaptive MTP 2)

**Main branch**:
```
decode_tok_s: 2.6708
cycles: 60
draft_n: 205, draft_n_accepted: 197 (96%)
accept_hist: [4,4,5,5,42]
```

**Experimental branch** (PLD disabled):
```
decode_tok_s: 0.6259 → 0.40 (latest)
cycles: 243
draft_n: 901, draft_n_accepted: 12 (1.3%) → 0% (latest)
accept_hist: [232,10,1,0,0,0,0,0,0]
```

### 16-token Test (16 GPU layers, non-adaptive MTP 1)

**Main branch**:
```
decode_tok_s: 1.1601
cycles: 8
draft_n: 8, draft_n_accepted: 8 (100%)
accept_hist: [0,8,0,0,0]
```

**Experimental branch**:
```
decode_tok_s: 0.5717
cycles: 15
draft_n: 15, draft_n_accepted: 0 (0%)
accept_hist: [15,0,0,0,0,0,0,0,0]
```

### 16-token Test (16 GPU layers, adaptive MTP 2)

**Main branch**:
```
decode_tok_s: 0.9175
cycles: 6
draft_n: 12, draft_n_accepted: 9 (75%)
accept_hist: [1,3,1,0,1]
```

**Experimental branch**:
```
decode_tok_s: 0.3980
cycles: 15
draft_n: 37, draft_n_accepted: 0 (0%)
accept_hist: [15,0,0,0,0,0,0,0,0]
```

### Code/Repetitive Benchmark (prompt_code_short.txt, 524 prompt + 256 decode)

**Main branch** (adaptive MTP):
```
decode_tok_s: 2.7511
cycles: 58
draft_n: 198, draft_n_accepted: 198 (100%)
```

**Experimental branch**:
```
decode_tok_s: 0.6797
cycles: 243
draft_n: 792, draft_n_accepted: 12 (1.5%)
```
**Token divergence**: Experimental output diverges from main at ~token 60.

### Non-adaptive MTP (prompt512.txt)

**Main branch**:
```
decode_tok_s: 2.5151
cycles: 88, draft_n: 176, accepted: 169 (96%)
```

**Experimental branch**:
```
decode_tok_s: 0.9086
cycles: 242, draft_n: 484, accepted: 13 (2.7%)
```

---

## Implementation Status

### ✅ Completed
1. **T≤8 verify batch**: `kRollT = 8` in `engine.hpp`, `gexec[16]` + `T <= 8` in `gpu_layers.cu`
2. **PLD 3/4-gram matcher**: Fixed open-address hash tables in `generator.cpp` (anonymous namespace)
3. **Cascade logic**: Verify batch = `[cur] + MTP_drafts + PLD_candidates`
4. **Buffer sizing**: `hv_`/`hv_next_` 8×H, `lrows_` 8×V, `bt(9)` `vt(8)` `dr(7)`, `accept_hist[9]`
5. **CLI flags**: `--pld`, `--pld-max`, `--pld-min-len` in `main.cpp`
6. **SASS gate**: 0 HFMA2/HMUL2/HADD2 ✅
7. **Greedy parity (free-text)**: 256/256 tokens identical ✅
8. **Acceptance loop**: `cur_new` from correct position (lrows_[k]) ✅
9. **Verify batch order**: `[cur] + MTP + PLD` ✅

### ❌ Issues Remaining
1. **MTP acceptance near 0%**: Both adaptive and non-adaptive modes reject nearly all drafts
2. **Greedy parity fails on repetitive code**: Tokens diverge after ~60 tokens
3. **Rollback test fails on both branches**: Pre-existing issue with batch vs sequential numerics (not introduced by this experiment)
4. **PLD candidates not accepted**: 792 candidates generated, 0 accepted

### Root Cause Hypothesis
The MTP head produces drafts that never match the main model's greedy predictions, despite:
- Identical MTP drafting logic (copied from main branch)
- Correct verify batch structure and acceptance loop
- Correct `hm`/`hm2` pointer handling (preserved by contiguous MTP positions)
- Greedy parity holding on free-text (main model output identical)

The MTP head's predictions diverge from main model's predictions. Possible causes:
- MTP KV cache state difference (subtle initialization difference)
- Hidden state passed to MTP forward differs (hv/hm/hm2 chain)
- MTP prefill difference (unlikely, same code)
- Memory layout difference from larger buffers (hv_ 8×H vs 5×H)

---

## VRAM / Memory

| Config | VRAM (MiB) | RSS (MiB) |
|--------|-----------|-----------|
| Main (adaptive, 16 layers) | 6070 | 12741 |
| Experimental (adaptive, 16 layers) | 6074 | 12727 |

VRAM increase: ~4 MiB (within budget)

---

## Next Steps (if continuing)
1. Debug MTP head: add logging to trace `dr[]` vs main model's greedy tokens per position
2. Verify MTP KV cache initialization and prefill match main branch exactly
3. Check `hm`/`hm2` pointer state after each MTP forward in the chain
4. Consider memory layout differences from 8-row buffers vs 5-row
5. Fix rollback test (pre-existing, separate fix needed)
6. Tune PLD hash table size and eviction policy

---

## Artifacts
- Branch: `exp/cascaded-pld-mtp` @ a5a63cf
- Main baseline: `origin/main` @ 28e9bb2 (worktree `main-parity`)
- Benchmark prompts: `bench/prompt512.txt`, `bench/prompt_code_short.txt`
- JSON logs: captured in this document