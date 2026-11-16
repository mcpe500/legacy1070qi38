---
name: iso3-kv-and-mtp
description: "The iso3 3-bit KV cache format (IsoQuant, block of 128 values, 50 bytes) and the MTP speculative decoding path with DeltaNet state rollback. Read before implementing or changing KV storage, KV quantization, attention on quantized K or V, MTP drafting, verification, or any state rollback. The reference is docs/ref_pr38_iso3_mtp.diff."
---

# iso3 KV cache and MTP

## Ground truth
`docs/ref_pr38_iso3_mtp.diff` is the offline copy of mcpe500/legacy1070qi38 PR #38 (merged 2026-11-15).
Its patch files `patches/llama.cpp/0001-*.patch` and `0002-*.patch` contain the iso3 code for
llama.cpp `3549c27`. The fork is mcpe500/llama.cpp, branch feat/iso3-kv, commit 36dbad3.
Use this diff as ground truth for rotation constants, block layout, and scales. Do not rely on
memory or on another project.

## iso3 format (from the reference)
- Block of 128 values, 50 bytes (0.195 x FP16):
  - `norm`: f16, the block L2 norm, divided by the L2 norm of the selected codebook values.
  - `qs`: 32 bytes, 2 low bits per value.
  - `signs`: 16 bytes, 1 high bit per value (3-bit code = high bit << 2 | low 2 bits).
- Encoding: divide the block by its L2 norm. Split it into 32 groups of 4 values. Rotate each group by
  a fixed unit quaternion (left product q x v, tables `iso3_quat_w/x/y/z[32]`). Map each rotated
  value to the nearest of 8 codebook entries (`iso3_codebook[8]`, 3-bit Lloyd-Max, ties go to the
  lower index).
- Decoding: codebook lookup, scale by `norm`, then the inverse rotation per group (conjugate
  quaternion). The CPU decoder works one whole block at a time, because the rotation mixes the 4 values of a group.
- The rotation is orthogonal. A block is rotated within itself, per group, with the same constants
  for every block.

**Conflict with the brief:** the brief asks for an orthogonal or Hadamard rotation. The reference
uses IsoQuant quaternion rotations. This skill follows the reference. Changing it to Hadamard needs
a user decision and new tests, because it changes the format.

## Attention on rotated K and V (derived; verify numerically)
- head_dim is 256, so one K or V head spans two 128-value blocks. Each block has its own norm.
- The rotation acts within groups of 4 and is the same for Q and K. So q . k = (Rq) . (R k).
  Rotate Q once per token and head, and score against the stored rotated K. This avoids inverse-rotating
  every K.
- Values: out = sum_i p_i v_i = R^T (sum_i p_i R v_i). Accumulate in the rotated space and apply R^T
  once per head to the output.
- Accept this only if a random-input test shows the same scores within the float tolerance.

## Measured in PR #38 (keep these as the bar to beat, not as targets)
| Test | Result | Threshold |
|---|---|---|
| Round trip, Gaussian, relative MSE | 0.0348 | 0.05 |
| CUDA SET_ROWS vs CPU reference, bytes | 1600 of 1600 identical | exact |
| FA on dequantized inputs, nmse | <= 2.0e-11 | 1e-9 |
| FA vs f16 original, nmse | 0.063 to 0.073 (quantization error) | informational |
| test-backend-ops CUDA0, iso3 | 28 of 28 | all pass |
| KLD vs f16 KV, 27B Q4_K_M | 0.0102 (same-top-p 95.3%) | informational |

## Known limits in the reference
- No native iso3 vector kernel for flash attention on CUDA. K and V are converted to f16 first.
  The FP32-only rule of pascal-sm61-cuda forbids that path for this engine.
- At ctx 2048, iso3 is not faster (KV is small). Its benefit is for long contexts, which is not
  measured yet.

## MTP drafting and verification
- Drafting: the MTP head (block 64, see qwen38-hybrid-arch) proposes d tokens, d = 1 to 3. The
  brief asks for d = 2 to 3. The engine exposes `mtp_steps` in 0 to 3.
- Verification: the main model runs the last accepted token plus the d drafts in one batch. Accept
  the longest prefix that matches the target. Emit the target token at the first mismatch.
- Rollback on rejection:
  - Attention KV: move the position counter back. Old entries are overwritten later. No copy.
  - DeltaNet state (48 layers): the recurrence is sequential, so the state after each draft position
    is needed. Keep one snapshot per draft position, or recompute from the last accepted state.
    Full f32 snapshots cost 144 MiB each. With d = 3 that is 432 MiB on top of the budget. Decide the
    strategy from the VRAM budget in memory-budget-24gb, not from preference.
- Draft KV type: the MTP block has its own KV. Use the same type as the main model unless measured
  otherwise (PR #38: `-ctkd/-ctvd` must be set explicitly).

## What the measurements say (27B on this card, llama.cpp)
- MTP depth 1 at -ngl 32, f16 KV: 2.17 tok/s vs 2.05 without MTP. Draft 66, accepted 60 (PR #38 table).
- Depth 3: 1.97 tok/s, draft 115, accepted 87. Slower.
- With `-ot` (FFN on CPU), MTP gave no gain at any tested KV type (run-002, C-56).
- Output with MTP is not bit-identical to output without MTP. The first divergence is a near-tie
  (PR #38). Parity tests must compare against the same MTP setting.

**Default for this engine: MTP off.** Turn it on only when the measured accept rate times the
per-step cost beats the no-MTP baseline on the same hardware.

## Checks before merge
- [ ] Round-trip test (random blocks, several scales) passes with the reference threshold.
- [ ] Bytes written by the CUDA path equal the CPU reference.
- [ ] Attention score test: rotated and unrotated paths agree within tolerance.
- [ ] Rollback test: after rejection, the state equals the state of a run that never drafted.
- [ ] Parity against llama.cpp at the same MTP and KV settings (webui-and-parity-bench, stage 1).
