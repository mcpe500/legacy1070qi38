# Depth matrix: Bonsai PTQ1_0 decode tok/s at deep contexts, with and without r3 MTP

Branch `exp/bonsai-overnight-v2-prefill-t17-drafter`, 2026-11-22. Measured with the depth probe
(`--depth-bench N -n 33`): the KV cache is allocated for the full context and zeroed, the DeltaNet state is
zero, and 32 single-token steps run at positions N-32 .. N-1. That measures the decode cost at a deep
position without paying a 60-minute prefill first; the attention span is the full context, so the number is
the honest capacity number (the KV content does not change the speed).

Model `Ternary-Bonsai-2-27B-PTQ1_0-mtpqwen.gguf`, 64 GPU layers, `--gpu-layers auto`, `taskset -c 0-3`,
GPU otherwise free (`scripts/gpu_isolate.sh`), greedy, seed 42. Raw output:
`depth/depth_ctx{N}_mtp{0,2}.json`.

| ctx | tanpa MTP | dengan r3 MTP | LM head | catatan |
|---|---|---|---|---|
| 32768 | 14.04 tok/s | 14.04 tok/s | GPU | 6350 / 6938 MiB |
| 65536 | 11.82 tok/s | 11.81 tok/s | GPU | 6758 / 6988 MiB |
| 98304 | 10.15 tok/s | 10.14 tok/s | GPU | 7166 / 7266 MiB; prefill chunk 512 → 256 supaya head muat |
| 114688 | 9.47 tok/s | 9.47 tok/s | GPU | 7370 / 7270 MiB; prefill chunk 512 → 256 |
| 131072 | **8.92 tok/s** | **3.32 tok/s** | MTP → CPU | 7348 / 7338 MiB |

## Two things the table does and does not say

1. The probe drives single-token forwards only, so "dengan r3 MTP" here measures whether the draft head and
   its rollback buffers still fit next to the LM head at that context — not the draft benefit. The draft
   benefit has to be measured on a real generation: on a real 64K prompt PR #56 measured 14.13 tok/s with
   r3 MTP against 11.58 without, and this branch's depth number (11.82) matches that without-MTP baseline.
2. At 131072 the MTP rollback buffers push the LM head off the GPU, and a CPU head costs about 0.2 s per
   token (1 GiB of DDR4 per token), which is the 3.32 tok/s row. This is the documented trade-off of
   `docs/DRAFTER_V2_CONTRACT.md` section 3, now measured on this branch: **at 128K run without MTP**; if the
   draft head is wanted at deep context, cap the context around 112K (the head stays on the GPU there, with
   the prefill chunk lowered to 256 tokens), or ship a smaller head.

## Reference points that did not change

- Prefill of the full 130965-token prompt: about 60.4 minutes (36 tok/s), unchanged by this branch.
- Real-prompt decode at 512 tokens, no MTP: 17.4 tok/s; with r3 MTP: 19-22 tok/s depending on the workload
  (`m3_cascade_speed.txt`).
- Verification batch T=17 attention: 429.5 µs per call at 32K, 865.8 µs at 64K (`verify_t17_test`).

## Follow-up (2026-11-22): the MTP boundary and the new `--verify-t` knob

The engine drops MTP entirely at 131072 (its rollback rows plus the partial buffer push the LM head off the
GPU, and a CPU head costs about 0.2 s per token), so the default deep-context answer is the no-MTP column.

`--verify-t N` (new CLI flag) sizes the buffers manually. At ctx 131072:

| setting | LM head | MTP | decode |
|---|---|---|---|
| default (verify_t 17) | CPU | disabled by the auto policy | 3.33 tok/s (CPU head) / 8.91 without MTP |
| `--verify-t 5` | **GPU** | **enabled** (4 drafts) | **8.92 tok/s** |

So 131072 is not a hard limit for MTP: with a 4-draft batch the head stays on the GPU and the buffers fit
(the prefill chunk is lowered to 64 tokens, which does not change the weight traffic per token). The engine
now also lowers the verification batch automatically in the head-fit decision instead of only in the
zero-layer plan, so the head stays on the GPU without the flag. Where the default (verify_t 17) boundary sits
measured: 114688 works, 131072 does not.

Real-generation throughput with MTP enabled at 131072 still needs a 131K-token prompt (about 30 minutes of
prefill per run) and was not measured; whether the draft acceptance rate holds at that depth is the open
question.

## Where deep-decode time goes, and the one lever left

Per token at 131072 (112 ms measured, 8.92 tok/s), estimated from the kernel measurements:
- GEMV weight reads: 5.66 GB at the PTQ1_0 T=1 effective 131 GB/s (59% of the DRAM ceiling) ≈ 43 ms. The M1
  analysis says the decode kernel is issue-bound there, so little is left.
- iso3 KV attention decode: 16 layers × 3.546 ms ≈ 57 ms. The KV traffic is 105 MB per layer per token, so
  the kernel runs at about 30 GB/s of the 222 GB/s DRAM ceiling — 13%. **This is the single biggest lever at
  128K**: reaching even 60 GB/s would cut about 28 ms per token, or 8.92 → about 11.8 tok/s. The cost is the
  iso3 dequantization (quaternion rotation + codebook) on 50-byte strided blocks; the v2 kernel already moved
  the rotation out of the loop and gained 2.48x, so the next step needs profiling (nsys) of where the
  remaining 3.5 ms per layer goes.

## Follow-up (2026-11-22): real generation at ~95K from a saved state — MTP loses 2x

The depth probe drives single-token forwards only, so the "dengan r3 MTP" column above could not show the draft
cost. A real check: prefill the 94836-token prompt once (`bench/pld_prompt100k.txt`, 99.1 min at 15.9 tok/s — the
prefill chunk is 128 at this context, not the 256 of the 112K probe row), save the state, then decode 64 tokens
twice from the same state (`-p "and" -n 64 --temp 0 --seed 42`, 64 layers GPU, LM head GPU, verify batch 8):

| run | decode | notes |
|---|---|---|
| `--gen-mtp-steps 0` | **10.31 tok/s** | `depth/state100k_mtp0.log` |
| default (2 drafts) | **5.27 tok/s** | `depth/state100k_mtp2.log`, `draft_n_accepted 27 / draft_n 27`, ids identical to the no-MTP run (lossless) |

100 % acceptance and still half the speed: at this context the MTP block sits on the CPU (`MTP block: CPU` in the
plan), so every one of the 13 draft cycles pays a CPU nextn + LM-head pass per draft token, while the GPU verify
batch saves only a fraction of the 43 ms GEMV + 57 ms KV-attention per token. The draft benefit that PR #56 measured
at 64K (14.13 vs 11.58) came with the MTP block still on the GPU; past ~96K the VRAM budget pushes it to the CPU and
the trade flips negative. **At 100K+, run without MTP** (same recommendation as the 128K row, now measured, not
extrapolated). To make MTP win at this depth, the draft head must come back to the GPU (smaller head or smaller
verify batch) or the KV kernel must get faster (the lever below).
