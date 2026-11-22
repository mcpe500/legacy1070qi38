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
