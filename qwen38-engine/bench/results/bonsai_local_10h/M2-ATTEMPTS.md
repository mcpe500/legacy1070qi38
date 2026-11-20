# M2: PTQ1_0 decode kernel attempts (issue #55)

Gate for every attempt: SASS gate (0 HFMA2/HMUL2/HADD2), `ternary_ab_test` 180/180 bit-identical, and higher end-to-end
decode tok/s (509-token prompt, 256 tokens, greedy, `--mtp-steps 0`). Baseline: 13.30 tok/s (PR #53), decode GEMV
`gemv_ptq1_0_kernel_v2<1,4>`, 48 registers, 5 blocks/SM, 86% of GPU time.

| # | Attempt | SASS | ab_test | Kernel (ffn_up 5120x17408, T=1) | Decode tok/s | Kept |
|---|---|---|---|---|---|---|
| 1 | v3 decode: TQ1_0 arithmetic digit decode (`((b * 3^i) mod 256) * 3 >> 8`) instead of 4 shared-memory table loads per lane/row/group; digits 0..2 with `dp4a(digits, a) - dp4a(0x01010101, a)` (activation sum shared by the R rows); weights canonicalized at upload (invalid codes -> byte 0, 0 bytes on the real file) | PASS (0 half ops) | 180/180 | 208.3 -> 164.7 us (R=4, 61 regs, 4 blocks/SM) | see #4 | yes |
| 2 | v3 rows per warp at T=1: R = 2 / 3 / 4 / 6 | PASS | 180/180 each | 226.0 / 192.9 / 165.8 / 196.5 us | - | R = 4 |
| 3 | v3 rows per warp at T = 2..5 (MTP verify): R = 1 / 2 / 3 / 4 | PASS | 180/180 each | ffn_up T=5: 646.6 / 500.8 / 337.1 / 376.3 us (v2: 591.8) | - | R=3 at T=2,5, R=4 at T=3,4 |
| 4 | v3 decode packing: positions (0,1)/(2,3) per word so one 32-bit product decodes both halves, `3x` as ISCADD (inline PTX; ptxas emitted 2 XMADs), one prmt packs the digits without masks. Kernel 722 -> 626 SASS instructions | PASS | 180/180 | 164.7 -> 150.1 us (lm_head 2778 -> 1997 us, T=5 ffn_up 592 -> 313 us) | **17.37** (2 runs: 17.366 / 17.314; same binary with `QWEN_PTQ1_DEC=2`: 13.20), ids identical | yes |
| 5 | tile3 prefill kernel (T > 5): the same arithmetic decode in tile2, codes via `((d | 0x80808080) - 0x01010101) ^ 0x80808080`, `__launch_bounds__(256, 2)` | PASS | 180/180 | T=16 ffn_up 783.7 (tile2) vs 776.0 (tile3 <16,2>), <16,4> 1408.7, <16,3> 782.4 | - | no: no gain, the tile kernels are bound by the activation loads, not the decode |
| 6 | Decode attention v2 (T <= 8, iso3, long-context path): one warp per position slice with the 6 heads of the KV group, prefill lane layout, no barrier in the loop, reduce-scatter scores; the iso3 quaternion rotation moved out of the loop (q' = M^T q once, V accumulated as sum p * nm * c and rotated once at the end). Replaces the split-K kernel (one warp per position, 30 shuffles per position, a barrier per tile, 4x redundant V decode) | PASS | n/a (no GEMV change); `attn_long_test` 52/52 (iso3 decode vs double reference rel 3.4e-6) | attention T=1 at n=131072: 8.80 -> 3.48 ms per layer; T=5: 37.0 -> 15.5 ms | 509-token prompt unchanged (17.33, ids identical); depth 128K 5.09 -> 8.91 | yes |
| 7 | Decode attention v2, V loop unrolled over the 16 tile positions | PASS | - | 3.56 -> 4.60 ms (spills) | - | no |
| 8 | Decode attention v2, `__launch_bounds__(128, 2)` (no spill, 8 warps/SM) / `(128, 4)` (128 registers, 376 B spill) | PASS | - | 4.95 / 7.48 ms vs 3.48 ms with `(128, 3)` | - | no |

## End-to-end decode after M2 (`m2_*`, 509-token prompt `bench/prompt512.txt`, 256 tokens, greedy)

| Run | tok/s | ids |
|---|---:|---|
| PR #53 baseline (`QWEN_PTQ1_DEC=2 QWEN_ATTN_DEC=0` on this build: 13.20 / PR #53: 13.30) | 13.30 | 26a58a94 |
| v3 GEMV, no MTP | **17.37** | 26a58a94 |
| v3 GEMV + r3 Q4_K_M MTP, 2 drafts | 25.88 (167/180 accepted) | 26a58a94 |
| v3 GEMV + r3 Q4_K_M MTP, adaptive | **26.65** (172/179 accepted) | 26a58a94 |
| PR #53 with the injected Qwen head, MTP 2 | 18.48 | 26a58a94 |

## Depth probe (`--depth-bench N -n 33`, zeroed KV, no MTP)

| Depth | PR #53 | v3 GEMV (`m2_depth_*`) | v3 GEMV + decode attention v2 (`m2_depth2_*`) | VRAM MiB |
|---:|---:|---:|---:|---:|
| 512 | 12.74 | 17.26 | 16.85 | 5949 |
| 4096 | 12.68 | 16.35 | 16.19 | 5995 |
| 16384 | 11.02 | 13.52 | 14.75 | 6147 |
| 32768 | 9.29 | 10.97 | 13.48 | 6351 |
| 65536 | 7.06 | 7.92 | 11.52 | 6759 |
| 131072 | 4.72 | 5.09 | **8.91** | 7349 |
