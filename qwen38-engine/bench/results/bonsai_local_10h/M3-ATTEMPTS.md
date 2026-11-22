# M3: 128K prefill attempts (issue #55)

Baseline (PR #53): 130965-token prefill 3774.7 s (62.9 min) = chunked GEMM about 2195 s + prefill attention about 1580 s
(nsys). The chunk is 64 tokens at 128K (lowered so the GPU LM head fits). Kernel timing below: `attn_long_test 131072 time`
(T = 64 query rows at n = 131072, iso3 KV) and `ternary_bench` (ffn_up 5120x17408, T = 16). Gate: `attn_long_test` 52/52,
SASS gate, `ternary_ab_test` 180/180 for the GEMM.

| # | Attempt | Result | Kept |
|---|---|---|---|
| 1 | tile3 GEMM: arithmetic digit decode in the T > 5 tile kernel (`<16,2>`, `<16,3>`, `<16,4>` with 128-register cap, `<8,4>`) | 783.7 us (tile2) vs 776.0 / 782.4 / 1408.7 us; bit-identical. The tile kernels are bound by the shared-memory activation loads and the 4 FP32 ops per row-token, not by the decode | no |
| 2 | Attention PV loop fully unrolled over the 16 tile positions (masked positions contribute exactly 0) | 96.9 -> 213.5 ms (231 registers, worse scheduling) | no |
| 3 | Attention PV loop `#pragma unroll 2` / `4` | 98.5 -> 97.2 / 117.1 ms | no |
| 4 | Attention with 3 or 2 heads per warp (`QWEN_ATTN_HPW`, 16 / 24 warps) | 108.5 / 137.1 ms | no (already the PR #53 choice) |
| 5 | Attention: raw iso3 bytes of the next K/V tile prefetched into registers during the dequant and compute of the current tile (one block per SM exposed the global-load latency of every tile between two barriers) | 98.5 -> 85.4 ms (-13%), same bytes so the same result | yes |
| 6 | Attention with 4 query tokens per block (128 threads, 246 registers, two blocks per SM so one block's barriers overlap the other's compute) | 98.5 -> 235.4 ms | no |
| 7 | tile2 GEMM: integer-to-float conversion of the dot with I2F (all tokens / odd tokens) instead of the magic-constant FADD, to move work off the FP32 pipe (bit-identical, 180/180) | T=64 ffn_up 3090.8 -> 3060.9 / 3092.9 us (within 1%) | no: the tile kernel is not bound by the FP32 issue rate |
| 8 | tile2 TT/R variants at T=64 (`QWEN_PTQ1_TILE` 1/5/6/2) | 3100.3 (default <16,2>) / 6066.9 / 4256.6 / 4624.9 us | default kept |

## Full run (`m3_PTQ1_0_ctx131072_statetest`, 130965 tokens, frozen build `qwen38-engine-m3`)

| Item | PR #53 | this branch |
|---|---:|---:|
| Prefill | 3774.68 s (62.9 min, 34.70 tok/s) | **3625.43 s (60.4 min, 36.12 tok/s)** |
| Needle `MAGENTA-FALCON-42` (in-memory / from file) | found / found | found / found |
| State round trip (1748.3 MiB) | PASS, save 4.46 s, load 2.75 s | PASS (tokens identical, first logits row bit-identical), save 4.49 s, load 2.73 s |
| Decode after 128K (real context) | 4.54 tok/s | **8.66 tok/s** (M2 decode attention v2) |
| Peak VRAM / max RSS | 7349 MiB / 674 MB | 7349 MiB / 675 MB |

The < 50 min target is missed by 10.4 min. What is left after these attempts:
- GEMM (about 2195 s): the bit-identical formulation (`acc += d * dx * float(dot)` per lane and group, the reference
  order) costs 4 issue slots per row-token, and the activation reads from shared memory cost about as much on the MIO pipe.
  Decode-side savings (tables, conversions) do not move it; more rows per warp run out of registers (2 blocks/SM need
  <= 128).
- Attention (about 1430 s after the prefetch): 229 registers per thread allow one block of 8 warps per SM, so barriers
  and dequantization are exposed. Double-buffering the dequantized tile needs 64 KiB of shared memory at 16 positions per
  tile (48 KiB limit per block).

## After the full run

| # | Attempt | Result | Kept |
|---|---|---|---|
| 9 | Attention double-buffered: 8-position tiles, two FP32 K/V tile buffers (2 x 16 KiB), one barrier per tile, every warp dequantizes its share of tile i+1 (byte loads from the cache) before computing tile i, so dequant and compute of different warps overlap | 83.5 -> 110.1 ms (T=64, n=131072, with slices): the per-tile softmax and reduce-scatter overhead doubles at 8 positions | no |
| 10 | tile2 GEMM with more rows per warp sharing each activation load: codes of R rows decoded first, tokens in slices of TH so fewer activation registers are live, `__launch_bounds__(256, 2)`. `<16,4,TH=8>` / `<16,4,TH=4>` / `<16,3,TH=8>` / `<32,2,TH=8>` | T=64 ffn_up 3090.8 (default) vs 4612.3 / 3795.1 / 3062.7 / 6028.3 us; the 128-register cap spills 40-248 B; `<32,2>` also gives 30/180 wrong results (the dx staging only covers 16 tokens). The restructured loop also pushed the default `<16,2>` from 127 to 146 registers (one block per SM, 762 -> 1064 us) | no, file reverted |

## Re-run on the final build (`m3_final_PTQ1_0_ctx131072_statetest`)

Prefill 3636.85 s (60.6 min, 36.01 tok/s); needle `MAGENTA-FALCON-42` in memory and from file; state round trip PASS
(save 4.56 s, load 2.77 s); decode after 128K 8.73 tok/s; peak 7349 MiB, RSS 674 MB. The < 50 min target stays missed.
