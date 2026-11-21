# M4: Q2_K GPU kernel, real __CUDA_ARCH__ dispatch, and the multi-arch compile check

Branch `exp/bonsai-overnight-v2-prefill-t17-drafter`. Q2_K (ggml type 10) had no GPU kernel in the engine, so
a Q2_K tensor dequantized on the CPU through the FP32 path (`gemv_row_bytes` returned 0, the roofline bench
listed it with the "no GPU kernel" row). This milestone gives it a `__dp4a` kernel and puts a real
`__CUDA_ARCH__` guard plus a compile check under the dispatch.

## Q2_K kernel (`src/cuda/dp4a_gemv.cu`, `Prep<GT_Q2_K>`)

ggml layout (`ggml-common.h block_q2_K`, 256 values in 84 bytes): `scales[16]` at 0 (one byte per 16 values,
low nibble = scale, high nibble = min), `qs[64]` at 16 (four 2-bit codes per byte), then the fp16 `d` at 80
and `dmin` at 82.

A unit is 32 values = one shift `s & 3` over the 32 payload bytes of one 128-value half (`s >> 2`), with the
two 16-value halves taking the two neighbouring scale bytes (`8 * half + 2 * j` and `+ 1`). Three details made
it work:

1. The codes expand with one `lop3_and(qq[g] >> sh, 0x03030303u)` per four values, exactly like Q4_K's
   `0x0F0F0F0F` mask, so the 2-bit tax is one instruction per 4 values, not a table lookup.
2. The payload is read as eight 4-byte words, not `ld128`: a block is 84 bytes, so `blk + 16 + 32 * half` is
   only 4-byte aligned and a `uint4` load faults (the first version of this kernel did).
3. The two halves carry *different* minimums, so the correction term needs each half's activation sum. The
   kernel gets it with a `__dp4a` against `0x01010101` (an exact integer byte sum, no rounding), selected by
   a new `Prep<TYPE>::TWO_MIN` flag. Every other format keeps its single-term correction, so their results are
   unchanged.

`GT_Q2_K` is now in `gemv_dp4a_supported`, in the `gemv_dp4a` dispatch and in `gemv_row_bytes`
(`in / 256 * 84`). The K-quant scale byte is read as `uint8_t`, so the high nibble shifts logically like
libggml's dequantizer (an `int8_t` read sign-extends and produced rel_rms 0.62 until it was fixed).

### Parity (`tests/cuda/q2_k_parity.cu`, new)

Random synthetic Q2_K (random payload, random scale bytes, random small fp16 d/dmin), reference =
`libggml`'s own `dequantize_row_q2_K` dotted in double precision with the same quantized activations, so only
the GPU dequantization and the summation order are measured. `in = 5120`, `out = 1741` (not a multiple of the
4 rows per block), T = 1, 3, 16 and 64:

```
[PASS] Q2_K in 5120 out 1741 T  1: rel_rms vs libggml q2_K dequant 8.314e-08 (bar 3.0e-06)
[PASS] Q2_K in 5120 out 1741 T  3: rel_rms vs libggml q2_K dequant 8.392e-08 (bar 3.0e-06)
[PASS] Q2_K in 5120 out 1741 T 16: rel_rms vs libggml q2_K dequant 8.255e-08 (bar 3.0e-06)
[PASS] Q2_K in 5120 out 1741 T 64: rel_rms vs libggml q2_K dequant 8.161e-08 (bar 3.0e-06)
```

### Roofline (`bench/quant_roofline_bench.cu`, ffn_up shape 5120 x 17408)

| format | bpw | T=1 us / GB/s | T=4 | T=16 | T=64 | GMAC/s T=64 | %dp4a T=64 |
|---|---|---|---|---|---|---|---|
| Q2_K (new) | 2.62 | 194.4 / 150.4 | 363.6 / 80.4 | 1336.3 / 21.9 | 5324.4 / 5.5 | 1071 | 7.3% |
| Q3_K | 3.44 | 578.4 / 66.2 | 729.4 / 52.5 | 1605.8 / 23.8 | 6397.5 / 6.0 | 892 | 6.1% |
| Q4_K | 4.50 | 257.2 / 194.9 | 364.4 / 137.6 | 1313.2 / 38.2 | 5226.5 / 9.6 | 1091 | 7.5% |

The Q2_K kernel reaches 98% of the Q4_K MAC rate (1071 vs 1091 G int8 MAC/s at T=64): the extra
`0x01010101` dots do not cost anything, because the tile kernel is not bound by the dp4a pipe (the M1
measurement). At 2.62 bits per weight it delivers 1.72x the weight bytes per second of Q4_K for the same
work, which is what a 2-bit format is for.

## Dispatch and the multi-arch compile check

1. `src/cuda/ternary_gemv.cu` now has a real `__CUDA_ARCH__` branch: `PTQ1_TILE_MIN_BLOCKS` is 4 on sm_90+
   (64 KiB of registers per SM, where a 256-thread block of the tile kernel fits four blocks instead of the
   two sm_61 fits) and 0 everywhere else, so the active sm_61 build keeps the occupancy the compiler picks
   (measured in M1: forcing three blocks on sm_61 spills and costs 23%).
2. `scripts/arch_compile_check.sh` (new) compiles every `src/cuda/*.cu` to a cubin for sm_61, sm_75, sm_86 and
   sm_89 in a temp directory and reports the failures. It does not link and does not need a GPU.

```
[PASS] sm_61: 7 of 7 kernels compile
[PASS] sm_75: 7 of 7 kernels compile
[PASS] sm_86: 7 of 7 kernels compile
[PASS] sm_89: 7 of 7 kernels compile
arch_compile_check: 28 of 28 (arch, kernel) pairs compile, 0 fail -> PASS
```

The active build is unchanged by the guard: `ternary_ab_test` still 180/180 bit-identical.

## What is NOT done

**IQ2_XXS / IQ2_XS / IQ2_S are still CPU-only.** Root cause: their value is `LUT[4] * sign` with a signed
quant, so every group of 4 values needs a 4-entry table lookup plus a sign flip before the `__dp4a`, and a
dynamic register LUT index on sm_61 spills to local memory. The arithmetic alternative
(`LUT0 + c*(LUT1-LUT0) + ...`) costs about 8 instructions per value against Q2_K's 1 per 4, which the M1
measurement (the tile kernels are issue-bound) says would lose to the CPU dequant. A 3-bit IQ2_S (two masks
plus a high-bit sign byte, `0x03030303` + `0x0f0f0f0f` like Q3_K) is the same work as Q3_K and is the next
candidate, not IQ2.

**Q3_K / Q4_K / Q6_K SIMD / LOP3 experiments** were not re-run: those kernels already unpack with one
`lop3.b32` per 32 values, and the roofline rows above are the "keep only if it wins" baseline. Nothing new
was tried because the M1 finding (the tile kernel is issue-bound, not dequant-bound) removes the case for
another unpack experiment on those formats.

**The dispatch to other architectures is compile-only.** No sm_75+ kernel was measured (there is no such
card here), and `doc/QUANT_MATH_AND_GPU_DISPATCH.md` still documents sm_75-sm_120 as documentation only.