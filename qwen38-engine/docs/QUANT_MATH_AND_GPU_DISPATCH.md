# Quantization math and GPU dispatch (issue #55)

This document covers three things:
- the arithmetic of every weight format the engine runs on the GPU;
- the identities the sm_61 kernels rely on;
- the measured roofline on the GTX 1070 Ti.

The numbers come from `bench/quant_roofline_bench.cu`. Results are in `bench/results/bonsai_local_10h/roofline_before.*` (before the batched Q8_0 kernel) and `roofline_after.*`.

## 1. Common activation side (Q8_1-style)

Each activation row x ∈ R^n is quantized in blocks of 32 values:

    d_b = max_i |x_i| / 127,   q_i = clamp(rint(x_i / d_b), -127, 127),   s_b = d_b * Σ_i q_i

`actq8_quantize`, `rotate_q8_kernel` and `quantize_q8_1_kernel` all produce exactly this.

A weight block w with integer codes c and scale(s) contributes

    Σ_i w_i x_i ≈ d_w * d_b * Σ_i c_i q_i        (+ zero-point / minimum terms, below)

The integer sum is computed with `__dp4a`. IDP.4A does four int8 × int8 products and accumulates them into an int32. The int32 is exact as long as |Σ| < 2^31; here |Σ| ≤ 32 · 127 · 127 per block.

The conversion of the integer sum to float is exact for |v| < 2^22 with the magic constant:

    float(v) = as_float(0x4B400000 + v) - 12582912.0f      (0x4B400000 = 1.5 · 2^23)

It is folded into the dp4a accumulator: `dp4a(c, a, 0x4B400000)`. Both this and I2F give the same float. The I2F variant was measured in M3 and was not faster.

## 2. Ternary formats (Ternary Bonsai, rotated basis)

The weights live in a Hadamard-rotated basis: W = W' · H_1024 · diag(s). Each input therefore goes through the same rotation, applied per 1024-value block with sign vector s:

    x' = (1/32) · H_1024 · (s ⊙ x)

`rotate_q8_kernel` computes this with an in-shared-memory FWHT of 10 butterfly stages. It then quantizes x' to Q8_1 in the same launch, while the values are still in shared memory, which saves one global round trip per GEMV input. It is bit-identical to `rotate` followed by `quantize` (`ternary_ab_test`).

A register-only FWHT is possible: stages 1-5 can use `__shfl_xor` inside a warp and stages 6-10 need 32 values per lane. It was not built, because on sm_61 shuffles share the MIO pipe that already limits the GEMVs, and the fused kernel is not on the critical path (less than 2% of decode time).

### PTQ1_0 (ggml type 143): 128 values per 28 bytes, 1.75 bpw

- **Layout:** 26 payload bytes and an fp16 scale d at byte 26.
- **Positions:**
  - payload bytes 0..23 each hold 5 base-3 digits;
  - bytes 24 and 25 each hold 4 digits (the last digit of the 5-digit code is zero).
- **Value:** w = d · (digit − 1) ∈ d · {−1, 0, 1}.
- **Packing (TQ1_0 style):** a code c ∈ [0, 243) is stored as b = ⌈c · 256 / 243⌉.
- **Decode identity**, checked for all 256 bytes against `kPtq1Code`, no division or modulo:

      digit_i(b) = ((b · 3^i) mod 256) · 3 >> 8,     i = 0..4, most significant first

- **Consequence:** digit_{i+1}(b) = digit_i((3b) mod 256).
- **Kernel use (v3):**
  - The kernel multiplies one 32-bit word holding two bytes in its 16-bit halves by 3^i. That works because b · 81 < 2^16, so the halves do not interact.
  - It masks with `0x00FF00FF` and multiplies by 3 with one ISCADD.
  - One `prmt` with selector `0x7531` takes byte 1 and byte 3 of each half: these hold exactly the digits, since 3 · 255 < 768.
- **Signed codes without a borrow:** c = ((digit | 0x80808080) − 0x01010101) ⊕ 0x80808080.
- **Zero point via the activation sum:** Σ (digit − 1) a = dp4a(digit, a) − dp4a(0x01010101, a). The decode kernel computes the second term once per token and group and shares it across its R rows.
- **Canonical weights:** bytes that are not valid codes (13 values, plus non-multiples of 3 at positions 24/25) are rewritten to 0 at upload. Byte 0 decodes to all-zero digits in every decoder, so all decoders agree. The real file has no such bytes (0 rewritten).

### Q2_0 / "PQ2_0" (type 42): 64 values per 18 bytes, 2.25 bpw

- **Layout:** fp16 d, then 16 bytes of 2-bit codes c ∈ {0, 1, 2}; w = d · (c − 1).
- **Zero point via the activation sum:** Σ (c − 1) q = dp4a(c, q) − Σ q. With the block sum s_b = d_b Σ q:

      Σ w x ≈ d · (d_b · dp4a(c, q) − s_b)

  This is the `- sx[b]` term of the kernel. It is the zero-point factorization the K-quants with a minimum use as well.

## 3. K-quants and IQ formats (Qwen path, MTP heads)

| Format | Block | bpw | Dot product on the GPU |
|---|---|---:|---|
| Q3_K | 256 values, 110 B: hmask 32, qs 64, 12 B 6-bit scales, fp16 d | 3.44 | (low 2 bits ∣ high bit) − 4 → int8, then dp4a; the high bit is merged with LOP3 masks |
| Q4_K | 256 values, 144 B: fp16 d, dmin, 8 × (6-bit scale, 6-bit min), 4-bit q | 4.50 | dp4a(q, x) · d · sc − dmin · m · s_b (the minimum folded into the Q8_1 block sum) |
| Q5_K | Q4_K + 1 high bit | 5.50 | same, 5-bit q |
| Q6_K | 256 values, 210 B: ql 128, qh 64, int8 scales 16, fp16 d | 6.56 | (q − 32) through dp4a(q, x) − 32 · Σ x |
| Q8_0 | 32 values, 34 B: fp16 d, int8 q | 8.50 | dp4a directly |
| IQ4_NL / IQ4_XS | 32 / 256 values, 4-bit indices into a 16-entry non-linear table | 4.50 / 4.25 | table lookup to int8x4, then dp4a |
| IQ3_S | 256 values: 8-bit grid indices + signs | 3.44 | grid lookup to int8x4, signs applied as int8x4 XOR/SUB, then dp4a |
| IQ2_XXS / IQ2_XS / IQ2_S, Q2_K | - | 2.06-2.62 | **no GPU kernel in the engine** (none of the shipped models uses them on the GPU) |

## 4. Measured roofline, sm_61 (ffn_up 5120 × 17408, `roofline_after.txt`)

Measured ceilings:
- DRAM read: 222.4 GB/s (86.9% of the nominal 256 GB/s).
- dp4a: 3.65 T instructions/s, i.e. 14.6 T int8 MAC/s.
- FFMA: 3.53 T/s.

| Format | T=1 GB/s (% of 222) | T=4 | T=16 | T=64 | int8 GMAC/s at T=64 (% of dp4a peak) |
|---|---:|---:|---:|---:|---:|
| PTQ1_0 | 119.7 (53.8%) | 64.7 | 21.9 | 5.5 | 1621 (11.1%) |
| Q2_0 | 105.2 (47.3%) | 46.1 | 13.8 | 3.5 | 791 (5.4%) |
| Q3_K | 66.4 (29.9%) | 52.7 | 24.0 | 6.0 | 898 (6.1%) |
| Q4_K | 195.4 (87.9%) | 138.3 | 38.4 | 9.7 | 1098 (7.5%) |
| Q5_K | 197.7 (88.9%) | 144.9 | 45.8 | 11.5 | 1071 (7.3%) |
| Q6_K | 115.9 (52.1%) | 86.5 | 45.7 | 11.5 | 894 (6.1%) |
| Q8_0 (batched, this branch) | 189.4 (85.2%) | 152.3 | 41.9 | 10.5 | 633 (4.3%) |
| Q8_0 (before) | 174.5 | 44.5 | 11.2 | 2.8 | 169 (1.2%) |
| IQ4_NL | 144.6 (65.0%) | 106.6 | 36.6 | 9.2 | 1045 (7.1%) |
| IQ4_XS | 201.6 (90.6%) | 141.0 | 36.6 | 9.2 | 1108 (7.6%) |
| IQ3_S | 118.7 (53.4%) | 79.4 | 27.3 | 6.8 | 1019 (7.0%) |

Readings:
- **T=1:** the 4-6 bpw K/IQ kernels reach 88-91% of the measured DRAM ceiling. PTQ1_0 reaches 54% even after the v3 decode, because it is bound by issue rate: at 1.75 bpw a byte carries 4.6 weights to decode.
- **T ≥ 16:** every format is compute or MIO bound. It reaches only 4-11% of the int8 MAC peak, because each row-token pays 3-4 non-dp4a instructions (FP32 scale products, int-to-float) next to one IDP.4A. PTQ1_0 has the best batched kernel (`tile2`) and the highest MAC rate.
- **Batched Q8_0:** the Q8_0 kernel was one warp per (row, token). The batched kernel decodes the weights once per 8 tokens: 3.4x faster at T=4 and 3.7x at T=64, bit-identical. It also helps the r3 Q8_0 MTP head: 26.96 tok/s adaptive on the 509-token prompt.
- **Run-to-run variation:** PTQ1_0 at T ≥ 4 differs between bench runs (T=16: 769 us in `roofline_before`, 890 us in `roofline_after`). The kernel is unchanged; `ternary_bench` measures 766 us for the same shape. The cause was not isolated (activation layout of the bench allocation, clocks).

## 5. GPU-specific dispatch

The engine is built for sm_61 only (`CMAKE_CUDA_ARCHITECTURES 61`): the hardware lock in AGENTS.md. Kernel choices are made at run time by T and format, with environment overrides for tuning:
- `QWEN_PTQ1_DEC`: v2 / v3 decode;
- `QWEN_PTQ1_R3`, `QWEN_PTQ1_RV`: rows per warp;
- `QWEN_PTQ1_TILE`: prefill tile;
- `QWEN_Q8_BT`: batched Q8_0;
- `QWEN_ATTN_DEC`: decode attention.

For other architectures, this is the intended `__CUDA_ARCH__` split. It is **not implemented or measured here**: no such GPU is available, and building for them would break the hardware lock.

| Target | What changes |
|---|---|
| sm_61 (this card) | dp4a int8; FP32 only (no HFMA2 per the FP32 rule); 48 KiB static smem per block; MIO-bound decode → arithmetic digit decode |
| sm_75 / sm_86 / sm_89 | `mma.sync` int8 tensor cores (m8n8k16 / m16n8k32) for T ≥ 16 batches; `cp.async` (sm_80+) for the activation tiles; up to 99 / 100 KiB dynamic smem → larger K/V reuse in prefill attention |
| sm_90 | wgmma + TMA for the batched GEMM; thread-block clusters for prefill attention |
| sm_120 | as sm_89/sm_90 plus FP4/FP6 tensor-core formats; a ternary weight could map to int2/fp4 mma operands |

A port would compile one kernel table per `__CUDA_ARCH__`:
- `#if __CUDA_ARCH__ >= 800` for cp.async and mma int8;
- `>= 900` for wgmma;
- the sm_61 path unchanged as the fallback.

The run-time dispatcher (`gemv_ptq1_0_dp4a` and friends) keeps its signature. The bit-identity contract (`ternary_ab_test` against the reference order) cannot hold for tensor-core paths, because mma accumulates in a different order. Those paths need a tolerance-based parity test against FP64 instead.
