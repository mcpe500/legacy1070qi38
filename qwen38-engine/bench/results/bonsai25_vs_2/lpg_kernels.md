# PTQ1_0 lane-per-group (LPG) kernels: decode and prefill

Commits `08109a6` (decode, T <= 5) and `6982f62` (prefill, T > 5) on `exp/bonsai-2.5-vs-2`. GTX 1070 Ti, `taskset -c 0-3`,
temp 0, seed 42, nothing else on the GPU.

## Idea

PTQ1_0 already stores 4 consecutive weights at the same digit position of 4 consecutive payload bytes (byte j digit s ->
weight j + 16 s; stage 2 byte 16 + k -> 80 + k + 8 s; tail bytes 24 / 25 -> 120 + 2 s / 121 + 2 s). One lane therefore decodes
a whole 128-weight group from its 7 words with ISCADD (x3), PRMT (four digits at once: 3 m < 768, so byte 1 / byte 3 of each
16-bit half is the digit) and LOP (mod 256), and feeds `__dp4a` in the natural activation order; the scale conversion and the
FP32 epilogue are paid once per group instead of once per 4 weights (v3: 24.6 warp instructions per 128 weights). No file
repacking, no tables. FP32 / INT32 only, SASS gate PASS.

## Kernel measurements

Decode GEMV (prototype harness, same buffer, alternating timing, best of 5; DRAM read ceiling 222.5 GB/s measured):

| shape | T | v3 (before) | LPG | speedup |
|---|---|---:|---:|---:|
| ffn_up 5120 x 17408 | 1 | 161.3 us, 121 GB/s | 95.8 us, 203 GB/s | 1.68x |
| ffn_up | 4 | 301.6 us | 135.9 us | 2.22x |
| attn_qkv 5120 x 10240 | 1 | 99.5 us | 57.8 us | 1.72x |
| ffn_down 17408 x 5120 | 1 | 176.3 us | 102.3 us | 1.72x |
| lm_head 5120 x 248320 | 1 | 2037 us | 1305 us, 213 GB/s | 1.56x |
| lm_head | 4 | 3919 us | 1703 us | 2.30x |

Prefill GEMV (`build/quant_roofline_bench`, ffn_up, T = 64): tile2 1765–1886 G int8 MAC/s, LPG prefill `<16 tokens, 2 rows per
lane>` 3516 G MAC/s (1.9x). In = 17408 uses `<8, 4>` (scales of 16 tokens do not fit 48 KiB of shared memory): 3370 G MAC/s.

Nsys at ctx 4096 (Bonsai 2.5, 128 decode steps): the ternary GEMV went from 6033 ms (47.1 ms per token) to 3680 ms
(28.8 ms per token, about 198 GB/s for the 5.7 GB of weights).

## Gates

- `build/ptq1_lpg_test`: 57 cases, 0 fail, worst rel_rms 9.9e-8 against an FP64 reference (decode T = 1..5 at in 5120 / 6144 /
  17408 and out 1027 / 4093 / 17408; prefill T = 6, 16, 37, 64). Decode rows are bit-identical across T = 1..5, which keeps
  MTP lossless.
- `ternary_ab_test` 180/180 bit-identical (pinned to v3 and tile2), `ptq1_fast_gate` 42/42, `verify_t17_test` 32/32,
  `attn_long_test` PASS, `q4_0_parity`, `gemv_f32_parity` PASS, SASS gate PASS.
- Greedy text differs from the old build after near-tie tokens (different FP32 order, smaller error against FP64).

## Engine results

| run | before | after | change |
|---|---:|---:|---:|
| Bonsai 2.5, ChatML coding, ctx 4096, no MTP | 17.93 tok/s | 26.65 tok/s | +49 % |
| Bonsai 2.5 -mtp, ChatML coding, MTP 2 | 29.05 tok/s (91.1 %) | 44.13 tok/s (92.0 %) | +52 % |
| Bonsai 2, ChatML coding, no MTP / uq4 MTP 2 | 17.93 / 28.10 | 26.77 / 42.58 | +49 % / +52 % |
| real 4K prompt (3936 tokens), prefill | 69.7 tok/s | 112.6 tok/s | +62 % |
| 32K needle + state round trip (Bonsai 2.5) | prefill 54.1, decode 12.60 | prefill 83.6, decode 18.94 | +55 % / +50 %, PASS |
| **102021-token ChatML request (Bonsai 2.5, no MTP)** | TTFT 2522.4 s, prefill 40.4, decode 9.36 | **TTFT 1862.6 s, prefill 54.8, decode 12.17** | −26 % / +36 % / +30 % |

All MTP rows stayed lossless (ids identical to the no-MTP run of the same build).

## What limits 100K now

Nsys at depth 102400 (decode): `attention_kernel_decode_iso3` 2.77 ms per layer x 16 = 44 ms of about 82 ms per token
(78 MiB of iso3 KV per layer read at about 29 GB/s); the ternary GEMV is 28.9 ms. Tried and reverted (no gain): more decode
slices (200 / 400 / 800: slower), heads split over 2 warps (T = 1 slower), a 16-position V prefetch and a shared-memory
staged double buffer (register spills, 1.3x to 2.5x slower). The attention kernel needs a redesign (fewer instructions per
position and hidden load latency) before 100K decode can move further.
