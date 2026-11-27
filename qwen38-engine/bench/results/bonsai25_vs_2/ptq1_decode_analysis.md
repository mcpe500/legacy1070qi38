# PTQ1_0 decode GEMV (gemv_ptq1_0_kernel_v3): where the time goes

Context: question whether the ideas of alesha-pro/llm.asm (hand-written x86-64 assembly, CPU only) help this engine on the GPU.
No public SASS assembler supports sm_61, so the GPU part is CUDA C plus inline PTX, as before. This note records what limits the
main decode kernel instead.

## Share of the decode

nsys (`--cuda-graph-trace=node`, Bonsai 2.5, 62-token prompt + 128 decode steps): `gemv_ptq1_0_kernel_v3` = 6033 ms of 8135 ms
of kernel time (74 %), 50928 launches, 118 us average.

## Roofline (build/quant_roofline_bench, ffn_up 5120 x 17408)

Measured ceilings: DRAM read 222.5 GB/s (86.9 % of the nominal 256 GB/s), dp4a 4005 G instr/s.

| format | T | us | GB/s | % of measured DRAM |
|---|---:|---:|---:|---:|
| PTQ1_0 | 1 | 153.8 | 126.8 | 57.0 % |
| Q2_0 | 1 | 235.4 | 106.5 | 47.9 % |
| Q4_K | 1 | 257.2 | 194.9 | 87.6 % |
| Q8_0 | 1 | 499.9 | 189.4 | 85.1 % |

The K-quant kernels reach 85 to 89 % of DRAM, so the memory path of the card is not the limit. PTQ1_0 stops at 57 %: the
kernel is instruction-issue bound by the base-3 decode.

## Instruction count (cuobjdump -sass, v3<TT=1, R=4>, inner loop)

394 instructions per loop iteration = KG 4 groups x R 4 rows = 16 row-groups of 128 weights, about 24.6 warp instructions per
128 weights. Largest classes: XMAD 93, PRMT 44, SHFL 32, LOP32I 30, ISCADD 24, FMUL 16, IDP 15, FFMA 13, LDG 12.
The XMAD count is the two 32-bit multiplies per row-group (`lo * 3^digit`, `hi * 3^digit`), 3 XMADs each on Pascal.
Rewrites tried in a scratch kernel (split halves, `__umul24`, PTX `mul.wide.u16` / `mad.wide.u16`) all compile to 3
instructions per product, so there is no cheap saving there.

ncu (DRAM throughput, stall reasons) was not run: the driver has `RmProfilingAdminOnly: 1` and this account has no sudo.

## Launch tuning (kept)

Paired runs in one session, engine decode, Bonsai 2.5 ChatML coding, 256 tokens, no MTP (tok/s):

| setting | run 1 | run 2 | run 3 |
|---|---:|---:|---:|
| rows per warp R = 2 / 3 / 4 / 6 (T = 1) | 15.86 / 16.59 / 17.77 / 16.65 | | |
| persistent blocks per SM: occupancy default | 17.65 | 17.64 | 17.64 |
| persistent blocks per SM: 8 | 17.87 | 17.87 | 17.86 |
| persistent blocks per SM: 10 | 17.55 | 17.55 | 17.56 |

With MTP (T = 3 verify) 8 blocks per SM was 0.4 % slower (29.02 / 28.96 vs 28.90 / 28.86), so only the T = 1 instances use 8.
Ids are identical (each row is still computed by one warp in the same order). After the change: 17.93 / 17.88 tok/s
(Bonsai 2.5), gates PASS (`ternary_ab_test` 180/180 bit-identical, `ptq1_fast_gate` 42/42, SASS gate).

## What would move it further

The remaining gap (57 % to about 85 % of DRAM) needs fewer instructions per weight in the decode, for example a different
GPU-side packing built at load time (more bytes, cheaper unpack) or a decode that shares one product between several digits.
Both are larger changes with their own parity work; neither is started here.
