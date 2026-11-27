# ssm_alpha / ssm_beta GEMV: why Bonsai 2.5 decoded slower, and the dense kernel

Setup: `nsys profile -t cuda --cuda-graph-trace=node`, engine `--gpu-layers auto --ctx 4096 -f bench/prompt64.txt -n 128 --temp 0 --seed 42 --gen-mtp-steps 0`,
`taskset -c 0-3`, nothing else on the GPU. Kernel times from `nsys stats -r cuda_gpu_kern_sum`
(62-token prompt plus 128 decode steps; 6144 alpha/beta launches = 48 DeltaNet layers x 128 steps).

The only tensor-type difference between the files is `ssm_alpha` / `ssm_beta` (96 tensors, 5120 x 48): BF16 in Bonsai 2, F32 in Bonsai 2.5.

| build | model | all kernels ms | alpha/beta GEMV ms | per launch us | effective read GB/s | decode tok/s |
|---|---|---:|---:|---:|---:|---:|
| before (generic `gemv_fp32_kernel`) | Bonsai 2 (BF16) | 8237.4 | 201.3 | 32.77 | 30 | 17.36 |
| before | Bonsai 2.5 (F32) | 8386.8 | 343.4 | 55.89 | 35 | 17.05 |
| after (`gemv_dense_kernel`) | Bonsai 2 (BF16) | 8094.2 | 66.1 | 10.75 | 91 | 17.68 |
| after | Bonsai 2.5 (F32) | 8135.2 | 90.4 | 14.71 | 134 | 17.60 |

Before: Bonsai 2.5 spent 149.4 ms more in kernels than Bonsai 2 (+1.81 %), and 142.1 ms of that was this one GEMV.
The generic kernel gives one warp a whole row and launches only out / 4 = 12 blocks per matrix (24 for the pair), and its F32
unpack copied bytes one by one, so the 1.97 MB pair was read at about 35 GB/s on a 256 GB/s card.
After: one 256-thread block per output row and token tile, 16-byte (F32) or 8-byte (BF16) vector loads, shuffle + shared-memory reduction.
The gap between the two models drops to 41.0 ms over the run (+0.51 %), which is about what reading 45 MiB more per token costs.

Kernel: `__launch_bounds__(256, 4)`. Checks: `build/gemv_f32_parity` PASS (F32 and BF16, T = 1, 3, 8, 11, 64, rel_rms about 1e-7 against
an FP64 reference, pair launch bit-identical to two single launches), `scripts/sass_gate.sh` PASS, `verify_t17_test`, `attn_long_test`,
`ternary_ab_test`, `q4_0_parity` PASS. MTP stays lossless (same ids as without MTP, ChatML coding and p512, both models):

| model | prompt | no MTP tok/s | MTP 2 tok/s (acceptance) | same ids |
|---|---|---:|---:|---|
| Bonsai 2 + uq4 | coding | 17.74 | 28.15 (90.2 %) | yes |
| Bonsai 2 + uq4 | p512 | 17.48 | 30.95 (97.4 %) | yes |
| Bonsai 2.5 -mtp, built-in head | coding | 17.76 | 29.07 (91.1 %) | yes |
| Bonsai 2.5 -mtp, built-in head | p512 | 17.43 | 28.83 (92.3 %) | yes |

The FP32 summation order changed (per-thread groups of 4, then a fixed tree), so greedy text can differ from the old build after a
near-tie token (raw-text prompt64: first different token at 30 for Bonsai 2 and at 4 for Bonsai 2.5). The tables in REPORT.md were
measured with the old kernel.
