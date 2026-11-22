# M1: fast prefill 128K (branch exp/bonsai-overnight-v2-prefill-t17-drafter)

Goal of the marathon: prefill of 131072 positions under 45 minutes. This milestone tried the PTQ1_0 prefill
GEMM first, because it is the largest single block of the 128K prefill (about 2195 s of the 3625 s total
measured on exp/bonsai-local-10h-opt-serving, see bench/results/bonsai_local_10h/M3-ATTEMPTS.md).

Baseline carried into this branch (both unchanged here): the bit-identical tile2 prefill GEMM
(`gemv_ptq1_0_dp4a`, `QWEN_PTQ1_TILE` default 1 = `<16,2>`) and the attention KV-reuse kernel with slices
(`tests/cuda/attn_long_test`, n = 131072: reuse 98.39 ms, reuse + slices 83.22 ms, old tiled 2180.68 ms).

## What was built

1. `gemv_ptq1_0_dp4a(..., bool fast)` in `src/cuda/ternary_gemv.{cu,cuh}`: fast = true runs
   `launch_ptq1_tile2<TT, R, true>` with two independent FP32 group accumulators (`acc`, `acc2`; even k
   groups in `acc`, odd ones in `acc2`), merged before the warp reduce. Same lane mapping, same integer
   dots, only the FP32 summation order is relaxed. The tile is picked by `QWEN_PTQ1_FAST`
   (1 `<16,2>`, 2 `<16,1>`, 3 `<32,1>`, 4 `<32,2>`, 5 `<8,2>`, 6 `<16,4>`).
   fast never reaches the decode kernels: `T <= 5` returns before the prefill dispatch.
2. `tests/cuda/ptq1_fast_gate.cu` (new): the numeric gate. Random canonical PTQ1_0 weights (code words
   from the empirical `kPtq1Code` table, tail bytes restricted to multiples of 3) over the three Bonsai
   widths 5120 / 6144 / 17408, both output counts, T = 6 .. 64, against a host FP64 reference built from
   the *same* quantized activations, so only the summation order is measured.
3. `bench/quant_roofline_bench.cu --fast`: a paired A/B section. The default and the fast kernel are timed
   alternately in one process (7 rounds, best of each side), because the clock of this card drifts several
   percent between processes and an A/B split across runs proves nothing. It also prints the occupancy of
   the prefill tiles (`ptq1_prefill_occupancy`).

## Measured (GTX 1070 Ti, sm_61, 19 SMs, taskset -c 0-3, GPU otherwise free)

### Numeric gate: PASS

```
ptq1_fast_gate: 42 cases, 42 pass, 0 fail | worst fast cos 1.00000000 | worst default cos 1.00000000
               | worst ref-kernel cos 1.00000000 | worst abs 3.815e-06 -> PASS
```

cosine against FP64 is 1.00000000 for both paths at every T, so the relaxed order is numerically
equivalent at the kernel level, and the greedy token (argmax over the output rows) is identical between
fast and default in all 42 cases. The default path stays bit-identical: `ternary_ab_test` 180/180.

### Speed: the fast path is a loss, so it is not wired into the engine

Paired A/B on the ffn_up shape 5120 x 17408, same process, best of 7 rounds:

| QWEN_PTQ1_FAST | tile | T=16 default / fast | T=64 default / fast | speedup T=16 | T=64 |
|---|---|---|---|---|---|
| 1 | `<16,2>` | 788.3 / 1113.9 us | 3081.4 / 4349.5 us | 0.708x | 0.708x |
| 2 | `<16,1>` | 1245.0 / 1264.2 us | 4908.0 / 4907.0 us | 0.985x | 1.000x |
| 3 | `<32,1>` | 3032.9 / 3031.9 us | 6152.6 / 6153.6 us | 1.000x | 1.000x |
| 4 | `<32,2>` | 2602.2 / 2601.0 us | 5343.8 / 5344.7 us | 1.000x | 1.000x |
| 5 | `<8,2>`  | 1015.0 / 1049.2 us | 4090.3 / 4101.9 us | 0.967x | 0.997x |
| 6 | `<16,4>` | 996.6 / 1047.1 us | 3937.9 / 3939.1 us | 0.952x | 1.000x |

No configuration reaches parity. T=1 and T=4 are 1.000x by construction (fast is never used at T <= 5).

Why: the tile2 kernel is issue-bound, not FP32-latency-bound.
- occupancy of `<16,2>`: 2 blocks of 256 threads per SM (127 registers, 27136 B of shared memory). 127
  registers x 256 threads = 32512 of the 65536 registers per SM, so exactly two blocks fit.
- Forcing 3 blocks with `__launch_bounds__(PW * 32, 3)` drops the kernel to 80 registers (local-memory
  spills) and makes T=16 946.6 us instead of 788.3 us: 23% slower. The fast variant collapses to 6661 us.
- `<32,1>` only fits 1 block per SM. `<16,4>` and `<8,4>` also fit 2.
- Per k step the thread issues 32 `IDP.4A` and about 96 FP32 instructions for 256 MACs (about 1.03 issue
  slots per MAC); the measured 1850-1870 G int8 MAC/s is about 47% of the 4096 G lane-slots/s issue
  ceiling of 19 SMs x 4 sub-partitions at 1.683 GHz. This agrees with attempt 7 of
  `bench/results/bonsai_local_10h/M3-ATTEMPTS.md`: "the tile kernel is not bound by the FP32 issue rate"
  was wrong about the pipe but right about the outcome - replacing the magic-constant FADD with I2F moves
  nothing.

Engine level, 509-token prefill of `bench/prompt512.txt` (PTQ1_0 model, `--mtp-steps 0`, `--temp 0`,
`--seed 42`, two runs each): default 7.16 / 7.18 s and ids identical, fast 9.50 / 9.54 s and ids identical
between the fast runs but diverge from the default at generated token 16
("Simon Stephens" becomes "John Tiffany"). The engine prefill is 0.76x, and the greedy output changes.

## Decision

The engine keeps the bit-identical default path (`src/cuda/gpu_layers.cu` calls `gemv_ptq1_0_dp4a` without
fast). The `--fast-prefill` CLI flag and the `fast_prefill` config plumbing were removed: the milestone
gate required greedy identity, and at the engine level the path fails it and is slower.
`fast = true`, its tile knob and its gate test stay in the tree as the measurement record.

## Consequence for the 45-minute target

Not reached in this milestone. The 128K prefill stays at the previous branch's number, about 3625 s
(60.4 min), and the GEMM part of it (about 2195 s) is at the floor of this formulation on this card after
11 documented attempts (10 in `bench/results/bonsai_local_10h/M3-ATTEMPTS.md`, this one here).
Remaining levers for later milestones, in order of expected value:
1. Prompt-lookup decoding (the `--pld` family already exists in `src/engine/generator.cpp`): cut the
   decoded token count, which does not touch the GEMM at all.
2. MTP r3 acceptance at long context (decode after 128K is 8.7 tok/s today).
3. Attention double buffering, blocked by the 48 KiB shared-memory limit per block at 16 positions per
   tile (`M3-ATTEMPTS.md` attempt 9).