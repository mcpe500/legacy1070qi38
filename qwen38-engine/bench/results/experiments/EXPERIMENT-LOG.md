# Experiment log: branch `exp/bit-lop3-cudagraph-prefill`

Every attempt of this branch, in the order it was made: why it was tried, the exact command or file, the output, and the decision.
Paths are relative to `qwen38-engine/`; `~` is the home directory. `M=~/models/27b/Qwen3.8-27B-UD-Q4_K_M.gguf`.
Common run wrapper: `bench/stage4_run.sh TAG PROMPT_FILE N [engine args]` (writes `bench/results/stage4/TAG.{txt,err}`; prints pp, tg, VRAM, RSS, MTP counters;
`BASELINE=<json>` compares with llama-server tokens, `SAME_AS=<tag>` with an earlier run).
All runs: `taskset -c 0-3`, temperature 0, seed 42, GPU free, no llama-server running. Builds: `ninja -C build -j4`.

## 0. Baseline recorded

| Step | Command | Output | Decision |
|---|---|---|---|
| tidy ignore rules | `qwen38-engine/.gitignore`: build trees, `archive/`, `*.gguf`, `*.bin`, `*.cubin`, reference dumps, raw sampler logs | 292 files staged, none above 1 MB except the vendored cpp-httplib / nlohmann sources | |
| commit | `git commit` (Stage 1-5 summary in the message) | `f90f3d7` | |
| push to main | `git push origin main` | `! [remote rejected] main -> main (protected branch hook declined)` | main is protected; not forced |
| push the same commit to a branch | `git push origin f90f3d7:refs/heads/baseline/stage1-5` | `* [new branch] f90f3d7 -> baseline/stage1-5` | baseline is on GitHub as `baseline/stage1-5`; merging it into main needs a PR |
| experiment branch | `git checkout -b exp/bit-lop3-cudagraph-prefill` | | |

## B. PTX bit tricks and CUDA graphs

**B1. 128-bit loads, PTX lop3.b32, int4 activation loads** (`src/cuda/dp4a_gemv.cu`)
- Why: the K-quant unpacking read weights 32 bits at a time; 128-bit loads cut the number of load instructions by 4, and `lop3.b32` fuses the mask/or of Q5_K.
- Change: `ld128()` / `ld64()` (uint4 / uint2) for Q4_K, Q5_K, IQ4_XS; `lop3_and` (immLut 0xC0) and `lop3_and_or` (0xEA) inline PTX; the 32 activation bytes as two `int4`.
  `prmt.b32` was already used through `__byte_perm` in the IQ4 table lookup (826 PRMT in the SASS).
- Check: `./build/gemv_parity $M` -> `result: PASS (0 failing)`.
- Speed: `bench/stage4_run.sh expB1_p5_r{1,2} bench/prompt5.txt 24 --mtp-steps 0` -> GPU part 43.2 / 41.8 ms/token (Stage 4: 46.3), tg 2.653 / 2.666, 24/24 tokens identical.
- SASS (`cuobjdump -sass`): baseline kernel 0 x `LDG.E.CI.128`, 338 x LOP3, 792 x PRMT; experiment 942 x `LDG.E.CI.128`. The compiler already produced LOP3 for the C++ `&`/`|`.
- Decision: kept (small gain, no numeric change).

**B2. CUDA graph of the 32 GPU layers for batches of 1..5 tokens** (`src/cuda/gpu_layers.cu`, `ops.cu`, `iso3_quant.cu`)
- Why: about 300 kernel launches per token; a graph replays them with one launch.
- Change: all GPU work on one blocking stream; the position is read from device memory (`pos_dev`) by `qk_prep`, `kv_store` and `attention` so the graph can be replayed; one graph per (T, snapshot on/off). `--cuda-graph 0|1`.
- Run: `bench/stage4_run.sh expB2_g{0,1}_mtp0 bench/prompt5.txt 24 --mtp-steps 0 --cuda-graph {0,1}` -> GPU part 43.5 ms (no graph) vs 44.2 ms (graph); both 24/24 identical.
  `SAME_AS=expB2_g0_mtp0 ... expB2_g1_mtp2 ... 48 --mtp-steps 2 --cuda-graph 1` printed `DIFFER(24)`: the script compares whole lists and the run had 48 tokens; the first 24 are identical (checked in Python), and `expB2_g0_mtp2` (24 tokens) is IDENTICAL.
- Decision: kept on (correct, harmless), but it gives no speed: kernel launch overhead is not what limits decode.

## A. Prefill over the PCIe Gen3 x1 link

**A1. CPU-only prefill for layers 32-63 (no PCIe streaming)**, as asked
- Already measured in Stage 4: `bench/stage4_run.sh p512_cpuprefill bench/prompt512.txt 16 --mtp-steps 0 --stream-min-t 0` -> prefill 150.5 s, 3.38 tok/s (`bench/results/stage4/p512_cpuprefill.txt`).
- Why it cannot reach 28.7 tok/s: 32 layers x about 0.42 B weights x 509 tokens x 2 = about 13.7 TOP of int8 work; the CPU path runs `vec_dot` once per token and row.
- Decision: not used. The streamed path was made faster instead (A2).

**A2. Overlapped streaming from pinned page-cache pages**
- Why: profiling (`QWEN_STREAM_DEBUG=1`) showed the streamed half spending its time in `pread` into a pinned bounce buffer and in waiting for the DMA, one after the other.
- Change: `cudaHostRegister(..., cudaHostRegisterReadOnly)` on the 6.96 GiB file range of layers 32-63 (DMA straight from the page cache); a second staging buffer; a non-blocking copy stream uploads matrix k+1 while matrix k is multiplied (events `up_done`/`comp_done`, the GEMV order is precomputed in `build_seq`). `--pin-stream 0` falls back.
- Run: `QWEN_STREAM_DEBUG=1 BASELINE=bench/results/stage2/baseline_p512_tokens.json bench/stage4_run.sh expA1_p512_mtp0 bench/prompt512.txt 16 --mtp-steps 0`
  -> `stream: 6.96 GiB ... registered`, `stream: layers 10.45 s (pread 0.00, dma wait 0.02)`, prefill 18.1 s = 28.18 tok/s (was 27.2 s), 16/16 tokens identical to llama-server.
- Breakdown: `forward T=509: GPU layers 7.33 s, CPU/streamed layers 10.67 s, head 0.05 s`. 7 GiB / 0.84 GB/s = 8.9 s is the floor of the streamed half.
- Decision: kept. Side effect found later (see "Open problems"): the first run after llama-server is cold and slower.

**A3. MTP prefill only on the last prompt tokens** (`--mtp-window N`, attention of the MTP layer starts at `mtp_kv_start_`)
- Why: the MTP prefill ran on the CPU for every prompt token (5 s for 509 tokens).
- Runs (16 tokens): `bench/stage4_run.sh expA2_w{0,1,16,64} bench/prompt512.txt 16 --mtp-steps 2 --mtp-window W` -> prefill 23.1 / 18.1 / 18.3 / 18.7 s, all identical output.
- Runs (256 tokens): `expA3_w{0,1,128}_n256` -> tg 2.621 (accept 96 %) / 1.817 (51 %) / 1.935 (58 %).
- Decision: rejected as a default (the MTP head needs the whole context; total time for 256 tokens grows from 121 s to 159 s). Kept as an opt-in flag, default 0 = full.

**A4. MTP KV prefill on the GPU** (`GpuLayers::mtp_kv`, `Engine::mtp_prefill`)
- Why: only the MTP KV is needed after the prefill, which takes eh_proj, attn_norm, wk, wv, k_norm and RoPE: about 54 MB of weights over PCIe.
- Run: `bench/stage4_run.sh expA4_gpumtp_n256 bench/prompt512.txt 256 --mtp-steps 2` -> prefill 18.3 s (27.86 tok/s), tg 2.622, accept 95 % (168/176), peak VRAM 7335 MiB;
  token ids identical to the CPU-MTP run `expA3_w0_n256` (Python check `True`). `--mtp-gpu-prefill 0` restores the CPU path.
- Decision: kept.

**A5. Larger dp4a token tiles (32, 64) for big batches**
- Why: the 32 GPU layers take 7.3 s at T = 509; the weight unpacking is repeated for every 16-token tile.
- Run: `QWEN_TT_MAX={16,32,64} QWEN_STREAM_DEBUG=1 ./build/qwen38-engine -m $M -p "$(cat bench/prompt512.txt)" -n 2 --mtp-steps 0` -> GPU layers 7.31 / 12.82 / 17.54 s.
- Decision: rejected and the knob removed (register pressure).

**A6. Read-only cache loads and more rows per block**
- Run: `__ldg` on the activation loads with `WARPS` 4 and 8 -> GPU layers 7.32 / 7.30 s.
- Profiling with `ncu` failed: `ERR_NVGPUCTRPERM` (GPU counters need root). `nvprof --print-gpu-summary` worked and is the source of the per-kernel times (IQ4_XS 3.0 s, Q4_K 1.6 s, Q5_K 1.1 s at T = 509).
- Decision: reverted (no effect). A shared-memory tiled kernel would be the next step; not built.

## C. Adaptive MTP and DDR4

**C1. Adaptive MTP** (`--mtp-adaptive 1`, `--mtp-p-keep 0.5`, `--mtp-p-cont 0.85`; server field `mtp_adaptive`)
- Rule: draft step 0 is kept when the MTP head's top-1 probability p >= 0.5; drafting continues while p >= 0.85, up to 4 drafts; zero drafts makes the cycle a plain decode step. Rollback buffers raised to 5 tokens.
- Check: `--rollback-test` -> PASS (bit-identical).
- Runs (48 tokens, `SAME_AS=expC_<p>_mtp0`): 5-token prompt: MTP 0 2.591, MTP 2 2.339 (74 %), adaptive 2.560 (86 %), adaptive 0.7/0.9 2.534 (88 %);
  62-token prompt: MTP 0 2.595, MTP 2 2.792 (100 %), adaptive 2.781 (100 %), adaptive 0.7/0.9 2.767. All IDENTICAL(48).
- 509-token benchmark: `expR_adapt_{1,2}` tg 2.570 / 2.572 (fixed MTP 2: 2.61), accept 96 % (198/207), peak VRAM 7359 MiB, ids identical to the baseline binary.
- Decision: opt-in. It removes most of the loss on hard text but is 2 % slower on easy text and uses 24 MiB more VRAM.

**C2. DDR4: CPU matmul chunking**
- Ceiling: `g++ -O3 -march=znver1 -pthread tests/tools/membw.cpp -o build/membw && taskset -c 0-3 build/membw 2 {1,2,4}` -> 19.6 / 30.5 / 33.1 GB/s (prefetch variant 17.6 / 27.8 / 32.5).
  The 32 CPU layers read 6.96 GiB in 296 ms per token = 25.2 GB/s (76 % of the ceiling).
- Runs: `QWEN_QM_GRAIN=G QWEN_QM_PF=P bench/stage4_run.sh expC2_gG_pfP bench/prompt5.txt 24 --mtp-steps 0`:
  G 8: 297.6 / 298.3 ms; G 32: 288.9; G 128: 286.4 / 287.1; G 256: 289.4; G 512: 309.5 ms per token (CPU layers). All 24/24 identical to llama-server.
- Decision: grain 128 is the new default (`src/cpu/qmat.cpp`).

**C3. Software prefetch of the next weight rows** (`QWEN_QM_PF=1,2`)
- Runs: 395.8 / 384.3 / 385.5 ms per token (grain 8 / 8 / 32).
- Decision: rejected, removed (the hardware prefetcher already streams the rows; extra prefetches only add traffic).

## Benchmark and verification

| Step | Command | Output |
|---|---|---|
| SASS gate | `scripts/sass_gate.sh` (`sass_gate.txt`) | PASS, IDP.4A x1632, half arithmetic x0 |
| head-to-head | `OUTDIR=bench/results/experiments TABLE=bench/results/experiments/compare_raw.md ADAPTIVE=1 bench/compare_llamacpp.sh` (`compare.log`) | llama-server -ngl 25: 27.25 / 2.519; engine runs right after it: 26.26 / 2.024 (MTP 2), 26.97 / 1.953 (MTP 0), 26.69 / 2.582 (adaptive) |
| first launch attempt | started with `pkill -f llama-server` in the same shell | the pattern matched the shell itself and killed it (known pitfall, use `pkill -x`); restarted |
| cold-cache check | `bench/stage4_run.sh expR_warm_mtp{0,2} bench/prompt512.txt 256 --mtp-steps {0,2}` | 28.22 / 2.558 and 27.89 / 2.614, load 42.8 s (cold runs: 73 s) |
| wait loop mistake | `while pgrep -f compare_llamacpp ...` | matched its own command line, looped until the 30-minute timeout; nothing else was affected |
| A/B against the baseline binary | `git worktree add /tmp/main-wt baseline/stage1-5`, build there, `bench/ab_main_vs_exp.sh /tmp/main-wt/qwen38-engine/build/qwen38-engine build/qwen38-engine 2 --mtp-steps 2` (`ab_mtp2.txt`) | baseline 15.90 / 2.578 and 15.84 / 2.557; experiment 27.81 / 2.625 and 27.81 / 2.608; all four runs 256/256 identical |
| regression | `tests/webui_test.py`, `tests/server_test.py`, `--rollback-test`, `gemv_parity`, greedy 24/24 | 46/46, 47/47, PASS, PASS, IDENTICAL |

## Open problems (measured, not fixed)
- **Cold start after llama-server:** with the 7 GiB pin, the first one or two engine runs after a llama-server run load in 73 s instead of 43 s and decode at 1.95-2.02 tok/s instead of 2.6. The cause is not isolated
  (candidates: page placement after the eviction, THP, pinning of freshly read pages). `--pin-stream 0` avoids the pin, at the cost of the A2 speed-up.
- Prompt processing equals llama.cpp (27.8 vs 27.3-28.7) but does not beat it. Remaining time: 8.9 s PCIe floor + 7.3 s of GPU-layer GEMV at T = 509.
- VRAM margin is 38 MiB with MTP 2 (7335 MiB) and 14 MiB with adaptive MTP (7359 MiB).
