# Stage 1: reference and baseline

Date: 2026-11-15. Hardware: GTX 1070 Ti (sm_61), Ryzen 5 2400G (CPUs 0-3 = four physical cores), 15905 MiB RAM, 4 GiB swap.
Model: `~/models/27b/Qwen3.8-27B-UD-Q4_K_M.gguf`.
Runtime: llama.cpp `3549c27` + iso3 patch (fork `mcpe500/llama.cpp` `feat/iso3-kv` @ `36dbad3`), built in `~/llama.cpp/build-iso3`.

## Verified inputs
| Item | Value |
|---|---|
| GGUF sha256 | `1bce504dfecd84153666928a41d8a66706f0158d3a317bc5196b72819d7dc529` (matches the LFS object id, CLAIMS C-52 source) |
| GGUF size | 16564259744 bytes |
| Prompt file `bench/prompt512.txt` | sha256 `e12144ad55a84cf46710dab34c164adbeaea9628138911f53841c8f89628649e`, **509 tokens** (not 512, the prefix was cut by bytes) |
| Reference tensors | `tests/ref_tensors/`, 69 files, SHA256SUMS included |

## Caps used (7.2 GiB = 7373 MiB)
Peak VRAM from `nvidia-smi` sampled every 200 ms.

## Baseline A: llama-bench, KV f16 (`-p 512 -n 256 -r 2 -t 4`, taskset 0-3)
Command: `llama-bench -m <GGUF> -ngl N -p 512 -n 256 -r 2 -t 4 -ctk f16 -ctv f16 -o json`

| -ngl | pp512 tok/s | tg256 tok/s | Peak VRAM MiB | Under 7373 MiB cap |
|---:|---:|---:|---:|---|
| 24 | 30.16 +- 6.27 | 2.22 +- 0.00 | 6733 | yes |
| 26 | 23.71 +- 9.12 | 2.18 +- 0.12 | 7927 | **no** |
| 28 | 30.09 +- 1.74 | 2.36 +- 0.00 | 7617 | **no** |

Peak RSS of llama-bench at -ngl 24: 14.8 GB (from `/usr/bin/time -v`, includes mmap'd file pages).

## Baseline B: llama-server with iso3 KV (`-ctk iso3 -ctv iso3`, `-c 2048 -np 1 -t 4`, one 509-token prompt, 256 decode, temp 0, top_k 1, seed 42)
`llama-bench` rejects `-ctk iso3` ("invalid parameter"), so iso3 is measured through `llama-server`.

| Config | -ngl | pp tok/s (n) | tg tok/s (n) | Draft / accepted | Peak VRAM MiB | Under cap |
|---|---:|---|---|---|---:|---|
| iso3, no MTP | 24 | 13.46 (509) | 2.15 (256) | - | 7402 | **no** |
| iso3 + MTP depth 1 | 24 | 10.83 (509) | 2.32 (256) | 131 / 123 (93.9%) | 6965 | yes |
| iso3 + MTP depth 2 | 24 | 27.98 (509) | 2.40 (256) | 179 / 165 (92.2%) | 7015 | yes |

Cap probe (iso3 + MTP depth 2, load plus 8 tokens): `-ngl 23` peak **6749 MiB** (under), `-ngl 22` peak 6549 MiB (under).

**Safe layer count set by this measurement: `-ngl 23`.** `-ngl 24` is at the edge: 6733 MiB (f16), 6965 MiB and 7015 MiB (iso3 with MTP), but 7402 MiB in the iso3 no-MTP server run.

## Findings that change the plan
1. **MTP is not bit-identical to no-MTP at temp 0.** Text diverges at character 98 of 1040. Depth 1 and depth 2 produce identical text to each other. Parity tests must compare against the same MTP setting (confirms PR #38 finding).
2. **MTP gives +8% to +12% decode** here (2.15 to 2.32 and 2.40 tok/s). The earlier best (+6%) was measured without the iso3 KV. Acceptance is about 92% to 94%.
3. **Prompt-eval numbers are not stable** with one request (13 to 28 tok/s for the same prompt). Stage 2 needs repeated runs before any pp number is cited.
4. **Only `-ngl 23` to `24` fits the cap.** With 23 GPU layers, 41 of 64 layers keep their FFN on the CPU. Decode is about 2.2 to 2.4 tok/s, the same range as before.
5. **Reference tensors are GPU-produced** (`-ngl 23`, CUDA numerics). A CPU-only dump was tried first and was killed by the OOM killer (16 GB model in 15.9 GB RAM). A CPU reference for Stage 2 needs another route.

## Reference tensors (layers 0 and 1)
Tool: `tests/tools/dump_ref_tensors` (built by `tests/tools/build_dump.sh` against the baseline build). Prompt "The capital of France is" (5 tokens), temp 0, seed 42, `-ngl 23 -c 2048`.
- 69 tensors written as raw little-endian f32, one file per graph name. Duplicate graph names get `__1` suffixes (e.g. `norm-0__1.bin`).
- All values are finite. Checksums in `tests/ref_tensors/SHA256SUMS`. Manifest: `tests/ref_tensors/manifest.tsv`.
- Covers the DeltaNet path of layer 0 and layer 1 (conv states, q/k/v, gates, `state_predelta-*`, `new_state-*`, FFN), and `model.input_embed`.

## Not done or not measured
- Swap use during the runs was not recorded (the brief asked for `SwapUsed`). Swap was 351 MB before the first sweep and 696 MB before the probe.
- RSS for the llama-server runs was not recorded (the variable was read after the process was stopped).
- Three llama-bench iso3 runs (`bench-kviso3-ngl{24,26,28}`) failed on the `-ctk iso3` argument. Their logs are kept in `bench/results/stage1/` as evidence.
- The llama.cpp iso3 flash-attention path converts KV to FP16 first. The baseline measures it as shipped. The engine must not copy it (decision 2).

## Reproduce
```
bench/stage1_baseline_bench.sh        # llama-bench sweep (f16 and iso3 attempts)
bench/stage1_server_mtp_bench.sh      # llama-server iso3 with and without MTP, -ngl 24
bench/stage1_cap_probe.sh             # NGL_LIST="23 22" by default
tests/tools/build_dump.sh && tests/tools/dump_ref_tensors -m <GGUF> -ngl 23 -c 2048 -t 4 --temp 0 --seed 42 -p "The capital of France is"
```
