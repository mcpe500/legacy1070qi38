# Experiment log: branch `exp/ternary-bonsai-27b-eval` (2026-11-17)

Every step in order: reason, command, output, decision. `M2=~/models/27b-bonsai/Ternary-Bonsai-2-27B-Q2_0.gguf`, `M1=...-PTQ1_0.gguf`,
`STOCK=~/llama.cpp/build-iso3/bin`, `PRISM=~/llama-prism/llama-prism-b12128-d7b2896`. All commands ran through RTK, on cores 0-3, with the GPU checked free first.

| # | Reason | Command | Output | Decision |
|---|---|---|---|---|
| 1 | Is the model on disk? | `find ~/models -iname "*bonsai*" -o -iname "*ternary*" ...` | nothing | inspect remotely first |
| 2 | Find the official repo | Hugging Face API `models?author=prism-ml` | `prism-ml/Ternary-Bonsai-2-27B-gguf` (Apache-2.0): F16 53.8 GB, PQ2_0 7.21, PTQ1_0 5.95, Q2_0 7.63 GB | |
| 3 | Map the architecture without downloading | `tests/tools/gguf_remote_header.py URL --json header_X.json` (HTTP range requests, 16 MiB read per file) | qwen35 hybrid, 64 blocks, no MTP block, types 142 / 143 / 42, `prism.hadamard` metadata | header_*.txt / .json |
| 4 | Which runtime runs which file | read model card + KNOWN_ISSUES; `grep prism.hadamard` in our llama.cpp; `strings libllama.so` | our stock build already has Q2_0 + the Hadamard runtime; PQ2_0/PTQ1_0 need the PrismML fork | use Q2_0 on stock, fork for PTQ1_0 |
| 5 | Download within the "no tens of GB" limit | `curl -C -` of Q2_0 (7.6 GB), then fork binary (369 MB) and PTQ1_0 (5.9 GB) | 1.4-1.6 MB/s; sha256 of all three verified (`sha256sum -c`) | PQ2_0 and F16 not downloaded |
| 6 | Kernel path in the engine | wrote `src/cuda/ternary_gemv.cu` (lop3 unpack, prmt gather, dp4a; FP32 Hadamard) + `tests/cuda/ternary_test.cu` | 15/15 PASS on the first build | kept |
| 7 | SASS gate | `scripts/sass_gate.sh`, `cuobjdump -sass` | PASS, IDP.4A x464, LOP3 x112, PRMT x976, half x0 | |
| 8 | First speed probe | `bench/bonsai_bench.sh probe $M2 $STOCK q2_f16_d0_ngl99 f16 0 99` | pp 210, tg 17.5, **7791 MiB (over cap)** | reduce GPU memory |
| 9 | Largest -ngl at the default ubatch | probes with -ngl 64 / 63 / 62 / 61 / 60 | 7685 / 7577 / 7471 / 7369 / 7263 MiB; tg 13.3 / 11.1 / 9.4 / 9.0 / 8.5 | -ngl 61 fits, but CPU layers cost about 15 ms each |
| 10 | Where does VRAM go? | server at -c 4096 f16 -ngl 99 | load 7611 MiB = 6921 weights + 256 KV + 150 state + about 284 overhead | shrink the compute buffer |
| 11 | Smaller ubatch | `UB=256` / `UB=128` probe, q4_0, -ngl 99 | 7513 (over) / **7371 MiB, pp 183, tg 17.1** | every layer on the GPU at short context |
| 12 | Depth sweep | `UB=128 probe ... q4_0 4096 64`, `16384 62`, `32768 59` | pp 139 / 89 / 59, tg 12.4 / 7.2 / 3.6, all OK | |
| 13 | KV type at 32K | server -c 32768 -ngl 59 with f16 / q8_0 / q4_0 / iso3 | f16 `cudaMalloc failed: out of memory`; q8_0 7855 (over); q4_0 7343; iso3 7167 | iso3 has the most room |
| 14 | 64K full prompt | server iso3 -c 65536 -ngl 56, prompt 64500 | **failed: request 69447 tokens > 65536** (my estimate of tokens per line was wrong) | size prompts with the server's `/tokenize` |
| 15 | 64K again | same, exact 64,988-token prompt, download paused (SIGSTOP) | pp 49.1, but **1 generated token** (greedy EOS) | add `ignore_eos` |
| 16 | 64K with ignore_eos | same | pp 49.05, **tg 2.04**, 7351 MiB | kept |
| 17 | 128K capacity | server iso3 -c 131072 -ngl 51 / 52, q4_0 -ngl 48 | 7739 / 7847 / 7899 MiB (over): my prediction ignored overhead growth | lower -ngl |
| 18 | 128K again | iso3 -ngl 47 / 46 / 48, q4_0 -ngl 43 | 7219 / 7113 / **7327** OK, q4_0 7227 OK; tg 3.8-4.0 at short depth | full 128K prefill not run (about 1 h) |
| 19 | Real-weight check of the kernel | `dump_ref_tensors` (DUMP_SUFFIXES=-0, -ngl 61), signs from `prism.hadamard.sign_values`, `ternary_real_test` | ffn_up / ffn_gate / ffn_down: **cosine 1.000000 vs llama.cpp** | semantics of rotation and signs confirmed |
| 20 | Qwen baseline, engine | `stage4_run.sh qwen_p4k_{2,adapt} bench/prompt4k.txt 128` (3924-token prompt from docs) | pp 24.2, tg 2.25 (MTP 2) / 2.45 (adaptive) | |
| 21 | Qwen baseline, llama.cpp | `probe Qwen Q4_K_M q4_0` d 0 / 4096 (-ngl 25) / 16384 (-ngl 24) | pp 30.0 / 34.8 / 32.0, tg 2.26 / 2.00 / 1.35 | |
| 22 | Quality sanity | `tests/tools/sanity_prompts.py` on llama-server (Bonsai) and qwen38-engine server | 6/6 and 6/6 | |
| 23 | Fork on Pascal | `cuobjdump --list-ptx libggml-cuda.so`; probe Q2_0 with the fork | sm_61 PTX present; pp 182.5, tg 18.7 (JIT on first start) | fork usable |
| 24 | PTQ1_0 depth sweep, every layer on the GPU | probes f16 d0, q4_0 d 4096 / 16384 / 32768 | pp 99 / 93 / 77 / 66, tg 19.6 / 15.2 / 11.7 / 7.5, max 7165 MiB | |
| 25 | PTQ1_0 capacity | server q4_0 64K -ngl 99 (ub 128/512), -ngl 62/64; 128K -ngl 48; q8_0 32K; f16 16K | 64K every layer 7471/7595 (over: fork adds about 350 MiB at the first request); -ngl 62 7361 OK; 128K 7303 OK | |
| 26 | PTQ1_0 64K full prompt | server -ngl 62, 64,988 tokens | pp 59.1, tg 3.36, 7361 MiB | |
| 27 | Overhead model | `bench/bonsai_overhead.py` over all server runs | overhead 320 MiB at 4K, +6.1 KiB per token (1080 MiB at 128K) | calculator uses it |
