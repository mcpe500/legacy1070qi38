#!/usr/bin/env bash
# Runs the whole intelligence suite for one model, sequentially.  bench/bonsai_intel_all.sh q2|ptq1|qwen
# The multiple-choice modes run 4 sequences in parallel (4 DeltaNet states, about +450 MiB), so Q2_0 uses 60 GPU layers there.
# Task sizes: WikiText-2 40 chunks x 512, HellaSwag 200, Winogrande 500, ARC-Challenge 299 (all), MMLU 500 (first 500 of 1531).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 2
S="$ROOT/bench/bonsai_intel.sh"
K="${1:-}"
case "$K" in
  q2)   M=~/models/27b-bonsai/Ternary-Bonsai-2-27B-Q2_0.gguf; B=~/llama.cpp/build-iso3/bin; NGL=99; NGL_MC=60; export UB=128 KV=q4_0 ;;
  ptq1) M=~/models/27b-bonsai/Ternary-Bonsai-2-27B-PTQ1_0.gguf; B=~/llama-prism/llama-prism-b12128-d7b2896; NGL=99; NGL_MC=99; export UB=8 KV=q4_0
        export LD_LIBRARY_PATH=$B CUDA_CACHE_MAXSIZE=4294967296 ;;
  qwen) M=~/models/27b/Qwen3.8-27B-UD-Q4_K_M.gguf; B=~/llama.cpp/build-iso3/bin; NGL=25; NGL_MC=25; export UB=512 KV=q4_0 ;;
  *) echo "usage: $0 q2|ptq1|qwen"; exit 2 ;;
esac
for t in "ppl 40" "hellaswag 200" "winogrande 500" "arc 299" "mmlu 500"; do
  set -- $t
  n=$NGL; [ "$1" != ppl ] && n=$NGL_MC
  [ "$1" = arc ] && [ "$K" = q2 ] && n=59   # ARC runs 5 sequences (one more DeltaNet state)
  "$S" "$M" "$B" "${K}_${1}" "$n" "$1" "$2" | tee -a "$ROOT/bench/results/bonsai/intel/run_lines.txt"
done
