#!/usr/bin/env bash
# Second pass of the intelligence suite (after bonsai_intel_all.sh q2): KL divergence against Qwen, ARC fix, Qwen and PTQ1_0 on smaller subsets.
# llama.cpp draws task subsets with mt19937(1) without replacement, so the first N tasks of a larger run are exactly the N-task subset:
# Bonsai's running score at task N (printed in its log) compares directly with a Qwen run of N tasks.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 2
S="$ROOT/bench/bonsai_intel.sh"
RL="$ROOT/bench/results/bonsai/intel/run_lines.txt"
STOCK=~/llama.cpp/build-iso3/bin
PRISM=~/llama-prism/llama-prism-b12128-d7b2896
M2=~/models/27b-bonsai/Ternary-Bonsai-2-27B-Q2_0.gguf
M1=~/models/27b-bonsai/Ternary-Bonsai-2-27B-PTQ1_0.gguf
MQ=~/models/27b/Qwen3.8-27B-UD-Q4_K_M.gguf
export KLD_BASE=~/kiro-tmp/qwen_q4km_wiki40.kld KV=q4_0
q()  { UB=512 "$S" "$MQ" "$STOCK" "qwen_$1" 25 "$1" "$2" | tee -a "$RL"; }
b2() { UB=128 "$S" "$M2" "$STOCK" "q2_$1" "$3" "$1" "$2" | tee -a "$RL"; }
b1() { LD_LIBRARY_PATH=$PRISM CUDA_CACHE_MAXSIZE=4294967296 UB=8 "$S" "$M1" "$PRISM" "ptq1_$1" 99 "$1" "$2" | tee -a "$RL"; }
q kldbase 40          # Qwen reference logits + Qwen PPL on the same 40 chunks
b2 kld 40 99
b1 kld 40
b2 arc 298 59         # 298 < 299 so the subset is drawn randomly (prefix property holds); 5 sequences need -np 5 and one GPU layer less
q hellaswag 100
q winogrande 200
q arc 100
q mmlu 150
b1 hellaswag 100
b1 winogrande 200
