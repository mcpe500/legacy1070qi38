#!/usr/bin/env bash
# Baseline capture BEFORE the B/C/A optimization changes (2026-11-16). Sequential, GPU guarded by stage4_run.sh.
set -u
cd $HOME/kiroween-2026/legacy1070qi38/qwen38-engine
MODEL="$HOME/models/27b/Qwen3.8-27B-UD-Q4_K_M.gguf"
pkill -x llama-server 2>/dev/null; sleep 1
./bench/stage4_run.sh pre_p5 bench/prompt5.txt 24 --mtp-steps 0
./bench/stage4_run.sh pre_p62 bench/prompt62.txt 8 --mtp-steps 0
./bench/stage4_run.sh pre_p512 bench/prompt512.txt 16 --mtp-steps 0
taskset -c 0-3 ./build/qwen38-engine -m "$MODEL" --rollback-test > bench/results/stage4/pre_rollback.txt 2> bench/results/stage4/pre_rollback.err
echo "rollback: $(tail -1 bench/results/stage4/pre_rollback.txt)"
taskset -c 0-3 ./build/qwen38-engine -m "$MODEL" -p "The capital of France is" -n 24 --temp 0 --seed 42 --parity-dir tests/ref_full > bench/results/stage4/pre_parity_p5.txt 2> bench/results/stage4/pre_parity_p5.err
grep -E "worst layer|logits cosine|top-1" bench/results/stage4/pre_parity_p5.txt || tail -5 bench/results/stage4/pre_parity_p5.err
echo DONE
