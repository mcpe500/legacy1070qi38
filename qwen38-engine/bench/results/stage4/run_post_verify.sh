#!/usr/bin/env bash
# Verification matrix AFTER the B/C/A optimization changes (2026-11-16), sequential, GPU guarded.
# Part 1 (QWEN_MT_DOT=0): B and C must be bit-identical to the pre-change baselines.
# Part 2 (mt_dot on, default): Point A parity bar (cosine >= 0.995, KLD, greedy tokens).
# Note: SAME_AS is an environment variable of stage4_run.sh (token-id comparison), not an engine
# argument. The first draft passed it as a positional arg and the engine aborted with
# "unknown argument: SAME_AS=..." (rc=2) — fixed here, kept as the record of that mistake.
set -u
cd $HOME/kiroween-2026/legacy1070qi38/qwen38-engine
MODEL="$HOME/models/27b/Qwen3.8-27B-UD-Q4_K_M.gguf"
OUT=bench/results/stage4
pkill -x llama-server 2>/dev/null; sleep 1
export QWEN_MT_DOT=0

# --- Part 1: B/C bit-identity with the multi-token kernel disabled ---
taskset -c 0-3 ./build/qwen38-engine -m "$MODEL" --rollback-test > "$OUT/post_rollback_mtdot0.txt" 2> "$OUT/post_rollback_mtdot0.err"
echo "rollback(mtdot0): $(tail -1 $OUT/post_rollback_mtdot0.txt)"

taskset -c 0-3 ./build/qwen38-engine -m "$MODEL" -p "The capital of France is" -n 24 --temp 0 --seed 42 --parity-dir tests/ref_full \
  > "$OUT/post_parity_p5_mtdot0.txt" 2> "$OUT/post_parity_p5_mtdot0.err"
grep -E "worst layer|logits cosine|top-1|logits KLD" "$OUT/post_parity_p5_mtdot0.txt" | head -5

SAME_AS=pre_p5 ./bench/stage4_run.sh postB_p5_mtdot0 bench/prompt5.txt 24 --mtp-steps 0

# --- Part 2: Point A with the multi-token kernel active ---
unset QWEN_MT_DOT

taskset -c 0-3 ./build/qwen38-engine -m "$MODEL" -p "The capital of France is" -n 24 --temp 0 --seed 42 --parity-dir tests/ref_full \
  > "$OUT/post_parity_p5_A.txt" 2> "$OUT/post_parity_p5_A.err"
grep -E "worst layer|logits cosine|top-1|logits KLD" "$OUT/post_parity_p5_A.txt" | head -5

SAME_AS=pre_p5 ./bench/stage4_run.sh post_p5 bench/prompt5.txt 24
SAME_AS=pre_p512 ./bench/stage4_run.sh post_p512 bench/prompt512.txt 16 --mtp-steps 0
echo DONE
