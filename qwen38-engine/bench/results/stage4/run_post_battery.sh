#!/usr/bin/env bash
# Post-optimization benchmark battery (2026-11-16): Points A+B+C active, Point D skipped.
# 3 prompts (p5 = 5 tok, p64 = 64 tok, p512 = 509 tok) x MTP steps {0, 2, 3} + adaptive, engine ctx 4096.
# The first two runs double as verification rows 5-6 in docs/OPT_LOG_2026-11-16.md (ids vs pre baselines).
# Sequential only — stage4_run.sh guards the GPU and refuses to run two engines at once.
set -u
cd $HOME/kiroween-2026/legacy1070qi38/qwen38-engine
pkill -x llama-server 2>/dev/null; sleep 1
unset QWEN_MT_DOT

# --- verification rows (id comparison against pre-change baselines) ---
SAME_AS=pre_p5 ./bench/stage4_run.sh post_p5 bench/prompt5.txt 24
SAME_AS=pre_p512 ./bench/stage4_run.sh post_p512 bench/prompt512.txt 16 --mtp-steps 0

# --- p5 battery (MTP 2 row = post_p5 above) ---
SAME_AS=pre_p5 ./bench/stage4_run.sh post_p5_mtp0 bench/prompt5.txt 24 --mtp-steps 0
./bench/stage4_run.sh post_p5_mtp3 bench/prompt5.txt 24 --mtp-steps 3
./bench/stage4_run.sh post_p5_adapt bench/prompt5.txt 24 --mtp-adaptive 1

# --- p64 battery (no pre baseline for prompt64; perf only) ---
./bench/stage4_run.sh post_p64_mtp0 bench/prompt64.txt 64 --mtp-steps 0
./bench/stage4_run.sh post_p64_mtp2 bench/prompt64.txt 64
./bench/stage4_run.sh post_p64_mtp3 bench/prompt64.txt 64 --mtp-steps 3
./bench/stage4_run.sh post_p64_adapt bench/prompt64.txt 64 --mtp-adaptive 1

# --- p512 battery (MTP 0 row = post_p512 above) ---
./bench/stage4_run.sh post_p512_mtp2 bench/prompt512.txt 16
./bench/stage4_run.sh post_p512_mtp3 bench/prompt512.txt 16 --mtp-steps 3
./bench/stage4_run.sh post_p512_adapt bench/prompt512.txt 16 --mtp-adaptive 1
echo BATTERY_DONE
