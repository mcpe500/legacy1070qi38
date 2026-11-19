#!/usr/bin/env bash
# One engine run with peak-VRAM sampling (nvidia-smi, 100 ms) for the exp/bonsai-128k-context sweep.
# Usage: bench/bonsai_ctx_run.sh <tag> <engine args...>
# Writes bench/results/bonsai128k/<tag>.txt (stdout), <tag>.err (stderr), <tag>.vram.csv and appends a peak line to <tag>.txt.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/bench/results/bonsai128k"
mkdir -p "$OUT"
tag="$1"; shift
if pgrep -x llama-server > /dev/null; then echo "llama-server is running, refusing to start the engine" >&2; exit 2; fi
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 100 > "$OUT/$tag.vram.csv" &
smi=$!
/usr/bin/time -v taskset -c 0-3 "$ROOT/build/qwen38-engine" "$@" > "$OUT/$tag.txt" 2> "$OUT/$tag.err"
rc=$?
kill "$smi" 2> /dev/null; wait "$smi" 2> /dev/null
peak=$(sort -n "$OUT/$tag.vram.csv" | tail -1)
rss=$(grep "Maximum resident set size" "$OUT/$tag.err" | awk '{print $NF}')
echo "{\"tag\":\"$tag\",\"rc\":$rc,\"peak_vram_mib\":${peak:-0},\"max_rss_kb\":${rss:-0}}" >> "$OUT/$tag.txt"
exit $rc
