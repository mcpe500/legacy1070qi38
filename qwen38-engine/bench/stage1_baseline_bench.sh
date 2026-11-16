#!/usr/bin/env bash
# Stage 1 baseline: llama-bench sweep, one run at a time (never concurrent with another GPU job).
# Workload: 512-token prompt, 256-token decode, 2 repetitions, 4 threads on physical cores 0-3.
# Records: JSON result, peak VRAM (nvidia-smi sampling), peak RSS and wall time (/usr/bin/time -v).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2
OUT="$ROOT/bench/results/stage1"
mkdir -p "$OUT"
B="$HOME/llama.cpp/build-iso3/bin"
M="$HOME/models/27b/Qwen3.8-27B-UD-Q4_K_M.gguf"

for kv in f16 iso3; do
  for ngl in 24 26 28; do
    tag="bench-kv${kv}-ngl${ngl}"
    echo "[$(date -u +%T)] start $tag"
    while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 20; done
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > "$OUT/$tag.vram.csv" 2>/dev/null &
    SMI=$!
    /usr/bin/time -v taskset -c 0,1,2,3 "$B/llama-bench" -m "$M" -ngl "$ngl" \
      -p 512 -n 256 -r 2 -t 4 -ctk "$kv" -ctv "$kv" -o json \
      > "$OUT/$tag.json" 2> "$OUT/$tag.stderr.txt"
    rc=$?
    kill "$SMI" 2>/dev/null; wait "$SMI" 2>/dev/null
    peak_vram=$(sort -n "$OUT/$tag.vram.csv" | tail -1)
    peak_rss=$(awk -F': ' '/Maximum resident/{print $2}' "$OUT/$tag.stderr.txt")
    wall=$(awk -F': ' '/Elapsed \(wall/{print $2}' "$OUT/$tag.stderr.txt")
    echo "[$(date -u +%T)] done $tag rc=$rc peak_vram_MiB=${peak_vram:-NA} peak_rss_kB=${peak_rss:-NA} wall=${wall:-NA}"
  done
done
echo "[$(date -u +%T)] sweep finished"
