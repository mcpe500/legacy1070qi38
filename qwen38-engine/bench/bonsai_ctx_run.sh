#!/usr/bin/env bash
# One engine run with peak-VRAM sampling (nvidia-smi) for the Bonsai long-context sweeps.
# Usage: bench/bonsai_ctx_run.sh <tag> <engine args...>
# Env: OUT_DIR (default bench/results/bonsai128k_v2), SAMPLE_MS (default 50), VRAM_CAP_MIB (default 7373).
# Writes <tag>.txt (stdout), <tag>.err (stderr), <tag>.vram.csv and appends a JSON line with the peak VRAM, the max RSS,
# and cap_ok to <tag>.txt. Refuses to start while another compute process holds the GPU (bench/gpu_isolate.sh check).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT_DIR:-$ROOT/bench/results/bonsai128k_v2}"
CAP="${VRAM_CAP_MIB:-7373}"
mkdir -p "$OUT"
tag="$1"; shift
if ! "$ROOT/bench/gpu_isolate.sh" check > /dev/null; then
    echo "GPU not free, refusing to start $tag: $("$ROOT/bench/gpu_isolate.sh" check)" >&2
    exit 3
fi
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms "${SAMPLE_MS:-50}" > "$OUT/$tag.vram.csv" &
smi=$!
start=$(date +%s)
/usr/bin/time -v taskset -c 0-3 "$ROOT/build/qwen38-engine" "$@" > "$OUT/$tag.txt" 2> "$OUT/$tag.err"
rc=$?
wall=$(( $(date +%s) - start ))
kill "$smi" 2> /dev/null; wait "$smi" 2> /dev/null
peak=$(sort -n "$OUT/$tag.vram.csv" | tail -1)
rss=$(grep "Maximum resident set size" "$OUT/$tag.err" | awk '{print $NF}')
ok=true
[ "${peak:-0}" -gt "$CAP" ] && ok=false
echo "{\"tag\":\"$tag\",\"rc\":$rc,\"wall_s\":$wall,\"peak_vram_mib\":${peak:-0},\"vram_cap_ok\":$ok,\"max_rss_kb\":${rss:-0}}" >> "$OUT/$tag.txt"
[ "$ok" = true ] || echo "WARNING $tag: peak VRAM ${peak} MiB > cap ${CAP} MiB" >&2
exit $rc
