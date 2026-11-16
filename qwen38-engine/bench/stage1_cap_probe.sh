#!/usr/bin/env bash
# Stage 1: find the largest -ngl whose peak VRAM stays under the 7.2 GiB cap (7373 MiB) with the
# iso3 KV configuration. Short probe: load, 8 generated tokens, sample VRAM, stop.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2
OUT="$ROOT/bench/results/stage1"
mkdir -p "$OUT"
B="$HOME/llama.cpp/build-iso3/bin"
M="$HOME/models/27b/Qwen3.8-27B-UD-Q4_K_M.gguf"
CAP=7373
PORT=8096
for ngl in ${NGL_LIST:-23 22}; do
  tag="cap-probe-kviso3-ngl${ngl}"
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 20; done
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > "$OUT/$tag.vram.csv" 2>/dev/null &
  SMI=$!
  taskset -c 0,1,2,3 "$B/llama-server" -m "$M" -ngl "$ngl" -c 2048 -np 1 -t 4 \
    --host 127.0.0.1 --port "$PORT" -ctk iso3 -ctv iso3 -ctkd iso3 -ctvd iso3 \
    --spec-type draft-mtp --spec-draft-n-max 2 > "$OUT/$tag.server.log" 2>&1 &
  SRV=$!
  ready=0
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ready=1; break; }
    kill -0 "$SRV" 2>/dev/null || break
    sleep 10
  done
  if [ "$ready" -eq 1 ]; then
    curl -s --max-time 600 -H 'Content-Type: application/json' \
      -d '{"prompt":"The capital of France is","n_predict":8,"temperature":0,"top_k":1,"seed":42,"cache_prompt":false}' \
      "http://127.0.0.1:$PORT/completion" > /dev/null
  fi
  kill "$SMI" 2>/dev/null; wait "$SMI" 2>/dev/null
  kill -TERM "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
  peak=$(sort -n "$OUT/$tag.vram.csv" | tail -1)
  verdict=$([ "${peak:-99999}" -le "$CAP" ] && echo UNDER_CAP || echo OVER_CAP)
  echo "$tag ready=$ready peak_vram_MiB=${peak:-NA} cap_MiB=$CAP $verdict"
done
echo "[$(date -u +%T)] cap probe finished"
