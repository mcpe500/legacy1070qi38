#!/usr/bin/env bash
# Stage 1 baseline with llama-server (llama-bench rejects -ctk iso3, llama-server accepts it).
# One server at a time. Each run: start server, wait for /health, sample VRAM, POST one 512-token
# prompt with 256 decode tokens (temp 0, seed 42, top_k 1), read timings, stop the server.
# Configs: iso3 KV without MTP, iso3 KV with MTP depth 1, iso3 KV with MTP depth 2. All at -ngl 24.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2
OUT="$ROOT/bench/results/stage1"
mkdir -p "$OUT"
B="$HOME/llama.cpp/build-iso3/bin"
M="$HOME/models/27b/Qwen3.8-27B-UD-Q4_K_M.gguf"
PORT=8095
PROMPT_FILE="$ROOT/bench/prompt512.txt"

run_cfg() {  # $1 tag, $2 spec-type, $3 draft-n-max (ignored when spec is none)
  local tag="$1" spec="$2" dmax="$3"
  local extra=()
  if [ "$spec" != "none" ]; then
    extra=(-ctkd iso3 -ctvd iso3 --spec-type "$spec" --spec-draft-n-max "$dmax")
  else
    extra=(--spec-type none)
  fi
  echo "[$(date -u +%T)] start $tag"
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 20; done
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > "$OUT/$tag.vram.csv" 2>/dev/null &
  local SMI=$!
  taskset -c 0,1,2,3 "$B/llama-server" -m "$M" -ngl 24 -c 2048 -np 1 -t 4 \
    --host 127.0.0.1 --port "$PORT" -ctk iso3 -ctv iso3 "${extra[@]}" \
    > "$OUT/$tag.server.log" 2>&1 &
  local SRV=$!
  local ready=0
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi
    kill -0 "$SRV" 2>/dev/null || break
    sleep 10
  done
  if [ "$ready" -ne 1 ]; then
    echo "[$(date -u +%T)] $tag: server not ready"
    kill "$SMI" 2>/dev/null; kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
    return 1
  fi
  python3 - "$PROMPT_FILE" > "$OUT/$tag.request.json" <<'PY'
import json, sys
text = open(sys.argv[1], encoding="utf-8").read()
print(json.dumps({"prompt": text, "n_predict": 256, "temperature": 0, "top_k": 1,
                  "seed": 42, "cache_prompt": False}))
PY
  curl -s --max-time 1500 -H 'Content-Type: application/json' \
    --data-binary @"$OUT/$tag.request.json" "http://127.0.0.1:$PORT/completion" > "$OUT/$tag.response.json"
  kill "$SMI" 2>/dev/null; wait "$SMI" 2>/dev/null
  curl -s "http://127.0.0.1:$PORT/metrics" > "$OUT/$tag.metrics.txt" 2>/dev/null
  kill -TERM "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
  local peak_vram rss
  peak_vram=$(sort -n "$OUT/$tag.vram.csv" | tail -1)
  rss=$(awk '/VmHWM/{print $2" kB"}' "/proc/$SRV/status" 2>/dev/null)
  python3 - "$OUT/$tag.response.json" "$tag" "${peak_vram:-NA}" <<'PY'
import json, sys
path, tag, vram = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    d = json.load(open(path))
except Exception as e:
    print(f"{tag}: bad response ({e})"); sys.exit(0)
t = d.get("timings", {})
print(f"{tag}: pp={t.get('prompt_per_second')} tok/s (n={t.get('prompt_n')}), "
      f"tg={t.get('predicted_per_second')} tok/s (n={t.get('predicted_n')}), "
      f"draft_n={t.get('draft_n')} accepted={t.get('draft_n_accepted')}, peak_vram_MiB={vram}")
PY
  echo "[$(date -u +%T)] done $tag"
}

run_cfg bench-server-kviso3-nomtp none 0
run_cfg bench-server-kviso3-mtp1 draft-mtp 1
run_cfg bench-server-kviso3-mtp2 draft-mtp 2
echo "[$(date -u +%T)] server sweep finished"
