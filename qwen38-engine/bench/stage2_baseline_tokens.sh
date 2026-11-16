#!/usr/bin/env bash
# Greedy reference tokens from the baseline llama-server (llama.cpp 3549c27 + iso3 patch), same GGUF and prompt as
# the engine run. One server at a time: the caller must make sure qwen38-engine is not running.
# KV f16, no MTP, -ngl 23 (the verified VRAM-safe split), temperature 0, top_k 1, seed 42.
# Writes bench/results/stage2/baseline_tokens.json with the generated token ids and text.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 2
cd "$ROOT" || exit 2
OUT="$ROOT/bench/results/stage2"
mkdir -p "$OUT"
B="$HOME/llama.cpp/build-iso3/bin"
M="$HOME/models/27b/Qwen3.8-27B-UD-Q4_K_M.gguf"
PORT=8097
N="${N_PREDICT:-24}"
TAG="${TAG:-}"
PROMPT="${PROMPT:-The capital of France is}"

pkill -x llama-server 2>/dev/null
pkill -x qwen38-engine 2>/dev/null
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 5; done

taskset -c 0-3 "$B/llama-server" -m "$M" -ngl 23 -c 2048 -np 1 -t 4 --host 127.0.0.1 --port "$PORT" \
  --spec-type none > "$OUT/baseline${TAG}.server.log" 2>&1 &
SRV=$!
ready=0
for _ in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && { ready=1; break; }
  kill -0 "$SRV" 2>/dev/null || break
  sleep 5
done
if [ "$ready" -ne 1 ]; then echo "server not ready"; kill "$SRV" 2>/dev/null; exit 1; fi

python3 - "$PROMPT" "$N" > "$OUT/baseline${TAG}.request.json" <<'PY'
import json, sys
print(json.dumps({"prompt": sys.argv[1], "n_predict": int(sys.argv[2]), "temperature": 0, "top_k": 1,
                  "seed": 42, "cache_prompt": False, "return_tokens": True}))
PY
curl -s --max-time 900 -H 'Content-Type: application/json' --data-binary @"$OUT/baseline${TAG}.request.json" \
  "http://127.0.0.1:$PORT/completion" > "$OUT/baseline${TAG}_tokens.json"
kill -TERM "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
python3 - "$OUT/baseline${TAG}_tokens.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("baseline tokens:", d.get("tokens"))
print("baseline text  :", repr(d.get("content")))
PY
echo "gpu after: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
