#!/usr/bin/env bash
# Sequential head-to-head benchmark (never both programs at the same time):
#   1. llama.cpp llama-server, iso3 KV (-ctk/-ctv/-ctkd/-ctvd iso3) + --spec-type draft-mtp, --spec-draft-n-max 2, with the layer count
#      raised from 24 until the peak VRAM of the full run (512-token prompt + 256 decode) would pass the 7373 MiB cap; the
#      largest count that stays under the cap is "llama_selected". One extra run without MTP at that count.
#   2. build/qwen38-engine --gpu-layers 32 --kv iso3 --dp4a all --mtp-steps 2 (and one extra run with --mtp-steps 0).
# Prompt: bench/prompt512.txt, 256 generated tokens, temperature 0, seed 42. Output: bench/results/stage5/ and
# bench/results/final_comparison.md.   Env: NGL_LIST="24 25 26"  CAP_MIB=7373  SKIP_LLAMA=1  SKIP_ENGINE=1  EXTRA=0
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 2
cd "$ROOT" || exit 2
OUT="$ROOT/bench/results/stage5"
mkdir -p "$OUT"
B="$HOME/llama.cpp/build-iso3/bin"
M="$HOME/models/27b/Qwen3.8-27B-UD-Q4_K_M.gguf"
PROMPT_FILE="$ROOT/bench/prompt512.txt"
PORT=8096
CAP_MIB="${CAP_MIB:-7373}"
NGL_LIST="${NGL_LIST:-24 25 26}"
EXTRA="${EXTRA:-1}"
SUF="${TAGSUF:-}"        # e.g. TAGSUF=_r2 for a repeat round (tags get the suffix)
N_PREDICT=256

pkill -f llama-server || true
pkill -x qwen38-engine || true
sleep 2
wait_gpu_free() { while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 5; done; }
wait_gpu_free

peak_of() { sort -n "$1" 2>/dev/null | tail -1; }

# run_llama TAG NGL SPEC(draft-mtp|none)  -> writes $OUT/TAG.summary.json; returns 0 when the run finished
run_llama() {
  local tag="$1" ngl="$2" spec="$3" extra=()
  if [ "$spec" != "none" ]; then extra=(-ctkd iso3 -ctvd iso3 --spec-type "$spec" --spec-draft-n-max 2); else extra=(--spec-type none); fi
  echo "[$(date -u +%T)] llama-server $tag (-ngl $ngl, $spec)"
  wait_gpu_free
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > "$OUT/$tag.vram.csv" 2>/dev/null &
  local SMI=$!
  taskset -c 0,1,2,3 "$B/llama-server" -m "$M" -ngl "$ngl" -c 2048 -np 1 -t 4 --host 127.0.0.1 --port "$PORT" \
    -ctk iso3 -ctv iso3 "${extra[@]}" > "$OUT/$tag.server.log" 2>&1 &
  local SRV=$! ready=0
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && { ready=1; break; }
    kill -0 "$SRV" 2>/dev/null || break
    sleep 5
  done
  local rss=""
  if [ "$ready" -eq 1 ]; then
    python3 - "$PROMPT_FILE" "$N_PREDICT" > "$OUT/$tag.request.json" <<'PY'
import json, sys
print(json.dumps({"prompt": open(sys.argv[1], encoding="utf-8").read(), "n_predict": int(sys.argv[2]), "temperature": 0, "top_k": 1,
                  "seed": 42, "cache_prompt": False, "return_tokens": True}))
PY
    curl -s --max-time 1800 -H 'Content-Type: application/json' --data-binary @"$OUT/$tag.request.json" \
      "http://127.0.0.1:$PORT/completion" > "$OUT/$tag.response.json"
    rss=$(awk '/VmHWM/{print $2}' "/proc/$SRV/status" 2>/dev/null)
  else
    echo "  server not ready (see $OUT/$tag.server.log)"
    : > "$OUT/$tag.response.json"
  fi
  kill "$SMI" 2>/dev/null; wait "$SMI" 2>/dev/null
  kill -TERM "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
  pkill -f llama-server || true
  python3 bench/bench_summary.py llama "$tag" "$OUT" "$ngl" "$spec" "${rss:-0}"
  [ "$ready" -eq 1 ]
}

run_engine() {  # TAG MTP_STEPS
  local tag="$1" mtp="$2"
  echo "[$(date -u +%T)] qwen38-engine $tag (--mtp-steps $mtp)"
  pkill -x llama-server || true
  wait_gpu_free
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > "$OUT/$tag.vram.csv" 2>/dev/null &
  local SMI=$!
  taskset -c 0-3 ./build/qwen38-engine -m "$M" -p "$(cat "$PROMPT_FILE")" -n "$N_PREDICT" --temp 0 --seed 42 \
    --gpu-layers 32 --kv iso3 --dp4a all --mtp-steps "$mtp" > "$OUT/$tag.txt" 2> "$OUT/$tag.err"
  kill "$SMI" 2>/dev/null; wait "$SMI" 2>/dev/null
  python3 bench/bench_summary.py engine "$tag" "$OUT" "$mtp"
}

if [ -z "${SKIP_LLAMA:-}" ]; then
  selected=""
  for ngl in $NGL_LIST; do
    run_llama "llama_ngl${ngl}${SUF}" "$ngl" draft-mtp
    peak=$(peak_of "$OUT/llama_ngl${ngl}${SUF}.vram.csv")
    echo "  peak VRAM at -ngl $ngl: ${peak:-NA} MiB (cap $CAP_MIB)"
    if [ -n "${peak:-}" ] && [ "$peak" -le "$CAP_MIB" ] && [ -s "$OUT/llama_ngl${ngl}${SUF}.response.json" ]; then selected="$ngl"; else break; fi
  done
  if [ -z "$selected" ]; then echo "no -ngl in NGL_LIST stays under the cap"; exit 1; fi
  echo "selected -ngl $selected"
  cp "$OUT/llama_ngl${selected}${SUF}.summary.json" "$OUT/llama_selected${SUF}.summary.json"
  python3 - "$OUT/llama_selected${SUF}.summary.json" "llama_selected${SUF}" <<'PY'
import json, sys
p = sys.argv[1]; j = json.load(open(p)); j["tag"] = sys.argv[2]; json.dump(j, open(p, "w"))
PY
  [ "$EXTRA" = "1" ] && run_llama "llama_nomtp_ngl${selected}${SUF}" "$selected" none
fi

if [ -z "${SKIP_ENGINE:-}" ]; then
  run_engine "engine_mtp2${SUF}" 2
  [ "$EXTRA" = "1" ] && run_engine "engine_mtp0${SUF}" 0
fi

python3 bench/bench_summary.py table "$OUT" "$ROOT/bench/results/final_comparison.md"
