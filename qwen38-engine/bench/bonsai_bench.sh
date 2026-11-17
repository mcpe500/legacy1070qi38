#!/usr/bin/env bash
# Ternary Bonsai 2 27B vs Qwen3.8-27B-UD-Q4_K_M on the GTX 1070 Ti, one process at a time (never two GPU programs together).
#   bench/bonsai_bench.sh probe   MODEL BIN TAG KV DEPTH NGL     one llama-bench run: pp512 + tg128 at depth DEPTH, peak VRAM
#   env: UB (ubatch, default 512), REPS (llama-bench repetitions, default 2)
#   bench/bonsai_bench.sh server  MODEL BIN TAG KV CTX NGL PROMPT_TOKENS   llama-server at context CTX (iso3 and others):
#                                 VRAM after load, then one prompt of PROMPT_TOKENS synthetic-text tokens + 128 generated tokens
# Output: bench/results/bonsai/<TAG>.{json,vram.csv,log}; a summary line on stdout.
# Guards: refuses to start while another program uses the GPU; -fit off so -ngl and the context are exactly what is asked.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 2
OUT="$ROOT/bench/results/bonsai"
mkdir -p "$OUT"
CAP=7373
mode="$1"; shift

gpu_free() {
  local n
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
  if [ "$n" != "0" ]; then echo "GPU busy (another process): refusing to run"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; return 1; fi
}

peak() { sort -n "$1" 2>/dev/null | tail -1; }

if [ "$mode" = probe ]; then
  MODEL="$1"; BIN="$2"; TAG="$3"; KV="$4"; DEPTH="$5"; NGL="$6"
  gpu_free || exit 3
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 100 > "$OUT/$TAG.vram.csv" 2>/dev/null &
  SMI=$!
  taskset -c 0-3 "$BIN/llama-bench" -m "$MODEL" -ngl "$NGL" -fa on -ctk "$KV" -ctv "$KV" -p 512 -n 128 -d "$DEPTH" -r "${REPS:-2}" -t 4 -ub "${UB:-512}" -o json \
    > "$OUT/$TAG.json" 2> "$OUT/$TAG.log"
  rc=$?
  kill "$SMI" 2>/dev/null; wait "$SMI" 2>/dev/null
  python3 - "$OUT/$TAG.json" "$TAG" "$(peak "$OUT/$TAG.vram.csv")" "$rc" "$CAP" <<'PY'
import json, sys
p, tag, vram, rc, cap = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
try:
    d = json.load(open(p))
except Exception:
    d = []
r = {x.get("n_prompt", 0) and "pp" or "tg": x for x in d}
pp = r.get("pp", {}).get("avg_ts"); tg = r.get("tg", {}).get("avg_ts")
ok = rc == 0 and pp is not None and tg is not None
v = int(vram) if vram and vram.isdigit() else None
print("%s: rc=%d pp512=%s tg128=%s peak_vram=%s MiB %s" % (tag, rc, "%.2f" % pp if pp else "-", "%.2f" % tg if tg else "-", v,
      "OK" if ok and v is not None and v <= cap else ("OVER CAP" if ok else "FAILED")))
PY
  exit 0
fi

if [ "$mode" = server ]; then
  MODEL="$1"; BIN="$2"; TAG="$3"; KV="$4"; CTX="$5"; NGL="$6"; PT="$7"
  gpu_free || exit 3
  PORT=8097
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 100 > "$OUT/$TAG.vram.csv" 2>/dev/null &
  SMI=$!
  taskset -c 0-3 "$BIN/llama-server" -m "$MODEL" -ngl "$NGL" -c "$CTX" -np 1 -t 4 -fa on -fit off -ctk "$KV" -ctv "$KV" \
    -ub "${UB:-512}" --host 127.0.0.1 --port "$PORT" --no-warmup > "$OUT/$TAG.log" 2>&1 &
  SRV=$!
  ready=0
  for _ in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && { ready=1; break; }
    kill -0 "$SRV" 2>/dev/null || break
    sleep 2
  done
  load_vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  res="{}"
  if [ "$ready" = 1 ]; then
    python3 - "$PT" "$PORT" > "$OUT/$TAG.request.json" <<'PY'
import json, sys, urllib.request
n, port = int(sys.argv[1]), int(sys.argv[2])
# synthetic but varied text (numbered report lines), sized to exactly n tokens with the server's own /tokenize
def line(i):
    return "Report %d: the station logged %d vehicles, %d cyclists and a temperature of %d degrees." % (i, (i * 37) % 911, (i * 13) % 97, (i * 7) % 41)
def count(text):
    req = urllib.request.Request("http://127.0.0.1:%d/tokenize" % port, data=json.dumps({"content": text}).encode(), headers={"Content-Type": "application/json"})
    return len(json.load(urllib.request.urlopen(req))["tokens"])
lines = [line(i) for i in range(1, n // 25 + 2)]
text = "\n".join(lines) + "\nSummary:"
while count(text) > n and len(lines) > 1:
    over = count(text) - n
    lines = lines[:max(1, len(lines) - max(1, over // 30))]
    text = "\n".join(lines) + "\nSummary:"
print(json.dumps({"prompt": text, "n_predict": 128, "temperature": 0, "top_k": 1, "seed": 42, "cache_prompt": False, "ignore_eos": True}))
PY
    curl -s --max-time 3600 -H 'Content-Type: application/json' --data-binary @"$OUT/$TAG.request.json" \
      "http://127.0.0.1:$PORT/completion" > "$OUT/$TAG.json"
  fi
  kill -TERM "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
  kill "$SMI" 2>/dev/null; wait "$SMI" 2>/dev/null
  python3 - "$OUT/$TAG.json" "$TAG" "$(peak "$OUT/$TAG.vram.csv")" "$load_vram" "$ready" "$CAP" <<'PY'
import json, sys
p, tag, vram, lv, ready, cap = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
try:
    t = json.load(open(p)).get("timings", {})
except Exception:
    t = {}
v = int(vram) if vram and vram.isdigit() else None
ok = ready == 1 and t.get("predicted_n")
print("%s: ready=%d load_vram=%s MiB peak_vram=%s MiB prompt_n=%s pp=%s tg=%s %s" % (tag, ready, lv, v, t.get("prompt_n"),
      "%.2f" % t["prompt_per_second"] if t.get("prompt_per_second") else "-", "%.2f" % t["predicted_per_second"] if t.get("predicted_per_second") else "-",
      "OK" if ok and v is not None and v <= cap else ("OVER CAP" if ok else "FAILED")))
PY
  exit 0
fi
echo "usage: $0 probe|server ..."; exit 2
