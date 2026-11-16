#!/usr/bin/env bash
# One measured run of qwen38-engine. Records pp tok/s, tg tok/s, peak VRAM (nvidia-smi, 200 ms), RSS and swap, then
# compares the generated tokens with a baseline llama-server response when one is given.
#   stage3_hybrid_bench.sh TAG PROMPT_FILE N_PREDICT [GPU_LAYERS=24] [KV=iso3] [BASELINE_JSON]
# Safety: refuses to start while llama-server or another engine runs, and checks that the GPU is free.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 2
cd "$ROOT" || exit 2
TAG="$1"; PF="$2"; N="$3"; GL="${4:-24}"; KV="${5:-iso3}"; BASE="${6:-}"
OUT="$ROOT/bench/results/stage3"
mkdir -p "$OUT"
MODEL="$HOME/models/27b/Qwen3.8-27B-UD-Q4_K_M.gguf"

pkill -x llama-server 2>/dev/null
if pgrep -x qwen38-engine > /dev/null; then echo "another qwen38-engine is running: abort"; exit 1; fi
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then echo "GPU is not free: abort"; exit 1; fi

nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > "$OUT/$TAG.vram.csv" 2>/dev/null &
SMI=$!
vmstat -n 2 > "$OUT/$TAG.vmstat" 2>&1 &
VM=$!
/usr/bin/time -v taskset -c 0-3 ./build/qwen38-engine -m "$MODEL" -p "$(cat "$PF")" -n "$N" --temp 0 --seed 42 \
  --gpu-layers "$GL" --kv "$KV" --dp4a "${DP4A:-none}" --resident-gib 11.0 > "$OUT/$TAG.txt" 2> "$OUT/$TAG.err"
rc=$?
kill "$SMI" "$VM" 2>/dev/null; wait "$SMI" "$VM" 2>/dev/null

python3 - "$OUT/$TAG" "$rc" <<'PY'
import json, sys
tag, rc = sys.argv[1], int(sys.argv[2])
line = [l for l in open(tag + ".txt", encoding="utf-8").read().splitlines() if l.startswith('{"prompt_tokens"')]
if not line:
    print(f"{tag}: no result line, rc={rc}"); sys.exit(1)
d = json.loads(line[-1])
vram = max(int(x) for x in open(tag + ".vram.csv").read().split())
vm = [l.split() for l in open(tag + ".vmstat").read().splitlines()[2:] if l.strip()]
si = sum(int(r[6]) for r in vm); so = sum(int(r[7]) for r in vm)
print(f"{tag}: rc={rc} prompt={d['prompt_tokens']} gen={d['generated_tokens']} "
      f"pp={d['prompt_tokens']/d['prefill_s']:.3f} tok/s (prefill {d['prefill_s']:.1f}s) tg={d['decode_tok_s']:.3f} tok/s "
      f"peak_vram={vram} MiB rss_peak={d['peak_rss_kb']/1024:.0f} MiB rss_anon={d['rss_anon_kb']/1024:.0f} MiB "
      f"gpu={d.get('decode_gpu_s',0):.2f}s cpu={d.get('decode_cpu_layers_s',0):.2f}s head={d.get('decode_head_s',0):.2f}s of decode {d['decode_s']:.2f}s "
      f"swap_in/out={si}/{so} KiB heap_allocs={d['heap_allocs_in_decode_forward']} parity_fail={d['parity_failures']}")
PY
if [ -n "$BASE" ]; then python3 tests/compare_tokens.py "$OUT/$TAG.txt" "$BASE" | tail -3; fi
