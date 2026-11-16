#!/usr/bin/env bash
# One measured run of qwen38-engine with the Stage 4 defaults (hybrid 32 GPU layers, iso3 KV, dp4a all, prefill streaming, MTP 2).
#   stage4_run.sh TAG PROMPT_FILE N_PREDICT [engine args...]      e.g.  stage4_run.sh p5_mtp0 bench/prompt5.txt 48 --mtp-steps 0
# Env: BASELINE=<baseline_tokens.json>  compare the generated tokens with a llama-server baseline
#      SAME_AS=<tag>                    compare the generated token ids with an earlier run of this script (e.g. the MTP 0 run)
# Records pp tok/s, tg tok/s, peak VRAM (nvidia-smi, 200 ms), RSS, swap and the MTP counters.
# Safety: refuses to start while llama-server or another engine runs, and checks that the GPU is free.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 2
cd "$ROOT" || exit 2
TAG="$1"; PF="$2"; N="$3"; shift 3
OUT="$ROOT/bench/results/stage4"
mkdir -p "$OUT"
MODEL="$HOME/models/27b/Qwen3.8-27B-UD-Q4_K_M.gguf"

pkill -x llama-server 2>/dev/null
if pgrep -x qwen38-engine > /dev/null; then echo "another qwen38-engine is running: abort"; exit 1; fi
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then echo "GPU is not free: abort"; exit 1; fi

nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > "$OUT/$TAG.vram.csv" 2>/dev/null &
SMI=$!
vmstat -n 2 > "$OUT/$TAG.vmstat" 2>&1 &
VM=$!
taskset -c 0-3 ./build/qwen38-engine -m "$MODEL" -p "$(cat "$PF")" -n "$N" --temp 0 --seed 42 "$@" \
  > "$OUT/$TAG.txt" 2> "$OUT/$TAG.err"
rc=$?
kill "$SMI" "$VM" 2>/dev/null; wait "$SMI" "$VM" 2>/dev/null

python3 - "$OUT/$TAG" "$rc" "${SAME_AS:+$OUT/$SAME_AS}" <<'PY'
import json, sys
tag, rc, same = sys.argv[1], int(sys.argv[2]), (sys.argv[3] if len(sys.argv) > 3 else "")
line = [l for l in open(tag + ".txt", encoding="utf-8").read().splitlines() if l.startswith('{"prompt_tokens"')]
if not line:
    print(f"{tag}: no result line, rc={rc}"); sys.exit(1)
d = json.loads(line[-1])
vram = max(int(x) for x in open(tag + ".vram.csv").read().split())
vm = [l.split() for l in open(tag + ".vmstat").read().splitlines()[2:] if l.strip()]
si = sum(int(r[6]) for r in vm); so = sum(int(r[7]) for r in vm)
acc = (100.0 * d["draft_n_accepted"] / d["draft_n"]) if d["draft_n"] else 0.0
msg = (f"{tag}: rc={rc} prompt={d['prompt_tokens']} gen={d['generated_tokens']} mtp={d['mtp_steps']} "
       f"pp={d['prompt_tokens']/d['prefill_s']:.2f} tok/s (prefill {d['prefill_s']:.1f}s) tg={d['decode_tok_s']:.3f} tok/s "
       f"peak_vram={vram} MiB rss_peak={d['peak_rss_kb']/1024:.0f} MiB rss_anon={d['rss_anon_kb']/1024:.0f} MiB "
       f"swap_in/out={si}/{so} KiB heap_allocs={d['heap_allocs_in_decode_forward']} "
       f"draft_n={d['draft_n']} draft_n_accepted={d['draft_n_accepted']} ({acc:.0f}%) cycles={d['cycles']} accept_hist={d['accept_hist']} "
       f"draft/verify/rollback={d['mtp_draft_s']:.2f}/{d['mtp_verify_s']:.2f}/{d['mtp_rollback_s']:.2f}s")
if same:
    ref = [l for l in open(same + ".txt", encoding="utf-8").read().splitlines() if l.startswith('{"prompt_tokens"')]
    r = json.loads(ref[-1])
    n = min(len(r["ids"]), len(d["ids"]))
    msg += f" ids_vs_{same.split('/')[-1]}={'IDENTICAL' if r['ids'][:n] == d['ids'][:n] and len(r['ids']) == len(d['ids']) else 'DIFFER'}({n})"
print(msg)
PY
if [ -n "${BASELINE:-}" ]; then python3 tests/compare_tokens.py "$OUT/$TAG.txt" "$BASELINE" | tail -3; fi
