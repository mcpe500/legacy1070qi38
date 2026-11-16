#!/usr/bin/env bash
# A/B run of two engine binaries on the benchmark prompt (509 tokens + 256 decode, temp 0, seed 42), alternating A, B, A, B so that
# both see the same page cache state. Each run is a full process (load, prefill, decode). Writes OUTDIR/ab_<label>_<i>.{txt,err,vram.csv}
# and prints one summary line per run plus a token-identity check of every run against the first A run.
#   bench/ab_main_vs_exp.sh MAIN_BIN EXP_BIN [ROUNDS=2] [extra engine args...]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 2
cd "$ROOT" || exit 2
A="$1"; B="$2"; R="${3:-2}"; shift 3 2>/dev/null || shift $#
OUT="${OUTDIR:-$ROOT/bench/results/experiments}"
mkdir -p "$OUT"
M="$HOME/models/27b/Qwen3.8-27B-UD-Q4_K_M.gguf"
pkill -x llama-server 2>/dev/null
for i in $(seq 1 "$R"); do
  for lab in main exp; do
    bin="$A"; [ "$lab" = exp ] && bin="$B"
    while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 3; done
    tag="ab_${lab}_$i"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > "$OUT/$tag.vram.csv" 2>/dev/null &
    SMI=$!
    taskset -c 0-3 "$bin" -m "$M" -p "$(cat bench/prompt512.txt)" -n 256 --temp 0 --seed 42 --gpu-layers 32 --kv iso3 --dp4a all "$@" \
      > "$OUT/$tag.txt" 2> "$OUT/$tag.err"
    kill "$SMI" 2>/dev/null; wait "$SMI" 2>/dev/null
    python3 - "$OUT" "$tag" <<'PY'
import json, sys, os
out, tag = sys.argv[1], sys.argv[2]
def load(t):
    l = [x for x in open(os.path.join(out, t + ".txt"), encoding="utf-8").read().splitlines() if x.startswith('{"prompt_tokens"')]
    return json.loads(l[-1]) if l else None
d = load(tag)
if not d:
    print(tag, "no result"); sys.exit(0)
v = [int(x) for x in open(os.path.join(out, tag + ".vram.csv")).read().split() if x.strip().isdigit()]
ref = load("ab_main_1")
same = "n/a" if not ref else ("IDENTICAL" if ref["ids"] == d["ids"] else "DIFFER")
acc = 100.0 * d["draft_n_accepted"] / d["draft_n"] if d["draft_n"] else 0
print(f"{tag}: pp {d['prompt_tokens']/d['prefill_s']:.2f} tok/s (prefill {d['prefill_s']:.1f} s) tg {d['decode_tok_s']:.3f} tok/s "
      f"MTP {d['draft_n_accepted']}/{d['draft_n']} ({acc:.0f}%) peak VRAM {max(v) if v else 'n/a'} MiB peak RSS {d['peak_rss_kb']/1024:.0f} MiB "
      f"load {d['load_s']:.1f} s ids vs ab_main_1: {same}")
PY
  done
done
