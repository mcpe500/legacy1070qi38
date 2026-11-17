#!/usr/bin/env bash
# Intelligence evaluation (log-likelihood scoring with llama-perplexity, no sampling), one GPU program at a time.
#   bench/bonsai_intel.sh MODEL BIN TAG NGL TASK N
#   TASK: ppl (WikiText-2 test, ctx 512, N chunks) | hellaswag | winogrande | arc (ARC-Challenge validation) | mmlu (MMLU validation); N = number of tasks
#   TASK kldbase / kld: KL divergence against reference logits in $KLD_BASE (kldbase writes them, kld reads them)
#   env: UB (ubatch, default 512), KV (cache type, default f16), DATA (dataset dir, default ~/datasets), CTX (default 512; 1024 for mmlu/arc if needed)
# Output: bench/results/bonsai/intel/<TAG>.{log,vram.csv}; one summary line on stdout (score, peak VRAM, wall time).
# Same tokenizer (vocabulary 248320) for both models, so perplexities are directly comparable.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 2
OUT="$ROOT/bench/results/bonsai/intel"
mkdir -p "$OUT"
CAP=7373
MODEL="$1"; BIN="$2"; TAG="$3"; NGL="$4"; TASK="$5"; N="$6"
DATA="${DATA:-$HOME/datasets}"
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
if [ "$n" != "0" ]; then echo "GPU busy: refusing to run"; exit 3; fi
case "$TASK" in
  ppl)        args=(-f "$DATA/wikitext-2-raw/wiki.test.raw" --chunks "$N") ;;
  kldbase)    args=(-f "$DATA/wikitext-2-raw/wiki.test.raw" --chunks "$N" --kl-divergence-base "$KLD_BASE") ;;   # writes the reference logits
  kld)        args=(--chunks "$N" --kl-divergence-base "$KLD_BASE" --kl-divergence) ;;                            # compares against them
  hellaswag)  args=(-f "$DATA/hellaswag_val_full.txt" --hellaswag --hellaswag-tasks "$N") ;;
  winogrande) args=(-f "$DATA/winogrande-debiased-eval.csv" --winogrande --winogrande-tasks "$N") ;;
  arc)        args=(-bf "$DATA/arc-challenge-validation.bin" --multiple-choice --multiple-choice-tasks "$N" -np 5) ;;  # some items have 5 choices
  mmlu)       args=(-bf "$DATA/mmlu-validation.bin" --multiple-choice --multiple-choice-tasks "$N") ;;
  *) echo "unknown task $TASK"; exit 2 ;;
esac
CTXV="${CTX:-512}"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 100 > "$OUT/$TAG.vram.csv" 2>/dev/null &
SMI=$!
trap 'kill "$SMI" 2>/dev/null' EXIT   # never leave the VRAM sampler running if this script is killed
t0=$(date +%s)
taskset -c 0-3 "$BIN/llama-perplexity" -m "$MODEL" -ngl "$NGL" -c "$CTXV" -b "$CTXV" -ub "${UB:-512}" -t 4 -fa on \
  -ctk "${KV:-f16}" -ctv "${KV:-f16}" --seed 42 "${args[@]}" > "$OUT/$TAG.log" 2>&1
rc=$?
t1=$(date +%s)
kill "$SMI" 2>/dev/null; wait "$SMI" 2>/dev/null
v=$(sort -n "$OUT/$TAG.vram.csv" | tail -1)
case "$TASK" in
  ppl)        score=$(grep -o "Final estimate: PPL = [0-9.]* +/- [0-9.]*" "$OUT/$TAG.log" | sed 's/Final estimate: //') ;;
  kldbase)    score=$(grep -o "Final estimate: PPL = [0-9.]* +/- [0-9.]*" "$OUT/$TAG.log" | sed 's/Final estimate: //') ;;
  kld)        score=$(grep -E "^Mean +KLD|^Same top p" "$OUT/$TAG.log" | tr -s " " | paste -sd ";") ;;
  hellaswag)  score=$(grep -E "^$N[[:space:]]" "$OUT/$TAG.log" | tail -1 | awk '{print "acc_norm " $2 " CI95 " $3 " " $4}') ;;
  winogrande) score=$(grep -o "Final Winogrande score([0-9]* tasks): [0-9.]* +/- [0-9.]*" "$OUT/$TAG.log") ;;
  *)          score=$(grep -o "Final result: [0-9.]* +/- [0-9.]*" "$OUT/$TAG.log") ;;
esac
st="OK"; [ "$rc" != 0 ] && st="FAILED rc=$rc"; [ -n "$v" ] && [ "$v" -gt "$CAP" ] && st="$st OVER CAP"
echo "$TAG: task=$TASK n=$N ngl=$NGL score=[${score:-none}] peak_vram=$v MiB wall=$((t1 - t0)) s $st"
