#!/usr/bin/env bash
# Verification queue of exp/bonsai-128k-prefill-opt-v2 (issue #52), strictly sequential, one engine on the GPU.
# Usage: bench/bonsai128k_v2_queue.sh STAGE...   stages: vram depth needle state64 state128load qwen
# Prompts in /tmp/b128 (bench/make_long_prompt.py, ChatML, --parse-special): p4032/p32704/p65472/p131008, pre*, question.txt.
# Every run goes through bench/bonsai_ctx_run.sh (refuses to start on a busy GPU, 50 ms VRAM sampling, cap check).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2
M=/home/ivan/models/27b-bonsai
Q=/home/ivan/models/27b/Qwen3.8-27B-UD-Q4_K_M.gguf
P=/tmp/b128
R="bench/bonsai_ctx_run.sh"
OUT="${OUT_DIR:-$ROOT/bench/results/bonsai128k_v2}"
log() { echo "[$(date '+%F %T')] $*"; }
run() { log "run $1"; "$R" "$@"; local rc=$?; log "done $1 rc=$rc $(tail -1 "$OUT/$1.txt" 2>/dev/null)"; return $rc; }

for stage in "$@"; do
  case "$stage" in
    vram)   # Q2_0 64K transient: same command as the #51 run that peaked at 7417 MiB, three times
      for i in 1 2 3; do run Q2_0_ctx65536_vram_r$i -m $M/Ternary-Bonsai-2-27B-Q2_0.gguf --gpu-layers auto --ctx 65536 --depth-bench 65536 -n 33; done ;;
    depth)
      for q in PTQ1_0 Q2_0; do for c in 4096 32768 65536 131072; do
        run ${q}_ctx${c}_depth -m $M/Ternary-Bonsai-2-27B-$q.gguf --gpu-layers auto --ctx $c --depth-bench $c -n 33
      done; done ;;
    needle)
      run PTQ1_0_ctx4k_needle -m $M/Ternary-Bonsai-2-27B-PTQ1_0.gguf --gpu-layers auto --ctx 4096 --parse-special -f $P/p4032.txt -n 24
      run Q2_0_ctx4k_needle -m $M/Ternary-Bonsai-2-27B-Q2_0.gguf --gpu-layers auto --ctx 4096 --parse-special -f $P/p4032.txt -n 24
      run Q2_0_ctx32768_needle -m $M/Ternary-Bonsai-2-27B-Q2_0.gguf --gpu-layers auto --ctx 32768 --parse-special -f $P/p32704.txt -n 24 ;;
    state64)
      run PTQ1_0_ctx65536_statetest -m $M/Ternary-Bonsai-2-27B-PTQ1_0.gguf --gpu-layers auto --ctx 65536 --parse-special \
          -f $P/pre65472.txt --state-test $P/ptq1_64k.state --cont-file $P/question.txt -n 24
      run PTQ1_0_ctx65536_load -m $M/Ternary-Bonsai-2-27B-PTQ1_0.gguf --gpu-layers auto --ctx 65536 --parse-special \
          --state-load $P/ptq1_64k.state -f $P/question.txt -n 24 ;;
    state128load)
      run PTQ1_0_ctx131072_load -m $M/Ternary-Bonsai-2-27B-PTQ1_0.gguf --gpu-layers auto --ctx 131072 --parse-special \
          --state-load $P/ptq1_128k.state -f $P/question.txt -n 24 ;;
    qwen)
      run qwen_q4km_p5_mtp0 -m $Q --gpu-layers 32 --ctx 4096 --mtp-steps 0 -n 48
      QWEN_MT_DOT=0 run qwen_rollback_test -m $Q --gpu-layers 32 --ctx 4096 --rollback-test ;;
    *) log "unknown stage $stage" ;;
  esac
done
log "queue finished: $*"
