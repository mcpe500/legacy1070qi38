#!/usr/bin/env bash
# Long-context verification queue of exp/bonsai-128k-context, run strictly one after the other (one engine on the GPU).
# Prompts: bench/make_long_prompt.py (ChatML: needle + source listing + question, run with --parse-special), e.g.
#   python3 bench/make_long_prompt.py MODEL.gguf 131008 /tmp/b128/p131008.txt --prefix-out /tmp/b128/pre131008.txt
# and /tmp/b128/question.txt = the question part alone (the continuation for --state-load / --state-test).
# Every run goes through bench/bonsai_ctx_run.sh (peak VRAM sampled at 100 ms, max RSS from /usr/bin/time).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2
M=/home/ivan/models/27b-bonsai
P=/tmp/b128
R="bench/bonsai_ctx_run.sh"
log() { echo "[$(date '+%F %T')] $*"; }
guard() {   # refuse to start while another process holds the GPU
    local n
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
    if [ "$n" -gt 0 ]; then log "GPU busy (another compute process), stopping"; exit 3; fi
}

# 1. decode at full depth (KV zeroed), every context, both packings
for q in PTQ1_0 Q2_0; do
  for c in 4096 32768 65536 131072; do
    guard; log "depth $q $c"; $R ${q}_ctx${c}_depth -m $M/Ternary-Bonsai-2-27B-$q.gguf --gpu-layers auto --ctx $c --depth-bench $c -n 33
  done
done
# 2. needle at 4K, both packings, and the MTP test file (Qwen's blk.64 injected, decode with drafts)
for q in PTQ1_0 Q2_0; do
  guard; log "needle 4k $q"; $R ${q}_ctx4k_needle_chat -m $M/Ternary-Bonsai-2-27B-$q.gguf --gpu-layers auto --ctx 4096 --parse-special -f $P/p4032.txt -n 24
done
guard; log "mtp p512"; $R PTQ1_0mtpqwen_p512_mtp -m $M/Ternary-Bonsai-2-27B-PTQ1_0-mtpqwen.gguf --gpu-layers auto --ctx 4096 -f bench/prompt512.txt -n 256 --mtp-adaptive 0 --mtp-steps 2
guard; log "nomtp p512"; $R PTQ1_0mtpqwen_p512_mtp0 -m $M/Ternary-Bonsai-2-27B-PTQ1_0-mtpqwen.gguf --gpu-layers auto --ctx 4096 -f bench/prompt512.txt -n 256 --mtp-steps 0
# 3. 128K: one prefill of the 130965-token prefix, in-process round trip (save, continue, reset, load, continue: bit-identical),
#    then a fresh process resumes from the file (instant continuation)
guard; log "ptq1 128k state-test"; $R PTQ1_0_ctx131072_statetest -m $M/Ternary-Bonsai-2-27B-PTQ1_0.gguf --gpu-layers auto --ctx 131072 --parse-special \
    -f $P/pre131008.txt --state-test $P/ptq1_128k.state --cont-file $P/question.txt -n 24
guard; log "ptq1 128k load"; $R PTQ1_0_ctx131072_load -m $M/Ternary-Bonsai-2-27B-PTQ1_0.gguf --gpu-layers auto --ctx 131072 --parse-special \
    --state-load $P/ptq1_128k.state -f $P/question.txt -n 24
# 4. 32K full needle prompt, 64K save + load
guard; log "ptq1 32k"; $R PTQ1_0_ctx32768_needle -m $M/Ternary-Bonsai-2-27B-PTQ1_0.gguf --gpu-layers auto --ctx 32768 --parse-special -f $P/p32704.txt -n 24
guard; log "ptq1 64k save"; $R PTQ1_0_ctx65536_save -m $M/Ternary-Bonsai-2-27B-PTQ1_0.gguf --gpu-layers auto --ctx 65536 --parse-special \
    -f $P/pre65472.txt --state-save $P/ptq1_64k.state
guard; log "ptq1 64k load"; $R PTQ1_0_ctx65536_load -m $M/Ternary-Bonsai-2-27B-PTQ1_0.gguf --gpu-layers auto --ctx 65536 --parse-special \
    --state-load $P/ptq1_64k.state -f $P/question.txt -n 24
log "done"
