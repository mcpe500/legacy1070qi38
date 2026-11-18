#!/usr/bin/env bash
# Long-context verification queue of exp/bonsai-128k-context, run strictly one after the other (one engine on the GPU).
# Prompts come from bench/make_long_prompt.py (needle + source listing + question, ChatML, --parse-special).
# Every run goes through bench/bonsai_ctx_run.sh (peak VRAM sampled at 100 ms, max RSS from /usr/bin/time).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2
M=/home/ivan/models/27b-bonsai
P=/tmp/b128
R="bench/bonsai_ctx_run.sh"
log() { echo "[$(date '+%F %T')] $*"; }

# Q2_0 4K: round trip again (pinned bounce buffer) + cross-process save / load
log "q2 4k round trip"; $R state_roundtrip_Q2_0_4k -m $M/Ternary-Bonsai-2-27B-Q2_0.gguf --gpu-layers auto --ctx 4096 --parse-special \
    -f $P/pre4032.txt --state-test $P/st_q2_4k.bin --cont-file $P/question.txt -n 16

# PTQ1_0 128K (all 64 layers on the GPU): prefill the 130970-token prefix once and save, then resume from the file.
log "ptq1 128k save"; $R PTQ1_0_ctx131072_save -m $M/Ternary-Bonsai-2-27B-PTQ1_0.gguf --gpu-layers auto --ctx 131072 --parse-special \
    -f $P/pre131008.txt --state-save $P/ptq1_128k.state
log "ptq1 128k load"; $R PTQ1_0_ctx131072_load -m $M/Ternary-Bonsai-2-27B-PTQ1_0.gguf --gpu-layers auto --ctx 131072 --parse-special \
    --state-load $P/ptq1_128k.state -f $P/question.txt -n 24
log "ptq1 128k load r2"; $R PTQ1_0_ctx131072_load_r2 -m $M/Ternary-Bonsai-2-27B-PTQ1_0.gguf --gpu-layers auto --ctx 131072 --parse-special \
    --state-load $P/ptq1_128k.state -f $P/question.txt -n 24

# PTQ1_0 32K and 64K: full needle prompt in one go (prefill + decode)
log "ptq1 32k"; $R PTQ1_0_ctx32768_needle -m $M/Ternary-Bonsai-2-27B-PTQ1_0.gguf --gpu-layers auto --ctx 32768 --parse-special \
    -f $P/p32704.txt -n 24
log "ptq1 64k save"; $R PTQ1_0_ctx65536_save -m $M/Ternary-Bonsai-2-27B-PTQ1_0.gguf --gpu-layers auto --ctx 65536 --parse-special \
    -f $P/pre65472.txt --state-save $P/ptq1_64k.state
log "ptq1 64k load"; $R PTQ1_0_ctx65536_load -m $M/Ternary-Bonsai-2-27B-PTQ1_0.gguf --gpu-layers auto --ctx 65536 --parse-special \
    --state-load $P/ptq1_64k.state -f $P/question.txt -n 24
log "done"
