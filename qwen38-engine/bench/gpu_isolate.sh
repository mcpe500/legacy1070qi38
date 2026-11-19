#!/usr/bin/env bash
# GPU isolation for long engine runs: another agent session (opencode) shares this machine and has started GPU jobs that
# killed engine runs. Usage:
#   bench/gpu_isolate.sh pause    SIGSTOP every opencode process, record the PIDs in $PIDS
#   bench/gpu_isolate.sh resume   SIGCONT every recorded PID (idempotent)
#   bench/gpu_isolate.sh check    exit 0 only when no compute process holds the GPU and no llama-server runs
#   bench/gpu_isolate.sh status   print the state of the recorded PIDs and of the GPU
# A driver script should run `pause`, then `trap 'bench/gpu_isolate.sh resume' EXIT INT TERM`.
set -u
PIDS=${GPU_ISOLATE_PIDS:-/tmp/bonsai_v2_paused.pids}
cmd=${1:-status}
case "$cmd" in
  pause)
    mapfile -t list < <(pgrep -f opencode || true)
    : > "$PIDS.new"
    for p in "${list[@]}"; do
      [ "$p" = "$$" ] && continue
      if kill -STOP "$p" 2>/dev/null; then echo "$p" >> "$PIDS.new"; echo "paused $p $(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-100)"; fi
    done
    cat "$PIDS.new" >> "$PIDS" 2>/dev/null; sort -u "$PIDS" -o "$PIDS"; rm -f "$PIDS.new"
    ;;
  resume)
    [ -f "$PIDS" ] || { echo "nothing to resume"; exit 0; }
    while read -r p; do [ -n "$p" ] && kill -CONT "$p" 2>/dev/null && echo "resumed $p"; done < "$PIDS"
    rm -f "$PIDS"
    ;;
  check)
    apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
    if [ -n "$apps" ]; then echo "GPU busy: $apps"; exit 1; fi
    if pgrep -x llama-server > /dev/null; then echo "llama-server running"; exit 1; fi
    echo "GPU free ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader))"
    ;;
  status)
    [ -f "$PIDS" ] && while read -r p; do ps -o pid=,stat=,cmd= -p "$p" 2>/dev/null | cut -c1-100; done < "$PIDS"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    ;;
  *) echo "usage: $0 pause|resume|check|status"; exit 2 ;;
esac
