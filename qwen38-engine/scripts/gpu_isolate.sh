#!/usr/bin/env bash
# GPU isolation guard for every long GPU test (branch exp/bonsai-overnight-v2-prefill-t17-drafter).
# Default: only verify safety. Opt-in --stall-opencode PID pauses that PID and always SIGCONTs it on exit.
# Usage:
#   scripts/gpu_isolate.sh <engine command...>
#   GPU_BUDGET_MIB=7373 scripts/gpu_isolate.sh --vram-budget-mib 7373 ninja -C build -j4 qwen38-engine
#   scripts/gpu_isolate.sh --stall-opencode "$OPENCODE_PID" taskset -c 0-3 ./build/qwen38-engine ...
set -euo pipefail

vram_budget_mib=7373
mem_budget_mib=14541
stall_pid=""
cmd=()

usage() {
  cat <<'EOF'
GPU isolation guard for long GPU tests.

Options:
  --vram-budget-mib N   Maximum nvidia-smi used GPU memory allowed at start (default 7373).
  --mem-budget-mib N    Maximum free-monitor "used" host memory allowed at start (default 14541 ~= 14.2 GiB).
  --stall-opencode PID  SIGSTOP PID before the command and SIGCONT it on exit (opt-in only).
  --help                Show this help.

Everything after -- is the command. Without --, all non-option arguments are the command.

Exit codes:
  0  safe to run the command
  2  bad arguments
  3  another GPU compute process or runtime is active
  4  VRAM above the requested budget
  5  host memory above the requested budget
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vram-budget-mib)
      [[ $# -ge 2 ]] || { echo "gpu_isolate: --vram-budget-mib needs a value" >&2; exit 2; }
      vram_budget_mib="$2"; shift 2 ;;
    --mem-budget-mib)
      [[ $# -ge 2 ]] || { echo "gpu_isolate: --mem-budget-mib needs a value" >&2; exit 2; }
      mem_budget_mib="$2"; shift 2 ;;
    --stall-opencode)
      [[ $# -ge 2 ]] || { echo "gpu_isolate: --stall-opencode needs a PID" >&2; exit 2; }
      stall_pid="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --) shift; while [[ $# -gt 0 ]]; do cmd+=("$1"); shift; done ;;
    *) cmd+=("$1"); shift ;;
  esac
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "gpu_isolate: nvidia-smi not found" >&2
  exit 3
fi

gpu_pids_raw="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader || true)"
if [[ -n "${gpu_pids_raw//[[:space:]]/}" ]]; then
  echo "gpu_isolate: refuse to start: a GPU compute process is active:" >&2
  printf '%s\n' "${gpu_pids_raw}" >&2
  exit 3
fi

for runtime in qwen38-engine llama-server llama-cli llama-bench; do
  if pgrep -x "${runtime}" >/dev/null 2>&1; then
    echo "gpu_isolate: refuse to start: runtime '${runtime}' is active" >&2
    pgrep -ax "${runtime}" >&2 || true
    exit 3
  fi
done

vram_line="$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits)"
IFS=',' read -r total_mib used_mib <<<"${vram_line}"
total_mib="${total_mib//[[:space:]]/}"
used_mib="${used_mib//[[:space:]]/}"
if [[ -z "${used_mib}" || -z "${total_mib}" ]]; then
  echo "gpu_isolate: cannot parse nvidia-smi memory fields" >&2
  exit 4
fi
if (( used_mib > vram_budget_mib )); then
  echo "gpu_isolate: refuse to start: used GPU ${used_mib} MiB > ${vram_budget_mib} MiB" >&2
  exit 4
fi

mem_used_mib="$(free -m | awk '/^Mem:/ {print $3}')"
if [[ -z "${mem_used_mib}" ]]; then
  echo "gpu_isolate: cannot parse free(1) memory fields" >&2
  exit 5
fi
if (( mem_used_mib > mem_budget_mib )); then
  echo "gpu_isolate: refuse to start: used host RAM ${mem_used_mib} MiB > ${mem_budget_mib} MiB" >&2
  exit 5
fi

stalled_now=""
cleanup() {
  local code=$?
  if [[ -n "${stalled_now}" ]]; then
    echo "gpu_isolate: SIGCONT PID ${stalled_now}" >&2
    kill -CONT "${stalled_now}" 2>/dev/null || true
    for _ in $(seq 1 100); do
      if ! kill -0 "${stalled_now}" 2>/dev/null; then break; fi
      proc_state="$(awk '{print $3}' "/proc/${stalled_now}/status" 2>/dev/null || true)"
      if [[ "${proc_state}" != "T" && "${proc_state}" != "t" ]]; then break; fi
      sleep 0.05
    done
  fi
  exit "${code}"
}
trap cleanup EXIT INT TERM

if [[ -n "${stall_pid}" ]]; then
  if ! kill -0 "${stall_pid}" 2>/dev/null; then
    echo "gpu_isolate: cannot stall PID ${stall_pid}: not running" >&2
    exit 3
  fi
  echo "gpu_isolate: SIGSTOP PID ${stall_pid} (explicit opt-in)" >&2
  kill -STOP "${stall_pid}"
  for _ in $(seq 1 100); do
    proc_state="$(awk '{print $3}' "/proc/${stall_pid}/status" 2>/dev/null || true)"
    [[ "${proc_state}" == "T" || "${proc_state}" == "t" ]] && break
    sleep 0.05
  done
  stalled_now="${stall_pid}"
fi

echo "gpu_isolate: safe: GPU ${used_mib}/${total_mib} MiB (budget ${vram_budget_mib}), host RAM ${mem_used_mib} MiB (budget ${mem_budget_mib})"

if [[ ${#cmd[@]} -eq 0 ]]; then
  exit 0
fi

exec "${cmd[@]}"