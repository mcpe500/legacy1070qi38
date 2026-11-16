#!/usr/bin/env bash
# Starts the engine as an HTTP/SSE server with the WebUI (default http://127.0.0.1:8080). Extra arguments go to qwen38-engine.
#   scripts/serve.sh                  # defaults: 32 GPU layers, iso3 KV, dp4a, MTP 2
#   PORT=9000 scripts/serve.sh --mtp-steps 0
set -eu
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if pgrep -x llama-server > /dev/null || pgrep -x qwen38-engine > /dev/null; then echo "llama-server or another qwen38-engine is running: stop it first" >&2; exit 1; fi
exec taskset -c 0-3 ./build/qwen38-engine -m "${MODEL:-$HOME/models/27b/Qwen3.8-27B-UD-Q4_K_M.gguf}" --server --port "${PORT:-8080}" "$@"
