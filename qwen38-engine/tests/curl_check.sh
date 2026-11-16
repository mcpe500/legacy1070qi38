#!/usr/bin/env bash
# curl checks against a running server (scripts/serve.sh). Usage: tests/curl_check.sh [BASE_URL]
BASE="${1:-http://127.0.0.1:8080}"
set -u
echo "== GET /health";            curl -s "$BASE/health"; echo
echo "== GET /v1/models";         curl -s "$BASE/v1/models"; echo
echo "== GET / (first 3 lines, size)"; curl -s "$BASE/" | head -3; echo "bytes: $(curl -s "$BASE/" | wc -c)"
echo "== GET /v1/stats";          curl -s "$BASE/v1/stats"; echo
echo "== POST /v1/chat/completions, stream, curl -N (thinking on, 40 tokens)"
curl -sN "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model":"qwen3.8-27b","stream":true,"max_tokens":40,"temperature":0.7,"top_k":20,"top_p":0.95,"seed":1,"mtp_steps":2,
  "enable_thinking":true,"reasoning_effort":"low",
  "messages":[{"role":"user","content":"Name three primary colours."}]}'
echo "== POST /v1/chat/completions, non-stream, mtp_steps 0 vs 2, temperature 0"
for m in 0 2; do
  curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
    "model":"qwen3.8-27b","max_tokens":24,"temperature":0,"mtp_steps":'"$m"',"enable_thinking":false,
    "messages":[{"role":"user","content":"Write a short greeting."}]}' | python3 -c "
import sys,json; j=json.load(sys.stdin); x=j['x_stats']
print('mtp_steps=$m', repr(j['choices'][0]['message']['content']), 'draft_n=%d accepted=%d tg=%.2f tok/s ttft=%.0f ms' % (x['draft_n'], x['draft_accepted'], x['tg_tok_s'], x['ttft_ms']))"
done
