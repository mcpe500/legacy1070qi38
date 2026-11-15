#!/bin/bash
# Ukur TPS dan acceptance MTP via llama-server lokal (127.0.0.1:8091), satu depth per panggilan.
# Pemakaian: mtp_server.sh <model> <ngl> <depth> <out-prefix> [extra args...]
M=$1; NGL=$2; D=$3; OUT=$4; shift 4
B=${B:-~/llama.cpp/build-cuda/bin}
X=""
[ "$D" != 0 ] && X="--spec-type draft-mtp --spec-draft-n-max $D"
$B/llama-server -m "$M" -ngl "$NGL" -c 2048 --host 127.0.0.1 --port 8091 $X "$@" > "$OUT.server.log" 2>&1 &
PID=$!
for i in $(seq 300); do
  curl -sf http://127.0.0.1:8091/health > /dev/null && break
  kill -0 $PID 2>/dev/null || { echo "server mati"; exit 1; }
  sleep 1
done
curl -s http://127.0.0.1:8091/completion -H 'Content-Type: application/json' -d '{"prompt":"<|im_start|>user\nTulis fungsi Python untuk menghitung bilangan prima sampai n, lalu jelaskan.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n","n_predict":128,"temperature":0,"seed":1,"cache_prompt":false}' > "$OUT.json"
kill -TERM $PID; wait $PID 2>/dev/null
python3 -c "
import json,hashlib;d=json.load(open('$OUT.json'));t=d['timings']
print('depth=$D','tg_tps=%.2f'%t['predicted_per_second'],'n=%d'%t['predicted_n'],'draft_n=',t.get('draft_n'),'accepted=',t.get('draft_n_accepted'),'md5=',hashlib.md5(d['content'].encode()).hexdigest()[:8])"
