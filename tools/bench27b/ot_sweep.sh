#!/bin/bash
# -ngl 99 dengan FFN layer k..64 dipindah ke CPU (--override-tensor). Pemakaian: ot_sweep.sh <model> <tag> k1 k2 ...
M=$1; TAG=$2; shift 2
B=${B:-~/llama.cpp/build-iso3/bin}
for k in "$@"; do
  L=$(seq -s '|' "$k" 64)
  OT="blk\.($L)\.ffn_(up|down|gate)\.weight=CPU"
  timeout 900 $B/llama-bench -m "$M" -ngl 99 -ot "$OT" -p 128 -n 32 -r 2 -t 4 -o json > "ot-$TAG-k$k.json" 2> "ot-$TAG-k$k.err"
  python3 - "$k" "ot-$TAG-k$k.json" <<'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[2]))
    print('ffn_gpu_layers=%s' % sys.argv[1], ' '.join('%s/%s:%.2f' % (r['n_prompt'], r['n_gen'], r['avg_ts']) for r in d))
except Exception:
    print('ffn_gpu_layers=%s gagal' % sys.argv[1])
EOF
  grep -m1 -i "out of memory" "ot-$TAG-k$k.err"
done
