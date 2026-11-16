#!/usr/bin/env bash
# SASS gate for every CUDA source of the engine (skill pascal-sm61-cuda, rule 8). Exit code 0 only if all checks pass.
#
#  1. Controls: a probe using __dp4a must produce IDP.4A, a probe using __hfma2 must produce HFMA2. Without both the
#     grep patterns cannot be trusted, so the gate fails.
#  2. Every src/cuda/*.cu is compiled to an sm_61 cubin (same flags as CMakeLists.txt) and must contain ZERO packed-half
#     arithmetic: HFMA2, HMUL2, HADD2.
#  3. A source that calls __dp4a must produce IDP.4A in its cubin.
#  4. The linked executable (build/qwen38-engine, embedded sm_61 SASS) is scanned the same way, and the engine as a whole
#     must contain IDP.4A.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 2
cd "$ROOT" || exit 2
OUT="$ROOT/build-sass"
rm -rf "$OUT"
mkdir -p "$OUT"
INC="-Isrc/cuda -Isrc/common -Isrc/packer"
HALF_RE='HFMA2|HMUL2|HADD2'
rc=0
pass() { printf '[PASS] %s\n' "$*"; }
fail() { printf '[FAIL] %s\n' "$*"; rc=1; }

# 1. controls
cat > "$OUT/ctl_dp4a.cu" <<'EOF'
__global__ void k(const int* a, const int* b, int* c) { int i = threadIdx.x; c[i] = __dp4a(a[i], b[i], c[i]); }
EOF
cat > "$OUT/ctl_half.cu" <<'EOF'
#include <cuda_fp16.h>
__global__ void k(const __half2* a, const __half2* b, __half2* c) { int i = threadIdx.x; c[i] = __hfma2(a[i], b[i], c[i]); }
EOF
for c in dp4a half; do
  nvcc -gencode arch=compute_61,code=sm_61 -cubin -O3 -o "$OUT/ctl_$c.cubin" "$OUT/ctl_$c.cu" 2> "$OUT/ctl_$c.log" || { fail "control $c did not compile"; }
  cuobjdump -sass "$OUT/ctl_$c.cubin" > "$OUT/ctl_$c.sass" 2>/dev/null
done
c1=$(grep -c 'IDP\.4A' "$OUT/ctl_dp4a.sass"); c2=$(grep -cE "$HALF_RE" "$OUT/ctl_half.sass")
if [ "$c1" -gt 0 ] && [ "$c2" -gt 0 ]; then pass "controls: __dp4a -> IDP.4A x$c1, __hfma2 -> half arithmetic x$c2 (patterns are sensitive)"
else fail "controls: IDP.4A x$c1, half arithmetic x$c2 (the grep is not trustworthy)"; fi

# 2 and 3. per-source cubins
total_dp4a=0
for src in src/cuda/*.cu; do
  name="$(basename "$src" .cu)"
  extra=""
  [ "$name" = "iso3_quant" ] && extra="--fmad=false"
  if ! nvcc -gencode arch=compute_61,code=sm_61 -cubin -O3 $extra $INC -o "$OUT/$name.cubin" "$src" 2> "$OUT/$name.nvcc.log"; then
    fail "$name: nvcc failed (see $OUT/$name.nvcc.log)"; continue
  fi
  cuobjdump -sass "$OUT/$name.cubin" > "$OUT/$name.sass"
  nd=$(grep -c 'IDP\.4A' "$OUT/$name.sass"); nh=$(grep -cE "$HALF_RE" "$OUT/$name.sass")
  total_dp4a=$((total_dp4a + nd))
  uses=$(grep -c '__dp4a' "$src")
  ok=1
  [ "$nh" -ne 0 ] && ok=0
  [ "$uses" -gt 0 ] && [ "$nd" -eq 0 ] && ok=0
  if [ "$ok" -eq 1 ]; then pass "$name: IDP.4A x$nd, half arithmetic x$nh"; else fail "$name: IDP.4A x$nd (source calls __dp4a $uses times), half arithmetic x$nh"; fi
done

# 4. linked executable
EXE="build/qwen38-engine"
if [ -x "$EXE" ]; then
  cuobjdump -sass "$EXE" > "$OUT/engine.sass" 2>/dev/null
  ed=$(grep -c 'IDP\.4A' "$OUT/engine.sass"); eh=$(grep -cE "$HALF_RE" "$OUT/engine.sass")
  if [ "$ed" -gt 0 ] && [ "$eh" -eq 0 ]; then pass "$EXE (linked): IDP.4A x$ed, half arithmetic x$eh"
  else fail "$EXE (linked): IDP.4A x$ed, half arithmetic x$eh"; fi
else
  fail "$EXE not built"
fi
[ "$total_dp4a" -gt 0 ] || fail "no IDP.4A anywhere in the engine sources"
printf '\nSASS gate: %s\n' "$([ $rc -eq 0 ] && echo PASS || echo FAIL)"
exit $rc
