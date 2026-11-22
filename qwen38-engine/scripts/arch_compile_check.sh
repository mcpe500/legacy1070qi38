#!/usr/bin/env bash
# Compile check of every engine .cu for the architectures the code claims to support, without touching the
# active sm_61 build. The engine ships for sm_61 only (GTX 1070 Ti, CMAKE_CUDA_ARCHITECTURES 61); a kernel that
# also has to build for newer cards (the dispatch table of docs/QUANT_MATH_AND_GPU_DISPATCH.md) is guarded with
# __CUDA_ARCH__, so the guard itself has to compile on each target to be worth anything.
#
# Usage: scripts/arch_compile_check.sh [arch ...]        (default: 61 75 86 89)
# For each arch it compiles every src/cuda/*.cu to a cubin in a temp directory and reports the failures.
# Exit 0 only if every (arch, file) pair compiles. It does not link and does not need a GPU.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NVCC="${NVCC:-/usr/local/cuda-12.6/bin/nvcc}"
[ -x "$NVCC" ] || NVCC="$(command -v nvcc)"
if [ -z "$NVCC" ]; then echo "arch_compile_check: nvcc not found" >&2; exit 2; fi
ARCHS=("$@")
[ ${#ARCHS[@]} -eq 0 ] && ARCHS=(61 75 86 89)

INC=(-I"$ROOT/src/cuda" -I"$ROOT/src/common" -I"$ROOT/src/packer" -I"$ROOT/src/cpu"
     -I"$HOME/llama.cpp/ggml/include" -I"$HOME/llama.cpp/include")
FLAGS=(-O3 -std=c++17 -w --cubin -Xcompiler -w)

tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
fail=0; total=0
for a in "${ARCHS[@]}"; do
    n=0
    for f in "$ROOT"/src/cuda/*.cu; do
        o="$tmp/$(basename "$f" .cu)_sm$a.cubin"
        total=$((total + 1))
        if ! "$NVCC" "${FLAGS[@]}" -arch=sm_"$a" -o "$o" "$f" "${INC[@]}" > "$tmp/log" 2>&1; then
            echo "[FAIL] sm_$a $(basename "$f")"
            grep -m3 "error" "$tmp/log" | sed 's/^/       /'
            fail=$((fail + 1))
        else
            n=$((n + 1))
        fi
    done
    printf "[%s] sm_%s: %d of %d kernels compile\n" "$([ "$n" -eq $(ls "$ROOT"/src/cuda/*.cu | wc -l) ] && echo PASS || echo FAIL)" "$a" "$n" "$(ls "$ROOT"/src/cuda/*.cu | wc -l)"
done
echo "arch_compile_check: $((total - fail)) of $total (arch, kernel) pairs compile, $fail fail -> $([ "$fail" -eq 0 ] && echo PASS || echo FAIL)"
exit $([ "$fail" -eq 0 ] && echo 0 || echo 1)