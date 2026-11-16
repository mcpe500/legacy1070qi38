#!/usr/bin/env python3
"""Compares the greedy token ids of qwen38-engine (JSON summary line) with the baseline llama-server response.
Usage: compare_tokens.py ENGINE_STDOUT_FILE BASELINE_JSON
Exit code 0 when every generated token is identical."""
import json
import sys


def main():
    engine_out, baseline_json = sys.argv[1], sys.argv[2]
    line = [l for l in open(engine_out, encoding="utf-8").read().splitlines() if l.startswith('{"prompt_tokens"')][-1]
    eng = json.loads(line)
    base = json.load(open(baseline_json, encoding="utf-8"))
    a, b = eng["ids"], base["tokens"]
    n = min(len(a), len(b))
    first = next((i for i in range(n) if a[i] != b[i]), None)
    print(f"engine tokens  : {len(a)}")
    print(f"baseline tokens: {len(b)}")
    print(f"compared       : {n}")
    if first is None and len(a) == len(b):
        print(f"result: IDENTICAL ({n}/{n} tokens)")
        print(f"text  : {eng['text']!r}")
        return 0
    print(f"result: DIFFER at index {first}" if first is not None else "result: LENGTH DIFFERS")
    if first is not None:
        print(f"  engine  : {a[first:first + 5]}")
        print(f"  baseline: {b[first:first + 5]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
