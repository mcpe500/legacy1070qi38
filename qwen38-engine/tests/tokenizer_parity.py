#!/usr/bin/env python3
"""Tokenizer parity: qwen38-engine --tokenize-file vs llama-tokenize (baseline llama.cpp).

Each corpus sample is tokenized by both. Samples are bucketed so the result shows where the hand-written
pre-tokenizer is exact (ASCII and Latin-1) and where the Unicode class approximation may differ.
Usage: tokenizer_parity.py [--model GGUF] [--engine BIN] [--tokenize BIN]
"""
import argparse
import ast
import os
import subprocess
import sys

HOME = os.path.expanduser("~")
RS = "\x1e"

ASCII = [
    "The capital of France is",
    "Hello, world!",
    "  leading and trailing  ",
    "line one\nline two\r\nline three\n\n\nend",
    "tabs\there\t\tand spaces    here",
    "It's John's book, isn't it? They've said we'll go; I'd say you're right. I'M SURE IT'S FINE.",
    "def f(x):\n    return x**2 + 3.14159  # comment\n",
    "for (int i = 0; i < n; ++i) { sum += a[i] * b[i]; }",
    "1234567890 and 3.14 and 1,000,000 and 0x1F and 2026-11-15",
    "email: user.name+tag@example.com, url: https://example.com/a/b?c=d&e=f#g",
    "a" * 200,
    " " * 40 + "x",
    "x" + " " * 40,
    "!!!???...,,,;;;:::---___===+++***///\\\\\\|||",
    "'s 't 're 've 'm 'll 'd 'S 'T 'RE 'VE 'M 'LL 'D 'x",
    "\n\n\n   \n  \n",
    "word\n\nword\n \nword",
    "Jakarta adalah ibu kota Indonesia. Saya sedang belajar menjalankan model bahasa besar di kartu grafis lama.",
    "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "{\"key\": [1, 2, 3], \"nested\": {\"a\": null, \"b\": true}}",
    "SELECT * FROM users WHERE id = 42 AND name LIKE '%abc%';",
    "A quick brown fox jumps over the lazy dog. " * 6,
    "",
    " ",
    "\n",
]

LATIN1 = [
    "café résumé naïve façade Zürich über ñandú",
    "Ångström ÆØÅ æøå ß ÿ",
    "price: £5, €7, ¥300, ¢50, ©2026, ®, ™",
    "½ ¼ ¾ x² y³ z¹",
    "non-breaking\u00a0space and soft\u00adhyphen",
]

NON_ASCII = [
    "你好，世界！今天天气很好。",
    "こんにちは世界。これはテストです。",
    "한국어 문장입니다.",
    "Привет, мир! Это тест.",
    "Γειά σου Κόσμε",
    "مرحبا بالعالم",
    "עולם שלום",
    "emoji: 😀🎉🚀 and symbols: → ← ✓ ✗ ★ ♥",
    "quotes: “curly” ‘single’ — dash – en … ellipsis",
    "math: ∑ ∫ √ ≈ ≠ ≤ ≥ ∞ π",
    "combining: e\u0301 a\u0308 n\u0303",
    "fullwidth: ＡＢＣ １２３ （括弧）",
    "mixed English 中文 and Indonesian kata",
]


def run_engine(engine, model, samples, parse_special):
    path = "/tmp/tokparity_corpus.txt"
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(RS.join(samples))
    cmd = [engine, "-m", model, "--tokenize-file", path]
    if parse_special:
        cmd.append("--parse-special")
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip("\n").split("\n")
    return [ast.literal_eval(l) if l else [] for l in out]


def run_ref(tokenize, model, text, parse_special):
    cmd = [tokenize, "-m", model, "--ids", "--no-bos", "--no-escape", "--log-disable", "-p", text]
    if not parse_special:
        cmd.append("--no-parse-special")
    r = subprocess.run(cmd, capture_output=True, text=True)
    line = [l for l in r.stdout.strip().split("\n") if l.startswith("[")]
    return ast.literal_eval(line[-1]) if line else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=f"{HOME}/models/27b/Qwen3.8-27B-UD-Q4_K_M.gguf")
    ap.add_argument("--engine", default="build/qwen38-engine")
    ap.add_argument("--tokenize", default=f"{HOME}/llama.cpp/build-iso3/bin/llama-tokenize")
    args = ap.parse_args()

    total = bad = 0
    summary = []
    for name, samples, special in (("ascii", ASCII, True), ("latin1", LATIN1, False), ("non-ascii", NON_ASCII, False)):
        ours = run_engine(args.engine, args.model, samples, special)
        nb = 0
        for text, mine in zip(samples, ours):
            ref = run_ref(args.tokenize, args.model, text, special)
            total += 1
            if mine != ref:
                bad += 1
                nb += 1
                print(f"[DIFF {name}] {text!r}\n    ours: {mine}\n    ref : {ref}")
        summary.append((name, len(samples), nb))
    print("\nbucket       samples  differ")
    for name, n, nb in summary:
        print(f"{name:<12} {n:>7}  {nb:>6}")
    print(f"total {total} samples, {bad} differ")
    return 1 if summary[0][2] or summary[1][2] else 0  # ASCII and Latin-1 must match exactly


if __name__ == "__main__":
    sys.exit(main())
