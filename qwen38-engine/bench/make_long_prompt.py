#!/usr/bin/env python3
"""Long-context prompt for the exp/bonsai-128k-context sweep: a needle sentence, then the engine's own source tree
(tracked text files, repeated if needed), then a question about the needle. Trimmed to an exact token count with the
engine tokenizer (qwen38-engine --tokenize-file, no weights loaded).

The text uses the ChatML template: run the engine with --parse-special.
Usage: bench/make_long_prompt.py MODEL.gguf N_TOKENS OUT.txt [--prefix-out PREFIX.txt]
  OUT.txt      needle + code + question (N_TOKENS tokens, +-1 at the join)
  PREFIX.txt   needle + code only (for --state-save; the question is the continuation for --state-load)
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(ROOT, "build", "qwen38-engine")
# ChatML (run the engine with --parse-special). Thinking is closed right away so the first tokens are the answer.
NEEDLE = ("<|im_start|>user\nRemember this fact for later: the secret passphrase of the vault is MAGENTA-FALCON-42.\n\n"
          "Below is a source listing.\n")
QUESTION = ("\n\nEnd of the source listing.\nQuestion: what is the secret passphrase of the vault, stated at the very "
            "beginning of this message? Reply with the passphrase only.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


def corpus():
    files = subprocess.run(["git", "ls-files", "src", "tests", "bench", "docs", "webui", "scripts"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.split()
    parts = []
    for f in sorted(files):
        if not f.endswith((".cpp", ".hpp", ".cu", ".cuh", ".py", ".sh", ".md", ".html", ".txt")):
            continue
        if "/results/" in f:
            continue
        try:
            with open(os.path.join(ROOT, f), encoding="utf-8") as fh:
                parts.append(f"\n===== FILE {f} =====\n" + fh.read())
        except (UnicodeDecodeError, OSError):
            pass
    return "".join(parts)


def count(model, texts):
    tmp = "/tmp/bonsai128k_tok.txt"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\x1e".join(texts))
    out = subprocess.run([ENGINE, "-m", model, "--parse-special", "--tokenize-file", tmp], capture_output=True, text=True, check=True).stdout
    return [line.count(",") + 1 for line in out.strip().splitlines()]


def main():
    model, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    prefix_out = sys.argv[5] if len(sys.argv) > 5 and sys.argv[4] == "--prefix-out" else None
    body = corpus()
    while len(body) < n * 5:
        body += body
    nq = count(model, [QUESTION])[0]
    lo, hi = 0, min(len(body), n * 6)
    while lo < hi:   # largest char cut whose needle + body + question is <= n tokens
        mid = (lo + hi + 1) // 2
        if count(model, [NEEDLE + body[:mid]])[0] + nq <= n:
            lo = mid
        else:
            hi = mid - 1
    prefix = NEEDLE + body[:lo]
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(prefix + QUESTION)
    if prefix_out:
        with open(prefix_out, "w", encoding="utf-8") as fh:
            fh.write(prefix)
    full, pre = count(model, [prefix + QUESTION, prefix])
    print(f"{out}: {full} tokens (prefix {pre}, question {nq})")


if __name__ == "__main__":
    main()
