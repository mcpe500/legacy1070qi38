#!/usr/bin/env python3
"""Answer-quality sanity check through an OpenAI-compatible server (llama-server or qwen38-engine --server).
  sanity_prompts.py URL LABEL OUT.json [--think-tokens N]
Six prompts with checkable answers; temperature 0. Non-thinking prompts disable reasoning (both server dialects are sent:
"enable_thinking": false for qwen38-engine and chat_template_kwargs for llama-server). One reasoning prompt runs with thinking
on (reasoning_effort "medium"). Generated code is only checked structurally, never executed."""
import json
import re
import sys
import time
import urllib.request

URL, LABEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
THINK = int(sys.argv[sys.argv.index("--think-tokens") + 1]) if "--think-tokens" in sys.argv else 3000

PROMPTS = [
    ("arith", "What is 17 * 23? Reply with only the number.", lambda a: "391" in a),
    ("capital", "What is the capital of Australia? Reply with one word.", lambda a: "canberra" in a.lower()),
    ("speed", "A train travels 120 km in 1.5 hours. What is its average speed in km/h? Reply with only the number.", lambda a: re.search(r"\b80(\.0+)?\b", a) is not None),
    ("translate", "Translate into Indonesian: 'The meeting starts at nine tomorrow morning.' Reply with only the translation.",
     lambda a: ("rapat" in a.lower() or "pertemuan" in a.lower()) and ("sembilan" in a.lower() or "09" in a or "9" in a) and "besok" in a.lower()),
    ("code", "Write a Python function is_prime(n) that returns True if n is a prime number and False otherwise. Reply with only the code.",
     lambda a: "def is_prime" in a and "return" in a),
]
THINK_PROMPT = ("discount", "A shirt costs $25 after a 20% discount. What was the original price? Give the final answer as a number.",
                lambda a: re.search(r"31\.25", a) is not None)


def ask(prompt, thinking, max_tokens):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "temperature": 0, "top_k": 1, "seed": 42,
            "max_tokens": max_tokens, "stream": False, "enable_thinking": thinking,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    if thinking:
        body["reasoning_effort"] = "medium"
        body["chat_template_kwargs"]["reasoning_effort"] = "medium"
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    j = json.load(urllib.request.urlopen(req, timeout=3600))
    dt = time.time() - t0
    m = j["choices"][0]["message"]
    content = m.get("content") or ""
    reasoning = m.get("reasoning_content") or ""
    if "</think>" in content:   # servers that return the raw block
        reasoning, content = content.split("</think>", 1)
    return content.strip(), reasoning.strip(), j.get("usage", {}).get("completion_tokens"), dt, j["choices"][0].get("finish_reason")


res = []
for name, p, ok in PROMPTS:
    a, r, n, dt, fin = ask(p, False, 300)
    res.append({"name": name, "thinking": False, "pass": bool(ok(a)), "answer": a, "tokens": n, "seconds": round(dt, 1), "finish": fin})
    print("%-8s %-9s %s  (%s tok, %.0f s) %r" % (LABEL, name, "PASS" if ok(a) else "FAIL", n, dt, a[:120]), flush=True)
name, p, ok = THINK_PROMPT
a, r, n, dt, fin = ask(p, True, THINK)
res.append({"name": name, "thinking": True, "pass": bool(ok(a)), "answer": a, "reasoning_chars": len(r), "tokens": n, "seconds": round(dt, 1), "finish": fin})
print("%-8s %-9s %s  (%s tok, %.0f s, reasoning %d chars, finish %s) %r" % (LABEL, name, "PASS" if ok(a) else "FAIL", n, dt, len(r), fin, a[:120]), flush=True)
json.dump({"label": LABEL, "url": URL, "results": res, "passed": sum(x["pass"] for x in res), "total": len(res)}, open(OUT, "w"), indent=1)
print("%s: %d / %d passed" % (LABEL, sum(x["pass"] for x in res), len(res)))
