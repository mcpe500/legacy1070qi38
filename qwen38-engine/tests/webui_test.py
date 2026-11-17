#!/usr/bin/env python3
"""Headless tests of the WebUI logic. Runs the real JavaScript of webui/index.html in QuickJS (pip install quickjs):
the <script id="core"> block (ThinkParser, SSEParser, buildRequest, ReplyState) is evaluated and fuzzed; the <script id="ui">
block is only compiled (syntax check), since it needs a DOM. Exit code 0 = all passed."""
import itertools
import json
import os
import re
import sys

import quickjs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
html = open(os.path.join(ROOT, "webui", "index.html"), encoding="utf-8").read()
core = re.search(r'<script id="core">(.*?)</script>', html, re.S).group(1)
ui = re.search(r'<script id="ui">(.*?)</script>', html, re.S).group(1)
fails = 0


def check(name, ok, detail=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ((" : " + detail) if (detail and not ok) else ""))
    if not ok:
        fails += 1


ctx = quickjs.Context()
ctx.eval(core)
ctx.eval("""
function runParser(chunks) {            // returns {think, answer, raw, thinking, thinkDone, switches}
  var r = new ReplyState(), sw = 0;
  for (var i = 0; i < chunks.length; i++) { var ev = r.push(chunks[i], false); for (var j = 0; j < ev.length; j++) if (ev[j].type === 'switch') sw++; }
  r.push('', true);
  return JSON.stringify({think: r.think, answer: r.answer, raw: r.raw, thinking: r.thinking, thinkDone: r.thinkDone, sw: sw});
}
function runSSE(chunks) { var p = new SSEParser(), out = []; for (var i = 0; i < chunks.length; i++) out = out.concat(p.push(chunks[i])); return JSON.stringify(out); }
function req(s, m) { return JSON.stringify(buildRequest(s, m)); }
""")


def parse(chunks):
    return json.loads(ctx.eval("runParser(%s)" % json.dumps(chunks)))


# ---- ThinkParser: every way of cutting the stream into 1, 2 or 3 chunks gives the same result as one chunk
samples = [
    "<think>\nI need to add 2 and 2.\n</think>\n\nThe answer is 4.",
    "<think>\n\n</think>\n\nOK",
    "<think>\nuse a < b and <thin and </thi in text\n</think>\n\nx <b>bold</b> <think",
    "plain answer without tags",
    "<think>\nunfinished reasoning that never closes <",
    "<think>\nA\n</think>\n\nB <think>\nC\n</think>\n\nD",
    "<think>é日本語🙂\n</think>\n\nünïcode",
]
for s in samples:
    ref = parse([s])
    bad = None
    for i in range(len(s) + 1):
        r = parse([s[:i], s[i:]])
        if (r["think"], r["answer"]) != (ref["think"], ref["answer"]):
            bad = ("2-split", i, r); break
    if not bad:
        for i, j in itertools.combinations(range(len(s) + 1), 2):
            r = parse([s[:i], s[i:j], s[j:]])
            if (r["think"], r["answer"]) != (ref["think"], ref["answer"]):
                bad = ("3-split", i, j, r); break
    if not bad:
        r = parse(list(s))  # one character per chunk
        if (r["think"], r["answer"]) != (ref["think"], ref["answer"]):
            bad = ("1-char", r)
    check("ThinkParser split-invariant: %r" % s[:40], bad is None, str(bad))
    check("ThinkParser keeps raw text: %r" % s[:40], parse(list(s))["raw"] == s)

r = parse(["<think>\nreasoning", " here\n</", "think>\n\nFinal", " answer"])
check("think/answer content", r["think"] == "\nreasoning here\n" and r["answer"] == "Final answer", str(r))
check("answer loses the blank lines after </think>", parse(["<think>x</think>\n\n  hi"])["answer"] == "hi")
check("two switches per block", parse(["<think>a</think>b"])["sw"] == 2)
check("unfinished block stays in thinking mode", parse(["<think>abc"])["thinking"] is True)
check("a lone '<' at the end is flushed at the end of the stream", parse(["hello <"])["answer"] == "hello <")
check("held-back tag prefix is not lost", parse(["a <thi", "nk>b</think>c"])["think"] == "b")

# ---- SSEParser
stream = 'data: {"a":1}\n\ndata: {"b":2}\n\ndata: [DONE]\n\n'
exp = ['{"a":1}', '{"b":2}', "[DONE]"]
ok = True
for i in range(len(stream) + 1):
    if json.loads(ctx.eval("runSSE(%s)" % json.dumps([stream[:i], stream[i:]]))) != exp:
        ok = False; break
check("SSEParser: events cut at every position", ok)
check("SSEParser: CRLF", json.loads(ctx.eval("runSSE(%s)" % json.dumps(['data: x\r\n\r\ndata: y\r\n\r\n']))) == ["x", "y"])
check("SSEParser: comment lines and partial event are ignored", json.loads(ctx.eval("runSSE(%s)" % json.dumps([': keepalive\n\ndata: z']))) == [])

# ---- buildRequest
s = dict(temperature="0.8", top_k="40", top_p="0.9", min_p="0.05", repeat_penalty="1.1", presence_penalty="0.3", seed="", max_tokens="128",
         mtp_steps="2", reasoning_effort="low", enable_thinking=True, cache_prompt=True)
b = json.loads(ctx.eval("req(%s, %s)" % (json.dumps(s), json.dumps([{"role": "user", "content": "hi"}]))))
check("buildRequest numbers", b["temperature"] == 0.8 and b["top_k"] == 40 and b["top_p"] == 0.9 and b["min_p"] == 0.05 and b["repeat_penalty"] == 1.1
      and b["presence_penalty"] == 0.3 and b["max_tokens"] == 128 and b["mtp_steps"] == 2 and b["stream"] is True)
check("buildRequest: empty seed is omitted", "seed" not in b)
s2 = dict(s, seed="7", mtp_steps="9", max_tokens="0", enable_thinking=False)
b = json.loads(ctx.eval("req(%s, [])" % json.dumps(s2)))
check("buildRequest: seed, clamps", b["seed"] == 7 and b["mtp_steps"] == 3 and b["max_tokens"] == 1 and b["enable_thinking"] is False)

# ---- stop strings, adaptive MTP and PLD in the request
ctx.eval("function stops(t) { return JSON.stringify(parseStops(t)); }")
check("parseStops: one per line, empty lines dropped", json.loads(ctx.eval("stops(%s)" % json.dumps("###\n\nEND\n"))) == ["###", "END"])
check("parseStops: \\n and \\t escapes", json.loads(ctx.eval("stops(%s)" % json.dumps("a\\nb\nc\\td"))) == ["a\nb", "c\td"])
check("parseStops: at most 4", len(json.loads(ctx.eval("stops(%s)" % json.dumps("1\n2\n3\n4\n5\n6")))) == 4)
check("parseStops: non-string -> []", json.loads(ctx.eval("stops(undefined)")) == [])
b0 = json.loads(ctx.eval("req(%s, [])" % json.dumps(s)))
check("buildRequest: adaptive MTP on and PLD off by default, no stop field", b0["mtp_adaptive"] is True and b0["pld"] is False and "stop" not in b0)
b1 = json.loads(ctx.eval("req(%s, [])" % json.dumps(dict(s, mtp_adaptive=False, pld=True, stop="###\n\\n\\nUser:"))))
check("buildRequest: toggles and stop array", b1["mtp_adaptive"] is False and b1["pld"] is True and b1["stop"] == ["###", "\n\nUser:"])

# ---- the UI script compiles (syntax) and only touches elements that exist in the page
try:
    quickjs.Context().eval("(function(){" + ui + "\n})")
    check("UI script compiles", True)
except Exception as e:
    check("UI script compiles", False, str(e))
ids = set(re.findall(r'\$\(\'([A-Za-z_]+)\'\)', ui))
html_ids = set(re.findall(r'id="([A-Za-z_]+)"', html))
missing = sorted(i for i in ids if i not in html_ids)
check("every $('id') used by the UI exists in the page", not missing, str(missing))
for k in ["temperature", "top_k", "top_p", "min_p", "repeat_penalty", "presence_penalty", "seed", "max_tokens", "mtp_steps",
          "t_pp", "t_tg", "t_ttft", "t_acc", "t_vram", "t_rss", "t_pld", "t_fill", "pld", "mtp_adaptive", "stop"]:
    check("control / telemetry element #%s exists" % k, k in html_ids)
no_ext = not re.search(r'(src|href)\s*=\s*["\']https?://', html) and "cdn" not in html.lower() and "@import" not in html
check("no external dependency (no CDN, no remote src/href)", no_ext)
check("page size under 100 KB", len(html.encode()) < 100 * 1024, str(len(html.encode())))

print("\nresult: %s (%d failed)" % ("PASS" if fails == 0 else "FAIL", fails))
sys.exit(1 if fails else 0)
