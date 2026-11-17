#!/usr/bin/env python3
"""End-to-end test of the embedded HTTP/SSE server. Starts build/qwen38-engine in server mode (loopback, port 8765) unless
--url is given, then checks: /health, /v1/models, /v1/stats, GET / (the WebUI), non-stream and stream chat (SSE format,
<think> block, usage, x_stats telemetry), multi-turn chat with prefix reuse, sampler parameters (seed, top_k, penalties,
temperature), mtp_steps 0 versus 2 (identical output, draft_n_accepted > 0), error handling, and with --long a prompt of more
than 2048 tokens (chunked prefill) with a needle-in-a-haystack question. The WebUI parser is tested by tests/webui_test.py;
here the real stream chunks are also fed through the same JavaScript (QuickJS).
Exit code 0 = all checks passed."""
import argparse
import http.client
import json
import os
import re
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--url", default="")
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--long", action="store_true", help="also run the 2k-token prompt test (about 3 minutes)")
ap.add_argument("--long-entries", type=int, default=28)
ap.add_argument("--long-only", action="store_true", help="run only the long-prompt test")
ap.add_argument("--think-tokens", type=int, default=320, help="max_tokens of the reasoning test")
ap.add_argument("--log", default=os.path.join(ROOT, "bench", "results", "stage5", "server_test.log"))
args = ap.parse_args()
os.makedirs(os.path.dirname(args.log), exist_ok=True)

fails = 0
results = []


def check(name, ok, detail=""):
    global fails
    line = ("PASS " if ok else "FAIL ") + name + ((" : " + str(detail)) if detail != "" else "")
    print(line, flush=True)
    results.append(line)
    if not ok:
        fails += 1


proc = None
if args.url:
    host, port = args.url.replace("http://", "").split(":")
    port = int(port)
else:
    host, port = "127.0.0.1", args.port
    model = os.path.expanduser("~/models/27b/Qwen3.8-27B-UD-Q4_K_M.gguf")
    subprocess.run(["pkill", "-x", "llama-server"], check=False)
    logf = open(args.log + ".server", "w")
    proc = subprocess.Popen(["taskset", "-c", "0-3", os.path.join(ROOT, "build", "qwen38-engine"), "-m", model, "--server", "--port", str(port)],
                            stdout=logf, stderr=subprocess.STDOUT)


def conn(timeout=900):
    return http.client.HTTPConnection(host, port, timeout=timeout)


def get(path):
    c = conn(60)
    c.request("GET", path)
    r = c.getresponse()
    body = r.read()
    c.close()
    return r.status, r.getheader("Content-Type"), body


def post(path, obj, timeout=900):
    c = conn(timeout)
    c.request("POST", path, body=json.dumps(obj), headers={"Content-Type": "application/json"})
    r = c.getresponse()
    body = r.read()
    c.close()
    try:
        return r.status, json.loads(body)
    except Exception:
        return r.status, body


def stream(obj, timeout=900):
    """Returns (events, deltas, t_first_byte, raw) where deltas is the list of delta.content strings in arrival order."""
    obj = dict(obj, stream=True)
    c = conn(timeout)
    t0 = time.time()
    c.request("POST", "/v1/chat/completions", body=json.dumps(obj), headers={"Content-Type": "application/json"})
    r = c.getresponse()
    ctype = r.getheader("Content-Type")
    buf, events, deltas, t_first = b"", [], [], None
    while True:
        chunk = r.read1(65536) if hasattr(r, "read1") else r.read(1)
        if not chunk:
            break
        if t_first is None:
            t_first = time.time() - t0
        buf += chunk
        while b"\n\n" in buf:
            ev, buf = buf.split(b"\n\n", 1)
            for line in ev.decode("utf-8").split("\n"):
                if line.startswith("data: "):
                    p = line[6:]
                    if p == "[DONE]":
                        events.append("[DONE]")
                    else:
                        j = json.loads(p)
                        events.append(j)
                        ch = j.get("choices", [{}])[0] if "choices" in j else {}
                        d = ch.get("delta", {}).get("content")
                        if d:
                            deltas.append(d)
    c.close()
    return r.status, ctype, events, deltas, t_first


def chat(messages, **kw):
    body = {"model": "qwen3.8-27b", "messages": messages, "enable_thinking": False, "temperature": 0, "max_tokens": 40}
    body.update(kw)
    return body


try:
    # ---- wait for the server
    t0 = time.time()
    up = False
    while time.time() - t0 < 600:
        if proc is not None and proc.poll() is not None:
            break
        try:
            if get("/health")[0] == 200:
                up = True
                break
        except Exception:
            time.sleep(2)
    check("server starts and /health answers", up, "%.0f s" % (time.time() - t0))
    if not up:
        raise SystemExit(1)

    if not args.long_only:
        st, ct, body = get("/health")
        check("GET /health", st == 200 and json.loads(body)["status"] == "ok", body.decode())
        st, ct, body = get("/v1/models")
        j = json.loads(body)
        check("GET /v1/models", st == 200 and j["object"] == "list" and j["data"][0]["id"] == "qwen3.8-27b", body.decode()[:120])
        st, ct, body = get("/v1/stats")
        j = json.loads(body)
        check("GET /v1/stats has VRAM and RSS", st == 200 and j["vram_mib"] > 1000 and j["rss_mib"] > 100, "vram %.0f MiB, rss %.0f MiB" % (j["vram_mib"], j["rss_mib"]))
        st, ct, body = get("/")
        page = body.decode()
        on_disk = open(os.path.join(ROOT, "webui", "index.html"), encoding="utf-8").read()
        check("GET / serves the embedded WebUI", st == 200 and "text/html" in ct and page == on_disk and "<title>qwen38-engine</title>" in page, "%d bytes" % len(body))
        st, _, _ = get("/nope")
        check("unknown path is 404", st == 404)

        # ---- errors
        st, j = post("/v1/chat/completions", {"messages": []})
        check("empty messages -> 400", st == 400 and "error" in j)
        st, j = post("/v1/chat/completions", chat([{"role": "user", "content": "hi"}], top_p=0))
        check("bad top_p -> 400", st == 400)
        st, j = post("/v1/chat/completions", chat([{"role": "user", "content": "hi"}], mtp_steps=7))
        check("mtp_steps 7 -> 400", st == 400)
        st, j = post("/v1/chat/completions", chat([{"role": "assistant", "content": "hi"}]))
        check("last message from the assistant -> 400", st == 400)
        c = conn(30); c.request("POST", "/v1/chat/completions", body="{not json", headers={"Content-Type": "application/json"}); r = c.getresponse(); r.read(); c.close()
        check("invalid JSON -> 400", r.status == 400)

        # ---- non-stream
        q = [{"role": "user", "content": "What is the capital of France? Answer with one word."}]
        st, j = post("/v1/chat/completions", chat(q, max_tokens=12))
        msg = j["choices"][0]["message"]
        check("non-stream chat answers Paris", st == 200 and "Paris" in msg["content"], repr(msg["content"]))
        check("non-stream response format", j["object"] == "chat.completion" and j["usage"]["completion_tokens"] > 0 and j["usage"]["prompt_tokens"] > 5
              and j["choices"][0]["finish_reason"] in ("stop", "length") and "x_stats" in j)
        base_text = msg["content"]

        # ---- stream: format, telemetry
        st, ct, ev, deltas, t_first = stream(chat(q, max_tokens=12))
        final = [e for e in ev if isinstance(e, dict) and e.get("usage")]
        check("stream is text/event-stream", st == 200 and "text/event-stream" in ct, ct)
        check("stream starts with the assistant role and ends with [DONE]", ev[0]["choices"][0]["delta"].get("role") == "assistant" and ev[-1] == "[DONE]")
        check("stream text equals the non-stream text (temperature 0)", "".join(deltas).strip() == base_text.strip(), repr("".join(deltas)))
        check("final chunk: finish_reason, usage, telemetry", len(final) == 1 and final[0]["choices"][0]["finish_reason"] in ("stop", "length")
              and final[0]["usage"]["completion_tokens"] > 0)
        x = final[0]["x_stats"]
        for k in ("pp_tok_s", "tg_tok_s", "ttft_ms", "accept_pct", "vram_mib", "rss_mib", "draft_n", "mtp_steps"):
            check("x_stats.%s present" % k, k in x, x.get(k))
        check("x_stats values are plausible", x["ttft_ms"] > 0 and x["tg_tok_s"] > 0.2 and x["vram_mib"] < 7373 and x["rss_mib"] > 100,
              "ttft %.0f ms, tg %.2f, vram %.0f, rss %.0f" % (x["ttft_ms"], x["tg_tok_s"], x["vram_mib"], x["rss_mib"]))

        # ---- multi-turn with prefix reuse
        m1 = [{"role": "system", "content": "You are a terse assistant."}, {"role": "user", "content": "My name is Zorblax. Please remember it and reply with just OK."}]
        st, j1 = post("/v1/chat/completions", chat(m1, max_tokens=8))
        a1 = j1["choices"][0]["message"]["content"]
        m2 = m1 + [{"role": "assistant", "content": a1}, {"role": "user", "content": "What is my name? Reply with only the name."}]
        st, j2 = post("/v1/chat/completions", chat(m2, max_tokens=12))
        a2 = j2["choices"][0]["message"]["content"]
        check("multi-turn: the model remembers the first turn", "Zorblax" in a2, repr(a2))
        check("multi-turn: the cached prefix of turn 1 is reused", j2["x_stats"]["reused_tokens"] > 0, "reused %s of %s tokens" % (j2["x_stats"]["reused_tokens"], j2["x_stats"]["prompt_tokens"]))
        st, j3 = post("/v1/chat/completions", chat(m2, max_tokens=12, cache_prompt=False))
        check("multi-turn: same answer without the cache (cache_prompt=false)", j3["choices"][0]["message"]["content"] == a2 and j3["x_stats"]["reused_tokens"] == 0,
              repr(j3["choices"][0]["message"]["content"]))
        st, j4 = post("/v1/chat/completions", chat(m2[:2] + [{"role": "user", "content": "unrelated: say hello"}], max_tokens=6))
        check("a different conversation falls back to a full prefill", j4["x_stats"]["reused_tokens"] == 0 and j4["choices"][0]["message"]["content"] != "")

        # ---- sampler parameters and mtp_steps
        prompt = [{"role": "user", "content": "Write one sentence about the sea."}]
        def run(**kw):
            st, j = post("/v1/chat/completions", chat(prompt, max_tokens=30, **kw))
            assert st == 200, j
            return j["choices"][0]["message"]["content"], j["x_stats"]
        g0, s0 = run(temperature=0, mtp_steps=0)
        g2, s2 = run(temperature=0, mtp_steps=2, mtp_adaptive=False)
        check("temperature 0: mtp_steps 2 output == mtp_steps 0 output", g0 == g2 and len(g0) > 10, repr(g2[:80]))
        check("mtp_steps 0 drafts nothing, mtp_steps 2 drafts and accepts", s0["draft_n"] == 0 and s2["draft_n"] > 0 and s2["draft_accepted"] > 0 and s2["mtp_steps"] == 2,
              "draft_n %s accepted %s" % (s2["draft_n"], s2["draft_accepted"]))
        a, _ = run(temperature=0.9, seed=5, mtp_steps=0)
        b, _ = run(temperature=0.9, seed=5, mtp_steps=2)
        c_, _ = run(temperature=0.9, seed=6, mtp_steps=0)
        check("same seed -> same sampled text, with and without MTP", a == b, repr(a[:60]) + " / " + repr(b[:60]))
        check("different seed -> different text", a != c_)
        k1, _ = run(temperature=1.5, top_k=1, seed=3, mtp_steps=0)
        check("top_k = 1 equals greedy", k1 == g0)
        hot, _ = run(temperature=2.0, top_k=0, top_p=1.0, seed=11, mtp_steps=0)
        check("temperature 2.0 without truncation differs from greedy", hot != g0)
        mp, _ = run(temperature=1.0, min_p=0.99, seed=4, mtp_steps=0)
        check("min_p 0.99 collapses to the top token (== greedy)", mp == g0)
        tp, _ = run(temperature=1.0, top_p=0.01, seed=4, mtp_steps=0)
        check("top_p 0.01 collapses to the top token (== greedy)", tp == g0)
        rp, _ = run(temperature=0, repeat_penalty=1.8, presence_penalty=1.0, mtp_steps=0)
        check("repeat/presence penalty changes the greedy text", rp != g0, repr(rp[:60]))
        rp2, _ = run(temperature=0, repeat_penalty=1.8, presence_penalty=1.0, mtp_steps=2)
        check("penalties: output identical with mtp_steps 2", rp == rp2)

        # ---- adaptive MTP is the default; output identical to fixed MTP and to MTP off
        ga, sa = run(temperature=0)
        check("adaptive MTP is on by default (x_stats.mtp_adaptive)", sa.get("mtp_adaptive") is True and s2.get("mtp_adaptive") is False, sa.get("mtp_adaptive"))
        check("adaptive MTP: greedy output identical to MTP off and to fixed MTP 2", ga == g0 == g2, repr(ga[:60]))
        check("adaptive MTP drafts and accepts", sa["draft_n"] > 0 and sa["draft_accepted"] > 0, "draft_n %s accepted %s" % (sa["draft_n"], sa["draft_accepted"]))
        gs, _ = run(temperature=0.9, seed=5)
        check("adaptive MTP: same seed -> same sampled text as MTP off", gs == a, repr(gs[:60]))

        # ---- PLD (prompt lookup decoding): opt-in, output identical, live telemetry fields
        rep_prompt = [{"role": "user", "content": "Repeat the following line exactly three times, nothing else:\nThe quick brown fox jumps over the lazy dog near the river bank."}]
        st, jp0 = post("/v1/chat/completions", chat(rep_prompt, max_tokens=60, pld=False))
        st, jp1 = post("/v1/chat/completions", chat(rep_prompt, max_tokens=60, pld=True))
        t0_, t1_ = jp0["choices"][0]["message"]["content"], jp1["choices"][0]["message"]["content"]
        x1 = jp1["x_stats"]
        check("PLD on: output identical to PLD off", t0_ == t1_ and len(t1_) > 20, repr(t1_[:80]))
        check("PLD on: x_stats.pld is true and PLD drafts were used (pld_hits > 0)", x1.get("pld") is True and x1.get("pld_hits", 0) > 0,
              "pld_hits %s pld_candidates %s mtp_fills %s" % (x1.get("pld_hits"), x1.get("pld_candidates"), x1.get("mtp_fills")))
        check("PLD off: no PLD drafts", jp0["x_stats"].get("pld_hits") == 0 and jp0["x_stats"].get("pld") is False)
        st, _, evp, dp, _ = stream(chat(rep_prompt, max_tokens=60, pld=True))
        live = [e["x_stats"] for e in evp if isinstance(e, dict) and e.get("x_stats", {}).get("phase") == "decode"]
        check("stream: live telemetry carries pld_hits and mtp_fills", len(live) > 0 and all("pld_hits" in x and "mtp_fills" in x for x in live), len(live))
        check("stream with PLD: text identical to non-stream", "".join(dp).strip() == t1_.strip())
        st, j = post("/v1/chat/completions", chat(rep_prompt, max_tokens=8, pld=True, pld_max=9))
        check("pld_max out of range -> 400", st == 400)

        # ---- stop strings (OpenAI `stop`): cut before the first match, finish_reason "stop", never leaked across chunks
        st, jb = post("/v1/chat/completions", chat(prompt, max_tokens=30, temperature=0))
        base, bx, bfin = jb["choices"][0]["message"]["content"], jb["x_stats"], jb["choices"][0]["finish_reason"]
        k = len(base) // 2
        stop_s = base[k:k + 5]
        expect = base[:base.find(stop_s)]
        st, js = post("/v1/chat/completions", chat(prompt, max_tokens=30, stop=[stop_s, "ZZZ_never"]))
        cs = js["choices"][0]["message"]["content"]
        check("stop: non-stream content ends right before the stop string", cs == expect.strip() or cs == expect, repr(cs[-30:]) + " | stop " + repr(stop_s))
        check("stop: finish_reason stop and x_stats.stop_matched", js["choices"][0]["finish_reason"] == "stop" and js["x_stats"].get("stop_matched") == stop_s)
        check("stop: fewer completion tokens than without stop", js["usage"]["completion_tokens"] < bx["completion_tokens"], "%s vs %s" % (js["usage"]["completion_tokens"], bx["completion_tokens"]))
        st, _, evs, ds, _ = stream(chat(prompt, max_tokens=30, stop=stop_s))
        joined = "".join(ds)
        fin = [e for e in evs if isinstance(e, dict) and e.get("usage")]
        check("stop: stream text identical to non-stream and the stop string never appears", joined.strip() == cs.strip() and stop_s not in joined, repr(joined[-30:]))
        check("stop: stream finish_reason stop", len(fin) == 1 and fin[0]["choices"][0]["finish_reason"] == "stop")
        st, jn = post("/v1/chat/completions", chat(prompt, max_tokens=30, stop=["ZZZ_never"]))
        check("stop that never matches: same text and finish_reason as without stop", jn["choices"][0]["message"]["content"] == base and jn["choices"][0]["finish_reason"] == bfin,
              repr(jn["choices"][0]["finish_reason"]) + " vs " + repr(bfin))
        for bad, why in [(["a", "b", "c", "d", "e"], "5 strings"), ([""], "empty string"), (7, "number"), (["x" * 129], "129 bytes"), ([1], "non-string item")]:
            st, j = post("/v1/chat/completions", chat(prompt, max_tokens=4, stop=bad))
            check("stop invalid (%s) -> 400" % why, st == 400, st)
        st, _, evt, dt, _ = stream({"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "What is 17 + 25? Think briefly."}],
                                    "temperature": 0, "max_tokens": args.think_tokens, "enable_thinking": True, "reasoning_effort": "low", "stop": ["42"]})
        rt = "".join(dt)
        if "</think>" in rt:
            ans = rt.split("</think>", 1)[1]
            check("stop with thinking: applied to the answer only (reasoning kept, answer cut before 42)", "42" not in ans and len(rt.split("</think>", 1)[0]) > 20, repr(ans[-40:]))
        else:
            print("INFO  stop+thinking: </think> not reached within %d tokens, check skipped" % args.think_tokens)

        # ---- thinking block through the stream, parsed by the real JavaScript of the WebUI
        st, ct, ev, deltas, t_first = stream({"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "What is 17 + 25? Think briefly."}],
                                              "temperature": 0, "max_tokens": args.think_tokens, "enable_thinking": True, "reasoning_effort": "low"})
        raw = "".join(deltas)
        check("thinking stream starts with <think>", raw.startswith("<think>\n"), repr(raw[:30]))
        has_close = "</think>" in raw
        print("INFO  </think> within %d tokens: %s (%d chunks, %d chars)" % (args.think_tokens, has_close, len(deltas), len(raw)))
        try:
            import quickjs
            html = open(os.path.join(ROOT, "webui", "index.html"), encoding="utf-8").read()
            core = re.search(r'<script id="core">(.*?)</script>', html, re.S).group(1)
            ctx = quickjs.Context(); ctx.eval(core)
            out = json.loads(ctx.eval("(function(c){var r=new ReplyState();for(var i=0;i<c.length;i++)r.push(c[i],false);r.push('',true);return JSON.stringify({t:r.think,a:r.answer,raw:r.raw,th:r.thinking,done:r.thinkDone});})(%s)" % json.dumps(deltas)))
            check("WebUI parser on the real stream: raw text preserved", out["raw"] == raw)
            check("WebUI parser on the real stream: reasoning extracted", len(out["t"].strip()) > 5, repr(out["t"][:60]))
            if has_close:
                check("WebUI parser: block closed, answer extracted", out["done"] and not out["th"] and len(out["a"].strip()) > 0, repr(out["a"][:60]))
                check("answer contains 42", "42" in out["a"], repr(out["a"][:80]))
                # next turn with the raw text as history: the server must accept it and keep the reasoning block
                hist = [{"role": "user", "content": "What is 17 + 25? Think briefly."}, {"role": "assistant", "content": raw},
                        {"role": "user", "content": "Add 8 to that. Reply with only the number."}]
                st, j = post("/v1/chat/completions", {"model": "qwen3.8-27b", "messages": hist, "temperature": 0, "max_tokens": 160, "enable_thinking": False})
                check("history with a raw <think> block is accepted (turn 2)", st == 200 and "50" in j["choices"][0]["message"]["content"], repr(j["choices"][0]["message"]["content"]) if st == 200 else str(j))
        except ImportError:
            print("SKIP  quickjs not installed: JS parser check on the real stream")

        # ---- non-stream thinking split
        st, j = post("/v1/chat/completions", {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "Say hi."}], "temperature": 0, "max_tokens": 48,
                                              "enable_thinking": True, "reasoning_effort": "low"})
        m = j["choices"][0]["message"]
        check("non-stream thinking: reasoning_content or open content", st == 200 and ("reasoning_content" in m or m["content"].startswith("<think>")), repr(m)[:100])

    # ---- long prompt: chunked prefill (2k+ tokens)
    if args.long or args.long_only:
        words = ("The harbour authority recorded wind speed, tide height and cargo tonnage every hour. " * 4 + "Ferries left at dawn. ")
        filler = "\n".join("Log entry %d: %s" % (i, words) for i in range(1, args.long_entries + 1))
        needle = "The secret access code of the archive is PINEAPPLE-7421."
        long_msg = [{"role": "user", "content": needle + "\n\n" + filler + "\n\nQuestion: what is the secret access code of the archive? Reply with the code only."}]
        t1 = time.time()
        st, j = post("/v1/chat/completions", chat(long_msg, max_tokens=16), timeout=1800)
        dt = time.time() - t1
        x = j["x_stats"] if st == 200 else {}
        n = j["usage"]["prompt_tokens"] if st == 200 else 0
        print("INFO  long prompt: %d tokens, %.0f s, pp %.1f tok/s" % (n, dt, x.get("pp_tok_s", 0)))
        check("long prompt has more than 2048 tokens and is accepted", st == 200 and n > 2048, n)
        check("needle-in-a-haystack over %d tokens (chunked prefill, iso3 KV)" % n, st == 200 and "7421" in j["choices"][0]["message"]["content"], repr(j["choices"][0]["message"]["content"]) if st == 200 else str(j))
        check("VRAM stays under the 7373 MiB cap during the long prefill", x.get("vram_mib", 1e9) < 7373, x.get("vram_mib"))
        st, j = post("/v1/chat/completions", chat([{"role": "user", "content": "word " * 5000}], max_tokens=4))
        check("a prompt longer than the context is rejected cleanly (400)", st == 400 and j["error"]["type"] == "context_length_exceeded")
finally:
    if proc is not None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
    open(args.log, "w").write("\n".join(results) + "\n")
    print("\nresult: %s (%d failed of %d)" % ("PASS" if fails == 0 else "FAIL", fails, len(results)))
sys.exit(1 if fails else 0)
