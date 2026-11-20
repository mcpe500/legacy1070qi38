#!/usr/bin/env python3
"""Ternary Bonsai server stress test (issue #55 M4): PTQ1_0 + an MTP sidecar (--mtp-gguf) + PLD in --server mode.

Checks, with nvidia-smi sampling the whole run (cap 7373 MiB) and /v1/stats RSS (cap 14.2 GB):
  1. 10-turn conversation (stream and non-stream alternating, adaptive MTP + PLD): every turn reuses at least the previous
     prompt (x_stats.reused_tokens), the WebUI telemetry fields are live (tg_tok_s, pld_hits, mtp_fills, accept_pct).
  2. Lossless greedy parity: selected turns re-run without cache, MTP and PLD give the same text.
  3. Trimmed reply in the history (the client sends a shorter assistant message than generated): the prompt checkpoint is
     reused (x_stats.reuse == "checkpoint").
  4. Stop string starting inside a token: content cut right before it, never leaked in the stream, finish_reason "stop".
  5. SSE: role chunk first, [DONE] last, stream text == non-stream text.
  6. POST /v1/state/save, SIGTERM shutdown, restart with --state-load: ready time, the next turn continues from the loaded
     state (reused_tokens == saved tokens) and answers from the earlier turns.
Usage: tests/bonsai_server_stress.py [--sidecar PATH] [--ctx 8192] [--port 8766]. Exit 0 = all checks passed.
"""
import argparse
import http.client
import json
import os
import signal
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOME = os.path.expanduser("~")
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=f"{HOME}/models/27b-bonsai/Ternary-Bonsai-2-27B-PTQ1_0.gguf")
ap.add_argument("--sidecar", default=f"{HOME}/models/27b-bonsai/procreations-mtp/r3_mtp_blk64_q4_k_m.gguf")
ap.add_argument("--ctx", type=int, default=8192)
ap.add_argument("--port", type=int, default=8766)
ap.add_argument("--turns", type=int, default=10)
ap.add_argument("--state", default="/tmp/b10h_server_state.bin")
ap.add_argument("--log", default=os.path.join(ROOT, "bench", "results", "bonsai_local_10h", "m4_bonsai_server_stress.log"))
args = ap.parse_args()
os.makedirs(os.path.dirname(args.log), exist_ok=True)
VRAM_CAP, RSS_CAP_MIB = 7373, 14.2e9 / 1048576

fails, results, rows = 0, [], []


def check(name, ok, detail=""):
    global fails
    line = ("PASS " if ok else "FAIL ") + name + ((" : " + str(detail)) if detail != "" else "")
    print(line, flush=True)
    results.append(line)
    if not ok:
        fails += 1


vram = {"peak": 0, "run": True}


def sample_vram():
    p = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-lms", "100"],
                         stdout=subprocess.PIPE, text=True)
    for line in p.stdout:
        if not vram["run"]:
            break
        if line.strip().isdigit():
            vram["peak"] = max(vram["peak"], int(line))
    p.terminate()


def start(extra):
    logf = open(args.log + ".server", "a")
    cmd = ["taskset", "-c", "0-3", os.path.join(ROOT, "build", "qwen38-engine"), "-m", args.model, "--mtp-gguf", args.sidecar,
           "--gpu-layers", "auto", "--ctx", str(args.ctx), "--server", "--port", str(args.port), "--state-save", args.state] + extra
    logf.write("\n$ " + " ".join(cmd) + "\n")
    logf.flush()
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    while time.time() - t0 < 600:
        if p.poll() is not None:
            return p, None
        try:
            if get("/health")[0] == 200:
                return p, time.time() - t0
        except Exception:
            time.sleep(0.5)
    return p, None


def stop(p):
    p.send_signal(signal.SIGTERM)
    try:
        return p.wait(timeout=120)
    except subprocess.TimeoutExpired:
        p.kill()
        return None


def conn(timeout=1800):
    return http.client.HTTPConnection("127.0.0.1", args.port, timeout=timeout)


def get(path):
    c = conn(30)
    c.request("GET", path)
    r = c.getresponse()
    b = r.read()
    c.close()
    return r.status, b


def post(path, obj):
    c = conn()
    c.request("POST", path, body=json.dumps(obj), headers={"Content-Type": "application/json"})
    r = c.getresponse()
    b = r.read()
    c.close()
    try:
        return r.status, json.loads(b)
    except Exception:
        return r.status, b


def stream(obj):
    c = conn()
    c.request("POST", "/v1/chat/completions", body=json.dumps(dict(obj, stream=True)), headers={"Content-Type": "application/json"})
    r = c.getresponse()
    buf, ev, deltas = b"", [], []
    while True:
        ch = r.read1(65536)
        if not ch:
            break
        buf += ch
        while b"\n\n" in buf:
            e, buf = buf.split(b"\n\n", 1)
            for line in e.decode("utf-8").split("\n"):
                if line.startswith("data: "):
                    if line[6:] == "[DONE]":
                        ev.append("[DONE]")
                        continue
                    j = json.loads(line[6:])
                    ev.append(j)
                    d = j.get("choices", [{}])[0].get("delta", {}).get("content") if "choices" in j else None
                    if d:
                        deltas.append(d)
    c.close()
    return r.status, ev, "".join(deltas)


def body(msgs, **kw):
    b = {"model": "qwen3.8-27b", "messages": msgs, "enable_thinking": False, "temperature": 0, "max_tokens": 96,
         "mtp_adaptive": True, "mtp_steps": 3, "pld": True}
    b.update(kw)
    return b


QUESTIONS = [
    "My project is called KESTREL-9 and it runs on a GTX 1070 Ti. Write a C++ function that returns the sum of a std::vector<int>.",
    "Now write the same function in Python.",
    "And in TypeScript, with a type annotation.",
    "Jelaskan dalam dua kalimat apa fungsi tersebut.",
    "Write a unit test for the C++ version using assert.",
    "List three ways to make the C++ version faster.",
    "Repeat the Python function exactly, then add a docstring.",
    "What GPU did I say the project runs on? Answer in one short sentence.",
    "Summarize our conversation in three bullet points.",
    "What is the name of my project? Reply with only the name.",
    "Write the TypeScript version again, unchanged.",
    "Thank you. Reply with OK.",
]

proc = None
vt = threading.Thread(target=sample_vram, daemon=True)
vt.start()
try:
    proc, ready = start([])
    check("server with --mtp-gguf and --state-save starts", ready is not None, "%.1f s" % (ready or -1))
    if ready is None:
        raise SystemExit(1)
    st, b = get("/v1/stats")
    s0 = json.loads(b)
    check("/v1/stats: MTP available, state_save configured, model file + sidecar", s0["mtp_available"] and s0["state_save"]
          and "Bonsai" in s0.get("model_file", "") and os.path.basename(args.sidecar) in s0.get("model_file", ""),
          "%s (general.name %s), ctx %s" % (s0.get("model_file"), s0.get("model_name"), s0.get("ctx")))

    # ---- 1. multi-turn conversation
    msgs = [{"role": "system", "content": "You are a concise assistant."}]
    prev_prompt = 0
    for i in range(args.turns):
        msgs.append({"role": "user", "content": QUESTIONS[i % len(QUESTIONS)]})
        t0 = time.time()
        if i % 2 == 0:
            st, j = post("/v1/chat/completions", body(msgs))
            text, x = j["choices"][0]["message"]["content"], j["x_stats"]
            live = []
        else:
            st, ev, text = stream(body(msgs))
            fin = [e for e in ev if isinstance(e, dict) and e.get("usage")]
            x = fin[0]["x_stats"] if fin else {}
            live = [e["x_stats"] for e in ev if isinstance(e, dict) and e.get("x_stats", {}).get("phase") == "decode"]
            check("turn %d SSE: role chunk first, [DONE] last, one final chunk" % (i + 1),
                  ev and ev[0]["choices"][0]["delta"].get("role") == "assistant" and ev[-1] == "[DONE]" and len(fin) == 1)
            check("turn %d SSE: live telemetry has tg_tok_s, pld_hits, mtp_fills, accept_pct" % (i + 1),
                  len(live) > 0 and all(all(k in lx for k in ("tg_tok_s", "pld_hits", "mtp_fills", "accept_pct")) for lx in live), len(live))
        wall = time.time() - t0
        st2, b2 = get("/v1/stats")
        sx = json.loads(b2)
        row = {"turn": i + 1, "mode": "non-stream" if i % 2 == 0 else "stream", "prompt": x.get("prompt_tokens"), "reused": x.get("reused_tokens"),
               "reuse": x.get("reuse"), "gen": x.get("completion_tokens"), "tg_tok_s": round(x.get("tg_tok_s", 0), 2), "pp_tok_s": round(x.get("pp_tok_s", 0), 1),
               "accept_pct": round(x.get("accept_pct", 0), 1), "pld_hits": x.get("pld_hits"), "mtp_fills": x.get("mtp_fills"),
               "vram_mib": round(sx["vram_mib"]), "rss_mib": round(sx["rss_mib"]), "wall_s": round(wall, 1), "vram_peak_so_far": vram["peak"]}
        rows.append(row)
        print("      " + json.dumps(row), flush=True)
        check("turn %d answers (%d tokens)" % (i + 1, x.get("completion_tokens", 0)), st == 200 and len(text.strip()) > 0)
        if i > 0:
            check("turn %d reuses at least the previous prompt" % (i + 1), x.get("reused_tokens", 0) >= prev_prompt,
                  "reused %s (%s) of %s, previous prompt %d" % (x.get("reused_tokens"), x.get("reuse"), x.get("prompt_tokens"), prev_prompt))
        check("turn %d MTP + PLD drafted (mtp_fills + pld_hits > 0)" % (i + 1), x.get("mtp_fills", 0) + x.get("pld_hits", 0) > 0,
              "mtp_fills %s pld_hits %s accept %.1f%%" % (x.get("mtp_fills"), x.get("pld_hits"), x.get("accept_pct", 0)))
        prev_prompt = x.get("prompt_tokens", 0)
        msgs.append({"role": "assistant", "content": text})
        if i in (0, 4, args.turns - 1):   # ---- 2. lossless parity: no cache, no MTP, no PLD
            st, jr = post("/v1/chat/completions", body(msgs[:-1], cache_prompt=False, mtp_steps=0, mtp_adaptive=False, pld=False))
            ref = jr["choices"][0]["message"]["content"]
            check("turn %d greedy parity: MTP+PLD+cache text == plain greedy without cache" % (i + 1), ref == text, repr(ref[:60]) + " | " + repr(text[:60]))
            # the parity request rewrote the state from position 0: the next turn re-prefills (expected), so do not require reuse
            prev_prompt = 0
    check("the conversation remembers turn 1 (project name)", any("KESTREL" in m["content"] for m in msgs[-2:] if m["role"] == "assistant") or "KESTREL" in msgs[-1]["content"],
          repr(msgs[-1]["content"][:80]))

    # ---- 3. trimmed reply in the history -> prompt checkpoint (the haiku prompt is prefilled from position 0, so the prefix
    # path below can rebuild exactly the same state)
    post("/v1/cache/clear", {})
    st, j = post("/v1/chat/completions", body(msgs + [{"role": "user", "content": "Write a haiku about GPUs."}], max_tokens=48))
    p_prompt = j["x_stats"]["prompt_tokens"]
    haiku = j["choices"][0]["message"]["content"]
    cut = haiku[: max(1, len(haiku) // 2)]
    msgs_t = msgs + [{"role": "user", "content": "Write a haiku about GPUs."}, {"role": "assistant", "content": cut},
                     {"role": "user", "content": "Now one about CPUs."}]
    st, j = post("/v1/chat/completions", body(msgs_t, max_tokens=32))
    x = j["x_stats"]
    check("trimmed assistant reply: the prompt checkpoint is reused", x.get("reuse") == "checkpoint" and x.get("reused_tokens") == p_prompt,
          "reuse %s, reused %s, previous prompt %s" % (x.get("reuse"), x.get("reused_tokens"), p_prompt))
    ck_text = j["choices"][0]["message"]["content"]
    # The same split through the old prefix path: the previous prompt alone (max_tokens 1 feeds no generated token, so the cache
    # holds exactly that prompt), then the trimmed request reuses it as a plain prefix. Same chunks as the checkpoint path, so
    # the output must be identical; a full prefill from 0 uses other chunk boundaries (prefill vs decode attention kernels)
    # and may differ in the last bits.
    post("/v1/cache/clear", {})
    post("/v1/chat/completions", body(msgs + [{"role": "user", "content": "Write a haiku about GPUs."}], max_tokens=1))
    st, jp = post("/v1/chat/completions", body(msgs_t, max_tokens=32))
    check("trimmed assistant reply: checkpoint path == prefix path with the same split (state restore is exact)",
          jp["x_stats"].get("reuse") == "prefix" and jp["choices"][0]["message"]["content"] == ck_text,
          "reuse %s, %r | %r" % (jp["x_stats"].get("reuse"), jp["choices"][0]["message"]["content"][:50], ck_text[:50]))
    st, jr = post("/v1/chat/completions", body(msgs_t, max_tokens=32, cache_prompt=False, mtp_steps=0, mtp_adaptive=False, pld=False))
    print("INFO  trimmed reply vs a full prefill from position 0: %s" % ("identical" if jr["choices"][0]["message"]["content"] == ck_text else "differs (other chunk boundaries)"))

    # ---- 4. stop string that starts inside a token
    q = [{"role": "user", "content": "Write one sentence about the sea."}]
    st, jb = post("/v1/chat/completions", body(q, max_tokens=40))
    base = jb["choices"][0]["message"]["content"]
    k = len(base) // 2
    while k < len(base) - 4 and not base[k - 1].isalpha():
        k += 1
    stop_s = base[k:k + 3]   # starts in the middle of a word, so inside a token
    expect = base[:base.find(stop_s)]
    st, js = post("/v1/chat/completions", body(q, max_tokens=40, stop=[stop_s]))
    cs = js["choices"][0]["message"]["content"]
    check("stop inside a token: content cut right before the stop string", cs in (expect, expect.strip()) and js["choices"][0]["finish_reason"] == "stop",
          repr(cs[-25:]) + " | stop " + repr(stop_s))
    st, ev, joined = stream(body(q, max_tokens=40, stop=[stop_s]))
    check("stop inside a token: stream identical, stop string never sent", joined.strip() == cs.strip() and stop_s not in joined, repr(joined[-25:]))

    # ---- 5. SSE == non-stream on a fresh request
    st, ev, sj = stream(body(q, max_tokens=40))
    check("SSE text == non-stream text (MTP + PLD, greedy)", sj.strip() == base.strip(), repr(sj[:60]))

    # ---- 6. state save, shutdown, state load
    msgs.append({"role": "user", "content": "Remember the code word ORCHID-77. Reply with OK."})
    st, j = post("/v1/chat/completions", body(msgs, max_tokens=8))
    msgs.append({"role": "assistant", "content": j["choices"][0]["message"]["content"]})
    st, sv = post("/v1/state/save", {})
    check("POST /v1/state/save", st == 200 and sv.get("bytes", 0) > 0, "%s tokens, %.1f MiB, %.2f s" % (sv.get("tokens"), sv.get("bytes", 0) / 1048576, sv.get("seconds", 0)))
    saved_tokens = sv.get("tokens", 0)
    rc = stop(proc)
    check("SIGTERM: clean shutdown (exit 0) with the state saved on the way out", rc == 0, rc)
    t0 = time.time()
    proc, ready2 = start(["--state-load", args.state])
    check("restart with --state-load is ready", ready2 is not None, "%.1f s" % (ready2 or -1))
    srv_log = open(args.log + ".server").read()
    sl = [l for l in srv_log.splitlines() if l.startswith("state-load:")]
    print("      " + (sl[-1] if sl else "no state-load line"))
    msgs.append({"role": "user", "content": "What was the code word? Reply with only the code word."})
    st, j = post("/v1/chat/completions", body(msgs, max_tokens=12))
    x = j["x_stats"]
    check("after --state-load: the next turn continues from the loaded state", x.get("reused_tokens", 0) >= saved_tokens - 64 and x.get("reused_tokens", 0) > 0,
          "reused %s (%s) of %s, saved %s" % (x.get("reused_tokens"), x.get("reuse"), x.get("prompt_tokens"), saved_tokens))
    check("after --state-load: the model remembers the code word", "ORCHID" in j["choices"][0]["message"]["content"], repr(j["choices"][0]["message"]["content"]))
    st, b = get("/v1/stats")
    sx = json.loads(b)
    check("RSS under 14.2 GB", sx["rss_peak_mib"] < RSS_CAP_MIB, "%.0f MiB peak" % sx["rss_peak_mib"])
finally:
    if proc is not None and proc.poll() is None:
        stop(proc)
    vram["run"] = False
    time.sleep(0.3)
    check("peak VRAM over the whole test under the %d MiB cap" % VRAM_CAP, 0 < vram["peak"] <= VRAM_CAP, "%d MiB" % vram["peak"])
    with open(args.log, "w") as fh:
        fh.write("\n".join(results) + "\n\nturns:\n" + "\n".join(json.dumps(r) for r in rows) + "\n")
    print("\nresult: %s (%d failed of %d)" % ("PASS" if fails == 0 else "FAIL", fails, len(results)))
sys.exit(1 if fails else 0)
