#!/usr/bin/env python3
"""M1 benchmark of issue #55: Ternary Bonsai PTQ1_0 decode with and without an MTP draft head, over four prompt suites.

One engine server per draft head (none, donor Qwen3.8 blk.64, r3 Q4_K_M, r3 Q8_0 sidecars via --mtp-gguf), so the 5.4 GiB
model is loaded once per head instead of once per run. Every request is greedy (temperature 0), thinking closed, no prompt
cache, 256 new tokens. The draft variant (mtp_steps, mtp_adaptive, pld) is a per-request parameter of the server.

Usage:
  bench/bonsai_mtp_suite.py prompts SIZE...            write bench/results/bonsai_local_10h/prompts/<suite>_<size>.json
  bench/bonsai_mtp_suite.py run SIZE HEAD VARIANT...   start the server for HEAD, run every suite x VARIANT, append JSONL
  bench/bonsai_mtp_suite.py report                     Markdown tables + greedy parity from the JSONL files

HEAD: none | donor | q4km | q8.  VARIANT: m0 | m1 | m2 | m3 | ad (adaptive, up to 4 drafts) | pld2 | pldad
Env: MODEL (PTQ1_0 GGUF), SIDECAR_DIR, PORT (8091), CTX (8192), N_PREDICT (256), SUITES (comma list).
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(ROOT, "build", "qwen38-engine")
OUT = os.path.join(ROOT, "bench", "results", "bonsai_local_10h")
PROMPTS = os.path.join(OUT, "prompts")
HOME = os.path.expanduser("~")
MODEL = os.environ.get("MODEL", f"{HOME}/models/27b-bonsai/Ternary-Bonsai-2-27B-PTQ1_0.gguf")
SIDECAR_DIR = os.environ.get("SIDECAR_DIR", f"{HOME}/models/27b-bonsai/procreations-mtp")
HEADS = {
    "none": None,
    "donor": "donor_qwen_blk64.gguf",
    "q4km": "r3_mtp_blk64_q4_k_m.gguf",
    "q8": "r3_mtp_blk64_q8_0.gguf",
}
VARIANTS = {
    "m0": dict(mtp_steps=0, mtp_adaptive=False, pld=False),
    "m1": dict(mtp_steps=1, mtp_adaptive=False, pld=False),
    "m2": dict(mtp_steps=2, mtp_adaptive=False, pld=False),
    "m3": dict(mtp_steps=3, mtp_adaptive=False, pld=False),
    "ad": dict(mtp_steps=3, mtp_adaptive=True, pld=False),
    "pld2": dict(mtp_steps=2, mtp_adaptive=False, pld=True),
    "pldad": dict(mtp_steps=3, mtp_adaptive=True, pld=True),
}
SUITES = ["coding", "chat_en", "chat_id", "repetitive"]


def read(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        return fh.read()


def git_files(*dirs, exts):
    files = subprocess.run(["git", "ls-files", *dirs], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    return [f for f in sorted(files) if f.endswith(exts) and "/results/" not in f]


def material(suite):
    if suite == "coding":
        fs = git_files("src/engine", "tools", "tests", "webui", exts=(".cpp", ".hpp", ".py", ".html"))
        return "".join(f"\n===== FILE {f} =====\n" + read(f) for f in fs)
    if suite == "chat_en":
        return read("docs/AFK_STATUS.md") + read("docs/OPT_LOG_2026-11-16.md") + read("docs/stage1-baseline.md")
    if suite == "chat_id":
        parts = ["../README.md", "README.md", "../spec/specs/SPEC-000-project-charter.md", "../spec/handoff/HANDOFF-LOG.md",
                 "../spec/handoff/SESSION-001-bootstrap.md", "../spec/specs/RFC-001-quantization-strategy.md"]
        return "".join("\n\n" + read(p) for p in parts)
    if suite == "repetitive":
        fs = git_files("src", exts=(".cpp", ".cu", ".hpp"))
        return "".join(f"\n===== FILE {f} =====\n" + read(f) for f in fs)
    raise ValueError(suite)


OPS = ["ADD", "SUB", "MUL", "DIV", "MOD", "AND", "OR", "XOR", "SHL", "SHR", "NOT", "NEG", "LOAD", "STORE", "PUSH", "POP",
       "JMP", "JZ", "JNZ", "CALL", "RET", "CMP", "INC", "DEC", "MOV", "LEA", "NOP", "HALT", "IN", "OUT", "SWAP", "DUP"]

QUESTIONS = {
    "coding": ("The text above is source code from an inference engine. Do three small tasks, each in a fenced code block: "
               "(1) C++17: a template class RingBuffer<T> with push, pop, size and a tiny test in main; "
               "(2) Python: a function that reads a CSV of benchmark rows (variant, tok_s) and prints the mean tok_s per variant; "
               "(3) TypeScript: a typed async fetchWithRetry(url, retries) helper with exponential backoff."),
    "chat_en": ("The text above is a project status report. In a friendly conversational tone, explain to a new teammate "
                "what the project does, what limits its speed on this GPU, and what they should try next. Use short paragraphs."),
    "chat_id": ("Teks di atas adalah dokumentasi proyek. Jelaskan dalam bahasa Indonesia yang santai kepada rekan baru: "
                "apa tujuan proyek ini, perangkat keras apa yang dipakai, apa kendala utamanya, dan langkah apa berikutnya. "
                "Gunakan paragraf pendek."),
    "repetitive": ("Write C++17 code only. Define enum class Op with the opcodes " + ", ".join(OPS) + " (values 0..31). "
                   "Then write const char * op_name(Op op) as a switch with one case per opcode returning its name, "
                   "and int op_arity(Op op) as a switch with one case per opcode."),
}
SYSTEM = "You are a helpful assistant."


def render(msgs):  # same ChatML layout as the server template with thinking closed
    s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
    return s + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def count(texts):
    tmp = "/tmp/b10h_suite_tok.txt"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\x1e".join(texts))
    out = subprocess.run([ENGINE, "-m", MODEL, "--parse-special", "--tokenize-file", tmp], capture_output=True, text=True,
                         check=True).stdout
    return [line.count(",") + 1 for line in out.strip().splitlines()]


def build(suite, n):
    body = material(suite)
    while len(body) < n * 6:
        body += body
    q = QUESTIONS[suite]

    def msgs(cut):
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body[:cut] + "\n\n" + q}]

    lo, hi = 0, min(len(body), n * 6)
    while lo < hi:   # largest char cut whose rendered prompt is <= n tokens
        mid = (lo + hi + 1) // 2
        if count([render(msgs(mid))])[0] <= n:
            lo = mid
        else:
            hi = mid - 1
    m = msgs(lo)
    return m, count([render(m)])[0]


def cmd_prompts(sizes):
    os.makedirs(PROMPTS, exist_ok=True)
    for n in sizes:
        for s in SUITES:
            m, nt = build(s, n)
            with open(os.path.join(PROMPTS, f"{s}_{n}.json"), "w", encoding="utf-8") as fh:
                json.dump({"suite": s, "target": n, "tokens": nt, "messages": m}, fh, ensure_ascii=False)
            print(f"{s}_{n}: {nt} tokens")


def http(port, path, body=None, timeout=7200):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def cmd_run(size, head, variants):
    port = int(os.environ.get("PORT", "8091"))
    ctx = int(os.environ.get("CTX", "8192"))
    n_pred = int(os.environ.get("N_PREDICT", "256"))
    suites = os.environ.get("SUITES", ",".join(SUITES)).split(",")
    args = ["taskset", "-c", "0-3", ENGINE, "-m", MODEL, "--gpu-layers", "auto", "--ctx", str(ctx), "--server", "--port", str(port),
            "--mtp-steps", "0" if head == "none" else "3"]
    if HEADS[head]:
        args += ["--mtp-gguf", os.path.join(SIDECAR_DIR, HEADS[head])]
    tag = f"suite_{size}_{head}"
    # /tmp/b10h_pause_srv: hold before starting the server, so a GPU test can run with no other process holding VRAM
    while os.path.exists("/tmp/b10h_pause_srv"):
        open("/tmp/b10h_pause_srv_ack", "w").close()
        time.sleep(2)
    if os.path.exists("/tmp/b10h_pause_srv_ack"):
        os.remove("/tmp/b10h_pause_srv_ack")
    log = open(os.path.join(OUT, tag + ".server.err"), "w")
    smi_f = open(os.path.join(OUT, tag + ".vram.csv"), "w")
    smi = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-lms", "100"], stdout=smi_f)
    t0 = time.time()
    srv = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
    try:
        while True:
            if srv.poll() is not None:
                raise RuntimeError(f"server exited with {srv.returncode}, see {tag}.server.err")
            try:
                http(port, "/health", timeout=2)
                break
            except Exception:
                time.sleep(1)
        ready_s = time.time() - t0
        print(f"[{tag}] ready in {ready_s:.1f} s", flush=True)
        with open(os.path.join(OUT, f"suite_{size}.jsonl"), "a", encoding="utf-8") as res:
            for s in suites:
                p = json.load(open(os.path.join(PROMPTS, f"{s}_{size}.json"), encoding="utf-8"))
                for v in variants:
                    # /tmp/b10h_pause: hold before the next request (kernel tests need an idle GPU); the ack file says it is idle
                    while os.path.exists("/tmp/b10h_pause"):
                        open("/tmp/b10h_pause_ack", "w").close()
                        time.sleep(1)
                    if os.path.exists("/tmp/b10h_pause_ack"):
                        os.remove("/tmp/b10h_pause_ack")
                    body = {"messages": p["messages"], "temperature": 0, "seed": 42, "max_tokens": n_pred, "enable_thinking": False,
                            "cache_prompt": False, "stream": False, **VARIANTS[v]}
                    w0 = time.time()
                    r = http(port, "/v1/chat/completions", body)
                    x = r["x_stats"]
                    row = {"size": size, "suite": s, "head": head, "variant": v, "prompt_tokens": x["prompt_tokens"],
                           "completion_tokens": x["completion_tokens"], "tg_tok_s": x["tg_tok_s"], "pp_tok_s": x["pp_tok_s"],
                           "prefill_s": x["prefill_s"], "accept_pct": x["accept_pct"], "draft_n": x["draft_n"],
                           "draft_accepted": x["draft_accepted"], "pld_hits": x["pld_hits"], "pld_candidates": x["pld_candidates"],
                           "mtp_fills": x["mtp_fills"], "cycles": x["cycles"], "mtp_steps": x["mtp_steps"], "vram_mib": x["vram_mib"],
                           "rss_peak_mib": x["rss_peak_mib"], "wall_s": round(time.time() - w0, 2), "ready_s": round(ready_s, 1),
                           "finish": r["choices"][0]["finish_reason"], "t_start": round(w0, 1), "t_end": round(time.time(), 1),
                           "text": r["choices"][0]["message"]["content"]}
                    res.write(json.dumps(row, ensure_ascii=False) + "\n")
                    res.flush()
                    print(f"[{tag}] {s:10s} {v:6s} tg {row['tg_tok_s']:6.2f} tok/s  accept {row['accept_pct']:5.1f}%  "
                          f"n {row['completion_tokens']}  pld_hits {row['pld_hits']}", flush=True)
    finally:
        srv.terminate()
        try:
            srv.wait(30)
        except subprocess.TimeoutExpired:
            srv.kill()
        smi.terminate()
        smi.wait()
        smi_f.close()
        vals = [int(l) for l in open(os.path.join(OUT, tag + ".vram.csv")) if l.strip().isdigit()]
        peak = max(vals) if vals else 0
        print(f"[{tag}] peak VRAM {peak} MiB (cap 7373): {'OK' if peak <= 7373 else 'OVER'}", flush=True)
        with open(os.path.join(OUT, "suite_vram.jsonl"), "a") as fh:
            fh.write(json.dumps({"tag": tag, "peak_vram_mib": peak, "cap_ok": peak <= 7373}) + "\n")


def cmd_report():
    rows = []
    for f in sorted(os.listdir(OUT)):
        if f.startswith("suite_") and f.endswith(".jsonl") and f != "suite_vram.jsonl":
            rows += [json.loads(l) for l in open(os.path.join(OUT, f), encoding="utf-8")]
    last = {}
    for r in rows:   # a re-run replaces the earlier row of the same configuration
        last[(r["size"], r["suite"], r["head"], r["variant"])] = r
    rows = list(last.values())
    out = []
    for size in sorted({r["size"] for r in rows}, key=int):
        rs = [r for r in rows if r["size"] == size]
        base = {r["suite"]: r for r in rs if r["head"] == "none" and r["variant"] == "m0"}
        out.append(f"\n### Prompt {size} tokens (decode {rs[0]['completion_tokens'] if rs else 0}-ish tokens, greedy)\n")
        same = {(r["suite"], r["head"]): r for r in rs if r["variant"] == "m0"}
        out.append("| suite | head | variant | prompt | gen | tg tok/s | vs none | vs m0 same server | accept % | drafts acc/n | pld hits | parity vs none |")
        out.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for r in sorted(rs, key=lambda r: (SUITES.index(r["suite"]), list(HEADS).index(r["head"]), list(VARIANTS).index(r["variant"]))):
            b = base.get(r["suite"])
            sp = f"{r['tg_tok_s'] / b['tg_tok_s']:.3f}x" if b and b["tg_tok_s"] else "-"
            m0 = same.get((r["suite"], r["head"]))
            sp0 = f"{r['tg_tok_s'] / m0['tg_tok_s']:.3f}x" if m0 and m0["tg_tok_s"] and r["head"] != "none" else "-"
            par = "-" if not b else ("identical" if r["text"] == b["text"] else "DIFFERS")
            out.append(f"| {r['suite']} | {r['head']} | {r['variant']} | {r['prompt_tokens']} | {r['completion_tokens']} | "
                       f"{r['tg_tok_s']:.2f} | {sp} | {sp0} | {r['accept_pct']:.1f} | {r['draft_accepted']}/{r['draft_n']} | {r['pld_hits']} | {par} |")
    print("\n".join(out))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    c = sys.argv[1]
    if c == "prompts":
        cmd_prompts([int(x) for x in sys.argv[2:]])
    elif c == "run":
        cmd_run(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif c == "report":
        cmd_report()
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
