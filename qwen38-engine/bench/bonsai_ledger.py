#!/usr/bin/env python3
"""Ledger of every Bonsai / Qwen run in bench/results/bonsai/: configuration from the tag, speeds from the run's JSON, exact peak VRAM from the
100 ms nvidia-smi samples (.vram.csv, kept locally; the ledger is the committed record), and the status against the 7373 MiB cap.
  python3 bench/bonsai_ledger.py [--md OUT.md]"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "bench", "results", "bonsai")
CAP = 7373


def peak(tag):
    p = os.path.join(R, tag + ".vram.csv")
    if not os.path.exists(p):
        return None
    v = [int(x) for x in open(p).read().split() if x.isdigit()]
    return max(v) if v else None


def first_steady(tag):   # load VRAM for server runs: the most frequent sample before the request starts is not recorded; use min of the top plateau
    return None


UB_DEFAULT = {"srv_q2_f16_c4096_ngl99": "512"}   # run before the UB option existed: llama-server default ubatch
tags = sorted(set(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(R, "*.json"))) |
              set(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(R, "*.log")) if not f.endswith((".server.log", ".log.server"))))
rows = []
for tag in tags:
    f = os.path.join(R, tag + ".json")
    if tag.startswith("header_") or tag.startswith("sanity_") or tag.endswith(".request"):
        continue
    try:
        d = json.load(open(f))
    except Exception:
        d = None
    if d is None and os.path.exists(os.path.join(R, tag + ".log")):
        lg = open(os.path.join(R, tag + ".log"), errors="replace").read()
        if "out of memory" in lg:
            err = "load failed: " + re.search(r"allocating [0-9.]+ MiB on device 0: cudaMalloc failed: out of memory", lg).group(0)
    kind = "server" if tag.startswith("srv_") else "llama-bench"
    pp = tg = n_prompt = None
    err = locals().get("err", "") if d is None else ""
    if kind == "llama-bench" and isinstance(d, list):
        for x in d:
            if x.get("n_prompt"):
                pp = x.get("avg_ts")
            elif x.get("n_gen"):
                tg = x.get("avg_ts")
        n_prompt = 512
    elif kind == "server" and isinstance(d, dict):
        t = d.get("timings", {})
        pp, tg, n_prompt = t.get("prompt_per_second"), t.get("predicted_per_second"), t.get("prompt_n")
        if "error" in d:
            err = d["error"].get("message", "")[:70]
        if t.get("predicted_n") == 1:
            err = "1 token generated (greedy EOS); rerun with ignore_eos"
            tg = None
    elif d is None or d == []:
        err = "no result (see .log)"
    lp = os.path.join(R, tag + ".log")
    if d is None and os.path.exists(lp):
        mm = re.search(r"allocating [0-9.]+ MiB on device 0: cudaMalloc failed: out of memory", open(lp, errors="replace").read())
        if mm:
            err = "load failed: " + mm.group(0)
    pk = peak(tag)
    model = "Qwen Q4_K_M" if tag.startswith("qwen") else ("PTQ1_0" if "ptq1" in tag else "Q2_0")
    runtime = "prism fork" if ("ptq1" in tag or tag.startswith("prism")) else "stock"
    m = {k: re.search(p, tag) for k, p in {"kv": r"_(f16|q8_0|q8|q4_0|q4|iso3)_", "d": r"_d(\d+)", "c": r"_c(\d+)", "ngl": r"_ngl(\d+)", "ub": r"_ub(\d+)"}.items()}
    kv = m["kv"].group(1) if m["kv"] else ""
    kv = {"q4": "q4_0", "q8": "q8_0"}.get(kv, kv)
    ctx = ("depth %s" % m["d"].group(1)) if m["d"] else (("ctx %s" % m["c"].group(1)) if m["c"] else "")
    ub = m["ub"].group(1) if m["ub"] else UB_DEFAULT.get(tag, "128" if (kind == "server" and model == "Q2_0") else "512")
    if "ignore_eos" in err:
        status, err = "superseded", err + "; VRAM samples overwritten by the rerun"
    elif pp is None and tg is None:
        status = "FAILED"
    else:
        status = "n/a" if pk is None else ("OK" if pk <= CAP else "OVER CAP")
    rows.append((tag, model, runtime, kind, kv, ctx, m["ngl"].group(1) if m["ngl"] else "", ub, n_prompt, pp, tg, pk, status, err))

L = ["| run | model | runtime | tool | KV | context | -ngl | -ub | prompt tok | pp tok/s | tg tok/s | peak VRAM MiB | vs 7373 MiB | note |",
     "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
for r in rows:
    tag, model, rt, kind, kv, ctx, ngl, ub, npr, pp, tg, pk, st, err = r
    L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        tag, model, rt, kind, kv, ctx, ngl, ub, npr if npr else "", "%.2f" % pp if pp else "-", "%.2f" % tg if tg else "-",
        pk if pk is not None else "-", st, err))
txt = "\n".join(L) + "\n"
print(txt)
if "--md" in sys.argv:
    open(sys.argv[sys.argv.index("--md") + 1], "w").write(txt)
