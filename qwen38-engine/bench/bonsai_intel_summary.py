#!/usr/bin/env python3
"""Summarises the intelligence suite into bench/results/bonsai/intel/INTEL.md and intel.json.
Inputs: llama-perplexity logs and VRAM samples in bench/results/bonsai/intel/, run lines (GPU layers, wall time) in intel/run_lines.txt.
Matched comparison: llama.cpp draws task subsets with mt19937(1) without replacement, so the first N tasks of a larger run are the N-task
subset; a model's running score at task N (printed per task in its log) is compared with another model's N-task run."""
import json
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "bench", "results", "bonsai", "intel")
NAMES = {"q2": "Bonsai `Q2_0`", "ptq1": "Bonsai `PTQ1_0`", "qwen": "Qwen3.8 Q4_K_M"}
MC = [("hellaswag", "HellaSwag (acc_norm)", 100), ("winogrande", "Winogrande", 200), ("arc", "ARC-Challenge", 100), ("mmlu", "MMLU", 150)]


def log(tag):
    p = os.path.join(D, tag + ".log")
    return open(p, errors="replace").read() if os.path.exists(p) else None


def running(task, text):
    """{n: accuracy %} from the per-task lines."""
    if text is None:
        return {}
    if task == "hellaswag":
        rows = re.findall(r"^(\d+)\t([0-9.]+)%\t\[", text, re.M)
    elif task == "winogrande":
        rows = re.findall(r"^(\d+)\t([0-9.]+)\t +-?[0-9.]+ +-?[0-9.]+ +\d +\d", text, re.M)
    else:
        rows = re.findall(r"^(\d+)\t([0-9.]+)\s*$", text, re.M)
    return {int(n): float(a) for n, a in rows}


def final_ok(task, text):
    if text is None:
        return False
    if task == "hellaswag":
        return "Final" in text or bool(re.search(r"^\d+\t[0-9.]+%\t\[", text, re.M))
    if task == "winogrande":
        return "Final Winogrande score" in text
    return "Final result:" in text


def per_item(task, text, n):
    """Correct (1) / wrong (0) for items 1..n, reconstructed from the running accuracy: c_i = round(a_i * i - a_(i-1) * (i-1))."""
    run = running(task, text)
    out, prev = [], 0.0
    for i in range(1, n + 1):
        if i not in run:
            return None
        tot = run[i] / 100 * i
        out.append(int(round(tot - prev)))
        prev = round(tot)
    return out


def mcnemar(a, b):
    """Exact two-sided McNemar test on paired outcomes; returns (only_a, only_b, p)."""
    x = sum(1 for u, v in zip(a, b) if u and not v)
    y = sum(1 for u, v in zip(a, b) if v and not u)
    n = x + y
    if n == 0:
        return x, y, 1.0
    k = min(x, y)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return x, y, min(1.0, 2 * p)


def se(p, n):
    p = p / 100
    return 100 * math.sqrt(max(p * (1 - p), 1e-9) / n)


def ppl(text):
    m = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", text or "")
    return (float(m.group(1)), float(m.group(2))) if m else None


def kld(text):
    if not text:
        return None
    m = re.search(r"Mean +KLD: +([0-9.]+) ± +([0-9.]+)", text)
    t = re.search(r"Same top p: +([0-9.]+) ± +([0-9.]+) %", text)
    r = re.search(r"Mean +PPL\(Q\) +: +([0-9.]+)", text)
    return dict(kld=float(m.group(1)), kld_se=float(m.group(2)), top1=float(t.group(1)), top1_se=float(t.group(2))) if m and t else None


def peak(tag):
    p = os.path.join(D, tag + ".vram.csv")
    if not os.path.exists(p):
        return None
    v = [int(x) for x in open(p).read().split() if x.isdigit()]
    return max(v) if v else None


def runs():
    r = {}
    p = os.path.join(D, "run_lines.txt")
    for line in open(p) if os.path.exists(p) else []:
        m = re.match(r"(\S+): task=(\S+) n=(\d+) ngl=(\d+) .* peak_vram=(\d+) MiB wall=(\d+) s (.*)", line.strip())
        if m:
            r[m.group(1)] = dict(task=m.group(2), n=int(m.group(3)), ngl=int(m.group(4)), peak=int(m.group(5)), wall=int(m.group(6)), status=m.group(7))
    return r


def collect():
    res = {"ppl": {}, "kld": {}, "mc": {}}
    for k, tag in (("q2", "q2_ppl"), ("ptq1", "ptq1_ppl"), ("qwen", "qwen_kldbase")):
        v = ppl(log(tag))
        if v:
            res["ppl"][k] = v
    for k in ("q2", "ptq1"):
        v = kld(log(k + "_kld"))
        if v:
            res["kld"][k] = v
    for task, _, n in MC:
        res["mc"][task] = {"n": n}
        for k in ("q2", "ptq1", "qwen"):
            text = log("%s_%s" % (k, task))
            run = running(task, text)
            if n in run and (final_ok(task, text) or max(run) > n):
                a = run[n]
                full_n = max(run)
                res["mc"][task][k] = dict(acc=a, se=se(a, n), full_n=full_n, full_acc=run[full_n] if final_ok(task, text) else None)
        qi = per_item(task, log("qwen_" + task), n)
        bi = per_item(task, log("q2_" + task), n)
        if qi and bi:
            x, y, pv = mcnemar(qi, bi)
            res["mc"][task]["paired"] = dict(only_qwen=x, only_bonsai=y, p=pv)
    return res


def main():
    res = collect()
    rl = runs()
    json.dump(dict(results=res, runs=rl), open(os.path.join(D, "intel.json"), "w"), indent=1)
    o = ["# Intelligence evaluation: Ternary Bonsai 2 27B vs Qwen3.8-27B Q4_K_M (2026-11-17)", "",
         "Log-likelihood scoring with `llama-perplexity` (no sampling). Both models use the same tokenizer (vocabulary 248320), so perplexity and",
         "KL divergence are directly comparable. Scripts: `bench/bonsai_intel.sh` (one task), `bench/bonsai_intel_all.sh` and `bench/bonsai_intel_plan.sh` (the runs below);",
         "this file: `python3 bench/bonsai_intel_summary.py`. Datasets: WikiText-2 test, HellaSwag validation, Winogrande debiased, ARC-Challenge and MMLU validation",
         "(llama.cpp evaluation sets). Every run alone on the GPU, peak VRAM sampled every 100 ms, cap 7373 MiB.", "",
         "## 1. Language modelling (WikiText-2 test, 40 chunks x 512 tokens = 20,480 tokens)", "",
         "| Model | PPL ↓ | KL divergence vs Qwen Q4_K_M ↓ | Same top-1 token as Qwen ↑ |", "|---|---:|---:|---:|"]
    for k in ("qwen", "q2", "ptq1"):
        p = res["ppl"].get(k)
        kd = res["kld"].get(k)
        o.append("| %s | %s | %s | %s |" % (NAMES[k], "%.3f ± %.3f" % p if p else "-", "%.4f ± %.4f" % (kd["kld"], kd["kld_se"]) if kd else ("reference" if k == "qwen" else "-"),
                                             "%.2f ± %.2f %%" % (kd["top1"], kd["top1_se"]) if kd else ("reference" if k == "qwen" else "-")))
    o += ["", "The reference logits are Qwen3.8-27B **Q4_K_M** (the project baseline), not the FP16 model, so KL divergence measures how far Bonsai is from the",
          "baseline we run, not from the original weights.", "",
          "## 2. Task accuracy on matched subsets (accuracy %, ± binomial standard error)", "",
          "| Task | N | " + " | ".join(NAMES[k] for k in ("qwen", "q2", "ptq1")) + " | Bonsai `Q2_0` on the larger run |", "|---|---:|---:|---:|---:|---|"]
    for task, tn, n in MC:
        row = res["mc"][task]
        cells = []
        for k in ("qwen", "q2", "ptq1"):
            v = row.get(k)
            cells.append("%.1f ± %.1f" % (v["acc"], v["se"]) if v else "-")
        big = row.get("q2")
        bigs = "%.1f %% at N = %d" % (big["full_acc"], big["full_n"]) if big and big.get("full_acc") is not None and big["full_n"] > n else "-"
        o.append("| %s | %d | %s | %s |" % (tn, n, " | ".join(cells), bigs))
    o += ["", "Paired test on the same items (Qwen vs Bonsai `Q2_0`; per-item results reconstructed from the running accuracy in the logs, exact McNemar test):", "",
          "| Task | N | only Qwen correct | only Bonsai correct | p (two-sided) | Reading |", "|---|---:|---:|---:|---:|---|"]
    for task, tn, n in MC:
        pr = res["mc"][task].get("paired")
        if pr:
            o.append("| %s | %d | %d | %d | %.3f | %s |" % (tn, n, pr["only_qwen"], pr["only_bonsai"], pr["p"],
                     "Qwen better (p < 0.05)" if pr["p"] < 0.05 and pr["only_qwen"] > pr["only_bonsai"] else
                     ("Bonsai better (p < 0.05)" if pr["p"] < 0.05 else "no significant difference")))
    o += ["", "## 3. Runs", "", "| Run | Task | N | GPU layers | Peak VRAM MiB | Wall s | Status |", "|---|---|---:|---:|---:|---:|---|"]
    for tag in sorted(rl):
        r = rl[tag]
        o.append("| `%s` | %s | %d | %d | %d | %d | %s |" % (tag, r["task"], r["n"], r["ngl"], r["peak"], r["wall"], r["status"]))
    open(os.path.join(D, "INTEL.md"), "w").write("\n".join(o) + "\n")
    if "-q" not in sys.argv:
        print("\n".join(o))


if __name__ == "__main__":
    main()
