#!/usr/bin/env python3
"""Long-prompt variant of the M1 suites (issue #55): one prefill per suite with --state-save, then every decode variant
from --state-load (the 32K prefill costs minutes; the variants only differ in the decode).

The prompt is the suite JSON (bench/bonsai_mtp_suite.py prompts) rendered as ChatML for the CLI (--parse-special). Literal
"<|" in the document text is written as "< |" so source files that mention the chat markup cannot become special tokens.
The state covers everything up to the end of the question; the continuation is the turn end plus the assistant header, so
the token sequence is the one of the full prompt.

Usage: bench/bonsai_suite_state.py SIZE SUITE[,SUITE] HEAD VARIANT...   (HEAD: none | donor | q4km | q8; VARIANT as in
bonsai_mtp_suite.py). Runs through bench/bonsai_ctx_run.sh (VRAM sampling, refuses a busy GPU), OUT_DIR bonsai_local_10h.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "bench", "results", "bonsai_local_10h")
sys.path.insert(0, os.path.join(ROOT, "bench"))
import bonsai_mtp_suite as S  # noqa: E402

TAIL = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
STATE_DIR = os.environ.get("STATE_DIR", "/tmp/b10h_states")


def render_prefix(msgs):
    s = ""
    for i, m in enumerate(msgs):
        c = m["content"].strip().replace("<|", "< |")
        s += f"<|im_start|>{m['role']}\n{c}"
        if i + 1 < len(msgs):
            s += "<|im_end|>\n"
    return s


def run(tag, args):
    env = dict(os.environ, OUT_DIR=OUT)
    rc = subprocess.run(["bash", os.path.join(ROOT, "bench", "bonsai_ctx_run.sh"), tag] + args, env=env).returncode
    rows = [json.loads(l) for l in open(os.path.join(OUT, tag + ".txt"), encoding="utf-8", errors="replace") if l.startswith("{")]
    return rc, rows


def main():
    size, suites, head, variants = sys.argv[1], sys.argv[2].split(","), sys.argv[3], sys.argv[4:]
    os.makedirs(STATE_DIR, exist_ok=True)
    n = int(size)
    ctx = str(n + 512)
    base = ["-m", S.MODEL, "--gpu-layers", "auto", "--ctx", ctx, "--parse-special"]
    if S.HEADS[head]:
        base += ["--mtp-gguf", os.path.join(S.SIDECAR_DIR, S.HEADS[head]), "--mtp-steps", "3"]
    else:
        base += ["--mtp-steps", "0"]
    cont = os.path.join(STATE_DIR, "cont.txt")
    open(cont, "w").write(TAIL)
    for suite in suites:
        p = json.load(open(os.path.join(S.PROMPTS, f"{suite}_{size}.json"), encoding="utf-8"))
        pre = os.path.join(STATE_DIR, f"{suite}_{size}.prefix.txt")
        open(pre, "w", encoding="utf-8").write(render_prefix(p["messages"]))
        state = os.path.join(STATE_DIR, f"{suite}_{size}_{head}.state")
        tag = f"st_{size}_{suite}_{head}_prefill"
        rc, rows = run(tag, base + ["-f", pre, "--state-save", state, "--gen-mtp-steps", "0" if head == "none" else "3",
                                    "--mtp-adaptive", "0"])
        info = rows[0] if rows else {}
        print(f"[{tag}] rc {rc} tokens {info.get('tokens')} prefill {info.get('prefill_s')} s ({info.get('prefill_tok_s')} tok/s) "
              f"state {info.get('state_mib')} MiB, peak VRAM {rows[-1].get('peak_vram_mib') if rows else '?'}", flush=True)
        if rc != 0:
            continue
        for v in variants:
            c = S.VARIANTS[v]
            tag = f"st_{size}_{suite}_{head}_{v}"
            rc, rows = run(tag, base + ["--state-load", state, "-f", cont, "-n", os.environ.get("N_PREDICT", "256"),
                                        "--gen-mtp-steps", str(c["mtp_steps"]), "--mtp-adaptive", "1" if c["mtp_adaptive"] else "0",
                                        "--pld", "1" if c["pld"] else "0"])
            d = next((r for r in rows if "ids" in r), {})
            print(f"[{tag}] rc {rc} decode {d.get('decode_tok_s')} tok/s accept {d.get('draft_n_accepted')}/{d.get('draft_n')} "
                  f"pld_hits {d.get('pld_hits')} state_load {d.get('state_load_s')} s, peak VRAM {rows[-1].get('peak_vram_mib') if rows else '?'}",
                  flush=True)
        os.remove(state)


if __name__ == "__main__":
    main()
