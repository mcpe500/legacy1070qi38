#!/usr/bin/env python3
"""Splits every measured llama-server load VRAM of the Bonsai runs into weights + KV + DeltaNet state + runtime overhead,
using the exact tensor sizes from the GGUF headers. With -ngl N llama.cpp puts the last N blocks on the GPU (blocks 64-N .. 63)
and output.weight only when N > 64; token_embd stays on the host.
  python3 bench/bonsai_overhead.py [--md OUT.md]"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "bench", "results", "bonsai")
MIB = 1048576.0
KV = {"f16": 4096, "q8_0": 2176, "q4_0": 1152, "iso3": 800}   # bytes per position per attention layer (K + V)
DN = 48 * 128 * 128 * 4 + 3 * 10240 * 4                        # bytes of recurrent state per DeltaNet layer

RUNS = [  # tag, packing, runtime, kv, ctx, ngl, ub
    ("srv_q2_f16_c4096_ngl99", "Q2_0", "stock", "f16", 4096, 99, 512),
    ("srv_q2_q8_0_c32768_ngl59", "Q2_0", "stock", "q8_0", 32768, 59, 128),
    ("srv_q2_q4_0_c32768_ngl59", "Q2_0", "stock", "q4_0", 32768, 59, 128),
    ("srv_q2_iso3_c32768_ngl59", "Q2_0", "stock", "iso3", 32768, 59, 128),
    ("srv_q2_iso3_c65536_ngl56_full", "Q2_0", "stock", "iso3", 65536, 56, 128),
    ("srv_q2_iso3_c131072_ngl46", "Q2_0", "stock", "iso3", 131072, 46, 128),
    ("srv_q2_iso3_c131072_ngl47", "Q2_0", "stock", "iso3", 131072, 47, 128),
    ("srv_q2_iso3_c131072_ngl48", "Q2_0", "stock", "iso3", 131072, 48, 128),
    ("srv_q2_iso3_c131072_ngl51", "Q2_0", "stock", "iso3", 131072, 51, 128),
    ("srv_q2_iso3_c131072_ngl52", "Q2_0", "stock", "iso3", 131072, 52, 128),
    ("srv_q2_q4_0_c131072_ngl43", "Q2_0", "stock", "q4_0", 131072, 43, 128),
    ("srv_q2_q4_0_c131072_ngl48", "Q2_0", "stock", "q4_0", 131072, 48, 128),
    ("srv_ptq1_f16_c16384_ngl99_ub128", "PTQ1_0", "prism", "f16", 16384, 99, 128),
    ("srv_ptq1_q8_0_c32768_ngl99_ub128", "PTQ1_0", "prism", "q8_0", 32768, 99, 128),
    ("srv_ptq1_q4_0_c65536_ngl99_ub128", "PTQ1_0", "prism", "q4_0", 65536, 99, 128),
    ("srv_ptq1_q4_0_c65536_ngl99_ub512", "PTQ1_0", "prism", "q4_0", 65536, 99, 512),
    ("srv_ptq1_q4_0_c65536_ngl64_ub128", "PTQ1_0", "prism", "q4_0", 65536, 64, 128),
    ("srv_ptq1_q4_0_c65536_ngl62_ub512", "PTQ1_0", "prism", "q4_0", 65536, 62, 512),
    ("srv_ptq1_q4_0_c131072_ngl48_ub512", "PTQ1_0", "prism", "q4_0", 131072, 48, 512),
]


def sizes(pack):
    t = json.load(open(os.path.join(R, "header_%s.json" % pack)))["tensors"]
    blk = [0] * 64
    for x in t:
        if x["name"].startswith("blk."):
            blk[int(x["name"].split(".")[1])] += x["nbytes"]
    out = sum(x["nbytes"] for x in t if x["name"] in ("output.weight", "output_norm.weight"))
    return blk, out


def summary(tag):
    lv = pv = None
    for line in open(os.path.join(R, tag + ".vram.csv")).read().split():
        if line.isdigit():
            pv = max(pv or 0, int(line))
    # load VRAM: the value printed by the bench script right after /health, recorded in the .log name order -> recompute from csv start
    return pv


def main():
    md = sys.argv[sys.argv.index("--md") + 1] if "--md" in sys.argv else None
    L = ["| run | packing / runtime | KV | ctx | -ngl | -ub | GPU weights MiB | KV MiB | state MiB | peak VRAM MiB | runtime overhead MiB (peak - parts) |",
         "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for tag, pack, rt, kv, ctx, ngl, ub in RUNS:
        p = os.path.join(R, tag + ".vram.csv")
        if not os.path.exists(p):
            continue
        blk, out = sizes(pack)
        n = min(ngl, 64)
        layers = list(range(64 - n, 64))
        w = sum(blk[i] for i in layers) + (out if ngl > 64 else 0)
        attn = [i for i in layers if i % 4 == 3]
        kvb = len(attn) * KV[kv] * ctx
        st = (len(layers) - len(attn)) * DN
        peak = summary(tag)
        L.append("| %s | %s / %s | %s | %d | %d | %d | %.0f | %.0f | %.0f | %d | %.0f |" % (
            tag, pack, rt, kv, ctx, ngl, ub, w / MIB, kvb / MIB, st / MIB, peak, peak - (w + kvb + st) / MIB))
    txt = "\n".join(L) + "\n"
    print(txt)
    if md:
        open(md, "w").write(txt)


if __name__ == "__main__":
    main()
