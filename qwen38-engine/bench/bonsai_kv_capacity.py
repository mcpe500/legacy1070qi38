#!/usr/bin/env python3
"""Exact memory budget of Ternary Bonsai 2 27B on the GTX 1070 Ti (VRAM cap 7373 MiB, RAM cap 14.2 GiB), all 64 layers on the GPU.
Inputs: the GGUF headers read by tests/tools/gguf_remote_header.py (bench/results/bonsai/header_*.json) and the fixed GPU overhead
of the runtime (CUDA context + compute buffers), measured with llama.cpp and passed as --fixed-mib (default: the measured value).
  python3 bench/bonsai_kv_capacity.py [--fixed-mib N] [--md OUT.md]
Model facts (from the header): 64 blocks, 16 full-attention layers (every 4th) with 4 KV heads x 256, 48 Gated-DeltaNet layers
(48 value heads, 128 x 128 FP32 state each, conv state 3 x 10240 FP32). Only the 16 attention layers have a KV cache that grows
with the context; the DeltaNet state is a constant 149.6 MiB per sequence.
llama.cpp keeps token_embd on the host (it is only gathered row-wise), so the GPU holds the 64 blocks plus output.weight."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAP_MIB = 7373.0
RAM_CAP_GIB = 14.2 / 1.073741824   # 14.2 GB (decimal) = 13.22 GiB
MIB = 1048576.0
args = sys.argv[1:]
FIXED = float(args[args.index("--fixed-mib") + 1]) if "--fixed-mib" in args else None
# Runtime overhead of llama.cpp (CUDA context + compute buffers) measured on this card, stock build (bench/bonsai_overhead.py):
# about 320 MiB at 4K context and +6.1 KiB per context token beyond that (590-630 MiB at 32K, 760 MiB at 64K, 1080 MiB at 128K).
OVH_BASE, OVH_PER_TOK = 295.0, 6.1 / 1024.0   # MiB, MiB per token
MD = args[args.index("--md") + 1] if "--md" in args else None

N_ATTN, N_KV_HEADS, HEAD_DIM, N_DN = 16, 4, 256, 48
ELEMS_PER_POS_LAYER = 2 * N_KV_HEADS * HEAD_DIM           # K and V
KV_BYTES = {   # bytes per position per attention layer (K + V), from the block sizes of each cache type
    "f16": ELEMS_PER_POS_LAYER * 2,
    "q8_0": ELEMS_PER_POS_LAYER // 32 * 34,
    "q4_0": ELEMS_PER_POS_LAYER // 32 * 18,
    "iso3": ELEMS_PER_POS_LAYER // 128 * 50,
}
DN_STATE = N_DN * (48 * 128 * 128 * 4 + 3 * 10240 * 4)    # bytes, per sequence
CTXS = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
SMEM = 48 * 1024                                            # bytes of shared memory per block on sm_61


def load(name):
    p = os.path.join(ROOT, "bench", "results", "bonsai", "header_%s.json" % name)
    j = json.load(open(p))
    t = j["tensors"]
    blocks = sum(x["nbytes"] for x in t if x["name"].startswith("blk."))
    out = sum(x["nbytes"] for x in t if x["name"] == "output.weight")
    emb = sum(x["nbytes"] for x in t if x["name"] == "token_embd.weight")
    small = sum(x["nbytes"] for x in t if not x["name"].startswith("blk.") and x["name"] not in ("output.weight", "token_embd.weight"))
    return {"file": j["size"], "blocks": blocks, "output": out, "embd": emb, "small": small}


def main():
    packs = [p for p in ("Q2_0", "PQ2_0", "PTQ1_0") if os.path.exists(os.path.join(ROOT, "bench", "results", "bonsai", "header_%s.json" % p))]
    def overhead(ctx):
        return FIXED if FIXED is not None else OVH_BASE + OVH_PER_TOK * ctx
    L = []
    L.append("### Weight footprint (from the GGUF headers)\n")
    L.append("| Packing | file GB | 64 blocks MiB | output.weight MiB | token_embd MiB (host) | GPU weights MiB | left under %d MiB before KV, state and runtime |" % CAP_MIB)
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    w = {}
    for p in packs:
        d = load(p)
        g = (d["blocks"] + d["output"] + d["small"]) / MIB
        w[p] = g
        L.append("| %s | %.2f | %.1f | %.1f | %.1f | %.1f | %.1f |" % (p, d["file"] / 1e9, d["blocks"] / MIB, d["output"] / MIB, d["embd"] / MIB, g, CAP_MIB - g))
    L.append("")
    L.append("### KV cache size (16 attention layers; DeltaNet state is constant %.1f MiB)\n" % (DN_STATE / MIB))
    L.append("| KV type | bytes / token (16 layers) | " + " | ".join("%dK" % (c // 1024) for c in CTXS) + " |")
    L.append("|---|---:|" + "---:|" * len(CTXS))
    for k, b in KV_BYTES.items():
        L.append("| %s | %d | " % (k, b * N_ATTN) + " | ".join("%.0f MiB" % (b * N_ATTN * c / MIB) for c in CTXS) + " |")
    L.append("")
    L.append("### Largest context with every layer on the GPU (runtime overhead %s)\n" %
             ("fixed at %.0f MiB" % FIXED if FIXED is not None else "from the measured model: %.0f MiB + %.1f KiB per token" % (OVH_BASE, OVH_PER_TOK * 1024)))
    L.append("| Packing | KV type | VRAM left at 0 tokens MiB | max context (tokens) | " + " | ".join("%dK" % (c // 1024) for c in CTXS) + " |")
    L.append("|---|---|---:|---:|" + "---|" * len(CTXS))
    for p in packs:
        base = CAP_MIB - w[p] - DN_STATE / MIB
        for k, b in KV_BYTES.items():
            per = b * N_ATTN / MIB
            fits = lambda c: base - overhead(c) - per * c >= 0
            mx = 0
            lo, hi = 0, 262144
            while lo < hi:   # largest c with fits(c)
                mid = (lo + hi + 1) // 2
                if fits(mid): lo = mid
                else: hi = mid - 1
            mx = lo
            L.append("| %s | %s | %.0f | %s | " % (p, k, base - overhead(0), "%d" % mx if mx > 0 else "does not fit") +
                     " | ".join("yes" if c <= mx else "no" for c in CTXS) + " |")
    L.append("")
    L.append("### Shared memory (48 KiB per block on sm_61)\n")
    L.append("- qwen38-engine keeps the attention scores of one query row in dynamic shared memory (4 bytes per position plus about 1.1 KiB): "
             "the kernel can address at most %d positions; the engine is limited to 4096 on purpose." % ((SMEM - 1152) // 4))
    L.append("- llama.cpp's flash-attention kernels tile the KV sequence, so shared memory does not limit the context there.")
    L.append("")
    L.append("### RAM (cap 14.2 GB = %.2f GiB)\n" % RAM_CAP_GIB)
    for p in packs:
        d = load(p)
        L.append("- %s: the mapped file is %.2f GiB; with every layer on the GPU only token_embd (%.0f MiB) and host buffers stay resident, far under the cap." %
                 (p, d["file"] / 1073741824.0, d["embd"] / MIB))
    txt = "\n".join(L) + "\n"
    print(txt)
    if MD:
        open(MD, "w").write(txt)


if __name__ == "__main__":
    main()
