#!/usr/bin/env python3
"""Draws docs/figures/66_dump_targets_fix.svg (style of the make_pr_figures family).

Panel A: measured per-sample cost breakdown of the fixed --dump-targets pass (GTX 1070 Ti, sm_61 dp4a
         path, seq 128, from the smoke logs /tmp/opencode/tg_smoke/dump_fix2.log quoted in PR #66:
         prefill 3763 ms, head 553 ms (~1177 GFLOPS), quant 7 ms, io 4 ms for a 2-sample shard).
Panel B: what the fix removes from the T4 head path (C-109): the per-call full LM head dequant (128x per
         sample), the 1.02 GB FP32 logits D2H per sample and the 254M exp + heap on 2 vCPUs -> GPU top-K
         with ~0.4 MB/sample of ids/vals/lse.
Panel C: the first T4 run that exposed the bug: 6.0 tok/s, 86.8% of the vocabulary corrupted by the beta=1
         chunk accumulation, fake descending ETA; after-fix T4 numbers are pending the user's re-run.
Usage: python3 scripts/make_fig66_dump_targets.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_pr_figures import C, Svg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H = 1100, 1040

V = 248320
K = 64
L = 1024
D2H_OLD = L * V * 4                      # 1,018,357,760 B
D2H_NEW = L * (K * 6 + 2)                # 394,240 B
CORRUPT_PCT = (V - 32768) / V * 100.0    # 86.79%


def gb(b):
    return b / 1e9


s = Svg(W, H, "dump-targets fix (PR #66 / C-109)")

# ---- Panel A: measured per-sample breakdown (local sm_61, seq 128) -------------------------------
s.rect(24, 64, W - 48, 300, C["soft"], C["line"])
s.text(44, 96, "A. measured per-sample breakdown after the fix", size=16, weight="700", color=C["ink"])
s.text(44, 118, "GTX 1070 Ti, sm_61 dp4a path, seq 128, shard = 2 samples (logs of PR #66); T4 fp16 path pending re-run",
       size=12, color=C["dim"])
comps = [("prefill", 1881.5, C["cpu"]), ("head ~1177 GFLOPS", 276.5, C["gpu"]), ("quant (CPU, inline)", 3.5, C["llama"]),
         ("io (npy shard)", 2.0, C["pcie"])]
x0, y0, bw, bh, gap = 230, 140, 620, 36, 14
mx = 1900.0
for i, (name, ms, col) in enumerate(comps):
    y = y0 + i * (bh + gap)
    w = max(4.0, bw * ms / mx)
    s.rect(x0, y, w, bh, col, stroke="none", rx=4)
    s.text(x0 - 12, y + 24, name, size=13, anchor="end", color=C["ink"])
    s.text(x0 + w + 10, y + 24, "%.1f ms" % ms, size=13, color=C["dim"])
s.text(44, y0 + 4 * (bh + gap) + 18, "per 2-sample shard: prefill ~3763 ms, head ~553 ms (~1177 GFLOPS), quant ~7 ms, io ~4 ms",
       size=12.5, color=C["dim"])
s.text(44, y0 + 4 * (bh + gap) + 42, "smoke 4 samples x 128 tok: 59-60 tok/s (52 before this PR); parity cosine 1.000000, |dlse| 0.0205",
       size=12.5, color=C["gpu"])
s.text(44, y0 + 4 * (bh + gap) + 64, "the shard log now prints this breakdown + an honest ETA (remaining tokens / measured rate)",
       size=12.5, color=C["dim"])

# ---- Panel B: removed from the head path (per 1024-token sample, T4) ------------------------------
yB = 388
s.rect(24, yB, W - 48, 300, C["gpu_bg"], C["line"])
s.text(44, yB + 32, "B. removed from the T4 head path (before -> after, per 1024-token sample)", size=16,
       weight="700", color=C["ink"])
rows = [
    ("LM head dequant", "full 248320-row dequant inside every head call, 128 calls/sample (~5.2 GB traffic each)",
     "dequantized ONCE into a budget-guarded FP16 cache (2.44 GiB)"),
    ("logits D2H", "%.2f GB FP32 logits to RAM, then 254M exp + heap on 2 vCPUs" % gb(D2H_OLD),
     "GPU top-K/lse kernel: %s (%.2f MB) ids/vals/lse, descending, ties -> lower id" % (f"{D2H_NEW:,} B", D2H_NEW / 1e6)),
    ("top-K/lse", "std::exp double x 254M + min-heap on 2 Colab vCPUs, threads re-spawned per chunk",
     "one 256-thread block per row in VRAM (arch-independent, runs on sm_61 too)"),
    ("beta accumulation", "chunks 2..N of the 32768-row GEMM added stale buffer contents (86.8% of vocab corrupted)",
     "beta = 0 unconditionally; unit test: dirty buffer + bit-identical idempotency + FP64 parity"),
]
yy = yB + 56
for name, before, after in rows:
    s.text(44, yy, name, size=13.5, weight="700", color=C["ink"])
    s.text(44, yy + 20, "before: " + before, size=12, color=C["bad"])
    s.text(44, yy + 38, "after:  " + after, size=12, color=C["gpu"])
    yy += 58
s.text(44, yB + 288, "D2H per sample: %.2f GB -> %.2f MB (%.0fx smaller)" % (gb(D2H_OLD), D2H_NEW / 1e6, D2H_OLD / D2H_NEW),
       size=12.5, weight="700", color=C["gpu"])

# ---- Panel C: the first T4 run that exposed it (C-109) --------------------------------------------
yC = 712
s.rect(24, yC, W - 48, 300, C["bad_bg"], C["line"])
s.text(44, yC + 32, "C. the run that exposed it (Colab T4, pre-fix) and what is still pending", size=16,
       weight="700", color=C["ink"])
s.text(44, yC + 58, "6.0 tok/s at seq 1024 (170.7 s/sample), 18 shards x 128 samples; fake ETA hid a ~4.5 h shard as '0.3 min'",
       size=12.5, color=C["ink"])
# corrupted vocabulary bar
bx, by, bwd, bhh = 44, yC + 78, W - 88, 34
ok_w = bwd * (32768 / V)
s.rect(bx, by, ok_w, bhh, C["gpu"], stroke="none", rx=4)
s.rect(bx + ok_w, by, bwd - ok_w, bhh, C["bad"], stroke="none", rx=0)
s.text(bx + ok_w / 2, by + 22, "13.2% clean", size=12, anchor="middle", color="#ffffff", weight="700")
s.text(bx, by + 58, "rows 32768..248319 (%.1f%% of the vocabulary) accumulated stale GEMM output:"
                    " beta=1 on disjoint chunks" % CORRUPT_PCT,
       size=12, color=C["dim"])
s.text(44, by + 80, "root causes fixed: beta=0 unconditional (unit test: dirty buffer, idempotency, FP64 parity);"
                    " dangling --dump-hidden-dtype c_str()",
       size=12, color=C["dim"])
s.text(44, by + 102, "root causes fixed (Colab): hardcoded CMAKE_CUDA_ARCHITECTURES 61 vs CUDA 13;"
                     " llama/llama-common targets, cmake --build, hf_hub_download",
       size=12, color=C["dim"])
s.text(44, by + 130, "T4 throughput AFTER the fix: pending the user's re-run - the per-shard breakdown line"
                     " (head GFLOPS vs ~65 TFLOPS fp16 roofline) is the instrument",
       size=12.5, weight="700", color=C["warn"])
s.text(44, by + 152, "delete /content/drive/MyDrive/bonsai_mtp/targets before re-running: its shards carry corrupt"
                     " top-64 targets and auto-resume would keep them",
       size=12, color=C["warn"])

s.save("66_dump_targets_fix.svg")
