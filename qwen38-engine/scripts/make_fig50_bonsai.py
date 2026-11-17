#!/usr/bin/env python3
"""Draws docs/figures/50_ternary_bonsai.svg (style of 48_cascade.svg).
Every number comes from bench/results/bonsai/RUNS.md and bench/results/ternary_bonsai_comparison.md.
Usage: python3 scripts/make_fig50_bonsai.py   (with cairosvg installed also writes /tmp/fig_50_ternary_bonsai.png)"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_pr_figures import C, Svg  # noqa: E402

W, H = 1000, 720
CAP = 7373
CTX = [512, 4096, 16384, 32768, 65536, 131072]
CTX_LABEL = ["512", "4K", "16K", "32K", "64K", "128K"]
# tg tok/s, peak VRAM MiB, GPU layers; the 128K point is speed at a 487-token depth (capacity check)
PTQ1 = dict(tg=[19.64, 15.19, 11.70, 7.51, 3.36, 2.24], vram=[6579, 6633, 6861, 7165, 7361, 7303], ngl=[99, 99, 99, 99, 62, 48])
Q2 = dict(tg=[17.13, 12.43, 7.20, 3.57, 2.04, 3.98], vram=[7371, 7337, 7339, 7355, 7351, 7327], ngl=[99, 64, 62, 59, 56, 48])
QWEN_ENGINE = [2.947, 2.245]          # qwen38-engine main, MTP 2, 509- and 3924-token prompts (limit 4096)
QWEN_LLAMA = [2.26, 2.00, 1.35]       # stock llama.cpp, q4_0 KV, -ngl 25/25/24
PURPLE, ORANGE, GREY = C["mtp"], C["pcie"], C["llama"]


def head(s):
    s.parts.pop()  # Svg() draws a 22 px title; this figure uses the 20 px title + subtitle of 48_cascade
    s.text(24, 38, "Qwen3.8-27B vs Ternary Bonsai 2 27B: arsitektur, kernel ternary, kapasitas KV", size=20, weight="700")
    s.text(24, 62, "exp/ternary-bonsai-27b-eval · GTX 1070 Ti sm_61 + Ryzen 5 2400G · rinci: bench/results/ternary_bonsai_comparison.md",
           size=14, color=C["dim"])


def layer_bar(s, x, y, w, gpu, total, gpu_txt, cpu_txt):
    gw = w * gpu / total
    s.rect(x, y, gw, 22, C["gpu_bg"], C["gpu"], rx=4, sw=1)
    s.text(x + gw / 2, y + 15, gpu_txt, size=11, weight="700", color=C["gpu"], anchor="middle")
    if gpu < total:
        s.rect(x + gw, y, w - gw, 22, C["cpu_bg"], C["cpu"], rx=4, sw=1)
        s.text(x + gw + (w - gw) / 2, y + 15, cpu_txt, size=11, weight="700", color=C["cpu"], anchor="middle")


def panel_arch(s):
    x0, y0, pw, ph = 24, 88, 452, 612
    s.rect(x0, y0, pw, ph, C["soft"], rx=6)
    s.text(x0 + 16, y0 + 28, "A. Model dan penempatan layer (konteks pendek)", size=14, weight="700")
    # Qwen
    y = y0 + 44
    s.rect(x0 + 16, y, pw - 32, 96, "#ffffff", C["line"], rx=5, sw=1)
    s.text(x0 + 28, y + 20, "Qwen3.8-27B-UD-Q4_K_M · 16.56 GB · 4.9 bpw · qwen38-engine", size=12, weight="700")
    s.text(x0 + 28, y + 37, "64 blok hybrid (48 Gated DeltaNet + 16 GQA) + blok MTP", size=12, color=C["dim"])
    layer_bar(s, x0 + 28, y + 46, pw - 56, 32, 64, "32 layer GPU (VRAM)", "32 layer CPU (DDR4)")
    s.text(x0 + 28, y + 86, "tg 2.947 tok/s (MTP 2) · 3.130 adaptif · 7337 MiB", size=12, weight="700", color=C["cpu"])
    # Bonsai
    y = y0 + 150
    s.rect(x0 + 16, y, pw - 32, 96, "#ffffff", C["line"], rx=5, sw=1)
    s.text(x0 + 28, y + 20, "Ternary Bonsai 2 27B PTQ1_0 · 5.95 GB · 1.75 bpw {-1, 0, +1}", size=12, weight="700")
    s.text(x0 + 28, y + 37, "arsitektur sama (qwen35), tanpa MTP · basis rotasi prism.hadamard", size=12, color=C["dim"])
    layer_bar(s, x0 + 28, y + 46, pw - 56, 64, 64, "64 / 64 layer di GPU · bobot 5395 MiB", "")
    s.text(x0 + 28, y + 86, "tg 19.64 tok/s (6.7x) · 6579 MiB · PrismML fork (PTX JIT)", size=12, weight="700", color=C["gpu"])

    # kernel pipeline
    y = y0 + 262
    s.text(x0 + 16, y + 6, "Kernel ternary qwen38-engine: ternary_gemv.cu (Q2_0)", size=13, weight="700")
    cx, bw = x0 + pw / 2, 380
    steps = [
        ("#eaeef2", C["ink"], "x (FP32, satu baris aktivasi)", None),
        (C["warn_bg"], C["warn"], "1) Rotasi Hadamard FP32", "x' = H_1024 · (s ⊙ x) / 32, butterfly di shared memory"),
        (C["cpu_bg"], C["cpu"], "2) Kuantisasi Q8_1", "int8 per 32 nilai + skala dx, jumlah sx"),
        (C["mtp_bg"], PURPLE, "3) lop3.b32 unpack + prmt.b32 gather", "16 kode 2-bit → 4 word int8x4 {k, k+4, k+8, k+12}"),
        (C["gpu_bg"], C["gpu"], "4) __dp4a + akumulasi FP32", "per blok: d · (dx · Σ code·q − sx)"),
    ]
    yy = y + 18
    for i, (bg, fg, t, sub) in enumerate(steps):
        h = 26 if sub is None else 40
        s.rect(cx - bw / 2, yy, bw, h, bg, rx=5)
        s.text(cx, yy + 17, t, size=12, weight="700", color=fg, anchor="middle")
        if sub:
            s.text(cx, yy + 33, sub, size=12, color=fg, anchor="middle")
        yy += h
        if i < len(steps) - 1:
            s.line(cx, yy, cx, yy + 10, color=C["dim"], arrow=True)
            yy += 12
    # verification
    yy += 10
    s.rect(x0 + 16, yy, pw - 32, 92, "#ffffff", C["gpu"], rx=5, sw=1.5)
    s.text(cx, yy + 22, "cosine 1.000000 vs llama.cpp", size=15, weight="700", color=C["gpu"], anchor="middle")
    s.text(cx, yy + 40, "bobot asli Bonsai, layer 0: ffn_up · ffn_gate · ffn_down", size=12, anchor="middle")
    s.text(cx, yy + 58, "SASS: IDP.4A ×464 · LOP3 ×112 · PRMT ×976 · aritmetika half ×0", size=12, anchor="middle")
    s.text(cx, yy + 76, "ternary_test 15/15 · belum di-wire ke forward pass engine", size=12, weight="700", color=C["warn"], anchor="middle")


def panel_curve(s):
    x0, y0, pw, ph = 500, 88, 476, 612
    s.rect(x0, y0, pw, ph, C["soft"], rx=6)
    s.text(x0 + 16, y0 + 28, "B. Decode vs konteks, konfigurasi terbaik ≤ 7373 MiB", size=14, weight="700")
    gx, gw = x0 + 58, pw - 86
    lx = [math.log2(c) for c in CTX]

    def X(i):
        return gx + (lx[i] - lx[0]) / (lx[-1] - lx[0]) * gw

    # --- throughput chart
    gy, gh, ymax = y0 + 64, 250, 24.0

    def Y(v):
        return gy + gh - v / ymax * gh

    for v in range(0, 21, 5):
        s.line(gx, Y(v), gx + gw, Y(v), color=C["line"], sw=1)
        s.text(gx - 8, Y(v) + 4, str(v), size=11, color=C["dim"], anchor="end")
    s.text(gx - 40, gy - 12, "tg tok/s", size=11, weight="700", color=C["dim"])
    for i, lab in enumerate(CTX_LABEL):
        s.line(X(i), gy + gh, X(i), gy + gh + 4, color=C["dim"], sw=1)
        s.text(X(i), gy + gh + 17, lab, size=11, color=C["dim"], anchor="middle")
    # 4K engine limit
    s.line(X(1) + 6, gy, X(1) + 6, gy + gh, color=C["dim"], sw=1, dash="3 3")
    s.text(X(1) + 2, Y(22.3), "batas engine Qwen 4096", size=10, color=C["dim"], anchor="end")

    def series(vals, color, label_last_hollow, dash=None):
        pts = [(X(i), Y(v)) for i, v in enumerate(vals)]
        solid = pts[:-1] if label_last_hollow else pts
        s.path("M" + " L".join("%.1f,%.1f" % p for p in solid), color=color, sw=2.2, dash=dash)
        if label_last_hollow:
            a, b = pts[-2], pts[-1]
            s.path("M%.1f,%.1f L%.1f,%.1f" % (a + b), color=color, sw=1.5, dash="4 3")
        for j, (px, py) in enumerate(pts):
            hollow = label_last_hollow and j == len(pts) - 1
            s.parts.append('<circle cx="%.1f" cy="%.1f" r="4" fill="%s" stroke="%s" stroke-width="1.8"/>'
                           % (px, py, "#ffffff" if hollow else color, color))

    series(PTQ1["tg"], C["gpu"], True)
    series(Q2["tg"], PURPLE, True)
    series(QWEN_LLAMA, GREY, False, dash="5 3")
    series(QWEN_ENGINE, C["cpu"], False)
    for i, v in enumerate(PTQ1["tg"][:5]):
        s.text(X(i) + (6 if i else 8), Y(v) - 8, "%.2f" % v, size=11, weight="700", color=C["gpu"], anchor="start" if i < 4 else "middle")
    for i, v in enumerate(Q2["tg"][:4]):
        s.text(X(i) - 6, Y(v) + 16, "%.2f" % v, size=11, weight="700", color=PURPLE, anchor="end" if i else "start")
    s.text(X(4) + 2, Y(Q2["tg"][4]) + 16, "2.04", size=11, weight="700", color=PURPLE, anchor="middle")
    s.text(X(0) + 8, Y(QWEN_ENGINE[0]) - 7, "2.95", size=11, weight="700", color=C["cpu"])
    # legend
    lgx, lgy = gx + gw - 168, gy + 4
    s.rect(lgx - 8, lgy - 4, 176, 84, "#ffffff", C["line"], rx=4, sw=1)
    for k, (col, txt, dash) in enumerate([(C["gpu"], "Bonsai PTQ1_0, fork", None), (PURPLE, "Bonsai Q2_0, stock", None),
                                          (C["cpu"], "Qwen, qwen38-engine", None), (GREY, "Qwen, llama.cpp", "5 3")]):
        yy = lgy + 10 + k * 18
        s.line(lgx, yy, lgx + 22, yy, color=col, sw=2.2, dash=dash)
        s.text(lgx + 28, yy + 4, txt, size=11)
    s.text(gx - 40, gy + gh + 34, "Marker kosong 128K: kecepatan pada kedalaman 487 token (cek kapasitas);", size=10, color=C["dim"])
    s.text(gx - 40, gy + gh + 47, "prefill 128K penuh (~1 jam) tidak dijalankan. Turun di 64K+: layer pindah ke CPU.", size=10, color=C["dim"])

    # --- VRAM chart
    vy, vh, vmin, vmax = gy + gh + 92, 140, 6500, 7500

    def V(v):
        return vy + vh - (v - vmin) / (vmax - vmin) * vh

    s.text(x0 + 16, vy - 14, "Peak VRAM per titik (MiB) dan GPU layer (-ngl)", size=13, weight="700")
    for v in (6600, 6800, 7000, 7200, 7400):
        s.line(gx, V(v), gx + gw, V(v), color=C["line"], sw=1)
        s.text(gx - 8, V(v) + 4, str(v), size=11, color=C["dim"], anchor="end")
    s.line(gx, V(CAP), gx + gw, V(CAP), color=C["bad"], sw=1.8, dash="6 4")
    s.text(gx + 4, V(CAP) - 5, "cap 7373 MiB", size=11, weight="700", color=C["bad"])
    for i, lab in enumerate(CTX_LABEL):
        s.text(X(i), vy + vh + 16, lab, size=11, color=C["dim"], anchor="middle")
    for d, col, off in ((PTQ1, C["gpu"], 16), (Q2, PURPLE, -9)):
        s.path("M" + " L".join("%.1f,%.1f" % (X(i), V(v)) for i, v in enumerate(d["vram"])), color=col, sw=2)
        for i, v in enumerate(d["vram"]):
            s.parts.append('<circle cx="%.1f" cy="%.1f" r="3.5" fill="%s"/>' % (X(i), V(v), col))
    pos = [("start", 4, 16), ("middle", 0, 16), ("middle", 0, 16), ("end", -7, 4), ("end", -7, 15), ("end", 0, 16)]
    for i, (v, n) in enumerate(zip(PTQ1["vram"], PTQ1["ngl"])):
        lab = "%d" % v + ("" if n == 99 else " · ngl %d" % n)
        anc, dx, dy = pos[i]
        s.text(X(i) + dx, V(v) + dy, lab, size=10, weight="700", color=C["gpu"], anchor=anc)
    s.text(gx - 40, vy + vh + 32, "Hijau PTQ1_0: semua 64 layer di GPU s.d. 32K, lalu ngl 62 (64K) dan 48 (128K).", size=10, color=C["gpu"])
    s.text(gx - 40, vy + vh + 45, "Ungu Q2_0: 7327-7371 MiB di semua titik; ngl 99 / 64 / 62 / 59 / 56 / 48.", size=10, color=PURPLE)


def main():
    s = Svg(W, H, "")
    head(s)
    panel_arch(s)
    panel_curve(s)
    s.save("50_ternary_bonsai.svg")


if __name__ == "__main__":
    main()
