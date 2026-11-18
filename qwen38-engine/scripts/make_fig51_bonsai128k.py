#!/usr/bin/env python3
"""Draws docs/figures/51_bonsai_128k.svg (style of 50_ternary_bonsai.svg).
Every number is read from bench/results/bonsai128k/ (JSON lines of the engine and of bench/bonsai_ctx_run.sh,
attn_long_test.txt). A run that has not finished yet is drawn as "pending".
Usage: python3 scripts/make_fig51_bonsai128k.py"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_pr_figures import C, Svg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "bench", "results", "bonsai128k")
W, H = 1000, 1080
CAP = 7373
CTX = [4096, 32768, 65536, 131072]
CTX_LABEL = ["4K", "32K", "64K", "128K"]


def jlines(tag):
    p = os.path.join(RES, tag + ".txt")
    out = {}
    if not os.path.exists(p):
        return None
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line.startswith("{"):
            try:
                out.update(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def depth(model, ctx):
    return jlines("%s_ctx%d_depth" % (model, ctx))


def head(s):
    s.parts.pop()
    s.text(24, 38, "Ternary Bonsai 2 27B pada konteks 128K: iso3 KV, attention tiled, state save/load", size=20, weight="700")
    s.text(24, 62, "exp/bonsai-128k-context · GTX 1070 Ti sm_61 (cap 7373 MiB) + Ryzen 5 2400G (cap RAM 14.2 GB) · data: bench/results/bonsai128k/",
           size=13, color=C["dim"])


def panel_vram(s):
    x0, y0, pw, ph = 24, 84, 952, 330
    s.rect(x0, y0, pw, ph, C["soft"], rx=6)
    s.text(x0 + 16, y0 + 26, "A. Rencana memori per konteks (--gpu-layers auto, diukur: puncak nvidia-smi 100 ms, RSS maks)", size=14, weight="700")
    gx, gy, gw, gh = x0 + 70, y0 + 50, pw - 330, 220
    vmax = 8192.0

    def V(v):
        return gy + gh - gh * v / vmax
    for v in (0, 2048, 4096, 6144, 8192):
        s.line(gx, V(v), gx + gw, V(v), color=C["line"], sw=1)
        s.text(gx - 8, V(v) + 4, "%d" % v, size=11, color=C["dim"], anchor="end")
    s.line(gx, V(CAP), gx + gw, V(CAP), color=C["bad"], sw=1.8, dash="6 4")
    s.text(gx + 4, V(CAP) - 6, "cap 7373 MiB", size=11, weight="700", color=C["bad"])
    s.text(gx - 52, gy - 10, "MiB VRAM", size=11, color=C["dim"])
    slot = gw / len(CTX)
    bw = slot * 0.32
    for i, c in enumerate(CTX):
        for j, (m, col, bg) in enumerate((("PTQ1_0", C["gpu"], C["gpu_bg"]), ("Q2_0", C["cpu"], C["cpu_bg"]))):
            d = depth(m, c)
            bx = gx + slot * i + slot * 0.14 + j * (bw + 6)
            if not d:
                s.text(bx + bw / 2, V(400), "pending", size=10, color=C["dim"], anchor="middle")
                continue
            w_mib = d["gpu_arena_mib"] - d["gpu_kv_mib"]
            kv = d["gpu_kv_mib"]
            peak = d.get("peak_vram_mib", d["vram_used_mib"])
            s.rect(bx, V(w_mib), bw, V(0) - V(w_mib), bg, col, rx=2, sw=1)
            s.rect(bx, V(w_mib + kv), bw, V(w_mib) - V(w_mib + kv), C["mtp_bg"], C["mtp"], rx=2, sw=1)
            s.rect(bx, V(peak), bw, V(w_mib + kv) - V(peak), "#eaeef2", C["base"], rx=2, sw=1)
            s.text(bx + bw / 2, V(peak) - 16, "%d" % peak, size=10, weight="700", color=col, anchor="middle")
            s.text(bx + bw / 2, V(peak) - 5, "%dL" % d["gpu_layers"], size=10, color=C["dim"], anchor="middle")
        s.text(gx + slot * i + slot / 2, gy + gh + 18, CTX_LABEL[i], size=12, weight="700", anchor="middle")
    # legend + host RAM table
    lx, ly = gx + gw + 24, y0 + 54
    for k, (bg, fg, t) in enumerate(((C["gpu_bg"], C["gpu"], "bobot + scratch PTQ1_0"), (C["cpu_bg"], C["cpu"], "bobot + scratch Q2_0"),
                                     (C["mtp_bg"], C["mtp"], "KV iso3 (12.5 KiB/posisi)"), ("#eaeef2", C["base"], "konteks CUDA + driver"))):
        s.rect(lx, ly + k * 20, 14, 12, bg, fg, rx=2, sw=1)
        s.text(lx + 20, ly + k * 20 + 11, t, size=11)
    s.text(lx, ly + 100, "RAM host (RSS maks, MB)", size=12, weight="700")
    for t, dx in (("ctx", 0), ("PTQ1_0", 90), ("Q2_0", 170)):
        s.text(lx + dx, ly + 118, t, size=11, color=C["dim"], anchor="start" if dx == 0 else "end")
    for i, c in enumerate(CTX):
        a, b = depth("PTQ1_0", c), depth("Q2_0", c)
        s.text(lx, ly + 136 + i * 16, CTX_LABEL[i], size=11, weight="700")
        s.text(lx + 90, ly + 136 + i * 16, "%.0f" % (a["max_rss_kb"] / 1000.0) if a else "pending", size=11, anchor="end")
        s.text(lx + 170, ly + 136 + i * 16, "%.0f" % (b["max_rss_kb"] / 1000.0) if b else "pending", size=11, anchor="end")
    s.text(lx, ly + 210, "cap 14200 MB · arena host dinamis", size=11, color=C["dim"])
    s.text(lx, ly + 226, "KV host hanya untuk layer CPU", size=11, color=C["dim"])
    s.text(gx, gy + gh + 40, "Angka di atas batang = puncak VRAM (MiB) dan jumlah layer GPU. Q2_0 menurunkan layer ke CPU saat KV tumbuh (64 → 62 → 59 → 53).",
           size=11, color=C["dim"])


def panel_attention(s):
    x0, y0, pw, ph = 24, 428, 470, 330
    s.rect(x0, y0, pw, ph, C["soft"], rx=6)
    s.text(x0 + 16, y0 + 26, "B. Attention GQA di GPU (op_attention)", size=14, weight="700")
    s.rect(x0 + 16, y0 + 40, pw - 32, 62, C["gpu_bg"], C["gpu"], rx=5, sw=1)
    s.text(x0 + 28, y0 + 60, "ctx_max ≤ 8192: kernel lama, tidak berubah", size=12, weight="700", color=C["gpu"])
    s.text(x0 + 28, y0 + 77, "skor [n] di shared memory (32 KiB dari 48 KiB sm_61), softmax 2 pass", size=11)
    s.text(x0 + 28, y0 + 93, "jalur Qwen tetap bit-identik (48/48 token sama dengan stage 4)", size=11)
    s.rect(x0 + 16, y0 + 112, pw - 32, 96, C["mtp_bg"], C["mtp"], rx=5, sw=1)
    s.text(x0 + 28, y0 + 132, "ctx_max > 8192: attention_kernel_tiled (sampai 131072)", size=12, weight="700", color=C["mtp"])
    for k, t in enumerate(("K/V dibaca per tile 256 posisi, skor tile 1 KiB di shared memory",
                           "online softmax FP32: m' = max(m, max tile), l = l·e^(m−m') + Σe^(s−m')",
                           "akumulator o diskalakan e^(m−m') per tile, dibagi l sekali di akhir",
                           "dekode iso3 di dalam kernel (FP32, 0 HFMA2/HMUL2)")):
        s.text(x0 + 28, y0 + 150 + k * 15, t, size=11)
    s.text(x0 + 16, y0 + 228, "Parity (tests/cuda/attn_long_test.cu), error relatif terhadap RMS:", size=12, weight="700")
    rows = []
    p = os.path.join(RES, "attn_long_test.txt")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            m = re.match(r"\[(\w+)\] kv=(\S+)\s+n=\s*(\d+) (fast|tiled)\S* .*rel ([0-9.e+-]+)", line)
            if m and m.group(4) == "tiled":
                rows.append("%s %-4s n=%6s  tiled vs ref double: %s" % (m.group(1), m.group(2), m.group(3), m.group(5)))
        fast = [l for l in open(p, encoding="utf-8") if "fast(ctx 8192)" in l]
        worst = max(float(re.search(r"rel ([0-9.e+-]+)", l).group(1)) for l in fast) if fast else None
        if worst is not None:
            rows.insert(0, "fast vs tiled (n ≤ 8192, 8 kasus): rel maks %.1e" % worst)
    for k, r in enumerate(rows[:9]):
        s.text(x0 + 28, y0 + 246 + k * 9.5, r, size=9, color=C["gpu"] if r.startswith("PASS") or k == 0 else C["bad"])


def panel_state(s):
    x0, y0, pw, ph = 506, 428, 470, 330
    s.rect(x0, y0, pw, ph, C["soft"], rx=6)
    s.text(x0 + 16, y0 + 26, "C. --state-save / --state-load", size=14, weight="700")
    cx = x0 + pw / 2
    save = jlines("PTQ1_0_ctx131072_save")
    load = jlines("PTQ1_0_ctx131072_load")
    steps = [
        (C["cpu_bg"], C["cpu"], "1) prefill dokumen sekali (-f, --state-save F)",
         "%d token · %.0f s · %.2f tok/s" % (save["tokens"], save["prefill_s"], save["prefill_tok_s"]) if save else "128K: pending"),
        (C["warn_bg"], C["warn"], "2) file state: header + token + KV + DeltaNet",
         "%.0f MiB · tulis %.1f s (via buffer pinned)" % (save["state_mib"], save["save_s"]) if save else "fingerprint model + layout, n_pos"),
        (C["gpu_bg"], C["gpu"], "3) proses baru: --state-load F -f pertanyaan.txt",
         "muat %.1f s · hanya %d token baru di-prefill" % (load["state_load_s"], load["prompt_tokens"] - load["reused_tokens"]) if load else "128K: pending"),
    ]
    yy = y0 + 42
    for i, (bg, fg, t, sub) in enumerate(steps):
        s.rect(x0 + 16, yy, pw - 32, 42, bg, rx=5)
        s.text(cx, yy + 17, t, size=12, weight="700", color=fg, anchor="middle")
        s.text(cx, yy + 34, sub, size=11, color=fg, anchor="middle")
        yy += 42
        if i < len(steps) - 1:
            s.line(cx, yy, cx, yy + 10, color=C["dim"], arrow=True)
            yy += 12
    if load:
        s.text(x0 + 16, yy + 18, "jawaban 128K: \"%s\"" % load.get("text", "")[:40], size=12, weight="700", color=C["gpu"])
    s.text(x0 + 16, yy + 40, "Uji round trip (satu proses, --state-test): token dan logit baris", size=12, weight="700")
    s.text(x0 + 16, yy + 55, "pertama dari file = state di memori (bit-identik):", size=12, weight="700")
    k = 0
    for tag, name in (("state_roundtrip_PTQ1_0_4k", "PTQ1_0 4K"), ("state_roundtrip_Q2_0_4k", "Q2_0 4K"),
                      ("state_roundtrip_qwen_q4km", "Qwen Q4_K_M 1K")):
        p = os.path.join(RES, tag + ".txt")
        if not os.path.exists(p):
            continue
        txt = open(p, encoding="utf-8").read()
        ok = "-> PASS" in txt
        m = re.search(r"state ([0-9.]+) MiB, save ([0-9.]+) s, load ([0-9.]+) s", txt)
        s.text(x0 + 28, yy + 74 + k * 16, "%s %s · %s MiB · tulis %s s · muat %s s" % ("PASS" if ok else "FAIL", name, *(m.groups() if m else ("?", "?", "?"))),
               size=11, color=C["gpu"] if ok else C["bad"])
        k += 1


def panel_decode(s):
    x0, y0, pw, ph = 24, 772, 952, 290
    s.rect(x0, y0, pw, ph, C["soft"], rx=6)
    s.text(x0 + 16, y0 + 26, "D. Decode pada kedalaman penuh (--depth-bench: KV nol sepanjang n posisi) dan uji needle", size=14, weight="700")
    gx, gy, gw, gh = x0 + 70, y0 + 46, 420, 190
    vmax = 3.0

    def Y(v):
        return gy + gh - gh * v / vmax
    for v in (0, 1, 2, 3):
        s.line(gx, Y(v), gx + gw, Y(v), color=C["line"], sw=1)
        s.text(gx - 8, Y(v) + 4, "%d" % v, size=11, color=C["dim"], anchor="end")
    s.text(gx - 52, gy - 8, "tok/s", size=11, color=C["dim"])
    slot = gw / len(CTX)
    for j, (m, col) in enumerate((("PTQ1_0", C["gpu"]), ("Q2_0", C["cpu"]))):
        pts = []
        for i, c in enumerate(CTX):
            d = depth(m, c)
            if d:
                pts.append((gx + slot * i + slot / 2, Y(d["decode_tok_s"]), d["decode_tok_s"]))
        if pts:
            s.path("M" + " L".join("%.1f,%.1f" % (a, b) for a, b, _ in pts), color=col, sw=2.2)
            for a, b, v in pts:
                s.rect(a - 3, b - 3, 6, 6, col, col, rx=3, sw=1)
                s.text(a, b - 8 - 10 * j, "%.2f" % v, size=10, weight="700", color=col, anchor="middle")
        s.line(gx + 10 + j * 120, gy + gh + 36, gx + 32 + j * 120, gy + gh + 36, color=col, sw=2.2)
        s.text(gx + 38 + j * 120, gy + gh + 40, m, size=11, weight="700", color=col)
    for i in range(len(CTX)):
        s.text(gx + slot * i + slot / 2, gy + gh + 16, CTX_LABEL[i], size=12, weight="700", anchor="middle")
    # needle table
    tx = gx + gw + 40
    s.text(tx, y0 + 52, "Needle (passphrase di awal, pertanyaan di akhir, ChatML):", size=12, weight="700")
    rows = [("PTQ1_0_ctx4k_needle_chat", "PTQ1_0 4K"), ("PTQ1_0_ctx32768_needle", "PTQ1_0 32K"), ("PTQ1_0_ctx65536_load", "PTQ1_0 64K (save+load)"),
            ("PTQ1_0_ctx131072_load", "PTQ1_0 128K (save+load)"), ("Q2_0_ctx4k_needle_chat", "Q2_0 4K")]
    for k, (tag, name) in enumerate(rows):
        d = jlines(tag)
        if d and "text" in d:
            ok = "MAGENTA-FALCON-42" in d["text"]
            s.text(tx, y0 + 74 + k * 18, "%s %-24s \"%s\" · tg %.2f tok/s" % ("PASS" if ok else "FAIL", name, d["text"][:24], d.get("decode_tok_s", 0)),
                   size=11, color=C["gpu"] if ok else C["bad"])
        else:
            s.text(tx, y0 + 74 + k * 18, "....  %-24s pending" % name, size=11, color=C["dim"])
    s.text(tx, y0 + 180, "Gate lain: SASS 0 HFMA2/HMUL2/HADD2 (IDP.4A ada) · rollback Qwen", size=11, color=C["dim"])
    s.text(tx, y0 + 196, "bit-identik · Qwen 48/48 token = stage 4 · Bonsai 16/16 = a89f467", size=11, color=C["dim"])
    s.text(tx, y0 + 212, "Kernel PTQ1_0: tabel digit ke shared memory → decode 2x,", size=11, color=C["dim"])
    s.text(tx, y0 + 228, "output 64/64 layer bit-identik dengan kernel lama.", size=11, color=C["dim"])


def main():
    s = Svg(W, H, "")
    head(s)
    panel_vram(s)
    panel_attention(s)
    panel_state(s)
    panel_decode(s)
    s.save("51_bonsai_128k.svg")


if __name__ == "__main__":
    main()
