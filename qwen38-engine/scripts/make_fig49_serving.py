#!/usr/bin/env python3
"""Draws docs/figures/49_serving_webui.svg (style of 48_cascade.svg).
Numbers: bench/results/stage5b/server_test_mtdot0.out and the "Serving round" section of docs/AFK_STATUS.md.
Usage: python3 scripts/make_fig49_serving.py   (with cairosvg installed also writes /tmp/fig_49_serving_webui.png)"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_pr_figures import C, Svg  # noqa: E402

W, H = 1000, 720
PURPLE = C["mtp"]


def head(s):
    s.parts.pop()  # 20 px title + subtitle, as in 48_cascade
    s.text(24, 38, "Serving: batas state <|im_end|>, stop string lintas chunk, telemetri PLD/MTP live", size=20, weight="700")
    s.text(24, 62, "exp/knapsack-ctx16k-webui · src/engine/generator.cpp, stop_filter.hpp, server.cpp, webui/index.html · rinci: docs/AFK_STATUS.md",
           size=14, color=C["dim"])


def chip(s, x, y, w, txt, bg, fg, dash=None, h=24):
    s.rect(x, y, w, h, bg, fg, rx=4, sw=1.2, dash=dash)
    s.text(x + w / 2, y + 16, txt, size=11, weight="700", color=fg, anchor="middle")
    return x + w + 4


def panel_boundary(s):
    x0, y0, pw, ph = 24, 88, 452, 330
    s.rect(x0, y0, pw, ph, C["soft"], rx=6)
    s.text(x0 + 16, y0 + 28, "A. Cache multi-turn: draft setelah <|im_end|>", size=14, weight="700")
    grey = ("#eaeef2", C["dim"])
    # before
    y = y0 + 50
    s.text(x0 + 16, y + 4, "Sebelum: verifikasi menerima draft di belakang token akhir", size=12, weight="700", color=C["bad"])
    x = x0 + 16
    x = chip(s, x, y + 14, 70, "… jawaban", *grey)
    x = chip(s, x, y + 14, 78, "<|im_end|>", C["gpu_bg"], C["gpu"])
    x = chip(s, x, y + 14, 62, "draft d2", C["bad_bg"], C["bad"])
    x = chip(s, x, y + 14, 62, "draft d3", C["bad_bg"], C["bad"])
    s.text(x + 4, y + 30, "→ masuk cached_", size=12, weight="700", color=C["bad"])
    s.text(x0 + 16, y + 58, "giliran 2: prompt = … <|im_end|>\\n<|im_start|>user …", size=12, color=C["dim"])
    s.text(x0 + 16, y + 76, "prefix token tidak cocok (DeltaNet tidak bisa di-rewind parsial)", size=12, color=C["dim"])
    s.rect(x0 + 16, y + 86, pw - 32, 26, C["bad_bg"], rx=5)
    s.text(x0 + pw / 2, y + 103, "reuse prefix gagal diam-diam → prefill penuh tiap giliran", size=12, weight="700", color=C["bad"], anchor="middle")
    # after
    y = y0 + 180
    s.text(x0 + 16, y + 4, "Sesudah: token akhir yang diterima memutus loop acceptance", size=12, weight="700", color=C["gpu"])
    x = x0 + 16
    x = chip(s, x, y + 14, 70, "… jawaban", *grey)
    x = chip(s, x, y + 14, 78, "<|im_end|>", C["gpu_bg"], C["gpu"])
    s.line(x + 2, y + 8, x + 2, y + 44, color=C["bad"], sw=2.5)
    x += 10
    x = chip(s, x, y + 14, 62, "draft d2", "#ffffff", C["dim"], dash="4 3")
    x = chip(s, x, y + 14, 62, "draft d3", "#ffffff", C["dim"], dash="4 3")
    s.text(x + 4, y + 30, "→ dibuang", size=12, weight="700", color=C["dim"])
    s.rect(x0 + 16, y + 54, pw - 32, 26, "#eaeef2", rx=5)
    s.text(x0 + pw / 2, y + 71, "generator.cpp: if (is_stop(t)) break;  // cached_ berhenti di sini", size=12, weight="700", anchor="middle")
    s.rect(x0 + 16, y + 88, pw - 32, 26, C["gpu_bg"], rx=5)
    s.text(x0 + pw / 2, y + 105, "server_test: reused 42 of 66 tokens pada giliran 2 · PASS", size=12, weight="700", color=C["gpu"], anchor="middle")
    s.text(x0 + 16, y + 132, "Terlihat karena MTP adaptif menerima hingga 4 draft per siklus.", size=11, color=C["dim"])


def panel_stop(s):
    x0, y0, pw, ph = 500, 88, 476, 330
    s.rect(x0, y0, pw, ph, C["soft"], rx=6)
    s.text(x0 + 16, y0 + 28, "B. StopFilter: holdback token parsial lintas chunk SSE", size=14, weight="700")
    s.text(x0 + 16, y0 + 50, "contoh dari server_test: stop = \"he ho\"", size=12, color=C["dim"])
    cols = [x0 + 16, x0 + 106, x0 + 226, x0 + 318]
    hdr = ["token baru", "dikirim ke klien", "ditahan", "alasan"]
    y = y0 + 64
    s.rect(x0 + 12, y, pw - 24, 24, "#eaeef2", rx=4)
    for c, t in zip(cols, hdr):
        s.text(c + 4, y + 16, t, size=11, weight="700", color=C["dim"])
    rows = [
        ("\"… to t\"", "\"… to t\"", "-", "bukan prefix", "#ffffff", C["ink"]),
        ("\"he\"", "-", "\"he\"", "prefix \"he ho\"", C["warn_bg"], C["warn"]),
        ("\" ho\"", "-", "(dibuang)", "cocok → stop", C["bad_bg"], C["bad"]),
    ]
    for i, (a, b, c_, d, bg, fg) in enumerate(rows):
        yy = y + 28 + i * 30
        s.rect(x0 + 12, yy, pw - 24, 26, bg, rx=4)
        for c, t in zip(cols, (a, b, c_, d)):
            s.text(c + 4, yy + 17, t, size=12, weight="700" if c == cols[3] else "400", color=fg)
    y = y + 28 + 3 * 30 + 8
    s.rect(x0 + 12, y, pw - 24, 26, C["gpu_bg"], rx=4)
    s.text(x0 + pw / 2, y + 17, "teks akhir \"e sea stretches endlessly to t\" · finish_reason \"stop\"", size=12, weight="700", color=C["gpu"], anchor="middle")
    notes = [
        "• push() menahan suffix terpanjang yang merupakan prefix stop string;",
        "  flush() di akhir mengirim sisa jika tidak ada yang cocok.",
        "• Dengan thinking: filter mulai setelah </think> (hanya jawaban).",
        "• stop: string atau array ≤ 4, 1-128 byte; selain itu HTTP 400.",
        "• stream = non-stream, stop string tidak pernah muncul; 11 vs 20 token.",
        "• stop_filter_test 36/36: semua split 1/2/3 bagian dan per byte.",
    ]
    s.lines(x0 + 16, y + 46, notes, size=12, gap=17)


def panel_telemetry(s):
    x0, y0, pw, ph = 24, 432, 952, 268
    s.rect(x0, y0, pw, ph, C["soft"], rx=6)
    s.text(x0 + 16, y0 + 28, "C. WebUI → server → generator → telemetri live (x_stats di setiap chunk SSE)", size=14, weight="700")
    by, bh = y0 + 46, 116
    boxes = [
        (x0 + 16, 200, C["cpu_bg"], C["cpu"], "WebUI (index.html)", ["[x] MTP adaptif (default on)", "[ ] PLD (off)", "kotak stop, satu per baris", "parseStops(): \\n escape"]),
        (x0 + 248, 210, "#eaeef2", C["ink"], "POST /v1/chat/completions", ["mtp_adaptive: true", "pld: true, pld_max 1..4", "stop: [\"he ho\"]", "stream: true"]),
        (x0 + 490, 210, C["mtp_bg"], PURPLE, "Generator (per siklus)", ["1) PLD 3/4-gram dulu", "2) kosong → draft MTP adaptif", "3) verifikasi batch, rollback", "4) StopFilter.push(teks)"]),
        (x0 + 732, 204, C["gpu_bg"], C["gpu"], "Telemetri WebUI", ["PLD hits 2 (dari 8 kandidat)", "MTP drafts 28", "accept % · cycles", "stop_matched \"he ho\""]),
    ]
    for i, (x, w, bg, fg, t, rows) in enumerate(boxes):
        s.rect(x, by, w, bh, bg, rx=5)
        s.text(x + w / 2, by + 20, t, size=12, weight="700", color=fg, anchor="middle")
        s.lines(x + 12, by + 42, rows, size=12, color=C["ink"], gap=18)
        if i < len(boxes) - 1:
            nx = boxes[i + 1][0]
            s.line(x + w + 2, by + bh / 2, nx - 3, by + bh / 2, color=C["dim"], arrow=True)
    s.path("M%.1f,%.1f V%.1f H%.1f V%.1f" % (x0 + 834, by + bh, by + bh + 10, x0 + 116, by + bh + 2), color=C["dim"], sw=1.2, dash="4 3", arrow=True)
    s.text(x0 + 475, by + bh + 26, "SSE: delta teks + x_stats {pld_hits, pld_candidates, mtp_fills, cycles, accept_pct}", size=11, color=C["dim"], anchor="middle")

    # speed bars
    sy = by + bh + 34
    s.text(x0 + 16, sy + 14, "Decode 509 + 256 token, temp 0 (256/256 token identik):", size=12, weight="700")
    bars = [("main, MTP 2 tetap", 2.947, C["base"], "95 %"), ("adaptif (default baru)", 3.130, C["gpu"], "96 %"), ("MTP 0", 2.604, C["llama"], "")]
    bx, bw, vmax = x0 + 560, 200, 3.3
    for i, (lab, v, col, acc) in enumerate(bars):
        yy = sy + 2 + i * 22
        s.text(bx - 8, yy + 13, lab, size=12, anchor="end")
        s.rect(bx, yy, bw * v / vmax, 16, col, rx=3)
        s.text(bx + bw * v / vmax + 6, yy + 13, "%.3f tok/s" % v + ("  · accept " + acc if acc else ""), size=12, weight="700", color=col)
    s.text(x0 + 16, sy + 36, "server_test 69/69 (QWEN_MT_DOT=0) · webui_test PASS", size=12, color=C["dim"])
    s.text(x0 + 16, sy + 54, "rollback bit-identik · SASS gate PASS", size=12, color=C["dim"])


def main():
    s = Svg(W, H, "")
    head(s)
    panel_boundary(s)
    panel_stop(s)
    panel_telemetry(s)
    s.save("49_serving_webui.svg")


if __name__ == "__main__":
    main()
