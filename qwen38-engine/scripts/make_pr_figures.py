#!/usr/bin/env python3
"""Draws the SVG figures used in the pull-request descriptions and docs (docs/figures/).
Every number comes from a measured run recorded in docs/AFK_STATUS.md, bench/results/final_comparison.md,
bench/results/experiment_comparison.md or bench/results/experiments/EXPERIMENT-LOG.md.
Usage: python3 scripts/make_pr_figures.py [44|45|all]   (writes docs/figures/*.svg; with cairosvg installed also a PNG preview in /tmp)"""
import os
import sys
from xml.sax.saxutils import escape

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures")
FONT = "Segoe UI, Helvetica Neue, Helvetica, Arial, DejaVu Sans, sans-serif"

C = {
    "ink": "#1f2328", "dim": "#57606a", "line": "#d0d7de", "bg": "#ffffff", "soft": "#f6f8fa",
    "gpu": "#1a7f37", "gpu_bg": "#dafbe1", "cpu": "#0969da", "cpu_bg": "#ddf4ff",
    "pcie": "#bc4c00", "pcie_bg": "#fff1e5", "mtp": "#8250df", "mtp_bg": "#fbefff",
    "bad": "#cf222e", "bad_bg": "#ffebe9", "warn": "#9a6700", "warn_bg": "#fff8c5",
    "llama": "#6e7781", "base": "#8c959f", "exp": "#1a7f37",
}


class Svg:
    def __init__(self, w, h, title):
        self.w, self.h, self.parts = w, h, []
        self.rect(0, 0, w, h, C["bg"], stroke="none", rx=0)
        self.text(24, 38, title, size=22, weight="700")

    def rect(self, x, y, w, h, fill, stroke=None, rx=8, sw=1.5, dash=None):
        d = ' stroke-dasharray="%s"' % dash if dash else ""
        self.parts.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="%d" fill="%s" stroke="%s" stroke-width="%.1f"%s/>'
                          % (x, y, w, h, rx, fill, stroke or "none", sw, d))

    def text(self, x, y, s, size=14, color=None, weight="400", anchor="start", italic=False):
        st = ' font-style="italic"' if italic else ""
        self.parts.append('<text x="%.1f" y="%.1f" font-family="%s" font-size="%d" font-weight="%s" fill="%s" text-anchor="%s"%s>%s</text>'
                          % (x, y, FONT, size, weight, color or C["ink"], anchor, st, escape(s)))

    def lines(self, x, y, rows, size=14, color=None, weight="400", anchor="start", gap=None):
        g = gap or size * 1.35
        for i, r in enumerate(rows):
            self.text(x, y + i * g, r, size=size, color=color, weight=weight, anchor=anchor)

    def line(self, x1, y1, x2, y2, color=None, sw=1.5, dash=None, arrow=False):
        d = ' stroke-dasharray="%s"' % dash if dash else ""
        m = ' marker-end="url(#arr_%s)"' % (color or C["ink"]).strip("#") if arrow else ""
        self.parts.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="%.1f"%s%s/>'
                          % (x1, y1, x2, y2, color or C["ink"], sw, d, m))
        if arrow:
            self.markers.add(color or C["ink"])

    def path(self, d, color=None, sw=1.5, fill="none", arrow=False, dash=None):
        da = ' stroke-dasharray="%s"' % dash if dash else ""
        m = ' marker-end="url(#arr_%s)"' % (color or C["ink"]).strip("#") if arrow else ""
        self.parts.append('<path d="%s" stroke="%s" stroke-width="%.1f" fill="%s"%s%s/>' % (d, color or C["ink"], sw, fill, da, m))
        if arrow:
            self.markers.add(color or C["ink"])

    markers = set()

    def box(self, x, y, w, h, fill, stroke, title, rows=(), tsize=15, rsize=13, tcolor=None):
        self.rect(x, y, w, h, fill, stroke)
        self.text(x + w / 2, y + 24, title, size=tsize, weight="700", anchor="middle", color=tcolor or stroke)
        self.lines(x + w / 2, y + 24 + rsize * 1.55, list(rows), size=rsize, anchor="middle", color=C["ink"])

    def save(self, name):
        defs = "".join('<marker id="arr_%s" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                       '<path d="M0,0 L10,5 L0,10 z" fill="%s"/></marker>' % (c.strip("#"), c) for c in sorted(self.markers))
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" role="img">'
               '<defs>%s</defs>%s</svg>\n' % (self.w, self.h, self.w, self.h, defs, "".join(self.parts)))
        os.makedirs(OUT, exist_ok=True)
        p = os.path.join(OUT, name)
        open(p, "w", encoding="utf-8").write(svg)
        Svg.markers = set()
        try:
            import cairosvg
            cairosvg.svg2png(bytestring=svg.encode(), write_to="/tmp/fig_" + name.replace(".svg", ".png"), output_width=self.w)
        except Exception:
            pass
        print("wrote", p)


def num(v, fmt):
    return (fmt % v).replace(".", ",")


def hbars(s, x, y, w, rows, vmax, unit, bar_h=30, gap=14, label_w=300, fmt="%.2f", note_w=None):
    """rows: (label, value, color, note). The value is printed after the bar, the note in grey after it; space for the
    longest value + note is reserved on the right so nothing runs off the figure."""
    if note_w is None:
        note_w = max(len(num(v, fmt) + unit + ("   " + n if n else "")) for _, v, _, n in rows) * 7.8 + 16
    for i, (lab, v, col, note) in enumerate(rows):
        yy = y + i * (bar_h + gap)
        s.text(x + label_w - 12, yy + bar_h * 0.68, lab, size=14, anchor="end")
        bw = (w - label_w - note_w) * v / vmax
        s.rect(x + label_w, yy, max(bw, 3), bar_h, col, stroke="none", rx=4)
        vt = num(v, fmt) + unit
        s.text(x + label_w + max(bw, 3) + 8, yy + bar_h * 0.68, vt, size=14, weight="700")
        if note:
            s.text(x + label_w + max(bw, 3) + 8 + len(vt) * 9.2 + 12, yy + bar_h * 0.68, note, size=13, color=C["dim"])


# ------------------------------------------------------------------------------------------------------ PR 44 figures

def fig44_architecture():
    s = Svg(1000, 700, "Cara kerja: model 64 layer dibagi antara GPU dan CPU")
    s.text(24, 62, "Setiap token jawaban melewati ke-64 layer. Panah = arah data untuk satu token.", size=14, color=C["dim"])
    # input
    s.box(24, 110, 170, 92, C["soft"], C["dim"], "Teks prompt", ["tokenizer BPE sendiri", "(identik llama-tokenize)"], tcolor=C["ink"])
    s.line(194, 156, 232, 156, C["dim"], 2, arrow=True)
    # GPU
    s.rect(236, 84, 300, 250, C["gpu_bg"], C["gpu"], rx=12, sw=2)
    s.text(386, 112, "GPU  GTX 1070 Ti  (VRAM)", size=16, weight="700", anchor="middle", color=C["gpu"])
    s.text(386, 146, "Layer 0 - 31", size=20, weight="700", anchor="middle", color=C["gpu"])
    s.lines(386, 178, ["bobot tetap di VRAM (6,6 GiB)", "perkalian matriks int8 __dp4a", "aritmetika lain FP32",
                       "cache KV iso3 (3-bit)"], size=14, anchor="middle")
    s.text(386, 296, "VRAM terpakai 7.245 MiB", size=14, weight="700", anchor="middle", color=C["gpu"])
    s.text(386, 316, "(batas 7.373 MiB)", size=13, anchor="middle", color=C["dim"])
    # PCIe
    s.rect(552, 172, 102, 68, C["pcie_bg"], C["pcie"], rx=8)
    s.lines(603, 200, ["PCIe Gen3 x1", "0,84 GB/s"], size=13, anchor="middle", color=C["pcie"], weight="700")
    s.line(536, 206, 548, 206, C["pcie"], 2, arrow=True)
    s.line(654, 206, 666, 206, C["pcie"], 2, arrow=True)
    s.lines(603, 262, ["hanya 5.120 angka", "per token lewat sini"], size=12, anchor="middle", color=C["dim"])
    # CPU
    s.rect(670, 84, 306, 250, C["cpu_bg"], C["cpu"], rx=12, sw=2)
    s.text(823, 112, "CPU  Ryzen 5 2400G  (RAM DDR4)", size=16, weight="700", anchor="middle", color=C["cpu"])
    s.text(823, 146, "Layer 32 - 63", size=20, weight="700", anchor="middle", color=C["cpu"])
    s.lines(823, 178, ["bobot di RAM lewat mmap (7,0 GiB)", "4 core, AVX2 (libggml-cpu)", "cache KV FP32"], size=14, anchor="middle")
    s.text(823, 296, "dibaca ulang tiap token", size=14, weight="700", anchor="middle", color=C["cpu"])
    s.text(823, 316, "~25 GB/s (batas RAM 33 GB/s)", size=13, anchor="middle", color=C["dim"])
    # LM head
    s.box(670, 380, 306, 96, C["cpu_bg"], C["cpu"], "Kepala LM + sampler", ["memilih 1 dari 248.320 token", "temperature, top_k, top_p, min_p"])
    s.line(823, 334, 823, 376, C["cpu"], 2, arrow=True)
    # output
    s.box(24, 380, 190, 96, C["soft"], C["dim"], "Token jawaban", ["ke CLI atau server", "(SSE streaming)"], tcolor=C["ink"])
    s.path("M 670 428 L 640 428", C["cpu"], 2)
    s.path("M 640 428 L 640 500 L 230 500 L 230 428 L 218 428", C["cpu"], 2, arrow=True)
    # MTP
    s.rect(250, 372, 376, 112, C["mtp_bg"], C["mtp"], rx=12, sw=2)
    s.text(438, 398, "MTP: kepala penebak (block 64)", size=16, weight="700", anchor="middle", color=C["mtp"])
    s.lines(438, 424, ["menebak 2 token berikutnya; model utama", "memeriksa [token + tebakan] dalam SATU langkah"],
            size=14, anchor="middle")
    s.path("M 438 372 L 438 344 L 386 344 L 386 338", C["mtp"], 2, arrow=True, dash="6,4")
    s.text(446, 362, "tebakan diperiksa", size=12, color=C["mtp"])
    # notes
    s.rect(24, 530, 952, 150, C["soft"], C["line"])
    s.lines(40, 558, ["• 48 dari 64 layer adalah Gated DeltaNet (rekuren, menyimpan “state”); 16 layer adalah atensi (layer 3, 7, 11, ...).",
                      "• Tebakan MTP yang salah dibuang, dan state 48 layer DeltaNet dikembalikan persis (rollback bit-identik).",
                      "• Karena itu jawaban dengan MTP selalu sama persis dengan jawaban tanpa MTP.",
                      "• Tidak ada pembacaan disk saat token dibuat, dan tidak ada alokasi memori baru (arena statis).",
                      "• 32 layer GPU adalah jumlah terbanyak yang muat di batas VRAM (33 layer = 7.420 MiB)."], size=14, gap=24)
    s.save("44_architecture.svg")


def fig44_memory():
    s = Svg(1000, 470, "Pemakaian memori (diukur, prompt 509 token + 256 token jawaban)")
    x0, W = 200, 720
    y0 = 110
    s.text(24, 84, "VRAM GPU (kapasitas 8.192 MiB)", size=16, weight="700")
    parts = [("bobot layer 0-31", 6646, C["gpu"]), ("buffer kerja", 459, "#4ac26b"), ("konteks CUDA", 112, C["base"]), ("cache KV iso3", 25, C["mtp"])]
    tot = 8192
    xx = x0
    for lab, v, col in parts:
        w = W * v / tot
        s.rect(xx, y0, w, 40, col, stroke="none", rx=0)
        xx += w
    s.rect(x0, y0, W, 40, "none", C["ink"], rx=0, sw=1)
    s.text(x0 - 12, y0 + 26, "terpakai", size=14, anchor="end")
    cap = x0 + W * 7373 / tot
    s.line(cap, y0 - 12, cap, y0 + 52, C["bad"], 2.5, dash="6,4")
    s.text(cap - 6, y0 - 16, "batas 7.373 MiB", size=13, weight="700", anchor="end", color=C["bad"])
    s.text(x0, y0 + 70, "total 7.245 MiB — sisa 128 MiB di bawah batas", size=14, weight="700")
    for i, (lab, v, col) in enumerate(parts):
        lx = x0 + (i % 2) * 360
        ly = y0 + 96 + (i // 2) * 24
        s.rect(lx, ly - 12, 14, 14, col, stroke="none", rx=2)
        s.text(lx + 22, ly, "%s: %s MiB" % (lab, format(v, ",").replace(",", ".")), size=14)
    y1 = 300
    s.text(24, y1 - 16, "RAM DDR4 (puncak per proses, 16 GB terpasang)", size=16, weight="700")
    scale = W / 16000.0
    s.text(x0 - 12, y1 + 21, "llama.cpp", size=14, anchor="end")
    s.rect(x0, y1, 14569 * scale, 30, C["llama"], stroke="none", rx=4)
    s.text(x0 + 14569 * scale - 8, y1 + 21, "14.569 MiB", size=14, weight="700", color="#ffffff", anchor="end")
    s.text(x0 - 12, y1 + 69, "qwen38-engine", size=14, anchor="end")
    s.rect(x0, y1 + 48, 9214 * scale, 30, C["cpu"], stroke="none", rx=4)
    s.text(x0 + 9214 * scale + 8, y1 + 69, "9.214 MiB  (37 % lebih hemat)", size=14, weight="700", color=C["cpu"])
    s.lines(24, y1 + 116, ["Halaman file untuk layer yang sudah ada di VRAM dibuang dari RAM. llama.cpp memetakan seluruh file 16 GB,",
                           "sehingga halaman layer GPU-nya ikut terhitung di RAM."], size=13, color=C["dim"], gap=20)
    s.save("44_memory.svg")


def fig44_mtp_cycle():
    s = Svg(1000, 500, "Satu siklus MTP (contoh: 2 tebakan, tebakan kedua salah)")
    xs = [24, 270, 516, 762]
    titles = ["1. Tebak", "2. Periksa sekaligus", "3. Bandingkan", "4. Rollback"]
    cols = [C["mtp"], C["gpu"], C["cpu"], C["warn"]]
    bgs = [C["mtp_bg"], C["gpu_bg"], C["cpu_bg"], C["warn_bg"]]
    rows = [["kepala MTP (murah)", "menebak 2 token:", "", "“Paris”  “,”"],
            ["model utama, 64 layer,", "menghitung 3 baris", "dalam satu langkah:", "", "[ is | Paris | , ]"],
            ["jawaban model utama:", "“Paris”  lalu  “.”", "", "tebakan 1: benar", "tebakan 2: salah"],
            ["state 48 layer DeltaNet", "dikembalikan ke posisi", "setelah “Paris”", "", "(snapshot + putar ulang)"]]
    for i in range(4):
        s.box(xs[i], 80, 214, 190, bgs[i], cols[i], titles[i], rows[i], tsize=16, rsize=14)
        if i < 3:
            s.line(xs[i] + 216, 248, xs[i + 1] - 4, 248, C["dim"], 2, arrow=True)
    s.rect(24, 300, 952, 76, C["soft"], C["line"])
    s.text(500, 330, "Hasil siklus ini: 2 token keluar (“Paris” dari tebakan + “.” dari model utama) dalam 1 langkah besar.",
           size=15, anchor="middle", weight="700")
    s.text(500, 356, "Tanpa MTP butuh 2 langkah. Jawaban selalu sama dengan tanpa MTP, karena setiap tebakan diperiksa.",
           size=14, anchor="middle", color=C["dim"])
    s.lines(24, 414, ["Kapan menguntungkan di mesin ini? Satu langkah 3 baris di CPU hampir 3x lebih mahal daripada 1 baris,",
                      "jadi MTP hanya untung bila tebakan sering benar:",
                      "• teks 62 token (100 % tebakan benar): 2,46 → 2,71 token/detik",
                      "• teks 5 token (75 % tebakan benar): 2,54 → 2,20 token/detik"], size=14, gap=21)
    s.save("44_mtp_cycle.svg")


def fig44_benchmark():
    s = Svg(1000, 470, "Benchmark: prompt 509 token + 256 token jawaban")
    s.text(24, 62, "temperature 0, seed 42. Kedua program dijalankan bergantian, tidak pernah bersamaan. Batang lebih panjang = lebih cepat.", size=14, color=C["dim"])
    s.text(24, 100, "Menulis jawaban (tg, token/detik)", size=16, weight="700")
    hbars(s, 24, 116, 940, [("llama.cpp + MTP 2", 2.50, C["llama"], ""), ("llama.cpp tanpa MTP", 2.25, C["base"], ""),
                            ("qwen38-engine + MTP 2", 2.58, C["exp"], "tercepat"), ("qwen38-engine tanpa MTP", 2.44, "#4ac26b", "")],
          3.0, "", bar_h=26, gap=10, label_w=260)
    s.text(24, 290, "Membaca prompt (pp, token/detik)", size=16, weight="700")
    hbars(s, 24, 306, 940, [("llama.cpp + MTP 2", 28.70, C["llama"], ""), ("llama.cpp tanpa MTP", 29.83, C["base"], ""),
                            ("qwen38-engine + MTP 2", 13.42, C["exp"], "lebih lambat (PCIe x1); PR #45: 27,8"),
                            ("qwen38-engine tanpa MTP", 16.97, "#4ac26b", "")], 34.0, "", bar_h=26, gap=10, label_w=260)
    s.save("44_benchmark.svg")


def fig44_progress():
    s = Svg(1000, 360, "Kecepatan menulis jawaban dari tahap ke tahap")
    hbars(s, 24, 84, 940, [("Tahap 2: semua di CPU", 0.059, C["bad"], "membaca disk tiap token"),
                           ("Tahap 3: 24 layer GPU, FP32", 1.963, C["base"], "33x lebih cepat"),
                           ("Tahap 3: + __dp4a semua jenis", 2.126, "#4ac26b", ""),
                           ("Tahap 3: 32 layer GPU", 2.524, C["exp"], "default sekarang"),
                           ("Tahap 4: + MTP 2", 2.714, C["mtp"], "prompt 62 token")],
          3.0, " token/detik", bar_h=30, gap=14, label_w=300, fmt="%.3f")
    s.text(24, 62, "Prompt 5 token tanpa MTP, kecuali baris terakhir. Satuan: token jawaban per detik.", size=14, color=C["dim"])
    s.text(24, 330, "Batas utama: 32 layer CPU dibaca dari RAM setiap token (7 GiB pada ~25 GB/s \u2248 0,29 detik per token).", size=14, color=C["dim"])
    s.save("44_progress.svg")


# ------------------------------------------------------------------------------------------------------ PR 45 figures

def fig45_prefill_timeline():
    s = Svg(1000, 400, "Ke mana waktu prefill pergi (prompt 509 token, MTP 2)")
    s.text(24, 62, "Satu batang = satu prefill. Panjang = detik. Pembagian baseline diperkirakan dari pengukuran terpisah.", size=14, color=C["dim"])
    x0, W, smax = 200, 720, 34.0
    sc = W / smax

    def bar(y, label, total, segs):
        s.text(x0 - 12, y + 30, label, size=15, weight="700", anchor="end")
        xx = x0
        for lab, v, col in segs:
            w = v * sc
            s.rect(xx, y, w, 46, col, stroke="#ffffff", rx=0, sw=2)
            if w > 60:
                s.text(xx + w / 2, y + 20, lab, size=13, anchor="middle", color="#ffffff", weight="700")
                s.text(xx + w / 2, y + 37, num(v, "%.1f") + " s", size=13, anchor="middle", color="#ffffff")
            xx += w
        s.text(xx + 10, y + 30, num(total, "%.1f") + " s", size=16, weight="700")

    bar(90, "baseline", 32.0, [("layer 0-31", 7.3, C["gpu"]), ("layer 32-63 (salin + hitung bergantian)", 19.9, C["pcie"]),
                               ("MTP di CPU", 4.8, C["mtp"])])
    bar(170, "PR ini", 18.3, [("layer 0-31", 7.3, C["gpu"]), ("32-63 (bertumpuk)", 10.8, C["pcie"]), ("", 0.2, C["mtp"])])
    floor = x0 + (7.3 + 8.9) * sc
    s.line(floor, 160, floor, 228, C["bad"], 2, dash="5,4")
    s.text(floor + 6, 246, "batas fisik PCIe: 7 GiB / 0,84 GB/s = 8,9 s", size=13, color=C["bad"], weight="700")
    # axis
    for t in range(0, 35, 5):
        xx = x0 + t * sc
        s.line(xx, 262, xx, 268, C["dim"], 1)
        s.text(xx, 284, "%d s" % t, size=12, anchor="middle", color=C["dim"])
    s.line(x0, 262, x0 + W, 262, C["dim"], 1)
    s.rect(24, 306, 952, 70, C["soft"], C["line"])
    s.text(40, 334, "Prefill 1,75x lebih cepat: 15,9 \u2192 27,8 token/detik. Dua perubahan memberi hampir semua keuntungan:", size=14, weight="700")
    s.text(40, 358, "penyalinan bertumpuk dengan perhitungan (oranye: 19,9 \u2192 10,8 s) dan prefill MTP di GPU (ungu: 4,8 \u2192 0,2 s).",
           size=14)
    s.save("45_prefill_timeline.svg")


def fig45_overlap():
    s = Svg(1000, 430, "Penyalinan bobot layer 32-63 ke GPU: bergantian vs bertumpuk")
    unit = 70

    def lane(y, label, items):
        s.text(190, y + 25, label, size=14, anchor="end", weight="700")
        for x, w, col, t in items:
            s.rect(200 + x * unit, y, w * unit - 4, 38, col, stroke="none", rx=5)
            s.text(200 + x * unit + (w * unit - 4) / 2, y + 24, t, size=13, anchor="middle", color="#ffffff", weight="700")

    s.text(24, 80, "Baseline: salin, tunggu, hitung, salin lagi (lewat buffer perantara di RAM)", size=16, weight="700", color=C["pcie"])
    lane(96, "PCIe", [(0, 1.6, C["pcie"], "salin M1"), (2.4, 1.6, C["pcie"], "salin M2"), (4.8, 1.6, C["pcie"], "salin M3"), (7.2, 1.6, C["pcie"], "salin M4")])
    lane(142, "GPU", [(1.6, 0.8, C["gpu"], "M1"), (4.0, 0.8, C["gpu"], "M2"), (6.4, 0.8, C["gpu"], "M3"), (8.8, 0.8, C["gpu"], "M4")])
    s.text(24, 230, "PR ini: dua buffer di VRAM; salinan berikutnya berjalan selagi GPU menghitung", size=16, weight="700", color=C["gpu"])
    lane(246, "PCIe", [(0, 1.6, C["pcie"], "salin M1"), (1.6, 1.6, C["pcie"], "salin M2"), (3.2, 1.6, C["pcie"], "salin M3"), (4.8, 1.6, C["pcie"], "salin M4")])
    lane(292, "GPU", [(1.6, 0.8, C["gpu"], "M1"), (3.2, 0.8, C["gpu"], "M2"), (4.8, 0.8, C["gpu"], "M3"), (6.4, 0.8, C["gpu"], "M4")])
    s.line(200 + 9.6 * unit, 90, 200 + 9.6 * unit, 186, C["bad"], 2, dash="5,4")
    s.line(200 + 7.2 * unit, 240, 200 + 7.2 * unit, 336, C["exp"], 2, dash="5,4")
    s.text(200 + 7.2 * unit + 8, 330, "selesai lebih awal", size=13, weight="700", color=C["exp"])
    s.text(24, 372, "Selain itu, halaman file (6,96 GiB) didaftarkan sebagai memori pinned (cudaHostRegister read-only), sehingga GPU",
           size=14)
    s.text(24, 394, "membaca langsung dari cache halaman RAM tanpa salinan perantara. Waktu tunggu PCIe kini 0,02 s (sebelumnya beberapa detik).",
           size=14)
    s.text(24, 416, "Skema, bukan skala waktu sebenarnya.", size=12, color=C["dim"], italic=True)
    s.save("45_overlap.svg")


def fig45_ab():
    s = Svg(1000, 430, "Hasil: baseline vs PR ini vs llama.cpp (509 + 256 token, MTP 2)")
    s.text(24, 62, "Baseline dan PR ini: dijalankan bergantian 2 putaran, cache hangat. llama.cpp: sesi yang sama.", size=14, color=C["dim"])
    s.text(24, 100, "Membaca prompt (pp, token/detik)", size=16, weight="700")
    hbars(s, 24, 114, 940, [("baseline (PR #44)", 15.9, C["base"], ""), ("PR ini", 27.8, C["exp"], "1,75x baseline"),
                            ("llama.cpp -ngl 25", 27.25, C["llama"], "(28,70 di putaran sebelumnya)")], 34.0, "", bar_h=28, gap=10,
          label_w=230, fmt="%.2f")
    s.text(24, 252, "Menulis jawaban (tg, token/detik)", size=16, weight="700")
    hbars(s, 24, 266, 940, [("baseline (PR #44)", 2.57, C["base"], ""), ("PR ini", 2.62, C["exp"], "+2 %"),
                            ("llama.cpp -ngl 25", 2.52, C["llama"], "")], 3.2, "", bar_h=28, gap=10, label_w=230, fmt="%.2f")
    s.text(24, 410, "Jawaban 256/256 token identik dengan baseline di semua run. VRAM 7.335 MiB, RAM 9.102 MiB.", size=14, weight="700")
    s.save("45_ab_result.svg")


def fig45_experiments():
    rows = [("A1", "Prefill layer 32-63 di CPU saja", "3,4 token/detik (jalur GPU: 18,7)", "bad"),
            ("A2", "Salin bertumpuk dari memori pinned", "prefill 27,2 \u2192 18,1 s", "ok"),
            ("A3", "MTP hanya di token terakhir prompt", "tebakan benar 96 % \u2192 51 %", "bad"),
            ("A4", "Prefill KV MTP di GPU", "MTP 5 s \u2192 0,2 s", "ok"),
            ("A5", "Tile token 32 / 64", "layer GPU 7,3 \u2192 12,8 / 17,5 s", "bad"),
            ("A6", "__ldg, 8 baris per blok", "tidak berubah (7,3 s)", "bad"),
            ("B1", "Baca 128-bit + PTX lop3", "GPU 46,3 \u2192 ~42 ms/token", "ok"),
            ("B2", "CUDA Graph", "tidak berubah (43,5 vs 44,2 ms)", "neutral"),
            ("C1", "MTP adaptif (1-4 tebakan)", "teks sulit 2,34 \u2192 2,56 token/detik", "opt"),
            ("C2", "Potongan kerja CPU 8 \u2192 128 baris", "layer CPU 298 \u2192 287 ms/token", "ok"),
            ("C3", "Software prefetch di CPU", "30 % lebih lambat", "bad")]
    s = Svg(1000, 100 + 44 * len(rows) + 50, "Semua eksperimen dan hasilnya")
    tags = {"ok": ("dipakai", C["exp"], C["gpu_bg"]), "bad": ("ditolak", C["bad"], C["bad_bg"]),
            "neutral": ("dipakai, netral", C["warn"], C["warn_bg"]), "opt": ("opsi", C["mtp"], C["mtp_bg"])}
    s.text(24, 76, "ID", size=13, weight="700", color=C["dim"])
    s.text(80, 76, "Ide", size=13, weight="700", color=C["dim"])
    s.text(470, 76, "Hasil terukur", size=13, weight="700", color=C["dim"])
    s.text(830, 76, "Keputusan", size=13, weight="700", color=C["dim"])
    for i, (k, idea, res, t) in enumerate(rows):
        y = 90 + i * 44
        s.rect(16, y, 968, 38, C["soft"] if i % 2 == 0 else C["bg"], stroke="none", rx=4)
        s.text(24, y + 25, k, size=14, weight="700")
        s.text(80, y + 25, idea, size=14)
        s.text(470, y + 25, res, size=14)
        lab, col, bg = tags[t]
        s.rect(830, y + 6, 140, 26, bg, col, rx=13)
        s.text(900, y + 24, lab, size=13, weight="700", anchor="middle", color=col)
    s.text(24, 90 + 44 * len(rows) + 30, "Rincian setiap baris (alasan, perintah, output): bench/results/experiments/EXPERIMENT-LOG.md",
           size=13, color=C["dim"])
    s.save("45_experiments.svg")


def fig45_adaptive():
    s = Svg(1000, 420, "MTP adaptif: menebak lebih banyak hanya saat yakin")
    groups = [("Teks sulit ditebak", "(prompt 5 token, 48 token)", [2.59, 2.34, 2.56]),
              ("Teks mudah ditebak", "(prompt 62 token, 48 token)", [2.60, 2.79, 2.78]),
              ("Benchmark", "(509 + 256 token)", [2.56, 2.61, 2.57])]
    names = [("MTP 0", C["base"]), ("MTP 2 tetap", C["mtp"]), ("MTP adaptif", C["exp"])]
    y0, H, vmin, vmax = 90, 220, 2.0, 2.9
    s.text(70, y0 - 12, "token/detik (sumbu mulai dari 2,0 agar selisih terlihat)", size=13, color=C["dim"])
    for t in (2.0, 2.2, 2.4, 2.6, 2.8):
        yy = y0 + H - H * (t - vmin) / (vmax - vmin)
        s.line(70, yy, 980, yy, C["line"], 1, dash="3,4")
        s.text(64, yy + 4, num(t, "%.1f"), size=12, anchor="end", color=C["dim"])
    for gi, (g, sub, vals) in enumerate(groups):
        gx = 90 + gi * 300
        for bi, v in enumerate(vals):
            bx = gx + bi * 74
            h = H * (v - vmin) / (vmax - vmin)
            s.rect(bx, y0 + H - h, 60, h, names[bi][1], stroke="none", rx=4)
            s.text(bx + 30, y0 + H - h - 8, ("%.2f" % v).replace(".", ","), size=14, weight="700", anchor="middle")
        s.text(gx + 104, y0 + H + 26, g, size=15, weight="700", anchor="middle")
        s.text(gx + 104, y0 + H + 46, sub, size=13, anchor="middle", color=C["dim"])
    s.line(70, y0 + H, 980, y0 + H, C["dim"], 1)
    lx = 600
    for n, col in names:
        s.rect(lx, 56, 14, 14, col, stroke="none", rx=2)
        s.text(lx + 20, 68, n, size=13)
        lx += 120
    s.text(24, 398, "Aturan: simpan tebakan bila peluangnya \u2265 0,5; tebak lagi bila \u2265 0,85; maksimal 4. Output selalu identik.", size=14)
    s.save("45_adaptive.svg")


def fig45_grain():
    s = Svg(1000, 330, "Layer CPU: ukuran potongan kerja per thread (lebih pendek = lebih cepat)")
    s.text(24, 62, "Waktu 32 layer CPU per token (ms). Batas bandwidth RAM terukur: 33 GB/s; sebelum: 25,2 GB/s (76 %).", size=14, color=C["dim"])
    hbars(s, 24, 84, 940, [("8 baris (baseline)", 298, C["base"], ""), ("32 baris", 289, "#4ac26b", ""), ("128 baris", 287, C["exp"], "dipakai"),
                           ("256 baris", 289, "#4ac26b", ""), ("512 baris", 310, C["warn"], "beban tidak merata"),
                           ("8 baris + software prefetch", 396, C["bad"], "ditolak")], 420, " ms", bar_h=26, gap=10, label_w=270, fmt="%.0f")
    s.save("45_grain.svg")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("44", "all"):
        fig44_architecture(); fig44_memory(); fig44_mtp_cycle(); fig44_benchmark(); fig44_progress()
    if which in ("45", "all"):
        fig45_prefill_timeline(); fig45_overlap(); fig45_ab(); fig45_experiments(); fig45_adaptive(); fig45_grain()
