#!/usr/bin/env python3
"""Figure 57: marathon v2 results (exp/bonsai-overnight-v2-prefill-t17-drafter, issue #57).

Panels:
  A. M1: the opt-in PTQ1_0 fast prefill path is a measured loss (paired in-process A/B).
  B. M2: the T=17 verification batch is lossless, and its per-row latency is what sizes a DFlash drafter.
  C. M3: the cascaded Suffix-PLD (1..16 drafts) never beats r3 MTP at 4K.
  D. M4: the Q2_K __dp4a kernel reaches 98% of the Q4_K MAC rate; the 4-arch compile check.
  E. The gates that ran, and the two targets that were not met.

Every number is copied from bench/results/bonsai_overnight_v2/ (M1-FAST-PREFILL.md, M2-VERIFY-T17.md,
M3-SUFFIX-PLD-CASCADE.md, M4-Q2K-AND-ARCH-DISPATCH.md, M5-DRAFTER-V2-PIPELINE.md).
Usage: scripts/make_fig57_bonsai_fastprefill_t17_drafter.py
Writes docs/figures/57_bonsai_fastprefill_t17_drafter.svg
"""
import os

W, H = 1200, 1480
F = "Segoe UI, Helvetica Neue, Helvetica, Arial, DejaVu Sans, sans-serif"
INK, MUT, GRID, BG, RED, GREEN, BLUE, PURPLE, GREY = "#1f2328", "#57606a", "#d0d7de", "#f6f8fa", "#cf222e", "#1a7f37", "#0969da", "#8250df", "#8c959f"
out = []


def esc(s):
    return s.replace("&;", "&;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=11, fill=INK, anchor="start", weight=400):
    out.append(f'<text x="{x}" y="{y}" font-family="{F}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{esc(s)}</text>')


def rect(x, y, w, h, fill, rx=2):
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="none" stroke-width="1.5"/>')


def line(x1, y1, x2, y2, c=GRID, w=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="{w}"{d}/>')


def panel(x, y, w, h, title):
    rect(x, y, w, h, BG, 6)
    text(x + 16, y + 26, title, 15, INK, weight=700)




# ---- canvas
out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img"><defs></defs>')
rect(0, 0, W, H, "#ffffff", 0)
text(24, 38, "Maraton v2: dinding prefill 128K, verifikasi T=17 lossless, Suffix-PLD/Drafter v2", 21, INK, weight=700)
text(24, 62, "exp/bonsai-overnight-v2-prefill-t17-drafter · issue #57 · GTX 1070 Ti sm_61 (cap 7373 MiB) · data: bench/results/bonsai_overnight_v2/", 13, MUT)

# ============================== A. M1: fast prefill is a loss ==============================
panel(24, 84, 1152, 320, "A. M1: varian prefill PTQ1_0 urutan FP32 direlaksasi (fast) — kerugian terukur, tidak dipasang")
text(40, 126, "A/B berpasangan dalam satu proses (7 ronde, terbaik tiap sisi), bentuk ffn_up 5120×17408", 12, INK)
base_y = 296.0   # 1.00x baseline
line(60, base_y, 1140, base_y, MUT, 1.0, "4,3")
text(56, base_y + 4, "1.00x", 10, MUT, "end")
bars = [
    ("<16,2>", "T=16", 0.708, BLUE), ("<16,2>", "T=64", 0.708, BLUE),
    ("<16,1>", "T=16", 0.985, PURPLE), ("<16,1>", "T=64", 1.000, PURPLE),
    ("<32,1>", "T=16", 1.000, GREEN), ("<32,1>", "T=64", 1.000, GREEN),
    ("<32,2>", "T=16", 1.000, GREEN), ("<32,2>", "T=64", 1.000, GREEN),
    ("<8,2>", "T=16", 0.967, GREY), ("<8,2>", "T=64", 0.997, GREY),
    ("<16,4>", "T=16", 0.952, "#9a6700"), ("<16,4>", "T=64", 1.000, "#9a6700"),
]
bx, bw, gap = 84.0, 62.0, 22.0
for i, (tile, tl, sp, col) in enumerate(bars):
    h = max(2.0, (sp - 0.5) / 0.5 * 140.0)          # 0.5x -> 2 px, 1.0x -> 140 px
    x = bx + i * (bw + gap)
    rect(x, base_y - h, bw, h, col)
    text(x + bw / 2, base_y - h - 8, f"{sp:.3f}x", 9, INK, "middle")
    text(x + bw / 2, base_y + 14, tile, 9, INK, "middle")
    text(x + bw / 2, base_y + 26, tl, 9, MUT, "middle")
text(84, 352, "Tidak satu pun tile mencapai paritas: yang terbaik 1.000x, terburuk 0.708x.", 11, INK)
text(84, 370, "Root cause: tile2 <16,2> terbatas occupancy — 127 register × 256 thread = 32512 dari 65536 register per SM,", 11, INK)
text(84, 386, "tepat 2 blok/SM. Memaksa 3 blok (__launch_bounds__(256,3)) → 80 register (spill) dan T=16 946,6 µs alih-alih 788,3.", 11, INK)
text(84, 402, "Jalur default tetap bit-identik: ternary_ab_test 180/180. Gate numerik lulus (cos 1.00000000), jadi flag tetap ada.", 11, MUT)

# ============================== B. M2: T=17 verify ==============================
panel(24, 424, 600, 430, "B. M2: batch verifikasi T=17 lossless dan latensinya")
text(40, 470, "Baris 0..16 batch T=17, baris yang sama dari batch T=5, dan 17 run satu-baris", 11, INK)
text(40, 486, "pada posisi yang sama: byte-identical di ctx 512/4096/32768/65536 (iso3).", 11, INK)
rows = [
    ("512", 57.8, 131.3, 438.5, 57.8, 26.3, 25.8),
    ("4096", 450.7, 1289.2, 3820.6, 450.7, 257.8, 224.7),
    ("32768", 1324.8, 3625.3, 7302.2, 1324.8, 725.1, 429.5),
    ("65536", 2613.7, 7319.7, 14719.4, 2613.7, 1463.9, 865.8),
]
ty = 512
text(48, ty, "ctx", 10, MUT, weight=700)
text(140, ty, "T=1 µs", 10, MUT, weight=700)
text(248, ty, "T=5 µs", 10, MUT, weight=700)
text(356, ty, "T=17 µs", 10, MUT, weight=700)
text(462, ty, "µs/baris  T=1 / T=5 / T=17", 10, MUT, weight=700)
line(48, ty + 6, 600, ty + 6, GRID, 0.8)
for i, (ctx, u1, u5, u17, p1, p5, p17) in enumerate(rows):
    y = ty + 30 + i * 24
    text(48, y, ctx, 10, INK)
    text(140, y, f"{u1:,.1f}".replace(",", "."), 10, INK)
    text(248, y, f"{u5:,.1f}".replace(",", "."), 10, INK)
    text(356, y, f"{u17:,.1f}".replace(",", "."), 10, INK)
    text(462, y, f"{p1:.0f} / {p5:.0f} / {p17:.0f}", 10, GREEN)
text(40, 640, "Batch 17 baris menghemat 41% biaya per baris dibanding batch 5 baris (865,8 vs 1463,9 µs", 11, INK)
text(40, 656, "pada 64K): itulah plafon untuk drafter DFlash B=16.", 11, INK)
text(40, 676, "Buffer: partial 40,05 MiB (17 × 24 × 100 × 258 × 4 B), rollback 51,3 MiB — keduanya", 11, INK)
text(40, 692, "dipilih dinamis oleh anggaran (17 → 8 → 5), dan pilihan itu tercetak di JSON ringkasan.", 11, INK)
text(40, 712, "Pilihan engine: ctx 4K/32K/64K/128K semua verify_t 17 (vram 6574/7002/7352 MiB, cap 7373);", 11, MUT)
text(40, 728, "anggaran 7500 MiB turun ke 5 (probe: 17 → 7531,5, 8 → 7504,3, 5 → 7474,0).", 11, MUT)
text(40, 748, "rollback: keep = 1,2,3,4,5,8,12,16,17 byte-identical (dn_rollback_test 9/9).", 11, MUT)

# ============================== C. M3: cascade ==============================
panel(640, 424, 536, 430, "C. M3: cascaded Suffix-PLD (1..16 draft) vs r3 MTP, ctx 4096")
text(656, 470, "decode tok/s, greedy, n=64; garis putus-putus = acuan r3 MTP", 11, INK)
pmax = [2, 4, 6, 8, 16]
series = [
    ("refactoring", [19.28, 18.81, 18.14, 17.85, 17.91], GREEN, 19.39),
    ("coding C++", [22.20, 22.33, 22.32, 22.15, 22.36], BLUE, 22.27),
    ("chat EN", [20.45, 20.46, 20.47, 20.42, 20.47], PURPLE, 20.56),
    ("chat ID", [19.76, 18.86, 15.82, 5.36, 3.29], RED, 20.19),
]
x0, x1, ytop, ybot = 740.0, 1140.0, 586.0, 700.0
lo, hi = 0.0, 24.0
for gv in (5, 10, 15, 20):
    y = ybot - (gv - lo) / (hi - lo) * (ybot - ytop)
    line(x0, y, x1, y, GRID, 0.8)
    text(x0 - 8, y + 4, str(gv), 9, MUT, "end")
line(x0, ybot, x1, ybot, MUT, 1.0)
line(x0, ytop, x0, ybot, MUT, 1.0)
for name, vals, col, r in series:
    pts = [(x0 + (x1 - x0) * i / (len(pmax) - 1), ybot - (v - lo) / (hi - lo) * (ybot - ytop)) for i, v in enumerate(vals)]
    for i in range(len(pts) - 1):
        line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], col, 2.2)
    ry = ybot - (r - lo) / (hi - lo) * (ybot - ytop)
    line(pts[0][0], ry, pts[-1][0], ry, col, 0.9, "3,3")
    for i in range(len(pts)):
        rect(pts[i][0] - 3, pts[i][1] - 3, 6, 6, col, 3)
for i, p in enumerate(pmax):
    x = x0 + (x1 - x0) * i / (len(pmax) - 1)
    text(x, ybot + 18, str(p), 10, INK, "middle")
text(x0 + (x1 - x0) / 2, ybot + 32, "pld-max (draft per siklus)", 10, INK, "middle")
# legend with the two endpoints of each series (no per-point labels: the values collide at 4K)
ly = 506
for name, vals, col, r in series:
    rect(660, ly - 9, 10, 10, col, 2)
    text(678, ly, f"{name}  {vals[0]:.2f} → {vals[-1]:.2f}".replace(".", ","), 10, INK)
    text(876, ly, f"acuan {r:.2f}".replace(".", ","), 10, MUT)
    ly += 18
text(656, 756, "Regresi parah hanya pada chat ID: satu-satunya workload yang sering match,", 11, INK)
text(656, 772, "dan daftar draft panjangnya sebagian besar ditolak (5,36 tok/s di pld-max 8).", 11, INK)
text(656, 788, "Default pld_max tetap 4; knob sampai 16. Langkah PLD 0,39–0,63 µs (anggaran 200),", 11, INK)
text(656, 804, "0 MiB VRAM. Cascade == r3 MTP (ids byte-identical di 4 workload).", 11, INK)

# ============================== D. M4: Q2_K kernel + arch check ==============================
panel(24, 866, 600, 330, "D. M4: kernel Q2_K GPU + dispatch arsitektur")
text(40, 912, "G int8 MAC/s pada T=64, bentuk ffn_up 5120×17408", 11, INK)
fmts = [("Q2_K (baru)", 1071, BLUE), ("Q3_K", 892, GREY), ("Q4_K", 1091, PURPLE)]
by = 1046.0
mx = 1200.0
for i, (n, v, c) in enumerate(fmts):
    h = v / mx * 100.0
    x = 90 + i * 150
    rect(x, by - h, 70, h, c)
    text(x + 35, by - h - 8, f"{v}", 10, INK, "middle")
    text(x + 35, by + 14, n, 10, INK, "middle")
line(84, by, 540, by, MUT, 1.0)
text(84, by + 36, "98% dari laju MAC Q4_K, dan 1,72x byte bobot per detik pada 2,62 bit/bobot.", 11, INK)
text(84, by + 54, "Paritas vs dequantizer libggml: rel_rms 8,2e-08 .. 8,4e-08 (T=1/3/16/64).", 11, INK)
text(84, by + 72, "Dua kesalahan layout sebelum benar: d/dmin di offset 80/82 (bukan 0/2), payload", 11, INK)
text(84, by + 88, "hanya 4-byte aligned (blok 84 byte → tidak boleh uint4), nibble tinggi byte skala", 11, INK)
text(84, by + 104, "digeser logis, dan dua paruh 16 nilai butuh koreksi minimum masing-masing.", 11, INK)
text(84, by + 124, "arch_compile_check: 28/28 pasangan (sm_61, sm_75, sm_86, sm_89) × 7 kernel compile.", 11, GREEN, weight=700)
text(84, by + 142, "Cabang __CUDA_ARCH__ nyata: 4 blok/SM minimum hanya untuk sm_90+; build sm_61 tak berubah.", 11, MUT)

# ============================== E. gates + misses ==============================
panel(640, 866, 536, 330, "E. Gate yang pernah dijalankan di branch ini")
gates = [
    ("ternary_ab_test", "180/180 bit-identical", GREEN),
    ("ptq1_fast_gate", "42/42, cos 1.00000000", GREEN),
    ("verify_t17_test", "32/32 lossless T=17", GREEN),
    ("dn_rollback_test", "9/9 keep 1..17", GREEN),
    ("q2_k_parity", "4/4 rel 8.4e-08", GREEN),
    ("attn_long_test", "PASS, reuse 98,27 ms vs 2179,82 ms", GREEN),
    ("ternary_test / dn_rec_ab_test", "PASS / PASS", GREEN),
    ("SASS gate", "IDP.4A x6205, half 0", GREEN),
    ("verify_m5.sh (CPU)", "6/6, trainer selftest PASS", GREEN),
    ("Target prefill 128K < 45 mnt", "tidak tercapai (~60,4 mnt)", RED),
    ("Target greedy lossless MTP", "belum: divergensi non-MTP vs MTP", RED),
]
for i, (n, v, c) in enumerate(gates):
    y = 914 + i * 26
    rect(656, y - 10, 10, 10, c, 2)
    text(676, y, n, 11, INK, weight=700)
    text(876, y, v, 11, c if c == RED else MUT)

text(24, 1216, "Catatan jujur untuk dua target merah: (1) prefill 128K tetap ~60,4 menit — bagian GEMM (~2195 s) sudah di floor formulasi ini", 11, INK)
text(24, 1234, "setelah 11 percobaan terdokumentasi (10 di M3-ATTEMPTS.md cabetang sebelumnya, satu di M1 branch ini); (2) lossless MTP belum —", 11, INK)
text(24, 1252, "near-tie argmax antara baris verifikasi dan decode satu-token, bukan efek batch 17 baris, dan sudah ada sebelum branch ini.", 11, INK)
text(24, 1272, "Yang berhasil dan bertahan: jalur default bit-identik/greedy parity pada semua tes, verifikasi T=17 lossless dengan buffer dinamis,", 11, INK)
text(24, 1288, "kernel Q2_K GPU, compile check 4 arsitektur, dan pipeline Drafter v2 yang sinkron dengan kontrak dan bisa direproduksi di CPU.", 11, MUT)
text(24, 1306, "Sumber angka: bench/results/bonsai_overnight_v2/M1-FAST-PREFILL.md, M2-VERIFY-T17.md, M3-SUFFIX-PLD-CASCADE.md,", 10, MUT)
text(24, 1320, "M4-Q2K-AND-ARCH-DISPATCH.md, M5-DRAFTER-V2-PIPELINE.md.", 10, MUT)

out.append("</svg>")
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(here, "docs", "figures", "57_bonsai_fastprefill_t17_drafter.svg")
with open(p, "w") as f:
    f.write("\n".join(out) + "\n")
print(f"wrote {p} ({len(out)} elements)")
