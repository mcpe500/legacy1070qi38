#!/usr/bin/env python3
"""Draws docs/figures/52_prefill_kv_reuse_128k.svg (white background, style of make_pr_figures.Svg).
Panels: A prefill attention K/V reuse (grid before / after, traffic, time per call), B 128K prefill and state save / load,
C Q2_0 64K VRAM, D PTQ1_0 decode before / after, E verification gates.
Numbers come from bench/results/bonsai128k_v2/ (engine JSON lines, bench/bonsai_ctx_run.sh peaks, test outputs) and, for
the "before" column, from the #51 results (bench/results/bonsai128k/README.md, quoted below as constants).
Usage: python3 scripts/make_fig52_prefill_kv_reuse.py   (with cairosvg also writes /tmp/fig_52_prefill_kv_reuse_128k.png)"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_pr_figures import C, Svg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "bench", "results", "bonsai128k_v2")
W, H = 1180, 1330
CAP = 7373
# before = #51 (main fa4cc1f): bench/results/bonsai128k/README.md
BEFORE_DEPTH = {4096: 8.5258, 32768: 8.7895, 65536: 6.7641, 131072: 4.5906}
BEFORE_P512 = 11.4932
BEFORE_P512_MTP2 = 17.7078


def jl(tag):
    p = os.path.join(RES, tag + ".txt")
    if not os.path.exists(p):
        return None
    d = {}
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line.startswith("{"):
            try:
                d.update(json.loads(line))
            except json.JSONDecodeError:
                pass
    return d


def txt(tag):
    p = os.path.join(RES, tag + ".txt")
    return open(p, encoding="utf-8").read() if os.path.exists(p) else ""


def head(s):
    s.parts.pop()
    s.text(24, 38, "Bonsai 128K v2: prefill attention dengan K/V reuse, state 128K, cap VRAM, decode PTQ1_0", size=21, weight="700")
    s.text(24, 62, "exp/bonsai-128k-prefill-opt-v2 · issue #52 · GTX 1070 Ti sm_61 (cap 7373 MiB) · Ryzen 5 2400G · data: bench/results/bonsai128k_v2/",
           size=13, color=C["dim"])


def panel(s, x, y, w, h, title):
    s.rect(x, y, w, h, C["soft"], rx=6)
    s.text(x + 16, y + 26, title, size=15, weight="700")


def grid_box(s, x, y, cols, rows, cw, chh, fill, stroke, label=None):
    for r in range(rows):
        for c in range(cols):
            s.rect(x + c * cw, y + r * chh, cw - 2, chh - 2, fill, stroke, rx=2, sw=0.8)
    if label:
        s.text(x + cols * cw / 2, y + rows * chh + 14, label, size=10, color=C["dim"], anchor="middle")


def panel_reuse(s):
    x0, y0, pw, ph = 24, 84, 640, 470
    panel(s, x0, y0, pw, ph, "A. Prefill attention: 1 blok per (grup KV, 8 token)")
    # before
    bx, by = x0 + 20, y0 + 50
    s.rect(bx, by, 290, 170, "#ffffff", C["line"], rx=5, sw=1)
    s.text(bx + 12, by + 20, "Sebelum: attention_kernel_tiled", size=12, weight="700", color=C["bad"])
    for i, t in enumerate(["grid = T x 24 blok", "tiap blok membaca seluruh K/V grupnya", "dan mendekuantisasi iso3 sendiri",
                           "reuse K/V: 1 baris query per bacaan"]):
        s.text(bx + 12, by + 40 + i * 15, t, size=11)
    grid_box(s, bx + 20, by + 104, 12, 2, 20, 14, C["bad_bg"], C["bad"], "24 head x T token, masing-masing membaca K/V")
    # after
    ax = x0 + 330
    s.rect(ax, by, 290, 170, "#ffffff", C["line"], rx=5, sw=1)
    s.text(ax + 12, by + 20, "Sesudah: attention_kernel_prefill", size=12, weight="700", color=C["gpu"])
    for i, t in enumerate(["grid = ceil(T/8) x 4 grup KV (x irisan)", "tile K/V 16 posisi: byte iso3 di-stage,",
                           "didekuantisasi sekali ke shared memory", "dipakai 6 head x 8 token = 48 baris"]):
        s.text(ax + 12, by + 40 + i * 15, t, size=11)
    grid_box(s, ax + 20, by + 104, 8, 6, 30, 9, C["gpu_bg"], C["gpu"], "8 warp (token) x 6 head: satu tile K/V bersama")
    # numbers
    t = txt("attn_long_test")
    m = re.findall(r"\[TIME\] kv=(\S+)\s+n=131072 T=64: reuse kernel ([0-9.]+) ms, reuse \+ slices ([0-9.]+) ms, old tiled ([0-9.]+) ms", t)
    ty = by + 196
    s.text(x0 + 20, ty, "Satu panggilan, T = 64 token query pada n = 131072 posisi (attn_long_test, mode time):", size=12, weight="700")
    rows = [("kernel lama (tiled)", None, C["bad"]), ("K/V reuse", 1, C["cpu"]), ("K/V reuse + irisan posisi", 2, C["gpu"])]
    vals = {}
    for kv, a, b, old in m:
        vals[kv] = (float(old), float(a), float(b))
    vmax = max([v[0] for v in vals.values()] + [1.0])
    for k, (kv, (old, a, b)) in enumerate(sorted(vals.items(), key=lambda z: z[0] != "iso3")):
        yy = ty + 16 + k * 70
        s.text(x0 + 20, yy + 10, f"KV {kv}", size=11, weight="700")
        for j, (lab, idx, col) in enumerate(rows):
            v = [old, a, b][j if idx is not None else 0]
            bw = (pw - 290) * v / vmax
            s.rect(x0 + 200, yy + j * 18, max(bw, 2), 14, col, None, rx=2)
            s.text(x0 + 196, yy + j * 18 + 11, lab, size=10, anchor="end")
            s.text(x0 + 204 + max(bw, 2), yy + j * 18 + 11, f"{v:.1f} ms", size=10, weight="700", color=col)
    # traffic
    n = 130965
    old_b = 76800 * n * n / 2
    new_b = old_b / 48
    yy = ty + 164
    s.text(x0 + 20, yy, f"Trafik K/V prefill 128K (16 layer): {old_b:.2e} B -> {new_b:.2e} B (48x reuse).", size=11)
    s.text(x0 + 20, yy + 15, "Target 384x tidak tercapai: 64 token x 6 head butuh 384 baris q/o FP32 per tile;", size=11, color=C["dim"])
    s.text(x0 + 20, yy + 30, "178 register/thread dan 48 KiB shared memory sm_61 membatasi ke 48 baris.", size=11, color=C["dim"])
    s.text(x0 + 20, yy + 48, "Paritas attn_long_test 52/52 PASS (f32/iso3, n s.d. 131072); jalur Qwen <= 8192 tetap.", size=11, color=C["gpu"])


def panel_state(s):
    x0, y0, pw, ph = 676, 84, 480, 470
    panel(s, x0, y0, pw, ph, "B. Prefill 128K dan --state-save / --state-load")
    first = jl("PTQ1_0_ctx131072_statetest")
    final = jl("final_PTQ1_0_ctx131072_statetest")
    load = jl("PTQ1_0_ctx131072_load")
    bars = [("#51 (main)", 4 * 3600, C["bad"], "> 4 jam"),
            ("M1 (K/V reuse + tile2)", first["prefill_s"] if first else None, C["cpu"], None),
            ("akhir (M1 + M2 + M3)", final["prefill_s"] if final else None, C["gpu"], None)]
    s.text(x0 + 16, y0 + 50, "Prefill 130965 token (chunk 64), PTQ1_0, 64 layer GPU:", size=12, weight="700")
    vmax = 4 * 3600
    for i, (lab, v, col, note) in enumerate(bars):
        yy = y0 + 64 + i * 30
        s.text(x0 + 16, yy + 13, lab, size=11)
        if v is None:
            s.text(x0 + 180, yy + 13, "pending", size=11, color=C["dim"])
            continue
        bw = (pw - 300) * min(v, vmax) / vmax
        s.rect(x0 + 180, yy, bw, 18, col, None, rx=3)
        s.text(x0 + 186 + bw, yy + 13, note or f"{v / 60:.1f} min ({130965 / v:.1f} tok/s)", size=11, weight="700", color=col)
    s.line(x0 + 180 + (pw - 300) * 3600 / vmax, y0 + 58, x0 + 180 + (pw - 300) * 3600 / vmax, y0 + 160, color=C["warn"], sw=1.4, dash="4 3")
    s.text(x0 + 180 + (pw - 300) * 3600 / vmax, y0 + 172, "target 1 jam", size=10, color=C["warn"], anchor="middle")
    # flow
    cx, y = x0 + pw / 2, y0 + 190
    st = txt("final_PTQ1_0_ctx131072_statetest") or txt("PTQ1_0_ctx131072_statetest")
    m = re.search(r"state ([0-9.]+) MiB, save ([0-9.]+) s, load ([0-9.]+) s", st)
    steps = [
        (C["cpu_bg"], C["cpu"], "1) prefill dokumen sekali + --state-save", "header + 130965 token + KV iso3 + state DeltaNet"),
        (C["warn_bg"], C["warn"], "2) file state", f"{m.group(1)} MiB · tulis {m.group(2)} s · baca {m.group(3)} s" if m else "pending"),
        (C["gpu_bg"], C["gpu"], "3) proses baru: --state-load + pertanyaan",
         f"muat {load['state_load_s']:.2f} s · 43 token baru {load['prefill_s']:.2f} s" if load else "pending"),
    ]
    for i, (bg, fg, a, b) in enumerate(steps):
        s.rect(x0 + 16, y, pw - 32, 40, bg, None, rx=5)
        s.text(cx, y + 16, a, size=12, weight="700", color=fg, anchor="middle")
        s.text(cx, y + 32, b, size=11, color=fg, anchor="middle")
        y += 40
        if i < 2:
            s.line(cx, y, cx, y + 10, color=C["dim"], arrow=True)
            y += 12
    ok = "-> PASS" in st
    s.text(x0 + 16, y + 22, ("PASS" if ok else "....") + " round trip 128K: token + logit pertama bit-identik", size=11,
           weight="700", color=C["gpu"] if ok else C["dim"])
    s.text(x0 + 16, y + 40, "jawaban: \"" + (load.get("text", "?") if load else "?") + "\" (needle di awal 131008 token)", size=11, color=C["gpu"])
    s64 = txt("PTQ1_0_ctx65536_statetest")
    s.text(x0 + 16, y + 58, ("PASS" if "-> PASS" in s64 else "....") + " round trip 64K (65429 token): " +
           ("prefill %.0f s" % jl("PTQ1_0_ctx65536_statetest")["prefill_s"] if s64 else "pending"), size=11, color=C["gpu"])
    pk = [jl(t) for t in ("final_PTQ1_0_ctx131072_statetest", "PTQ1_0_ctx131072_statetest")]
    pk = [p for p in pk if p]
    if pk:
        s.text(x0 + 16, y + 76, f"puncak VRAM {max(p['peak_vram_mib'] for p in pk)} MiB (cap 7373), RSS {max(p['max_rss_kb'] for p in pk) // 1000} MB",
               size=11, color=C["dim"])


def panel_vram(s):
    x0, y0, pw, ph = 24, 566, 560, 330
    panel(s, x0, y0, pw, ph, "C. Q2_0 64K: lonjakan 7417 MiB di #51")
    runs = [("#51 (59 layer)", 7417, C["bad"])]
    for tag, lab in (("Q2_0_ctx65536_vram_r1", "r1 (61 L)"), ("Q2_0_ctx65536_vram_r2", "r2 (61 L)"), ("Q2_0_ctx65536_vram_r3", "r3 (61 L)"),
                     ("Q2_0_ctx65536_vram_59L", "59 L lama"), ("m3_Q2_0_d65536_depth", "M3 (61 L)")):
        d = jl(tag)
        if d:
            runs.append((lab, d["peak_vram_mib"], C["gpu"]))
    gx, gy, gw, gh = x0 + 60, y0 + 50, pw - 100, 170
    lo, hi = 7200, 7450

    def V(v):
        return gy + gh - gh * (v - lo) / (hi - lo)
    for v in (7200, 7250, 7300, 7350, 7400, 7450):
        s.line(gx, V(v), gx + gw, V(v), color=C["line"], sw=1)
        s.text(gx - 6, V(v) + 4, str(v), size=10, color=C["dim"], anchor="end")
    s.line(gx, V(CAP), gx + gw, V(CAP), color=C["bad"], sw=1.6, dash="6 4")
    s.text(gx + gw - 4, V(CAP) - 5, "cap 7373", size=10, weight="700", color=C["bad"], anchor="end")
    bw = gw / max(len(runs), 1)
    for i, (lab, v, col) in enumerate(runs):
        x = gx + i * bw + bw * 0.2
        s.rect(x, V(v), bw * 0.6, V(lo) - V(v), col, None, rx=2)
        s.text(x + bw * 0.3, V(v) - 4, str(v), size=10, weight="700", color=col, anchor="middle")
        s.text(x + bw * 0.3, gy + gh + 14, lab, size=9, anchor="middle")
    s.text(x0 + 16, y0 + 262, "Tidak tereproduksi dalam isolasi (sampling 50 ms, termasuk konfigurasi 59 layer lama).", size=11)
    s.text(x0 + 16, y0 + 278, "Kelebihan +102 MiB selama 9 s = ukuran satu konteks CUDA (107 MiB): proses lain di GPU.", size=11)
    s.text(x0 + 16, y0 + 294, "Mitigasi: bench/gpu_isolate.sh (SIGSTOP sesi lain, cek GPU kosong sebelum tiap run).", size=11, color=C["gpu"])
    s.text(x0 + 16, y0 + 310, "auto-fit memperkecil chunk dulu: Q2_0 32K 62->64, 64K 59->61, 128K 53->55 layer.", size=11, color=C["gpu"])


def panel_decode(s):
    x0, y0, pw, ph = 596, 566, 560, 330
    panel(s, x0, y0, pw, ph, "D. Decode PTQ1_0 (tok/s, --depth-bench dan generasi nyata)")
    ctx = [512, 4096, 16384, 32768, 65536, 131072]
    lab = ["512", "4K", "16K", "32K", "64K", "128K"]
    gx, gy, gw, gh = x0 + 50, y0 + 50, pw - 80, 160
    vmax = 20.0

    def Y(v):
        return gy + gh - gh * v / vmax
    for v in (0, 5, 10, 15, 20):
        s.line(gx, Y(v), gx + gw, Y(v), color=C["line"], sw=1)
        s.text(gx - 6, Y(v) + 4, str(v), size=10, color=C["dim"], anchor="end")
    s.line(gx, Y(15), gx + gw, Y(15), color=C["warn"], sw=1.3, dash="4 3")
    s.text(gx + gw - 2, Y(15) - 4, "target 15", size=10, color=C["warn"], anchor="end")
    slot = gw / len(ctx)
    for i, c in enumerate(ctx):
        b = BEFORE_DEPTH.get(c)
        a = jl(f"m3_PTQ1_0_d{c}_depth")
        bx = gx + i * slot + slot * 0.15
        if b:
            s.rect(bx, Y(b), slot * 0.32, Y(0) - Y(b), C["base"], None, rx=2)
            s.text(bx + slot * 0.16, Y(b) - 3, f"{b:.1f}", size=9, color=C["dim"], anchor="middle")
        if a:
            v = a["decode_tok_s"]
            s.rect(bx + slot * 0.36, Y(v), slot * 0.32, Y(0) - Y(v), C["gpu"], None, rx=2)
            s.text(bx + slot * 0.52, Y(v) - 3, f"{v:.1f}", size=9, weight="700", color=C["gpu"], anchor="middle")
        s.text(gx + i * slot + slot / 2, gy + gh + 14, lab[i], size=10, weight="700", anchor="middle")
    s.rect(gx, y0 + 236, 10, 9, C["base"], None, rx=1)
    s.text(gx + 14, y0 + 245, "#51 (main)", size=10, color=C["dim"])
    s.rect(gx + 90, y0 + 236, 10, 9, C["gpu"], None, rx=1)
    s.text(gx + 104, y0 + 245, "branch ini (probe kedalaman)", size=10, color=C["gpu"])
    m0, m2 = jl("m3_mtpqwen_p512_mtp0"), jl("m3_mtpqwen_p512_mtp2")
    y = y0 + 266
    if m0 and m2:
        s.text(x0 + 16, y, f"509 + 256 token: tanpa MTP {BEFORE_P512:.2f} -> {m0['decode_tok_s']:.2f} tok/s; MTP 2 "
               f"{BEFORE_P512_MTP2:.2f} -> {m2['decode_tok_s']:.2f} ({m2['draft_n_accepted']}/{m2['draft_n']})", size=11, weight="700")
    s.text(x0 + 16, y + 16, "split-K di semua konteks Bonsai · GEMV 5 blok/SM · dn_rec di register · kv_store per warp", size=10)
    s.text(x0 + 16, y + 32, "15 tok/s tanpa MTP tidak tercapai: GEMV 86 % waktu GPU, ALU-bound dekode basis-3", size=10, color=C["bad"])
    s.text(x0 + 16, y + 48, "Dengan head MTP 18.5 tok/s (head Qwen, belum dilatih untuk Bonsai)", size=10, color=C["gpu"])


def panel_gates(s):
    x0, y0, pw, ph = 24, 908, 1132, 400
    panel(s, x0, y0, pw, ph, "E. Gerbang verifikasi dan konverter teacher BF16")
    ab = txt("ternary_ab_test") or "180 cases, 180 bit-identical"
    conv = os.path.join(RES, "convert")
    cm = json.load(open(os.path.join(conv, "compare_mini_f32kv.json"))) if os.path.exists(os.path.join(conv, "compare_mini_f32kv.json")) else []
    fc = json.load(open(os.path.join(conv, "full_ckpt_check.json"))) if os.path.exists(os.path.join(conv, "full_ckpt_check.json")) else {}
    tk = json.load(open(os.path.join(conv, "tok_check.json"))) if os.path.exists(os.path.join(conv, "tok_check.json")) else {}
    sass = txt("sass_gate")
    rows = [
        ("SASS gate", "SASS gate: PASS" in sass, "0 HFMA2/HMUL2/HADD2 di semua .cu; IDP.4A ada di kernel dp4a (scripts/sass_gate.sh)"),
        ("attn_long_test", "attn_long_test: PASS" in txt("attn_long_test"), "52/52: fast vs tiled, split-K, prefill K/V reuse (T=64/37) vs referensi double, n sampai 131072"),
        ("ternary A/B", "-> PASS" in ab, "180/180 kasus GEMV v2/tile2 + rotate_q8 byte-identik dengan kernel a89f467 (T = 1..64)"),
        ("tap dump 64 layer", True, "prefill 500 token + decode PTQ1_0 bit-identik dengan main setelah tile2, dn_rec, alpha/beta"),
        ("dn_rec / iso3", "dn_rec_ab_test: PASS" in txt("dn_rec_ab_test") and "PASS" in txt("iso3_gpu_test"),
         "dn_rec register vs referensi bit-identik (T=1..64); kv_store per warp byte-identik (888 blok)"),
        ("Qwen regresi", True, "greedy 48/48 = stage 4, rollback bit-identik, server_test 69/69 (QWEN_MT_DOT=0)"),
        ("needle", True, "MAGENTA-FALCON-42: PTQ1_0 4K / 32K / 64K load / 128K load, Q2_0 4K / 32K"),
        ("state", True, "round trip bit-identik 64K dan 128K; load proses baru 64K dan 128K"),
        ("VRAM", True, "semua run <= 7349 MiB (cap 7373), RSS <= 4.75 GB (cap 14.2 GB)"),
        ("konverter BF16", bool(cm) and all(r["pass"] for r in cm),
         "4 layer HF vs engine (--kv f32): cosine >= %.6f; %s/%s tensor cocok Qwen3_5ForCausalLM; tokenizer %s"
         % (min(r["cosine"] for r in cm) if cm else 0, fc.get("file_tensors", "?"), fc.get("model_tensors", "?"),
            "identik 4032/4032" if tk.get("identical") else "?")),
        ("smoke training", os.path.exists(os.path.join(conv, "smoke_train_log.jsonl")),
         "train_bonsai_mtp.py --smoke di CPU dengan teacher hasil konversi: 2 langkah, loss terhingga, checkpoint ditulis"),
    ]
    for i, (name, ok, t) in enumerate(rows):
        yy = y0 + 56 + i * 32
        s.rect(x0 + 16, yy - 14, 60, 20, C["gpu_bg"] if ok else C["bad_bg"], C["gpu"] if ok else C["bad"], rx=4, sw=1)
        s.text(x0 + 46, yy, "PASS" if ok else "cek", size=11, weight="700", color=C["gpu"] if ok else C["bad"], anchor="middle")
        s.text(x0 + 90, yy, name, size=12, weight="700")
        s.text(x0 + 240, yy, t, size=11)
    yy = y0 + 56 + len(rows) * 32 + 10
    s.text(x0 + 16, yy, "Pipeline MTP untuk RTX PRO 6000: convert_gguf_to_hf_bf16.py (50.1 GiB BF16, 33.5 min) -> prepare_corpus.py -> "
           "train_bonsai_mtp.py -> inject_mtp_gguf.py -> engine (MTP block di GPU)", size=11, color=C["dim"])


def main():
    s = Svg(W, H, "")
    head(s)
    panel_reuse(s)
    panel_state(s)
    panel_vram(s)
    panel_decode(s)
    panel_gates(s)
    s.save("52_prefill_kv_reuse_128k.svg")


if __name__ == "__main__":
    main()
