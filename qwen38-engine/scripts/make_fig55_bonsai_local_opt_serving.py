#!/usr/bin/env python3
"""Draws docs/figures/55_bonsai_local_opt_serving.svg (white background, style of make_pr_figures.Svg) for issue #55.
Panels: A M1 MTP speedup per suite (r3 Q4_K_M / donor / Q8_0), B M2 decode at 509 tokens, C M2 depth probe before / after,
D M3 128K prefill, E M4 server and the verification gates.
Numbers come from bench/results/bonsai_local_10h/ (suite JSONL, engine JSON lines, test logs); the PR #53 "before" values
are quoted as constants (bench/results/bonsai128k_v2/, docs/AFK_STATUS.md).
Usage: python3 scripts/make_fig55_bonsai_local_opt_serving.py"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_pr_figures import C, Svg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "bench", "results", "bonsai_local_10h")
W, H = 1200, 1240
BEFORE_DEPTH = {512: 12.74, 4096: 12.68, 16384: 11.02, 32768: 9.29, 65536: 7.06, 131072: 4.72}   # PR #53
BEFORE_P512, BEFORE_P512_MTP = 13.30, 18.48                                                     # PR #53 (Qwen blk.64, MTP 2)
BEFORE_PREFILL_S, BEFORE_GEMM_S, BEFORE_ATTN_S = 3774.7, 2195.0, 1580.0                       # PR #53, nsys split
SUITES = [("coding", "coding"), ("chat_en", "chat EN"), ("chat_id", "chat ID"), ("repetitive", "repetitif")]


def jl(tag):
    p = os.path.join(RES, tag + ".txt")
    d = {}
    if not os.path.exists(p):
        return d
    for line in open(p, encoding="utf-8", errors="replace"):
        line = line.strip()
        if line.startswith("{"):
            try:
                d.update(json.loads(line))
            except json.JSONDecodeError:
                pass
    return d


def suite_rows(size):
    rows = {}
    p = os.path.join(RES, f"suite_{size}.jsonl")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            rows[(r["suite"], r["head"], r["variant"])] = r
    return rows


def panel(s, x, y, w, h, title):
    s.rect(x, y, w, h, C["soft"], rx=6)
    s.text(x + 16, y + 26, title, size=15, weight="700")


def panel_m1(s):
    x0, y0, pw, ph = 24, 84, 1152, 300
    panel(s, x0, y0, pw, ph, "A. M1: speedup MTP per suite (greedy, 256 token, semua run identik dengan tanpa MTP)")
    r512, r4k = suite_rows(512), suite_rows(4096)
    st = {}
    for suite in ("coding", "repetitive"):
        for v in ("m0", "ad"):
            d = jl(f"st_32768_{suite}_{'none' if v == 'm0' else 'q4km'}_{v}")
            st[(suite, v)] = d.get("decode_tok_s")
    groups = []
    for key, name in SUITES:
        def sp(rows, head, v, base_head="none"):
            r, b = rows.get((key, head, v)), rows.get((key, base_head, "m0"))
            return r["tg_tok_s"] / b["tg_tok_s"] if r and b else None
        g = [("512 q4km m1", sp(r512, "q4km", "m1", "q4km")), ("512 q4km ad", sp(r512, "q4km", "ad", "q4km")),
             ("512 donor ad", sp(r512, "donor", "ad", "donor")), ("512 q8 ad", sp(r512, "q8", "ad", "q8")),
             ("512 q4km pldad", sp(r512, "q4km", "pldad", "q4km")),
             ("4K q4km ad", sp(r4k, "q4km", "ad")), ("4K q4km m2", sp(r4k, "q4km", "m2"))]
        if (key, "ad") in st and st[(key, "ad")] and st[(key, "m0")]:
            g.append(("32K q4km ad", st[(key, "ad")] / st[(key, "m0")]))
        groups.append((name, g))
    colors = {"q4km": C["gpu"], "donor": C["cpu"], "q8": C["mtp"], "pldad": C["warn"]}
    bx, by, bh = x0 + 40, y0 + 250, 190   # baseline y of the bars (1.0x at by - scale(1.0))
    scale = bh / 1.6
    gw = (pw - 80) / len(groups)
    s.line(bx - 10, by - scale * 1.0, x0 + pw - 20, by - scale * 1.0, C["dim"], sw=1, dash="4,3")
    s.text(bx - 14, by - scale * 1.0 + 4, "1.0x", size=10, color=C["dim"], anchor="end")
    for gi, (name, g) in enumerate(groups):
        gx = bx + gi * gw
        bw = (gw - 30) / max(1, len(g))
        for i, (lab, v) in enumerate(g):
            if v is None:
                continue
            col = colors["pldad"] if "pld" in lab else colors["donor"] if "donor" in lab else colors["q8"] if "q8" in lab else colors["q4km"]
            hgt = scale * min(v, 1.6)
            s.rect(gx + i * bw, by - hgt, bw - 3, hgt, col, rx=2)
            s.text(gx + i * bw + (bw - 3) / 2, by - hgt - 4, "%.2f" % v, size=9, anchor="middle")
        s.text(gx + (gw - 30) / 2, by + 16, name, size=12, weight="700", anchor="middle")
    lx = x0 + 20
    for i, (lab, col) in enumerate([("r3 Q4_K_M", C["gpu"]), ("donor Qwen blk.64", C["cpu"]), ("r3 Q8_0", C["mtp"]), ("PLD + MTP adaptive", C["warn"])]):
        s.rect(lx + i * 190, y0 + 40, 12, 12, col, rx=2)
        s.text(lx + i * 190 + 18, y0 + 51, lab, size=11)
    s.text(x0 + 780, y0 + 51, "bar per grup: 512 m1 / ad / donor ad / q8 ad / pldad, 4K ad / m2, 32K ad", size=10, color=C["dim"])


def panel_m2(s):
    x0, y0, pw, ph = 24, 400, 560, 380
    panel(s, x0, y0, pw, ph, "B. M2: decode PTQ1_0, prompt 509 + 256 token")
    a = jl("m2_dec2_p512_mtp0").get("decode_tok_s") or jl("m2_v3_p512_mtp0_a").get("decode_tok_s")
    b = jl("m2_dec2_p512_q4km_ad").get("decode_tok_s") or jl("m2_v3_p512_q4km_ad").get("decode_tok_s")
    bars = [("PR #53, tanpa MTP", BEFORE_P512, C["base"]), ("PR #53, MTP 2 (Qwen blk.64)", BEFORE_P512_MTP, C["base"]),
            ("v3 GEMV, tanpa MTP", a, C["gpu"]), ("v3 + r3 Q4_K_M adaptive", b, C["mtp"]),
            ("v3 + r3 Q8_0 adaptive (Q8 batched)", jl("q8bt_p512_q8head_ad").get("decode_tok_s"), C["mtp"])]
    mx = 30.0
    for i, (lab, v, col) in enumerate(bars):
        yy = y0 + 56 + i * 46
        s.text(x0 + 16, yy + 16, lab, size=12)
        if v:
            w = (pw - 260) * v / mx
            s.rect(x0 + 220, yy, w, 26, col, rx=3)
            s.text(x0 + 226 + w, yy + 18, "%.2f tok/s" % v, size=12, weight="700")
    s.line(x0 + 220 + (pw - 260) * 15 / mx, y0 + 50, x0 + 220 + (pw - 260) * 15 / mx, y0 + 270, C["bad"], sw=1.2, dash="4,3")
    s.text(x0 + 224 + (pw - 260) * 15 / mx, y0 + 284, "target 15 tok/s", size=10, color=C["bad"])
    s.lines(x0 + 16, y0 + 312, ["GEMV v3: decode digit TQ1_0 aritmetis, bukan tabel shared memory;",
                                "bit-identik (ternary_ab_test 180/180), ids 509-token identik (26a58a94);",
                                "ffn_up T=1 208 -> 150 us, lm_head 2778 -> 1997 us, T=5 592 -> 313 us"], size=11, color=C["dim"], gap=16)


def panel_depth(s):
    x0, y0, pw, ph = 600, 400, 576, 380
    panel(s, x0, y0, pw, ph, "C. M2: depth probe tanpa MTP (tok/s per token pada posisi N)")
    after = {d: jl(f"m2_depth2_d{d}").get("decode_tok_s") for d in BEFORE_DEPTH}
    px, py, gw, gh = x0 + 60, y0 + 60, pw - 100, 240
    mx = 20.0
    s.line(px, py + gh, px + gw, py + gh, C["dim"], sw=1)
    s.line(px, py, px, py + gh, C["dim"], sw=1)
    for v in (5, 10, 15, 20):
        yy = py + gh - gh * v / mx
        s.line(px, yy, px + gw, yy, C["line"], sw=0.8)
        s.text(px - 6, yy + 4, str(v), size=10, color=C["dim"], anchor="end")
    ds = sorted(BEFORE_DEPTH)
    for series, col, lab in ((BEFORE_DEPTH, C["base"], "PR #53"), (after, C["gpu"], "sekarang (GEMV v3 + attention decode v2)")):
        pts = []
        for i, d in enumerate(ds):
            v = series.get(d)
            if v:
                pts.append((px + gw * i / (len(ds) - 1), py + gh - gh * v / mx, v))
        for a, b in zip(pts, pts[1:]):
            s.line(a[0], a[1], b[0], b[1], col, sw=2.2)
        for x, y, v in pts:
            s.rect(x - 3, y - 3, 6, 6, col, rx=3)
            s.text(x, y - 8, "%.2f" % v, size=10, color=col, anchor="middle", weight="700")
    for i, d in enumerate(ds):
        s.text(px + gw * i / (len(ds) - 1), py + gh + 16, "%dK" % (d // 1024) if d >= 1024 else str(d), size=10, anchor="middle")
    s.rect(px + 10, y0 + 330, 12, 12, C["base"], rx=2)
    s.text(px + 28, y0 + 341, "PR #53", size=11)
    s.rect(px + 110, y0 + 330, 12, 12, C["gpu"], rx=2)
    s.text(px + 128, y0 + 341, "sekarang: GEMV v3 + attention decode v2 (rotasi iso3 di luar loop)", size=11)


def panel_m3(s):
    x0, y0, pw, ph = 24, 796, 560, 420
    panel(s, x0, y0, pw, ph, "D. M3: prefill 128K (130965 token, PTQ1_0)")
    d = jl("m3_final_PTQ1_0_ctx131072_statetest") or jl("m3_PTQ1_0_ctx131072_statetest")
    now = d.get("prefill_s")
    s.text(x0 + 16, y0 + 56, "PR #53: %.1f min" % (BEFORE_PREFILL_S / 60), size=13, weight="700", color=C["base"])
    mx = BEFORE_PREFILL_S * 1.05
    w = (pw - 40)
    s.rect(x0 + 16, y0 + 66, w * BEFORE_GEMM_S / mx, 26, C["cpu"], rx=2)
    s.rect(x0 + 16 + w * BEFORE_GEMM_S / mx, y0 + 66, w * BEFORE_ATTN_S / mx, 26, C["mtp"], rx=2)
    s.text(x0 + 22, y0 + 84, "GEMM ~2195 s", size=11, color="#ffffff", weight="700")
    s.text(x0 + 22 + w * BEFORE_GEMM_S / mx, y0 + 84, "attention ~1580 s", size=11, color="#ffffff", weight="700")
    if now:
        s.text(x0 + 16, y0 + 126, "sekarang: %.1f min (%.1f tok/s)" % (now / 60, d.get("prefill_tok_s", 0)), size=13, weight="700", color=C["gpu"])
        s.rect(x0 + 16, y0 + 136, w * now / mx, 26, C["gpu"], rx=2)
    s.line(x0 + 16 + w * 3000 / mx, y0 + 50, x0 + 16 + w * 3000 / mx, y0 + 172, C["bad"], sw=1.2, dash="4,3")
    s.text(x0 + 20 + w * 3000 / mx, y0 + 186, "target 50 min", size=10, color=C["bad"])
    rows = ["attention prefill: tile iso3 berikutnya di-prefetch ke register",
            "  (98.5 -> 85.4 ms per layer, T=64, n=131072)",
            "GEMM prefill: tile3, I2F, TT/R lain: tidak ada gain",
            "  (batas: load aktivasi shared memory + 4 op FP32 per baris-token)",
            "target < 50 min tidak tercapai (kurang 10.4 min)",
            "state round trip: %s; needle: %s" % (
                "bit-identik" if "PASS" in open(os.path.join(RES, "m3_final_PTQ1_0_ctx131072_statetest.txt"), errors="replace").read() else "?"
                if os.path.exists(os.path.join(RES, "m3_final_PTQ1_0_ctx131072_statetest.txt")) else "?",
                "MAGENTA-FALCON-42" if "MAGENTA-FALCON-42" in (open(os.path.join(RES, "m3_final_PTQ1_0_ctx131072_statetest.txt"), errors="replace").read()
                                                              if os.path.exists(os.path.join(RES, "m3_final_PTQ1_0_ctx131072_statetest.txt")) else "") else "?"),
            "peak VRAM %s MiB (cap 7373); decode setelah 128K: %.2f tok/s (PR #53: 4.54)" % (d.get("peak_vram_mib", "?"), d.get("decode_tok_s", 0))]
    s.lines(x0 + 16, y0 + 220, rows, size=11, gap=20)


def panel_m4(s):
    x0, y0, pw, ph = 600, 796, 576, 420
    panel(s, x0, y0, pw, ph, "E. M4: server Bonsai + MTP + PLD, gate verifikasi")
    p = os.path.join(RES, "GATES.json")
    gates = json.load(open(p)) if os.path.exists(p) else []
    for i, g in enumerate(gates[:12]):
        yy = y0 + 58 + i * 28
        ok = g.get("ok")
        s.rect(x0 + 16, yy - 14, 56, 20, C["gpu_bg"] if ok else C["bad_bg"], C["gpu"] if ok else C["bad"], rx=4, sw=1)
        s.text(x0 + 44, yy, "PASS" if ok else "FAIL", size=11, weight="700", color=C["gpu"] if ok else C["bad"], anchor="middle")
        s.text(x0 + 84, yy, g.get("name", ""), size=11, weight="700")
        s.text(x0 + 250, yy, g.get("detail", ""), size=10)


def main():
    s = Svg(W, H, "")
    s.parts.pop()
    s.text(24, 38, "Bonsai lokal 10 jam: MTP ProCreations, kernel decode/prefill sm_61, hardening server", size=21, weight="700")
    s.text(24, 62, "exp/bonsai-local-10h-opt-serving · issue #55 · GTX 1070 Ti sm_61 (cap 7373 MiB) · Ryzen 5 2400G · data: bench/results/bonsai_local_10h/",
           size=13, color=C["dim"])
    panel_m1(s)
    panel_m2(s)
    panel_depth(s)
    panel_m3(s)
    panel_m4(s)
    s.save("55_bonsai_local_opt_serving.svg")


if __name__ == "__main__":
    main()
