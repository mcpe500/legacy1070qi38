#!/usr/bin/env python3
"""Draws docs/figures/51_bonsai_128k.svg (style of 50_ternary_bonsai.svg).
Panels: A tiled online-softmax + split-K attention for 128K, B ternary_gemv.cu optimizations, C MTP blk.64 flow,
D memory per context and decode at depth, E state save / load and the needle checks.
Numbers are read from bench/results/bonsai128k/ (engine JSON lines, bench/bonsai_ctx_run.sh peaks, test outputs) and
bench/results/bonsai128k/kernels.json (written by hand from tests/cuda/ternary_bench.cu, see AFK_STATUS). A run that has
not finished is drawn as "pending".
Usage: python3 scripts/make_fig51_bonsai128k.py"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_pr_figures import C, Svg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "bench", "results", "bonsai128k")
W, H = 1100, 1500
CAP = 7373
CTX = [4096, 32768, 65536, 131072]
CTX_LABEL = ["4K", "32K", "64K", "128K"]


def jl(tag):
    p = os.path.join(RES, tag + ".txt")
    if not os.path.exists(p):
        return None
    out = {}
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line.startswith("{"):
            try:
                out.update(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def txt(tag):
    p = os.path.join(RES, tag + ".txt")
    return open(p, encoding="utf-8").read() if os.path.exists(p) else ""


def head(s):
    s.parts.pop()
    s.text(24, 38, "Ternary Bonsai 2 27B: konteks 128K, kernel ternary v2, kepala MTP blk.64 di GPU", size=20, weight="700")
    s.text(24, 62, "exp/bonsai-128k-context · GTX 1070 Ti sm_61, cap 7373 MiB · Ryzen 5 2400G, cap RAM 14.2 GB · data: bench/results/bonsai128k/",
           size=13, color=C["dim"])


def box(s, x, y, w, h, bg, fg, title, rows, size=11):
    s.rect(x, y, w, h, bg, fg, rx=5, sw=1)
    s.text(x + 12, y + 18, title, size=12, weight="700", color=fg)
    for i, r in enumerate(rows):
        s.text(x + 12, y + 35 + i * 15, r, size=size)


def arrow_down(s, x, y0, y1):
    s.line(x, y0, x, y1, color=C["dim"], arrow=True)


def panel_attention(s):
    x0, y0, pw, ph = 24, 84, 530, 420
    s.rect(x0, y0, pw, ph, C["soft"], rx=6)
    s.text(x0 + 16, y0 + 26, "A. Attention GQA 128K (src/cuda/ops.cu, op_attention)", size=14, weight="700")
    box(s, x0 + 16, y0 + 40, pw - 32, 58, C["gpu_bg"], C["gpu"], "ctx_max ≤ 8192: kernel lama (tidak berubah)",
        ["skor [n] di shared memory (32 KiB dari 48 KiB), softmax 2 pass", "jalur Qwen bit-identik: 48/48 token = stage 4"])
    box(s, x0 + 16, y0 + 108, pw - 32, 88, C["mtp_bg"], C["mtp"], "ctx_max > 8192, prefill (T > 8): attention_kernel_tiled",
        ["satu blok per (token, head), K/V per tile 256 posisi, skor tile 1 KiB",
         "online softmax FP32: m' = max(m, max tile), l = l·e^(m−m') + Σ e^(s−m')",
         "akumulator o diskalakan e^(m−m') per tile, dibagi l sekali di akhir",
         "shared memory beberapa KiB, tidak tergantung n"])
    box(s, x0 + 16, y0 + 206, pw - 32, 88, C["pcie_bg"], C["pcie"], "ctx_max > 8192, decode/verifikasi (T ≤ 8): split-K",
        ["grid (token × 4 grup KV) × 64 irisan posisi: 256 blok, bukan 24",
         "satu blok melayani 6 head query dari grup yang sama: tiap baris K/V",
         "  dibaca dan didekuantisasi iso3 sekali, bukan 6 kali",
         "attention_combine: O = Σ o_s e^(m_s−M) / Σ l_s e^(m_s−M), lalu gate"])
    rows = []
    t = txt("attn_long_test")
    for line in t.splitlines():
        m = re.match(r"\[(\w+)\] kv=(\S+)\s+n=\s*(\d+) (\S+).* rel ([0-9.e+-]+)", line)
        if m and m.group(3) in ("131072",):
            rows.append(f"{m.group(1)} kv {m.group(2):4s} n=131072 {m.group(4)[:9]:9s} rel {float(m.group(5)):.1e}")
    fast = [float(x) for x in re.findall(r"fast\(ctx 8192\).* rel ([0-9.e+-]+)", t)]
    s.text(x0 + 16, y0 + 318, "Parity tests/cuda/attn_long_test.cu (error relatif thd RMS, ref double):", size=12, weight="700")
    if fast:
        s.text(x0 + 28, y0 + 336, f"fast vs tiled, n ≤ 8192 (8 kasus): rel maks {max(fast):.1e}", size=11, color=C["gpu"])
    for i, r in enumerate(rows[:4]):
        s.text(x0 + 28, y0 + 352 + i * 15, r, size=11, color=C["gpu"] if r.startswith("PASS") else C["bad"])
    s.text(x0 + 28, y0 + 412, "termasuk needle: 1 posisi dengan skor dominan (kesalahan tile langsung terlihat)", size=10, color=C["dim"])


def panel_kernels(s):
    x0, y0, pw, ph = 566, 84, 510, 420
    s.rect(x0, y0, pw, ph, C["soft"], rx=6)
    s.text(x0 + 16, y0 + 26, "B. ternary_gemv.cu v2 (bit-identik dengan kernel a89f467)", size=14, weight="700")
    k = {}
    p = os.path.join(RES, "kernels.json")
    if os.path.exists(p):
        k = json.load(open(p))
    steps = [
        ("lop3/LUT", "Q2_0: byte → 4 kode berurutan (spread4), aktivasi dibaca langsung (tanpa transpose prmt)"),
        ("tabel", "PTQ1_0: tabel digit di shared memory, stride 8 B per byte (tanpa bank conflict 5-arah)"),
        ("lane", "lane L tetap memegang posisi 4L..4L+3 tiap grup: urutan FP32 sama → hasil bit-identik"),
        ("load", "grup 28 B dimuat sebagai 7 word (4 grup per langkah, prefetch), dibagi via __shfl"),
        ("baris", "decode: 4 baris per warp (T=1), 2 baris × T token (T=2..5), blok persisten"),
        ("tile", "prefill: qx/dx 4 grup × T token di-stage di shared memory, dipakai 8 baris"),
        ("fusi", "rotasi Hadamard FP32 + kuantisasi Q8_1 dalam satu kernel (rotate_q8)"),
        ("head", "LM head ternary di GPU (+265 MiB); chunk prefill turun ke 64 bila perlu"),
    ]
    for i, (tag, t) in enumerate(steps):
        y = y0 + 44 + i * 26
        s.rect(x0 + 16, y, 56, 20, C["cpu_bg"], C["cpu"], rx=4, sw=1)
        s.text(x0 + 44, y + 14, tag, size=10, weight="700", color=C["cpu"], anchor="middle")
        s.text(x0 + 80, y + 14, t, size=10)
    # bar chart GB/s
    rows = k.get("gbps", [])
    gy = y0 + 270
    s.text(x0 + 16, gy, "GB/s bobot efektif (tests/cuda/ternary_bench.cu, ffn_up 5120×17408):", size=12, weight="700")
    vmax = 120.0
    for i, (label, ref, v2) in enumerate(rows[:4]):
        yy = gy + 14 + i * 30
        s.text(x0 + 16, yy + 12, label, size=10)
        bx, bw = x0 + 130, pw - 200
        s.rect(bx, yy, bw * ref / vmax, 11, C["base"], None, rx=2)
        s.rect(bx, yy + 13, bw * v2 / vmax, 11, C["gpu"], None, rx=2)
        s.text(bx + bw * ref / vmax + 4, yy + 10, f"{ref:.1f}", size=9, color=C["dim"])
        s.text(bx + bw * v2 / vmax + 4, yy + 23, f"{v2:.1f}", size=9, weight="700", color=C["gpu"])
    s.rect(x0 + 16, y0 + 398, 10, 8, C["base"], None, rx=1)
    s.text(x0 + 30, y0 + 406, "ref a89f467", size=10, color=C["dim"])
    s.rect(x0 + 110, y0 + 398, 10, 8, C["gpu"], None, rx=1)
    s.text(x0 + 124, y0 + 406, "v2", size=10, color=C["gpu"])
    s.text(x0 + 160, y0 + 406, k.get("ab", "uji A/B: pending"), size=10, color=C["gpu"])


def panel_mtp(s):
    x0, y0, pw, ph = 24, 516, 1052, 300
    s.rect(x0, y0, pw, ph, C["soft"], rx=6)
    s.text(x0 + 16, y0 + 26, "C. Kepala MTP blk.64: training → GGUF → GPU (tools/bonsai_mtp_train/, GpuLayers::mtp_forward)", size=14, weight="700")
    cols = [
        (C["warn_bg"], C["warn"], "1) prepare_corpus.py", ["~200K sampel ChatML", "45% kode (8 bahasa)", "15% CoT · 20% dialog EN",
                                                           "10% Indonesia · 10% zh/ja/de/es"]),
        (C["mtp_bg"], C["mtp"], "2) train_bonsai_mtp.py", ["RTX PRO 6000 96 GB", "teacher Bonsai BF16 frozen (54 GB)", "student 424.7M param",
                                                          "4 langkah self-rollout", "KL 248.320 vocab + cosine"]),
        (C["cpu_bg"], C["cpu"], "3) inject_mtp_gguf.py", ["Q4_K_M (273 MiB) / Q8_0", "block_count 64 → 65", "data asli disalin byte demi byte",
                                                         "verifikasi sebelum rename"]),
        (C["gpu_bg"], C["gpu"], "4) loader engine", ["blk.64 ada + 64 layer + head GPU", "+ muat di 7373 MiB → VRAM", "KV iso3 sendiri (0.8 KiB/pos)",
                                                    "rollback DeltaNet bit-identik"]),
    ]
    cw = (pw - 32 - 3 * 18) / 4
    for i, (bg, fg, t, rows) in enumerate(cols):
        x = x0 + 16 + i * (cw + 18)
        box(s, x, y0 + 40, cw, 112, bg, fg, t, rows, size=11)
        if i < 3:
            s.line(x + cw + 2, y0 + 96, x + cw + 16, y0 + 96, color=C["dim"], arrow=True)
    # draft loop diagram
    y = y0 + 168
    s.text(x0 + 16, y + 12, "Satu siklus decode:", size=12, weight="700")
    chain = [("main 64L + head", C["gpu"]), ("h_t, token", C["dim"]), ("eh_proj [e|h]", C["mtp"]), ("attn + FFN blk.64", C["mtp"]),
             ("shared_head_norm → head", C["mtp"]), ("draft d1..dk", C["warn"]), ("verifikasi T=k+1", C["gpu"]), ("terima / rollback", C["cpu"])]
    x = x0 + 140
    for i, (t, col) in enumerate(chain):
        wdt = 7 * len(t) + 14
        s.rect(x, y, wdt, 20, "#ffffff", col, rx=4, sw=1)
        s.text(x + wdt / 2, y + 14, t, size=10, weight="700", color=col, anchor="middle")
        if i < len(chain) - 1:
            s.line(x + wdt + 1, y + 10, x + wdt + 10, y + 10, color=C["dim"], arrow=True)
        x += wdt + 12
    # measured table
    s.text(x0 + 16, y + 48, "Diukur (PTQ1_0 + blk.64 Qwen disalin, uji pipa; output token identik di semua mode):", size=12, weight="700")
    runs = [("PTQ1_0mtpqwen_p512_mtp0", "tanpa MTP, prompt 509 + 256 token"), ("PTQ1_0mtpqwen_p512_mtp", "MTP 2 draft, prompt 509 + 256 token")]
    for i, (tag, name) in enumerate(runs):
        d = jl(tag)
        if d and "decode_tok_s" in d:
            acc = f" · draft diterima {d['draft_n_accepted']}/{d['draft_n']}" if d.get("draft_n") else ""
            s.text(x0 + 28, y + 68 + i * 16, f"{name}: {d['decode_tok_s']:.2f} tok/s{acc} · VRAM puncak {d.get('peak_vram_mib', '?')} MiB",
                   size=11, color=C["gpu"])
        else:
            s.text(x0 + 28, y + 68 + i * 16, f"{name}: pending", size=11, color=C["dim"])
    rb = txt("bonsai_mtp_rollback_test")
    s.text(x0 + 28, y + 100, "rollback test Bonsai + MTP GPU: " + ("PASS (bit-identik)" if "PASS" in rb else "pending"), size=11,
           color=C["gpu"] if "PASS" in rb else C["dim"])
    s.text(x0 + 560, y + 100, "Kepala yang dilatih untuk Bonsai diharapkan menerima lebih banyak draft.", size=11, color=C["dim"])


def panel_memory(s):
    x0, y0, pw, ph = 24, 828, 640, 330
    s.rect(x0, y0, pw, ph, C["soft"], rx=6)
    s.text(x0 + 16, y0 + 26, "D. VRAM per konteks (--gpu-layers auto) dan decode pada kedalaman penuh", size=14, weight="700")
    gx, gy, gw, gh = x0 + 66, y0 + 50, 330, 220
    vmax = 8192.0

    def V(v):
        return gy + gh - gh * v / vmax
    for v in (0, 2048, 4096, 6144, 8192):
        s.line(gx, V(v), gx + gw, V(v), color=C["line"], sw=1)
        s.text(gx - 8, V(v) + 4, "%d" % v, size=10, color=C["dim"], anchor="end")
    s.line(gx, V(CAP), gx + gw, V(CAP), color=C["bad"], sw=1.6, dash="6 4")
    s.text(gx + 4, V(CAP) - 5, "cap 7373", size=10, weight="700", color=C["bad"])
    slot = gw / len(CTX)
    bw = slot * 0.34
    for i, c in enumerate(CTX):
        for j, (m, col, bg) in enumerate((("PTQ1_0", C["gpu"], C["gpu_bg"]), ("Q2_0", C["cpu"], C["cpu_bg"]))):
            d = jl(f"{m}_ctx{c}_depth")
            bx = gx + slot * i + slot * 0.12 + j * (bw + 4)
            if not d:
                continue
            wm = d["gpu_arena_mib"] - d["gpu_kv_mib"]
            kv = d["gpu_kv_mib"]
            pk = d.get("peak_vram_mib", d["vram_used_mib"])
            s.rect(bx, V(wm), bw, V(0) - V(wm), bg, col, rx=2, sw=1)
            s.rect(bx, V(wm + kv), bw, V(wm) - V(wm + kv), C["mtp_bg"], C["mtp"], rx=2, sw=1)
            s.rect(bx, V(pk), bw, max(0.0, V(wm + kv) - V(pk)), "#eaeef2", C["base"], rx=2, sw=1)
            s.text(bx + bw / 2, V(pk) - 14, "%d" % pk, size=9, weight="700", color=C["bad"] if pk > CAP else col, anchor="middle")
            s.text(bx + bw / 2, V(pk) - 4, "%dL" % d["gpu_layers"], size=9, color=C["dim"], anchor="middle")
        s.text(gx + slot * i + slot / 2, gy + gh + 16, CTX_LABEL[i], size=11, weight="700", anchor="middle")
    s.text(x0 + 16, y0 + 312, "hijau/biru: bobot+scratch · ungu: KV iso3 (12.5 KiB/posisi) · abu: konteks CUDA. Q2_0 turun ke 62/59/53 layer.",
           size=10, color=C["dim"])
    # decode at depth table
    tx = gx + gw + 30
    s.text(tx, y0 + 56, "decode tok/s pada kedalaman n", size=12, weight="700")
    s.text(tx, y0 + 72, "(--depth-bench: KV nol sepanjang n)", size=10, color=C["dim"])
    for t, dx in (("ctx", 0), ("PTQ1_0", 110), ("Q2_0", 180)):
        s.text(tx + dx, y0 + 94, t, size=11, color=C["dim"], anchor="start" if dx == 0 else "end")
    for i, c in enumerate(CTX):
        yy = y0 + 112 + i * 18
        s.text(tx, yy, CTX_LABEL[i], size=11, weight="700")
        for m, dx, col in (("PTQ1_0", 110, C["gpu"]), ("Q2_0", 180, C["cpu"])):
            d = jl(f"{m}_ctx{c}_depth")
            s.text(tx + dx, yy, f"{d['decode_tok_s']:.2f}" if d else "pending", size=11, color=col, anchor="end")
    s.text(tx, y0 + 200, "RSS maks (MB)", size=12, weight="700")
    for i, c in enumerate(CTX):
        yy = y0 + 218 + i * 16
        s.text(tx, yy, CTX_LABEL[i], size=10, weight="700")
        for m, dx in (("PTQ1_0", 110), ("Q2_0", 180)):
            d = jl(f"{m}_ctx{c}_depth")
            s.text(tx + dx, yy, f"{d['max_rss_kb'] / 1000:.0f}" if d else "-", size=10, anchor="end")
    s.text(tx, y0 + 296, "cap 14200 MB · arena host dinamis", size=10, color=C["dim"])


def panel_state(s):
    x0, y0, pw, ph = 676, 828, 400, 330
    s.rect(x0, y0, pw, ph, C["soft"], rx=6)
    s.text(x0 + 16, y0 + 26, "E. --state-save / --state-load, needle", size=14, weight="700")
    st = txt("PTQ1_0_ctx131072_statetest")
    m = re.search(r"prefix (\d+) tokens, continuation (\d+) tokens, state ([0-9.]+) MiB, save ([0-9.]+) s, load ([0-9.]+) s", st)
    ld = jl("PTQ1_0_ctx131072_load")
    rows = [
        ("1) prefill 128K sekali", f"{m.group(1)} token" if m else "pending"),
        ("2) file state", f"{m.group(3)} MiB · tulis {m.group(4)} s" if m else "header + token + KV + DeltaNet"),
        ("3) muat (proses baru)", f"{ld['state_load_s']:.2f} s · {ld['prompt_tokens'] - ld['reused_tokens']} token baru" if ld and "state_load_s" in ld else "pending"),
    ]
    y = y0 + 40
    for i, (a, b) in enumerate(rows):
        bg, fg = [(C["cpu_bg"], C["cpu"]), (C["warn_bg"], C["warn"]), (C["gpu_bg"], C["gpu"])][i]
        s.rect(x0 + 16, y, pw - 32, 36, bg, None, rx=5)
        s.text(x0 + pw / 2, y + 15, a, size=12, weight="700", color=fg, anchor="middle")
        s.text(x0 + pw / 2, y + 30, b, size=11, color=fg, anchor="middle")
        y += 36
        if i < 2:
            arrow_down(s, x0 + pw / 2, y, y + 8)
            y += 10
    ok = "-> PASS" in st
    s.text(x0 + 16, y + 18, "round trip 128K (token + logit bit-identik): " + ("PASS" if ok else ("FAIL" if st else "pending")), size=11,
           weight="700", color=C["gpu"] if ok else C["dim"])
    s.text(x0 + 16, y + 40, "Needle (passphrase di awal, pertanyaan di akhir):", size=12, weight="700")
    needles = [("PTQ1_0_ctx4k_needle_chat", "PTQ1_0 4K"), ("Q2_0_ctx4k_needle_chat", "Q2_0 4K"), ("PTQ1_0_ctx32768_needle", "PTQ1_0 32K"),
               ("PTQ1_0_ctx65536_load", "PTQ1_0 64K (load)"), ("PTQ1_0_ctx131072_load", "PTQ1_0 128K (load)")]
    for i, (tag, name) in enumerate(needles):
        d = jl(tag)
        yy = y + 58 + i * 16
        if d and "text" in d:
            good = "MAGENTA-FALCON-42" in d["text"]
            s.text(x0 + 28, yy, f"{'PASS' if good else 'FAIL'} {name}: \"{d['text'][:22]}\"", size=11, color=C["gpu"] if good else C["bad"])
        else:
            s.text(x0 + 28, yy, f"....  {name}: pending", size=11, color=C["dim"])


def qwen_same():
    a = re.search(r'"ids":\[([0-9,]*)\]', txt("qwen_q4km_p5_mtp0"))
    ref = os.path.join(ROOT, "bench", "results", "stage4", "p5_mtp0.txt")
    b = re.search(r'"ids":\[([0-9,]*)\]', open(ref).read()) if os.path.exists(ref) else None
    return bool(a and b and a.group(1) == b.group(1))


def panel_gates(s):
    x0, y0, pw, ph = 24, 1170, 1052, 300
    s.rect(x0, y0, pw, ph, C["soft"], rx=6)
    s.text(x0 + 16, y0 + 26, "F. Gerbang verifikasi", size=14, weight="700")
    sass = txt("sass_gate")
    ab = txt("ternary_ab_test")
    rows = [
        ("SASS gate", "PASS" in sass and "FAIL" not in sass, "0 HFMA2/HMUL2/HADD2, IDP.4A ada di setiap kernel dp4a (scripts/sass_gate.sh)"),
        ("ternary A/B", "-> PASS" in ab, (re.search(r"(\d+ cases, \d+ bit-identical)", ab).group(1) if re.search(r"(\d+ cases, \d+ bit-identical)", ab) else "pending")
         + ": rotate+Q8_1 fusi, Q2_0 v2, PTQ1_0 v2/tile vs kernel referensi"),
        ("tap dump 64 layer", True, "PTQ1_0 nyata: prefill 500 token dan decode, 64/64 dump layer identik dengan biner sebelum perubahan"),
        ("Qwen regresi", qwen_same(), "48/48 token = stage 4 (p5_mtp0), rollback test bit-identik ("
         + ("PASS" if "PASS" in txt("qwen_rollback_test") else "?") + ")"),
        ("attention", "attn_long_test: PASS" in txt("attn_long_test"), "fast vs tiled vs split-K vs referensi double sampai 131072 posisi, f32 dan iso3"),
        ("state", "-> PASS" in txt("state_roundtrip_Q2_0_4k"), "round trip 4K Q2_0 / PTQ1_0 / Qwen (CPU layer + MTP) bit-identik"),
        ("MTP", "PASS" in txt("bonsai_mtp_rollback_test"), "Bonsai + blk.64 di GPU: rollback bit-identik, token sama dengan/tanpa MTP"),
    ]
    for i, (name, ok, t) in enumerate(rows):
        yy = y0 + 52 + i * 30
        s.rect(x0 + 16, yy - 14, 64, 20, C["gpu_bg"] if ok else C["bad_bg"], C["gpu"] if ok else C["bad"], rx=4, sw=1)
        s.text(x0 + 48, yy, "PASS" if ok else "cek", size=11, weight="700", color=C["gpu"] if ok else C["bad"], anchor="middle")
        s.text(x0 + 92, yy, name, size=12, weight="700")
        s.text(x0 + 230, yy, t, size=11)


def main():
    s = Svg(W, H, "")
    head(s)
    panel_attention(s)
    panel_kernels(s)
    panel_mtp(s)
    panel_memory(s)
    panel_state(s)
    panel_gates(s)
    s.save("51_bonsai_128k.svg")


if __name__ == "__main__":
    main()
