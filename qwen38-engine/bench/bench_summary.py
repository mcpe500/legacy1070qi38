#!/usr/bin/env python3
"""Helpers of bench/compare_llamacpp.sh.
  bench_summary.py llama  TAG OUTDIR NGL SPEC RSS_KB     -> OUTDIR/TAG.summary.json  (from TAG.response.json and TAG.vram.csv)
  bench_summary.py engine TAG OUTDIR MTP                 -> OUTDIR/TAG.summary.json  (from TAG.txt, the engine's JSON line, and TAG.vram.csv)
  bench_summary.py table  OUTDIR OUTFILE                 -> the Markdown comparison"""
import json
import os
import sys


def vram_peak(path):
    try:
        v = [int(x) for x in open(path).read().split() if x.strip().isdigit()]
        return max(v) if v else None
    except Exception:
        return None


def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def llama(tag, out, ngl, spec, rss_kb):
    p = os.path.join(out, tag)
    s = {"tag": tag, "engine": "llama-server", "ngl": int(ngl), "spec": spec, "ok": False}
    try:
        d = json.load(open(p + ".response.json"))
        t = d["timings"]
        s.update(ok=True, pp_tok_s=t["prompt_per_second"], tg_tok_s=t["predicted_per_second"], prompt_n=t["prompt_n"], predicted_n=t["predicted_n"],
                 prompt_ms=t["prompt_ms"], draft_n=t.get("draft_n", 0) or 0, draft_acc=t.get("draft_n_accepted", 0) or 0, tokens=d.get("tokens", []),
                 text=d.get("content", ""))
    except Exception as e:
        s["error"] = str(e)
    s["peak_vram_mib"] = vram_peak(p + ".vram.csv")
    s["peak_rss_mib"] = int(rss_kb) / 1024.0 if str(rss_kb).isdigit() else None
    json.dump(s, open(p + ".summary.json", "w"))
    print(tag, {k: v for k, v in s.items() if k not in ("tokens", "text")})


def engine(tag, out, mtp):
    p = os.path.join(out, tag)
    s = {"tag": tag, "engine": "qwen38-engine", "mtp": int(mtp), "ok": False}
    try:
        lines = [l for l in open(p + ".txt", encoding="utf-8").read().splitlines() if l.startswith('{"prompt_tokens"')]
        d = json.loads(lines[-1])
        s.update(ok=True, pp_tok_s=d["prompt_tokens"] / d["prefill_s"], tg_tok_s=d["decode_tok_s"], prompt_n=d["prompt_tokens"], predicted_n=d["generated_tokens"],
                 prompt_ms=d["prefill_s"] * 1000.0, ttft_ms=d.get("ttft_s", 0) * 1000.0, draft_n=d["draft_n"], draft_acc=d["draft_n_accepted"], tokens=d["ids"],
                 text=d["text"], peak_rss_mib=d["peak_rss_kb"] / 1024.0, rss_anon_mib=d["rss_anon_kb"] / 1024.0, heap_allocs=d["heap_allocs_in_decode_forward"],
                 gpu_layers=d["gpu_layers"], kv=d["kv"], load_s=d["load_s"])
    except Exception as e:
        s["error"] = str(e)
    s["peak_vram_mib"] = vram_peak(p + ".vram.csv")
    json.dump(s, open(p + ".summary.json", "w"))
    print(tag, {k: v for k, v in s.items() if k not in ("tokens", "text")})


def table(out, outfile):
    S = {}
    for f in sorted(os.listdir(out)):
        if f.endswith(".summary.json"):
            j = json.load(open(os.path.join(out, f)))
            S[j["tag"]] = j
    cap = 7373

    def row(s, label):
        if not s.get("ok"):
            return "| %s | failed: %s | | | | | | |" % (label, s.get("error", "no result"))
        acc = ("%.0f %% (%d / %d)" % (100.0 * s["draft_acc"] / s["draft_n"], s["draft_acc"], s["draft_n"])) if s.get("draft_n") else "no MTP"
        vram = s.get("peak_vram_mib")
        return "| %s | %.2f | %.2f | %s | %s | %s | %s | %s |" % (
            label, s["pp_tok_s"], s["tg_tok_s"], acc, "%d%s" % (vram, "" if vram <= cap else " (over cap)") if vram else "n/a",
            "%.0f" % s["peak_rss_mib"] if s.get("peak_rss_mib") else "n/a", s["prompt_n"], s["predicted_n"])

    main_l = S.get("llama_selected")
    main_e = S.get("engine_mtp2")
    L = []
    L.append("# Head-to-head: llama.cpp (iso3 KV + draft-mtp 2) vs qwen38-engine\n")
    L.append("Prompt: `bench/prompt512.txt` (509 tokens), 256 generated tokens, temperature 0, seed 42 (top_k 1 for llama-server). "
             "GTX 1070 Ti (8192 MiB, VRAM cap 7373 MiB), Ryzen 5 2400G pinned to cores 0-3, 16 GB DDR4, same GGUF (Qwen3.8-27B-UD-Q4_K_M). "
             "The two programs ran one after the other, never together. Peak VRAM = highest `nvidia-smi` memory.used sample (200 ms) over the whole run, "
             "Peak RSS = VmHWM of the process (includes the file pages of the mapped model).\n")
    L.append("## Headline\n")
    L.append("| Program (configuration) | pp tok/s | tg tok/s | MTP accept | Peak VRAM MiB | Peak DDR4 RSS MiB | prompt tokens | generated |")
    L.append("|---|---:|---:|---|---:|---:|---:|---:|")
    if main_l:
        L.append(row(main_l, "llama-server `-ngl %d` iso3 KV + draft-mtp 2" % main_l["ngl"]))
    if main_e:
        L.append(row(main_e, "qwen38-engine `--gpu-layers 32 --kv iso3 --dp4a all --mtp-steps 2`"))
    L.append("")
    if main_l and main_e and main_l.get("ok") and main_e.get("ok"):
        L.append("Ratios (engine / llama-server): pp %.2fx, tg %.2fx.\n" % (main_e["pp_tok_s"] / main_l["pp_tok_s"], main_e["tg_tok_s"] / main_l["tg_tok_s"]))
        d = first_diff(main_e["tokens"], main_l["tokens"])
        L.append("Generated tokens, engine vs llama-server: %s.\n" % ("identical (%d tokens)" % len(main_e["tokens"]) if d is None else
                 "first difference at token %d of %d / %d (both use an iso3 KV cache but with different attention numerics: llama.cpp quantizes the KV of all 16 attention layers and converts it to f16 for flash attention, the engine quantizes the 8 GPU layers and keeps FP32 on the CPU layers; greedy outputs of a 3-bit KV cache are not expected to match for 256 tokens)" % (d, len(main_e["tokens"]), len(main_l["tokens"]))))
    r2l, r2e = S.get("llama_selected_r2"), S.get("engine_mtp2_r2")
    if r2l and r2e:
        L.append("## Repeat round (same commands again; the page cache is in a different state, which moves the prompt-processing numbers)\n")
        L.append("| Program (configuration) | pp tok/s | tg tok/s | MTP accept | Peak VRAM MiB | Peak DDR4 RSS MiB | prompt tokens | generated |")
        L.append("|---|---:|---:|---|---:|---:|---:|---:|")
        L.append(row(r2l, "llama-server `-ngl %d` iso3 KV + draft-mtp 2" % r2l["ngl"]))
        L.append(row(r2e, "qwen38-engine `--gpu-layers 32 --kv iso3 --dp4a all --mtp-steps 2`"))
        L.append("")
    L.append("## All runs\n")
    L.append("| Run | pp tok/s | tg tok/s | MTP accept | Peak VRAM MiB | Peak DDR4 RSS MiB | prompt tokens | generated |")
    L.append("|---|---:|---:|---|---:|---:|---:|---:|")
    for tag in sorted(S):
        s = S[tag]
        if tag.startswith("llama_selected"):
            continue
        label = ("llama-server -ngl %d, %s" % (s["ngl"], "draft-mtp 2" if s["spec"] != "none" else "no MTP")) if s["engine"] == "llama-server" else \
                ("qwen38-engine, --mtp-steps %d" % s["mtp"])
        L.append(row(s, "`%s`: %s" % (tag, label)))
    L.append("")
    probes = [s for t, s in S.items() if t.startswith("llama_ngl") and not t.endswith("_r2")]
    if probes:
        L.append("## llama-server layer count (largest -ngl whose peak VRAM stays at or under %d MiB)\n" % cap)
        L.append("| -ngl | peak VRAM MiB | under cap | pp tok/s | tg tok/s |")
        L.append("|---:|---:|---|---:|---:|")
        for s in sorted(probes, key=lambda x: x["ngl"]):
            v = s.get("peak_vram_mib")
            L.append("| %d | %s | %s | %s | %s |" % (s["ngl"], v if v else "n/a", "yes" if v and v <= cap and s.get("ok") else "no", "%.2f" % s["pp_tok_s"] if s.get("ok") else "failed", "%.2f" % s["tg_tok_s"] if s.get("ok") else "failed"))
        L.append("")
    L.append("## Notes\n")
    L.append("- **Decode (tg):** the engine is about 3 % faster in both rounds (2.58 / 2.59 against 2.50 / 2.51 tok/s). Both programs are bound by reading the CPU-resident weights from DDR4 each step; the engine keeps 32 layers on the GPU, llama.cpp 25, and the engine's MTP accepted 96 % of its drafts against 91 %. The engine without MTP decodes at 2.44, llama-server without MTP at 2.25 tok/s.")
    L.append("- **Prompt processing (pp):** the engine is slower, 0.47x. The GPU sits on a PCIe Gen3 x1 link (0.84 GB/s measured), and the engine streams the 32 CPU-resident layers (7 GiB) through VRAM for every prefill, which costs about 9 s whatever the prompt length; the engine's pp also includes the MTP prefill (about 5 s for 509 tokens: 16.97 tok/s with `--mtp-steps 0`, 13.4 with 2). llama.cpp's own pp moves with the layer count and the page cache (28.7 tok/s at -ngl 25, 17.0 to 17.4 at -ngl 24 and 26 in the same conditions).")
    L.append("- **Memory:** the engine's peak RSS is 9.2 GiB against 14.6 GiB for llama-server. llama-server's figure includes the whole mapped 16 GB GGUF; the engine drops the file pages of the layers it keeps in VRAM and streams the rest.")
    L.append("- **VRAM:** both stay under the 7373 MiB cap (7239 and 7245 MiB). llama-server at -ngl 26 needs 7473 MiB and is excluded.")
    L.append("- **MTP accept %** is drafts accepted / drafts proposed. The two programs draft with different rules (llama.cpp stops drafting under a confidence threshold, the engine always drafts the configured number), so the percentages are not strictly comparable.")
    L.append("- Commands: `bench/compare_llamacpp.sh` (this file), engine run = `taskset -c 0-3 build/qwen38-engine -m GGUF -p \"$(cat bench/prompt512.txt)\" -n 256 --temp 0 --seed 42 --gpu-layers 32 --kv iso3 --dp4a all --mtp-steps 2`, llama-server run = `taskset -c 0-3 llama-server -m GGUF -ngl N -c 2048 -np 1 -t 4 -ctk iso3 -ctv iso3 -ctkd iso3 -ctvd iso3 --spec-type draft-mtp --spec-draft-n-max 2`, then POST /completion with `n_predict 256, temperature 0, top_k 1, seed 42, cache_prompt false`. Raw data: `bench/results/stage5/`.\n")
    open(outfile, "w").write("\n".join(L) + "\n")
    print("wrote", outfile)


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "llama":
        llama(*sys.argv[2:7])
    elif c == "engine":
        engine(*sys.argv[2:5])
    elif c == "table":
        table(*sys.argv[2:4])
