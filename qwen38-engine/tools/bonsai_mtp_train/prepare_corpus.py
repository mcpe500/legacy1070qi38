#!/usr/bin/env python3
"""Streams and mixes the MTP distillation corpus for Ternary Bonsai 2 27B (about 200,000 samples) from Hugging Face.

Mix (share of the samples, override with --mix FILE.json):
  45 %  multi-language code: nvidia/OpenCodeInstruct (instruction + solution) and bigcode/starcoderdata
        (C++, CUDA, Rust, Zig, Python, TypeScript, Go, PHP source files)
  15 %  reasoning / chain of thought: open-thoughts/OpenThoughts-114k
  20 %  English dialogue: HuggingFaceH4/ultrachat_200k (train_sft)
  10 %  Indonesian: uonlp/CulturaX (id) and wikimedia/wikipedia (20231101.id)
  10 %  multilingual zh / ja / de / es: uonlp/CulturaX and CohereForAI/aya_collection (aya_dataset)

Every sample is written as ChatML text (the template Bonsai and qwen38-engine use), so the teacher sees the same token
layout at distillation time as at inference time. Plain documents (CulturaX, Wikipedia, starcoderdata) become a user turn
asking to continue / explain, with the document as the assistant answer, so the MTP head learns on assistant-side tokens.

Output: OUT_DIR/shard-00000.jsonl ... ({"text", "source", "lang"} per line), OUT_DIR/manifest.json (counts per source,
seed, filters). Deterministic for a given --seed; exact-duplicate texts are dropped (sha1); too short / too long texts
are skipped (--min-chars / --max-chars). Resumable: sources already complete in the manifest are skipped.

Needs: pip install datasets huggingface_hub. bigcode/starcoderdata and uonlp/CulturaX are gated: accept their terms on the
Hub and run `huggingface-cli login` (or set HF_TOKEN) first.

Usage:
  python3 prepare_corpus.py --out /data/bonsai_mtp_corpus --total 200000
  python3 prepare_corpus.py --selftest        # formatters and quota arithmetic on synthetic rows, no network
"""
import argparse
import hashlib
import json
import os
import random
import sys

# (name, share, loader kwargs, formatter, lang). Shares inside a group are fractions of the total.
DEFAULT_MIX = [
    # 45 % code
    {"name": "opencodeinstruct", "share": 0.15, "path": "nvidia/OpenCodeInstruct", "split": "train", "fmt": "instr", "lang": "code"},
    {"name": "starcoder-cpp", "share": 0.05, "path": "bigcode/starcoderdata", "data_dir": "cpp", "split": "train", "fmt": "code:C++", "lang": "code"},
    {"name": "starcoder-cuda", "share": 0.05, "path": "bigcode/starcoderdata", "data_dir": "cuda", "split": "train", "fmt": "code:CUDA", "lang": "code"},
    {"name": "starcoder-rust", "share": 0.04, "path": "bigcode/starcoderdata", "data_dir": "rust", "split": "train", "fmt": "code:Rust", "lang": "code"},
    {"name": "starcoder-zig", "share": 0.02, "path": "bigcode/starcoderdata", "data_dir": "zig", "split": "train", "fmt": "code:Zig", "lang": "code"},
    {"name": "starcoder-python", "share": 0.04, "path": "bigcode/starcoderdata", "data_dir": "python", "split": "train", "fmt": "code:Python", "lang": "code"},
    {"name": "starcoder-typescript", "share": 0.04, "path": "bigcode/starcoderdata", "data_dir": "typescript", "split": "train", "fmt": "code:TypeScript", "lang": "code"},
    {"name": "starcoder-go", "share": 0.03, "path": "bigcode/starcoderdata", "data_dir": "go", "split": "train", "fmt": "code:Go", "lang": "code"},
    {"name": "starcoder-php", "share": 0.03, "path": "bigcode/starcoderdata", "data_dir": "php", "split": "train", "fmt": "code:PHP", "lang": "code"},
    # 15 % reasoning
    {"name": "openthoughts", "share": 0.15, "path": "open-thoughts/OpenThoughts-114k", "split": "train", "fmt": "sharegpt", "lang": "en"},
    # 20 % English dialogue
    {"name": "ultrachat", "share": 0.20, "path": "HuggingFaceH4/ultrachat_200k", "split": "train_sft", "fmt": "messages", "lang": "en"},
    # 10 % Indonesian
    {"name": "culturax-id", "share": 0.06, "path": "uonlp/CulturaX", "name_cfg": "id", "split": "train", "fmt": "doc:id", "lang": "id"},
    {"name": "wikipedia-id", "share": 0.04, "path": "wikimedia/wikipedia", "name_cfg": "20231101.id", "split": "train", "fmt": "doc:id", "lang": "id"},
    # 10 % multilingual
    {"name": "culturax-zh", "share": 0.015, "path": "uonlp/CulturaX", "name_cfg": "zh", "split": "train", "fmt": "doc:zh", "lang": "zh"},
    {"name": "culturax-ja", "share": 0.015, "path": "uonlp/CulturaX", "name_cfg": "ja", "split": "train", "fmt": "doc:ja", "lang": "ja"},
    {"name": "culturax-de", "share": 0.015, "path": "uonlp/CulturaX", "name_cfg": "de", "split": "train", "fmt": "doc:de", "lang": "de"},
    {"name": "culturax-es", "share": 0.015, "path": "uonlp/CulturaX", "name_cfg": "es", "split": "train", "fmt": "doc:es", "lang": "es"},
    {"name": "aya", "share": 0.04, "path": "CohereForAI/aya_collection", "name_cfg": "aya_dataset", "split": "train", "fmt": "aya",
     "lang": "multi", "languages": ["Chinese", "Simplified Chinese", "Japanese", "German", "Spanish"]},
]

DOC_PROMPTS = {
    "id": "Tuliskan sebuah teks dalam bahasa Indonesia.",
    "zh": "请用中文写一段文字。",
    "ja": "日本語で文章を書いてください。",
    "de": "Schreibe einen Text auf Deutsch.",
    "es": "Escribe un texto en español.",
}


def chatml(turns):
    """turns: [(role, content)] -> ChatML string, thinking closed for assistant turns without <think>."""
    out = []
    for role, content in turns:
        content = content.strip()
        if role == "assistant" and not content.startswith("<think>"):
            content = "<think>\n\n</think>\n\n" + content
        out.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(out)


def first_str(row, keys):
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def fmt_row(spec, row):
    """Returns ChatML text or None when the row has no usable content."""
    f = spec["fmt"]
    if f == "instr":   # OpenCodeInstruct: input (task) + output (solution)
        q, a = first_str(row, ["input", "instruction", "prompt", "question"]), first_str(row, ["output", "response", "solution", "answer"])
        return chatml([("user", q), ("assistant", a)]) if q and a else None
    if f.startswith("code:"):   # raw source file
        code = first_str(row, ["content", "code", "text"])
        if not code:
            return None
        lang = f.split(":", 1)[1]
        path = row.get("max_stars_repo_path") or row.get("path") or ""
        ask = f"Write the {lang} source file{(' ' + path) if path else ''}."
        return chatml([("user", ask), ("assistant", f"```\n{code}\n```")])
    if f == "sharegpt":   # OpenThoughts: system + conversations [{from, value}]
        conv = row.get("conversations") or []
        turns = []
        if isinstance(row.get("system"), str) and row["system"].strip():
            turns.append(("system", row["system"]))
        for m in conv:
            role = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant"}.get(m.get("from") or m.get("role"))
            val = m.get("value") or m.get("content")
            if role and isinstance(val, str):
                turns.append((role, val))
        return chatml(turns) if any(r == "assistant" for r, _ in turns) else None
    if f == "messages":   # ultrachat: messages [{role, content}]
        turns = [(m["role"], m["content"]) for m in row.get("messages") or [] if m.get("role") in ("user", "assistant", "system")]
        return chatml(turns) if any(r == "assistant" for r, _ in turns) else None
    if f.startswith("doc:"):   # plain document
        text = first_str(row, ["text", "content"])
        return chatml([("user", DOC_PROMPTS[f.split(":", 1)[1]]), ("assistant", text)]) if text else None
    if f == "aya":
        if spec.get("languages") and row.get("language") not in spec["languages"]:
            return None
        q, a = first_str(row, ["inputs", "input"]), first_str(row, ["targets", "output"])
        return chatml([("user", q), ("assistant", a)]) if q and a else None
    raise ValueError(f"unknown formatter {f}")


def quotas(mix, total):
    s = sum(m["share"] for m in mix)
    q = {m["name"]: int(round(total * m["share"] / s)) for m in mix}
    diff = total - sum(q.values())
    q[mix[0]["name"]] += diff   # rounding remainder to the first source
    return q


def stream(spec, seed, buffer):
    from datasets import load_dataset   # noqa: E402
    kw = dict(split=spec["split"], streaming=True)
    if "data_dir" in spec:
        kw["data_dir"] = spec["data_dir"]
    args = [spec["path"]] + ([spec["name_cfg"]] if "name_cfg" in spec else [])
    ds = load_dataset(*args, **kw)
    return ds.shuffle(seed=seed, buffer_size=buffer)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out")
    ap.add_argument("--total", type=int, default=200000)
    ap.add_argument("--mix", help="JSON file with a list like DEFAULT_MIX")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-chars", type=int, default=24000, help="longer texts are cut at a line break (about 6K tokens)")
    ap.add_argument("--shard-size", type=int, default=10000)
    ap.add_argument("--shuffle-buffer", type=int, default=10000)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.out:
        ap.error("--out is required")
    mix = json.load(open(a.mix)) if a.mix else DEFAULT_MIX
    q = quotas(mix, a.total)
    os.makedirs(a.out, exist_ok=True)
    man_p = os.path.join(a.out, "manifest.json")
    man = json.load(open(man_p)) if os.path.exists(man_p) else {"seed": a.seed, "done": {}, "counts": {}, "skipped": {}}
    seen = set()
    rows = []
    for spec in mix:
        name = spec["name"]
        part = os.path.join(a.out, f"part-{name}.jsonl")
        if man["done"].get(name):
            print(f"{name}: done earlier ({man['counts'][name]} samples)")
            continue
        want, got, skipped = q[name], 0, 0
        print(f"{name}: streaming {want} samples from {spec['path']} {spec.get('name_cfg', spec.get('data_dir', ''))}", flush=True)
        with open(part, "w", encoding="utf-8") as f:
            for row in stream(spec, a.seed, a.shuffle_buffer):
                text = fmt_row(spec, row)
                if text is None or len(text) < a.min_chars:
                    skipped += 1
                    continue
                if len(text) > a.max_chars:
                    cut = text.rfind("\n", 0, a.max_chars)
                    text = text[: cut if cut > a.max_chars // 2 else a.max_chars] + "<|im_end|>\n"
                h = hashlib.sha1(text.encode("utf-8")).hexdigest()
                if h in seen:
                    skipped += 1
                    continue
                seen.add(h)
                f.write(json.dumps({"text": text, "source": name, "lang": spec["lang"]}, ensure_ascii=False) + "\n")
                got += 1
                if got % 5000 == 0:
                    print(f"  {name}: {got}/{want}", flush=True)
                if got >= want:
                    break
        man["done"][name] = got >= want
        man["counts"][name] = got
        man["skipped"][name] = skipped
        json.dump(man, open(man_p, "w"), indent=1)
        if got < want:
            print(f"  WARNING {name}: stream ended at {got} of {want}")
    # interleave all parts into shuffled shards
    for spec in mix:
        with open(os.path.join(a.out, f"part-{spec['name']}.jsonl"), encoding="utf-8") as f:
            rows.extend(f.readlines())
    random.Random(a.seed).shuffle(rows)
    for i in range(0, len(rows), a.shard_size):
        with open(os.path.join(a.out, f"shard-{i // a.shard_size:05d}.jsonl"), "w", encoding="utf-8") as f:
            f.writelines(rows[i: i + a.shard_size])
    man["total"] = len(rows)
    man["shares"] = {s["name"]: s["share"] for s in mix}
    json.dump(man, open(man_p, "w"), indent=1)
    print(f"wrote {len(rows)} samples in {(len(rows) + a.shard_size - 1) // a.shard_size} shards to {a.out}")


def selftest():
    q = quotas(DEFAULT_MIX, 200000)
    ok = sum(q.values()) == 200000
    groups = {}
    for m in DEFAULT_MIX:
        g = m["lang"] if m["lang"] in ("code", "id") else ("reason" if m["name"] == "openthoughts" else ("dialog" if m["name"] == "ultrachat" else "multi"))
        groups[g] = groups.get(g, 0) + q[m["name"]]
    want = {"code": 90000, "reason": 30000, "dialog": 40000, "id": 20000, "multi": 20000}
    ok &= all(abs(groups[k] - v) <= 2 for k, v in want.items())
    print("quotas per group:", groups)
    samples = [
        ({"fmt": "instr"}, {"input": "Reverse a list in Python.", "output": "def rev(x):\n    return x[::-1]"}),
        ({"fmt": "code:CUDA"}, {"content": "__global__ void k(float *x) { x[threadIdx.x] *= 2; }", "max_stars_repo_path": "k.cu"}),
        ({"fmt": "sharegpt"}, {"system": "Think step by step.", "conversations": [{"from": "human", "value": "2+2?"}, {"from": "gpt", "value": "<think>2+2=4</think>4"}]}),
        ({"fmt": "messages"}, {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}),
        ({"fmt": "doc:id"}, {"text": "Jakarta adalah ibu kota Indonesia."}),
        ({"fmt": "aya", "languages": ["German"]}, {"inputs": "Wie geht's?", "targets": "Gut, danke.", "language": "German"}),
        ({"fmt": "aya", "languages": ["German"]}, {"inputs": "x", "targets": "y", "language": "French"}),
    ]
    for spec, row in samples:
        t = fmt_row(spec, row)
        exp_none = spec["fmt"] == "aya" and row["language"] == "French"
        good = (t is None) if exp_none else (t is not None and t.count("<|im_start|>assistant") == 1 and t.endswith("<|im_end|>\n"))
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] {spec['fmt']}: {repr(t[:70]) if t else None}")
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
