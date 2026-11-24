#!/usr/bin/env python3
"""Corpus -> tokens for ./build/qwen38-engine --dump-targets (the teacher-free T4 precompute pass).

Selects balanced samples from the prepare_corpus.py shards ({"text","source","lang"} jsonl), tokenizes them
with the ENGINE'S OWN tokenizer (`qwen38-engine -m GGUF --tokenize-file`, the exact BPE the dump pass uses —
no Python-side BPE to maintain or drift), and packs the result as:

  OUT/tokens.bin  int32 little-endian stream of all sample tokens (offsets in plan.json)
  OUT/plan.json   {"seq_len", "samples", "tokens", "items": [{"off", "len", "source"}, ...]}

Every dumped sample is exactly --seq-len tokens, so the C++ pass needs no JSONL/BPE parsing at all.

  python3 tokenize_for_engine.py --corpus /drive/bonsai_mtp/corpus --gguf PTQ1_0.gguf \
      --engine ./build/qwen38-engine --out /content/tok --samples 4000 --seq-len 1024
  python3 tokenize_for_engine.py --selftest
"""
import argparse
import collections
import glob
import json
import os
import random
import subprocess
import sys
import tempfile

import numpy as np

DEFAULT_MIX = [   # from prepare_corpus.py; unknown sources keep an equal share of what is left
    ["OpenCodeInstruct", 0.20], ["OpenThoughts", 0.15], ["UltraChat", 0.20],
    ["Wikipedia-ID", 0.25], ["Aya", 0.20],
]
SEP = "\x1e"   # the --tokenize-file sample separator


def select_samples(corpus, samples, seq_len, seed, mix=None):
    """Round-robin over the corpus shards picking one candidate per source turn, longest-first inside a source,
    so the mix stays balanced without loading everything. Returns [(text, source), ...]."""
    mix = mix or DEFAULT_MIX
    weights = {name: w for name, w in mix}
    shards = sorted(glob.glob(os.path.join(corpus, "shard-*.jsonl")))
    if not shards:
        raise SystemExit(f"no shard-*.jsonl in {corpus} (run prepare_corpus.py first)")
    rng = random.Random(seed)
    min_chars = seq_len * 3          # ~3-4 chars per token: keeps only samples that can fill seq_len
    keep_chars = seq_len * 8         # trim before encoding; seq_len tokens need ~4 chars/token at most
    buckets = collections.defaultdict(list)
    for sp in shards:
        with open(sp, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = r.get("text") or ""
                if len(t) >= min_chars:
                    buckets[r.get("source") or "unknown"].append(t[:keep_chars])
    if not buckets:
        raise SystemExit(f"no sample with >= {min_chars} chars in {corpus}")
    names = sorted(buckets)
    # per-source quotas: DEFAULT_MIX weights among known sources, the rest split evenly
    known_w = {n: weights.get(n) for n in names}
    unk = [n for n in names if known_w[n] is None]
    rest_w = max(0.0, 1.0 - sum(w for w in known_w.values() if w)) / max(1, len(unk))
    quota = {n: (known_w[n] if known_w[n] is not None else rest_w) for n in names}
    for n in names:
        rng.shuffle(buckets[n])
    out, src_count = [], collections.Counter()
    pool = {n: 0 for n in names}
    total_w = sum(quota.values())
    while len(out) < samples:
        progressed = False
        for n in names:                     # one sample per source per round keeps the mix live
            if len(out) >= samples:
                break
            want = quota[n] / total_w
            if src_count[n] >= max(1, round(want * samples)) or pool[n] >= len(buckets[n]):
                continue
            out.append((buckets[n][pool[n]], n))
            pool[n] += 1
            src_count[n] += 1
            progressed = True
        if not progressed:
            break
    if len(out) < samples:
        print(f"note: corpus exhausted at {len(out)} samples (< --samples {samples})", flush=True)
    return out


def write_tokens_bin(out_dir, items_ids, seq_len):
    """items_ids: list of full id lists (already tokenized). Keeps exactly seq_len tokens per sample
    (drops shorter ones), writes tokens.bin + plan.json, returns (kept, plan)."""
    os.makedirs(out_dir, exist_ok=True)
    items, arrs = [], []
    off = 0
    for ids, source in items_ids:
        if len(ids) < seq_len:
            continue
        ids = ids[:seq_len]
        items.append({"off": off, "len": seq_len, "source": source})
        arrs.append(np.asarray(ids, dtype="<i4"))
        off += seq_len
    if not arrs:
        raise SystemExit("no sample reached --seq-len tokens; lower --seq-len or use longer texts")
    binp = os.path.join(out_dir, "tokens.bin")
    tmp = binp + ".tmp"
    with open(tmp, "wb") as f:
        for a in arrs:
            f.write(a.tobytes())
    os.replace(tmp, binp)
    plan = {"seq_len": seq_len, "samples": len(items), "tokens": off,
            "source_counts": dict(collections.Counter(i["source"] for i in items)), "items": items}
    with open(os.path.join(out_dir, "plan.json"), "w") as f:
        json.dump(plan, f)
    return len(items), plan


def tokenize_with_engine(engine, gguf, texts, parse_special=True):
    """The engine tokenizer path: one --tokenize-file call, one [ids] line per sample."""
    fd, txt = tempfile.mkstemp(prefix="tok_", suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(SEP.join(texts))
    try:
        r = subprocess.run([engine, "-m", gguf, "--tokenize-file", txt] +
                           (["--parse-special"] if parse_special else []),
                           capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"engine tokenizer failed ({e.returncode}):\n{e.stderr[-2000:]}") from e
    finally:
        os.unlink(txt)
    out = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            out.append([int(x) for x in line[1:-1].split(",") if x])
    if len(out) != len(texts):
        raise SystemExit(f"engine tokenizer returned {len(out)} lines for {len(texts)} samples")
    return out


def selftest():
    ok = True
    tmp = tempfile.mkdtemp(prefix="tfe_selftest_")
    rng = random.Random(1)
    # fake corpus: 3 sources, uneven sizes; seq_len 32
    d = os.path.join(tmp, "corpus")
    os.makedirs(d)
    n = 0
    for src, count, base_len in (("OpenCodeInstruct", 6, 300), ("Aya", 4, 150), ("Wikipedia-ID", 8, 400)):
        for i in range(count):
            with open(os.path.join(d, f"shard-{n // 4:02d}.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"text": (f"{src} " + "x" * base_len + " "), "source": src, "lang": "id"}) + "\n")
            n += 1
    sel = select_samples(d, samples=10, seq_len=32, seed=7)
    good = len(sel) == 10 and all(len(t) >= 96 for t, _ in sel)
    srcs = {s for _, s in sel}
    ok &= good and len(srcs) == 3
    print(f"[{'PASS' if good and len(srcs) == 3 else 'FAIL'}] balanced selection: {len(sel)} samples from "
          f"{len(srcs)} sources (mix {[s for _, s in sel]})")
    # packing: short samples dropped, offsets contiguous, tokens.bin matches the ids
    fake = [list(rng.sample(range(1000), 20)) for _ in range(3)] + [list(rng.sample(range(1000), 40)) for _ in range(3)]
    items_ids = [(fake[i], "s") for i in range(6)]
    kept, plan = write_tokens_bin(os.path.join(tmp, "tok"), items_ids, 32)
    back = np.fromfile(os.path.join(tmp, "tok", "tokens.bin"), dtype="<i4")
    good = (kept == 3 and plan["tokens"] == 96
            and np.array_equal(back, np.concatenate([np.asarray(i[:32], dtype="<i4") for i, _ in items_ids if len(i) >= 32]))
            and [it["off"] for it in plan["items"]] == [0, 32, 64])
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] packing: kept {kept}/6 (short dropped), offsets {[i['off'] for i in plan['items']]}")
    # determinism
    sel2 = select_samples(d, samples=10, seq_len=32, seed=7)
    ok &= sel == sel2
    print(f"[{'PASS' if sel == sel2 else 'FAIL'}] selection is seed-deterministic")
    print(f"tokenize_for_engine selftest: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", help="prepare_corpus.py output dir (shard-*.jsonl)")
    ap.add_argument("--gguf", help="model GGUF (its tokenizer.ggml.* metadata drives the engine tokenizer)")
    ap.add_argument("--engine", default="./build/qwen38-engine", help="qwen38-engine binary for --tokenize-file")
    ap.add_argument("--out", help="output dir for tokens.bin + plan.json")
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-special", action="store_true", help="tokenize without --parse-special")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.corpus and a.gguf and a.out):
        ap.error("--corpus, --gguf and --out are required (or --selftest)")
    sel = select_samples(a.corpus, a.samples, a.seq_len, a.seed)
    print(f"selected {len(sel)} samples; tokenizing with {a.engine} ...", flush=True)
    ids = tokenize_with_engine(a.engine, a.gguf, [t for t, _ in sel], not a.no_special)
    kept, plan = write_tokens_bin(a.out, list(zip(ids, [s for _, s in sel])), a.seq_len)
    print(f"wrote {a.out}/tokens.bin: {kept} samples x {a.seq_len} tokens = {plan['tokens'] / 1e6:.2f}M tokens; "
          f"sources {plan['source_counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
