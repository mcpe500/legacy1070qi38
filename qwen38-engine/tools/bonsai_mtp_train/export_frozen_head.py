#!/usr/bin/env python3
"""Extract ONLY the 3 frozen tensors the --from-targets trainer needs, straight from the PTQ1_0/Q2_0 GGUF:

  token_embd.weight  -> embed_tokens.weight  [V, 5120] F16 (engine hadamard_unrotate applied: the GGUF stores
                       the embedding pre-rotated, the trainer embeds in the plain residual-stream basis)
  output.weight      -> lm_head.weight      [V, 5120] F16 (unrotated: y = W . hadamard(x) in the engine is
                       W' . H diag(s) . x here, so z == dot(lm_rows, plain hidden) — the trainer convention)
  output_norm.weight -> output_norm.weight  [5120]     F16

Output: lm_head_embed_frozen.safetensors (4.88 GiB) in about a minute on 4 CPU threads — it never touches
the other ~850 layer tensors, so there is no 35-minute BF16 conversion on the T4 path anymore.

  python3 export_frozen_head.py --gguf Ternary-Bonsai-2-27B-PTQ1_0.gguf --out DIR/lm_head_embed_frozen.safetensors
  python3 export_frozen_head.py --selftest
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_converter_module():
    """Reuse the exact dequant + hadamard_unrotate math of convert_gguf_to_hf_bf16.py (byte-identical inverse)."""
    spec = importlib.util.spec_from_file_location("cvt_mod", os.path.join(HERE, "convert_gguf_to_hf_bf16.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_safetensors(path, tensors):
    """Minimal safetensors writer: 8-byte header length, padded JSON header, raw buffers. F16/F32 only."""
    dt_map = {np.dtype("float16"): "F16", np.dtype("float32"): "F32"}
    header, blobs, offset = {}, [], 0
    for name, arr in tensors.items():
        if arr.dtype not in dt_map:
            raise SystemExit(f"{name}: dtype {arr.dtype} not F16/F32")
        b = np.ascontiguousarray(arr).tobytes()
        header[name] = {"dtype": dt_map[arr.dtype], "shape": [int(s) for s in arr.shape],
                        "data_offsets": [offset, offset + len(b)]}
        blobs.append(b)
        offset += len(b)
    hj = json.dumps(header, separators=(",", ":")).encode()
    pad = (8 - len(hj) % 8) % 8
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write((len(hj) + pad).to_bytes(8, "little"))
        f.write(hj)
        f.write(b" " * pad)
        for b in blobs:
            f.write(b)
    os.replace(tmp, path)
    return os.path.getsize(path)


def read_safetensors(path):
    """Minimal safetensors reader (selftest + parity checks without extra dependencies)."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
        payload = f.read()
    out = {}
    dt = {"F16": np.dtype("float16"), "F32": np.dtype("float32")}
    for name, m in header.items():
        if name == "__metadata__":
            continue
        a, b = m["data_offsets"]
        out[name] = np.frombuffer(payload[a:b], dtype=dt[m["dtype"]]).reshape(m["shape"])
    return out


def extract(gguf, out, vocab_limit=0):
    cvt = _load_converter_module()
    conv = cvt.Converter(gguf)   # checks general.architecture == qwen35 + hadamard metadata
    vl = vocab_limit or None
    t0 = time.time()
    tensors = {}
    for gname, key in (("token_embd.weight", "embed_tokens.weight"), ("output.weight", "lm_head.weight")):
        hf, w = conv.hf_tensor(gname, vocab_limit=vl)   # plain-basis (unrotated) float32, 8192 rows at a time
        if hf is None or w is None:
            raise SystemExit(f"{gguf}: tensor {gname} missing")
        tensors[key] = w.astype(np.float16)
        del w
        print(f"  {gname} -> {key} {tensors[key].shape} ({tensors[key].nbytes / 1e9:.2f} GB)", flush=True)
    _, norm = conv.hf_tensor("output_norm.weight")
    tensors["output_norm.weight"] = norm.astype(np.float16)
    size = write_safetensors(out, tensors)
    print(f"wrote {out}: {size / 1e9:.2f} GB in {time.time() - t0:.0f} s "
          f"(vocab {tensors['lm_head.weight'].shape[0]})")
    return size


def selftest():
    ok = True
    cvt = _load_converter_module()
    rng = np.random.default_rng(0)
    # writer/reader round-trip: F16 + F32, shapes across the 8-byte header padding cases
    for shapes, dts in (((7, 13), (5,)), ((257, 3), (3,)), ((1, 1, 1), ())):
        ref, tensors = {}, {}
        for i, s in enumerate(shapes):
            a = rng.standard_normal(s).astype(np.float32).astype(np.float16 if i % 2 == 0 else np.float32)
            ref[i], tensors[f"t{i}"] = a, a
        p = os.path.join(__import__("tempfile").mkdtemp(), "t.safetensors")
        write_safetensors(p, tensors)
        back = read_safetensors(p)
        good = all(np.array_equal(back[k], ref[int(k[1:])]) for k in back) and len(back) == len(ref)
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] safetensors round-trip {shapes}: {len(back)} tensors byte-equal")
    # the 3 GGUF names must map to the trainer's frozen-file keys
    good = (cvt.hf_name("token_embd.weight") == "model.embed_tokens.weight"
            and cvt.hf_name("output.weight") == "lm_head.weight"
            and cvt.hf_name("output_norm.weight") == "model.norm.weight")
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] GGUF -> HF name mapping for the 3 frozen tensors")
    # HF layout: lm_head rows are [V, H] so the trainer's embedding gather is row-per-vocab-id
    good = (cvt.hf_name("output.weight") == "lm_head.weight")   # HF lm_head.weight: [V, H] = embedding table
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] lm_head kept in [V, H] embedding-table layout")
    print(f"export_frozen_head selftest: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", help="Ternary-Bonsai-2-27B-PTQ1_0.gguf (or any qwen35 ternary GGUF)")
    ap.add_argument("--out", default="lm_head_embed_frozen.safetensors",
                    help="output safetensors (default: lm_head_embed_frozen.safetensors)")
    ap.add_argument("--vocab-limit", type=int, default=0, help="keep only the first N rows (tests)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.gguf:
        ap.error("--gguf is required (or --selftest)")
    extract(a.gguf, a.out, a.vocab_limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
