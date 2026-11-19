#!/usr/bin/env python3
"""End-to-end check of convert_gguf_to_hf_bf16.py: the converted Hugging Face model and qwen38-engine must produce the same
residual stream after every decoder layer on the same tokens.

  1. Engine side (GPU or CPU): QWEN_TAP_DUMP=DIR qwen38-engine -m BONSAI.gguf -p "..." -n 1   writes DIR/l_out-<il>.bin
     ([T][5120] FP32 after layer il).
  2. HF side: this script loads the converted checkpoint (any number of layers, FP32), runs the same token ids and hooks the
     output of every decoder layer.
Pass: cosine >= 0.9999 and relative max error <= 1e-3 for every converted layer (the engine quantizes activations to
Q8_1 for its dp4a GEMVs; the HF model multiplies in FP32, so the two differ by that rounding only).

Usage: compare_engine_hf.py --hf DIR --taps DIR --ids "[760,6511,314,9338,369]"
"""
import argparse
import json
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--taps", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", help="write the per-layer table as JSON")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM
    ids = json.loads(a.ids)
    m = AutoModelForCausalLM.from_pretrained(a.hf, dtype=torch.float32)
    m.eval()
    outs = {}
    for i, layer in enumerate(m.model.layers):
        layer.register_forward_hook(lambda mod, inp, out, i=i: outs.__setitem__(i, (out[0] if isinstance(out, tuple) else out).detach()))
    with torch.no_grad():
        m(input_ids=torch.tensor([ids]), use_cache=False)
    rows, ok = [], True
    for i in sorted(outs):
        p = os.path.join(a.taps, f"l_out-{i}.bin")
        eng = np.fromfile(p, dtype=np.float32).reshape(len(ids), -1)
        hf = outs[i][0].numpy()
        cos = float((eng * hf).sum() / np.sqrt((eng ** 2).sum() * (hf ** 2).sum()))
        rel = float(np.abs(eng - hf).max() / np.abs(hf).max())
        good = cos >= 0.9999 and rel <= 1e-3
        ok &= good
        rows.append({"layer": i, "cosine": cos, "rel_max_err": rel, "pass": good})
        print(f"[{'PASS' if good else 'FAIL'}] layer {i:2d} ({'attention' if (i + 1) % 4 == 0 else 'DeltaNet '}): cosine {cos:.7f}, rel max err {rel:.2e}")
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
    print("compare_engine_hf:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
