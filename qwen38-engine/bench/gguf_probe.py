#!/usr/bin/env python3
"""Standalone GGUF header probe (no gguf-py, no numpy): works for tensor type ids that gguf-py does not know (PTQ1_0 = 143, Q2_0 = 42).

  bench/gguf_probe.py RESULTS_DIR MODEL.gguf [MODEL.gguf ...]    -> RESULTS_DIR/probe.json and probe.txt (the first model is the reference
                                                                    for the "only in" and "type diff" sections)

Reads GGUF v2/v3: magic, version, tensor_count, kv_count, typed key/value pairs (arrays and strings included), then the tensor infos
(name, n_dims, dims, type, offset). Tensor data is never read. Only the header is parsed, so this is fast and CPU only.
"""
import hashlib
import json
import os
import struct
import sys

# GGUF metadata value types
_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9

# ggml type id -> (name, block size in elements, bytes per block). Unknown ids are reported as "type<id>" with no size.
_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292), 16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2), 42: ("Q2_0", 128, 32), 143: ("PTQ1_0", 128, 34),
}
_KEYS = ("architecture", "name", "file_type", "block_count", "nextn", "mtp", "context_length", "embedding_length", "vocab", "quantization",
         "expert", "ssm", "full_attention")


class _R:
    def __init__(self, fh):
        self.fh = fh

    def read(self, n):
        b = self.fh.read(n)
        if len(b) != n:
            raise EOFError("truncated GGUF header")
        return b

    def unpack(self, fmt):
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self, ver):
        n = self.unpack("<Q" if ver >= 2 else "<I")
        return self.read(n)

    def value(self, vt, ver, want_hash=None):
        """Scalar -> python value, string -> bytes, array -> (n, list or None). `want_hash` set: hash string arrays instead of keeping them."""
        if vt in _SCALARS:
            return self.unpack(_SCALARS[vt])
        if vt == _STRING:
            return self.string(ver)
        if vt == _ARRAY:
            et = self.unpack("<I")
            n = self.unpack("<Q" if ver >= 2 else "<I")
            if et == _STRING:
                h = hashlib.sha256() if want_hash else None
                for _ in range(n):
                    s = self.string(ver)
                    if h is not None:
                        h.update(s)
                        h.update(b"\0")
                return ("array", n, h.hexdigest() if h is not None else None)
            if et in _SCALARS:
                sz = struct.calcsize(_SCALARS[et])
                self.read(sz * n)      # skipped: arrays of numbers (token types, scores) are not needed
                return ("array", n, None)
            raise ValueError("unsupported array element type %d" % et)
        raise ValueError("unsupported GGUF value type %d" % vt)


def tensor_bytes(tid, dims):
    t = _TYPES.get(tid)
    if not t:
        return None
    n = 1
    for d in dims:
        n *= d
    return n // t[1] * t[2]


def probe(path):
    out = {"file": os.path.basename(path), "size": os.path.getsize(path) if os.path.exists(path) else 0}
    if not os.path.exists(path):
        out["error"] = "missing"
        return out
    try:
        with open(path, "rb") as fh:
            r = _R(fh)
            if r.read(4) != b"GGUF":
                raise ValueError("not a GGUF file")
            ver = r.unpack("<I")
            if ver not in (2, 3):
                raise ValueError("unsupported GGUF version %d" % ver)
            n_t = r.unpack("<Q")
            n_kv = r.unpack("<Q")
            meta, tok_hash, tok_n = {}, None, None
            for _ in range(n_kv):
                key = r.string(ver).decode("utf-8", "replace")
                vt = r.unpack("<I")
                v = r.value(vt, ver, want_hash=(key == "tokenizer.ggml.tokens"))
                if key == "tokenizer.ggml.tokens" and isinstance(v, tuple):
                    tok_n, tok_hash = v[1], v[2]
                if key.startswith("tokenizer.ggml.merges"):
                    continue
                if any(w in key for w in _KEYS):
                    if isinstance(v, bytes):
                        v = v.decode("utf-8", "replace")
                    elif isinstance(v, tuple):
                        v = "array[%d]" % v[1]
                    meta[key] = v
            tensors = []
            for _ in range(n_t):
                name = r.string(ver).decode("utf-8", "replace")
                nd = r.unpack("<I")
                dims = [r.unpack("<Q") for _ in range(nd)]
                tid = r.unpack("<I")
                off = r.unpack("<Q")
                tensors.append((name, tid, dims, off))
    except (OSError, ValueError, EOFError, struct.error) as e:
        out["error"] = "parse: %s" % e
        return out
    out["version"] = ver
    out["meta"] = meta
    out["tokenizer_tokens"] = tok_n
    out["tokenizer_sha256"] = tok_hash
    hist, head, types, maxblk = {}, [], {}, -1
    for name, tid, dims, off in tensors:
        tn = _TYPES.get(tid, ("type%d" % tid,))[0]
        nb = tensor_bytes(tid, dims)
        h = hist.setdefault(tn, [0, 0])
        h[0] += 1
        h[1] += nb or 0
        types[name] = {"type": tn, "shape": dims, "bytes": nb}
        if name.startswith("blk."):
            try:
                maxblk = max(maxblk, int(name.split(".")[1]))
            except ValueError:
                pass
        if name.startswith("blk.64.") or "nextn" in name or "mtp" in name.lower():
            head.append({"name": name, "type": tn, "shape": dims, "bytes": nb})
    out.update(n_tensors=len(tensors), max_block_index=maxblk, type_hist={k: {"n": v[0], "bytes": v[1]} for k, v in hist.items()},
               head_tensors=head, probe_types={k: v["type"] for k, v in types.items()})
    for k in ("blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight"):
        if k in types:
            out.setdefault("ssm_ab", {})[k] = types[k]["type"]
    bc = next((v for k, v in meta.items() if k.endswith(".block_count")), None)
    out["block_count"] = bc
    return out


def main():
    res, paths = sys.argv[1], sys.argv[2:]
    os.makedirs(res, exist_ok=True)
    P = [probe(p) for p in paths]
    ref = P[0] if P else {}
    ref_names = set(ref.get("probe_types", {}))
    ref_types = ref.get("probe_types", {})
    for p in P[1:]:
        if "error" in p or "error" in ref:
            continue
        n = set(p["probe_types"])
        p["only_in_this_vs_bonsai2"] = sorted(n - ref_names)[:60]
        p["only_in_bonsai2_vs_this"] = sorted(ref_names - n)[:60]
        diff = [(k, ref_types[k], p["probe_types"][k]) for k in sorted(n & ref_names) if ref_types[k] != p["probe_types"][k]]
        summary = {}
        for k, a, b in diff:
            s = summary.setdefault("%s -> %s" % (a, b), [])
            s.append(k)
        p["type_diff_vs_bonsai2"] = {"n": len(diff), "groups": {g: {"n": len(v), "examples": v[:4]} for g, v in summary.items()}}
    for p in P:
        p.pop("probe_types", None)
    with open(os.path.join(res, "probe.json"), "w") as fh:
        json.dump(P, fh, indent=1)
    with open(os.path.join(res, "probe.txt"), "w") as fh:
        for p in P:
            fh.write("== %s (%d bytes)\n" % (p["file"], p["size"]))
            if "error" in p:
                fh.write("  %s\n" % p["error"])
                continue
            for k, v in sorted(p["meta"].items()):
                fh.write("  %s = %s\n" % (k, v))
            fh.write("  GGUF v%d, block_count %s, tensors: %d, highest blk index %d, tokenizer %s tokens sha256 %s\n"
                     % (p["version"], p["block_count"], p["n_tensors"], p["max_block_index"], p["tokenizer_tokens"], p["tokenizer_sha256"]))
            fh.write("  ssm_alpha / ssm_beta (blk.0) types: %s\n" % json.dumps(p.get("ssm_ab", {})))
            fh.write("  types: %s\n" % json.dumps(p["type_hist"]))
            fh.write("  blk.64 / nextn / mtp tensors (%d):\n" % len(p["head_tensors"]))
            for t in p["head_tensors"]:
                fh.write("    %-48s %-8s %s %s B\n" % (t["name"], t["type"], t["shape"], t["bytes"]))
            for k in ("only_in_this_vs_bonsai2", "only_in_bonsai2_vs_this"):
                if k in p:
                    fh.write("  %s: %s\n" % (k, p[k]))
            if "type_diff_vs_bonsai2" in p:
                d = p["type_diff_vs_bonsai2"]
                fh.write("  type diff vs Bonsai 2: %d tensors differ\n" % d["n"])
                for g, v in d["groups"].items():
                    fh.write("    %s: %d tensors, e.g. %s\n" % (g, v["n"], ", ".join(v["examples"])))
    print(open(os.path.join(res, "probe.txt")).read())
    return 1 if any("error" in p and p["error"] != "missing" for p in P) else 0


if __name__ == "__main__":
    sys.exit(main())
