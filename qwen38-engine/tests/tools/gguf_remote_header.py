#!/usr/bin/env python3
"""Reads only the header of a remote GGUF file (metadata + tensor table) with HTTP Range requests, so a 6-50 GB model can be
inspected without downloading it. Unknown (fork-specific) tensor type ids are reported as numbers.
  gguf_remote_header.py URL [--json OUT.json]
Prints the architecture keys, the tensor-type histogram (count and bytes per type, from the offsets), and the per-layer
byte totals. Pure standard library."""
import json
import struct
import sys
import urllib.request

URL = sys.argv[1]
OUT = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None

GGML_NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
              13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
              22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16",
              34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4"}


class Remote:
    def __init__(self, url):
        self.url, self.buf, self.base = url, b"", 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers.get("Content-Length", 0))

    def ensure(self, end):
        while self.base + len(self.buf) < end:
            start = self.base + len(self.buf)
            n = max(8 << 20, end - start)
            req = urllib.request.Request(self.url, headers={"Range": "bytes=%d-%d" % (start, start + n - 1)})
            with urllib.request.urlopen(req) as r:
                self.buf += r.read()


class Reader:
    def __init__(self, rem):
        self.r, self.p = rem, 0

    def take(self, n):
        self.r.ensure(self.p + n)
        b = self.r.buf[self.p - self.r.base:self.p - self.r.base + n]
        self.p += n
        return b

    def u32(self): return struct.unpack("<I", self.take(4))[0]
    def u64(self): return struct.unpack("<Q", self.take(8))[0]
    def str(self): return self.take(self.u64()).decode("utf-8", "replace")

    def value(self, t):
        fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
        if t in fmt:
            s = struct.calcsize(fmt[t])
            return struct.unpack(fmt[t], self.take(s))[0]
        if t == 8:
            return self.str()
        if t == 9:
            et, n = self.u32(), self.u64()
            if et in fmt and n > 64:   # large numeric arrays (token types, scores): skip the payload
                self.take(struct.calcsize(fmt[et]) * n)
                return "<array of %d>" % n
            if et == 8 and n > 64:     # vocabulary strings: skip
                for _ in range(n):
                    self.take(self.u64())
                return "<%d strings>" % n
            return [self.value(et) for _ in range(n)]
        raise ValueError("unknown gguf value type %d" % t)


rem = Remote(URL)
rd = Reader(rem)
assert rd.take(4) == b"GGUF", "not a GGUF file"
ver, n_t, n_kv = rd.u32(), rd.u64(), rd.u64()
kv = {}
for _ in range(n_kv):
    k = rd.str()
    kv[k] = rd.value(rd.u32())
align = kv.get("general.alignment", 32)
tensors = []
for _ in range(n_t):
    name = rd.str()
    nd = rd.u32()
    dims = [rd.u64() for _ in range(nd)]
    tensors.append({"name": name, "dims": dims, "type": rd.u32(), "offset": rd.u64()})
data_start = (rd.p + align - 1) // align * align
order = sorted(range(n_t), key=lambda i: tensors[i]["offset"])
file_size = rem.size
for j, i in enumerate(order):   # byte size of each tensor from the next offset (works for unknown type ids)
    nxt = tensors[order[j + 1]]["offset"] if j + 1 < len(order) else file_size - data_start
    tensors[i]["nbytes"] = nxt - tensors[i]["offset"]

arch = kv.get("general.architecture", "?")
print("file: %s\n  size %.3f GB, GGUF v%d, %d tensors, %d metadata keys, header %.1f MiB read" % (URL.rsplit("/", 1)[-1], file_size / 1e9, ver, n_t, n_kv, rem.base + len(rem.buf) >> 20))
for k in sorted(kv):
    if k.startswith(arch + ".") or k.startswith("general.") and k in ("general.architecture", "general.name", "general.file_type", "general.quantization_version", "general.alignment"):
        print("  %-48s %s" % (k, kv[k]))
for k in sorted(kv):
    if not (k.startswith(arch + ".") or k.startswith("general.") or k.startswith("tokenizer.")):
        print("  %-48s %s" % (k, str(kv[k])[:120]))
hist = {}
for t in tensors:
    n = GGML_NAMES.get(t["type"], "type%d" % t["type"])
    h = hist.setdefault(n, [0, 0, 0])
    h[0] += 1
    h[1] += t["nbytes"]
    e = 1
    for d in t["dims"]:
        e *= d
    h[2] += e
print("  tensor types (count, MiB, bits/weight):")
for n, (c, b, e) in sorted(hist.items(), key=lambda x: -x[1][1]):
    print("    %-10s %5d %10.1f  %6.3f" % (n, c, b / 1048576, 8.0 * b / max(e, 1)))
per = {}
for t in tensors:
    if t["name"].startswith("blk."):
        il = int(t["name"].split(".")[1])
        per[il] = per.get(il, 0) + t["nbytes"]
nb = len(per)
print("  blocks %d, block bytes MiB min %.1f max %.1f total %.1f; non-block MiB %.1f" % (
    nb, min(per.values()) / 1048576, max(per.values()) / 1048576, sum(per.values()) / 1048576,
    sum(t["nbytes"] for t in tensors if not t["name"].startswith("blk.")) / 1048576))
for t in tensors:
    if t["name"].startswith(("blk.0.", "blk.3.")) or not t["name"].startswith("blk."):
        print("    %-40s %-8s %-24s %9.2f MiB" % (t["name"], GGML_NAMES.get(t["type"], "type%d" % t["type"]), t["dims"], t["nbytes"] / 1048576))
if OUT:
    json.dump({"url": URL, "size": file_size, "version": ver, "kv": {k: v for k, v in kv.items() if not k.startswith("tokenizer.ggml.")},
               "tensors": tensors, "data_start": data_start}, open(OUT, "w"), indent=1, default=str)
