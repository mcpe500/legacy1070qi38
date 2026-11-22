#!/usr/bin/env python3
"""Inject an MTP draft head (blk.64, one decoder layer + nextn projection) into a Ternary Bonsai 2 27B GGUF.

The Bonsai release has 64 blocks and no MTP head. qwen38-engine enables speculative decoding when the GGUF says
qwen35.block_count = 65 and holds these tensors (same names and shapes as Qwen3.8-27B):

    blk.64.attn_norm.weight [5120] F32          blk.64.post_attention_norm.weight [5120] F32
    blk.64.attn_q.weight [5120 -> 12288]        blk.64.attn_k.weight / attn_v.weight [5120 -> 1024]
    blk.64.attn_output.weight [6144 -> 5120]    blk.64.attn_q_norm.weight / attn_k_norm.weight [256] F32
    blk.64.ffn_gate.weight / ffn_up.weight [5120 -> 17408]   blk.64.ffn_down.weight [17408 -> 5120]
    blk.64.nextn.eh_proj.weight [10240 -> 5120] blk.64.nextn.enorm / hnorm / shared_head_norm [5120] F32

The MTP weights live in the ordinary (not Hadamard-rotated) basis: the engine feeds them the un-rotated residual stream.

Sources (one of):
  --from-checkpoint FILE   .safetensors or torch .pt state dict written by train_bonsai_mtp.py (GGUF tensor names,
                           FP32 / BF16 / F16 values); quantized here with --quant.
  --from-gguf FILE         copy the blk.64.* tensors raw from another qwen35 GGUF (e.g. Qwen3.8-27B: same shapes).
                           Useful to test the engine path; that head was trained on Qwen, not on Bonsai.

Quantization (--quant, matrices only; norms stay F32):
  q8_0     every matrix Q8_0 (8.5 bpw, about 450 MiB for the whole head)
  q4_k_m   Q4_K (4.5 bpw) for every matrix except attn_v and ffn_down, which use Q8_0 (the llama.cpp Q4_K_M rule gives
           those two a higher type; Q6_K is not implemented here, Q8_0 stands in for it). About 270 MiB.

The output is a new file (the input is only read): the original header with block_count = 65 and
qwen35.nextn_predict_layers = 1, the original tensor infos and data copied byte for byte, then the new tensors.
Use --in-place to replace the input file after the new one is complete and verified (atomic rename).

Standalone head file (--mtp-only-out FILE.gguf, no --base needed): a small GGUF with only the 15 quantized blk.64 tensors
(about 273 MiB for q4_k_m, 450 MiB for q8_0). Copy it to the PC with the Bonsai GGUF and inject it there with
--from-gguf FILE.gguf: the tensors are copied raw, so no torch is needed on that machine.

Usage:
  python3 inject_mtp_gguf.py --from-checkpoint mtp_final.safetensors --quant q4_k_m --mtp-only-out mtp_blk64_q4_k_m.gguf
  python3 inject_mtp_gguf.py --base Ternary-Bonsai-2-27B-PTQ1_0.gguf --from-gguf mtp_blk64_q4_k_m.gguf --out ...-mtp.gguf
  python3 inject_mtp_gguf.py --base Ternary-Bonsai-2-27B-PTQ1_0.gguf --from-checkpoint mtp_blk64.safetensors \
      --quant q4_k_m --out Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf
  python3 inject_mtp_gguf.py --base Ternary-Bonsai-2-27B-PTQ1_0.gguf --from-gguf Qwen3.8-27B-UD-Q4_K_M.gguf --out ...
  python3 inject_mtp_gguf.py --selftest      # quantizer round trips (needs numpy; uses llama.cpp gguf-py if found)
"""
import argparse
import hashlib
import os
import struct
import sys

import numpy as np

# ------------------------------------------------------------------------------------------------------------ GGUF IO
GGUF_MAGIC = b"GGUF"
T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)
SCALAR = {T_U8: "<B", T_I8: "<b", T_U16: "<H", T_I16: "<h", T_U32: "<I", T_I32: "<i", T_F32: "<f", T_BOOL: "<?",
          T_U64: "<Q", T_I64: "<q", T_F64: "<d"}
GGML_F32, GGML_F16, GGML_Q8_0, GGML_Q4_K, GGML_Q6_K, GGML_BF16 = 0, 1, 8, 12, 14, 30

MTP_SHAPES = {   # name -> (ne0 = input, ne1 = output) ; 1-D tensors have ne1 = None
    "blk.64.attn_norm.weight": (5120, None), "blk.64.post_attention_norm.weight": (5120, None),
    "blk.64.attn_q.weight": (5120, 12288), "blk.64.attn_k.weight": (5120, 1024), "blk.64.attn_v.weight": (5120, 1024),
    "blk.64.attn_output.weight": (6144, 5120), "blk.64.attn_q_norm.weight": (256, None), "blk.64.attn_k_norm.weight": (256, None),
    "blk.64.ffn_gate.weight": (5120, 17408), "blk.64.ffn_up.weight": (5120, 17408), "blk.64.ffn_down.weight": (17408, 5120),
    "blk.64.nextn.eh_proj.weight": (10240, 5120), "blk.64.nextn.enorm.weight": (5120, None),
    "blk.64.nextn.hnorm.weight": (5120, None), "blk.64.nextn.shared_head_norm.weight": (5120, None),
}


class Reader:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        self.size = os.fstat(self.f.fileno()).st_size
        if self.f.read(4) != GGUF_MAGIC:
            raise SystemExit(f"{path}: not a GGUF file")
        self.version, n_tensors, n_kv = struct.unpack("<IQQ", self.f.read(20))
        if self.version != 3:
            raise SystemExit(f"{path}: GGUF version {self.version}, only 3 is supported")
        self.kv = []   # (key, type, raw value bytes, decoded value or None)
        for _ in range(n_kv):
            key = self._str()
            (vt,) = struct.unpack("<I", self.f.read(4))
            start = self.f.tell()
            val = self._value(vt)
            end = self.f.tell()
            self.f.seek(start)
            raw = self.f.read(end - start)
            self.kv.append((key, vt, raw, val))
        self.tensors = []   # (name, dims, type, offset)
        for _ in range(n_tensors):
            name = self._str()
            (nd,) = struct.unpack("<I", self.f.read(4))
            dims = list(struct.unpack(f"<{nd}Q", self.f.read(8 * nd)))
            ttype, off = struct.unpack("<IQ", self.f.read(12))
            self.tensors.append((name, dims, ttype, off))
        self.alignment = int(self.get("general.alignment", 32))
        here = self.f.tell()
        self.data_start = (here + self.alignment - 1) // self.alignment * self.alignment

    def _str(self):
        (n,) = struct.unpack("<Q", self.f.read(8))
        return self.f.read(n).decode("utf-8")

    def _value(self, vt):
        if vt in SCALAR:
            return struct.unpack(SCALAR[vt], self.f.read(struct.calcsize(SCALAR[vt])))[0]
        if vt == T_STR:
            return self._str()
        if vt == T_ARR:
            et, n = struct.unpack("<IQ", self.f.read(12))
            if et in SCALAR:
                sz = struct.calcsize(SCALAR[et])
                self.f.seek(sz * n, 1)
                return None   # arrays are copied raw, not decoded
            for _ in range(n):
                self._value(et)
            return None
        raise SystemExit(f"unknown GGUF value type {vt}")

    def get(self, key, default=None):
        for k, _, _, v in self.kv:
            if k == key:
                return v
        return default

    def tensor_bytes(self, name):
        """Raw bytes of one tensor (size from the next offset; only used for --from-gguf copies)."""
        offs = sorted(t[3] for t in self.tensors)
        for n, dims, tt, off in self.tensors:
            if n == name:
                nxt = [o for o in offs if o > off]
                end = nxt[0] if nxt else self.size - self.data_start
                self.f.seek(self.data_start + off)
                return dims, tt, self.f.read(end - off)
        raise KeyError(name)


def w_str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def row_bytes(ttype, ne0):
    if ttype == GGML_F32:
        return 4 * ne0
    if ttype in (GGML_F16, GGML_BF16):
        return 2 * ne0
    if ttype == GGML_Q8_0:
        return ne0 // 32 * 34
    if ttype == GGML_Q4_K:
        return ne0 // 256 * 144
    if ttype == GGML_Q6_K:
        return ne0 // 256 * 210
    raise ValueError(f"no row size for type {ttype}")


# ------------------------------------------------------------------------------------------------------------ quantizers
def fp16(x):
    return np.asarray(x, dtype=np.float32).astype(np.float16)


def quant_q8_0(w):
    """ggml quantize_row_q8_0_ref: blocks of 32, d = amax / 127 (fp16), q = round(x / d)."""
    rows, n = w.shape
    b = w.reshape(rows, n // 32, 32).astype(np.float32)
    amax = np.abs(b).max(axis=2, keepdims=True)
    d = amax / 127.0
    idn = np.where(d > 0, 1.0 / np.where(d > 0, d, 1), 0.0)
    q = np.clip(np.trunc(b * idn + np.copysign(0.5, b * idn)), -127, 127).astype(np.int8)   # roundf: half away from 0
    out = np.empty((rows, n // 32, 34), dtype=np.uint8)
    out[:, :, 0:2] = fp16(d[:, :, 0]).view(np.uint8).reshape(rows, n // 32, 2)
    out[:, :, 2:] = q.view(np.uint8)
    return out.reshape(rows, -1)


def _make_qkx2(x, w, nmax=15, rmin=-1.0, rdelta=0.1, nstep=20):
    """Vectorized make_qkx2_quants (ggml-quants.c) over sub-blocks x[S][32] with weights w[S][32]: returns scale[S],
    min[S] (the_min = -min) and codes L[S][32] in 0..nmax."""
    mn = np.minimum(x.min(axis=1), 0.0)
    mx = x.max(axis=1)
    sum_w = w.sum(axis=1)
    sum_x = (w * x).sum(axis=1)
    rng_ = mx - mn
    flat = rng_ <= 0
    rng_safe = np.where(flat, 1.0, rng_)
    iscale = nmax / rng_safe
    scale = 1.0 / iscale
    L = np.clip(np.rint(iscale[:, None] * (x - mn[:, None])), 0, nmax)
    best = (w * (scale[:, None] * L + mn[:, None] - x) ** 2).sum(axis=1)
    best_min = mn.copy()
    for k in range(nstep + 1):
        isc = (rmin + rdelta * k + nmax) / rng_safe
        La = np.clip(np.rint(isc[:, None] * (x - mn[:, None])), 0, nmax)
        sl, sl2, sxl = (w * La).sum(1), (w * La * La).sum(1), (w * La * x).sum(1)
        D = sum_w * sl2 - sl * sl
        ok = D > 0
        Ds = np.where(ok, D, 1.0)
        ts = (sum_w * sxl - sum_x * sl) / Ds
        tm = (sl2 * sum_x - sl * sxl) / Ds
        pos = tm > 0
        ts = np.where(pos, sxl / np.where(sl2 > 0, sl2, 1.0), ts)
        tm = np.where(pos, 0.0, tm)
        mad = (w * (ts[:, None] * La + tm[:, None] - x) ** 2).sum(axis=1)
        better = ok & (mad < best)
        L = np.where(better[:, None], La, L)
        best = np.where(better, mad, best)
        scale = np.where(better, ts, scale)
        best_min = np.where(better, tm, best_min)
    scale = np.where(flat, 0.0, scale)
    L = np.where(flat[:, None], 0, L)
    return scale, -best_min, L.astype(np.int32)


def quant_q4_k(w):
    """ggml quantize_row_q4_K_ref: super-blocks of 256 = 8 x 32, 6-bit scales and mins, fp16 d / dmin, 4-bit codes."""
    rows, n = w.shape
    nb = n // 256
    x = w.reshape(rows * nb, 8, 32).astype(np.float32)
    S = rows * nb
    xs = x.reshape(S * 8, 32)
    av = np.sqrt((xs * xs).sum(axis=1) / 32.0)
    wt = av[:, None] + np.abs(xs)
    sc, mi, _ = _make_qkx2(xs, wt)
    sc = sc.reshape(S, 8)
    mi = mi.reshape(S, 8)
    max_scale = sc.max(axis=1)
    max_min = mi.max(axis=1)
    inv_s = np.where(max_scale > 0, 63.0 / np.where(max_scale > 0, max_scale, 1), 0.0)
    inv_m = np.where(max_min > 0, 63.0 / np.where(max_min > 0, max_min, 1), 0.0)
    ls = np.clip(np.rint(inv_s[:, None] * sc), 0, 63).astype(np.int32)
    lm = np.clip(np.rint(inv_m[:, None] * mi), 0, 63).astype(np.int32)
    d = fp16(max_scale / 63.0)
    dmin = fp16(max_min / 63.0)
    scales = np.zeros((S, 12), dtype=np.int32)
    for j in range(8):
        if j < 4:
            scales[:, j] = ls[:, j]
            scales[:, j + 4] = lm[:, j]
        else:
            scales[:, j + 4] = (ls[:, j] & 0xF) | ((lm[:, j] & 0xF) << 4)
            scales[:, j - 4] |= (ls[:, j] >> 4) << 6
            scales[:, j] |= (lm[:, j] >> 4) << 6
    # recompute the codes with the quantized scales (as the reference does)
    dd = d.astype(np.float32)[:, None] * ls
    dm = dmin.astype(np.float32)[:, None] * lm
    ddn = np.where(dd > 0, dd, 1.0)
    L = np.where(dd[:, :, None] > 0, np.clip(np.rint((x + dm[:, :, None]) / ddn[:, :, None]), 0, 15), 0).astype(np.uint8)
    L = L.reshape(S, 256)
    qs = np.zeros((S, 128), dtype=np.uint8)
    for j in range(0, 256, 64):
        qs[:, j // 2: j // 2 + 32] = L[:, j: j + 32] | (L[:, j + 32: j + 64] << 4)
    out = np.empty((S, 144), dtype=np.uint8)
    out[:, 0:2] = d.view(np.uint8).reshape(S, 2)
    out[:, 2:4] = dmin.view(np.uint8).reshape(S, 2)
    out[:, 4:16] = scales.astype(np.uint8)
    out[:, 16:] = qs
    return out.reshape(rows, nb * 144)


def quantize(name, arr, mode):
    """Returns (ggml type, bytes) for one tensor in GGUF layout [ne1 rows][ne0]."""
    ne0, ne1 = MTP_SHAPES[name]
    a = np.asarray(arr, dtype=np.float32)
    if ne1 is None:
        if a.shape != (ne0,):
            raise SystemExit(f"{name}: shape {a.shape}, expected ({ne0},)")
        return GGML_F32, a.astype("<f4").tobytes()
    if a.shape != (ne1, ne0):
        raise SystemExit(f"{name}: shape {a.shape}, expected ({ne1}, {ne0}) = [out][in]")
    hi = name.endswith(("attn_v.weight", "ffn_down.weight"))
    if mode == "q8_0" or (mode == "q4_k_m" and hi):
        return GGML_Q8_0, quant_q8_0(a).tobytes()
    if mode == "q4_k_m":
        return GGML_Q4_K, quant_q4_k(a).tobytes()
    raise SystemExit(f"unknown --quant {mode}")


# ------------------------------------------------------------------------------------------------------------ sources
# Hugging Face MTP layout (Qwen3.5 `mtp.*`, e.g. ProCreations/Ternary-Bonsai-2-27B-MTP model_mtp.safetensors) -> GGUF names.
# Same mapping as their training/export_mtp.py and llama.cpp's converter. Qwen3.5 RMSNorm weights are zero-centered
# (the layer multiplies by 1 + w), the GGUF and qwen38-engine store the effective multiplier, so 1-D tensors get + 1.
HF_MTP_NAMES = {
    "mtp.fc.weight": "blk.64.nextn.eh_proj.weight",
    "mtp.pre_fc_norm_embedding.weight": "blk.64.nextn.enorm.weight",
    "mtp.pre_fc_norm_hidden.weight": "blk.64.nextn.hnorm.weight",
    "mtp.norm.weight": "blk.64.nextn.shared_head_norm.weight",
    "mtp.layers.0.input_layernorm.weight": "blk.64.attn_norm.weight",
    "mtp.layers.0.post_attention_layernorm.weight": "blk.64.post_attention_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight": "blk.64.attn_q.weight",
    "mtp.layers.0.self_attn.k_proj.weight": "blk.64.attn_k.weight",
    "mtp.layers.0.self_attn.v_proj.weight": "blk.64.attn_v.weight",
    "mtp.layers.0.self_attn.o_proj.weight": "blk.64.attn_output.weight",
    "mtp.layers.0.self_attn.q_norm.weight": "blk.64.attn_q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight": "blk.64.attn_k_norm.weight",
    "mtp.layers.0.mlp.gate_proj.weight": "blk.64.ffn_gate.weight",
    "mtp.layers.0.mlp.up_proj.weight": "blk.64.ffn_up.weight",
    "mtp.layers.0.mlp.down_proj.weight": "blk.64.ffn_down.weight",
}


def from_hf_mtp(sd):
    """Renames a Hugging Face `mtp.*` state dict to GGUF names and turns zero-centered norms into multipliers."""
    out = {}
    for k, v in sd.items():
        if k not in HF_MTP_NAMES:
            raise SystemExit(f"unknown Hugging Face MTP tensor {k}")
        out[HF_MTP_NAMES[k]] = v + 1.0 if v.ndim == 1 else v
    return out


def load_checkpoint(path):
    """blk.64 weights as float32 numpy arrays with GGUF names. Accepts the GGUF naming of train_bonsai_mtp.py and the
    Hugging Face `mtp.*` naming (detected from the keys)."""
    sd = _load_raw(path)
    if any(k.startswith("mtp.") for k in sd):
        print(f"{os.path.basename(path)}: Hugging Face MTP layout (mtp.*), renamed to blk.64 and norms + 1")
        sd = from_hf_mtp(sd)
    return sd


def _load_raw(path):
    if path.endswith(".safetensors"):
        try:   # torch reads FP32 / F16 / BF16; safetensors.numpy has no bfloat16
            from safetensors.torch import load_file   # noqa: E402
            return {k: v.float().numpy() for k, v in load_file(path).items()}
        except ImportError:
            from safetensors.numpy import load_file   # noqa: E402
            return {k: v.astype(np.float32) for k, v in load_file(path).items()}
    import torch   # noqa: E402
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if "mtp" in sd and isinstance(sd["mtp"], dict):
        sd = sd["mtp"]
    return {k: v.float().numpy() for k, v in sd.items()}


def write_mtp_only(ckpt, quant, out, alignment=32, donor=None):
    """A GGUF v3 holding only the blk.64 tensors, usable as --from-gguf donor: quantized from a checkpoint, or (donor =
    path of a qwen35 GGUF with blk.64, e.g. Qwen3.8-27B) copied raw without requantizing."""
    new = []
    if donor:
        d = Reader(donor)
        for name, (ne0, ne1) in MTP_SHAPES.items():
            dims, tt, raw = d.tensor_bytes(name)
            want = [ne0] if ne1 is None else [ne0, ne1]
            if list(dims) != want:
                raise SystemExit(f"{name}: donor shape {dims}, expected {want}")
            nbytes = row_bytes(tt, ne0) * (ne1 or 1) if tt in (GGML_F32, GGML_F16, GGML_BF16, GGML_Q8_0, GGML_Q4_K, GGML_Q6_K) else len(raw)
            new.append((name, want, tt, bytes(raw[:nbytes])))
        ckpt, quant = donor, "raw copy"
    else:
        sd = load_checkpoint(ckpt)
        missing = [n for n in MTP_SHAPES if n not in sd]
        if missing:
            raise SystemExit(f"checkpoint lacks {missing}")
        for name, (ne0, ne1) in MTP_SHAPES.items():
            tt, raw = quantize(name, sd[name], quant)
            new.append((name, [ne0] if ne1 is None else [ne0, ne1], tt, raw))
            print(f"  {name:40s} type {tt:2d}  {len(raw) / 2**20:7.2f} MiB")
    kv = [("general.architecture", T_STR, w_str("qwen35")), ("general.alignment", T_U32, struct.pack("<I", alignment)),
          ("general.name", T_STR, w_str("Ternary Bonsai 2 27B MTP head (blk.64)")),
          ("bonsai.mtp.source", T_STR, w_str(f"{os.path.basename(ckpt)} quantized {quant}"))]
    infos, cur = [], 0
    for name, dims, tt, raw in new:
        infos.append((name, dims, tt, cur))
        cur = (cur + len(raw) + alignment - 1) // alignment * alignment
    tmp = out + ".tmp"
    with open(tmp, "wb") as f:
        f.write(GGUF_MAGIC + struct.pack("<IQQ", 3, len(new), len(kv)))
        for k, vt, raw in kv:
            f.write(w_str(k) + struct.pack("<I", vt) + raw)
        for name, dims, tt, off in infos:
            f.write(w_str(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", tt, off))
        f.write(b"\0" * ((-f.tell()) % alignment))
        data0 = f.tell()
        for (name, dims, tt, raw), (_, _, _, off) in zip(new, infos):
            f.write(b"\0" * (data0 + off - f.tell()))
            f.write(raw)
        f.write(b"\0" * ((-f.tell()) % alignment))
    chk = Reader(tmp)   # verify: every tensor is there with the planned bytes
    for name, dims, tt, raw in new:
        d2, t2, got = chk.tensor_bytes(name)
        if list(d2) != dims or t2 != tt or got[: len(raw)] != raw:
            raise SystemExit(f"verify failed for {name}")
    os.replace(tmp, out)
    print(f"wrote {out}: {len(new)} blk.64 tensors, {os.path.getsize(out) / 2**20:.1f} MiB ({quant})")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", help="Ternary Bonsai GGUF (64 blocks)")
    ap.add_argument("--out", help="output GGUF (default: <base>-mtp.gguf)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--from-checkpoint")
    src.add_argument("--from-gguf")
    ap.add_argument("--quant", default="q4_k_m", choices=["q4_k_m", "q8_0"])
    ap.add_argument("--in-place", action="store_true", help="replace --base with the result (after it is complete)")
    ap.add_argument("--mtp-only-out", help="write only the blk.64 tensors to this GGUF: quantized from --from-checkpoint, "
                                           "or copied raw from --from-gguf (no --base)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.mtp_only_out:
        if not (a.from_checkpoint or a.from_gguf):
            ap.error("--mtp-only-out needs --from-checkpoint or --from-gguf")
        return write_mtp_only(a.from_checkpoint, a.quant, a.mtp_only_out, donor=a.from_gguf)
    if not a.base or not (a.from_checkpoint or a.from_gguf):
        ap.error("--base and one of --from-checkpoint / --from-gguf are required")

    base = Reader(a.base)
    names = {t[0] for t in base.tensors}
    if any(n.startswith("blk.64.") for n in names):
        raise SystemExit(f"{a.base} already has blk.64 tensors")
    if base.get("qwen35.block_count") != 64:
        raise SystemExit(f"{a.base}: qwen35.block_count = {base.get('qwen35.block_count')}, expected 64")

    new = []   # (name, dims, type, bytes)
    if a.from_gguf:
        donor = Reader(a.from_gguf)
        for name, (ne0, ne1) in MTP_SHAPES.items():
            dims, tt, raw = donor.tensor_bytes(name)
            want = [ne0] if ne1 is None else [ne0, ne1]
            if list(dims) != want:
                raise SystemExit(f"{name}: donor shape {dims}, expected {want}")
            new.append((name, want, tt, raw))   # donor padding is kept, the engine reads only the tensor's own bytes
        src_desc = f"copied from {os.path.basename(a.from_gguf)}"
    else:
        sd = load_checkpoint(a.from_checkpoint)
        missing = [n for n in MTP_SHAPES if n not in sd]
        if missing:
            raise SystemExit(f"checkpoint lacks {missing}")
        for name, (ne0, ne1) in MTP_SHAPES.items():
            tt, raw = quantize(name, sd[name], a.quant)
            new.append((name, [ne0] if ne1 is None else [ne0, ne1], tt, raw))
            print(f"  {name:40s} type {tt:2d}  {len(raw) / 2**20:7.2f} MiB")
        src_desc = f"{os.path.basename(a.from_checkpoint)} quantized {a.quant}"

    out = a.out or (a.base[:-5] + "-mtp.gguf")
    tmp = out + ".tmp"
    al = base.alignment
    orig_data = base.size - base.data_start
    cur = (orig_data + al - 1) // al * al
    infos = []
    for name, dims, tt, raw in new:
        infos.append((name, dims, tt, cur))
        cur = (cur + len(raw) + al - 1) // al * al

    kv = []
    have_nextn = False
    for k, vt, raw, val in base.kv:
        if k == "qwen35.block_count":
            raw = struct.pack("<I", 65) if vt == T_U32 else struct.pack(SCALAR[vt], 65)
        if k == "qwen35.nextn_predict_layers":
            have_nextn = True
            raw = struct.pack(SCALAR[vt], 1)
        kv.append((k, vt, raw))
    if not have_nextn:
        kv.append(("qwen35.nextn_predict_layers", T_U32, struct.pack("<I", 1)))
    kv.append(("bonsai.mtp.source", T_STR, w_str(src_desc)))

    with open(tmp, "wb") as f:
        f.write(GGUF_MAGIC + struct.pack("<IQQ", 3, len(base.tensors) + len(new), len(kv)))
        for k, vt, raw in kv:
            f.write(w_str(k) + struct.pack("<I", vt) + raw)
        for name, dims, tt, off in base.tensors:
            f.write(w_str(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", tt, off))
        for name, dims, tt, off in infos:
            f.write(w_str(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", tt, off))
        f.write(b"\0" * ((-f.tell()) % al))
        data0 = f.tell()
        base.f.seek(base.data_start)
        left = orig_data
        while left > 0:   # original tensor data, byte for byte (64 MiB at a time)
            chunk = base.f.read(min(left, 64 << 20))
            f.write(chunk)
            left -= len(chunk)
        for (name, dims, tt, raw), (_, _, _, off) in zip(new, infos):
            f.write(b"\0" * (data0 + off - f.tell()))
            f.write(raw)
        f.write(b"\0" * ((-f.tell()) % al))
    # verify: the new file parses, has 65 blocks and the new tensors at the planned offsets with the planned bytes
    chk = Reader(tmp)
    assert chk.get("qwen35.block_count") == 65
    got = {t[0]: t for t in chk.tensors}
    for name, dims, tt, raw in new:
        _, d2, t2, off = got[name]
        chk.f.seek(chk.data_start + off)
        if chk.f.read(len(raw)) != raw:
            raise SystemExit(f"verify failed for {name}")
    os.replace(tmp, a.base if a.in_place else out)
    print(f"wrote {a.base if a.in_place else out}: {len(new)} MTP tensors ({sum(len(r) for *_, r in new) / 2**20:.1f} MiB), {src_desc}")


def selftest():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((64, 5120)).astype(np.float32) * 0.02
    sys.path.insert(0, os.path.expanduser("~/llama.cpp/gguf-py"))
    try:
        from gguf import quants, GGMLQuantizationType   # noqa: E402
    except ImportError:
        print("gguf-py not found: only the shape checks run")
        quants = None
    ok = True
    for name, fn, gt in (("Q8_0", quant_q8_0, "Q8_0"), ("Q4_K", quant_q4_k, "Q4_K")):
        q = fn(w)
        exp = row_bytes(GGML_Q8_0 if name == "Q8_0" else GGML_Q4_K, 5120)
        if q.shape != (64, exp):
            print(f"[FAIL] {name} row bytes {q.shape[1]} != {exp}")
            ok = False
            continue
        if quants is not None:
            deq = quants.dequantize(q, getattr(GGMLQuantizationType, gt)).reshape(w.shape)
            rel = np.sqrt(((deq - w) ** 2).mean()) / np.sqrt((w ** 2).mean())
            cos = float((deq * w).sum() / np.sqrt((deq ** 2).sum() * (w ** 2).sum()))
            lim = 0.01 if name == "Q8_0" else 0.12
            good = rel < lim
            ok &= good
            print(f"[{'PASS' if good else 'FAIL'}] {name}: relative RMS error {rel:.4f} (limit {lim}), cosine {cos:.6f} vs gguf-py dequantize")
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
