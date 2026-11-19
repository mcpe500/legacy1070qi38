#!/usr/bin/env python3
"""Convert a Ternary Bonsai 2 27B GGUF (PTQ1_0 or Q2_0, Hadamard-rotated basis) into a Hugging Face checkpoint
(Qwen3_5ForCausalLM, BF16 or FP32 safetensors) that train_bonsai_mtp.py loads as its frozen teacher.

What the GGUF holds and how it is undone (the inverse of llama.cpp conversion/qwen.py, Qwen3_5TextModel, plus the
PrismML Hadamard fold that qwen38-engine implements in src/cpu/ternary_cpu.cpp):

  * Quantization. PTQ1_0 (ggml type 143): 128 values per 28-byte group, base-3 digits of the payload bytes through the
    empirical byte -> code table kPtq1Code (src/common/ternary_spec.hpp), value = (digit - 1) * fp16 scale at byte 26.
    Q2_0 (type 42): 64 values per 18-byte block, fp16 scale + 2-bit codes, value = (code - 1) * scale. BF16 / F16 / F32
    are read as they are.
  * Hadamard basis. For every name in prism.hadamard.weight_names the engine computes y = W' . H(s * x), with H the
    normalized 1024-block Sylvester-Walsh matrix (symmetric) and s the sign vector of the input width. The plain weight is
    W = W' H diag(s): every row of W' goes through the engine's hadamard_unrotate (butterfly, * 1/32, * s). The token
    embedding (prism.hadamard.inverse_weight_names) is stored rotated the same way and is undone with the same function.
  * Linear-attention V heads. llama.cpp stores the 48 value heads in tiled order (v' = k + 16 r) instead of the grouped
    order of the checkpoint (v = 3 k + r) for in_proj_qkv (V rows), in_proj_z, in_proj_a / in_proj_b, A_log, dt_bias and
    the V channels of conv1d; these are permuted back. out_proj is folded (prism.hadamard.gdn_v_grouped = 1) and already
    in grouped order.
  * Names and value transforms: blk.N.attn_norm -> model.layers.N.input_layernorm (HF Qwen3.5 RMSNorm multiplies by
    1 + w, the GGUF stores w + 1: subtract 1; same for post_attention_norm, output_norm, attn_q_norm, attn_k_norm; the
    gated linear_attn.norm is stored as it is), ssm_a = -exp(A_log) -> A_log = log(-ssm_a), ssm_dt.bias -> dt_bias,
    ssm_conv1d [4, C] -> conv1d.weight [C, 1, 4].

Output: OUT/config.json (Qwen3_5TextConfig, architectures Qwen3_5ForCausalLM), OUT/model-0000N-of-0000M.safetensors and
OUT/model.safetensors.index.json, OUT/generation_config.json, tokenizer files (from --tokenizer-from DIR_OR_REPO, or
built from the GGUF vocabulary and merges with the `tokenizers` library), OUT/conversion.json (source, checks).
Memory: one tensor at a time (the token embedding / LM head in row chunks); peak RAM about 3 GB.

Usage:
  python3 convert_gguf_to_hf_bf16.py --gguf Ternary-Bonsai-2-27B-PTQ1_0.gguf --out /models/Ternary-Bonsai-2-27B-bf16
  python3 convert_gguf_to_hf_bf16.py --gguf ... --out /tmp/mini --layers 4 --vocab-limit 32768 --dtype f32   # test model
  python3 convert_gguf_to_hf_bf16.py --check --gguf Ternary-Bonsai-2-27B-PTQ1_0.gguf      # numeric checks of the inverse
  python3 convert_gguf_to_hf_bf16.py --selftest                                          # synthetic, no files
"""
import argparse
import json
import mmap
import os
import re
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_ROOT = os.path.dirname(os.path.dirname(HERE))
SPEC_HPP = os.path.join(ENGINE_ROOT, "src", "common", "ternary_spec.hpp")

GT_F32, GT_F16, GT_BF16, GT_Q2_0, GT_PTQ1_0 = 0, 1, 30, 42, 143

# ------------------------------------------------------------------------------------------------------------ GGUF read
T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)
SCALAR = {T_U8: "<B", T_I8: "<b", T_U16: "<H", T_I16: "<h", T_U32: "<I", T_I32: "<i", T_F32: "<f", T_BOOL: "<?",
          T_U64: "<Q", T_I64: "<q", T_F64: "<d"}
NP_OF = {T_U8: np.uint8, T_I8: np.int8, T_U16: np.uint16, T_I16: np.int16, T_U32: np.uint32, T_I32: np.int32,
         T_F32: np.float32, T_BOOL: np.bool_, T_U64: np.uint64, T_I64: np.int64, T_F64: np.float64}


class Gguf:
    """Minimal GGUF v3 reader: every metadata value decoded (arrays as lists / numpy), tensors as raw byte views."""

    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.pos = 0
        if self._raw(4) != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF file")
        version, n_t, n_kv = struct.unpack("<IQQ", self._raw(20))
        if version != 3:
            raise SystemExit(f"{path}: GGUF version {version}")
        self.kv = {}
        for _ in range(n_kv):
            k = self._str()
            (vt,) = struct.unpack("<I", self._raw(4))
            self.kv[k] = self._val(vt)
        self.tensors = {}
        order = []
        for _ in range(n_t):
            name = self._str()
            (nd,) = struct.unpack("<I", self._raw(4))
            ne = list(struct.unpack(f"<{nd}Q", self._raw(8 * nd)))
            tt, off = struct.unpack("<IQ", self._raw(12))
            self.tensors[name] = (ne, tt, off)
            order.append((off, name))
        al = int(self.kv.get("general.alignment", 32))
        self.data0 = (self.pos + al - 1) // al * al
        order.sort()
        self.size_of = {}
        for i, (off, name) in enumerate(order):
            end = order[i + 1][0] if i + 1 < len(order) else len(self.mm) - self.data0
            self.size_of[name] = end - off

    def _raw(self, n):
        b = self.mm[self.pos: self.pos + n]
        self.pos += n
        return b

    def _str(self):
        (n,) = struct.unpack("<Q", self._raw(8))
        return self._raw(n).decode("utf-8", errors="replace")

    def _val(self, vt):
        if vt in SCALAR:
            return struct.unpack(SCALAR[vt], self._raw(struct.calcsize(SCALAR[vt])))[0]
        if vt == T_STR:
            return self._str()
        if vt == T_ARR:
            et, n = struct.unpack("<IQ", self._raw(12))
            if et in NP_OF:
                a = np.frombuffer(self._raw(n * np.dtype(NP_OF[et]).itemsize), dtype=NP_OF[et]).copy()
                return a
            return [self._val(et) for _ in range(n)]
        raise SystemExit(f"unknown GGUF value type {vt}")

    def raw(self, name):
        ne, tt, off = self.tensors[name]
        return ne, tt, memoryview(self.mm)[self.data0 + off: self.data0 + off + self.size_of[name]]


# ------------------------------------------------------------------------------------------------------------ decode
def load_ptq1_tables():
    txt = open(SPEC_HPP, encoding="utf-8").read()
    m = re.search(r"kPtq1Code\[256\]\s*=\s*\{([^}]*)\}", txt)
    if not m:
        raise SystemExit(f"kPtq1Code not found in {SPEC_HPP}")
    code = np.array([int(x) for x in m.group(1).replace("\n", " ").split(",") if x.strip()], dtype=np.int32)
    assert code.size == 256
    d5 = np.zeros((256, 5), dtype=np.int8)
    d4 = np.zeros((256, 4), dtype=np.int8)
    for b in range(256):   # same construction as digit_table5 / digit_table4 of src/cpu/ternary_cpu.cpp
        c = int(code[b])
        if c > 242:
            d5[b] = -1
        else:
            v = c
            for d in range(4, -1, -1):
                d5[b, d] = v % 3
                v //= 3
        if c > 242 or c % 3 != 0:
            d4[b] = -1
        else:
            v = c // 3
            for d in range(3, -1, -1):
                d4[b, d] = v % 3
                v //= 3
    return d5, d4


_TABLES = None


def dequant_rows(tt, raw, ne0, r0, r1):
    """Rows r0 .. r1-1 of a [ne1][ne0] tensor as float32."""
    global _TABLES
    n = r1 - r0
    if tt == GT_F32:
        return np.frombuffer(raw, dtype="<f4", count=n * ne0, offset=r0 * ne0 * 4).reshape(n, ne0).copy()
    if tt == GT_F16:
        return np.frombuffer(raw, dtype="<f2", count=n * ne0, offset=r0 * ne0 * 2).reshape(n, ne0).astype(np.float32)
    if tt == GT_BF16:
        u = np.frombuffer(raw, dtype="<u2", count=n * ne0, offset=r0 * ne0 * 2).astype(np.uint32) << 16
        return u.view(np.float32).reshape(n, ne0)
    if tt == GT_Q2_0:
        rb = ne0 // 64 * 18
        b = np.frombuffer(raw, dtype=np.uint8, count=n * rb, offset=r0 * rb).reshape(n, ne0 // 64, 18)
        d = b[:, :, 0:2].copy().view("<f2").astype(np.float32)          # [n, nb, 1]
        q = b[:, :, 2:]                                                    # [n, nb, 16]
        codes = np.stack([(q >> (2 * k)) & 3 for k in range(4)], axis=-1)  # [n, nb, 16, 4] element 4j + k
        vals = codes.reshape(n, ne0 // 64, 64).astype(np.float32) - 1.0
        return (vals * d).reshape(n, ne0)
    if tt == GT_PTQ1_0:
        if _TABLES is None:
            _TABLES = load_ptq1_tables()
        d5, d4 = _TABLES
        rb = ne0 // 128 * 28
        g = np.frombuffer(raw, dtype=np.uint8, count=n * rb, offset=r0 * rb).reshape(-1, 28)
        d = g[:, 26:28].copy().view("<f2").astype(np.float32)              # [G, 1]
        out = np.empty((g.shape[0], 128), dtype=np.float32)
        s1 = d5[g[:, 0:16]]                                                 # [G, 16, 5]
        for s in range(5):
            out[:, 16 * s: 16 * s + 16] = s1[:, :, s]
        s2 = d5[g[:, 16:24]]                                                # [G, 8, 5]
        for s in range(5):
            out[:, 80 + 8 * s: 88 + 8 * s] = s2[:, :, s]
        ta, tb = d4[g[:, 24]], d4[g[:, 25]]                                 # [G, 4]
        for s in range(4):
            out[:, 120 + 2 * s] = ta[:, s]
            out[:, 121 + 2 * s] = tb[:, s]
        if (out < 0).any():
            raise SystemExit("PTQ1_0: a payload byte outside the code table (invalid digit)")
        out = (out - 1.0) * d
        return out.reshape(n, ne0)
    raise SystemExit(f"unsupported tensor type {tt}")


def fwht_rows(x):
    """Unnormalized Walsh-Hadamard transform over the last axis (length 1024), butterfly order of the engine."""
    n = x.shape[-1]
    h = 1
    y = x.reshape(-1, n).copy()
    while h < n:
        y = y.reshape(-1, n // (2 * h), 2, h)
        a, b = y[:, :, 0, :].copy(), y[:, :, 1, :].copy()
        y[:, :, 0, :] = a + b
        y[:, :, 1, :] = a - b
        y = y.reshape(-1, n)
        h *= 2
    return y.reshape(x.shape)


def hadamard_unrotate(rows, signs):
    """Engine hadamard_unrotate on every row: blockwise FWHT(1024) * 1/32 * signs."""
    n, w = rows.shape
    blk = fwht_rows(rows.reshape(n, w // 1024, 1024)).reshape(n, w)
    return blk * np.float32(0.03125) * signs[None, :]


def hadamard_rotate(rows, signs):
    n, w = rows.shape
    return fwht_rows((rows * signs[None, :]).reshape(n, w // 1024, 1024)).reshape(n, w) * np.float32(0.03125)


# ------------------------------------------------------------------------------------------------------------ mapping
NK, NV, HK, HV = 16, 48, 128, 128


def tiled_to_grouped(a, axis, head_dim):
    """Inverse of llama.cpp _reorder_v_heads: [r(3), k(16), hd] -> [k, r, hd] along `axis`."""
    a = np.moveaxis(a, axis, 0)
    sh = a.shape
    r = NV // NK
    a = a.reshape(r, NK, head_dim, *sh[1:]).swapaxes(0, 1).reshape(sh)
    return np.moveaxis(a, 0, axis)


def hf_name(gname):
    if gname == "token_embd.weight":
        return "model.embed_tokens.weight"
    if gname == "output_norm.weight":
        return "model.norm.weight"
    if gname == "output.weight":
        return "lm_head.weight"
    m = re.match(r"blk\.(\d+)\.(.+)$", gname)
    if not m:
        return None
    b, rest = m.group(1), m.group(2)
    table = {
        "attn_norm.weight": "input_layernorm.weight", "post_attention_norm.weight": "post_attention_layernorm.weight",
        "ffn_gate.weight": "mlp.gate_proj.weight", "ffn_up.weight": "mlp.up_proj.weight", "ffn_down.weight": "mlp.down_proj.weight",
        "attn_q.weight": "self_attn.q_proj.weight", "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight", "attn_output.weight": "self_attn.o_proj.weight",
        "attn_q_norm.weight": "self_attn.q_norm.weight", "attn_k_norm.weight": "self_attn.k_norm.weight",
        "attn_qkv.weight": "linear_attn.in_proj_qkv.weight", "attn_gate.weight": "linear_attn.in_proj_z.weight",
        "ssm_alpha.weight": "linear_attn.in_proj_a.weight", "ssm_beta.weight": "linear_attn.in_proj_b.weight",
        "ssm_out.weight": "linear_attn.out_proj.weight", "ssm_norm.weight": "linear_attn.norm.weight",
        "ssm_conv1d.weight": "linear_attn.conv1d.weight", "ssm_a": "linear_attn.A_log", "ssm_dt.bias": "linear_attn.dt_bias",
    }
    if rest not in table:
        return None
    return f"model.layers.{b}.{table[rest]}"


class Converter:
    def __init__(self, path):
        self.g = Gguf(path)
        kv = self.g.kv
        if kv.get("general.architecture") != "qwen35":
            raise SystemExit("not a qwen35 GGUF")
        self.n_layer = int(kv["qwen35.block_count"])
        widths = [int(x) for x in kv["prism.hadamard.sign_widths"]]
        sv = np.asarray(kv["prism.hadamard.sign_values"], dtype=np.float32)
        if sum(widths) != sv.size:
            raise SystemExit("sign_values length does not match sign_widths")
        self.signs = {}
        o = 0
        for w in widths:
            self.signs[w] = sv[o: o + w]
            o += w
        self.rot = set(kv.get("prism.hadamard.weight_names", []))
        self.inv = set(kv.get("prism.hadamard.inverse_weight_names", []))
        self.v_grouped = bool(int(kv.get("prism.hadamard.gdn_v_grouped", 0)))

    def rotated(self, gname):
        return gname in self.rot

    def tensor(self, gname, r0=0, r1=None):
        """Plain-basis float32 rows r0..r1 of a GGUF tensor ([ne1][ne0]; 1-D tensors as [1][ne0])."""
        ne, tt, raw = self.g.raw(gname)
        ne0 = ne[0]
        ne1 = int(np.prod(ne[1:])) if len(ne) > 1 else 1
        r1 = ne1 if r1 is None else r1
        w = dequant_rows(tt, raw, ne0, r0, r1)
        if gname in self.rot or gname in self.inv:
            if ne0 not in self.signs:
                raise SystemExit(f"{gname}: no sign vector of width {ne0}")
            w = hadamard_unrotate(w, self.signs[ne0])
        return w

    def hf_tensor(self, gname, vocab_limit=None, out_bf16=False):
        """(hf_name, array in HF layout) for one GGUF tensor: float32, or the bfloat16 payload (uint16) with out_bf16.
        The token embedding and the LM head (248320 x 5120, 5 GB in FP32) are converted 8192 rows at a time."""
        name = hf_name(gname)
        if name is None:
            return None, None
        ne, tt, _ = self.g.raw(gname)
        rows = int(np.prod(ne[1:])) if len(ne) > 1 else 1
        if gname in ("token_embd.weight", "output.weight") and vocab_limit:
            rows = min(rows, vocab_limit)
        if gname in ("token_embd.weight", "output.weight"):
            w = np.empty((rows, ne[0]), dtype=np.uint16 if out_bf16 else np.float32)
            for r in range(0, rows, 8192):
                c = self.tensor(gname, r, min(rows, r + 8192))
                w[r: r + c.shape[0]] = bf16_bytes(c) if out_bf16 else c
            return name, w
        else:
            w = self.tensor(gname, 0, rows)
        if len(ne) == 1:
            w = w.reshape(ne[0])
        w = self._hf_layout(gname, ne, w)
        return name, (bf16_bytes(w) if out_bf16 else w)

    def _hf_layout(self, gname, ne, w):
        rest = gname.split(".", 2)[-1] if gname.startswith("blk.") else gname
        if rest in ("attn_norm.weight", "post_attention_norm.weight", "attn_q_norm.weight", "attn_k_norm.weight") or gname == "output_norm.weight":
            w = w - 1.0
        elif rest == "ssm_a":
            w = tiled_to_grouped(np.log(-w.astype(np.float64)).astype(np.float32), 0, 1)
        elif rest == "ssm_dt.bias":
            w = tiled_to_grouped(w, 0, 1)
        elif rest == "ssm_conv1d.weight":   # GGUF [C][4] rows -> HF [C, 1, 4]
            qk = 2 * NK * HK
            w = np.concatenate([w[:qk], tiled_to_grouped(w[qk:], 0, HV)], axis=0).reshape(-1, 1, ne[0])
        elif rest == "attn_qkv.weight":
            qk = 2 * NK * HK
            w = np.concatenate([w[:qk], tiled_to_grouped(w[qk:], 0, HV)], axis=0)
        elif rest == "attn_gate.weight":
            w = tiled_to_grouped(w, 0, HV)
        elif rest in ("ssm_alpha.weight", "ssm_beta.weight"):
            w = tiled_to_grouped(w, 0, 1)
        elif rest == "ssm_out.weight" and not self.v_grouped:
            w = tiled_to_grouped(w, 1, HV)
        return w

    def config(self, n_layers, vocab):
        kv = self.g.kv
        p = "qwen35."
        cfg = {
            "architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text",
            "vocab_size": vocab, "hidden_size": int(kv[p + "embedding_length"]), "intermediate_size": int(kv[p + "feed_forward_length"]),
            "num_hidden_layers": n_layers, "num_attention_heads": int(kv[p + "attention.head_count"]),
            "num_key_value_heads": int(kv[p + "attention.head_count_kv"]), "head_dim": int(kv[p + "attention.key_length"]),
            "rms_norm_eps": float(kv[p + "attention.layer_norm_rms_epsilon"]), "hidden_act": "silu",
            "max_position_embeddings": int(kv.get(p + "context_length", 262144)), "tie_word_embeddings": False,
            "attention_bias": False, "attention_dropout": 0.0,
            "linear_conv_kernel_dim": int(kv[p + "ssm.conv_kernel"]), "linear_key_head_dim": int(kv[p + "ssm.state_size"]),
            "linear_value_head_dim": int(kv[p + "ssm.state_size"]), "linear_num_key_heads": int(kv[p + "ssm.group_count"]),
            "linear_num_value_heads": int(kv[p + "ssm.time_step_rank"]),
            "layer_types": ["full_attention" if (i + 1) % int(kv.get(p + "full_attention_interval", 4)) == 0 else "linear_attention"
                            for i in range(n_layers)],
            "rope_parameters": {"rope_type": "default", "rope_theta": float(kv[p + "rope.freq_base"]),
                                "partial_rotary_factor": int(kv[p + "rope.dimension_count"]) / int(kv[p + "attention.key_length"]),
                                "mrope_section": [11, 11, 10], "mrope_interleaved": True},
            "torch_dtype": "bfloat16", "bos_token_id": None,
            "eos_token_id": int(kv.get("tokenizer.ggml.eos_token_id", 248046)),
        }
        return cfg


def bf16_bytes(a):
    """float32 -> bfloat16 with round to nearest even (as torch.to(bfloat16))."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    r = ((u >> 16) & 1) + 0x7FFF
    out = ((u + r) >> 16).astype(np.uint16)
    nan = np.isnan(a)
    if nan.any():
        out[nan] = 0x7FC0
    return out


def write_safetensors(path, tensors):
    """tensors: list of (name, np.ndarray (float32 or uint16 bf16 payload), dtype string)."""
    header, off = {}, 0
    for name, arr, dt in tensors:
        n = arr.nbytes
        header[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + n]}
        off += n
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(path + ".tmp", "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for _, arr, _ in tensors:
            f.write(np.ascontiguousarray(arr).tobytes())
    os.replace(path + ".tmp", path)


def build_tokenizer(conv, out, source):
    if source:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
        tok.save_pretrained(out)
        return f"copied from {source}"
    kv = conv.g.kv
    tokens = kv["tokenizer.ggml.tokens"]
    merges = kv.get("tokenizer.ggml.merges", [])
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, AddedToken
    vocab = {t: i for i, t in enumerate(tokens)}
    bpe = models.BPE(vocab=vocab, merges=[tuple(m.split(" ", 1)) for m in merges], byte_fallback=False, fuse_unk=False)
    tk = Tokenizer(bpe)
    pat = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=__import__("tokenizers").Regex(pat), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tk.decoder = decoders.ByteLevel()
    ttype = kv.get("tokenizer.ggml.token_type")
    special = [tokens[i] for i in range(len(tokens)) if ttype is not None and int(ttype[i]) in (3, 4)]
    tk.add_special_tokens([AddedToken(t, special=True, normalized=False) for t in special])
    tk.save(os.path.join(out, "tokenizer.json"))
    eos = tokens[int(kv.get("tokenizer.ggml.eos_token_id", 0))]
    tpl = kv.get("tokenizer.chat_template")
    json.dump({"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": eos, "pad_token": eos,
               "model_max_length": 262144, **({"chat_template": tpl} if tpl else {})},
              open(os.path.join(out, "tokenizer_config.json"), "w"), indent=1)
    return f"built from the GGUF vocabulary ({len(tokens)} tokens, {len(merges)} merges, {len(special)} special)"


def convert(a):
    conv = Converter(a.gguf)
    n_layers = min(a.layers or conv.n_layer, conv.n_layer)
    if conv.n_layer == 65 and not a.layers:
        n_layers = 64   # blk.64 is the MTP head, not a decoder layer of the trunk
    vocab_full = int(conv.g.raw("token_embd.weight")[0][1])
    vocab = min(vocab_full, a.vocab_limit) if a.vocab_limit else vocab_full
    os.makedirs(a.out, exist_ok=True)
    names = ["token_embd.weight"] + [n for n in conv.g.tensors if n.startswith("blk.") and int(n.split(".")[1]) < n_layers] + \
            ["output_norm.weight", "output.weight"]
    dt = "BF16" if a.dtype == "bf16" else "F32"
    limit = a.shard_gib * (1 << 30)
    t0 = time.time()
    shard, shard_bytes, files, index = [], 0, [], {}

    def flush():
        nonlocal shard, shard_bytes
        if shard:
            fn = f"model-{len(files) + 1:05d}.safetensors.part"
            write_safetensors(os.path.join(a.out, fn), shard)
            for n, _, _ in shard:
                index[n] = fn
            files.append(fn)
        shard, shard_bytes = [], 0

    for gname in names:
        hname, w = conv.hf_tensor(gname, vocab_limit=vocab, out_bf16=dt == "BF16")
        if hname is None:
            print(f"  skip {gname}")
            continue
        arr = w if dt == "BF16" else np.ascontiguousarray(w, dtype=np.float32)
        if shard and shard_bytes + arr.nbytes > limit:
            flush()
        shard.append((hname, arr, dt))
        shard_bytes += arr.nbytes
        print(f"  {gname:36s} -> {hname:56s} {list(w.shape)}  {time.time() - t0:7.1f} s", flush=True)
    flush()
    total = len(files)
    final = {}
    for k, fn in enumerate(files, 1):
        dst_name = f"model-{k:05d}-of-{total:05d}.safetensors"
        os.replace(os.path.join(a.out, fn), os.path.join(a.out, dst_name))
        final[fn] = dst_name
    weight_map = {n: final[f] for n, f in index.items()}
    size = sum(os.path.getsize(os.path.join(a.out, f)) for f in set(weight_map.values()))
    json.dump({"metadata": {"total_size": size}, "weight_map": weight_map}, open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=1)
    cfg = conv.config(n_layers, vocab)
    cfg["torch_dtype"] = "bfloat16" if dt == "BF16" else "float32"
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    json.dump({"eos_token_id": cfg["eos_token_id"], "do_sample": False}, open(os.path.join(a.out, "generation_config.json"), "w"), indent=1)
    tok = "skipped (--no-tokenizer)"
    if not a.no_tokenizer:
        try:
            tok = build_tokenizer(conv, a.out, a.tokenizer_from)
        except Exception as e:   # the weights are still usable; train_bonsai_mtp.py needs a tokenizer for the corpus only
            tok = f"FAILED: {e}"
    json.dump({"source": os.path.basename(a.gguf), "layers": n_layers, "vocab": vocab, "dtype": cfg["torch_dtype"],
               "hadamard_rotated_tensors": len(conv.rot), "inverse_rotated": sorted(conv.inv), "gdn_v_grouped": conv.v_grouped,
               "tokenizer": tok, "seconds": round(time.time() - t0, 1), "bytes": size},
              open(os.path.join(a.out, "conversion.json"), "w"), indent=1)
    print(f"wrote {a.out}: {len(weight_map)} tensors, {size / 2**30:.2f} GiB in {total} shards, {time.time() - t0:.0f} s; tokenizer {tok}")


# ------------------------------------------------------------------------------------------------------------ checks
def check(a):
    """Numeric checks on real tensors: the plain weight reproduces the engine's rotated product."""
    conv = Converter(a.gguf)
    rng = np.random.default_rng(0)
    ok = True
    cand = [n for n in conv.g.tensors if n in conv.rot][:3] + ["blk.3.attn_q.weight", "blk.0.ffn_down.weight"]
    for gname in dict.fromkeys(cand):
        if gname not in conv.g.tensors:
            continue
        ne, tt, raw = conv.g.raw(gname)
        rows = 64
        wq = dequant_rows(tt, raw, ne[0], 0, rows)                # stored (rotated) weight
        x = rng.standard_normal((4, ne[0])).astype(np.float32)
        y_engine = (hadamard_rotate(x, conv.signs[ne[0]]) @ wq.T) if gname in conv.rot else x @ wq.T
        y_plain = x @ conv.tensor(gname, 0, rows).T
        rel = np.abs(y_engine - y_plain).max() / np.abs(y_engine).max()
        good = rel < 1e-5
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] {gname:28s} type {tt:3d} rotated={gname in conv.rot}: W_plain x vs W' H(s x), max rel {rel:.2e}")
    # embedding: engine row = hadamard_unrotate(stored row)
    ne, tt, raw = conv.g.raw("token_embd.weight")
    ids = [0, 11751, 248000]
    stored = np.concatenate([dequant_rows(tt, raw, ne[0], i, i + 1) for i in ids])
    eng = hadamard_unrotate(stored, conv.signs[ne[0]]) if "token_embd.weight" in conv.inv else stored
    mine = np.concatenate([conv.tensor("token_embd.weight", i, i + 1) for i in ids])
    good = np.array_equal(eng, mine)
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] token_embd rows {ids}: identical to the engine's unrotate")
    # round trip of the V-head permutation
    v = np.arange(48 * 3, dtype=np.float32).reshape(48 * 3, 1)
    fwd = v.reshape(16, 3, 3, 1).swapaxes(0, 1).reshape(v.shape)   # llama.cpp _reorder_v_heads with head_dim 3
    back = np.moveaxis(fwd, 0, 0)
    back = back.reshape(3, 16, 3, 1).swapaxes(0, 1).reshape(v.shape)
    good = np.array_equal(back, v) and np.array_equal(tiled_to_grouped(fwd, 0, 3), v)
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] V-head permutation tiled -> grouped inverts llama.cpp _reorder_v_heads")
    print("check:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def selftest():
    ok = True
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 2048)).astype(np.float32)
    s = np.where(rng.random(2048) < 0.5, -1.0, 1.0).astype(np.float32)
    r = hadamard_rotate(x, s)
    back = hadamard_unrotate(r, s)
    e = np.abs(back - x).max()
    ok &= e < 1e-5
    print(f"[{'PASS' if e < 1e-5 else 'FAIL'}] unrotate(rotate(x)) = x, max err {e:.2e}")
    Wr = rng.standard_normal((5, 2048)).astype(np.float32)   # a stored (rotated-basis) weight W'
    y_engine = hadamard_rotate(x, s) @ Wr.T
    y_plain = x @ hadamard_unrotate(Wr, s).T
    e = np.abs(y_engine - y_plain).max() / np.abs(y_engine).max()
    ok &= e < 1e-5
    print(f"[{'PASS' if e < 1e-5 else 'FAIL'}] W' H(s x) = unrotate(W') x, max rel {e:.2e}")
    a = np.arange(48 * 2).reshape(48 * 2)
    fwd = a.reshape(16, 3, 2).swapaxes(0, 1).reshape(-1)
    good = np.array_equal(tiled_to_grouped(fwd, 0, 2), a)
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] tiled_to_grouped inverts the llama.cpp V-head reorder")
    f = np.array([1.0, -2.5, 3.14159, 1e-8, 65504.0], dtype=np.float32)
    import torch
    good = np.array_equal(bf16_bytes(f), torch.from_numpy(f).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16))
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] float32 -> bfloat16 rounding matches torch")
    if os.path.exists(SPEC_HPP):
        d5, d4 = load_ptq1_tables()
        good = d5.shape == (256, 5) and ((d5 >= 0).all(axis=1).sum() == 243)
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] PTQ1_0 code table: {(d5 >= 0).all(axis=1).sum()} valid bytes (expected 243)")
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf")
    ap.add_argument("--out")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "f32"])
    ap.add_argument("--layers", type=int, default=0, help="convert only the first N decoder layers (tests)")
    ap.add_argument("--vocab-limit", type=int, default=0, help="keep only the first N rows of the embedding and LM head (tests)")
    ap.add_argument("--shard-gib", type=float, default=2.0, help="shard size; also the RAM held before a shard is written")
    ap.add_argument("--tokenizer-from", help="HF repo or directory to copy the tokenizer from (default: build from the GGUF)")
    ap.add_argument("--no-tokenizer", action="store_true")
    ap.add_argument("--check", action="store_true", help="numeric checks of the inverse transform on real tensors")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.gguf:
        ap.error("--gguf is required")
    if a.check:
        return check(a)
    if not a.out:
        ap.error("--out is required")
    convert(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
