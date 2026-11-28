#!/usr/bin/env python3
"""Live online distillation of an MTP draft head (blk.64) for Ternary Bonsai 2 27B, on one RTX PRO 6000 (96 GB).

Teacher: Ternary Bonsai 2 27B in BF16 (a Hugging Face checkpoint of the unpacked weights, about 54 GB), frozen, eval mode.
Student: one Qwen3.8-style decoder layer + nextn projection, the layout qwen38-engine runs as blk.64 (about 424M params):

    e = RMSNorm(embed(x_p), enorm)        h = RMSNorm(h_{p-1}, hnorm)           (h_{p-1}: teacher hidden after output_norm)
    z = eh_proj([e | h])                                                         (10240 -> 5120)
    z = z + o_proj( sigmoid(gate) * Attn( RoPE(RMSNorm_head(q)), RoPE(RMSNorm_head(k)), v ) )   on RMSNorm(z, attn_norm)
          q_proj: 5120 -> 24 heads x [q(256) | gate(256)],  k/v: 4 heads x 256 (GQA), NeoX RoPE on 64 of 256 dims, theta 1e7
    z = z + down( silu(gate_proj(n)) * up_proj(n) ),  n = RMSNorm(z, post_attention_norm)                   (5120 -> 17408)
    f = RMSNorm(z, shared_head_norm)  ->  logits = teacher lm_head(f)  (shared, frozen);  f is also the next draft's h

Objective, per training sequence of L tokens (all on the GPU, nothing precomputed: the teacher runs live every step):
  * teacher forward -> H[i] (output-normed hidden) for every position; teacher logits are recomputed per chunk from H
    with the teacher LM head (exactly the teacher's own logits) so the 248,320-wide rows never all sit in memory.
  * K = 4 unrolled self-rollout steps (EAGLE-3 training-time test / HASS): step 1 feeds (H[i], x_{i+1}); step k > 1 feeds the
    student's own feature f^(k-1)[i] with the token x_{i+k}, as the drafting loop of the engine does.
  * loss_k = KL( softmax(Z[i+k] / T) || softmax(student_k / T) ) over the full vocabulary  +  lambda * (1 - cos(f^(k)[i], H[i+k]))
    total = sum_k decay^(k-1) * loss_k.
  * on-policy: with probability --onpolicy-frac a batch is a teacher continuation (sampled, temp 0.8, top-p 0.95) of a corpus
    prefix instead of corpus text, so the head also sees the teacher's own distribution of tokens.
Logged per step k: KL, cosine, and top-1 agreement with the teacher (a proxy of the draft acceptance rate in the engine).

Memory plan (96 GB): teacher BF16 54 GB; student FP32 master 1.7 GB + AdamW 3.4 GB + grads 1.7 GB; activations of the student
with gradient checkpointing per unroll step; KL in chunks of --kl-chunk positions (logits 248,320 x chunk in FP32).

Output: OUT/mtp_step{N}.safetensors (FP32, GGUF tensor names: feed it to inject_mtp_gguf.py; the newest --keep-last are
kept), OUT/train_state.pt (student + optimizer + step, written atomically), OUT/mtp_final.safetensors when --steps is
reached, OUT/train_log.jsonl.

Resume: --resume-from auto (default) continues from OUT/train_state.pt when it exists; --resume-from PATH takes a
train_state.pt (or a directory holding one) from elsewhere; --resume-from none starts fresh. A resumed run reshuffles the
corpus with seed + step (batches are not replayed one by one).
Warm start: --init-mtp-safetensors FILE (a blk.64 safetensors: an mtp_step*.safetensors of an earlier run or the output of
--export-init), or --init gguf:QWEN.gguf / st:FILE. The init is ignored when a train state is resumed.

Usage:
  python3 train_bonsai_mtp.py --teacher /models/Ternary-Bonsai-2-27B-bf16 --corpus /data/bonsai_mtp_corpus --out runs/mtp1 \
      --init gguf:/models/Qwen3.8-27B-UD-Q4_K_M.gguf --steps 12000
  python3 train_bonsai_mtp.py --selftest        # tiny random teacher on CPU: shapes, losses, backward, export names,
                                                 # external warm-start mapping, warning-free generate, save triggers
  python3 train_bonsai_mtp.py --export-init gguf:QWEN.gguf --out x.safetensors   # dequantized init only (warm start file)
  python3 train_bonsai_mtp.py --from-targets /drive/bonsai_mtp/targets --out runs/t4 --steps 2000 --optim adamw8bit \
      --seq-len 1024   # T4/local-GPU mode: no teacher loaded, sparse top-64 KL against the precomputed package
"""
import argparse
import glob
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GGUF_NAMES = {   # student parameter -> GGUF tensor name (blk.64)
    "enorm": "blk.64.nextn.enorm.weight", "hnorm": "blk.64.nextn.hnorm.weight", "eh_proj": "blk.64.nextn.eh_proj.weight",
    "attn_norm": "blk.64.attn_norm.weight", "q_proj": "blk.64.attn_q.weight", "k_proj": "blk.64.attn_k.weight",
    "v_proj": "blk.64.attn_v.weight", "o_proj": "blk.64.attn_output.weight", "q_norm": "blk.64.attn_q_norm.weight",
    "k_norm": "blk.64.attn_k_norm.weight", "post_norm": "blk.64.post_attention_norm.weight",
    "gate_proj": "blk.64.ffn_gate.weight", "up_proj": "blk.64.ffn_up.weight", "down_proj": "blk.64.ffn_down.weight",
    "head_norm": "blk.64.nextn.shared_head_norm.weight",
}

# External warm-start schemas (other names for the same 15 tensors). ProCreations/Ternary-Bonsai-2-27B-MTP ships
# model_mtp.safetensors with the Qwen HF nextn naming (mtp.fc, mtp.layers.0.*, mtp.norm) for the same single layer.
EXTERNAL_ALIASES = {
    "enorm": ("mtp.pre_fc_norm_embedding.weight",),
    "hnorm": ("mtp.pre_fc_norm_hidden.weight",),
    "eh_proj": ("mtp.fc.weight",),
    "attn_norm": ("mtp.layers.0.input_layernorm.weight",),
    "q_proj": ("mtp.layers.0.self_attn.q_proj.weight",),
    "k_proj": ("mtp.layers.0.self_attn.k_proj.weight",),
    "v_proj": ("mtp.layers.0.self_attn.v_proj.weight",),
    "o_proj": ("mtp.layers.0.self_attn.o_proj.weight",),
    "q_norm": ("mtp.layers.0.self_attn.q_norm.weight",),
    "k_norm": ("mtp.layers.0.self_attn.k_norm.weight",),
    "post_norm": ("mtp.layers.0.post_attention_layernorm.weight",),
    "gate_proj": ("mtp.layers.0.mlp.gate_proj.weight",),
    "up_proj": ("mtp.layers.0.mlp.up_proj.weight",),
    "down_proj": ("mtp.layers.0.mlp.down_proj.weight",),
    "head_norm": ("mtp.norm.weight",),
}


class Dims:
    def __init__(self, hidden=5120, n_head=24, n_kv=4, head_dim=256, rope_dim=64, ffn=17408, theta=1e7, eps=1e-6):
        self.hidden, self.n_head, self.n_kv, self.head_dim = hidden, n_head, n_kv, head_dim
        self.rope_dim, self.ffn, self.theta, self.eps = rope_dim, ffn, theta, eps


def rms(x, w, eps):   # engine rmsnorm: x * rsqrt(mean(x^2) + eps) * w  (w as stored in the GGUF, no 1 + w)
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


class MTPHead(nn.Module):
    def __init__(self, d: Dims):
        super().__init__()
        self.d = d
        H, nh, nkv, hd = d.hidden, d.n_head, d.n_kv, d.head_dim
        p = lambda *s: nn.Parameter(torch.empty(*s))   # noqa: E731
        self.enorm, self.hnorm = nn.Parameter(torch.ones(H)), nn.Parameter(torch.ones(H))
        self.eh_proj = p(H, 2 * H)
        self.attn_norm = nn.Parameter(torch.ones(H))
        self.q_proj, self.k_proj, self.v_proj = p(nh * 2 * hd, H), p(nkv * hd, H), p(nkv * hd, H)
        self.o_proj = p(H, nh * hd)
        self.q_norm, self.k_norm = nn.Parameter(torch.ones(hd)), nn.Parameter(torch.ones(hd))
        self.post_norm = nn.Parameter(torch.ones(H))
        self.gate_proj, self.up_proj, self.down_proj = p(d.ffn, H), p(d.ffn, H), p(H, d.ffn)
        self.head_norm = nn.Parameter(torch.ones(H))
        for n, w in self.named_parameters():
            if w.dim() == 2:
                nn.init.normal_(w, std=0.02 / math.sqrt(2.0) if n in ("o_proj", "down_proj") else 0.02)
        inv = 1.0 / (d.theta ** (torch.arange(0, d.rope_dim, 2, dtype=torch.float64) / d.rope_dim))
        self.register_buffer("inv_freq", inv.float(), persistent=False)

    def rope(self, x, pos):   # x [B, heads, L, hd]; NeoX pairs (p, p + rope_dim/2) on the first rope_dim dims
        r = self.d.rope_dim
        ang = pos[:, None, :, None].float() * self.inv_freq   # [B,1,L,r/2]
        c, s = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
        x0, x1, rest = x[..., : r // 2], x[..., r // 2: r], x[..., r:]
        return torch.cat([x0 * c - x1 * s, x0 * s + x1 * c, rest], dim=-1)

    def forward(self, emb, h, pos, kv63=None, pos_abs=None):
        """emb, h: [B, L, H] (token embedding of x_p, hidden of p-1), pos: [B, L] absolute positions. Returns f [B, L, H].
        Shared-KV training (issue #70 P1-rev, --kv63): kv63 = (K, V) teacher layer-63 K/V [B, L, nkv, hd], already
        k_norm+RoPE'd by the engine at the ABSOLUTE positions pos_abs [B, L]; the q side is rotated with pos_abs so the
        phases match, and each row attends only the E-D2b selection (top-32 blocks + window 128 + sink, causal)."""
        d = self.d
        B, L, _ = emb.shape
        if not torch.is_autocast_enabled():   # CPU smoke run: BF16 teacher tensors into the FP32 student
            emb, h = emb.to(self.eh_proj.dtype), h.to(self.eh_proj.dtype)
        z = F.linear(torch.cat([rms(emb, self.enorm, d.eps), rms(h, self.hnorm, d.eps)], -1), self.eh_proj)
        n = rms(z, self.attn_norm, d.eps)
        qg = F.linear(n, self.q_proj).view(B, L, d.n_head, 2, d.head_dim)
        q, gate = qg[:, :, :, 0], qg[:, :, :, 1]
        q = self.rope(rms(q, self.q_norm, d.eps).transpose(1, 2), pos_abs if kv63 is not None else pos)
        if kv63 is None:
            k = F.linear(n, self.k_proj).view(B, L, d.n_kv, d.head_dim)
            v = F.linear(n, self.v_proj).view(B, L, d.n_kv, d.head_dim)
            k = self.rope(rms(k, self.k_norm, d.eps).transpose(1, 2), pos)
            v = v.transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)   # scale 1/sqrt(256) as the engine
        else:
            k63, v63 = kv63                                                     # [B, L, nkv, hd] teacher K/V
            k63 = k63.to(q.dtype).transpose(1, 2)                               # [B, nkv, L, hd]
            v63 = v63.to(q.dtype).transpose(1, 2)
            qg6 = q.view(B, d.n_kv, d.n_head // d.n_kv, L, d.head_dim)
            sc = torch.einsum("bgtld,bgjd->bgtlj", qg6.float(), k63.float()).amax(2)   # [B, nkv, L, L] max over the group's heads
            from iso3_kv import sparse_kv_mask                                  # the E-D2b selection, same rule as the engine
            m = sparse_kv_mask(sc, pos)                                         # [B, L, nkv, L] True = attend (rel causal)
            m4 = m.permute(0, 2, 1, 3).to(torch.bool)                           # [B, nkv, L, L] True = attend
            mk = m4.repeat_interleave(d.n_head // d.n_kv, dim=1)                # broadcast over each group's heads
            a = F.scaled_dot_product_attention(q, k63.repeat_interleave(d.n_head // d.n_kv, dim=1),
                                               v63.repeat_interleave(d.n_head // d.n_kv, dim=1), attn_mask=mk)
        a = a.transpose(1, 2) * torch.sigmoid(gate)
        z = z + F.linear(a.reshape(B, L, d.n_head * d.head_dim), self.o_proj)
        n = rms(z, self.post_norm, d.eps)
        z = z + F.linear(F.silu(F.linear(n, self.gate_proj)) * F.linear(n, self.up_proj), self.down_proj)
        return rms(z, self.head_norm, d.eps)

    def gguf_state(self):
        return {GGUF_NAMES[n]: w.detach().float().cpu().contiguous() for n, w in self.named_parameters()}

    def load_gguf_state(self, sd, strict=True):
        own = dict(self.named_parameters())
        for n, g in GGUF_NAMES.items():
            if g not in sd:
                if strict:
                    raise KeyError(g)
                continue
            t = torch.as_tensor(sd[g]).float().reshape(own[n].shape)
            with torch.no_grad():
                own[n].copy_(t)


def student_shapes(d: Dims):
    """Parameter name -> tensor shape of the student layout (shape checks for the external warm-start mapping)."""
    H, nh, nkv, hd, f = d.hidden, d.n_head, d.n_kv, d.head_dim, d.ffn
    return {"enorm": (H,), "hnorm": (H,), "eh_proj": (H, 2 * H), "attn_norm": (H,),
            "q_proj": (nh * 2 * hd, H), "k_proj": (nkv * hd, H), "v_proj": (nkv * hd, H),
            "o_proj": (H, nh * hd), "q_norm": (hd,), "k_norm": (hd,), "post_norm": (H,),
            "gate_proj": (f, H), "up_proj": (f, H), "down_proj": (H, f), "head_norm": (H,)}


def map_external_state(sd, d: Dims):
    """Renames an external blk.64 schema (GGUF names or the ProCreations mtp.* / HF nextn names) into the internal
    GGUF names, checking every tensor shape. SystemExits with the offending names when a tensor is missing or has
    the wrong shape, instead of a cryptic reshape error later."""
    shapes = student_shapes(d)
    out = {}
    for param, gguf_name in GGUF_NAMES.items():
        src = next((c for c in (gguf_name,) + EXTERNAL_ALIASES.get(param, ()) if c in sd), None)
        if src is None:
            raise SystemExit(f"warm start: no tensor for {gguf_name} (looked for {gguf_name} and "
                             f"{EXTERNAL_ALIASES.get(param, ())}); keys present: {sorted(sd)[:10]} ...")
        t = torch.as_tensor(sd[src]).float()
        if tuple(t.shape) != shapes[param]:
            raise SystemExit(f"warm start: {src} has shape {list(t.shape)}, expected {list(shapes[param])} for {gguf_name}")
        out[gguf_name] = t
    known = set(GGUF_NAMES.values()) | {a for v in EXTERNAL_ALIASES.values() for a in v}
    extras = [k for k in sd if k not in known]
    if extras:
        print(f"warm start: ignoring {len(extras)} unknown tensors: {sorted(extras)[:6]} ...")
    return out


def load_init(spec, d):
    """gguf:PATH (dequantize blk.64 of a qwen35 GGUF with llama.cpp gguf-py), st:PATH (safetensors; a foreign schema
    such as ProCreations model_mtp.safetensors is auto-mapped, shapes checked), or random."""
    if spec is None or spec == "random":
        return None
    kind, path = spec.split(":", 1)
    if kind == "st":
        from safetensors.torch import load_file
        sd = load_file(path)
        if not any(k in sd for k in GGUF_NAMES.values()) and any(k.startswith("mtp.") for k in sd):
            print("warm start: external schema detected (mtp.* keys), mapping to the blk.64 names")
            return map_external_state(sd, d)
        return sd
    if kind == "gguf":
        sys.path.insert(0, os.path.expanduser("~/llama.cpp/gguf-py"))
        from gguf import GGUFReader, quants   # noqa: E402
        r = GGUFReader(path)
        out = {}
        for t in r.tensors:
            if t.name.startswith("blk.64."):
                a = quants.dequantize(t.data, t.tensor_type)
                out[t.name] = torch.from_numpy(a.reshape([int(x) for x in reversed(t.shape)]) if a.ndim > 1 else a.copy())
        return out
    raise SystemExit(f"bad --init {spec}")


def generation_pad_id(tok):
    """A pad token id that never collides with eos, so generate() padding can be told apart from real eos tokens.
    Prefers the tokenizer's own pad token, then common Qwen-style reserve pad tokens. None = keep pad == eos
    (the caller then rewrites the generated tail: every position after a row's first eos is eos anyway)."""
    pad = getattr(tok, "pad_token_id", None)
    eos = getattr(tok, "eos_token_id", None)
    if pad is not None and eos is not None and pad != eos:
        return pad
    for special in ("<|finetune_right_pad_id|>", "<|video_pad|>", "<|fim_pad|>", "<|object_ref_pad|>"):
        try:
            i = tok.convert_tokens_to_ids(special)
        except Exception:
            i = None
        if isinstance(i, int) and i >= 0 and (eos is None or i != eos):
            return i
    return None


def report_accelerators():
    """Prints whether the fused teacher kernels are importable (flash-linear-attention, causal-conv1d) or the run
    falls back to the slower PyTorch reference paths. A causal-conv1d fallback costs roughly 4x the step time."""
    for mod, what in (("flash-linear-attention", "Triton kernels for the 48 Gated DeltaNet layers"),
                      ("causal-conv1d", "fused conv1d of the Gated DeltaNet convs")):
        try:
            m = __import__(mod.replace("-", "_"))
            if mod == "causal-conv1d" and not hasattr(m, "causal_conv1d_fn"):
                raise RuntimeError("causal_conv1d_fn missing")
            print(f"accelerator: {mod}: ACTIVE ({what})", flush=True)
        except Exception as e:
            print(f"accelerator: {mod}: FALLBACK to the PyTorch reference path "
                  f"({type(e).__name__}: {str(e)[:90]})", flush=True)


# ------------------------------------------------------------------------------------------------------------ teacher
class Teacher:
    """Frozen Hugging Face causal LM. hidden(): output-normed last hidden state (hook on the final norm), lm_head shared."""

    def __init__(self, path, device, dtype=torch.bfloat16, attn_impl=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.tok.padding_side = "left"   # generation with padded batches must pad on the left (decoder-only model)
        kw = {"attn_implementation": attn_impl} if attn_impl else {}
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map={"": device}, trust_remote_code=True, **kw)
        self.model.eval().requires_grad_(False)
        self.embed = self.model.get_input_embeddings()
        self.lm_head = self.model.get_output_embeddings()
        norm = None
        for name, mod in self.model.named_modules():
            if name.endswith("model.norm") or name.endswith("language_model.norm"):
                norm = mod
        if norm is None:
            raise SystemExit("teacher: no final norm module (model.norm) found")
        self._h = None
        norm.register_forward_hook(lambda m, i, o: setattr(self, "_h", o))
        cfg = self.model.config
        tc = getattr(cfg, "text_config", cfg)
        rp = getattr(tc, "rope_parameters", None) or {}   # transformers 5 keeps theta / partial factor here
        hd = getattr(tc, "head_dim", 256)
        prf = rp.get("partial_rotary_factor", getattr(tc, "partial_rotary_factor", 0.25))
        self.dims = Dims(hidden=tc.hidden_size, n_head=tc.num_attention_heads, n_kv=tc.num_key_value_heads,
                         head_dim=hd, ffn=tc.intermediate_size, rope_dim=int(round(prf * hd)),
                         theta=float(rp.get("rope_theta", getattr(tc, "rope_theta", 1e7))), eps=tc.rms_norm_eps)

    @torch.no_grad()
    def hidden(self, ids):
        self.model(input_ids=ids, use_cache=False, logits_to_keep=1)   # logits of the last row only; H comes from the hook
        return self._h

    @torch.no_grad()
    def continue_(self, ids, n_new, temp=0.8, top_p=0.95):
        """Teacher continuation for the on-policy batches: left padding side, an explicit attention_mask, and a
        pad id that never collides with eos, so transformers emits neither the missing-attention-mask nor the
        right-padding warning, and the generated rows never contain pad ids (they are rewritten to eos, which in
        ChatML is just a turn boundary, a token the corpus contains everywhere)."""
        tok = self.tok
        eos = tok.eos_token_id
        pad = generation_pad_id(tok)
        out = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=n_new,
                                  do_sample=True, temperature=temp, top_p=top_p,
                                  pad_token_id=pad if pad is not None else eos)
        if pad is not None and pad != eos:
            out[out == pad] = eos   # rows that finished early were padded with `pad`; eos instead, never pad in the data
        return out


# ------------------------------------------------------------------------------------------------------------ data
def packed_batches(corpus, tok, seq_len, batch, seed):
    files = sorted(glob.glob(os.path.join(corpus, "shard-*.jsonl")))
    if not files:
        raise SystemExit(f"no shard-*.jsonl in {corpus} (run prepare_corpus.py)")
    rng = random.Random(seed)
    buf = []
    while True:
        rng.shuffle(files)
        for fpath in files:
            with open(fpath, encoding="utf-8") as f:
                lines = f.readlines()
            rng.shuffle(lines)
            for line in lines:
                buf.extend(tok(json.loads(line)["text"], add_special_tokens=False)["input_ids"])
                while len(buf) >= seq_len * batch:
                    chunk, buf = buf[: seq_len * batch], buf[seq_len * batch:]
                    yield torch.tensor(chunk, dtype=torch.long).view(batch, seq_len)


# ------------------------------------------------------------------------------------------------------------ loss
def chunk_loss(f, h_tgt, lm_w, T, cos_w):
    """f: student features [N, H], h_tgt: teacher hidden of the target positions [N, H]. Teacher logits = lm_head(h_tgt).
    Returns (kl_sum, cos_sum, top1_agree_count) over the N rows."""
    zs = F.linear(f.to(lm_w.dtype), lm_w).float() / T
    with torch.no_grad():
        zt = F.linear(h_tgt.to(lm_w.dtype), lm_w).float() / T
        lp_t = F.log_softmax(zt, -1)
        agree = (zt.argmax(-1) == zs.detach().argmax(-1)).sum()
    lq = F.log_softmax(zs, -1)
    kl = (lp_t.exp() * (lp_t - lq)).sum()
    cos = (1.0 - F.cosine_similarity(f.float(), h_tgt.float(), dim=-1)).sum()
    return kl, cos * cos_w, agree


def unroll_K(args):
    """Unrolled self-rollout steps of this run: --unroll (1..8) for the `mtp` drafter, --df3-block (16) for the
    `df3` block schedule, which is the draft count of the 17-row verification batch of the engine
    (1 target row + B drafts, qwen38-engine src/engine/generator.cpp kVerifyMax)."""
    return args.df3_block if args.drafter == "df3" else args.unroll


def unrolled_loss(student, emb_fn, lm_w, ids, H, args):
    """ids [B, L], H [B, L, Hd] (teacher output-normed hidden). Steps k = 1..K as described in the module docstring."""
    B, L = ids.shape
    K = unroll_K(args)
    total = 0.0
    logs = []
    h_in = H[:, : L - 1]                    # step 1: h of position i feeds the token x_{i+1}
    for k in range(1, K + 1):
        n = L - k                           # positions i = 0 .. L-1-k
        tok = ids[:, k: k + n]              # x_{i+k}
        pos = torch.arange(k, k + n, device=ids.device).expand(B, n)
        emb = emb_fn(tok)

        def run(e, h, p):
            return student(e, h, p)
        f = torch.utils.checkpoint.checkpoint(run, emb, h_in[:, :n], pos, use_reentrant=False) if args.ckpt else run(emb, h_in[:, :n], pos)
        tgt = H[:, k: k + n]                # teacher hidden of position i+k: its logits predict x_{i+k+1}
        fr, tr = f.reshape(-1, f.shape[-1]), tgt.reshape(-1, tgt.shape[-1])
        kl_s = cos_s = 0.0
        agree = 0
        for c in range(0, fr.shape[0], args.kl_chunk):
            kl, cs, ag = torch.utils.checkpoint.checkpoint(chunk_loss, fr[c: c + args.kl_chunk], tr[c: c + args.kl_chunk], lm_w,
                                                           args.temperature, args.cos_weight, use_reentrant=False)
            kl_s, cos_s, agree = kl_s + kl, cos_s + cs, agree + ag
        rows = fr.shape[0]
        lk = (kl_s + cos_s) / rows
        total = total + (args.step_decay ** (k - 1)) * lk
        logs.append((float(kl_s.detach()) / rows, float(cos_s.detach()) / rows / max(args.cos_weight, 1e-12), float(agree) / rows))
        h_in = f                            # self-rollout: the next step feeds the student's own feature
    return total, logs


# --------------------------------------------------------------------------- from-targets mode (T4 / local GPU)
class Frozen:
    """embed_tokens + lm_head from the precompute package's lm_head_embed_frozen.safetensors. The 27B teacher never
    loads in this mode; these 4.7 GiB of frozen FP16 rows are everything the loss touches of it. Parked on the CPU
    when the student is big (--frozen-device cpu / auto): a [N, K, H] row gather is tiny, so the PCIe hop is cheap."""

    def __init__(self, path, dev, gpu):
        from safetensors.torch import load_file
        if not os.path.exists(path):
            raise SystemExit(f"--lm-head-file {path} not found (run precompute_teacher_targets.py first)")
        sd = load_file(path)
        self.dev = dev if gpu else torch.device("cpu")
        self.emb = sd["embed_tokens.weight"].to(self.dev)
        self.lm = sd["lm_head.weight"].to(self.dev)
        self.norm = sd.get("output_norm.weight", torch.zeros(1)).to(self.dev)
        self.vocab = self.lm.shape[0]

    def emb_fn(self, tok):
        e = torch.nn.functional.embedding(tok.to(self.emb.device), self.emb)
        return e.to(tok.device)

    def lm_rows(self, idx):
        r = torch.nn.functional.embedding(idx.to(self.lm.device), self.lm)
        return r.to(idx.device)


def sparse_chunk_loss(f, tk_ids, tk_vals, tk_lse, h_tgt, frozen, T, cos_w):
    """From-targets replacement of chunk_loss. f [N, H] student features; tk_* the stored teacher top-K of the
    target positions; h_tgt the (dequantized) teacher hidden of the target positions.
    Teacher probabilities are renormalized over the stored K set exactly: p_tilde = softmax(topk_vals) = the
    teacher's own conditional distribution restricted to its top-K support, so the loss
    D_KL(p_tilde || q_tilde) is a true KL divergence and >= 0 in exact arithmetic (fp16 vals round it to
    ~1e-4). The stored lse is no longer needed by the loss (it stays in the package for tail-mass audits);
    the student logits are the K gathered lm_head rows renormalized over K — the standard fast-distillation
    approximation that removes the 1.27B-parameter full-vocab lm_head backward."""
    K = tk_ids.shape[-1]
    rows = frozen.lm_rows(tk_ids)                                   # [N, K, H], frozen, no grad
    zs = torch.einsum("nh,nkh->nk", f.to(rows.dtype), rows).float() / T
    with torch.no_grad():
        lp_tilde = torch.log_softmax(tk_vals.float(), -1)           # renormalized teacher log-probs over K
        p_tilde = lp_tilde.exp()
        win = zs.detach().argmax(-1)                                # local winner 0..K-1 ...
        agree = (torch.gather(tk_ids, 1, win[:, None])[:, 0] == tk_ids[:, 0]).sum()   # ... mapped to its vocab id
    lq = torch.log_softmax(zs, -1)
    kl = (p_tilde * (lp_tilde - lq)).sum()
    cos = (1.0 - torch.nn.functional.cosine_similarity(f.float(), h_tgt.float(), dim=-1)).sum()
    return kl, cos * cos_w, agree


def unrolled_loss_targets(student, emb_fn, frozen, ids, H, tk_ids, tk_vals, tk_lse, args, kv63=None, offs=None):
    """The unrolled self-rollout of unrolled_loss, with stored targets instead of a live teacher.
    H is the dequantized teacher hidden of the window, tk_* the stored top-K/lse of every position.
    kv63 = (K, V) teacher layer-63 K/V of the window (--kv63, issue #70 P1-rev); offs [B] = the crop's
    absolute start in the sample, so the q-side RoPE matches the K/V phase the engine cached."""
    B, L = ids.shape
    K = unroll_K(args)
    Kw = tk_ids.shape[-1]   # stored teacher ids per position (sparse-KL width) — unrelated to the unroll K
    if L < K + 2:
        raise SystemExit(f"--seq-len {L} too small for K={K} unrolled steps (need >= {K + 2})")
    total = 0.0
    logs = []
    h_in = H[:, : L - 1]
    pa = None
    if kv63 is not None:
        pa = (torch.arange(L, device=ids.device).unsqueeze(0) + (offs.to(ids.device) if offs is not None else 0).unsqueeze(1))
    for k in range(1, K + 1):
        n = L - k
        tok = ids[:, k: k + n]
        pos = torch.arange(k, k + n, device=ids.device).expand(B, n)
        emb = emb_fn(tok)
        kc = kv63   # teacher positions 0..L-1; the sparse mask clamps each row to j <= its absolute position
        pax = None if pa is None else pa[:, k: k + n]   # the rows' own absolute positions (q-side RoPE phase)
        run = lambda e, h, p: student(e, h, p, kv63=kc, pos_abs=pax)   # noqa: E731
        f = torch.utils.checkpoint.checkpoint(run, emb, h_in[:, :n], pos, use_reentrant=False) if args.ckpt else run(emb, h_in[:, :n], pos)
        tgt = H[:, k: k + n]
        fr, tr = f.reshape(-1, f.shape[-1]), tgt.reshape(-1, tgt.shape[-1])
        tid = tk_ids[:, k: k + n].reshape(-1, Kw)
        tv = tk_vals[:, k: k + n].reshape(-1, Kw)
        tl = tk_lse[:, k: k + n].reshape(-1)
        kl_s = cos_s = 0.0
        agree = 0
        for c in range(0, fr.shape[0], args.kl_chunk):
            kl, cs, ag = sparse_chunk_loss(fr[c: c + args.kl_chunk], tid[c: c + args.kl_chunk],
                                           tv[c: c + args.kl_chunk], tl[c: c + args.kl_chunk],
                                           tr[c: c + args.kl_chunk], frozen, args.temperature, args.cos_weight)
            kl_s, cos_s, agree = kl_s + kl, cos_s + cs, agree + ag
        rows = fr.shape[0]
        lk = (kl_s + cos_s) / rows
        total = total + (args.step_decay ** (k - 1)) * lk
        logs.append((float(kl_s.detach()) / rows, float(cos_s.detach()) / rows / max(args.cos_weight, 1e-12), float(agree) / rows))
        h_in = f
    return total, logs


def target_batches(from_dir, args, dev):
    """Infinite batch iterator over the precompute package: per epoch a fresh shard shuffle and per row a random
    crop offset, so multi-epoch runs see different block/anchor positions (what the df3 schedule wants)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import precompute_teacher_targets as pc
    files = sorted(p for p in glob.glob(os.path.join(from_dir, "target-shard-*"))
                   if p.endswith(".npz") or os.path.isdir(p))   # Python .npz shards or engine shard directories
    if not files:
        raise SystemExit(f"no target-shard-*.npz in {from_dir} (run precompute_teacher_targets.py first)")
    epoch = 0
    dtype_note = False
    kv63_files = args.kv63
    while True:
        rng = random.Random(args.seed + 131 * epoch)
        order = files[:]
        rng.shuffle(order)
        for fp in order:
            local = pc.stage_shard(fp, args.stage_dir)
            d, meta = pc.load_shard(local)
            dt = pc.hidden_dtype_of(meta, fp)
            if kv63_files and not meta.get("kv63"):
                raise SystemExit(f"--kv63 set but shard {fp} was dumped without --dump-kv63 (re-run the dump)")
            dt = pc.hidden_dtype_of(meta, fp)
            if not dtype_note and dt != args.hidden_dtype:
                print(f"note: shards store hidden as {dt}, --hidden-dtype {args.hidden_dtype} is ignored for decoding",
                      flush=True)
                dtype_note = True
            if not dtype_note and meta.get("top_k") not in (None, args.top_k_kl):
                print(f"note: shards store top_k={meta.get('top_k')}, --top-k-kl {args.top_k_kl} is informational only; "
                      f"the loss uses the stored ids", flush=True)
                dtype_note = True
            H = torch.from_numpy(pc.dequantize_hidden(d["hq"], d["hs"], dt)).half()
            ids = torch.from_numpy(d["ids"].astype(np.int64))
            ti = torch.from_numpy(d["tk_ids"].astype(np.int64))
            tv = torch.from_numpy(d["tk_vals"])
            tl = torch.from_numpy(d["tk_lse"])
            S, Ls = ids.shape
            L = min(args.seq_len, Ls)
            if L < unroll_K(args) + 2:
                raise SystemExit(f"stored seq {Ls} < --seq-len {args.seq_len} needed for K={unroll_K(args)} steps")
            for b in range(0, S, args.batch):
                idxs = list(range(b, min(b + args.batch, S)))
                offs = [rng.randint(0, Ls - L) for _ in idxs] if Ls > L else [0] * len(idxs)
                sl = lambda t: torch.stack([t[i, o: o + L] for i, o in zip(idxs, offs)])   # noqa: E731
                extra = None
                if kv63_files:
                    import iso3_kv
                    kv = iso3_kv.dequant_shard(local)
                    if kv is None:
                        raise SystemExit(f"shard {fp} meta says kv63 but kv63_k.npy/kv63_v.npy are missing")
                    kt = (torch.from_numpy(kv[0]).half(), torch.from_numpy(kv[1]).half())
                    extra = (sl(kt[0]).to(dev), sl(kt[1]).to(dev),
                             torch.tensor(offs, dtype=torch.long))
                yield (sl(ids).to(dev), sl(H).to(dev), sl(ti).to(dev), sl(tv).to(dev), sl(tl).to(dev), extra)
        epoch += 1
        if args.epochs and epoch >= args.epochs:
            print(f"--epochs {args.epochs} exhausted; stopping at an optimizer boundary", flush=True)
            return


def make_optim(name, params, lr, dev_type="cpu", allow_bnb=True):
    """Optimizer chain for low-VRAM cards: adamw8bit (bitsandbytes) -> adafactor -> adamw. Returns (opt, name_used)
    so the train state records what really ran (a resumed run resets optimizer state on a mismatch, as live mode)."""
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr), "sgd"
    if name == "adamw8bit" and allow_bnb:
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.01), "adamw8bit"
        except ImportError as e:
            print(f"--optim adamw8bit: bitsandbytes unavailable ({type(e).__name__}); falling back to Adafactor",
                  flush=True)
            name = "adafactor"
    if name in ("adafactor", "adamw8bit"):
        try:
            from transformers.optimization import Adafactor
            return Adafactor(params, lr=lr, scale_parameter=False, relative_step=False, warmup_init=False), "adafactor"
        except ImportError:
            print("Adafactor unavailable; falling back to AdamW (FP32 optimizer state, more memory)", flush=True)
    return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.01,
                             fused=dev_type == "cuda"), "adamw"


def train_dtype(dev):
    """(autocast dtype, GradScaler) for the from-targets loop. sm_75 (T4) has FP16/INT8 tensor cores but no BF16
    tensor cores: BF16 there silently emulates in FP32, so pre-Ampere cards train FP16 with a GradScaler."""
    if dev.type != "cuda":
        return torch.float32, False
    major = torch.cuda.get_device_capability(dev)[0]
    return (torch.float16, True) if major < 8 else (torch.bfloat16, False)


def run_training(args, dev, student, opt, batch_iter, loss_fn, amp_dtype, use_scaler, save, step0):
    """The shared optimizer loop of the live and from-targets modes: warmup+cosine LR, grad accumulation, atomic
    saves on --save-every / --save-every-min / signals / time limit, and the per-step JSONL telemetry."""
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler and dev.type == "cuda")

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        p = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))

    t0 = time.time()
    log_f = open(os.path.join(args.out, "train_log.jsonl"), "a")
    last_save_t, last_log_t, last_log_step = t0, t0, step0
    stop = {"sig": None}
    import signal

    def _sig(s, _f):
        stop["sig"] = s
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _sig)

    for step in range(step0, args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        acc_logs = []
        for _ in range(args.grad_accum):
            try:
                payload = next(batch_iter)
            except StopIteration:      # --epochs exhausted: save what we have at an optimizer boundary
                if time.time() - last_save_t > 1.0:
                    save(step)
                print(f"epoch limit reached after step {step}, stopping", flush=True)
                log_f.close()
                return
            with torch.autocast(dev.type, dtype=amp_dtype, enabled=dev.type == "cuda" and amp_dtype != torch.float32):
                loss, logs = loss_fn(payload)
            if use_scaler and dev.type == "cuda":
                scaler.scale(loss / args.grad_accum).backward()
            else:
                (loss / args.grad_accum).backward()
            acc_logs.append(logs)
        if use_scaler and dev.type == "cuda":
            scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        if use_scaler and dev.type == "cuda":
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        now = time.time()
        if step % args.log_every == 0:
            K = unroll_K(args)
            avg = [[sum(l[k][j] for l in acc_logs) / len(acc_logs) for j in range(3)] for k in range(K)]
            sps = (now - last_log_t) / max(1, step - last_log_step)
            rec = {"step": step, "lr": lr_at(step), "grad_norm": float(gn), "s": time.time() - t0,
                   "kl": [a[0] for a in avg], "cos_loss": [a[1] for a in avg], "top1_agree": [a[2] for a in avg],
                   "sec_per_step": round(sps, 3),
                   "tok_per_sec": round(args.batch * args.grad_accum * args.seq_len / sps, 1) if sps else None,
                   "elapsed_min": round((now - t0) / 60.0, 1),
                   "eta_h": round((args.steps - step - 1) * sps / 3600.0, 2)}
            print(json.dumps(rec), flush=True)
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            last_log_t, last_log_step = now, step
        if save_due(step + 1, args, now - last_save_t):
            save(step + 1)
            last_save_t = now
        if args.time_limit_min and (now - t0) / 60.0 >= args.time_limit_min:
            print(f"time limit of {args.time_limit_min} min reached after step {step + 1}, stopping", flush=True)
            break
        if stop["sig"] is not None:
            if now - last_save_t > 1.0:
                save(step + 1)
            print(f"{signal.Signals(stop['sig']).name} received after step {step + 1}, stopping", flush=True)
            break
    log_f.close()


def train_targets(args):
    """--from-targets: the whole training loop without the teacher. Loads the frozen lm_head/embed rows, dequantizes
    the stored INT4 hidden per batch, and trains the same blk.64 student with the sparse top-K KL."""
    dev = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(args.out, exist_ok=True)
    dims_p = os.path.join(args.from_targets, "dims.json")
    if not os.path.exists(dims_p):
        raise SystemExit(f"{dims_p} not found (the precompute package ships it)")
    d = Dims(**json.load(open(dims_p)))
    student = MTPHead(d).to(dev)
    init_spec = f"st:{args.init_mtp_safetensors}" if args.init_mtp_safetensors else args.init
    resume = find_resume(args.resume_from, args.out)
    init = None if resume else load_init(init_spec, d)
    if init is not None:
        student.load_gguf_state(init)
        print(f"init: {init_spec}")
    n_par = sum(p.numel() for p in student.parameters())
    gpu_frozen = args.frozen_device == "gpu" or (args.frozen_device == "auto" and n_par <= 5e8)
    frozen = Frozen(args.lm_head_file or os.path.join(args.from_targets, "lm_head_embed_frozen.safetensors"),
                    dev, gpu_frozen)
    print(f"student: {n_par / 1e6:.1f}M parameters; frozen lm_head/embed on {frozen.dev} "
          f"(vocab {frozen.vocab}); dims {vars(d)}", flush=True)
    opt, optim_used = make_optim(args.optim, student.parameters(), args.lr, dev.type)
    print(f"optimizer: {optim_used}", flush=True)
    step0 = 0
    st_path = os.path.join(args.out, "train_state.pt")
    if resume:
        st = torch.load(resume, map_location="cpu", weights_only=False)
        student.load_state_dict(st["student"])
        if st.get("optim") in (None, optim_used):
            opt.load_state_dict(st["opt"])
        else:
            print(f"resume: optimizer {st.get('optim')} != {optim_used}, optimizer state reset")
        step0 = int(st["step"])
        print(f"resumed at step {step0} from {resume}")
    if step0 >= args.steps:
        print(f"nothing to do: step {step0} >= --steps {args.steps}")
        if not os.path.exists(os.path.join(args.out, "mtp_final.safetensors")):
            from safetensors.torch import save_file
            save_file(student.gguf_state(), os.path.join(args.out, "mtp_final.safetensors"))
        return

    def save(done_steps):
        from safetensors.torch import save_file
        sd = student.gguf_state()
        ck = os.path.join(args.out, f"mtp_step{done_steps}.safetensors")
        save_file(sd, ck + ".tmp")
        os.replace(ck + ".tmp", ck)
        torch.save({"student": student.state_dict(), "opt": opt.state_dict(), "step": done_steps,
                    "optim": optim_used, "args": vars(args)}, st_path + ".tmp")
        os.replace(st_path + ".tmp", st_path)
        prune_checkpoints(args.out, args.keep_last)
        if done_steps == args.steps:
            save_file(sd, os.path.join(args.out, "mtp_final.safetensors"))
        print(f"saved step {done_steps}: {ck}", flush=True)

    dtype, use_scaler = train_dtype(dev)
    print(f"precision: {str(dtype).replace('torch.', '')}"
          f"{' + GradScaler' if use_scaler else ''} (auto per GPU)", flush=True)
    batches = target_batches(args.from_targets, args, dev)

    def loss_fn(payload):
        ids, H, ti, tv, tl, extra = payload
        kv63 = offs = None
        if extra is not None:
            kv63, offs = extra[0], extra[2]
        return unrolled_loss_targets(student, frozen.emb_fn, frozen, ids, H, ti, tv, tl, args, kv63=kv63, offs=offs)

    run_training(args, dev, student, opt, batches, loss_fn, dtype, use_scaler, save, step0)


# ------------------------------------------------------------------------------------------------------------ main
def random_batches(vocab, seq_len, batch, seed):
    """--smoke: random token ids (no corpus, no tokenizer), for a pipeline test of the teacher + student + loss + step."""
    g = torch.Generator().manual_seed(seed)
    while True:
        yield torch.randint(0, vocab, (batch, seq_len), generator=g)


def find_resume(spec, out):
    """--resume-from: 'auto' -> OUT/train_state.pt if present, 'none' -> None, else a train_state.pt or a directory holding one."""
    if spec in (None, "", "none"):
        return None
    if spec == "auto":
        p = os.path.join(out, "train_state.pt")
        return p if os.path.exists(p) else None
    p = os.path.join(spec, "train_state.pt") if os.path.isdir(spec) else spec
    if not os.path.exists(p):
        raise SystemExit(f"--resume-from {spec}: no train_state.pt found")
    return p


def prune_checkpoints(out, keep):
    """Keeps the newest `keep` mtp_step*.safetensors in OUT (0 keeps all)."""
    if keep <= 0:
        return
    files = sorted(glob.glob(os.path.join(out, "mtp_step*.safetensors")),
                   key=lambda p: int(re.search(r"mtp_step(\d+)\.safetensors$", p).group(1)))
    for p in files[:-keep]:
        os.remove(p)


def latest_checkpoint(out):
    """Path of the newest mtp_step*.safetensors (or mtp_final.safetensors) in OUT, None if there is none."""
    fin = os.path.join(out, "mtp_final.safetensors")
    if os.path.exists(fin):
        return fin
    files = glob.glob(os.path.join(out, "mtp_step*.safetensors"))
    return max(files, key=lambda p: int(re.search(r"mtp_step(\d+)\.safetensors$", p).group(1))) if files else None


def save_due(next_step, args, since_last_s):
    """A checkpoint is due at a --save-every multiple, at --steps, or once --save-every-min minutes have passed
    since the last save (whichever comes first; 0 disables the time trigger, 0 for --save-every disables the
    step trigger). Long Colab steps (71-76 s without causal-conv1d) then never lose more than 20 minutes."""
    if args.save_every and next_step % args.save_every == 0:
        return True
    if next_step == args.steps:
        return True
    return bool(args.save_every_min) and since_last_s >= args.save_every_min * 60.0


def train(args):
    dev = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out, exist_ok=True)
    report_accelerators()
    resume = find_resume(args.resume_from, args.out)
    teacher = Teacher(args.teacher, dev, dtype=torch.float32 if args.teacher_f32 else torch.bfloat16, attn_impl=args.attn_impl)
    d = teacher.dims
    student = MTPHead(d).to(dev)
    init_spec = f"st:{args.init_mtp_safetensors}" if args.init_mtp_safetensors else args.init
    init = None if resume else load_init(init_spec, d)
    if init is not None:
        student.load_gguf_state(init)
        print(f"init: {init_spec}")
    n_par = sum(p.numel() for p in student.parameters())
    print(f"student: {n_par / 1e6:.1f}M parameters; teacher dims {vars(d)}")
    if args.optim == "sgd":   # --smoke on a small machine: no optimizer state
        opt = torch.optim.SGD(student.parameters(), lr=args.lr)
    else:
        opt = torch.optim.AdamW(student.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01, fused=dev.type == "cuda")
    step0 = 0
    st_path = os.path.join(args.out, "train_state.pt")
    if resume:
        st = torch.load(resume, map_location="cpu", weights_only=False)
        student.load_state_dict(st["student"])
        if st.get("optim") in (None, args.optim):
            opt.load_state_dict(st["opt"])
        else:
            print(f"resume: optimizer {st.get('optim')} != --optim {args.optim}, optimizer state reset")
        step0 = int(st["step"])
        print(f"resumed at step {step0} from {resume}")
    if step0 >= args.steps:
        print(f"nothing to do: step {step0} >= --steps {args.steps}")
        if not os.path.exists(os.path.join(args.out, "mtp_final.safetensors")):
            from safetensors.torch import save_file
            save_file(student.gguf_state(), os.path.join(args.out, "mtp_final.safetensors"))
        return

    vocab = teacher.lm_head.weight.shape[0]
    data = random_batches(vocab, args.seq_len, args.batch, args.seed + step0) if args.smoke else \
        packed_batches(args.corpus, teacher.tok, args.seq_len, args.batch, args.seed + step0)
    rng = random.Random(args.seed + 7 * step0)
    lm_w = teacher.lm_head.weight
    emb_fn = lambda t: teacher.embed(t)   # noqa: E731  (frozen, BF16)

    def save(done_steps):
        from safetensors.torch import save_file
        sd = student.gguf_state()
        ck = os.path.join(args.out, f"mtp_step{done_steps}.safetensors")
        save_file(sd, ck + ".tmp")
        os.replace(ck + ".tmp", ck)
        torch.save({"student": student.state_dict(), "opt": opt.state_dict(), "step": done_steps, "optim": args.optim,
                    "args": vars(args)}, st_path + ".tmp")
        os.replace(st_path + ".tmp", st_path)
        prune_checkpoints(args.out, args.keep_last)
        if done_steps == args.steps:
            save_file(sd, os.path.join(args.out, "mtp_final.safetensors"))
        print(f"saved step {done_steps}: {ck}", flush=True)

    def loss_fn(payload):
        ids = payload.to(dev)
        if rng.random() < args.onpolicy_frac:   # teacher continuation of a corpus prefix (on-policy tokens)
            keep = args.seq_len - args.onpolicy_new
            ids = teacher.continue_(ids[:, :keep], args.onpolicy_new)[:, : args.seq_len]
        H = teacher.hidden(ids)
        return unrolled_loss(student, emb_fn, lm_w, ids, H, args)

    # live mode keeps its BF16 autocast and no GradScaler exactly as before (an A100/RTX PRO card has BF16 cores)
    run_training(args, dev, student, opt, data, loss_fn, torch.bfloat16, False, save, step0)


def selftest():
    """Tiny random teacher on CPU: forward shapes, 4-step unrolled loss, backward, a few optimizer steps, GGUF names."""
    torch.manual_seed(0)
    d = Dims(hidden=64, n_head=4, n_kv=2, head_dim=16, rope_dim=8, ffn=96, theta=1e7, eps=1e-6)
    V, B, L = 300, 2, 24
    emb = nn.Embedding(V, d.hidden)
    proj = nn.Linear(d.hidden, d.hidden)
    lm = nn.Linear(d.hidden, V, bias=False)

    def teacher_hidden(ids):   # a "teacher": embedding, causal mean mixing, projection, norm
        x = emb(ids)
        x = torch.cumsum(x, 1) / torch.arange(1, ids.shape[1] + 1)[None, :, None]
        return rms(torch.tanh(proj(x)), torch.ones(d.hidden), d.eps)

    class A:
        unroll, kl_chunk, temperature, cos_weight, step_decay, ckpt = 4, 16, 1.0, 0.5, 0.8, True
        drafter = "mtp"      # the selftest exercises the `mtp` drafter (the `df3` block schedule uses unroll = df3_block)
        df3_block = 4        # only read when drafter == "df3"
    s = MTPHead(d)
    opt = torch.optim.AdamW(s.parameters(), lr=3e-3)
    ids = torch.randint(0, V, (B, L))
    with torch.no_grad():
        H = teacher_hidden(ids)
    losses = []
    for _ in range(30):
        opt.zero_grad()
        loss, logs = unrolled_loss(s, lambda t: emb(t).detach(), lm.weight.detach(), ids, H, A)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    ok = losses[-1] < losses[0] * 0.8 and all(math.isfinite(x) for x in losses)
    names = set(s.gguf_state())
    ok &= names == set(GGUF_NAMES.values())
    full = MTPHead(Dims())
    n_par = sum(p.numel() for p in full.parameters())
    shapes_ok = full.gguf_state()["blk.64.attn_q.weight"].shape == (12288, 5120) and \
        full.gguf_state()["blk.64.nextn.eh_proj.weight"].shape == (5120, 10240)
    ok &= shapes_ok
    print(f"loss {losses[0]:.4f} -> {losses[-1]:.4f}; per-step (kl, cos, top1) at the end: "
          + ", ".join(f"k{k + 1}=({a:.3f},{b:.3f},{c:.2f})" for k, (a, b, c) in enumerate(logs)))
    # external warm-start mapping: a small student re-keyed to the ProCreations mtp.* schema must load back unchanged,
    # and a wrong shape must SystemExit with a message that names the tensor
    dsm = Dims(hidden=64, n_head=4, n_kv=2, head_dim=16, rope_dim=8, ffn=96, theta=1e7, eps=1e-6)
    sm = MTPHead(dsm)
    ref = {n: w.detach().clone() for n, w in sm.named_parameters()}
    external = {al[0]: ref[param] for param, al in EXTERNAL_ALIASES.items()}
    mapped = map_external_state(external, dsm)
    map_ok = all(torch.equal(mapped[GGUF_NAMES[p]], ref[p]) for p in ref)
    bad = dict(external)
    bad["mtp.fc.weight"] = torch.zeros(1)
    try:
        map_external_state(bad, dsm)
        shape_exit = False
    except SystemExit:
        shape_exit = True
    ok &= map_ok and shape_exit
    print(f"external warm-start mapping (ProCreations mtp.* -> blk.64): {'ok' if map_ok else 'WRONG'}; "
          f"wrong shape rejected: {'yes' if shape_exit else 'NO'}")

    # generation pad id: a tokenizer whose pad collides with eos and knows a reserve pad token gets that token,
    # one without any gets None (pad == eos; the generated tail is rewritten to eos instead)
    class _Tok:
        pad_token_id = eos_token_id = 0

        def convert_tokens_to_ids(self, s):
            return {"<|video_pad|>": 7}.get(s)
    pad_id = generation_pad_id(_Tok())
    class _Tok2:
        pad_token_id = eos_token_id = 0

        def convert_tokens_to_ids(self, s):
            return None
    pad_none = generation_pad_id(_Tok2())
    pad_ok = pad_id == 7 and pad_none is None
    ok &= pad_ok
    print(f"generation pad id: reserve={pad_id}, none-fallback={pad_none}: {'ok' if pad_ok else 'WRONG'}")

    # generate() with pad_token_id == eos_token_id must not emit a transformers warning when the batch is
    # explicitly masked (the two warnings the on-policy path used to print on every step)
    gen_ok, gen_note = True, "skipped (transformers not installed)"
    try:
        import logging as _logging
        from transformers import GPT2Config, GPT2LMHeadModel
        gm = GPT2LMHeadModel(GPT2Config(vocab_size=64, n_embd=32, n_layer=1, n_head=2, n_positions=32,
                                        bos_token_id=1, eos_token_id=0))
        gm.eval()
        recs = []

        class _H(_logging.Handler):
            def emit(self, r):
                recs.append(r.getMessage())
        h = _H()
        lg = _logging.getLogger("transformers")
        lg.addHandler(h)
        gids = torch.randint(0, 64, (2, 8))
        with torch.no_grad():
            gm.generate(input_ids=gids, attention_mask=torch.ones_like(gids), max_new_tokens=4, do_sample=False,
                        pad_token_id=0)
        lg.removeHandler(h)
        bad_recs = [m for m in recs if any(k in m.lower() for k in ("attention mask", "right-pad", "padding side", "pad_token"))]
        gen_ok = not bad_recs
        gen_note = "no warning" if not bad_recs else f"{len(bad_recs)} warning(s): {bad_recs[:2]}"
    except ImportError:
        pass
    ok &= gen_ok
    print(f"generate() with pad == eos, explicit mask: {gen_note}")

    # save triggers: step multiples, the final step, the wall-clock trigger, and the disabled combinations
    class _S:
        save_every, save_every_min, steps = 500, 20.0, 2000
    sv = _S()
    trig_ok = (save_due(500, sv, 0.0) and save_due(2000, sv, 0.0) and not save_due(499, sv, 0.0)
               and not save_due(499, sv, 19 * 60.0))
    sv.save_every_min = 0.0
    trig_ok &= not save_due(499, sv, 1e9)   # time trigger disabled
    sv.save_every_min, sv.save_every = 20.0, 0
    trig_ok &= save_due(499, sv, 21 * 60.0) and not save_due(499, sv, 1.0) and save_due(2000, sv, 0.0)
    ok &= trig_ok
    print(f"save triggers (step / final / minutes / disabled combos): {'ok' if trig_ok else 'WRONG'}")

    # ---- from-targets (T4 / local-GPU) end-to-end on synthetic shards: the whole path, no teacher anywhere
    import tempfile
    from types import SimpleNamespace
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import precompute_teacher_targets as pc
    td = tempfile.mkdtemp(prefix="tg_selftest_")
    ids_t = torch.randint(0, V, (3, 32))
    with torch.no_grad():
        H_t = teacher_hidden(ids_t)
    z_t = lm(H_t)                                   # [3, 32, V] toy-teacher logits
    Kt = 8
    ti_t, tv_t, tl_t = pc.topk_targets(z_t.reshape(-1, V), Kt)
    ti_t = ti_t.reshape(3, 32, Kt)
    tv_t = tv_t.reshape(3, 32, Kt)
    tl_t = tl_t.reshape(3, 32)
    hq_t, hs_t = pc.quantize_hidden(H_t.numpy(), "int4g64")
    for si, lo, hi in ((0, 0, 2), (1, 2, 3)):       # shard 0: rows 0-1, shard 1: row 2
        pc.write_shard(pc.shard_path(td, si), ids_t[lo:hi].numpy().astype(np.int32), hq_t[lo:hi], hs_t[lo:hi],
                       ti_t[lo:hi].astype(np.int32), tv_t[lo:hi], tl_t[lo:hi],
                       {"version": 1, "hidden_dtype": "int4g64", "top_k": Kt, "hidden": d.hidden,
                        "seq_len": 32, "samples": hi - lo})
    from safetensors.torch import save_file
    save_file({"embed_tokens.weight": emb.weight.detach().half(), "lm_head.weight": lm.weight.detach().half(),
               "output_norm.weight": torch.ones(d.hidden).half()},
              os.path.join(td, "lm_head_embed_frozen.safetensors"))
    with open(os.path.join(td, "dims.json"), "w") as f:
        json.dump(vars(d), f)

    def run_targets(drafter, steps):
        a_t = SimpleNamespace(
            from_targets=td, lm_head_file=None, top_k_kl=Kt, hidden_dtype="int4g64", epochs=0, stage_dir=None,
            frozen_device="cpu", out=os.path.join(td, f"run_{drafter}"), init="random",
            init_mtp_safetensors=None, resume_from="none", keep_last=1, time_limit_min=0,
            steps=steps, seq_len=24, batch=2, grad_accum=1, lr=5e-3, warmup=1, drafter=drafter,
            df3_block=4, df3_layers=2, unroll=4, temperature=1.0, cos_weight=0.5, step_decay=0.8,
            kl_chunk=16, onpolicy_frac=0.0, onpolicy_new=8, ckpt=True, log_every=1, save_every=0,
            save_every_min=0.0, seed=7, device="cpu", optim="adamw8bit", smoke=False, teacher=None,
            corpus=None, selftest=False, export_init=None, latest=None, attn_impl=None, teacher_f32=False)
        train_targets(a_t)
        log = [json.loads(l) for l in open(os.path.join(a_t.out, "train_log.jsonl"))]
        kls0, kls1 = log[0]["kl"], log[-1]["kl"]
        agree1 = log[-1]["top1_agree"]
        good = (len(log) >= 2
                and all(math.isfinite(x) for x in kls0 + kls1)
                and min(kls1) >= -1e-3                                   # renormalized KL: >= 0 up to fp16 eps
                and kls1[0] < kls0[0]                                    # and it descends
                and min(agree1) >= 0.50                                  # the acceptance proxy must really learn
                and all(k in log[-1] for k in ("sec_per_step", "tok_per_sec", "elapsed_min", "eta_h"))
                and os.path.exists(os.path.join(a_t.out, "mtp_final.safetensors")))
        return good, kls0[0], kls1[0], min(agree1)

    ok_m, k0, k1, a1 = run_targets("mtp", 12)
    ok &= ok_m
    print(f"[{'PASS' if ok_m else 'FAIL'}] --from-targets mtp: kl {k0:.3f} -> {k1:.3f} (>= 0), "
          f"top1_agree {a1:.2f} (>= 0.50), final saved")
    ok_d, k0d, k1d, a1d = run_targets("df3", 16)
    ok &= ok_d
    print(f"[{'PASS' if ok_d else 'FAIL'}] --from-targets df3 (block schedule K=4): kl {k0d:.3f} -> {k1d:.3f} (>= 0), "
          f"top1_agree {a1d:.2f} (>= 0.50)")

    # optimizer fallback chain without bitsandbytes: adamw8bit -> adafactor (or adamw) must never hard-fail
    _p = [torch.zeros(2, requires_grad=True)]
    _o1, n1 = make_optim("adamw8bit", _p, 1e-3, "cpu", allow_bnb=False)
    _o2, n2 = make_optim("sgd", _p, 1e-3)
    fb_ok = n1 in ("adafactor", "adamw") and n2 == "sgd"
    ok &= fb_ok
    print(f"[{'PASS' if fb_ok else 'FAIL'}] optimizer fallback chain: adamw8bit -> {n1}, sgd -> {n2}")

    # precision policy: pre-Ampere (T4) FP16 + GradScaler, Ampere+ BF16, CPU FP32 (cuda branches documented)
    dt_cpu, sc_cpu = train_dtype(torch.device("cpu"))
    prec_ok = dt_cpu == torch.float32 and not sc_cpu
    ok &= prec_ok
    print(f"[{'PASS' if prec_ok else 'FAIL'}] train_dtype on cpu: fp32, no scaler (sm<80 -> fp16+scaler on cuda)")

    print(f"full-size student: {n_par / 1e6:.1f}M parameters, GGUF shapes {'ok' if shapes_ok else 'WRONG'}")
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher")
    ap.add_argument("--corpus")
    ap.add_argument("--out")
    ap.add_argument("--init", default="random", help="random | gguf:PATH (blk.64 of a qwen35 GGUF) | st:PATH")
    ap.add_argument("--init-mtp-safetensors", help="warm start from a blk.64 safetensors (same as --init st:PATH)")
    ap.add_argument("--resume-from", default="auto", help="auto (OUT/train_state.pt if present) | none | PATH (file or dir)")
    ap.add_argument("--keep-last", type=int, default=2, help="keep the newest N mtp_step*.safetensors in --out (0 = all)")
    ap.add_argument("--time-limit-min", type=float, default=0, help="save and stop after this many minutes (0 = no limit)")
    ap.add_argument("--attn-impl", default=None, help="teacher attention implementation: sdpa | flash_attention_2 | eager")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--drafter", default="mtp", choices=["mtp", "df3"],
                    help="mtp: the single-layer nextn head the engine loads as blk.64, trained with --unroll (1..8) "
                         "unrolled self-rollout steps. df3: the block-diffusion schedule, --df3-block (default 16) drafts "
                         "per verification batch; the student is the same head, while a 2-3 layer DFlash head needs a "
                         "loader path the engine does not have yet (docs/DRAFTER_V2_CONTRACT.md section 2)")
    ap.add_argument("--df3-block", type=int, default=16,
                    help="draft block of the df3 schedule (B of a B+1 row verification batch); 16 is the batch the "
                         "engine verifies losslessly since M2")
    ap.add_argument("--df3-layers", type=int, default=2, choices=[2, 3],
                    help="decoder layers of the target DFlash head, recorded in the run config and the export "
                         "metadata; the engine loader knows one layer, so a real 2-3 layer head needs M4-style work")
    ap.add_argument("--unroll", type=int, default=4, help="unrolled self-rollout steps of the `mtp` drafter (1..8)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--cos-weight", type=float, default=0.5)
    ap.add_argument("--step-decay", type=float, default=0.8)
    ap.add_argument("--kl-chunk", type=int, default=256)
    ap.add_argument("--onpolicy-frac", type=float, default=0.25)
    ap.add_argument("--onpolicy-new", type=int, default=256)
    ap.add_argument("--no-ckpt", dest="ckpt", action="store_false")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--save-every-min", type=float, default=20.0,
                    help="also save when this many wall-clock minutes passed since the last save (0 = only --save-every)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda", help="cuda (training) or cpu (--smoke tests)")
    ap.add_argument("--optim", default="adamw", choices=["adamw", "sgd", "adamw8bit", "adafactor"],
                    help="adamw (live default) | sgd (CPU smoke) | adamw8bit / adafactor (from-targets on 16 GB cards; "
                         "both fall back down the chain when the dependency is missing)")
    ap.add_argument("--from-targets", help="train without the teacher: the target package dir written by "
                    "precompute_teacher_targets.py (T4 16 GB / local-GPU mode)")
    ap.add_argument("--lm-head-file", help="frozen safetensors (default: <from-targets>/lm_head_embed_frozen.safetensors)")
    ap.add_argument("--top-k-kl", type=int, default=64, help="sparse KL over this many stored teacher ids per token "
                    "(from-targets mode; skipping the full 248K lm_head backward is what makes a T4 fast)")
    ap.add_argument("--kv63", action="store_true",
                    help="shared-KV training (issue #70 P1-rev): the shards carry kv63_k/v.npy (engine --dump-kv63); "
                    "the draft head attends the teacher's layer-63 K/V through the E-D2b sparse selection instead "
                    "of its own k/v projections")
    ap.add_argument("--hidden-dtype", default="int4g64", choices=["int4g64", "int8"],
                    help="quantization of the stored teacher hidden in the package")
    ap.add_argument("--epochs", type=int, default=1, help="passes over the target package; each epoch reshuffles the "
                    "shards and crops a random block/anchor offset per row (training saves and stops when exhausted)")
    ap.add_argument("--stage-dir", help="copy shards here once before training (local SSD instead of a Drive FUSE)")
    ap.add_argument("--frozen-device", default="auto", choices=["auto", "cpu", "gpu"],
                    help="where the frozen lm_head/embed rows live (auto: GPU while the student is <= 500M params)")
    ap.add_argument("--teacher-f32", action="store_true", help="load the teacher in FP32 (CPU smoke tests)")
    ap.add_argument("--smoke", action="store_true", help="random token ids instead of --corpus (pipeline test)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--export-init", help="write the dequantized --export-init source as safetensors to --out and exit")
    ap.add_argument("--latest", metavar="DIR", help="print the newest checkpoint safetensors of a run directory and exit")
    a = ap.parse_args()
    # The `mtp` drafter trains 1..8 unrolled steps; `df3` runs the block schedule of --df3-block drafts, which is
    # the draft count of the verification batch the engine verifies losslessly (M2 of issue #57).
    if a.drafter == "mtp" and not (1 <= a.unroll <= 8):
        ap.error("--unroll must be 1..8 for the `mtp` drafter (a larger block is the `df3` schedule)")
    if a.drafter == "df3" and not (1 <= a.df3_block <= 32):
        ap.error("--df3-block must be 1..32 (the engine verifies 16 drafts + 1 target row since M2)")
    if a.drafter == "df3":
        print(f"df3 schedule: block of {a.df3_block} drafts, target head of {a.df3_layers} decoder layers "
              "(recorded in the export metadata; the engine loader knows one layer)", flush=True)
    if a.latest:
        p = latest_checkpoint(a.latest)
        if not p:
            print(f"no checkpoint in {a.latest}", file=sys.stderr)
            return 1
        print(p)
        return 0
    if a.selftest:
        return selftest()
    if a.export_init:
        from safetensors.torch import save_file
        if not a.out:
            ap.error("--export-init needs --out FILE.safetensors")
        sd = load_init(a.export_init, Dims())
        s = MTPHead(Dims())
        s.load_gguf_state(sd)
        save_file(s.gguf_state(), a.out)
        print(f"wrote {a.out}: {len(sd)} tensors")
        return 0
    if a.from_targets:
        if "--onpolicy-frac" in sys.argv and a.onpolicy_frac:
            ap.error("--onpolicy-frac needs the live teacher; --from-targets has no teacher to continue from")
        a.onpolicy_frac = 0.0   # the live default (0.25) is meaningless without a teacher
        if a.teacher or a.corpus:
            print("note: --teacher/--corpus are ignored in --from-targets mode")
        if not a.out:
            ap.error("--out is required")
        train_targets(a)
        return 0
    if not (a.teacher and (a.corpus or a.smoke) and a.out):
        ap.error("--teacher, --corpus (or --smoke) and --out are required")
    train(a)
    return 0

if __name__ == "__main__":
    sys.exit(main())
