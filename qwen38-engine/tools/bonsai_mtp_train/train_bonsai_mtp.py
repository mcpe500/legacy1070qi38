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
  python3 train_bonsai_mtp.py --selftest        # tiny random teacher on CPU: shapes, losses, backward, export names
  python3 train_bonsai_mtp.py --export-init gguf:QWEN.gguf --out x.safetensors   # dequantized init only (warm start file)
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

    def forward(self, emb, h, pos):
        """emb, h: [B, L, H] (token embedding of x_p, hidden of p-1), pos: [B, L] absolute positions. Returns f [B, L, H]."""
        d = self.d
        B, L, _ = emb.shape
        if not torch.is_autocast_enabled():   # CPU smoke run: BF16 teacher tensors into the FP32 student
            emb, h = emb.to(self.eh_proj.dtype), h.to(self.eh_proj.dtype)
        z = F.linear(torch.cat([rms(emb, self.enorm, d.eps), rms(h, self.hnorm, d.eps)], -1), self.eh_proj)
        n = rms(z, self.attn_norm, d.eps)
        qg = F.linear(n, self.q_proj).view(B, L, d.n_head, 2, d.head_dim)
        q, gate = qg[:, :, :, 0], qg[:, :, :, 1]
        k = F.linear(n, self.k_proj).view(B, L, d.n_kv, d.head_dim)
        v = F.linear(n, self.v_proj).view(B, L, d.n_kv, d.head_dim)
        q = self.rope(rms(q, self.q_norm, d.eps).transpose(1, 2), pos)
        k = self.rope(rms(k, self.k_norm, d.eps).transpose(1, 2), pos)
        v = v.transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)   # scale 1/sqrt(256) as the engine
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


def load_init(spec, d):
    """gguf:PATH (dequantize blk.64 of a qwen35 GGUF with llama.cpp gguf-py), st:PATH (safetensors), or random."""
    if spec is None or spec == "random":
        return None
    kind, path = spec.split(":", 1)
    if kind == "st":
        from safetensors.torch import load_file
        return load_file(path)
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


# ------------------------------------------------------------------------------------------------------------ teacher
class Teacher:
    """Frozen Hugging Face causal LM. hidden(): output-normed last hidden state (hook on the final norm), lm_head shared."""

    def __init__(self, path, device, dtype=torch.bfloat16, attn_impl=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
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
        return self.model.generate(input_ids=ids, max_new_tokens=n_new, do_sample=True, temperature=temp, top_p=top_p,
                                   pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)


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


def train(args):
    dev = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out, exist_ok=True)
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

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        p = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))

    vocab = teacher.lm_head.weight.shape[0]
    data = random_batches(vocab, args.seq_len, args.batch, args.seed + step0) if args.smoke else \
        packed_batches(args.corpus, teacher.tok, args.seq_len, args.batch, args.seed + step0)
    rng = random.Random(args.seed + 7 * step0)
    lm_w = teacher.lm_head.weight
    emb_fn = lambda t: teacher.embed(t)   # noqa: E731  (frozen, BF16)
    t0 = time.time()
    log_f = open(os.path.join(args.out, "train_log.jsonl"), "a")

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

    for step in range(step0, args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        acc_logs = []
        for _ in range(args.grad_accum):
            ids = next(data).to(dev)
            if rng.random() < args.onpolicy_frac:   # teacher continuation of a corpus prefix (on-policy tokens)
                keep = args.seq_len - args.onpolicy_new
                ids = teacher.continue_(ids[:, :keep], args.onpolicy_new)[:, : args.seq_len]
            H = teacher.hidden(ids)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                loss, logs = unrolled_loss(student, emb_fn, lm_w, ids, H, args)
            (loss / args.grad_accum).backward()
            acc_logs.append(logs)
        gn = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0:
            K = unroll_K(args)
            avg = [[sum(l[k][j] for l in acc_logs) / len(acc_logs) for j in range(3)] for k in range(K)]
            rec = {"step": step, "lr": lr_at(step), "grad_norm": float(gn), "s": time.time() - t0,
                   "kl": [a[0] for a in avg], "cos_loss": [a[1] for a in avg], "top1_agree": [a[2] for a in avg]}
            print(json.dumps(rec), flush=True)
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            save(step + 1)
        elif args.time_limit_min and (time.time() - t0) / 60.0 >= args.time_limit_min:
            save(step + 1)   # stop cleanly before a session limit; the next run resumes from here
            print(f"time limit of {args.time_limit_min} min reached after step {step + 1}, stopping", flush=True)
            break
        if args.time_limit_min and (time.time() - t0) / 60.0 >= args.time_limit_min:
            print(f"time limit of {args.time_limit_min} min reached after step {step + 1}, stopping", flush=True)
            break
    log_f.close()


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
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda", help="cuda (training) or cpu (--smoke tests)")
    ap.add_argument("--optim", default="adamw", choices=["adamw", "sgd"])
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
    if not (a.teacher and (a.corpus or a.smoke) and a.out):
        ap.error("--teacher, --corpus (or --smoke) and --out are required")
    train(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
