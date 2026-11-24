#!/usr/bin/env python3
"""One-shot teacher burst for the T4 / local-GPU training path (PRECOMPUTE_T4_DESIGN.md).

Runs the frozen BF16 teacher forward ONLY (torch.inference_mode, no student, no backward) over a curated subset of
the ChatML corpus and stores, per token position, everything the draft-head trainer needs without the teacher:

    ids        int32   [S, L]        the token ids
    hq/hs      INT4 group-64 (or INT8)  the teacher output-normed hidden H[i] (the step-1 `h` input and the
                                       cosine target of every unrolled step)
    tk_ids     int32   [S, L, K]    top-K teacher logit ids (teacher logits = lm_head(H[i]), as in live mode)
    tk_vals    fp16    [S, L, K]    the matching logit values
    tk_lse     fp16    [S, L]       logsumexp of the full teacher row, so p_k = exp(v_k - lse) is exact inside K

The package is consumed by `train_bonsai_mtp.py --from-targets`, which trains the blk.64 head (mtp or df3 schedule)
with a sparse top-K KL + cosine loss on a 16 GB GPU, with the 27B teacher absent. Budgets (hidden 5120, K=64):

    8K x 1024 tok, INT4-g64:  22.3 GB hidden + 3.1 GB top-64 + 0.03 GB ids  =  ~25.4 GB on Drive
    frozen lm_head+embed FP16 (written once):                                ~4.7 GiB

Usage:
  python3 precompute_teacher_targets.py --teacher /content/models/bonsai-hf-bf16 --corpus /content/drive/MyDrive/bonsai_mtp/corpus \
      --out /content/drive/MyDrive/bonsai_mtp/targets --samples 8000 --seq-len 1024 --max-gb 100
  python3 precompute_teacher_targets.py --selftest   # CPU: quantizer, shard format, tiny end-to-end
"""
import argparse
import glob
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import torch

GROUP = 64   # hidden quantization group; 5120 / 64 = 80 groups per token
INT4_SCALE_FLOOR = 1e-4   # keep fp16 scales in normal range (a saturated tiny group is ~0 anyway)


# ---------------------------------------------------------------------------------------------------- quantizer
def _pack_int4(q):
    """int8 [..., H] -> packed uint8 [..., H//2] (two nibbles per byte)."""
    u = (q.astype(np.uint8) & 0x0F).reshape(*q.shape[:-1], q.shape[-1] // 2, 2)
    return u[..., 0] | (u[..., 1] << 4)


def _unpack_int4(p):
    """Inverse of _pack_int4: uint8 [..., H//2] -> int8 [..., H]."""
    lo = (p & 0x0F).astype(np.int8)
    lo[lo >= 8] -= 16
    hi = (p >> 4).astype(np.int8)
    hi[hi >= 8] -= 16
    return np.stack([lo, hi], axis=-1).reshape(*p.shape[:-1], -1)


def quantize_hidden(x, dtype):
    """x: np.float32 [..., H] -> (hq, hs). int4g64: packed uint8 [..., H//2] + fp16 scales [..., H//GROUP];
    int8: int8 [..., H] + fp16 scales [..., H//GROUP]. Symmetric, per group of GROUP."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    H = x.shape[-1]
    if H % GROUP:
        raise ValueError(f"hidden {H} not divisible by group {GROUP}")
    g = x.reshape(*x.shape[:-1], H // GROUP, GROUP)
    amax = np.abs(g).max(-1)
    qmax = 7 if dtype == "int4g64" else 127
    scale = np.maximum(amax / qmax, INT4_SCALE_FLOOR).astype(np.float32)
    q = np.clip(np.rint(g / scale[..., None]), -qmax - (1 if dtype == "int4g64" else 0), qmax)
    hs = scale.astype(np.float16)
    if dtype == "int4g64":
        hq = _pack_int4(q.astype(np.int8).reshape(*x.shape[:-1], H))
        return np.ascontiguousarray(hq), np.ascontiguousarray(hs)
    if dtype == "int8":
        return np.ascontiguousarray(q.astype(np.int8).reshape(*x.shape[:-1], H)), np.ascontiguousarray(hs)
    raise SystemExit(f"unknown --hidden-dtype {dtype}")


def dequantize_hidden(hq, hs, dtype):
    """Inverse of quantize_hidden. Returns np.float32 [..., H]."""
    H = hs.shape[-1] * GROUP
    if dtype == "int4g64":
        q = _unpack_int4(np.asarray(hq, dtype=np.uint8))
    elif dtype == "int8":
        q = np.asarray(hq, dtype=np.int8)
    else:
        raise SystemExit(f"unknown --hidden-dtype {dtype}")
    return (q.astype(np.float32).reshape(*hs.shape, GROUP) * hs[..., None].astype(np.float32)).reshape(
        *hs.shape[:-1], H)


def topk_targets(logits_t, K):
    """logits_t: torch [L, V] (any float dtype, grad or not) -> (ids int32 [L,K], vals fp16 [L,K], lse fp16 [L])."""
    z = logits_t.detach().float()
    lse = torch.logsumexp(z, dim=-1).half()
    tv, ti = z.topk(min(K, z.shape[-1]), dim=-1)
    return ti.to(torch.int32).numpy(), tv.half().numpy(), lse.numpy()


# ---------------------------------------------------------------------------------------------------- shard io
def shard_path(out, idx):
    return os.path.join(out, f"target-shard-{idx:05d}.npz")


def write_shard(path, ids, hq, hs, tk_ids, tk_vals, tk_lse, meta):
    """Atomic shard write: all arrays as one .npz plus a sidecar .json meta (the npz stays self-describing enough)."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, ids=ids, hq=hq, hs=hs, tk_ids=tk_ids, tk_vals=tk_vals, tk_lse=tk_lse)
    os.replace(tmp, path)
    with open(path + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    return os.path.getsize(path)


def load_shard(path):
    with open(path, "rb") as f:
        z = np.load(f)
        d = {k: z[k] for k in ("ids", "hq", "hs", "tk_ids", "tk_vals", "tk_lse")}
    meta = {}
    mp = path + ".json"
    if os.path.exists(mp):
        meta = json.load(open(mp))
    return d, meta


def stage_shard(path, stage_dir):
    """Copy the shard to local storage on first use (Drive FUSE is slow for training reads); returns the local path."""
    if not stage_dir:
        return path
    os.makedirs(stage_dir, exist_ok=True)
    local = os.path.join(stage_dir, os.path.basename(path))
    if not os.path.exists(local + ".ok"):
        tmp = local + ".tmp"
        import shutil
        shutil.copyfile(path, tmp)
        os.replace(tmp, local)
        shutil.copyfile(path + ".json", local + ".json")
        open(local + ".ok", "w").close()
    return local


def hidden_dtype_of(meta, path):
    dt = (meta or {}).get("hidden_dtype", "int4g64")
    if dt not in ("int4g64", "int8"):
        raise SystemExit(f"shard {path}: unknown hidden_dtype {dt!r}")
    return dt


# ---------------------------------------------------------------------------------------------------- teacher io
class NormProbe:
    """Finds the teacher's final norm weight once (same rule as the trainer's hook)."""
    def __init__(self, model):
        self.w = None
        for name, mod in model.named_modules():
            if name.endswith("model.norm") or name.endswith("language_model.norm"):
                self.w = mod.weight
        if self.w is None:
            raise SystemExit("teacher: no final norm module (model.norm) found")


def save_frozen(out, teacher):
    """The 4.7 GiB FP16 file the T4 training loop needs instead of the 50.1 GiB teacher."""
    from safetensors.torch import save_file
    nw = NormProbe(teacher.model).w
    sd = {"embed_tokens.weight": teacher.embed.weight.detach().half().cpu().contiguous(),
          "lm_head.weight": teacher.lm_head.weight.detach().half().cpu().contiguous(),
          "output_norm.weight": nw.detach().half().cpu().contiguous()}
    p = os.path.join(out, "lm_head_embed_frozen.safetensors")
    tmp = p + ".tmp"
    save_file(sd, tmp)
    os.replace(tmp, p)
    return p


def save_dims(out, d):
    p = os.path.join(out, "dims.json")
    with open(p + ".tmp", "w") as f:
        json.dump(vars(d), f, indent=1)
    os.replace(p + ".tmp", p)
    return p


def corpus_samples(corpus, n, seq_len, tok, seed):
    """Seeded pick of up to n corpus texts tokenized to seq_len ids (ChatML jsonl shards of prepare_corpus.py)."""
    files = sorted(glob.glob(os.path.join(corpus, "shard-*.jsonl")))
    if not files:
        raise SystemExit(f"no shard-*.jsonl in {corpus} (run prepare_corpus.py first)")
    rng = random.Random(seed)
    rng.shuffle(files)
    lines = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            lines.extend(l for l in f if len(l) > 200)   # skip blank / stub rows cheaply
    rng.shuffle(lines)
    plan_sha = hashlib.sha1("".join(lines).encode("utf-8", "ignore")).hexdigest()
    out, i, seen = [], 0, 0
    while len(out) < n and i < len(lines):
        try:
            ids = tok(json.loads(lines[i])["text"], add_special_tokens=False)["input_ids"]
        except Exception:
            ids = []
        i += 1
        seen += 1
        if len(ids) >= seq_len:
            out.append(ids[:seq_len])
    if not out:
        raise SystemExit(f"no corpus sample reached {seq_len} tokens (checked {seen} rows)")
    print(f"corpus: {len(out)} of {n} planned samples usable ({seen} rows scanned)", flush=True)
    return out, plan_sha


# ---------------------------------------------------------------------------------------------------- burst
def precompute(a):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    from train_bonsai_mtp import Teacher   # reuse the frozen-HF wrapper (same taps as the live trainer)
    dev = torch.device(a.device)
    os.makedirs(a.out, exist_ok=True)
    man_p = os.path.join(a.out, "manifest.json")
    teacher = Teacher(a.teacher, dev, dtype=torch.bfloat16, attn_impl=a.attn_impl)
    lm_w = teacher.lm_head.weight
    print(f"teacher dims {vars(teacher.dims)}", flush=True)
    save_frozen(a.out, teacher)
    save_dims(a.out, teacher.dims)
    samples, plan_sha = corpus_samples(a.corpus, a.samples, a.seq_len, teacher.tok, a.seed)
    done_shards, bytes_done = 0, 0
    if os.path.exists(man_p) and not a.overwrite:
        man = json.load(open(man_p))
        if man.get("plan_sha") == plan_sha and man.get("config") == vars(a):
            done_shards, bytes_done = len(man["shards"]), man["total_bytes"]
            print(f"resume: {done_shards} shards / {bytes_done / 1e9:.1f} GB already written", flush=True)
        else:
            print("manifest config or corpus plan changed: starting the package over", flush=True)
    per = a.shard_samples
    t0 = time.time()
    n_tok_done = 0
    for s0 in range(done_shards * per, len(samples), per):
        rows = samples[s0: s0 + per]
        B, L, H = len(rows), a.seq_len, teacher.dims.hidden
        ids_full = torch.tensor(rows, dtype=torch.long, device=dev)
        hq_all = np.empty((B, L, H // 2 if a.hidden_dtype == "int4g64" else H),
                          dtype=np.uint8 if a.hidden_dtype == "int4g64" else np.int8)
        hs_all = np.empty((B, L, H // GROUP), dtype=np.float16)
        tk_ids = np.empty((B, L, a.top_k), dtype=np.int32)
        tk_vals = np.empty((B, L, a.top_k), dtype=np.float16)
        tk_lse = np.empty((B, L), dtype=np.float16)
        for c0 in range(0, B, a.batch):                  # teacher forward in --batch chunks (fits 24-40 GB cards too)
            ids = ids_full[c0: c0 + a.batch]
            with torch.inference_mode():
                Hh = teacher.hidden(ids)                 # [b, L, Hd] output-normed, BF16
            for r in range(ids.shape[0]):                # per row: the 248K-wide teacher logits stay small
                with torch.inference_mode():
                    zt = torch.nn.functional.linear(Hh[r], lm_w).float()
                ti, tv, lse = topk_targets(zt, a.top_k)
                tk_ids[c0 + r], tk_vals[c0 + r], tk_lse[c0 + r] = ti, tv, lse
                hq_all[c0 + r], hs_all[c0 + r] = quantize_hidden(Hh[r].float().cpu().numpy(), a.hidden_dtype)
        meta = {"version": 1, "hidden_dtype": a.hidden_dtype, "top_k": a.top_k, "hidden": H, "seq_len": L,
                "samples": B, "group": GROUP, "index": s0 // per}
        nb = write_shard(shard_path(a.out, s0 // per), ids.cpu().numpy(), hq_all, hs_all,
                         tk_ids, tk_vals, tk_lse, meta)
        bytes_done += nb
        n_tok_done += B * L
        dt = time.time() - t0
        man = {"version": 1, "config": vars(a), "plan_sha": plan_sha,
               "shards": list(range(0, s0 // per + 1)), "total_bytes": bytes_done,
               "total_samples": s0 + B, "total_tokens": n_tok_done, "hidden_dtype": a.hidden_dtype,
               "top_k": a.top_k}
        with open(man_p + ".tmp", "w") as f:
            json.dump(man, f, indent=1)
        os.replace(man_p + ".tmp", man_p)
        eta = (len(samples) - (s0 + B)) / max(1.0, B * L / dt) if dt > 0 else 0.0
        print(f"shard {s0 // per}: {B} samples, {nb / 1e6:.0f} MB, package {bytes_done / 1e9:.1f} GB, "
              f"{n_tok_done / dt:.0f} tok/s, eta {eta / 60:.1f} min", flush=True)
        if a.max_gb and bytes_done > a.max_gb * 1e9:
            print(f"--max-gb {a.max_gb} reached ({bytes_done / 1e9:.1f} GB): stopping with a consistent manifest",
                  flush=True)
            break
    print(f"done: {len(json.load(open(man_p))['shards'])} shards, {bytes_done / 1e9:.1f} GB, "
          f"{n_tok_done / 1e6:.1f}M tokens", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------------- selftest
def selftest():
    import tempfile
    ok = True
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # 1. quantizer round-trip: random hidden rows, int4g64 must keep cosine >= 0.99, int8 >= 0.999
    # (a synthetic gaussian row is the worst case for max-scaled symmetric int4; real teacher hidden is smooth and
    # quantizes above this — the Phase A burst prints the real cosine in the package manifest)
    x = rng.standard_normal((64, 5120)).astype(np.float32) * 2.0
    for dt, floor in (("int4g64", 0.99), ("int8", 0.999)):
        hq, hs = quantize_hidden(x, dt)
        x2 = dequantize_hidden(hq, hs, dt)
        cos = float(np.dot(x.ravel(), x2.ravel()) / (np.linalg.norm(x) * np.linalg.norm(x2)))
        good = cos >= floor and x2.shape == x.shape and np.isfinite(x2).all()
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] quantize {dt}: cosine {cos:.5f} (floor {floor}), "
              f"{x.nbytes / 1e6:.1f} MB -> {(hq.nbytes + hs.nbytes) / 1e6:.1f} MB")
    # int4 exact value round-trip for the full code range
    q = np.arange(-8, 8, dtype=np.int8).reshape(1, 16)
    back = _unpack_int4(_pack_int4(q))
    good = np.array_equal(back.astype(np.int8), q)
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] int4 code range -8..7 round-trip")

    # 2. topk/lse math: p_k sums to the real tail, ids are the argmax order
    z = torch.randn(7, 500)
    ti, tv, lse = topk_targets(z, 64)
    ref_lse = torch.logsumexp(z, -1)
    good = (np.allclose(lse, ref_lse.numpy(), atol=2e-3) and np.array_equal(ti[:, 0], z.argmax(-1).numpy())
            and (np.exp(tv - lse[:, None]).sum(-1) <= 1.0 + 2e-3).all())   # fp16 rounding of vals/lse
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] topk ids/lse: p_k tail mass <= 1 (+fp16 eps), argmax first, lse matches")

    # 3. shard write/read/quantize/byte accounting + a tiny end-to-end burst with a toy teacher
    tmp = tempfile.mkdtemp(prefix="pc_selftest_")
    H = 5120
    S, L, K = 3, 32, 16
    ids = rng.integers(0, 1000, size=(S, L)).astype(np.int32)
    Hf = rng.standard_normal((S, L, H)).astype(np.float32)
    hq, hs = quantize_hidden(Hf, "int4g64")
    zt = torch.tensor(rng.standard_normal((S * L, 400)).astype(np.float32))
    ti, tv, lse = topk_targets(zt, K)
    ti, tv, lse = ti.reshape(S, L, K), tv.reshape(S, L, K), lse.reshape(S, L)
    meta = {"version": 1, "hidden_dtype": "int4g64", "top_k": K, "hidden": H, "seq_len": L, "samples": S}
    p = shard_path(tmp, 0)
    nb = write_shard(p, ids, hq, hs, ti, tv, lse, meta)
    d, m = load_shard(stage_shard(p, os.path.join(tmp, "stage")))
    H2 = dequantize_hidden(d["hq"], d["hs"], hidden_dtype_of(m, p))
    cos = float(np.dot(Hf.ravel(), H2.ravel()) / (np.linalg.norm(Hf) * np.linalg.norm(H2)))
    bpt = nb / (S * L)
    good = (d["ids"].shape == (S, L) and d["tk_ids"].shape == (S, L, K) and cos >= 0.99
            and abs(bpt - (4 + 5120 * 0.53125 + K * 6 + 2)) < 60)
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] shard round-trip: cosine {cos:.4f}, {bpt:.0f} bytes/token "
          f"(formula {4 + 5120 * 0.53125 + K * 6 + 2:.0f})")
    # manifest accounting: two shards of S samples each
    man_p = os.path.join(tmp, "manifest.json")
    total = 0
    for i in range(2):
        total += nb
        man = {"shards": list(range(i + 1)), "total_bytes": total, "total_samples": (i + 1) * S,
               "total_tokens": (i + 1) * S * L}
        with open(man_p + ".tmp", "w") as f:
            json.dump(man, f)
        os.replace(man_p + ".tmp", man_p)
    man = json.load(open(man_p))
    good = man["total_bytes"] == 2 * nb and man["total_samples"] == 2 * S
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] manifest accounting after 2 shards")

    # 4. dims.json round-trip through the trainer's Dims (what --from-targets reads)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_bonsai_mtp import Dims
    dd = Dims(hidden=5120, n_head=24, n_kv=4, head_dim=256, rope_dim=64, ffn=17408, theta=1e7, eps=1e-6)
    save_dims(tmp, dd)
    dd2 = Dims(**json.load(open(os.path.join(tmp, "dims.json"))))
    good = vars(dd) == vars(dd2)
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] dims.json round-trip")
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", help="the BF16 Hugging Face teacher directory (as the live pipeline uses)")
    ap.add_argument("--corpus", help="corpus dir with shard-*.jsonl (prepare_corpus.py output)")
    ap.add_argument("--out", help="target package dir (shards + manifest.json + dims.json + frozen lm_head/embed)")
    ap.add_argument("--samples", type=int, default=8000, help="corpus samples to run through the teacher")
    ap.add_argument("--seq-len", type=int, default=1024, help="tokens per stored sample (random block/anchor crops per epoch later)")
    ap.add_argument("--batch", type=int, default=8, help="teacher forward batch (rows per forward)")
    ap.add_argument("--shard-samples", type=int, default=128, help="samples per shard file")
    ap.add_argument("--top-k", type=int, default=64, help="sparse teacher logits per token")
    ap.add_argument("--hidden-dtype", default="int4g64", choices=["int4g64", "int8"])
    ap.add_argument("--max-gb", type=float, default=0, help="stop writing shards past this package size (0 = no cap)")
    ap.add_argument("--overwrite", action="store_true", help="ignore the manifest and rebuild the package")
    ap.add_argument("--attn-impl", default=None, help="teacher attention implementation: sdpa | flash_attention_2 | eager")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda", help="cuda (burst) or cpu (tests)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.teacher and a.corpus and a.out):
        ap.error("--teacher, --corpus and --out are required")
    return precompute(a)


if __name__ == "__main__":
    sys.exit(main())