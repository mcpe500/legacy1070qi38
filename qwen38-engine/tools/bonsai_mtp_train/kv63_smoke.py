"""CPU smoke test of the shared-KV training path (issue #70 P1-rev, --kv63): the MTP head with teacher layer-63
K/V must produce finite outputs, respect the E-D2b causal mask (rows never see teacher positions above their
own), and keep gradients flowing into the q/o projections (the only weights the shared-KV path trains through
attention). Run: CUDA_VISIBLE_DEVICES="" python3 tools/bonsai_mtp_train/kv63_smoke.py"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_bonsai_mtp import Dims, MTPHead  # noqa: E402
from iso3_kv import sparse_kv_mask  # noqa: E402


def main():
    torch.manual_seed(70)
    d = Dims(hidden=256, n_head=8, n_kv=2, head_dim=64, rope_dim=32, ffn=512)
    B, L = 2, 128
    head = MTPHead(d)
    offs = torch.tensor([40, 400])                       # two crops at different absolute offsets
    pos_rel = torch.arange(L).expand(B, L)
    pos_abs = pos_rel + offs[:, None]
    emb = torch.randn(B, L, d.hidden)
    h = torch.randn(B, L, d.hidden)
    kv63 = (torch.randn(B, L, d.n_kv, d.head_dim), torch.randn(B, L, d.n_kv, d.head_dim))

    f = head(emb, h, pos_rel, kv63=kv63, pos_abs=pos_abs)
    assert f.shape == (B, L, d.hidden) and torch.isfinite(f).all(), "kv63 forward shape/finiteness"
    f_dense = head(emb, h, pos_rel)
    assert f_dense.shape == f.shape and torch.isfinite(f_dense).all(), "dense forward still works"

    # causal: perturbing teacher K/V at positions ABOVE the last row's position must not change the output
    kv63b = (kv63[0].clone(), kv63[1].clone())
    kv63b[0][:, L // 2:] *= -7.0
    kv63b[1][:, L // 2:] *= -7.0
    f2 = head(emb, h, pos_rel, kv63=kv63b, pos_abs=pos_abs)
    # rows 0..L//2-1 have pos_abs < L//2 (offs 40: rows up to 87 see L//2=64... use the strict subset): compare
    # the FIRST rows whose absolute position is below L//2 for every batch row: pos_abs - offs < L//2
    safe = (pos_abs - offs[:, None]) < L // 2
    d1 = (f - f2).abs().amax(dim=-1)
    assert (d1[safe] < 1e-4).all(), f"causality leak: future teacher K/V changed early rows (max {d1[safe].max()})"

    # gradients reach the q/o projections (shared-KV trains through them) but not through v_proj/k_proj at step
    f.sum().backward()
    assert head.q_proj.grad is not None and head.q_proj.grad.abs().sum() > 0, "no gradient into q_proj"
    assert head.o_proj.grad is not None and head.o_proj.grad.abs().sum() > 0, "no gradient into o_proj"

    # mask agreement: the selection keeps sink + window + top-k and clamps causally
    sc = torch.randn(1, d.n_kv, 8, 200)
    pa = torch.tensor([[150]]).expand(1, 8) * torch.ones(1, 8, dtype=torch.long)
    m = sparse_kv_mask(sc, pa)
    assert m.dtype == torch.bool and m.shape == (1, 8, 2, 200)   # [B, R, G, J]
    j = torch.arange(200)
    assert bool(m[0, 0, 0, 0]) and not bool(m[0, 0, 0, 151]), "sink kept, causal clamp"
    print(f"[PASS] kv63 smoke: forward finite, causality holds, grads reach q/o, mask {tuple(m.shape)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
