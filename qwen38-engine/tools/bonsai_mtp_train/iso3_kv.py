"""iso3 K/V shard loader + dequant + DFly2 sparse selection (issue #70 P1-rev / E-D2b).

The engine dump (`--dump-kv63`) stores the K/V of the teacher's last attention layer (63) EXACTLY as the
engine caches them: 400 bytes per position per tensor, 8 slots of 50 B (slot = kv_group * 2 + head_half).
Each slot holds 128 dims of one head half: [0:2] f16 norm, [2:34] 16 little-endian u16 code words
(one nibble = two dims: 2+2 code bits | 2 sign bits << 4), [34:36] sign bits (2 bits per code word),
rest unused. The value of a slot is  nm * M_g . c  with M_g the iso3 quaternion merge of the decode
kernel's o = M o' epilogue, which is exactly iso3_ref.cpp rotate(g, inv=True) on each 4-dim sub-block
(quaternion index gi0 + k, gi0 = (2*lane) & 31, lane = dim block / 8 within the half).

Everything here mirrors src/cuda/ops.cu and src/common/{iso3_ref.cpp,tables.inc}; the selftest checks the
round-trip construction and the quaternion orthonormality so a mismatch with the engine shows up on CPU.

Selection (training-time twin of the E-D2b selector): per KV group and query row, keep the top-32 blocks
of 64 positions by block-max score, plus the blocks covering the last 128 positions (window) and block 0
(sink); ties broken toward the lower block via a tiny index bias, matching the engine's deterministic rank
rule up to float noise.
"""
import glob
import os
import re

import numpy as np
import torch

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "common", "tables.inc")
CB_MAX = None  # filled by _tables()


def _tables():
    txt = open(_SRC, encoding="utf-8").read()
    out = {}
    global CB_MAX
    for name in ("iso3_codebook", "iso3_quat_w", "iso3_quat_x", "iso3_quat_y", "iso3_quat_z"):
        m = re.search(r"QT_BEGIN\(float, %s, (\d+)\)(.*?)QT_END\(\)" % name, txt, re.S)
        vals = [float(x) for x in re.findall(r"(-?\d+\.\d+)f", m.group(2))]
        assert len(vals) == int(m.group(1)), name
        out[name] = np.array(vals, dtype=np.float32)
    CB_MAX = float(np.abs(out["iso3_codebook"]).max())
    return out


_T = _tables()
CB = _T["iso3_codebook"]
QUAT = np.stack([_T["iso3_quat_w"], _T["iso3_quat_x"], _T["iso3_quat_y"], _T["iso3_quat_z"]], axis=0)  # [4, 32]
K_UNIT = 400        # iso3 bytes per position per tensor
SLOT_DIMS = 128     # dims per slot
N_SLOTS = 8         # 4 KV groups x 2 head halves
NH, NKV, HD = 24, 4, 256
QG = NH // NKV      # 6 query heads per KV group
BLOCK = 64
TOP_BLOCKS = 32
WINDOW = 128

# sign LUT: 6-bit entry (4 code bits | 2 sign bits << 4) -> the two codebook values of that nibble
_LUT = np.zeros((64, 2), np.float32)
for e in range(64):
    _LUT[e, 0] = CB[(e & 3) | (((e >> 4) & 1) << 2)]
    _LUT[e, 1] = CB[((e >> 2) & 3) | (((e >> 5) & 1) << 2)]

# 128x128 merge matrix: out = P . c (rotate(inv=True) per 4-dim block; see iso3_ref.cpp rotate)
_P = np.zeros((SLOT_DIMS, SLOT_DIMS), np.float32)
for lane in range(16):
    gi0 = (2 * lane) & 31
    base = 8 * lane
    for k in range(2):
        g = gi0 + k
        w, x, y, z = QUAT[0][g], QUAT[1][g], QUAT[2][g], QUAT[3][g]
        rows = ((w, x, y, z), (-x, w, z, -y), (-y, -z, w, x), (-z, y, -x, w))  # o = M o' epilogue of the decode kernel
        for r in range(4):
            for c in range(4):
                _P[base + 4 * k + r, base + 4 * k + c] = rows[r][c]


def dequant_slots(blk):
    """blk: [..., 50] uint8 (the bytes of one slot) -> [..., 128] float32 values nm * M . c. Fully vectorized."""
    blk = np.ascontiguousarray(blk)
    shape = blk.shape[:-1]
    b = blk.reshape(-1, 50)
    nm = b[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1)         # f16 norm per slot
    q16 = b[:, 2:34].copy().view("<u2").astype(np.uint32)                          # [n, 16] code words, lane-major
    s8 = b[:, 34:50]                                                               # [n, 16] sign bytes, lane-major
    e = np.zeros((q16.shape[0], 16, 4), np.uint32)
    for p in range(4):   # nibble p of each code word, sign bits 2p..2p+1 of the lane's sign byte
        e[:, :, p] = ((q16 >> (4 * p)) & 15) | (((s8 >> (2 * p)) & 3) << 4)
    c = _LUT[e]                                                                    # [n, 16, 4, 2]
    c = c.reshape(-1, SLOT_DIMS)                                                   # dims 8*lane + 2p (+1) in order
    out = c @ _P.T
    out *= nm[:, None]
    return out.reshape(*shape, SLOT_DIMS)


def dequant_position(kb, vb):
    """kb, vb: [..., 400] uint8 -> K, V: [..., 8, 128] = [kv_group*2 + half, dim-in-half]."""
    k = dequant_slots(kb.reshape(kb.shape[:-1] + (N_SLOTS, 50))).reshape(kb.shape[:-1] + (N_SLOTS, SLOT_DIMS))
    v = dequant_slots(vb.reshape(vb.shape[:-1] + (N_SLOTS, 50))).reshape(vb.shape[:-1] + (N_SLOTS, SLOT_DIMS))
    return k, v


def dequant_shard(shard_dir):
    """Load kv63_k.npy / kv63_v.npy of one engine shard dir -> (K, V) float32 [samples, seq, 4, 256], or None.
    The slots (kv_group * 2 + head_half, 128 dims each) are folded into full 256-dim heads."""
    fk = os.path.join(shard_dir, "kv63_k.npy")
    fv = os.path.join(shard_dir, "kv63_v.npy")
    if not (os.path.exists(fk) and os.path.exists(fv)):
        return None
    kb = np.load(fk, mmap_mode="r")
    vb = np.load(fv, mmap_mode="r")
    assert kb.shape == vb.shape and kb.shape[-1] == K_UNIT, "kv63 shard shape mismatch"
    S, Ls = kb.shape[0], kb.shape[1]
    Kd, Vd = dequant_position(np.asarray(kb).reshape(-1, K_UNIT), np.asarray(vb).reshape(-1, K_UNIT))
    fold = lambda t: np.ascontiguousarray(t.reshape(S, Ls, NKV, 2, SLOT_DIMS)).reshape(S, Ls, NKV, HD)
    return fold(Kd).astype(np.float32), fold(Vd).astype(np.float32)


# ---- training-time selection (the torch twin of ops.cu dfly2_select_kernel) ----

def select_blocks(scores_blockmax, pos_abs, window=WINDOW, top_k=TOP_BLOCKS, block=BLOCK):
    """scores_blockmax: [B, G, R, NB] block-max scores (R rows, NB blocks); pos_abs: [B, R] absolute row positions.
    Returns a bool keep mask [B, G, R, NB]: top-k blocks (ties -> lower block, via an index bias negligible against
    fp16 score noise) union the window (last `window` positions of each row) union the sink block 0, and blocks
    fully beyond pos_abs are never kept."""
    B, G, R, NB = scores_blockmax.shape
    bias = scores_blockmax - torch.arange(NB, device=scores_blockmax.device).float() * 1e-4
    idx = bias.topk(min(top_k, NB), dim=-1).indices                     # [B, G, R, K]
    keep = torch.zeros_like(scores_blockmax, dtype=torch.bool)
    keep.scatter_(-1, idx, True)
    blk_of_row = pos_abs // block                                       # last live block per row
    pa4 = pos_abs[:, None, :, None]                                     # [B, 1, R, 1] broadcast over the groups
    blk4 = blk_of_row[:, None, :, None]
    ar = torch.arange(NB, device=scores_blockmax.device).view(1, 1, 1, NB)
    keep |= ar == 0                                                     # sink
    keep |= (ar * block > pa4 - window) & (ar <= blk4)                  # window
    keep &= ar <= blk4                                                  # causal clamp
    return keep


def sparse_kv_mask(scores, pos_abs, block=BLOCK, window=WINDOW, top_k=TOP_BLOCKS):
    """scores: [B, G, R, J] full q.K scores of the rows against the teacher positions j = 0..J-1.
    Returns a bool attention mask [B, R, G, J] (True = attend), the training-time twin of the E-D2b list."""
    B, G, R, J = scores.shape
    NB = (J + block - 1) // block
    pad = NB * block - J
    sp = torch.nn.functional.pad(scores, (0, pad), value=float("-inf"))
    bm = sp.view(B, G, R, NB, block).max(dim=-1).values                 # block-max
    bm = torch.where(torch.isfinite(bm), bm, torch.full_like(bm, -1e30))
    kb = select_blocks(bm, pos_abs, window, top_k, block)               # [B, G, R, NB]
    ar = torch.arange(J, device=scores.device).view(1, 1, 1, J)
    blk = ar // block
    m = kb.gather(-1, blk.expand(B, G, R, J))                           # [B, G, R, J] keep per position
    causal = ar <= pos_abs[:, None, :, None]                            # j <= pos_abs per row, broadcast over groups
    return (m & causal).permute(0, 2, 1, 3)                             # [B, R, G, J]


if __name__ == "__main__":
    rng = np.random.default_rng(70)
    # 1. round-trip: encode a known slot (nm, nibbles, signs), dequant, compare against the direct construction
    n = 512
    nm = rng.uniform(0.5, 2.0, n).astype(np.float16)
    e = rng.integers(0, 64, (n, 16, 4)).astype(np.uint32)
    s8 = np.zeros((n, 16), np.uint8)
    for p in range(4):   # the sign bits of nibble p live at bits 2p..2p+1 of the lane's sign byte
        s8 |= (((e[:, :, p] >> 4) & 3) << (2 * p)).astype(np.uint8)
    q16 = np.zeros((n, 16), np.uint32)
    for p in range(4):
        q16 |= ((e[:, :, p] & 15) << (4 * p)).astype(np.uint32)
    blk = np.zeros((n, 50), np.uint8)
    blk[:, 0:2] = nm[:, None].view(np.uint8)
    lo16 = np.ascontiguousarray(q16.astype("<u4").view("<u2")[:, 0::2])   # low u16 of each word, lane-major
    blk[:, 2:34] = lo16.view(np.uint8)                                    # LE bytes in lane order
    blk[:, 34:50] = 0
    blk[:, 34 + np.arange(16)] = s8   # kernel reads the sign byte of lane L at blk + 34 + L
    got = dequant_slots(blk)
    c = np.zeros((n, 128), np.float32)
    for lane in range(16):
        for p in range(4):
            c[:, 8 * lane + 2 * p] = _LUT[e[:, lane, p], 0]
            c[:, 8 * lane + 2 * p + 1] = _LUT[e[:, lane, p], 1]
    want = (c @ _P.T) * nm.astype(np.float32)[:, None]
    err = np.abs(got - want).max()
    assert err < 1e-6, f"round-trip mismatch {err}"
    # 2. merge matrix rows are the unit quaternions (orthonormal 4x4 blocks)
    for lane in range(16):
        for k in range(2):
            b0 = 8 * lane + 4 * k
            sub = _P[b0:b0 + 4, b0:b0 + 4]
            assert np.abs(sub @ sub.T - np.eye(4)).max() < 1e-5, "quaternion block not orthogonal"
    # 3. magnitude sanity: random slots, but only those whose stored f16 norm is finite
    rnd = rng.integers(0, 256, (56, 50)).astype(np.uint8)
    nmf = rnd[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1)
    out = dequant_slots(rnd)
    fin = np.isfinite(nmf) & (nmf > 0)
    assert np.isfinite(out[fin]).all(), "dequant produced non-finite values for a finite norm"
    bound = np.abs(nmf[fin]) * 3.0 * CB_MAX   # |row abs sum| of the merge matrix is <= 4 unit-quaternion coeffs
    assert (np.abs(out[fin]) <= bound[:, None]).all(), "dequant magnitude out of range"
    Kd, Vd = dequant_position(rng.integers(0, 256, (7, 400)).astype(np.uint8),
                              rng.integers(0, 256, (7, 400)).astype(np.uint8))
    assert Kd.shape == (7, 8, 128) and Vd.shape == (7, 8, 128), "per-position dequant shape"
    print(f"[PASS] iso3_kv: round-trip (max err {err:.2e}), quaternion blocks orthogonal, magnitudes in range")
