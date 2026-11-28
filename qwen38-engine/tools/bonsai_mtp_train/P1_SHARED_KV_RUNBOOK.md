# P1-rev runbook — shared-KV MTP head training (issue #70, teacher PTQ1_0-native)

Goal: retrain the Bonsai 2.5 draft head (blk.64) so its attention reads the **teacher's layer-63 K/V**
(the E-D2a zero-shot test proved the current head cannot: acceptance 2.6 % vs 57.9 %). No BF16
teacher anywhere: the K/V come from the **PTQ1_0 GGUF run by the engine itself**, in the exact byte
format the kernels consume (iso3, 400 B/position/tensor).

## 0. What is already in place (CPU-verified)

| piece | file | check |
|---|---|---|
| sideband block-max + off/on bench | `src/cuda/ops.cu`, `bench/attn_dec_bench.cu` | E-D1: bit-identical, +0.033 ms/layer @ 102K |
| zero-shot shared-KV flag | `QWEN_MTP_SHARED_KV=1` | E-D2a: lossless, acceptance 2.6 % (expected pre-training) |
| selector + sparse decode | `src/cuda/dfly2_select.cuh`, `op_attention_sparse`, `QWEN_MTP_SPARSE=1` | `dfly2_select_test` PASS (CPU); GPU test pending |
| teacher K/V dump | engine `--dump-kv63` (kv63_k/v.npy per shard) | compiles; run on the T4 (below) |
| dequant + selection (python) | `tools/bonsai_mtp_train/iso3_kv.py` | selftest PASS (round-trip exact, quaternions orthogonal) |
| shared-KV training path | `train_bonsai_mtp.py --kv63` | `kv63_smoke.py` PASS (finite, causal, grads reach q/o) |

## 1. Dump the teacher K/V (T4 or the 1070 Ti, one burst)

The dump runs the PTQ1_0 GGUF on the GPU, one sample at a time (`eng.reset()` per sample), so the
layer-63 cache of positions 0..seq_len-1 is exactly what a decode at that position would read.

```bash
# tokenize the corpus once (already part of the old pipeline, same plan.json)
python3 tools/bonsai_mtp_train/tokenize_for_engine.py <corpus.txt> --out /data/p

scripts/gpu_isolate.sh taskset -c 0-3 ./build/qwen38-engine \
  -m /home/ivan/models/27b-bonsai-2.5/Ternary-Bonsai-2.5-27B-PTQ1_0.gguf \
  --dump-targets /data/pkg --tokens-bin /data/p/tokens.bin \
  --dump-samples 2000 --dump-seq-len 1024 --shard-samples 32 \
  --dump-top-k 64 --dump-hidden-dtype int4g64 \
  --dump-kv63 --max-gb 0
```

Size: hidden int4g64 ≈ 22.3 GB + top-64 ≈ 3.1 GB + **kv63 = 2000 × 1024 × 800 B ≈ 1.6 GB**.
On Colab mount the same Drive paths; the dump is resumable (complete shard dirs are skipped).

## 2. Train on a T4 (Colab)

```bash
python3 tools/bonsai_mtp_train/train_bonsai_mtp.py \
  --from-targets /data/pkg --kv63 \
  --drafter mtp --unroll 3 \
  --steps 4000 --batch 2 --seq-len 1024
```

Notes
- `--kv63` switches the draft head's attention to the teacher K/V (`MTPHead.forward(kv63=...)`),
  with the E-D2b selection computed per batch from the real scores (the training twin of the GPU
  selector; the sparse-mask builder is `iso3_kv.sparse_kv_mask` — top-32 blocks + window 128 + sink).
- The k/v projections of the head become dead weight at training time (their gradients are zero on
  the attention path); keep them in the state dict — the engine GGUF still carries them and the
  zero-shot shared-KV path skips them at run time.
- Crop alignment: rows of a crop that starts at offset `o` use absolute RoPE on the q side
  (`pos_abs = pos + o`), matching the phase the engine baked into the dumped K/V. The causal mask
  stays relative (`j <= row`), which the kv63_smoke causality test pins down.
- Expected gate (issue #70): acceptance of the trained head, measured by re-running the E-D2a
  zero-shot command, must recover to within 3 points of the 57.9 % dense baseline at 4K, and the
  sparse (`QWEN_MTP_SPARSE=1`) variant must stay within 3 points of that.

## 3. Acceptance re-measurement after training (the P1 gate)

```bash
# dense shared-KV with the NEW head (same command as E-D2a, new GGUF from inject_mtp_gguf.py)
QWEN_MTP_SHARED_KV=1 ./build/qwen38-engine -m <new-mtp.gguf> -f /tmp/opencode/prompt4k.txt \
  -n 256 --ctx 8192 --temp 0 --seed 42 --mtp-steps 2 --mtp-adaptive 0 --pld 0 --gpu-layers 64 --gpu-mtp 1
# sparse shared-KV: adds QWEN_MTP_SPARSE=1 (reads only top-32 blocks + window + sink, ~2180 positions)
```

Gate table (from issue #70, unchanged): coverage ≥ 0.80 @ 32K; acceptance −3 points vs dense;
greedy ids identical (lossless); VRAM ≤ 7373 MiB; MTP cycle @ 102K ≤ 160 ms.

## 4. Open items (GPU-free day)

1. Run `bench/attn_dec_bench` off/on for the sparse pointer path and the E-D1b idea if needed.
2. First real `QWEN_MTP_SPARSE=1` engine run (compile + CPU logic verified; kernel timing + parity
   vs dense-list on real data is the remaining unknown).
3. Port `pc.load_shard` staging if Drive FUSE stumbles on the extra 1.6 GB (it is tiny vs 25 GB).
