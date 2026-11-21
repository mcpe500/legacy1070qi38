# M5: Drafter v2 training pipeline synced with the contract (CPU-only, no local GPU)

Branch `exp/bonsai-overnight-v2-prefill-t17-drafter`. Milestone 2 delivered the engine side of the contract
section 4 (decode attention v2 extended to T <= 17 with a fixed slice count, plus the 40.05 MiB partial
buffer, lossless). This milestone brings `tools/bonsai_mtp_train/` to the same contract and makes every claim
reproducible on a CPU.

## What changed

1. **Unrolled MTP, 4..8 steps.** `unroll_K(args)` is now the single source of the unroll count and
   `--unroll` is validated to 1..8 for the `mtp` drafter. The loss loop, the per-step logging and the decay
   already followed `args.unroll`, so only the bound and the helper are new; the selftest still passes
   unchanged.
2. **DFlash B = 16 block schedule.** `--drafter {mtp,df3}`, `--df3-block` (default 16) and
   `--df3-layers {2,3}`. `df3` runs the same unrolled self-rollout with K = the block, i.e. 16 drafts, which
   is exactly the 17-row verification batch the engine verifies losslessly since M2. The layer count is
   recorded for the export metadata.
3. **Colab notebook regenerated.** `train_bonsai_mtp_colab.ipynb` now carries the three new parameters
   (`DRAFTER`, `DF3_BLOCK`, `DF3_LAYERS`) and passes them to the trainer; `check_colab_notebook.py` parses
   every code cell and every script call it contains.
4. **`tools/bonsai_mtp_train/verify_m5.sh` (new)**: the reproducible CPU gate, run before a Colab run and
   after any change to the trainer, the notebook generator or the injector.

## What the gate checks and measures

```
== 1. trainer selftest (CPU) ==             selftest: PASS  (loss 2.4854 -> 0.0430, GGUF shapes ok, 424.7M params)
== 2. drafter schedule matrix ==            schedule matrix: PASS
   mtp unroll=1/2/4/6/8 df3_block=16 -> 1/2/4/6/8 step(s)
   df3 unroll=4 df3_block=12/16 -> 12/16 step(s)
== 3. CLI rejects --unroll 12 ==            PASS
== 4. notebook regenerate + check ==        check_colab_notebook: PASS
== 5. GGUF names the trainer exports ==     15 tensors, no missing and no extra; eh_proj (5120, 10240)
== 6. the converter unrotates token_embd == embedding fix audit: PASS
verify_m5: PASS
```

Full log: `bench/results/bonsai_overnight_v2/verify_m5.txt`.

### Embedding-fix audit (contract section 1)

- The converter (`convert_gguf_to_hf_bf16.py`) lists `token_embd.weight` in
  `prism.hadamard.inverse_weight_names`, and `tensor()` applies `hadamard_unrotate` to every row of those
  tensors, writing them as `model.embed_tokens.weight` in the plain basis. The trainer therefore sees the
  plain-basis embedding, and `MTPHead` feeds `RMSNorm(embed(x_p), enorm)` with it.
- The engine builds exactly the same input on its side: `Engine::mtp_forward` does
  `qmat_row_to_f32(tok_embd_)` followed by `hadamard_unrotate(proj_, signs_5120_, kHidden, 1)`.
- So the ProCreations "embedding fix" is consistent across the three stages: converter (GGUF -> plain-basis
  HF BF16), trainer (student trained on the plain-basis embedding), engine (rebuilds the plain-basis row at
  run time). No second fix is needed, and the selftest's GGUF shape assertions are the ones that would catch
  a drift in the norm convention.

### GGUF sidecar compatibility

`MTPHead.gguf_state()` emits 15 tensors, exactly the names `Engine::load` reads for `blk.64` (4 `nextn.*`
plus the 11 layer tensors), and `nextn.eh_proj.weight` is `(5120, 10240)`, which is the check
`mtp_eh_.in != 2 * kHidden` in `engine.cpp` enforces. `inject_mtp_gguf.py` (the exporter the notebook calls)
was not changed and still consumes the same `mtp_step*.safetensors`.

## What is NOT done

- **No training run was made.** The trainer needs the 54 GB BF16 HF teacher and the corpus on an RTX PRO
  6000; this card (GTX 1070 Ti) is not a training target and the milestone is CPU-only by contract. The gate
  above is the part that can run here.
- **A real 2-3 layer DFlash head is not implemented.** `--df3-layers` only records the target architecture:
  the engine loader knows the one-layer `nextn` layout, so a multi-layer drafter needs the loader path and
  the tensor names of contract section 2 first (EAGLE-3's `fc` 3x5120 -> 5120 and the three hidden taps are
  the same class of work).
- **`T = 17` verification of a real sidecar was not run**, because no sidecar was trained here. The engine
  side it needs was measured in M2 (`verify_t17_test` 32/32 lossless, `dn_rollback_test` 9/9), and the
  generator batch that consumes 16 drafts was measured in M3.
