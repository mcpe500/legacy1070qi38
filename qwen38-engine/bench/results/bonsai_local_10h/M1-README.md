# M1: ProCreations Bonsai MTP head, `--mtp-gguf` sidecars, A/B/C/D benchmark (issue #55)

## Download and audit (`ProCreations/Ternary-Bonsai-2-27B-MTP`)

Downloaded only the metadata, `model_mtp.safetensors` (sha256 `7a4a18b2...`, matches the repo), `mtp_config.json`, the
runtime patch `runtime/bonsai-mtp-embedding.patch`, `training/*.py` and `reports/*` to `~/models/27b-bonsai/procreations-mtp/`.
The 7.66 GB GGUF was not downloaded.

Audit of the "Bonsai MTP embedding fix":
- The patch makes the llama.cpp MTP graph apply the inverse Hadamard rotation and the signs to the latent `tok_embd` rows
  before `enorm`. qwen38-engine already does this: `Engine::mtp_forward` and `mtp_prefill` build the embedding half of the
  concat with `qmat_row_to_f32(tok_embd_)` followed by `hadamard_unrotate` in ternary mode, and the GPU MTP block receives
  that host-built concat. No engine change was needed.
- `train_bonsai_mtp.py` is consistent with it: the teacher embedding comes from the converted HF model, which the converter
  already unrotated.
- Their hidden state is the output-normed hidden (`logits = h @ head.T`) and position p is paired with h[p-1]: the engine
  convention.
- Their norm weights are zero-centred (+1 at export, as in their `export_mtp.py`); `inject_mtp_gguf.py` adds the 1.
- The r3 head is the Qwen donor head fine-tuned by a few percent: matrix cosine >= 0.9995 against the donor, identical norm
  means after the +1 (`/tmp/b10h_r3_vs_donor.txt` during the run).

## Sidecars (`tools/bonsai_mtp_train/inject_mtp_gguf.py --mtp-only-out`)

| File | Content | Size |
|---|---|---:|
| `r3_mtp_blk64_q4_k_m.gguf` | r3 head, Q4_K (Q8_0 for attn_v and ffn_down, as llama.cpp Q4_K_M) | 272.9 MiB |
| `r3_mtp_blk64_q8_0.gguf` | r3 head, Q8_0 | 430.4 MiB |
| `donor_qwen_blk64.gguf` | Qwen3.8 `blk.64` copied raw from the Qwen GGUF (Q6_K / Q8_0) | 334.7 MiB |

`--mtp-gguf PATH` (CLI and `--server`) mmaps the sidecar next to the 5.4 GiB PTQ1_0 file; nothing is duplicated. Plumbing
checks (`m1_sidecar_*`): the donor sidecar reproduces the injected `-mtpqwen.gguf` run exactly (163/186 accepted, same ids,
same VRAM); every head gives the ids of the plain greedy run. Placement: GPU block at 4K and 64K; at 128K the MTP rollback
buffers push the LM head to the CPU (2.46 tok/s), so the auto mode now drops MTP there and keeps the GPU head
(`m1_sidecar_q4km_ctx131072_place3`: 13.52 tok/s at ctx 131072 with a 509-token prompt, ids equal to the non-MTP run).
`--gpu-mtp 0` (CPU MTP block) now uses the GPU LM head for the draft logits: 5.08 -> 15.05 tok/s.

## Benchmark

Tables: `M1-SUITES-512-4K.md` (server per head, `bench/bonsai_mtp_suite.py`), `M1-SUITES-32K.md` (state-based,
`bench/bonsai_suite_state.py`). Greedy, thinking closed, 256 new tokens. Variants: m1/m2/m3 = fixed 1/2/3 drafts
(verify batch T = 2/3/4), ad = adaptive up to 4 drafts (T up to 5), pld2/pldad = cascaded PLD + MTP. Every one of the 128
(512/4K) and 10 (32K) runs gives exactly the text / ids of the non-MTP run (lossless).

Best variant per suite (speedup against m0 of the same server at 512; against the no-MTP server at 4096, whose sessions
had no m0 row; against the no-MTP state run at 32768):

| Prompt | coding | chat EN | chat ID | repetitive |
|---|---|---|---|---|
| 512 | q4km m1 1.26x, ad 1.24x | q4km m1 1.18x, ad 1.13x | q4km m1 1.23x, ad 1.19x | q4km m2 1.50x |
| 4096 | q4km ad 1.34x | q4km ad 1.16x | q4km ad 1.20x | q4km m2 1.46x |
| 32768 | q4km ad 1.22x | - | - | q4km ad 1.43x |

Readings:
- r3 Q4_K_M is the best head on every suite. Q8_0 has the same acceptance but its larger weights cost more per draft than
  Q4_K (the Q8_0 GEMV of the engine is the slower path), the donor head accepts less.
- Draft depth: T = 2 or 3 wins on chat, T = 3..5 on code; fixed 3 drafts (T = 4) loses on chat. The adaptive mode is the
  best or near-best everywhere and is the recommended setting.
- Cascaded PLD + MTP never beats MTP alone here: PLD candidates take the verify batch first and are rejected more often than
  the MTP drafts (at 512, pld2 is below m2 on 11 of 12 suite/head pairs and pldad below ad on 12 of 12).
- The 512/4K tables were measured on the PR #53 decode kernel; the 32K table on the M2 v3 kernel (see M2-ATTEMPTS.md).
  Speedups are within one build.

Incident: during the 512-token q8 server session a GPU unit test (`attn_long_test`, about 1.2 GB) ran while the idle
server held 6.6 GB, so the GPU total reached 7855 MiB (`suite_vram.jsonl`). The engine itself stayed under the cap; no
benchmark request was running at that time. Later GPU tests only ran with no other process on the GPU.
