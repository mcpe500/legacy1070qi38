# M4: server hardening for Bonsai + MTP + PLD (issue #55)

## Changes

- `--mtp-gguf` works in `--server` (the engine config is shared with the CLI). `/v1/stats` reports `model_file`
  (GGUF + sidecar basenames) and `model_name` (`general.name`; it is "Hf" in the Bonsai file). The API model id stays
  `qwen3.8-27b`, which clients and the WebUI send.
- `--state-save PATH` in `--server`: `POST /v1/state/save` writes the current state to that fixed path (clients cannot
  choose a path), and SIGINT / SIGTERM now shut the server down cleanly (the running generation finishes under the
  generation lock) and save the state on the way out. Before, SIGTERM killed the process.
- `--state-load` in `--server`: the loaded state is also kept as a pinned checkpoint, so every request that extends the
  loaded document reuses it, even after a request about something else replaced the cached sequence.
- Multi-turn prefix cache: the cache was all-or-nothing on the cached sequence (prompt + generated tokens). Any turn whose
  history did not round-trip token for token fell back to a full prefill. This happened when the reply was cut by a stop
  string, by `max_tokens` in the middle of an accepted MTP batch (the accepted drafts were already in the state), when the
  client trimmed or edited the reply, or when it re-tokenized differently. The server now keeps a prompt checkpoint: after
  each prefill the recurrent state (`Engine::save_state(0)`, 150 MiB of host RAM; the positional KV stays in place) is
  copied aside. A later prompt that extends that prompt restores it and prefills only the rest. `x_stats.reuse` says
  `prefix`, `checkpoint` or `none`.
- PLD runs on the MTP verify path: without a draft head `pld` is now reported `false` in `x_stats` instead of `true` with
  0 hits, and the engine message no longer says that `--pld` stays available without MTP.
- CLI: `--gen-mtp-steps N` (drafts of one run; lets a state file of an MTP engine be decoded with 0 drafts).

## Regression and stress

| Test | Result |
|---|---|
| `tests/bonsai_server_stress.py` (PTQ1_0 + r3 Q4_K_M sidecar, ctx 8192, adaptive MTP + PLD) | **57/57 PASS** (`m4_bonsai_server_stress.log`) |
| `tests/server_test.py`, Qwen Q4_K_M, `QWEN_MT_DOT=0` | **69/69 PASS** (`m4_server_test.log`) |
| `tests/webui_test.py` | PASS (`m4_webui_test.log`) |
| `build/stop_filter_test` | 36/36 PASS |
| Qwen Q4_K_M greedy, 48 tokens (`m4_qwen_q4km_p5_mtp0`) | 48/48 identical to `bench/results/stage4/p5_mtp0.txt` |
| Qwen rollback test (`m4_qwen_rollback_test`) | PASS, bit-identical |
| Bonsai + r3 sidecar rollback test (`m4_bonsai_q4km_rollback_test`) | PASS, bit-identical |
| `dn_rec_ab_test`, `iso3_gpu_test` | PASS |

The 10-turn stress (alternating non-stream and SSE, peak VRAM 6545 MiB, RSS peak 1123 MiB):

| Turn | Mode | Prompt | Reused | Gen | tg tok/s | Accept % | PLD hits | MTP drafts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | non-stream | 61 | 0 | 44 | 26.35 | 86.1 | 0 | 36 |
| 2 | stream | 127 | 0 (after the parity request) | 22 | 24.51 | 71.4 | 1 | 17 |
| 3 | non-stream | 172 | 150 | 34 | 27.60 | 89.3 | 0 | 28 |
| 4 | stream | 229 | 207 | 41 | 19.93 | 66.7 | 1 | 17 |
| 5 | non-stream | 296 | 270 | 96 | 28.90 | 89.3 | 1 | 71 |
| 6 | stream | 417 | 0 (after the parity request) | 96 | 20.76 | 66.7 | 3 | 48 |
| 7 | non-stream | 539 | 512 | 36 | 28.83 | 96.2 | 0 | 26 |
| 8 | stream | 605 | 575 | 15 | 30.98 | 100.0 | 0 | 11 |
| 9 | non-stream | 644 | 620 | 95 | 24.52 | 83.3 | 2 | 58 |
| 10 | stream | 767 | 739 | 5 | 16.31 | 50.0 | 0 | 8 |

Other checks in the same run:
- Greedy parity: turns 1, 5 and 10 with MTP + PLD + cache equal plain greedy without cache.
- Trimmed reply: the checkpoint is reused (793 tokens). Its output equals the old prefix path with the same split.
- Stop string starting inside a token: content cut right before it, and it is never sent in the SSE stream.
- SSE output equals the non-stream output.
- State save: 803 tokens, 160 MiB in 0.43-1.41 s. SIGTERM exits with 0.
- Restart with `--state-load`: ready in 39.1 s, almost all of it the model load. The state load itself takes 0.55 s
  (2.73 s for the 128K state in M3). The next turn reuses all 803 tokens and recalls the code word.

Reading: a full prefill from position 0 of the trimmed conversation gives a different greedy text from the checkpoint and
prefix paths, which agree with each other. The prefill is not chunk-invariant in the last bits: the decode attention
kernel (T <= 8) and the prefill kernel (with position slices) round differently, and the chunk boundaries differ. This is
a property of the engine, not of the cache; the GEMV is batch-invariant (bit-identical for every T).
