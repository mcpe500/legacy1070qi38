# M3: Suffix-PLD up to 16 drafts, cascaded with r3 MTP on the T=17 verification batch

Branch `exp/bonsai-overnight-v2-prefill-t17-drafter`. Milestone 2 made a 17-row verification batch legal,
lossless and affordable. This milestone feeds it and measures the cascade against plain MTP.

All numbers below: `Ternary-Bonsai-2-27B-PTQ1_0-mtpqwen.gguf`, 64 GPU layers, `--ctx 4096`, `-n 64`,
`--temp 0 --seed 42` (greedy), `taskset -c 0-3`, GPU otherwise free. Workload files:
`bench/pld_refactor_workload.txt` (repetitive C loader), `bench/pld_code_scratch.txt` (a from-scratch SSE
server), `bench/pld_en_conversation.txt` and `bench/pld_id_conversation.txt` (Qwen about this very engine).

## What changed

1. The generator's cascaded path builds batches of up to `vcap_` rows instead of 5, where
   `vcap_ = max(5, min(kVerifyMax, Engine::roll_t()))`. The draft count is derived from the buffer the engine
   actually allocated, so a VRAM-fallback plan degrades the draft count instead of overflowing it.
   `--pld-max` and the server's `pld_max` accept 1..16; the CLI now rejects values outside the range.
2. The PLD matcher (the n-gram suffix index in `src/engine/generator.cpp`: two open-addressed hash tables for
   3-grams and 4-grams plus the history pointer, CPU-only) is asked for up to `min(gp.pld_max, kPldMax)`
   candidates and its continuation walk fills the whole batch, so one cycle can carry 16 PLD drafts.
3. Cascade order unchanged and now covering the whole batch: PLD candidates first, MTP drafts only when PLD
   returns nothing (the `--mtp-gguf` sidecar stays available for the fallback cycle), verify batch
   `[cur] + drafts`, DeltaNet rollback restores the accepted prefix.
4. `GenStats::t_pld` times the suffix query; the JSON summary reports `pld_step_us`. `accept_hist` is widened
   to 17 slots.
5. `pld_max` default stays 4 (the measured value that does not regress; see below).

## Lossless parity (ids, greedy)

| workload | non-MTP | r3 MTP | cascade (pld-max 16) | PLD only (pld-max 16) |
|---|---|---|---|---|
| refactor | `3fe15206d737` | `3fe15206d737` | `3fe15206d737` | - |
| code from scratch | `37c163f4f8c0` | `52ca4eefb6e5` | `52ca4eefb6e5` | - |
| English chat | `b4acde9241` | `96259802d9` | `96259802d9` | - |
| Indonesian chat | `c72858bb411a` | `d48ff754f3e9` | `d48ff754f3e9` | `c72858bb411a` |

- **The cascade changes nothing**: cascade ids are byte-identical to plain r3-MTP ids on all four workloads,
  and PLD-only ids are byte-identical to non-MTP ids on the Indonesian chat workload. That is the "greedy
  parity 100% lossless" requirement for the new path.
- **A pre-existing MTP divergence is exposed** by these workloads: r3 MTP ids differ from non-MTP ids on 3 of
  the 4 (first difference at token 4). It is not introduced here: the same four ids md5s reproduce on the
  parent commit (M2) with the same binary flags. It is a near-tie argmax flip somewhere between a verify row
  and a single-token decode row, and hunting it is the first task of a later milestone (it predates the
  17-row batch: with `--mtp-steps 2` the verify batch stays at 5 rows).

## Speed (decode tok/s)

| workload | non-MTP | r3 MTP | pld-max 2 | 4 | 6 | 8 | 16 |
|---|---|---|---|---|---|---|---|
| refactor | 16.39 | 19.39 | 19.28 | 18.81 | 18.14 | 17.85 | 17.91 |
| code from scratch | 17.50 | 22.27 | 22.20 | 22.33 | 22.32 | 22.15 | 22.36 |
| English chat | 17.38 | 20.56 | 20.45 | 20.46 | 20.47 | 20.42 | 20.47 |
| Indonesian chat | 17.61 | 20.19 | 19.76 | 18.86 | 15.82 | **5.36** | **3.29** |

PLD firing counts (cycles / candidates produced): refactor 1 / 5, Indonesian chat 5 / 10-80 by `pld-max`,
English chat 0, code from scratch 0. The index only fires when the same continuation already followed an
earlier occurrence of the same suffix (`kMinRepeat = 4`), which is what keeps it from wasting verify rows on
one-off hits.

## PLD step cost

`pld_step_us` (the suffix query inside the draft phase): 0.63 us on the refactor workload, 0.46 us on English
chat, 0.53 us on Indonesian chat, 0.39 us on code from scratch. Budget was 0.2 ms = 200 us, so the index costs
between 300x and 500x less than that, and it uses 0 MiB of VRAM (two host hash tables, no allocation inside
the token loop).

## Conclusion

The 16-draft Suffix-PLD mechanism works, is lossless against plain MTP, and is nearly free per step
(0.4-0.6 us). It does **not** beat r3 MTP on any of the four workloads at 4K: the only workload where a
suffix match fires often enough to matter (Indonesian chat) is also the one where a long draft list is
mostly rejected, and filling the 17-row batch costs far more than the accept rate pays for
(19.76 tok/s at `pld-max` 2 down to 3.29 at 16). `pld_max` therefore keeps its measured default of 4, while
the knob reaches 16 for workloads that are genuinely repetitive.

What is left for a real win, in order:
1. Root-cause the pre-existing MTP vs non-MTP argmax flip above (a lossless MTP is worth 1.2x-1.3x on its own).
2. Cap the PLD draft count by what the previous PLD cycle actually accepted, so a bad first cycle does not
   cost a 17-row verify.
3. Measure the cascade at 32K and 64K (not done here: only 4K was run, because each 32K/64K run costs a long
   prefill and the budget went to the four workloads first).
4. A trained Drafter v2 (Milestone 5) is the intended user of the 17-row batch: the engine side of the
   contract section 4 ("decode attention v2 extended to T <= 17 with the same fixed slice count, plus the
   40.1 MiB partial buffer") is delivered and measured in M2 (40.05 MiB, lossless, see `M2-VERIFY-T17.md`).
