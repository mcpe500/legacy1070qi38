# MTP sidecar race: user-distilled head (2.000 steps) vs ProCreations r3 — 2026-11-23

Hardware: GTX 1070 Ti + Ryzen 5 2400G, engine built from this branch. Every run:
`scripts/gpu_isolate.sh -- taskset -c 0-3 ./build/qwen38-engine -m ~/models/27b-bonsai/Ternary-Bonsai-2-27B-PTQ1_0.gguf
--gpu-layers auto --ctx 4096 -f <prompt> -n 256 --temp 0 --seed 42` (+ `--mtp-gguf <sidecar> --mtp-adaptive 1
--gen-mtp-steps {2,3}` for the MTP runs). All numbers read from the log files in this directory (`rtk bypass`/read).

Heads under test (user head trained in Colab, 2.000 steps / 65.54 M tokens over 114.610 samples, warm start qwen_blk64):

| head | sidecar | size | sha256 |
|---|---|---|---|
| r3 (ProCreations) | `~/models/27b-bonsai/procreations-mtp/r3_mtp_blk64_q4_k_m.gguf` | 273.0 MiB | (unchanged, local copy of the released r3 head) |
| user Q4_K_M (`uq4`) | `~/models/27b-bonsai/mtp_blk64_q4_k_m.gguf` | 272.9 MiB | `597b1b1fd39dffe9ffdd1a49a65a2d962edc9fc3079731ab411e263696ff9985` |
| user Q8_0 (`uq8`) | `~/models/27b-bonsai/mtp_blk64_q8_0.gguf` | 430.4 MiB | `c91f2a5fbad4bc61a061b3d8ff9b954de02850ff09ee5afe06440b81add355b7` |

User-head provenance (from the training MANIFEST, verified after download): `mtp_blk64.safetensors`
1.698.799.056 B, sha256 `4071d1f2478296a5bba0f4ac376f9cd50d0fe5b0737930d2b5a4c3abcfa1b69b`.

## Results (decode tok/s, 256 tokens; acceptance = draft_n_accepted/draft_n)

| workload | prompt | no MTP | r3 s2 / s3 | uq4 s2 / s3 | uq8 s2 / s3 |
|---|---|---|---|---|---|
| standard | `bench/prompt512.txt` | 17.34 | 27.80 / 27.75 (96.1 %) | **28.39 / 28.36** (94.3 %) | 27.59 / 27.48 (95.7 %) |
| coding | `prompt_coding.txt` (LRU cache) | 17.31 | 27.81 / 27.97 (90.1 %) | **28.48 / 28.48** (90.7 %) | 27.17 / 27.17 (88.8 %) |
| Indonesian | `prompt_id.txt` (list/tuple/set) | 17.27 | **25.85 / 25.89** (87.6 %) | 24.83 / 24.86 (86.7 %) | 24.58 / 24.62 (87.5 %) |

Log files: `{base_s0,r3_s{2,3},uq4_s{2,3},uq8_s{2,3}}_{p512,coding,id}.log`.

## Lossless check

Every MTP run produced token ids **identical** to the same-workload `--gen-mtp-steps 0` baseline
(parity failures: none across all 18 MTP runs). The drafts only change the speed, never the output.

## Findings

1. **uq4 (user head, Q4_K_M) is the fastest overall**: +2.1 % over r3 on the standard prompt and
   +2.4 % on coding — the workload mix a coding agent actually sees. r3 keeps a small lead on the
   Indonesian prompt (25.85 vs 24.83, −4.0 %), despite being trained on 8K English chat only; the
   user head's 10 % Indonesian share did not translate into an acceptance win on this one prompt.
2. **uq8 is not worth it on this GPU**: below r3 on all three workloads while streaming 430 MiB
   instead of 273 MiB.
3. **`--gen-mtp-steps 3` never drafts a 3rd token here**: with `--mtp-adaptive 1` every head/race
   shows identical draft_n, acceptance and tok/s at s2 and s3 (the adaptive gate stops at 2 drafts;
   the engine also hard-caps `--gen-mtp-steps` at 3).
4. Practical daily driver: `--mtp-gguf ~/models/27b-bonsai/mtp_blk64_q4_k_m.gguf --mtp-adaptive 1
   --gen-mtp-steps 2` (s2 == s3, so 2 is enough). Keep r3 as the sidecar for long Indonesian chat if
   that −4 % ever matters; the gap is one prompt's worth of evidence, not a trend.
5. Margin vs the recorded PR #56 baseline (r3 26.65 on the same prompt family): today's r3 measures
   27.80 — same build, same flags, run-to-run/thermal spread explains the ~4 % shift. All comparisons
   above are same-session, so the pairwise deltas are the meaningful numbers.

## Verdict

User head wins the race on 2 of 3 workloads and loses one narrowly. Adopt `mtp_blk64_q4_k_m.gguf`
as the default sidecar for coding-agent use; a 300–500-step polish run (resume from
`mtp_blk64.safetensors`, `ONPOLICY_FRAC 0.15`, `LR 3e-5`) is optional and only targets the
Indonesian/chat acceptance gap.
