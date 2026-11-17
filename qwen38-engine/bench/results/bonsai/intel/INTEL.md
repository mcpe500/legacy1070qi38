# Intelligence evaluation: Ternary Bonsai 2 27B vs Qwen3.8-27B Q4_K_M (2026-11-17)

Log-likelihood scoring with `llama-perplexity` (no sampling). Both models use the same tokenizer (vocabulary 248320), so perplexity and
KL divergence are directly comparable. Scripts: `bench/bonsai_intel.sh` (one task), `bench/bonsai_intel_all.sh` and `bench/bonsai_intel_plan.sh` (the runs below);
this file: `python3 bench/bonsai_intel_summary.py`. Datasets: WikiText-2 test, HellaSwag validation, Winogrande debiased, ARC-Challenge and MMLU validation
(llama.cpp evaluation sets). Every run alone on the GPU, peak VRAM sampled every 100 ms, cap 7373 MiB.

## 1. Language modelling (WikiText-2 test, 40 chunks x 512 tokens = 20,480 tokens)

| Model | PPL ↓ | KL divergence vs Qwen Q4_K_M ↓ | Same top-1 token as Qwen ↑ |
|---|---:|---:|---:|
| Qwen3.8 Q4_K_M | 5.916 ± 0.140 | reference | reference |
| Bonsai `Q2_0` | 7.995 ± 0.204 | 0.3561 ± 0.0059 | 76.48 ± 0.42 % |
| Bonsai `PTQ1_0` | 8.000 ± 0.204 | 0.3561 ± 0.0059 | 76.45 ± 0.42 % |

The reference logits are Qwen3.8-27B **Q4_K_M** (the project baseline), not the FP16 model, so KL divergence measures how far Bonsai is from the
baseline we run, not from the original weights.

## 2. Task accuracy on matched subsets (accuracy %, ± binomial standard error)

| Task | N | Qwen3.8 Q4_K_M | Bonsai `Q2_0` | Bonsai `PTQ1_0` | Bonsai `Q2_0` on the larger run |
|---|---:|---:|---:|---:|---|
| HellaSwag (acc_norm) | 100 | 83.0 ± 3.8 | 72.0 ± 4.5 | 72.0 ± 4.5 | 76.5 % at N = 200 |
| Winogrande | 200 | 77.5 ± 3.0 | 73.5 ± 3.1 | 73.5 ± 3.1 | 72.0 % at N = 500 |
| ARC-Challenge | 100 | 52.0 ± 5.0 | 48.0 ± 5.0 | - | 50.7 % at N = 298 |
| MMLU | 150 | 43.3 ± 4.0 | 46.7 ± 4.1 | - | 43.6 % at N = 500 |

Paired test on the same items (Qwen vs Bonsai `Q2_0`; per-item results reconstructed from the running accuracy in the logs, exact McNemar test):

| Task | N | only Qwen correct | only Bonsai correct | p (two-sided) | Reading |
|---|---:|---:|---:|---:|---|
| HellaSwag (acc_norm) | 100 | 11 | 0 | 0.001 | Qwen better (p < 0.05) |
| Winogrande | 200 | 24 | 16 | 0.268 | no significant difference |
| ARC-Challenge | 100 | 11 | 7 | 0.481 | no significant difference |
| MMLU | 150 | 8 | 13 | 0.383 | no significant difference |

## 3. Runs

| Run | Task | N | GPU layers | Peak VRAM MiB | Wall s | Status |
|---|---|---:|---:|---:|---:|---|
| `ptq1_hellaswag` | hellaswag | 100 | 99 | 6185 | 347 | OK |
| `ptq1_kld` | kld | 40 | 99 | 5691 | 426 | OK |
| `ptq1_ppl` | ppl | 40 | 99 | 5691 | 371 | OK |
| `ptq1_winogrande` | winogrande | 200 | 99 | 6185 | 124 | OK |
| `q2_arc` | arc | 298 | 59 | 7307 | 967 | OK |
| `q2_hellaswag` | hellaswag | 200 | 60 | 7279 | 1012 | OK |
| `q2_kld` | kld | 40 | 99 | 7371 | 201 | OK |
| `q2_mmlu` | mmlu | 500 | 60 | 7283 | 1620 | OK |
| `q2_ppl` | ppl | 40 | 99 | 7371 | 147 | OK |
| `q2_winogrande` | winogrande | 500 | 60 | 7273 | 952 | OK |
| `qwen_arc` | arc | 100 | 25 | 7159 | 1776 | OK |
| `qwen_hellaswag` | hellaswag | 100 | 25 | 7113 | 3624 | OK |
| `qwen_kldbase` | kldbase | 40 | 25 | 6945 | 799 | OK |
| `qwen_mmlu` | mmlu | 150 | 25 | 7117 | 2947 | OK |
| `qwen_winogrande` | winogrande | 200 | 25 | 7093 | 2152 | OK |
