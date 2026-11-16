# Head-to-head: llama.cpp (iso3 KV + draft-mtp 2) vs qwen38-engine

Prompt: `bench/prompt512.txt` (509 tokens), 256 generated tokens, temperature 0, seed 42 (top_k 1 for llama-server). GTX 1070 Ti (8192 MiB, VRAM cap 7373 MiB), Ryzen 5 2400G pinned to cores 0-3, 16 GB DDR4, same GGUF (Qwen3.8-27B-UD-Q4_K_M). The two programs ran one after the other, never together. Peak VRAM = highest `nvidia-smi` memory.used sample (200 ms) over the whole run, Peak RSS = VmHWM of the process (includes the file pages of the mapped model).

## Headline

| Program (configuration) | pp tok/s | tg tok/s | MTP accept | Peak VRAM MiB | Peak DDR4 RSS MiB | prompt tokens | generated |
|---|---:|---:|---|---:|---:|---:|---:|
| llama-server `-ngl 25` iso3 KV + draft-mtp 2 | 28.70 | 2.50 | 91 % (164 / 181) | 7239 | 14569 | 509 | 256 |
| qwen38-engine `--gpu-layers 32 --kv iso3 --dp4a all --mtp-steps 2` | 13.42 | 2.58 | 96 % (169 / 176) | 7245 | 9214 | 509 | 256 |

Ratios (engine / llama-server): pp 0.47x, tg 1.03x.

Generated tokens, engine vs llama-server: first difference at token 12 of 256 / 256 (both use an iso3 KV cache but with different attention numerics: llama.cpp quantizes the KV of all 16 attention layers and converts it to f16 for flash attention, the engine quantizes the 8 GPU layers and keeps FP32 on the CPU layers; greedy outputs of a 3-bit KV cache are not expected to match for 256 tokens).

## Repeat round (same commands again; the page cache is in a different state, which moves the prompt-processing numbers)

| Program (configuration) | pp tok/s | tg tok/s | MTP accept | Peak VRAM MiB | Peak DDR4 RSS MiB | prompt tokens | generated |
|---|---:|---:|---|---:|---:|---:|---:|
| llama-server `-ngl 25` iso3 KV + draft-mtp 2 | 28.81 | 2.51 | 91 % (164 / 181) | 7239 | 14656 | 509 | 256 |
| qwen38-engine `--gpu-layers 32 --kv iso3 --dp4a all --mtp-steps 2` | 13.50 | 2.59 | 96 % (169 / 176) | 7245 | 9219 | 509 | 256 |

## All runs

| Run | pp tok/s | tg tok/s | MTP accept | Peak VRAM MiB | Peak DDR4 RSS MiB | prompt tokens | generated |
|---|---:|---:|---|---:|---:|---:|---:|
| `engine_mtp0`: qwen38-engine, --mtp-steps 0 | 16.97 | 2.44 | no MTP | 7163 | 8643 | 509 | 256 |
| `engine_mtp2`: qwen38-engine, --mtp-steps 2 | 13.42 | 2.58 | 96 % (169 / 176) | 7245 | 9214 | 509 | 256 |
| `engine_mtp2_r2`: qwen38-engine, --mtp-steps 2 | 13.50 | 2.59 | 96 % (169 / 176) | 7245 | 9219 | 509 | 256 |
| `llama_ngl24`: llama-server -ngl 24, draft-mtp 2 | 17.42 | 2.46 | 92 % (165 / 179) | 7015 | 14310 | 509 | 256 |
| `llama_ngl25`: llama-server -ngl 25, draft-mtp 2 | 28.70 | 2.50 | 91 % (164 / 181) | 7239 | 14569 | 509 | 256 |
| `llama_ngl25_r2`: llama-server -ngl 25, draft-mtp 2 | 28.81 | 2.51 | 91 % (164 / 181) | 7239 | 14656 | 509 | 256 |
| `llama_ngl26`: llama-server -ngl 26, draft-mtp 2 | 17.01 | 2.59 | 92 % (165 / 179) | 7473 (over cap) | 14641 | 509 | 256 |
| `llama_nomtp_ngl25`: llama-server -ngl 25, no MTP | 29.83 | 2.25 | no MTP | 6605 | 14605 | 509 | 256 |

## llama-server layer count (largest -ngl whose peak VRAM stays at or under 7373 MiB)

| -ngl | peak VRAM MiB | under cap | pp tok/s | tg tok/s |
|---:|---:|---|---:|---:|
| 24 | 7015 | yes | 17.42 | 2.46 |
| 25 | 7239 | yes | 28.70 | 2.50 |
| 26 | 7473 | no | 17.01 | 2.59 |

## Notes

- **Decode (tg):** the engine is about 3 % faster in both rounds (2.58 / 2.59 against 2.50 / 2.51 tok/s). Both programs are bound by reading the CPU-resident weights from DDR4 each step; the engine keeps 32 layers on the GPU, llama.cpp 25, and the engine's MTP accepted 96 % of its drafts against 91 %. The engine without MTP decodes at 2.44, llama-server without MTP at 2.25 tok/s.
- **Prompt processing (pp):** the engine is slower, 0.47x. The GPU sits on a PCIe Gen3 x1 link (0.84 GB/s measured), and the engine streams the 32 CPU-resident layers (7 GiB) through VRAM for every prefill, which costs about 9 s whatever the prompt length; the engine's pp also includes the MTP prefill (about 5 s for 509 tokens: 16.97 tok/s with `--mtp-steps 0`, 13.4 with 2). llama.cpp's own pp moves with the layer count and the page cache (28.7 tok/s at -ngl 25, 17.0 to 17.4 at -ngl 24 and 26 in the same conditions).
- **Memory:** the engine's peak RSS is 9.2 GiB against 14.6 GiB for llama-server. llama-server's figure includes the whole mapped 16 GB GGUF; the engine drops the file pages of the layers it keeps in VRAM and streams the rest.
- **VRAM:** both stay under the 7373 MiB cap (7239 and 7245 MiB). llama-server at -ngl 26 needs 7473 MiB and is excluded.
- **MTP accept %** is drafts accepted / drafts proposed. The two programs draft with different rules (llama.cpp stops drafting under a confidence threshold, the engine always drafts the configured number), so the percentages are not strictly comparable.
- Commands: `bench/compare_llamacpp.sh` (this file), engine run = `taskset -c 0-3 build/qwen38-engine -m GGUF -p "$(cat bench/prompt512.txt)" -n 256 --temp 0 --seed 42 --gpu-layers 32 --kv iso3 --dp4a all --mtp-steps 2`, llama-server run = `taskset -c 0-3 llama-server -m GGUF -ngl N -c 2048 -np 1 -t 4 -ctk iso3 -ctv iso3 -ctkd iso3 -ctvd iso3 --spec-type draft-mtp --spec-draft-n-max 2`, then POST /completion with `n_predict 256, temperature 0, top_k 1, seed 42, cache_prompt false`. Raw data: `bench/results/stage5/`.

