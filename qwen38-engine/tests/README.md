# tests

Parity numerik, kernel CUDA, rollback, state round-trip, server API, dan logika WebUI.

| Tes | Cakupan |
|---|---|
| `layer01_parity` | parity layer CPU terhadap dump referensi |
| `foundation_test` | konversi FP16 dan iso3 byte-identik |
| `mt_dot_test` | kernel multi-token W4A8 |
| `stop_filter_test` | stop string yang terpotong antar-token/chunk |
| `gemv_parity`, `q8_gemv_parity` | dequantisasi dan GEMV vs libggml |
| `gpu_layers_parity` | parity layer CUDA |
| `iso3_gpu_test` | bytes iso3 GPU vs referensi host |
| `ternary_test`, `ternary_real_test` | kernel ternary sintetis dan bobot Bonsai nyata |
| `attn_long_test` | fast vs tiled dan tiled vs double reference sampai 131072 posisi |
| `server_test.py`, `curl_check.sh` | endpoint, SSE, error, cache, dan multi-turn |
| `webui_test.py` | parser `<think>`/SSE dan wiring DOM di QuickJS |

Status: **sudah terisi dan dipakai**. Ikuti `.agents/skills/webui-and-parity-bench/SKILL.md`.
