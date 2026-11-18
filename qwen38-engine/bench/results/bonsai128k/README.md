# Ringkasan konteks panjang `qwen38-engine` (2026-11-18)

Hardware: GTX 1070 Ti 8 GB (SM 6.1), Ryzen 5 2400G pada empat core fisik, 16 GB DDR4.
Budget proses: **7373 MiB VRAM** dan **14,2 GB RAM host**. Semua run memakai
`taskset -c 0-3`; hanya satu proses GPU pada satu waktu.

## Metode

`--depth-bench N` adalah **probe decode-only**:

1. state DeltaNet di-reset;
2. seluruh KV GPU dan CPU dikosongkan;
3. satu langkah warm-up dijalankan di posisi `N-9`, lalu delapan langkah decode terukur di posisi `N-8 ... N-1` (perintah memakai `-n 9`).

Attention tetap membaca seluruh rentang hingga `N`, tetapi isinya nol. Karena itu tabel ini
menjawab “berapa biaya decode pada kedalaman N?”, **bukan** “berapa lama prefill N token?” dan
**bukan** “seberapa baik model pada konteks N?”.

## Probe kedalaman

| Model | Konteks | Layer GPU | Decode probe | Puncak VRAM | RSS puncak |
|---|---:|---:|---:|---:|---:|
| Bonsai `PTQ1_0` | 4096 | 64 | 0,9582 tok/s | 5711 MiB | 542 MiB |
| Bonsai `PTQ1_0` | 32768 | 64 | 0,7569 tok/s | 6069 MiB | 549 MiB |
| Bonsai `PTQ1_0` | 65536 | 64 | 0,6111 tok/s | 6477 MiB | 557 MiB |
| Bonsai `PTQ1_0` | 131072 | 64 | 0,4390 tok/s | 7293 MiB | 572 MiB |
| Bonsai `Q2_0` | 4096 | 64 | 2,2497 tok/s | 7161 MiB | 617 MiB |
| Bonsai `Q2_0` | 32768 | 62 | 1,1379 tok/s | 7287 MiB | 1076 MiB |
| Bonsai `Q2_0` | 65536 | 59 | 0,6914 tok/s | 7303 MiB | 2166 MiB |
| Bonsai `Q2_0` | 131072 | 53 | 0,3786 tok/s | 7285 MiB | 4867 MiB |

`Q2_0` lebih besar, sehingga auto-fit memindahkan lebih banyak layer ke CPU pada konteks besar.
Ini menjelaskan turunnya bandwidth GPU sekaligus bertambahnya KV FP32 pada layer CPU.

## Generasi nyata pada konteks 4096

| Model | Prompt | Decode | Heap alloc di decode | Puncak VRAM |
|---|---:|---:|---:|---:|
| Bonsai `PTQ1_0` | 5 token | 1,0709 tok/s | 0 | 5723 MiB |
| Bonsai `Q2_0` | 5 token | 2,8018 tok/s | 0 | 7173 MiB |

Kedua model menghasilkan teks yang koheren pada prompt “The capital of France is”. Angka ini
adalah implementasi engine khusus, **bukan** benchmark runtime stock llama.cpp/PrismML yang
menghasilkan sekitar 17–20 tok/s.

## Bukti kebenaran

| Pemeriksaan | Hasil |
|---|---|
| `attn_long_test 131072` | 16/16 PASS; fast-vs-tiled dan tiled-vs-reference double pada FP32/iso3 |
| Galat relatif tertinggi | 2,112e-05 (batas 1e-4) |
| Rollback DeltaNet Qwen | PASS bit-identik untuk `keep=1..4` |
| SASS gate | PASS; executable tertaut `IDP.4A x2125`, aritmetika half x0 |
| State round-trip | `state_roundtrip_qwen_q4km.txt`, `state_roundtrip_PTQ1_0_4k.txt`, dan `state_roundtrip_Q2_0_4k.txt` semuanya PASS |
| Contoh CLI state 4K | `qwen_state_save_1k.txt` + `qwen_state_load_1k.txt`: 957 token dipakai ulang, jawaban needle tetap `MAGENTA-FALCON-42` |

## Reproduksi

```bash
# Contoh probe 128K
bench/bonsai_ctx_run.sh PTQ1_0_ctx131072_depth \
  -m "$HOME/models/27b-bonsai/Ternary-Bonsai-2-27B-PTQ1_0.gguf" \
  --gpu-layers auto --ctx 131072 --depth-bench 131072 -n 9

# Probe 128K Q2_0
bench/bonsai_ctx_run.sh Q2_0_ctx131072_depth \
  -m "$HOME/models/27b-bonsai/Ternary-Bonsai-2-27B-Q2_0.gguf" \
  --gpu-layers auto --ctx 131072 --depth-bench 131072 -n 9
```

## Berkas yang dilacak

- `*_depth.txt`: JSON probe kedalaman.
- `*_short.txt`: JSON generasi nyata 4K.
- `qwen_q4km_p5_mtp0.txt`: run regresi Qwen tanpa MTP.
- `qwen_rollback_test.txt`: bukti rollback DeltaNet.
- `attn_long_test.txt`: 16 kasus paritas attention.
- `sass_gate.txt`: bukti SASS.
- `state_roundtrip_qwen_q4km.txt`, `state_roundtrip_PTQ1_0_4k.txt`, `state_roundtrip_Q2_0_4k.txt`: bukti round-trip state.
- `qwen_state_save_1k.txt` dan `qwen_state_load_1k.txt`: contoh CLI state save/load pada prompt 1000 token.

Sampel VRAM mentah (`.vram.csv`), stderr (`.err`), dan log antrian tidak dilacak; `.err` kini masuk
`.gitignore` engine agar artefak itu tidak ter-commit lagi.
