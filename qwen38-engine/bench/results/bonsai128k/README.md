# Ringkasan konteks panjang `qwen38-engine` (2026-11-18/19)

Hardware: GTX 1070 Ti 8 GB (SM 6.1), Ryzen 5 2400G pada empat core fisik, 16 GB DDR4.
Budget proses: **7373 MiB VRAM** dan **14,2 GB RAM host**. Semua run memakai
`taskset -c 0-3`; hanya satu proses GPU pada satu waktu.

## Metode `--depth-bench`

Probe ini **decode-only**:

1. state DeltaNet di-reset;
2. seluruh KV GPU dan CPU dikosongkan;
3. satu langkah warm-up dijalankan di posisi `N-33`, lalu 32 langkah decode terukur di posisi
   `N-32 ... N-1` (perintah memakai `-n 33`).

Attention tetap membaca seluruh rentang hingga `N`, tetapi isinya nol. Karena itu tabel ini
menjawab “berapa biaya decode pada kedalaman N?”, **bukan** “berapa lama prefill N token?” dan
**bukan** “seberapa baik model pada konteks N?”.

## Probe kedalaman

| Model | Konteks | Layer GPU | Decode probe | Puncak VRAM | RSS puncak |
|---|---:|---:|---:|---:|---:|
| Bonsai `PTQ1_0` | 4096 | 64 | 8,5258 tok/s | 5983 MiB | 542 MiB |
| Bonsai `PTQ1_0` | 32768 | 64 | 8,7895 tok/s | 6351 MiB | 549 MiB |
| Bonsai `PTQ1_0` | 65536 | 64 | 6,7641 tok/s | 6759 MiB | 556 MiB |
| Bonsai `PTQ1_0` | 131072 | 64 | 4,5906 tok/s | 7349 MiB | 573 MiB |
| Bonsai `Q2_0` | 4096 | 64 | 8,3228 tok/s | 7315 MiB | 617 MiB |
| Bonsai `Q2_0` | 32768 | 62 | 1,6474 tok/s | 7297 MiB | 1073 MiB |
| Bonsai `Q2_0` | 65536 | 59 | 0,9636 tok/s | **7417 MiB** | 2165 MiB |
| Bonsai `Q2_0` | 131072 | 53 | 0,4655 tok/s | 7295 MiB | 4854 MiB |

`Q2_0` lebih besar, sehingga auto-fit memindahkan lebih banyak layer ke CPU pada konteks besar.
Sampel `Q2_0` 64K melampaui target 7373 MiB; ini dicatat apa adanya sebagai transient
`nvidia-smi`, bukan konfigurasi yang direkomendasikan.

## Generasi nyata

### Prompt 5 token, kernel v1 (`a89f467`)

| Model | Decode | Puncak VRAM |
|---|---:|---:|
| Bonsai `PTQ1_0` | 1,0709 tok/s | 5723 MiB |
| Bonsai `Q2_0` | 2,8018 tok/s | 7173 MiB |

### Prompt needle 4K (4032 token), kernel v2

| Model | Decode | Jawaban | Puncak VRAM |
|---|---:|---|---:|
| Bonsai `PTQ1_0` | 8,1536 tok/s | `MAGENTA-FALCON-42` | 5983 MiB |
| Bonsai `Q2_0` | 8,2092 tok/s | `MAGENTA-FALCON-42` | 7315 MiB |

### Plumbing MTP (head `blk.64` Qwen disuntikkan ke `PTQ1_0`)

Prompt 509 token + 256 token decode, ctx 4096, 64 layer GPU + LM head GPU:

| Mode | Decode | Draft diterima | Bukti |
|---|---:|---:|---|
| MTP 0 | 11,4932 tok/s | – | `PTQ1_0mtpqwen_p512_mtp0.txt` |
| MTP 2 tetap | 17,7078 tok/s | 158/198 (79,8 %) | `PTQ1_0mtpqwen_p512_mtp.txt` |

Output 256 token identik pada kedua mode. Rollback MTP bit-identik untuk `keep=1..4`
(`bonsai_mtp_rollback_test.txt`). Head ini **belum dilatih pada Bonsai**; angkanya bukti
plumbing, bukan kualitas draft trained.

## Kernel v2

- `ternary_ab_test`: **135/135** output byte-identical terhadap kernel referensi `a89f467`
  (`ternary_ab_test.txt`).
- Microbench (`ternary_bench.txt`), PTQ1_0 T=1: ffn_up 15,6→77,6 GB/s, ffn_down 18,5→85,5 GB/s,
  LM head 18,7→91,1 GB/s. Q2_0 praktis tidak berubah karena sudah dekat batas bandwidth.

## Bukti kebenaran

| Pemeriksaan | Hasil |
|---|---|
| `attn_long_test 131072` | 28/28 kasus PASS; fast-vs-tiled, split-K, dan tiled vs referensi double pada FP32/iso3 |
| Galat relatif tertinggi attention | 2,581e-05 (batas 1e-4) |
| Rollback DeltaNet Qwen | PASS bit-identik untuk `keep=1..4` |
| Rollback MTP Bonsai | PASS bit-identik untuk `keep=1..4` |
| SASS gate | PASS; executable tertaut `IDP.4A x2813`, aritmetika half x0 |
| State round-trip 4K | Qwen Q4_K_M, `PTQ1_0`, `Q2_0` semuanya PASS |
| Contoh CLI state 4K | 957 token dipakai ulang; jawaban needle tetap `MAGENTA-FALCON-42` |
| Server multi-request | `server_test.py` + `QWEN_MT_DOT=0`: 69/69 PASS setelah perbaikan slot `Pool` |

## Reproduksi

```bash
# Probe 128K
bench/bonsai_ctx_run.sh PTQ1_0_ctx131072_depth \
  -m "$HOME/models/27b-bonsai/Ternary-Bonsai-2-27B-PTQ1_0.gguf" \
  --gpu-layers auto --ctx 131072 --depth-bench 131072 -n 33

# Kernel A/B dan microbench
./build/ternary_ab_test
./build/ternary_bench

# Server deterministik
env QWEN_MT_DOT=0 python3 tests/server_test.py
```

## Berkas yang dilacak

- `*_depth.txt`: JSON probe kedalaman.
- `*_short.txt`: JSON generasi nyata prompt pendek (kernel v1).
- `*_ctx4k_needle_chat.txt`: generasi nyata prompt 4K (kernel v2).
- `PTQ1_0mtpqwen_p512_mtp*.txt`: A/B plumbing MTP.
- `bonsai_mtp_rollback_test.txt`: rollback blok MTP.
- `ternary_ab_test.txt`, `ternary_bench.txt`: bukti kernel v2.
- `qwen_q4km_p5_mtp0.txt`, `qwen_rollback_test.txt`, `attn_long_test.txt`, `sass_gate.txt`.
- `state_roundtrip_*.txt` dan `qwen_state_{save,load}_1k.txt`: bukti state.

Sampel VRAM mentah (`.vram.csv`), stderr (`.err`), dan log antrian tidak dilacak; `.err` masuk
`.gitignore` engine agar artefak itu tidak ter-commit.
