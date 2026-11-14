# Laporan contoh: `bench analyze` pada konfigurasi nyata (Qwen3.5-0.8B IQ4_XS, CPU)

Tanggal: 2026-11-13. Mesin: Ryzen 5 2400G, RAM 15.5 GiB. Binary: fork `08e025c` (`llama-bench`). GPU dimatikan
dengan `CUDA_VISIBLE_DEVICES=` karena VRAM dipegang server lain. Model kecil dipilih agar hasil cepat dan tidak
mengganggu server; **ini contoh format laporan, bukan hasil 27B**.

## Perintah

```bash
CUDA_VISIBLE_DEVICES= python3 tools/legacy.py llama bench --model Qwen3.5-0.8B-IQ4_XS.gguf \
    --bin-dir ~/llama-cpp-turboquant/build/bin --ngl 0 --prompt-tokens 128 --gen-tokens 64 --runs 3 --json > bench-iq4.json
python3 tools/legacy.py hw membw --json                # 14.15 GB/s (2 thread)
python3 tools/legacy.py bench analyze --bench-json bench-iq4.json --gguf Qwen3.5-0.8B-IQ4_XS.gguf \
    --gpu-fraction 0 --ram-bw-gbs 14.15 --json
```

## Hasil

| Besaran | Nilai |
|---|---|
| TPS generasi terukur (`llama-bench`, 3 ulangan) | 38.20 tok/s |
| Bobot dibaca per token (total tensor, `token_embd` di-tie) | 0.4486 GiB |
| Bandwidth efektif = byte x TPS | **18.4 GB/s** |
| Bandwidth RAM yang diberikan (`hw membw`, laju salin) | 14.15 GB/s |
| Batas atas TPS pada 14.15 GB/s | 29.38 tok/s |
| Efisiensi = terukur / batas | **1.30** |
| Putusan alat | "lewat batas yang diasumsikan: bandwidth yang diberikan terlalu rendah (bandwidth efektif minimal 18.4 GB/s)" |

## Cara membaca

- Efisiensi > 1 bukan prestasi. Artinya laju salin `memmove` (14.15 GB/s) **meremehkan** bandwidth baca CPU. Sebagai gantinya kita
  mendapat batas bawah yang lebih baik: **CPU membaca bobot minimal 18.4 GB/s** (C-34).
- Konsekuensinya untuk 27B: asumsi 20 GB/s untuk single channel dan 40 GB/s untuk dual channel (MASTER) tidak bisa dibedakan
  dari angka ini; keduanya masih mungkin. Dugaan dual channel datang dari lalu lintas salin 28 GB/s yang melebihi puncak teoretis
  single channel DDR4-2933 (23.4 GB/s) (C-35), dan harus dikonfirmasi dengan `dmidecode`.
- Model 0.8B (0.45 GiB) sebagian ada di cache CPU, sehingga bandwidth efektif tidak murni dari DRAM. Untuk model 16 GiB angkanya akan lebih rendah.

## Contoh sisi GPU (informal, bukan benchmark terkontrol)

```bash
python3 tools/legacy.py bench analyze --gen-tps 20.4 --gguf Qwen3.8-9B-Q4_K_M.gguf --gpu-fraction 1 --json
```

| Besaran | Nilai |
|---|---|
| TPS (rata-rata `/metrics` server 9B milik pengguna: 9147 token / 447.7 dtk) | 20.4 tok/s |
| Bobot dibaca per token (total - `token_embd`, `output.weight` terpisah) | 4.840 GiB |
| Bandwidth efektif | 106 GB/s dari 256 GB/s |
| Efisiensi | **0.41** ("sedang: bukan sekadar bandwidth; cek kernel dan transfer") |

Angka ini memakai konteks 114.688 dengan KV iso3 dan rata-rata percakapan, bukan decode murni. Untuk perbandingan, pihak lain
melaporkan ±86% (RTX 4090), ±76% (RTX 5090 Linux), ±58% (RX 6800 ROCm) pada qwen35 (C-27).
Benchmark terkontrol di GPU kosong dikerjakan di issue 27B-1 dan 27B-2.
