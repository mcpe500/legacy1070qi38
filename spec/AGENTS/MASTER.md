# MASTER AGENT — legacy1070qi38

> **Baca dokumen ini pertama sebelum mengerjakan apapun dalam proyek ini.**
> Semua agents, baik role-based maupun phase-based, tunduk pada aturan di dokumen ini.
>
> **Revisi (pasca-bootstrap):** target model dikoreksi dari "Qwen3-8B / Qwen3-27B" menjadi
> **Qwen3.8-27B saja**. Versi awal memakai asumsi arsitektur yang salah. Lihat "Fakta Model" di bawah.

---

## Identitas Proyek

**Nama:** `legacy1070qi38`
**Spirit:** *Old Hardware, New Intelligence*
**Tema:** Resurrection, memberi kehidupan kedua pada GPU generasi lama
**Lisensi:** Apache 2.0

Proyek ini adalah **LLM inference engine open-source** untuk hardware lama, khususnya NVIDIA GTX 1070 Ti
(8 GB VRAM, Pascal SM 6.1) dengan RAM 16 GB DDR4. **Goal utama: menjalankan Qwen3.8-27B** dengan tokens per
second setinggi mungkin, memakai VRAM dan RAM sehemat mungkin, dan kualitas output yang masih layak.
Tidak diasumsikan seluruh model muat di VRAM.

---

## Hardware Constraints (NON-NEGOTIABLE)

| Komponen | Spesifikasi |
|---|---|
| GPU | NVIDIA GTX 1070 Ti |
| Arsitektur | Pascal (GP104, SM 6.1) |
| VRAM | 8 GB GDDR5 |
| Bandwidth GPU | 256 GB/s |
| CUDA Cores | 2432 |
| System RAM | 16 GB DDR4 |
| Bandwidth RAM | **Belum diketahui.** ~20 GB/s bila single-channel, ~40 GB/s bila dual-channel (asumsi, ukur dengan `hw membw`) |
| CPU | **Belum diketahui** (model, jumlah core, dukungan AVX2) |
| PCIe | 3.0 x16 (~12–13 GB/s praktis) |
| TDP GPU | 180W |

### Yang tidak ada di GPU ini
- ❌ **Tensor Cores** (mulai Volta)
- ❌ **Native INT8 (DP4A) untuk matmul cepat** yang dipakai GPTQ/AWQ
- ❌ **BF16 hardware**
- ❌ **FP16 fast path** (GP104 FP16 ≈ FP32)
- ⚠️ **FlashAttention (library Dao-AILab) tidak mendukung SM 6.1.** Apakah implementasi FA bawaan
  llama.cpp (`-fa`) berjalan baik di Pascal **belum diverifikasi**. Uji, jangan diasumsikan.

---

## Fakta Model: Qwen3.8-27B

> Sumber: model card komunitas di Hugging Face
> ([6block](https://huggingface.co/6block/Qwen3.8-27B-GGUF),
> [guideboardlabs](https://huggingface.co/guideboardlabs/Qwen3.8-27B-Q3-DOWN-XS-GGUF)) dan satu wiki.
> **Belum dicocokkan dengan kartu resmi `Qwen/Qwen3.8-27B`.** Verifikasi dengan `gguf inspect` pada file asli.

| Properti | Nilai | Status |
|---|---|---|
| Jenis | Dense, vision-language (teks, gambar, video masuk; teks keluar) | Dari model card |
| Parameter | 27B teks, 27.8B termasuk vision tower | Dari model card |
| Layer | **64**: 16 full attention (GQA) + 48 Gated DeltaNet (linear attention); `full_attention_interval=4` | Dari model card (satu repo menulis 49 layer DeltaNet, kemungkinan menghitung MTP head) |
| Hidden dim | 5120 | Dari model card |
| Vocab | 248,320, embedding dan output **tidak di-tie** (dua salinan) | Dari model card |
| Konteks | 262,144 native | Dari model card |
| KV cache | ≈ **64 KiB per token** (hanya 16 layer full attention); 4096 token ≈ 256 MiB, 32k ≈ 2 GiB (FP16) | Dari model card |
| KV heads × head dim | 1024 (turunan dari 64 KiB/token: 4 KiB per layer per token) | ⚠️ Turunan, belum dikonfirmasi |
| State DeltaNet | Ukuran tetap, tidak tumbuh dengan konteks; **besarnya belum diketahui** | ⚠️ Ukur |
| MTP head | Ada, bisa dipakai untuk speculative decoding (`--spec-type draft-mtp`) | Dari model card, belum diuji di build kita |
| Vision | `mmproj` terpisah (~0.9 GiB), opsional; tanpa mmproj model jalan sebagai teks | Dari model card |
| Ukuran lain dalam keluarga | Wiki hanya menyebut 27B dan Max (2.4T MoE). **Tidak ada 8B yang terverifikasi** | ⚠️ Cek |

Dampak: rumus KV cache lama (semua layer punya KV) **salah** untuk model ini. Lihat `RFC-001`.

---

## Pilihan Quantization (ukuran file nyata)

> Sumber: tabel 6block. PPL diukur dalam satu setup internal; **hanya bisa dibandingkan di dalam tabel ini**.

| Quant | Ukuran (GiB) | PPL | Catatan |
|---|---|---|---|
| Q5_K_M | 18.70 | 6.8010 | |
| Q4_K_M | 16.53 | 6.8103 | |
| IQ4_XS | 15.28 | 6.8529 | |
| Q3_K_M | 13.23 | 7.1020 | |
| IQ3_M | 12.56 | 7.0306 | Lebih kecil dari Q3_K_M, PPL setara atau lebih baik |
| Q3_K_S | 12.16 | 7.2176 | |
| IQ3_XXS | 11.60 | 7.1994 | |
| Q3-DOWN-XS (repo lain) | 7.68 | 8.27 (protokol berbeda) | Muat penuh di 8 GB; lihat strategi B |

---

## Memory Budget (ESTIMASI, belum diukur)

Total memori fisik = 8 GiB VRAM + 16 GiB RAM = 24 GiB. Bobot yang masuk VRAM **tidak** perlu ada di RAM.

```
Kapasitas VRAM untuk bobot ≈ 8.0 − 0.4 (CUDA context) − 0.25 (KV, 4096 ctx) − ~0.5 (compute buffer, tebakan)
                          ≈ 6.5–6.8 GiB
Bobot di RAM ≈ ukuran file − 6.5 GiB
RAM total    ≈ bobot di RAM + ~2.5 GiB (OS, runtime, KV layer CPU)
Batas atas TPS ≈ bandwidth RAM ÷ bobot di RAM   (bobot GPU menambah ~0.03 s per token)
```

| Quant | Bobot di RAM | RAM total | Batas atas TPS (20 GB/s) | Batas atas TPS (40 GB/s) | Status |
|---|---|---|---|---|---|
| Q5_K_M | ~12.2 GiB | ~14.7 GiB | ~1.5 | ~3 | ⚠️ RAM mepet |
| Q4_K_M | ~10.0 GiB | ~12.5 GiB | ~1.9 | ~3.7 | ✅ Muat |
| IQ4_XS | ~8.8 GiB | ~11.3 GiB | ~2.1 | ~4.2 | ✅ |
| Q3_K_M | ~6.7 GiB | ~9.2 GiB | ~2.8 | ~5.5 | ✅ |
| IQ3_M | ~6.1 GiB | ~8.6 GiB | ~3.1 | ~6.1 | ✅ |
| IQ3_XXS | ~5.1 GiB | ~7.6 GiB | ~3.6 | ~7.3 | ✅ |

> Angka TPS adalah **batas atas teoritis**. Praktisnya lebih rendah. Koreksi dari versi awal: dulu ditulis
> bahwa Q4_K_M "tidak muat di 16 GB RAM". Itu salah, karena sebagian bobot berada di VRAM.

---

## Strategi yang Akan Diuji

| | Strategi | Gagasan | Risiko |
|---|---|---|---|
| A | Hybrid CPU+GPU, Q4_K_M sampai IQ3 | Sebagian layer di VRAM, sisanya di RAM | Tertahan bandwidth RAM |
| B | Muat penuh di 8 GB, ultra-low-bit (~7.7 GiB) | Tidak ada layer di CPU. Model card melaporkan 24.5 tok/s di RX 5700 XT 8 GB (Vulkan), kualitas 86.5% terhadap kontrol 3-bit | Kualitas turun; di 1070 Ti kurang ruang untuk KV dan compute buffer; bandwidth 256 vs 448 GB/s (skala kasar ≈ 14 tok/s, tebakan) |
| C | Speculative decoding dengan MTP head | Verifikasi beberapa token per satu kali baca bobot, mengurangi beban bandwidth RAM | Dukungan di build kita belum diuji |
| D | Backend Vulkan sebagai pembanding CUDA | Hasil B didapat dengan Vulkan | Performa Vulkan di Pascal belum diketahui |

---

## Target Performa (usulan, harus dikalibrasi setelah pengukuran)

| Skenario | Minimum | Target |
|---|---|---|
| A: Q4_K_M hybrid | 1.5 tok/s | 3 tok/s |
| A: IQ3_M / Q3_K_M hybrid | 2 tok/s | 5 tok/s |
| B: muat penuh di VRAM | 8 tok/s | 14 tok/s |
| C: dengan MTP | +30% dari A | +80% dari A |
| RAM puncak | ≤ 15 GiB | ≤ 12 GiB |
| Kualitas | PPL tidak lebih dari 10% di atas Q4_K_M | |

---

## Primary Backend

**llama.cpp** (build CUDA SM 61). Vulkan diuji sebagai pembanding.

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

Dua cara mengatur offload, **uji keduanya**:

```bash
# 1. Biarkan llama.cpp memilih jumlah layer (anjuran model card 6block: jangan set -ngl manual)
./build/bin/llama-cli -m Qwen3.8-27B-Q4_K_M.gguf -c 4096

# 2. Manual
./build/bin/llama-cli -m Qwen3.8-27B-Q4_K_M.gguf -c 4096 -ngl 28 -ctk q8_0 -ctv q8_0
```

> ⚠️ Belum diverifikasi: perilaku auto-fit di build kita, dukungan arsitektur Gated DeltaNet di kernel CUDA
> untuk SM 6.1, dan apakah KV quantization butuh `-fa`.

---

## Approaches yang DILARANG tanpa disclaimer

| Approach | Alasan |
|---|---|
| **GPTQ / AWQ** | Mengandalkan INT8 DP4A yang tidak ada di Pascal |
| **vLLM** | Butuh SM ≥ 8.0 |
| **TensorRT-LLM** | Tidak ada optimasi bermakna untuk Pascal |
| **FlashAttention library (Dao-AILab)** | Tidak mendukung SM 6.1 |
| **BF16 inference** | Tidak ada hardware BF16 |
| **EXL2 / ExLlamaV2** | Jalur cepatnya butuh Tensor Core |

---

## Struktur Spec System

```
spec/
├── specs/          ← Spesifikasi fitur dan RFC arsitektur
├── AGENTS/         ← Instruksi untuk agents (kamu berada di sini)
└── handoff/        ← Log dan dokumen transisi antar sesi
```

### Urutan baca sebelum mengerjakan sebuah spec
1. `spec/AGENTS/MASTER.md` (dokumen ini)
2. Role kamu di `spec/AGENTS/roles/`
3. Phase kamu di `spec/AGENTS/phases/`
4. `spec/handoff/HANDOFF-LOG.md` untuk state terkini
5. Spec yang akan dikerjakan di `spec/specs/`

### Penamaan
- Feature spec: `SPEC-NNN-nama-fitur/` (requirements.md, design.md, tasks.md)
- RFC: `RFC-NNN-nama.md`

---

## Aturan Umum Agent

- Dokumentasi teknis: Bahasa Indonesia. Komentar kode, nama file, variabel: Bahasa Inggris.
- Gunakan tabel untuk perbandingan dan code block untuk perintah.
- Tandai ⚠️ ASSUMPTION pada klaim yang belum diverifikasi. **Sertakan sumber untuk klaim faktual.**
- Konflik dengan spec: catat `[CONFLICT]` di `HANDOFF-LOG.md`, jangan berasumsi.
- Selesai satu task: tambahkan entri di `HANDOFF-LOG.md` (append-only).

---

## Success Criteria Proyek

1. Qwen3.8-27B berjalan di GTX 1070 Ti + 16 GB RAM dan menghasilkan output koheren.
2. TPS terukur (bukan estimasi) ≥ minimum di tabel target untuk minimal satu strategi.
3. RAM puncak ≤ 15 GiB tanpa swap.
4. Kualitas terukur (perplexity atau KL divergence) untuk quant yang dipilih.
5. Setup terdokumentasi dan bisa direproduksi.

---

## Open Questions

Daftar aktif dan statusnya ada di `spec/handoff/HANDOFF-LOG.md`. Yang paling menentukan:

- [ ] Apakah kernel Gated DeltaNet di llama.cpp CUDA jalan di SM 6.1? (bisa langsung gagal)
- [ ] Nama dan konfigurasi persis dari kartu resmi `Qwen/Qwen3.8-27B` (layer, KV heads, head dim)?
- [ ] Berapa bandwidth RAM aktual dan apakah single atau dual channel?
- [ ] Apakah MTP speculative decoding jalan di build kita, dan berapa percepatannya?
- [ ] Apakah strategi B (muat penuh di 8 GB) cukup ruang di 1070 Ti, dan seberapa turun kualitasnya?
- [ ] Apakah auto-fit layer lebih baik dari `-ngl` manual?

---

*Direvisi setelah verifikasi model. Lihat `spec/handoff/HANDOFF-LOG.md` untuk riwayat koreksi.*
