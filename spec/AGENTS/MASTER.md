# MASTER AGENT — legacy1070qi38

> **Baca dokumen ini pertama sebelum mengerjakan apapun dalam proyek ini.**
> Semua agents, baik role-based maupun phase-based, tunduk pada aturan di dokumen ini.
>
> **Revisi 2:** target model dikoreksi dari "Qwen3-8B / Qwen3-27B" menjadi **Qwen3.8-27B saja**.
> **Revisi 3 (2026-11-13):** fakta model kini dicocokkan dengan kartu resmi `Qwen/Qwen3.8-27B`, hardware
> diukur, dan setiap klaim faktual punya baris di [`spec/research/CLAIMS.md`](../research/CLAIMS.md)
> (ID `C-nn` dipakai di bawah). Klaim tanpa baris di sana tidak boleh ditambahkan.

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
| Bandwidth RAM | **Terukur ≥ 18.4 GB/s baca** (C-34; laju salin 14.15 GB/s, C-33). Dual channel kemungkinan besar (C-35) tetapi **belum dikonfirmasi**. Angka 20/40 GB/s di bawah tetap dipakai sebagai skenario, bukan fakta |
| CPU | Ryzen 5 2400G, 4 core / 8 thread, AVX2, tanpa AVX-512 (C-31) |
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

> Sumber: kartu resmi [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), dicocokkan 2026-11-13.
> Repo resmi hanya berisi safetensors; **GGUF berasal dari pihak ketiga** (C-13). Verifikasi file GGUF
> yang dipakai dengan `gguf inspect` dan `gguf tensors`.

| Properti | Nilai | ID | Status |
|---|---|---|---|
| Jenis | Dense, vision-language (teks, gambar, video masuk; teks keluar); thinking aktif secara default | C-10 | ✅ Resmi |
| Parameter | 27B teks, 27.8B dengan vision tower | C-12 | ⚠ Komunitas |
| Layer | **64** = 16 x (3 x Gated DeltaNet + 1 x Gated Attention), tiap layer diikuti FFN dense (intermediate 17.408) | C-01, C-02 | ✅ Resmi |
| Hidden dim | 5120 | C-02 | ✅ Resmi |
| Gated Attention | 24 head Q, **4 head KV**, head dim 256 | C-03 | ✅ Resmi |
| Gated DeltaNet | 48 head V, 16 head QK, head dim 128 | C-04 | ✅ Resmi |
| Vocab | 248.320 (padded); embedding dan output tercantum terpisah | C-07 | ⚠ "untied" belum dikonfirmasi di GGUF 27B |
| Konteks | 262.144 native, 1.000.000 dengan YaRN statis | C-08 | ✅ Resmi |
| KV cache | **64 KiB per token** (FP16) = 16 x 2 x 4 x 256 x 2 B; 4096 token = 256 MiB, 32k = 2 GiB | C-05 | ✅ Turunan dari resmi |
| State DeltaNet | **≈ 0.146 GiB**, ukuran tetap (tidak tumbuh dengan konteks) | C-06 | ⚠ Turunan; rumus cocok persis pada 0.8B, belum diukur pada 27B |
| MTP head | Ada, dilatih multi-langkah. Di llama.cpp utama: `--spec-type draft-mtp`; **fork kita tidak punya** | C-09, C-18, C-19 | ✅ ada; ✅ jalan di llama.cpp utama (C-51, C-52); percepatan kecil di mesin ini (C-51, C-56) |
| Vision | `mmproj` terpisah (±0.9 GiB), opsional | C-12 | ⚠ Komunitas |
| Model lain dengan arsitektur sama | Ada GGUF **9B** `qwen35` (32 blok, `full_attention_interval=4`) yang dipakai sebagai pengganti arsitektur untuk menguji kernel dan alat. Asalnya diduga distilasi komunitas, **bukan** rilis resmi Qwen | C-20, C-21 | ✅ terukur; ⚠ asal file |

Dampak: rumus KV cache lama (semua layer punya KV) **salah** untuk model ini. Lihat `RFC-001`.

---

## Pilihan Quantization (ukuran file nyata)

> Sumber: tabel 6block (C-13). PPL diukur dalam satu setup internal; **hanya bisa dibandingkan di dalam tabel ini**.
> Kita punya alat untuk mengukur sendiri dengan satu protokol: `quality ppl` dan `quality kld`.

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

## Memory Budget (ESTIMASI)

> Kalibrasi nyata (C-29): pada model 9B yang sama, `budget estimate` memberi 6.99 GiB sedangkan
> `nvidia-smi` mencatat 7.70 GiB (7883 MiB). Overhead di luar bobot dan KV sekitar **1.6 GiB**, bukan
> 0.9 GiB (CUDA 0.4 + compute 0.5) yang diasumsikan di bawah. Tabel di bawah masih memakai asumsi lama,
> jadi **RAM total kemungkinan sedikit lebih besar dan layer di GPU sedikit lebih sedikit** dari yang tertulis.
> Rumus sekarang menghitung juga state DeltaNet (0.146 GiB).

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
| B | Hampir penuh di GPU, ultra-low-bit (~7.7 GiB) | Model card melaporkan 24.5 tok/s di RX 5700 XT 8 GB (Vulkan), kualitas 86.5% terhadap kontrol 3-bit. **Hitungan kita (C-30): 7.68 GiB + KV + state + overhead ≈ 1.6 GiB tidak muat 8 GiB**, sehingga ±1.5 GiB tetap di RAM | Kualitas turun; bandwidth 256 vs 448 GB/s (skala kasar ≈ 14 tok/s, tebakan); sebagian kecil di CPU tetap membatasi TPS |
| C | Speculative decoding dengan MTP head | Verifikasi beberapa token per satu kali baca bobot, mengurangi beban bandwidth RAM. Pihak lain melaporkan 1.3-1.5x di GPU penuh (C-25) dan 1.1x di ROCm (C-26) | **Fork kita tidak punya `draft-mtp`** (C-19): perlu build llama.cpp utama; percepatan pada hybrid CPU belum diketahui |
| D | Backend Vulkan sebagai pembanding CUDA | Hasil B didapat dengan Vulkan; di RDNA2 Vulkan pernah crash pada arsitektur ini (C-46) | Performa dan kestabilan Vulkan di Pascal belum diketahui |

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

**Hasil terukur (2026-11-15, 27B di 1070 Ti, bench pendek):**

| Skenario | Terukur | Terhadap target |
|---|---|---|
| A: Q4_K_M hybrid (`-ngl 99 -ot` FFN 17..64) | 2.85 tok/s (C-54) | minimum 1.5 lulus; target 3 belum |
| A: IQ3_M / IQ3_XXS | 1.15 tok/s (IQ3_XXS, C-50). IQ3_M belum diuji | minimum 2 **tidak** lulus; IQ3_XXS lebih lambat dari IQ4_XS |
| B: muat penuh di VRAM | belum diuji | - |
| C: dengan MTP | +6% terbaik (depth 1, `-ngl 32`, C-51); dengan `-ot` tidak ada percepatan (C-56) | minimum +30% **tidak tercapai** di mesin ini |
| RAM puncak | tersedia terendah 14.12 GiB, swap +0.175 GiB | RAM lulus; "tanpa swap tambahan" tidak penuh |
| Kualitas | PPL IQ4_XS 0.4% di atas Q4_K_M; Q4_0 2.0%; IQ3_XXS 4.1% (C-55) | lulus (batas 10%) |

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

## Aturan Pull Request

| Aturan | Isi |
|---|---|
| Satu tugas, satu PR | Pekerjaan untuk satu issue atau permintaan masuk ke **satu** PR. Perintah lanjutan pada issue yang sama menambah ke PR itu, bukan membuka PR baru |
| Deskripsi: teks + visual | Deskripsi PR memang teks, tapi **visualkan sebisa mungkin dengan SVG**: alur, struktur, perbandingan, hasil ukur. Perubahan lebih dari satu berkas atau fitur baru wajib punya minimal satu SVG |
| Bukti test | Tulis test apa yang dijalankan, hasilnya, dan di mana. Jangan mengklaim lulus bila tidak dijalankan |
| Yang belum | Tulis eksplisit apa yang belum dikerjakan atau diuji, dan asumsi yang dipakai |
| Merge | Hanya oleh manusia. `main` dilindungi |

### Aturan SVG

| Aturan | Alasan |
|---|---|
| Simpan di `spec/specs/assets/` | Lokasi tetap, mudah ditemukan |
| XML valid, tanpa `<script>`, event handler, atau `foreignObject` | Keamanan; pemeriksa otomatis menolaknya |
| Huruf minimal 12 px | Terbaca di tampilan PR |
| Teks harus muat di kotaknya (perkiraan lebar: 0,55 × ukuran huruf per karakter) | Teks yang meluap tidak terbaca |
| Satu gagasan per gambar, ada judul dan legenda | Mudah dipahami orang yang baru melihatnya |
| Sematkan dengan tautan permanen (SHA commit), bukan nama branch | Branch dihapus setelah merge, tautan ke branch akan putus |

Pada PR dari agen `/kiro`, skrip menyematkan SVG otomatis di deskripsi dan memberi peringatan bila PR berisi banyak berkas tanpa SVG.
Pemeriksaan tata letak hanya perkiraan (tanpa renderer). Lihat gambarnya langsung di PR sebelum merge.

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

- [x] Kernel Gated DeltaNet di SM 6.1: **ya untuk 9B di fork** (C-17). Untuk 27B dan llama.cpp utama: lihat issue uji 27B
- [x] Konfigurasi kartu resmi (layer, KV heads, head dim): **terjawab** (C-01..C-05)
- [~] Bandwidth RAM: ≥ 18.4 GB/s baca terukur; **single/dual channel belum dikonfirmasi** (C-35)
- [x] Apakah MTP jalan di build kita, dan berapa percepatannya? **Jalan di llama.cpp utama; paling tinggi +6%, dengan `-ot` tidak mempercepat** (C-51, C-56)
- [~] Strategi B: **tidak muat penuh** menurut hitungan (C-30); kualitas dan TPS nyata belum diukur
- [ ] Apakah auto-fit layer lebih baik dari `-ngl` manual? (issue uji 27B)

---

*Direvisi setelah verifikasi model. Lihat `spec/handoff/HANDOFF-LOG.md` untuk riwayat koreksi.*
