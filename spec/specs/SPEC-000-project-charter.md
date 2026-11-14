# SPEC-000: Project Charter — legacy1070qi38

> Status: **Active (direvisi: target model dikoreksi ke Qwen3.8-27B)**
> Lisensi: Apache 2.0

---

## Identitas Proyek

| Field | Value |
|---|---|
| Nama | `legacy1070qi38` |
| Tagline | *Old Hardware, New Intelligence* |
| Tema | Resurrection, Kiroween 2026 |
| Jenis | Open-source LLM inference engine / toolkit |
| Hardware | NVIDIA GTX 1070 Ti (8 GB, Pascal SM 6.1), RAM 16 GB DDR4 |
| Model target | **Qwen3.8-27B** (dense, hibrida Gated DeltaNet + full attention) |
| Backend utama | llama.cpp (CUDA, SM 61) |

---

## Latar Belakang

AI semakin bergantung pada GPU modern dengan memori besar yang tidak terjangkau semua orang. Banyak GPU
lama masih punya tenaga komputasi yang bisa dipakai. Pertanyaan proyek ini:

> *Seberapa jauh kita bisa mendorong batas hardware lama melalui rekayasa perangkat lunak?*

Fokus: optimasi inference, quantization, manajemen memori, dan strategi eksekusi CPU–GPU. Model 27B
dijalankan **tanpa mengasumsikan seluruh bobot muat di VRAM**.

---

## Model Target

Rincian lengkap dan sumber ada di `spec/AGENTS/MASTER.md`, bagian "Fakta Model".

| Properti | Nilai |
|---|---|
| Parameter | 27B teks (27.8B dengan vision tower) |
| Layer | 64 (16 full attention + 48 Gated DeltaNet) |
| KV cache | ≈ 64 KiB per token |
| Konteks native | 262,144 |
| MTP head | Ada (untuk speculative decoding) |

> ⚠️ Fakta di atas dari model card komunitas. Verifikasi ke kartu resmi dan dengan `gguf inspect`.

---

## Keterbatasan Hardware

| Fitur | Status | Implikasi |
|---|---|---|
| Tensor Cores | ❌ | Matmul lewat CUDA core |
| INT8 DP4A | ❌ | GPTQ/AWQ tidak berguna |
| BF16 | ❌ | Emulasi, lambat |
| FP16 fast path | ❌ (GP104) | FP16 ≈ FP32 |
| FlashAttention (library) | ❌ | FA bawaan llama.cpp perlu diuji |
| Bandwidth RAM | ❓ | Menentukan TPS hybrid; ukur |

---

## Strategi

| | Strategi |
|---|---|
| A | Hybrid CPU+GPU dengan Q4_K_M sampai IQ3 |
| B | Muat penuh di 8 GB dengan quant ultra-rendah (~7.7 GiB) |
| C | Speculative decoding dengan MTP head |
| D | Vulkan sebagai pembanding CUDA |

Detail dan estimasi: `MASTER.md` dan `RFC-001`.

---

## Kriteria Keberhasilan

Dievaluasi dari penggunaan memori, kecepatan token, latency, stabilitas, dan kualitas output.

- [ ] Qwen3.8-27B berjalan dan menghasilkan output koheren di GTX 1070 Ti + 16 GB RAM
- [ ] TPS terukur memenuhi minimum di `MASTER.md` untuk minimal satu strategi
- [ ] RAM puncak ≤ 15 GiB tanpa swap
- [ ] Kualitas terukur (perplexity atau KL divergence) untuk quant terpilih
- [ ] Setup terdokumentasi dan bisa direproduksi

---

## Spec Roadmap

| Nomor | Nama | Status |
|---|---|---|
| SPEC-000 | Project Charter | Active |
| RFC-001 | Quantization dan Offload Strategy | Proposed (direvisi) |
| RFC-002 | CLI Tooling | Proposed |
| SPEC-002 | Inference Baseline (build dan model uji kecil) | Ready |
| SPEC-003 | Memory Manager | Skeleton |
| SPEC-004 | KV Cache dan State | Skeleton |
| SPEC-005 | CPU-GPU Offload untuk Qwen3.8-27B | Partial |
| SPEC-006 | Benchmark Suite | Skeleton |

---

## Spirit

> **"Old Hardware, New Intelligence"**

Inovasi tidak selalu soal hardware termahal. Proyek ini ingin membuktikan bahwa optimasi tingkat rendah dan
pemahaman arsitektur hardware tetap berharga, dan membuat inference AI modern lebih praktis, efisien,
terukur, dan dapat diakses pengguna hardware generasi lama.

---

*Lihat `spec/handoff/HANDOFF-LOG.md` untuk riwayat koreksi.*
