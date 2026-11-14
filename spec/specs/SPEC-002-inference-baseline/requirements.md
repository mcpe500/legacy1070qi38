# Requirements — SPEC-002: Inference Baseline

> Spec: SPEC-002 | Status: **Ready to Implement (direvisi)**
> Referensi: RFC-001, SPEC-000
> Priority: P0

> **Revisi:** versi awal memakai Qwen3-8B sebagai target. Goal proyek adalah Qwen3.8-27B. Spec ini
> sekarang hanya memvalidasi **jalur kerja** (build, GPU terdeteksi, pengukuran) memakai model uji kecil,
> lalu memeriksa apakah llama.cpp bisa membaca arsitektur Qwen3.8-27B.

---

## Problem Statement

Sebelum mengukur Qwen3.8-27B, kita perlu build llama.cpp untuk Pascal SM 6.1 yang terbukti jalan, dan tahu
apakah arsitektur Gated DeltaNet didukung. Dua hal ini bisa menggagalkan seluruh rencana, jadi dicek paling awal.

---

## Functional Requirements

- **FR-01:** llama.cpp ter-build dengan CUDA untuk SM 61 (`-DCMAKE_CUDA_ARCHITECTURES=61`)
- **FR-02:** Sebuah model uji kecil (GGUF apa saja yang lazim, dipilih saat mengerjakan) dimuat penuh ke GPU dan menghasilkan teks koheren
- **FR-03:** Header GGUF Qwen3.8-27B dibaca (tanpa memuat bobot) dan arsitektur serta jumlah layer dicatat
- **FR-04:** Dicoba memuat Qwen3.8-27B dengan sebagian layer di GPU, dan hasilnya (berhasil, gagal, atau pesan galat) dicatat
- **FR-05:** Statistik TPS dari llama.cpp terbaca dan tercatat

---

## Non-Functional Requirements

- **NFR-01: Reproducibility** — commit llama.cpp, versi CUDA, dan driver tercatat
- **NFR-02: Compatibility** — hanya jalur yang kompatibel dengan Pascal (lihat daftar larangan di MASTER)
- **NFR-03: Stability** — tidak crash pada 100–500 token keluaran model uji

---

## Out of Scope

- Optimasi offload dan perbandingan quant → SPEC-005
- Pengukuran kualitas → SPEC-006
- Speculative decoding → spec terpisah setelah baseline

---

## Success Criteria

- [ ] Build selesai dan `llama-cli --version` menunjukkan CUDA
- [ ] Model uji berjalan di GPU, TPS tercatat
- [ ] Arsitektur Qwen3.8-27B terbaca dari GGUF dan dicatat (layer, tipe attention)
- [ ] Hasil percobaan memuat Qwen3.8-27B tercatat, termasuk jika gagal di kernel Gated DeltaNet
- [ ] Jawaban untuk "apakah kernel Gated DeltaNet jalan di SM 6.1" tercatat di handoff

---

## Dependencies

- **Requires:** tidak ada
- **Blocks:** SPEC-003, SPEC-004, SPEC-005, SPEC-006
