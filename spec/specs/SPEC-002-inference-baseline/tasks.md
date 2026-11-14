# Tasks — SPEC-002: Inference Baseline

> Spec: SPEC-002 | Status: **Not Started**
> Dependencies: tidak ada
> Blocks: SPEC-003, SPEC-004, SPEC-005, SPEC-006

---

## Task List

- [ ] **Task 1: Catat environment**
  - Implementation: `nvidia-smi`, `nvcc --version`, `lscpu`, `free -h`, `sudo dmidecode -t memory` (untuk single atau dual channel)
  - Done when: semua tercatat di SESSION-002. Menjawab sebagian OQ tentang CPU dan channel RAM

- [ ] **Task 2: Build llama.cpp untuk SM 61**
  - Implementation: sesuai design.md; catat commit hash
  - Test: `llama-cli --version` menampilkan CUDA
  - Done when: binary ada dan GPU terdeteksi

- [ ] **Task 3: Jalankan model uji kecil penuh di GPU**
  - Implementation: sesuai design.md
  - Test: output koheren, TPS tercatat
  - Done when: jalur CUDA dan pengukuran terbukti bekerja

- [ ] **Task 4: Baca arsitektur Qwen3.8-27B dari GGUF**
  - Implementation: unduh satu quant (misalnya Q4_K_M, ~16.5 GiB); baca metadata (manual atau `gguf inspect`)
  - Test: jumlah layer, tipe attention, KV heads, head dim tercatat
  - Done when: tabel "Fakta Model" di MASTER dicocokkan dan koreksinya dicatat

- [ ] **Task 5: Coba memuat Qwen3.8-27B parsial**
  - Implementation: dua cara offload (auto dan `-ngl` manual); catat pesan pemuatan, RAM, VRAM
  - Test: berhasil atau gagal, dua-duanya dicatat lengkap
  - Done when: ada jawaban untuk "apakah Gated DeltaNet jalan di SM 6.1"

- [ ] **Task 6: Update handoff**
  - Implementation: buat `SESSION-002`, tambahkan entri ke `HANDOFF-LOG.md`, tandai OQ yang terjawab
  - Done when: spec berikutnya bisa dimulai atau diputuskan beralih ke Vulkan

---

## Catatan

- Task 4 dan 5 menentukan arah proyek. Jika Task 5 gagal di kernel, hentikan dan diskusikan sebelum SPEC-003 sampai SPEC-006.
