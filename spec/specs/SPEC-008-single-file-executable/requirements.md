# Requirements — Single-File Executable (bobot di dalam binary)

> Spec: SPEC-008 | Status: **Draft**
> Referensi: RFC-004 (Embedding Format), RFC-001 (Quantization Strategy), SPEC-005 (CPU-GPU Offload), SPEC-007 (CLI Tooling)
> Priority: P2

---

## Problem Statement

`qwen38-engine` sudah terbangun menjadi satu executable lengkap dengan WebUI dan server HTTP/SSE di dalam
binary, tetapi **bobot tetap berkas `.gguf` terpisah** (±15 GB untuk Q4_K_M) yang harus disalin berdampingan
dengan engine, cocok dengan path, dan tidak terikat versi terhadap engine yang membacanya. Spec ini menetapkan
agar satu berkas executable memuat engine + WebUI + bobot, dan langsung menyala sebagai webserver
(`./legacy1070qi38` → `http://127.0.0.1:8080`) tanpa instalasi, tanpa file sebelah, tanpa salah path.

Spec ini **tidak** mempercepat inferensi. Decode LLM terikat bandwidth
(`tok/s ≈ bandwidth / bytes bobot per token`, lihat C-71); menempatkan bobot yang sama di dalam executable
tidak mengubah jumlah byte per token. Nilai spec ini adalah distribusi, replikasi, dan pengikatan versi
engine↔bobot — bukan angka tok/s.

---

## Functional Requirements

- **FR-01:** Tanpa argumen apa pun, `./legacy1070qi38` mendeteksi payload bobot di dalam dirinya sendiri,
  memuatnya, dan **langsung menjalankan server** pada `127.0.0.1:8080` (WebUI + `/v1/chat/completions`
  streaming SSE), sesuai perilaku `--server` hari ini
- **FR-02:** `--version` mencetak: commit engine, arsitektur target (`sm_61`/`znver1`), nama+versi payload,
  ukuran payload, dan **digest SHA-256 payload** (verifikasi integritas tanpa menjalankan inferensi)
- **FR-03:** Payload dibaca dengan `mmap` **zero-copy** dari `/proc/self/exe`; tidak ada ekstraksi ke disk,
  tidak ada salinan ke buffer lain; tensor tetap di page cache seperti jalur `-m` saat ini
- **FR-04:** Format payload dan trailer sesuai RFC-004: magic, versi format, offset, panjang, dan digest.
  Trailer dibaca dari ujung berkas (`file_size − N`), sehingga tidak perlu memindai isi berkas
- **FR-05:** Payload rusak, digest tidak cocok, offset tidak align halaman, atau trailer tidak dikenal →
  **galat jelas dengan kode keluar 2** (konvensi kode SPEC-007), tanpa crash, tanpa menjalankan server
- **FR-06:** Dua cara membuat executable fat, keduanya harus berhasil:
  (a) target build `embed_payload` (CMake custom command + `tools/embed_payload.py`, hanya library standar),
  dan (b) penyambungan manual `cat engine.bin payload.bin trailer.bin > legacy1070qi38 && chmod +x`.
  Menambah/ganti payload **tidak** boleh memicu kompilasi ulang engine
- **FR-07:** Mode lama tidak boleh berubah: `-m MODEL.gguf` (file eksternal) tetap jalan penuh.
  Bila payload ada **dan** `-m` diberikan, `-m` menang. `--list-payload` mencetak info payload + verifikasi digest
- **FR-08:** Kembalinya sama: prompt, seed, dan parameter identik harus menghasilkan token greedy identik
  antara mode payload dan mode `-m` eksternal (protocol parity webui-and-parity-bench: cosine ≥ 0.998,
  temp 0 seed 42)

---

## Non-Functional Requirements

- **NFR-01: Performance** — Waktu start sampai server siap terima koneksi **tidak boleh lebih lambat** dari
  mode `-m` (mmap kedua jalur identik; setelah pemanasan, tok/s harus setara). Ukur dan laporkan; jangan
  diklaim lebih cepat
- **NFR-02: Memory (VRAM)** — ≤ 7.2 GiB (6866 MiB), sama seperti batas kerja engine hari ini
  (`qwen38-engine/AGENTS.md` "Open conflicts" #3)
- **NFR-03: Memory (RAM)** — ≤ 14.2 GiB konsumsi proses. `mlock` penuh **tidak diminta**: limit 1.94 GiB
  tanpa root (C-70). Boleh `mlock` sebagian (≤ 1.8 GiB) untuk tensor panas (LM head, MTP head) seperti
  opsi #4 di `prompt2.md`, dan wajib pantau `SwapUsed`
- **NFR-04: Compatibility** — Linux x86-64, CUDA 12.6 (sm_61), `-march=znver1` AVX2/FMA. Tetap patuh
  larangan MASTER: tanpa Tensor Core path, tanpa GPTQ/AWQ/EXL2, tanpa BF16, tanpa FlashAttention Dao-AILab
- **NFR-05: Stability** — Payload tidak ada dan `-m` tidak diberikan → pesan usage, kode keluar 2.
  Server yang sudah berjalan tidak boleh mati karena payload dilepas/ditimpa
- **NFR-06: Security** — Payload hanya dipetakan `PROT_READ` (tidak pernah `PROT_EXEC`, tidak pernah
  dieksekusi). Galat digest memblokir jalannya inferensi, bukan hanya peringatan
- **NFR-07: Build hygiene** — Berkas executable multi-GiB **tidak pernah masuk git** (pola `*.gguf` sudah
  ada; perlu pola baru untuk binary fat). Artefak rilis lewat GitHub Release, bukan LFS (5+ GB melewati
  kuota LFS wajar)

---

## Out of Scope

- Klaim percepatan dari "compile bobot ke binary" — terbukti salah secara bandwidth (C-71); dokumen ini
  eksplisit menolaknya sebagai non-goal
- Kompresi payload (zstd/rsyncable): ia **eksperimen terukur** di RFC-004; default payload **mentah**
- Menulis ulang engine ke C, Rust, atau Goose sebagai mesin inferensi (eksperimen hanya di level kernel,
  Task 6, dengan decision gate)
- AOT codegen seluruh graph inferensi (gaya tinygrad/MLC): belum tentu menembus batas bandwidth di GPU ini
- Dukungan macOS/Windows, multi-GPU, atau mode tanpa GPU (engine hari ini mengasumsikan CUDA)
- Baris CLAIMS untuk eksperimen Ternary Bonsai — dokumen terpisah (lihat HANDOFF-LOG `[OPEN]` terkait)

---

## Success Criteria

- [ ] `./legacy1070qi38` (tanpa argumen) menjalankan server; `curl http://127.0.0.1:8080/` mengembalikan WebUI
- [ ] `--version` mencetak digest SHA-256 yang cocok dengan `sha256sum` payload pembangunnya
- [ ] Parity FR-08 lulus pada minimal 3 prompt (pendek, multi-turn, 4096 token)
- [ ] Payload sengaja diperiksa 1 byte → server menolak jalan, kode keluar 2, pesan jelas
- [ ] Ukuran berkas, waktu start (mode payload vs `-m`), RAM puncak, VRAM puncak: terukur dan ditulis di
      `bench/` + laporan; tidak ada angka yang diklaim tanpa sumber
- [ ] `python3 tools/legacy.py spec validate` lulus dan `python3 tools/test_legacy.py` tidak regresi

---

## Dependencies

- **Requires:** RFC-004 (keputusan format embedding), SPEC-005 (penempatan layer CPU/GPU yang berlaku juga
  pada mode payload), SPEC-007 (konvensi CLI dan kode keluar), qwen38-engine stage 5 (server + WebUI, sudah ada)
- **Blocks:** SPEC-009 (rilis artefak "satu berkas" ke GitHub Release), PR #52 (eksperimen bahasa),
  PR #53 (integrasi forward Bonsai) — keduanya prasyarat agar executable fat membawa model **cepat**,
  bukan hanya model besar