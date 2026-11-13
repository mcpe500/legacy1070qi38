# Tasks — SPEC-002: Inference Baseline

> Spec: SPEC-002 | Status: **Not Started**
> Dependencies: Tidak ada (ini adalah spec pertama)
> Blocks: SPEC-003, SPEC-004, SPEC-005, SPEC-006

---

## Task List

- [ ] **Task 1: Setup environment dan verifikasi hardware**
  - Implementation: Jalankan `nvidia-smi`, catat GPU model, VRAM, driver version. Jalankan `nvcc --version`, catat CUDA version. Pastikan Pascal terdeteksi (`Device 0: "NVIDIA GeForce GTX 1070 Ti"`)
  - Test: `nvidia-smi` menampilkan GTX 1070 Ti dengan 8192 MB VRAM; `nvcc --version` menampilkan versi CUDA yang terinstall
  - Done when: Semua info environment tercatat di SESSION-002 handoff doc

- [ ] **Task 2: Clone llama.cpp dan compile untuk Pascal SM 61**
  - Implementation: `git clone https://github.com/ggerganov/llama.cpp`, catat commit hash. Run cmake dengan `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61`. Build dengan `cmake --build build -j$(nproc)`
  - Test: Build selesai tanpa error. `./build/bin/llama-cli --version` output mengandung "CUDA"
  - Done when: Binary `llama-cli` ada di `build/bin/` dan bisa dijalankan

- [ ] **Task 3: Download model Qwen3-8B GGUF**
  - Implementation: Download `qwen3-8b-q5_k_m.gguf` dari HuggingFace ke `./models/`. Verifikasi ukuran file (~6.1 GB). Catat model card URL dan tanggal download
  - Test: File ada, ukuran ~6.1 GB, tidak corrupt (cek dengan `./llama-cli -m ... --no-warmup -n 0` yang harus selesai tanpa error)
  - Done when: Model dapat di-load oleh llama.cpp tanpa error

- [ ] **Task 4: Jalankan inference pertama dan verifikasi output**
  - Implementation: Jalankan command baseline dari design.md. Catat full output termasuk stats line
  - Test: Output teks koheren (bukan random characters). Stats line menampilkan tok/s
  - Done when: Bisa menghasilkan 50+ token output yang dapat dibaca

- [ ] **Task 5: Ukur dan catat baseline TPS**
  - Implementation: Jalankan 3x benchmark run dengan prompt standar (lihat `roles/benchmarker.md`). Catat: decode TPS, prefill TPS, VRAM usage, RAM usage, ctx-size
  - Test: Decode TPS ≥ 10 tok/s pada minimal 2 dari 3 run. VRAM ≤ 7500 MB
  - Done when: Angka tercatat di `spec/handoff/HANDOFF-LOG.md` dengan format benchmark result

- [ ] **Task 6: Test context size limits**
  - Implementation: Coba bertahap: ctx 1024, 2048, 3072, 4096. Catat VRAM usage dan TPS untuk masing-masing. Identify ctx maksimum sebelum OOM
  - Test: Semua ctx yang dicoba tidak crash. Ditemukan batas VRAM
  - Done when: Tabel ctx_size vs VRAM vs TPS tercatat di handoff doc; open question di design.md ter-update

- [ ] **Task 7: Update status dan handoff**
  - Implementation: Update status spec dari "Not Started" → "Implemented" di file ini. Buat `spec/handoff/SESSION-002-inference-baseline.md` dengan hasil semua pengukuran. Update `spec/handoff/HANDOFF-LOG.md`
  - Test: Semua success criteria di requirements.md terceklis
  - Done when: Review phase dapat dimulai dengan informasi yang cukup

---

## Notes

- Task 1–3 bisa dilakukan tanpa GPU (preparation)
- Task 4–6 membutuhkan GTX 1070 Ti dan model file
- Jika TPS di bawah target di Task 5: lihat `roles/inference-optimizer.md` untuk troubleshooting sebelum escalate
- Jika OOM di Task 6 pada ctx < 2048: update MASTER.md dan RFC-001 dengan temuan baru
