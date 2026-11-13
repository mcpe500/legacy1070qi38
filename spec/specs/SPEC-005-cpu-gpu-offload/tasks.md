# Tasks — SPEC-005: CPU-GPU Offload Strategy

> Spec: SPEC-005 | Status: **Not Started** (bisa mulai setelah SPEC-002)
> Dependencies: SPEC-002 (llama.cpp build)
> Blocks: —

---

## Task List

- [ ] **Task 1: Download Qwen3-27B GGUF (Q3_K_M)**
  - Implementation: Download file dari HuggingFace; verifikasi ukuran ~13.5 GB; pastikan RAM cukup
  - Test: File ada dan bisa di-load tanpa error
  - Done when: Model dapat diinisialisasi oleh llama.cpp

- [ ] **Task 2: Verifikasi model architecture dari GGUF**
  - Implementation: Jalankan `llama-cli -m qwen3-27b-q3_k_m.gguf --no-warmup -n 0` dan catat layer count, KV heads, hidden dim dari output
  - Test: Layer count terverifikasi (expected ~52, bisa berbeda)
  - Done when: Semua ASSUMPTION tentang Qwen3-27B architecture ter-update di MASTER.md dan RFC-001

- [ ] **Task 3: Test baseline hybrid inference**
  - Implementation: Jalankan dengan `-ngl 27 --ctx-size 2048 --no-mmap`; monitor RAM dengan `htop`; catat VRAM dan TPS
  - Test: Tidak OOM; output koheren; TPS ≥ 2 tok/s
  - Done when: Angka baseline tercatat

- [ ] **Task 4: Sweep -ngl values**
  - Implementation: Test -ngl 20, 24, 27, 30, 35; catat TPS dan RAM untuk setiap nilai
  - Test: Ditemukan nilai -ngl optimal (highest TPS tanpa RAM OOM)
  - Done when: Tabel -ngl vs TPS vs RAM tersedia di handoff doc

- [ ] **Task 5: Test Q2_K sebagai alternatif**
  - Implementation: Download Q2_K; jalankan dengan -ngl 35; bandingkan TPS dan kualitas output vs Q3_K_M
  - Test: TPS ≥ 4 tok/s; kualitas output masih acceptable
  - Done when: Comparison report tersedia

- [ ] **Task N: Update status dan handoff**
  - Implementation: Update status, buat session doc, update HANDOFF-LOG.md
  - Done when: Review phase dapat dimulai
