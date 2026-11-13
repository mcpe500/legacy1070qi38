# Tasks — SPEC-006: Benchmark Suite

> Spec: SPEC-006 | Status: **Not Started** (design bisa paralel dengan SPEC-002)
> Dependencies: SPEC-002 (untuk eksekusi benchmark)
> Blocks: —

---

## Task List

- [ ] **Task 1: Buat direktori benchmark dan prompt files**
  - Implementation: Buat `benchmarks/prompts/` dengan 3 file prompt standar. Pastikan token count sesuai (gunakan llama.cpp tokenizer untuk verifikasi)
  - Test: 3 file prompt ada; token count diverifikasi dengan tokenizer
  - Done when: Prompt files ready

- [ ] **Task 2: Buat benchmark runner script**
  - Implementation: Buat `benchmarks/run_benchmark.sh` yang menerima params: model path, quant, ngl, ctx-size. Jalankan 3x, parse TPS dari output llama.cpp, hitung average
  - Test: Script berjalan dan menghasilkan output yang bisa dibaca
  - Done when: Script bisa dijalankan dengan satu command

- [ ] **Task 3: Tambahkan VRAM dan RAM monitoring**
  - Implementation: Gunakan `nvidia-smi` dan `/proc/meminfo` untuk capture peak usage. Integrasikan ke script
  - Test: Report berisi VRAM dan RAM peak yang akurat
  - Done when: Memory metrics ada di output

- [ ] **Task 4: Jalankan full benchmark untuk semua skenario dari MASTER.md**
  - Implementation: Benchmark Qwen3-8B Q5_K_M, Q6_K, Q4_K_M (semua full GPU). Benchmark Qwen3-27B Q3_K_M dan Q2_K (hybrid). Simpan semua result
  - Test: Semua hasil ada; TPS matches atau exceeds targets di MASTER.md
  - Done when: Semua result tersimpan di `benchmarks/results/`

- [ ] **Task N: Update status dan handoff**
  - Done when: Semua success criteria di requirements.md terceklis
