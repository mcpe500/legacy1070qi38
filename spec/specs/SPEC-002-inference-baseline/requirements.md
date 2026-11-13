# Requirements — SPEC-002: Inference Baseline

> Spec: SPEC-002 | Status: **Ready to Implement**
> Referensi: RFC-001 (Quantization Strategy), SPEC-000 (Project Charter)
> Priority: P0 — Ini adalah komponen pertama yang harus berjalan

---

## Problem Statement

Sebelum mengoptimalkan apapun, kita perlu **baseline yang berjalan**: llama.cpp ter-compile untuk Pascal SM 6.1 yang bisa menjalankan Qwen3-8B dengan GGUF quantization dan menghasilkan output yang koheren. Semua spec lain (memory management, KV cache optimization, benchmark suite) bergantung pada baseline ini.

---

## Functional Requirements

- **FR-01:** llama.cpp ter-compile dengan CUDA support untuk Pascal SM 6.1 (`-DCMAKE_CUDA_ARCHITECTURES=61`)
- **FR-02:** Model Qwen3-8B GGUF (Q5_K_M atau Q4_K_M) dapat di-load tanpa error
- **FR-03:** Inference menghasilkan output teks yang koheren (bukan garbage/nonsense)
- **FR-04:** Semua layer di-offload ke GPU (`-ngl 99`) secara default untuk Qwen3-8B Q5_K_M
- **FR-05:** KV cache quantization q8_0 aktif secara default
- **FR-06:** llama.cpp menampilkan statistik TPS (decode dan prefill) di output

---

## Non-Functional Requirements

- **NFR-01: Performance** — Decode TPS ≥ 10 tok/s untuk Qwen3-8B Q5_K_M pada GTX 1070 Ti
- **NFR-02: Memory (VRAM)** — Total VRAM usage ≤ 7.5 GB saat inference Qwen3-8B Q5_K_M, ctx 2048
- **NFR-03: Compatibility** — Berjalan pada Pascal SM 6.1 (GTX 1070 Ti); tidak menggunakan Flash Attention, GPTQ, AWQ, BF16
- **NFR-04: Stability** — Tidak crash atau OOM selama inference normal (100–500 token output)
- **NFR-05: Reproducibility** — Setup bisa diikuti dari README tanpa manual debugging

---

## Out of Scope (di SPEC-002 ini)

- Optimasi lanjutan (KV cache eviction, sliding window) → SPEC-004
- Qwen3-27B hybrid inference → SPEC-005
- Automated benchmark suite → SPEC-006
- Python wrapper/API → future spec
- Batched inference → future spec

---

## Success Criteria

- [ ] `cmake --build` selesai tanpa error, menghasilkan binary `llama-cli`
- [ ] `llama-cli --version` menampilkan CUDA dan SM 61 di output
- [ ] `llama-cli -m qwen3-8b-q5_k_m.gguf -ngl 99 -n 50` menghasilkan teks koheren
- [ ] Output llama-cli menampilkan baris seperti: `llama_perf_sampler_print: ... X tok/s`
- [ ] Measured decode TPS ≥ 10 tok/s
- [ ] `nvidia-smi` menunjukkan VRAM usage ≤ 7500 MB saat inference berjalan
- [ ] Tidak ada error CUDA OOM pada ctx-size 2048
