# Requirements — SPEC-006: Benchmark Suite

> Spec: SPEC-006 | Status: **Skeleton**
> Referensi: RFC-001, SPEC-002
> Priority: P1 (mulai bisa didesign paralel dengan SPEC-002)

---

> 📋 **Status: Skeleton**
>
> Benchmark suite adalah komponen evaluasi. Design bisa dimulai paralel dengan
> implementasi SPEC-002, tapi eksekusi benchmark membutuhkan SPEC-002 selesai.

---

## Problem Statement

Tanpa measurement yang konsisten dan reproducible, kita tidak tahu apakah optimasi
memberikan hasil nyata. Benchmark suite menyediakan tooling untuk mengukur TPS,
memory usage, dan kualitas output secara standar, sehingga hasil bisa dibandingkan
antar konfigurasi dan waktu.

---

## Functional Requirements

- **FR-01:** Script yang menjalankan benchmark TPS (decode + prefill) untuk konfigurasi yang diberikan
- **FR-02:** Script yang mengukur VRAM dan RAM peak usage selama inference
- **FR-03:** Prompt standar yang digunakan konsisten di semua benchmark run
- **FR-04:** Output benchmark tersimpan dalam format yang mudah dibandingkan (JSON atau Markdown)
- **FR-05:** Benchmark berjalan 3x per konfigurasi dan melaporkan rata-rata

---

## Non-Functional Requirements

- **NFR-01: Reproducibility** — Benchmark yang dijalankan dua kali dengan konfigurasi sama menghasilkan angka dalam ±10% variance
- **NFR-02: Minimal overhead** — Benchmark script tidak menambah overhead yang mengubah TPS secara signifikan
- **NFR-03: Compatibility** — Berjalan di Linux dengan bash; tidak membutuhkan dependensi Python kecuali untuk parsing output

---

## Out of Scope

- Automated CI/CD benchmark runner
- Web dashboard untuk visualisasi
- Benchmark selain pada hardware target (GTX 1070 Ti)

---

## Success Criteria

- [ ] Satu command menjalankan full benchmark dan menghasilkan report
- [ ] Report berisi: TPS (avg dari 3 run), VRAM peak, RAM peak, ctx-size, model, quant
- [ ] Benchmark report tersimpan di `benchmarks/results/` dengan timestamp
- [ ] Variance antar 3 run ≤ 10% untuk TPS

---

## Dependencies

- **Requires:** SPEC-002 (llama.cpp build dan model tersedia)
- **Blocks:** —
