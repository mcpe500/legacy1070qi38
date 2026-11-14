# Requirements — SPEC-005: CPU-GPU Offload untuk Qwen3.8-27B

> Spec: SPEC-005 | Status: **Skeleton (direvisi)**
> Referensi: RFC-001, SPEC-002
> Priority: P0 (ini inti goal proyek)

---

## Problem Statement

Qwen3.8-27B (11.6–18.7 GiB tergantung quant) tidak muat penuh di VRAM 8 GB. Diperlukan strategi yang
memaksimalkan bobot di VRAM, menjaga RAM puncak di bawah 15 GiB, dan mengukur TPS serta kualitas nyata.

---

## Functional Requirements

- **FR-01:** Qwen3.8-27B dimuat dengan sebagian layer di GPU, sisanya di CPU, pada minimal satu quant
- **FR-02:** Output koheren pada 100+ token
- **FR-03:** Dibandingkan dua cara offload: auto-fit dan `-ngl` manual
- **FR-04:** Dibandingkan minimal tiga quant (misalnya Q4_K_M, IQ3_M, IQ3_XXS)
- **FR-05:** Strategi B (muat penuh, ~7.7 GiB) dicoba dan hasilnya dicatat termasuk jika gagal
- **FR-06:** MTP speculative decoding dicoba dan hasilnya dicatat

---

## Non-Functional Requirements

- **NFR-01: Performance** — target di `MASTER.md` (tabel "Target Performa")
- **NFR-02: Memory** — VRAM tidak OOM, RAM puncak ≤ 15 GiB tanpa swap
- **NFR-03: Compatibility** — Pascal SM 6.1

---

## Out of Scope

- Redistribusi layer dinamis saat berjalan
- Input gambar dan video (`mmproj`)

---

## Success Criteria

- [ ] Satu konfigurasi Qwen3.8-27B berjalan stabil dengan TPS dan RAM puncak terukur
- [ ] Tabel perbandingan quant × cara offload tersedia di handoff
- [ ] Strategi B dan C punya hasil terukur (berhasil atau gagal dengan alasan)

---

## Dependencies

- **Requires:** SPEC-002 (build dan keputusan apakah CUDA jalan untuk Gated DeltaNet)
- **Blocks:** tidak ada
