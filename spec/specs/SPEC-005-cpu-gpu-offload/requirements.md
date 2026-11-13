# Requirements — SPEC-005: CPU-GPU Offload Strategy

> Spec: SPEC-005 | Status: **Skeleton**
> Referensi: RFC-001, SPEC-002
> Priority: P2

---

> 📋 **Status: Skeleton**
>
> Spec ini fokus pada Qwen3-27B yang tidak muat penuh di VRAM 8 GB.
> Membutuhkan strategi offloading yang optimal antara GPU (256 GB/s GDDR5)
> dan CPU (40–45 GB/s DDR4).

---

## Problem Statement

Qwen3-27B (~13.5–16.4 GB tergantung quant) tidak bisa sepenuhnya dimuat ke VRAM 8 GB.
Diperlukan strategi CPU-GPU hybrid offloading yang memaksimalkan layer yang diproses
GPU sambil menjaga total RAM usage di bawah 16 GB.

---

## Functional Requirements

- **FR-01:** Qwen3-27B Q3_K_M dapat di-load dengan ~27 layer di GPU, ~25 layer di CPU
- **FR-02:** Inference menghasilkan output koheren meski dalam hybrid mode
- **FR-03:** Total RAM usage ≤ 15 GB saat inference Qwen3-27B Q3_K_M

---

## Non-Functional Requirements

- **NFR-01: Performance** — Decode TPS ≥ 2 tok/s untuk Qwen3-27B Q3_K_M hybrid
- **NFR-02: Memory** — VRAM ≤ 7.5 GB, RAM ≤ 15 GB
- **NFR-03: Compatibility** — Pascal SM 6.1

---

## Out of Scope

- Optimasi KV cache untuk 27B → SPEC-004 (apply lessons learned)
- Dynamic layer redistribution (terlalu kompleks untuk tahap ini)

---

## Success Criteria

- [ ] Qwen3-27B Q3_K_M berjalan dengan -ngl 27 tanpa OOM
- [ ] Decode TPS ≥ 2 tok/s diukur empiris
- [ ] RAM usage aktual ≤ 15 GB (tidak ada swap ke disk)
- [ ] Output koheren untuk 100+ token generation

---

## Dependencies

- **Requires:** SPEC-002 (Inference Baseline — llama.cpp build)
- **Blocks:** —
