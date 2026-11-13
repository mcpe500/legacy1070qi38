# Role Agent: Spec Writer

> Baca `spec/AGENTS/MASTER.md` terlebih dahulu sebelum dokumen ini.

## Tanggung Jawab

Agent dengan role ini bertanggung jawab atas **pembuatan, pemeliharaan, dan konsistensi dokumentasi spec** — memastikan semua spesifikasi lengkap, akurat, dan mudah diikuti oleh agent implementer.

Scope tanggung jawab:
- Membuat spec baru (feature spec dan RFC) dari brief/requirement
- Mereview dan mengupdate spec yang ada
- Menjaga konsistensi antar spec (tidak ada kontradiksi)
- Mengelola handoff log dan session documents
- Membuat dan memperbarui template

## Yang TIDAK Termasuk Scope

- Implementasi kode — Spec Writer hanya menulis dokumen
- Keputusan teknis unilateral — semua keputusan perlu justifikasi atau referensi riset
- Modifikasi `MASTER.md` tanpa escalation ke user

## Format Feature Spec

Setiap feature spec (`SPEC-NNN-nama/`) WAJIB memiliki 3 file:

### `requirements.md`

```markdown
# Requirements — [Nama Fitur]

> Spec: SPEC-NNN | Status: [Draft/Review/Approved/Implemented]
> Referensi: [RFC atau SPEC terkait]

## Problem Statement

[Masalah spesifik yang diselesaikan spec ini, dalam 2-3 kalimat]

## Functional Requirements

- FR-01: [requirement 1]
- FR-02: [requirement 2]
- ...

## Non-Functional Requirements

- NFR-01: Performance — [target angka spesifik]
- NFR-02: Memory — [batas VRAM/RAM]
- NFR-03: Compatibility — [harus berjalan pada Pascal SM 6.1]
- ...

## Out of Scope

- [Hal yang TIDAK akan dikerjakan di spec ini]
- ...

## Success Criteria

- [ ] [Kriteria terukur 1]
- [ ] [Kriteria terukur 2]
- ...
```

### `design.md`

```markdown
# Design — [Nama Fitur]

> Spec: SPEC-NNN | Status: [Draft/Review/Approved]

## Architecture Overview

[Diagram atau deskripsi high-level]

## Components

### [Komponen 1]
- Tanggung jawab: ...
- Interface: ...
- Dependencies: ...

## Configuration

[Parameter dan nilai default yang relevan]

## Edge Cases & Error Handling

| Kasus | Penanganan |
|---|---|
| [edge case 1] | [solusi] |

## Open Questions

- [ ] [Pertanyaan yang belum terjawab]
```

### `tasks.md`

```markdown
# Tasks — [Nama Fitur]

> Spec: SPEC-NNN | Status: [Not Started/In Progress/Done]

## Task List

- [ ] Task 1: [deskripsi singkat]
  - Implementation: [panduan implementasi]
  - Test: [apa yang harus diverifikasi]
  - Done when: [definisi selesai]

- [ ] Task 2: ...

## Dependencies

- Requires: [SPEC-NNN atau RFC-NNN yang harus selesai dulu]
- Blocks: [SPEC-NNN yang menunggu spec ini]
```

## Format RFC

RFC (`RFC-NNN-nama.md`) untuk keputusan arsitektur:

```markdown
# RFC-NNN: [Judul Keputusan]

> Status: [Proposed/Accepted/Superseded by RFC-XXX]
> Tanggal: [YYYY-MM-DD]
> Penulis: [role yang menulis]

## Summary

[Keputusan dalam 1-2 kalimat]

## Motivation

[Mengapa keputusan ini perlu dibuat? Masalah apa yang dipecahkan?]

## Design

[Detail teknis keputusan]

## Trade-offs

| Pro | Con |
|---|---|
| [keuntungan 1] | [kerugian 1] |

## Rejected Alternatives

### [Alternatif 1]
Alasan ditolak: ...

### [Alternatif 2]  
Alasan ditolak: ...

## References

- [Link atau referensi yang relevan]
```

## Konvensi Penomoran

```
SPEC-000  → Project Charter (satu kali, tidak berulang)
SPEC-001  → (reserved untuk future core spec)
SPEC-002  → Inference Baseline
SPEC-003  → Memory Manager
SPEC-004  → KV Cache Optimizer
SPEC-005  → CPU-GPU Offload
SPEC-006  → Benchmark Suite
SPEC-007+ → Spec baru berikutnya

RFC-001   → Quantization Strategy
RFC-002+  → RFC baru berikutnya
```

## Checklist Review Spec

Sebelum menandai spec sebagai `Approved`, verifikasi:

- [ ] Semua requirement memiliki success criteria yang terukur
- [ ] Tidak ada kontradiksi dengan `MASTER.md`
- [ ] Tidak ada kontradiksi dengan RFC yang sudah Accepted
- [ ] Semua asumsi diberi label ⚠️ ASSUMPTION
- [ ] Ada referensi ke spec/RFC terkait
- [ ] Task list cukup granular (tiap task ≤ 1 hari kerja)
- [ ] Dependencies antar spec terdokumentasi

## Referensi

- `spec/specs/_TEMPLATE-feature-spec/` — template lengkap untuk spec baru
- `spec/AGENTS/MASTER.md` — aturan global
- `spec/handoff/HANDOFF-LOG.md` — log perubahan spec
