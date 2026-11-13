# Phase Agent: Reviewer

> Baca `spec/AGENTS/MASTER.md` terlebih dahulu sebelum dokumen ini.

## Kapan Phase Ini Aktif

Phase Reviewer aktif ketika:
- Implementasi selesai dan diklaim ready
- Ada RFC baru yang perlu divalidasi sebelum Accepted
- Ada perubahan pada spec yang sudah Approved
- Ditemukan regresi performa atau kualitas

## Tanggung Jawab

Agent dalam phase ini bertugas **memverifikasi bahwa implementasi sesuai spec dan tidak melanggar constraints**.

Output utama phase ini:
1. Review report (pass/fail per requirement)
2. Performance verification (angka TPS vs target di MASTER.md)
3. Memory verification (penggunaan VRAM/RAM vs budget)
4. Rekomendasi: `APPROVED` | `NEEDS WORK` | `BLOCKED`

## Review Checklist

### Functional Review

Untuk setiap requirement di `requirements.md`:

- [ ] FR-01: [requirement] — **Status: PASS/FAIL** — Evidence: [command output atau screenshot]
- [ ] FR-02: ...
- [ ] NFR-01: Performance ≥ [target] — **Measured: X tok/s** — PASS/FAIL
- [ ] NFR-02: Memory ≤ [budget] — **Measured: X MB** — PASS/FAIL
- [ ] NFR-03: Pascal compatibility — **Verified on GTX 1070 Ti** — PASS/FAIL

### Code Review Checklist

- [ ] Tidak ada penggunaan forbidden approaches (GPTQ, AWQ, vLLM, BF16, Flash Attention)
- [ ] Ada comment untuk Pascal-specific decisions
- [ ] Error handling untuk OOM ada
- [ ] Memory budget tidak melebihi limits di MASTER.md
- [ ] Kode berjalan di Pascal (SM 6.1) — bukan hanya di GPU yang lebih modern
- [ ] Demo command berjalan dan menghasilkan output yang benar

### Spec Consistency Review

- [ ] Implementasi konsisten dengan RFC-001 (quantization strategy)
- [ ] Tidak ada kontradiksi dengan MASTER.md
- [ ] Tidak ada kontradiksi dengan spec yang sudah Implemented

## Review Output Format

```markdown
## Review Report — SPEC-NNN — [Tanggal]

**Reviewer phase:** Reviewer
**Verdict:** APPROVED / NEEDS WORK / BLOCKED

### Functional Requirements

| ID | Requirement | Status | Evidence |
|---|---|---|---|
| FR-01 | [req] | ✅ PASS | [command/output] |
| FR-02 | [req] | ❌ FAIL | [reason] |

### Performance Results

| Metric | Target | Measured | Status |
|---|---|---|---|
| Decode TPS | ≥10 tok/s | X tok/s | ✅/❌ |
| VRAM usage | ≤7500 MB | X MB | ✅/❌ |
| RAM usage | ≤4000 MB | X MB | ✅/❌ |

### Issues Found

1. **[CRITICAL/MAJOR/MINOR]**: [deskripsi masalah]
   - Lokasi: [file:line]
   - Rekomendasi: [solusi]

### Decision

**[APPROVED / NEEDS WORK / BLOCKED]**

Alasan: [justifikasi keputusan]

Jika NEEDS WORK, sebelum re-review:
- [ ] [action item 1]
- [ ] [action item 2]
```

## Kriteria BLOCKED

Reviewer mengeluarkan BLOCKED (bukan hanya NEEDS WORK) jika:

1. Implementasi menggunakan forbidden approach dari MASTER.md
2. Memory usage melebihi budget dan tidak ada fix yang jelas
3. Implementasi crash atau OOM pada hardware target (GTX 1070 Ti)
4. Ada security issue (misalnya arbitrary code execution dari model output)
5. Konflik fundamental dengan RFC yang sudah Accepted

## Referensi

- `spec/AGENTS/MASTER.md` — success criteria dan forbidden approaches
- `spec/AGENTS/roles/benchmarker.md` — cara mengukur performance
- `spec/specs/RFC-001-quantization-strategy.md` — baseline keputusan quantization
- `spec/handoff/HANDOFF-LOG.md` — log review results
