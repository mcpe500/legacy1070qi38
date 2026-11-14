# HANDOFF LOG — legacy1070qi38

> File ini adalah continuous log dari semua sesi dan keputusan penting proyek.
> **Append-only** — jangan hapus atau edit entri lama; hanya tambahkan entri baru di bawah.
> Format: `[YYYY-MM-DD] [SESSION-NNN] [TYPE] Deskripsi singkat`

---

## Cara Membaca Log Ini

- **[SESSION]** — Sesi kerja baru dimulai; link ke file SESSION-NNN yang detail
- **[DECISION]** — Keputusan arsitektur atau teknis dibuat
- **[FINDING]** — Temuan baru dari eksperimen atau research
- **[CONFLICT]** — Ditemukan konflik antar spec; perlu resolusi
- **[OOM]** — Ditemukan OOM event; catat kondisi dan workaround
- **[UPDATE]** — Spec atau RFC di-update
- **[OPEN]** — Open question baru yang perlu dijawab
- **[RESOLVED]** — Open question terjawab; referensikan ke evidence

---

## Index Sesi

| Session | Tanggal | Deskripsi | Status |
|---|---|---|---|
| SESSION-001 | Bootstrap | Setup spec system, analisis hardware, keputusan teknis awal | ✅ Complete |

---

## Log Entries

### SESSION-001 — Bootstrap

```
[Bootstrap] [SESSION-001] Sesi pertama: setup spec-driven development framework
  → Lihat: spec/handoff/SESSION-001-bootstrap.md
```

```
[Bootstrap] [DECISION] Backend: llama.cpp dengan CUDA SM 61 dipilih sebagai satu-satunya backend
  → Alasan: satu-satunya backend yang support Pascal SM 6.1 secara eksplisit dengan GGUF dequant kernels
  → Lihat: spec/specs/RFC-001-quantization-strategy.md
```

```
[Bootstrap] [DECISION] Quantization Qwen3-8B: Q5_K_M sebagai primary, Q4_K_M sebagai fallback
  → Alasan: Q5_K_M ~7.1 GB total VRAM (dengan KV q8_0, 4096 ctx) — nyaman di 8 GB
  → Lihat: spec/specs/RFC-001-quantization-strategy.md
```

```
[Bootstrap] [DECISION] Quantization Qwen3-27B: Q3_K_M sebagai max viable, Q2_K sebagai safe fallback
  → Alasan: Q4_K_M ~16.4 GB melebihi 16 GB RAM; Q3_K_M ~15.5 GB total (marginal)
  → Lihat: spec/specs/RFC-001-quantization-strategy.md
```

```
[Bootstrap] [DECISION] KV cache quantization: q8_0 default untuk 8B, q4_0 untuk 27B hybrid
  → Alasan: Hemat 144–288 MB VRAM tanpa signifikan impact kualitas (perlu verifikasi empiris)
  → Lihat: spec/specs/RFC-001-quantization-strategy.md
```

```
[Bootstrap] [OPEN] Berapa layer count Qwen3-8B dan Qwen3-27B yang sebenarnya?
  → Status: UNRESOLVED
  → Akan dijawab di: SPEC-002 Task 2 dan SPEC-005 Task 2
```

```
[Bootstrap] [OPEN] Apakah KV cache q4_0 memengaruhi kualitas output secara signifikan?
  → Status: UNRESOLVED
  → Akan dijawab di: SPEC-004 (KV Cache Optimizer)
```

```
[Bootstrap] [OPEN] Apakah system RAM 16 GB cukup untuk Q3_K_M 27B tanpa swap?
  → Status: UNRESOLVED
  → Akan dijawab di: SPEC-005 Task 3
```

```
[Bootstrap] [UPDATE] Spec system bootstrap selesai:
  - spec/AGENTS/MASTER.md ✅
  - spec/AGENTS/KIRO-USAGE.md ✅
  - spec/AGENTS/roles/ (4 files) ✅
  - spec/AGENTS/phases/ (3 files) ✅
  - spec/specs/SPEC-000-project-charter.md ✅
  - spec/specs/RFC-001-quantization-strategy.md ✅
  - spec/specs/SPEC-002-inference-baseline/ (3 files) ✅
  - spec/specs/SPEC-003-memory-manager/ (skeleton) ✅
  - spec/specs/SPEC-004-kv-cache-optimizer/ (skeleton) ✅
  - spec/specs/SPEC-005-cpu-gpu-offload/ (skeleton) ✅
  - spec/specs/SPEC-006-benchmark-suite/ (skeleton) ✅
  - spec/specs/_TEMPLATE-feature-spec/ ✅
  - spec/handoff/ (dokumen ini) ✅
```

---

## Open Questions Aktif

Daftar open questions yang belum terjawab (update saat resolved):

| ID | Pertanyaan | Akan dijawab di | Status |
|---|---|---|---|
| OQ-001 | Berapa layer count Qwen3-8B yang sebenarnya? | SPEC-002 Task 2 | ❓ Open |
| OQ-002 | Berapa layer count Qwen3-27B yang sebenarnya? | SPEC-005 Task 2 | ❓ Open |
| OQ-003 | Berapa KV head count Qwen3 yang sebenarnya? | SPEC-002 Task 2 | ❓ Open |
| OQ-004 | Apakah KV q4_0 menurunkan kualitas output secara signifikan? | SPEC-004 | ❓ Open |
| OQ-005 | Apakah RAM 16 GB cukup untuk Q3_K_M 27B tanpa swap? | SPEC-005 Task 3 | ❓ Open |
| OQ-006 | Berapa TPS aktual (bukan estimasi) untuk masing-masing skenario? | SPEC-002 Task 5 | ❓ Open |
| OQ-007 | Apakah llama.cpp terbaru ada regresi di Pascal? | SPEC-002 Task 2 | ❓ Open |
| OQ-008 | Berapa ctx-size maksimum sebelum VRAM OOM untuk Q5_K_M? | SPEC-002 Task 6 | ❓ Open |

---

*Log dibuat pada sesi bootstrap. Append entri baru di bawah bagian "Log Entries" setelah header SESSION yang sesuai.*

### Post-bootstrap — Custom MCP

```
[Post-bootstrap] [OPEN] Desain MCP server kustom (4 kandidat: budget, hw, llama, spec)
  → Status: PROPOSED, belum diputuskan
  → Lihat: spec/specs/RFC-002-custom-mcp-servers.md
  → Diskusi: GitHub issues #1 (umbrella), #2 (A), #3 (B), #4 (C), #5 (D)
  → Pertanyaan kunci: timeout tool call di Kiro, llama-cli vs llama-server, 1 vs 4 server
```

```
[Post-bootstrap] [DECISION] RFC-002 direvisi: 4 MCP server diganti satu CLI (tools/legacy.py)
  → Alasan: lebih mudah dipasang dan dipakai orang lain; MCP menjadi pembungkus opsional
  → File RFC diganti nama: RFC-002-custom-mcp-servers.md → RFC-002-cli-tooling.md
  → Entri OPEN di atas yang menyebut 4 MCP server digantikan oleh entri ini
  → Status: masih PROPOSED; diskusi di GitHub issues #1..#5
```
