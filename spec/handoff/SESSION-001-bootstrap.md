# SESSION-001: Bootstrap — Spec System Setup

> ⚠️ **Koreksi (setelah sesi ini):** keputusan dan angka di dokumen ini memakai model yang salah (Qwen3-8B dan
> Qwen3-27B transformer biasa). Target sebenarnya Qwen3.8-27B (hibrida, 64 layer). Dokumen ini dibiarkan sebagai
> catatan historis. Versi benar ada di `MASTER.md`, `RFC-001`, dan entri `[CORRECTION]` di `HANDOFF-LOG.md`.

> Tanggal: Bootstrap session (awal proyek)
> Status: **Complete**
> Agent: Planning + Execution (Kiro)
> Dibuat dengan: Kiro CLI (`kiro-cli chat`)

---

## Tujuan Sesi Ini

Menyiapkan seluruh struktur dokumentasi dan spec-driven development framework untuk proyek `legacy1070qi38` dari nol. Sesi ini bukan implementasi kode — fokus pada fondasi dokumentasi, template, aturan agent, dan spec awal berdasarkan analisis teknis nyata.

---

## Konteks Awal Sesi

**Hardware yang dimiliki:**
- GPU: NVIDIA GTX 1070 Ti (8 GB VRAM, Pascal SM 6.1)
- RAM: 16 GB DDR4

**Model yang ingin dijalankan:**
- Qwen3-8B (primary target)
- Qwen3-27B (stretch target — ambisius mengingat RAM 16 GB)

**Pertanyaan utama yang ingin dijawab:**
- Seberapa jauh GPU generasi lama bisa didorong untuk menjalankan LLM modern?
- Teknik apa yang paling efektif: quantization apa, offloading strategy mana?

---

## Research yang Dilakukan

Sesi ini melakukan analisis teknis mendalam melalui context-gathering agent yang menghasilkan:

### Hardware Analysis: GTX 1070 Ti (Pascal SM 6.1)

**Keterbatasan kritis yang ditemukan:**
1. **Tidak ada Tensor Cores** — semua matmul via CUDA FP32 cores, tidak ada 8× speedup FP16
2. **Tidak ada Flash Attention** — requires SM ≥ 7.5 (v1) atau SM ≥ 8.0 (v2)
3. **Tidak ada native INT8** — GPTQ/AWQ tidak ada benefit di Pascal
4. **FP16 di GP104 tidak lebih cepat dari FP32** — tidak seperti GP100

**Yang tersedia:**
- CUDA core FP32 matmul
- llama.cpp GGUF dequantization kernels untuk SM 6.1
- 256 GB/s GDDR5 bandwidth (decode adalah memory-bandwidth-bound)

### Model Architecture Analysis (ASSUMPTIONS — perlu verifikasi)

| Model | Layers | KV heads | Hidden dim | FFN dim |
|---|---|---|---|---|
| Qwen3-8B | ~36 | 8 | 4096 | 28,672 |
| Qwen3-27B | ~52 | 8 | 5120 | ~27,648–32,768 |

> ⚠️ Semua angka di atas diekstrapolasi dari Qwen2/Qwen2.5 family. Verifikasi dari model card diperlukan.

### Memory Feasibility Analysis

**Qwen3-8B — dapat dimuat penuh ke VRAM 8 GB:**
- Q5_K_M (6.1 GB) + KV q8_0 (0.29 GB) + overhead (0.4 GB) = **~7.1 GB** ✅ Optimal
- Q6_K (7.0 GB) + KV q8_0 (0.29 GB) + overhead (0.4 GB) = **~7.7 GB** ⚠️ Tight di 2048 ctx

**Qwen3-27B — harus hybrid offloading:**
- Q4_K_M (~16.4 GB) **melebihi 16 GB RAM** ❌ Tidak feasible
- Q3_K_M (~13.5 GB) + overhead → ~15.5 GB total ⚠️ Marginal, perlu verifikasi empiris
- Q2_K (~10.4 GB) + overhead → ~12.4 GB total ✅ Aman, kualitas menurun

---

## Keputusan yang Dibuat

### DECISION-001: Primary backend = llama.cpp (CUDA SM 61)
- **Alasan:** Satu-satunya backend yang support Pascal SM 6.1 dengan GGUF dequant kernels, partial GPU offload, dan KV cache quantization
- **Ditolak:** GPTQ (no INT8), AWQ (no INT8), vLLM (SM 8.0+), TensorRT-LLM (no Pascal optimization)
- **Dicatat di:** `spec/specs/RFC-001-quantization-strategy.md`

### DECISION-002: Quantization format = GGUF Q5_K_M (8B) dan Q3_K_M (27B)
- **Alasan 8B:** Q5_K_M ~7.1 GB total VRAM — nyaman di 8 GB dengan headroom untuk KV cache
- **Alasan 27B:** Q3_K_M adalah max viable yang muat di 16 GB RAM. Q4_K_M tidak muat.
- **Dicatat di:** `spec/specs/RFC-001-quantization-strategy.md`

### DECISION-003: KV cache quantization wajib aktif
- **Alasan:** Menghemat 144–288 MB VRAM yang krusial di hardware ini
- **Default:** q8_0 untuk Qwen3-8B, q4_0 untuk Qwen3-27B hybrid
- **Dicatat di:** `spec/specs/RFC-001-quantization-strategy.md`

### DECISION-004: Struktur spec menggunakan hybrid (feature spec + RFC)
- **Feature spec:** Subfolder `SPEC-NNN-nama/` dengan requirements.md + design.md + tasks.md
- **RFC:** File tunggal `RFC-NNN-nama.md` untuk keputusan arsitektur besar
- **Dicatat di:** `spec/AGENTS/MASTER.md`

### DECISION-005: AGENTS hirarkis dengan MASTER.md + roles/ + phases/
- **Master:** Aturan global, hardware constraints, forbidden approaches
- **Roles:** Inference Optimizer, Memory Manager, Benchmarker, Spec Writer
- **Phases:** Researcher, Implementer, Reviewer
- **Dicatat di:** `spec/AGENTS/MASTER.md`

---

## Output Sesi Ini

### File yang Dibuat

```
spec/
├── AGENTS/
│   ├── MASTER.md                      ✅ Fully written
│   ├── KIRO-USAGE.md                  ✅ Fully written
│   ├── roles/
│   │   ├── inference-optimizer.md     ✅ Fully written
│   │   ├── memory-manager.md          ✅ Fully written
│   │   ├── benchmarker.md             ✅ Fully written
│   │   └── spec-writer.md             ✅ Fully written
│   └── phases/
│       ├── researcher.md              ✅ Fully written
│       ├── implementer.md             ✅ Fully written
│       └── reviewer.md                ✅ Fully written
├── specs/
│   ├── SPEC-000-project-charter.md    ✅ Fully written
│   ├── RFC-001-quantization-strategy.md ✅ Fully written
│   ├── SPEC-002-inference-baseline/
│   │   ├── requirements.md            ✅ Fully written
│   │   ├── design.md                  ✅ Fully written
│   │   └── tasks.md                   ✅ Fully written
│   ├── SPEC-003-memory-manager/       ✅ Skeleton
│   ├── SPEC-004-kv-cache-optimizer/   ✅ Skeleton
│   ├── SPEC-005-cpu-gpu-offload/      ✅ Partially written (27B design known)
│   ├── SPEC-006-benchmark-suite/      ✅ Skeleton + planned structure
│   └── _TEMPLATE-feature-spec/       ✅ Template siap pakai
└── handoff/
    ├── HANDOFF-LOG.md                 ✅ (dokumen ini dibuat bersamaan)
    ├── SESSION-001-bootstrap.md       ✅ (dokumen ini)
    └── _TEMPLATE-session.md          ✅ Template sesi baru
```

---

## Open Questions dari Sesi Ini

| ID | Pertanyaan | Priority | Target |
|---|---|---|---|
| OQ-001 | Berapa layer count Qwen3-8B yang sebenarnya? | High | SPEC-002 Task 2 |
| OQ-002 | Berapa layer count Qwen3-27B yang sebenarnya? | High | SPEC-005 Task 2 |
| OQ-003 | Berapa KV head count Qwen3 yang sebenarnya? | High | SPEC-002 Task 2 |
| OQ-004 | Apakah KV q4_0 menurunkan kualitas output? | Medium | SPEC-004 |
| OQ-005 | Apakah RAM 16 GB cukup untuk Q3_K_M 27B tanpa swap? | Critical | SPEC-005 Task 3 |
| OQ-006 | Berapa TPS aktual untuk masing-masing skenario? | High | SPEC-002 Task 5 |
| OQ-007 | Apakah llama.cpp terbaru ada regresi di Pascal? | Medium | SPEC-002 Task 1 |
| OQ-008 | Berapa ctx-size maksimum sebelum OOM untuk Q5_K_M? | Medium | SPEC-002 Task 6 |

---

## Next Steps

Setelah sesi ini, urutan pengerjaan yang direkomendasikan:

1. **SPEC-002: Inference Baseline** (P0) — mulai dari sini
   - Build llama.cpp untuk Pascal SM 61
   - Download Qwen3-8B Q5_K_M
   - Jalankan inference pertama
   - Ukur TPS baseline
   - Jawab OQ-001, OQ-003, OQ-006, OQ-007, OQ-008

2. **SPEC-006: Benchmark Suite** (P1) — bisa mulai design paralel
   - Buat prompt standar dan benchmark script

3. **SPEC-005: CPU-GPU Offload** (P2) — setelah SPEC-002
   - Download Qwen3-27B Q3_K_M
   - Test hybrid inference
   - Jawab OQ-002, OQ-005

4. **SPEC-003 dan SPEC-004** (P1) — isi setelah baseline data tersedia

---

## Catatan untuk Agent Berikutnya

Jika kamu membaca dokumen ini sebagai agent yang baru masuk:

1. Mulai dari `spec/AGENTS/MASTER.md` — baca hardware constraints dan forbidden approaches
2. Cek `spec/handoff/HANDOFF-LOG.md` untuk state terkini (mungkin ada update setelah sesi ini)
3. Spec pertama yang harus dikerjakan: `spec/specs/SPEC-002-inference-baseline/`
4. Semua angka TPS dan memory adalah **ESTIMASI** — belum ada pengukuran empiris
5. Qwen3 architecture masih ASSUMPTION — verifikasi dari model card adalah prioritas pertama

---

*Sesi ini dibuat menggunakan Kiro CLI (`kiro-cli chat`) dengan planning + execution agent.*
*Lihat `spec/AGENTS/KIRO-USAGE.md` untuk dokumentasi penggunaan Kiro dalam proyek ini.*
