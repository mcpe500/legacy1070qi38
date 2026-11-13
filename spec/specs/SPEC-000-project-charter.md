# SPEC-000: Project Charter — legacy1070qi38

> Status: **Active**
> Dibuat: Sesi Bootstrap (SESSION-001)
> Lisensi: Apache 2.0

---

## Identitas Proyek

| Field | Value |
|---|---|
| Nama | `legacy1070qi38` |
| Tagline | *Old Hardware, New Intelligence* |
| Tema | Resurrection — Kiroween 2026 |
| Jenis | Open-source LLM inference engine |
| Target Hardware | NVIDIA GTX 1070 Ti (8GB VRAM, Pascal SM 6.1) |
| System RAM | 16 GB DDR4 |
| Primary Backend | llama.cpp (CUDA build, SM 61) |

---

## Latar Belakang

Perkembangan AI semakin bergantung pada GPU modern dengan kapasitas memori besar yang tidak terjangkau oleh semua orang. Namun banyak GPU lama — seperti NVIDIA GTX 1070 Ti — masih memiliki kemampuan komputasi yang dapat dimanfaatkan untuk inference LLM modern.

Proyek ini berangkat dari pertanyaan sederhana:

> *Seberapa jauh kita bisa mendorong batas kemampuan hardware lama melalui rekayasa perangkat lunak?*

`legacy1070qi38` menjawab pertanyaan ini dengan fokus pada optimasi inference, quantization, dan manajemen memori yang efisien — alih-alih mengganti hardware.

---

## Target Model

### Qwen3-8B (Primary Target)

| Properti | Nilai |
|---|---|
| Parameter | ~8.19 miliar |
| Layers | ~36 transformer layers |
| Hidden dim | 4096 |
| KV heads (GQA) | 8 |
| Head dim | 128 |
| FFN intermediate | 28,672 |
| Vocabulary | 152,064 tokens |

> ⚠️ ASSUMPTION: Arsitektur diekstrapolasi dari Qwen2-7B/Qwen2.5-7B. Verifikasi ke model card resmi.

### Qwen3-27B (Stretch Target)

| Properti | Nilai |
|---|---|
| Parameter | ~27.4 miliar |
| Layers | ~52 transformer layers |
| Hidden dim | 5,120 |
| KV heads (GQA) | 8 |
| Head dim | 128 |
| FFN intermediate | ~27,648–32,768 |

> ⚠️ ASSUMPTION: Arsitektur diekstrapolasi. Verifikasi ke model card resmi.

---

## Hardware Constraints

### GPU: NVIDIA GTX 1070 Ti

| Spesifikasi | Nilai |
|---|---|
| Architecture | Pascal (GP104) |
| Compute Capability | SM 6.1 |
| VRAM | 8 GB GDDR5 |
| Memory Bandwidth | 256 GB/s |
| CUDA Cores | 2,432 |
| TDP | 180W |
| PCIe | 3.0 x16 |

### Keterbatasan Pascal yang Kritis

| Fitur | Status | Implikasi |
|---|---|---|
| Tensor Cores | ❌ Tidak ada | Semua matmul via CUDA FP32 cores |
| Flash Attention v1 | ❌ Butuh SM ≥ 7.5 | Gunakan standard SDPA kernel |
| Flash Attention v2 | ❌ Butuh SM ≥ 8.0 | Sama |
| Native INT8 (DP4A) | ❌ Tidak ada | GPTQ/AWQ tidak ada speedup |
| BF16 hardware | ❌ Tidak ada | Emulasi = lambat |
| FP16 fast path | ⚠️ GP104 tidak ada speedup | FP16 ≈ FP32 kecepatan |

### System RAM

| Spesifikasi | Nilai |
|---|---|
| Kapasitas | 16 GB DDR4 |
| Bandwidth (est.) | ~40–45 GB/s (dual-channel) |
| PCIe (practical) | ~12–13 GB/s |

---

## Memory Budget dan Feasibility

### Qwen3-8B (Full VRAM Inference)

| Quant | Weights | +KV (4096 ctx) | +KV (q8_0) | +Overhead | Total | Status |
|---|---|---|---|---|---|---|
| Q8_0 | 8.8 GB | 0.58 GB | 0.29 GB | 0.4 GB | 9.8 GB | ❌ OOM |
| Q6_K | 7.0 GB | 0.58 GB | 0.29 GB | 0.4 GB | 7.7 GB | ⚠️ Tight |
| **Q5_K_M** | **6.1 GB** | **0.58 GB** | **0.29 GB** | **0.4 GB** | **7.1 GB** | **✅ Recommended** |
| Q4_K_M | 5.0 GB | 0.58 GB | 0.29 GB | 0.4 GB | 6.0 GB | ✅ Comfortable |
| Q3_K_M | 4.1 GB | 0.58 GB | 0.29 GB | 0.4 GB | 5.1 GB | ✅ Comfortable |

### Qwen3-27B (Hybrid CPU-GPU Offloading)

| Quant | Model Size | RAM Total | GPU Layers | CPU Layers | Tok/s est. | Status |
|---|---|---|---|---|---|---|
| Q4_K_M | 16.4 GB | ~18 GB | ~22 | ~30 | 2–4 | ❌ RAM OOM |
| **Q3_K_M** | **13.5 GB** | **~15.5 GB** | **~27** | **~25** | **3–6** | **⚠️ Max Viable** |
| Q2_K | 10.4 GB | ~12.4 GB | ~35 | ~17 | 6–10 | ✅ Fits, quality↓ |

> ⚠️ ASSUMPTION: Semua angka di atas berdasarkan analisis teoritis. Harus diverifikasi empiris.

---

## Performance Targets

| Skenario | Minimum Acceptable | Target | Stretch |
|---|---|---|---|
| Qwen3-8B Q5_K_M, full GPU | 10 tok/s | 15 tok/s | 20+ tok/s |
| Qwen3-8B Q6_K, full GPU | 8 tok/s | 12 tok/s | 16+ tok/s |
| Qwen3-27B Q3_K_M, hybrid | 2 tok/s | 4 tok/s | 6+ tok/s |
| Qwen3-27B Q2_K, hybrid | 4 tok/s | 7 tok/s | 10+ tok/s |
| VRAM usage (8B, any config) | ≤8.0 GB | ≤7.5 GB | ≤7.0 GB |
| RAM usage (27B, Q3_K_M) | ≤16 GB | ≤15 GB | ≤14 GB |
| Output quality (8B Q5_K_M) | Koheren | Faktual | Setara FP16 |

---

## Arsitektur Teknis High-Level

```
legacy1070qi38
├── Backend: llama.cpp (CUDA build, SM 61)
│   ├── Quantization: GGUF (Q4_K_M – Q6_K untuk 8B, Q2_K – Q3_K_M untuk 27B)
│   ├── GPU Offload: --n-gpu-layers (partial untuk 27B)
│   └── KV Cache: quantized (q8_0 atau q4_0)
│
├── Inference Pipeline
│   ├── Qwen3-8B: Full VRAM inference
│   └── Qwen3-27B: Hybrid CPU-GPU offloading
│
└── Evaluation
    ├── Tokens per second (decode + prefill)
    ├── Memory usage (VRAM + RAM)
    ├── Time to first token
    └── Output quality (coherence + factuality)
```

---

## Approaches yang Dilarang

Lihat `spec/AGENTS/MASTER.md` bagian "Approaches yang DILARANG" untuk daftar lengkap.

Ringkasan: **GPTQ, AWQ, vLLM, TensorRT-LLM, Flash Attention, BF16** — semua tidak kompatibel dengan Pascal SM 6.1.

---

## Spec Roadmap

| Nomor | Nama | Status | Prioritas |
|---|---|---|---|
| SPEC-000 | Project Charter (dokumen ini) | ✅ Active | — |
| RFC-001 | Quantization Strategy | ✅ Accepted | — |
| SPEC-002 | Inference Baseline | 📝 Ready to implement | P0 |
| SPEC-003 | Memory Manager | 📋 Skeleton | P1 |
| SPEC-004 | KV Cache Optimizer | 📋 Skeleton | P1 |
| SPEC-005 | CPU-GPU Offload | 📋 Skeleton | P2 |
| SPEC-006 | Benchmark Suite | 📋 Skeleton | P1 |

---

## Success Criteria Proyek

Proyek dianggap sukses jika semua kriteria berikut terpenuhi:

- [ ] Qwen3-8B berjalan di GTX 1070 Ti dengan ≥10 tok/s pada Q5_K_M
- [ ] Qwen3-27B berjalan dengan hybrid offloading dan menghasilkan ≥2 tok/s pada Q3_K_M
- [ ] VRAM usage Qwen3-8B ≤7.5 GB (ada headroom)
- [ ] RAM usage Qwen3-27B ≤15 GB (tidak swap ke disk)
- [ ] Output model koheren dan dapat dievaluasi kualitasnya
- [ ] Proses setup terdokumentasi dan reproducible dalam satu README
- [ ] Benchmark suite menghasilkan angka terukur dan comparable

---

## Spirit dan Filosofi

> **"Old Hardware, New Intelligence"**

Inovasi bukan selalu tentang hardware paling mahal. Proyek ini membuktikan bahwa optimasi tingkat rendah, kreativitas engineering, dan pemahaman mendalam terhadap arsitektur hardware masih sangat berharga di tengah perkembangan AI yang cepat.

`legacy1070qi38` adalah tentang **aksesibilitas** — membuat inference AI modern dapat dijalankan oleh siapapun yang memiliki GPU generasi lama yang masih berfungsi.

---

*Lihat `spec/handoff/SESSION-001-bootstrap.md` untuk konteks pembuatan dokumen ini.*
*Lihat `spec/AGENTS/MASTER.md` untuk aturan dan instruksi agent.*
