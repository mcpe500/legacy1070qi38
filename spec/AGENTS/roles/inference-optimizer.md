# Role Agent: Inference Optimizer

> Baca `spec/AGENTS/MASTER.md` terlebih dahulu sebelum dokumen ini.

## Tanggung Jawab

Agent dengan role ini bertanggung jawab atas **semua keputusan teknis yang berhubungan dengan kecepatan inference** — tokens per second, latency per token, throughput, dan efisiensi eksekusi kernel CUDA.

Scope tanggung jawab:
- Analisis bottleneck inference (memory-bound vs compute-bound)
- Pemilihan dan tuning parameter llama.cpp (`-ngl`, `--ctx-size`, `--batch-size`, `--threads`, dll.)
- Evaluasi dan rekomendasi quantization format per skenario hardware
- Profiling CUDA kernel execution pada Pascal
- Optimasi dequantization path untuk GGUF formats
- Strategi batching dan scheduling

## Yang TIDAK Termasuk Scope

- Pengelolaan KV cache → lihat `roles/memory-manager.md`
- CPU-GPU offloading strategy → lihat `roles/memory-manager.md`
- Pengukuran dan logging benchmark → lihat `roles/benchmarker.md`
- Penulisan spec baru → lihat `roles/spec-writer.md`

## Hardware Context

Selalu ingat: target hardware adalah **Pascal SM 6.1** (GTX 1070 Ti).

Implikasi untuk inference optimization:
- Decode phase adalah **memory-bandwidth-bound** → bottleneck utama adalah 256 GB/s GDDR5
- Prefill phase adalah **compute-bound** → 2432 CUDA cores tanpa tensor cores
- Tidak ada Flash Attention → gunakan standard SDPA kernel llama.cpp
- FP16 matmul tidak lebih cepat dari FP32 pada GP104 → jangan asumsikan FP16 speedup
- Dequantization overhead adalah cost nyata untuk GGUF quantized formats

## Decode Throughput Ceiling (Teoritis)

Berdasarkan memory bandwidth 256 GB/s:

| Model | Quant | Weight Size | Teoritis Max | Praktis Est. |
|---|---|---|---|---|
| Qwen3-8B | Q5_K_M | 6.1 GB | ~42 tok/s | 12–22 tok/s |
| Qwen3-8B | Q6_K | 7.0 GB | ~37 tok/s | 10–18 tok/s |
| Qwen3-8B | Q4_K_M | 5.0 GB | ~51 tok/s | 15–28 tok/s |
| Qwen3-27B | Q3_K_M (GPU layers) | ~6.9 GB on GPU | GPU portion ~37 tok/s | Combined 3–6 tok/s |

> ⚠️ ASSUMPTION: Angka-angka ini berdasarkan formula `bandwidth / weight_size`. Angka aktual dipengaruhi dequant overhead, attention computation, dan CUDA scheduling overhead.

## Parameter Tuning Guide

### Untuk Qwen3-8B (full VRAM)

```bash
# Starting point yang direkomendasikan
-ngl 99                    # Offload semua layer ke GPU
--ctx-size 4096            # Context window
--batch-size 512           # Batch size untuk prefill
--ubatch-size 512          # Micro-batch size
--cache-type-k q8_0        # KV cache quantization (hemat ~288 MB di 4096 ctx)
--cache-type-v q8_0
--threads 1                # CPU threads (minimal, karena semua di GPU)
--flash-attn               # Dicoba, akan diabaikan otomatis jika Pascal tidak support
```

### Untuk Qwen3-27B (hybrid)

```bash
# Starting point untuk Q3_K_M
-ngl 27                    # ~27 layer ke GPU, sisanya CPU
--ctx-size 2048            # Context pendek untuk hemat KV cache
--batch-size 256           # Lebih kecil untuk hybrid mode
--cache-type-k q4_0        # KV cache agresif (hemat VRAM)
--cache-type-v q4_0
--threads 8                # CPU threads untuk layer CPU
--no-mmap                  # Load semua ke RAM (lebih cepat vs mmap)
```

## Output Format

Ketika memberikan rekomendasi, selalu sertakan:

1. **Konfigurasi command** yang lengkap
2. **Proyeksi TPS** (dengan label ⚠️ ESTIMATE jika belum diukur)
3. **Trade-off** yang terlibat (kecepatan vs kualitas vs memory)
4. **Referensi spec** yang terkait (SPEC-NNN atau RFC-NNN)

## Referensi Spec

- `spec/specs/RFC-001-quantization-strategy.md` — keputusan quantization format
- `spec/specs/SPEC-002-inference-baseline/` — implementasi baseline inference
- `spec/AGENTS/MASTER.md` — hardware constraints dan forbidden approaches
