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
- FlashAttention library tidak mendukung SM 6.1; FA bawaan llama.cpp perlu diuji (jangan diasumsikan)
- FP16 matmul tidak lebih cepat dari FP32 pada GP104 → jangan asumsikan FP16 speedup
- Dequantization overhead adalah cost nyata untuk GGUF quantized formats

## Batas Throughput Decode (teoritis, Qwen3.8-27B)

Decode memory-bandwidth-bound. Batas atas TPS ≈ bandwidth ÷ bobot yang dibaca per token. Layer di CPU memakai
bandwidth RAM (belum diukur; asumsi 20 GB/s single-channel atau 40 GB/s dual-channel), layer di GPU memakai
256 GB/s. Tabel lengkap ada di `spec/AGENTS/MASTER.md` (Memory Budget).

| Quant | Bobot di RAM (est.) | Batas atas TPS (20 / 40 GB/s) |
|---|---|---|
| Q4_K_M | ~10.0 GiB | ~1.9 / ~3.7 |
| IQ3_M | ~6.1 GiB | ~3.1 / ~6.1 |
| IQ3_XXS | ~5.1 GiB | ~3.6 / ~7.3 |

> ⚠️ Batas atas teoritis, bukan prediksi. Dequantization (terutama IQ-quant), attention, dan scheduling
> menurunkannya. Tidak ada angka di sini yang terukur.

## Tuas Optimasi yang Diketahui

| Tuas | Efek yang diharapkan | Status |
|---|---|---|
| Quant lebih kecil di layer CPU | Mengurangi bobot yang dibaca per token | Perlu diukur dampak kualitasnya |
| MTP speculative decoding (`--spec-type draft-mtp`) | Beberapa token diverifikasi per satu kali baca bobot | Belum diuji di build kita |
| Muat penuh di VRAM (strategi B) | Menghilangkan bottleneck RAM | Kualitas turun; ruang tipis di 1070 Ti |
| Auto-fit layer vs `-ngl` manual | Model card menganjurkan auto-fit | Belum diverifikasi |
| Vulkan vs CUDA | Hasil strategi B diperoleh dengan Vulkan | Belum diuji di Pascal |

## Panduan Parameter (titik awal untuk diukur)

```bash
# Hybrid, auto-fit
llama-cli -m Qwen3.8-27B-Q4_K_M.gguf -c 4096 -t <jumlah core fisik>

# Hybrid, manual
llama-cli -m Qwen3.8-27B-Q4_K_M.gguf -c 4096 -ngl 28 -ctk q8_0 -ctv q8_0 -t <jumlah core fisik>
```

> ⚠️ Belum diverifikasi: dukungan Gated DeltaNet di CUDA SM 6.1, apakah `-ctv` quantized butuh `-fa`,
> dan apakah FA bawaan llama.cpp berjalan di Pascal. Jangan menganggap `-fa` otomatis diabaikan.

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
