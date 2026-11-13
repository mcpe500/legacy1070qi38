# RFC-001: Quantization Strategy untuk Pascal SM 6.1

> Status: **Accepted**
> Tanggal: Sesi Bootstrap (SESSION-001)
> Penulis: Researcher Phase (bootstrap analysis)
> Berlaku untuk: Semua spec yang menyentuh quantization format

---

## Summary

Untuk hardware target Pascal SM 6.1 (GTX 1070 Ti), proyek ini menggunakan **GGUF format dengan llama.cpp** sebagai satu-satunya backend inference yang direkomendasikan. Quantization optimal adalah **Q5_K_M untuk Qwen3-8B** dan **Q3_K_M (atau Q2_K jika RAM terbatas) untuk Qwen3-27B**.

---

## Motivation

GTX 1070 Ti memiliki keterbatasan hardware yang signifikan dibanding GPU modern:
- Tidak ada Tensor Cores (mulai Volta)
- Tidak ada native INT8 acceleration (DP4A)
- Tidak ada BF16 hardware support
- Flash Attention tidak bisa berjalan (membutuhkan SM ≥ 7.5)

Ini membuat banyak inference framework modern (vLLM, GPTQ, AWQ) tidak memberikan manfaat di hardware ini. Pemilihan quantization format yang salah bisa menghasilkan performa lebih lambat dari yang seharusnya, atau bahkan gagal sama sekali.

---

## Decision: GGUF via llama.cpp

### Alasan Pemilihan

| Kriteria | GGUF + llama.cpp | GPTQ + AutoGPTQ | AWQ | vLLM |
|---|---|---|---|---|
| Pascal SM 6.1 support | ✅ Eksplisit | ⚠️ No benefit | ⚠️ No benefit | ❌ Tidak support |
| Dequant CUDA kernels | ✅ Ada untuk Q4_K–Q6_K | — | — | — |
| Partial GPU offload | ✅ --n-gpu-layers | ❌ Tidak ada | ❌ Tidak ada | ❌ |
| KV cache quantization | ✅ --cache-type-k | ❌ | ❌ | ❌ |
| Memory efficiency | ✅ GGUF k-quants | ⚠️ Hanya weight | ⚠️ Hanya weight | — |
| Community support | ✅ Aktif, besar | ⚠️ Menurun | ⚠️ Terbatas | — |

---

## Quantization Format per Model

### Qwen3-8B: Rekomendasi Q5_K_M

**Primary choice: Q5_K_M**
- File size: ~6.1 GB
- Total VRAM (4096 ctx, KV q8_0): ~7.1 GB → **muat di 8 GB** ✅
- Expected TPS: 12–22 tok/s
- Quality: Near-identical dengan FP16 untuk sebagian besar tasks

**Fallback: Q4_K_M**
- File size: ~5.0 GB
- Total VRAM (4096 ctx, KV q8_0): ~6.0 GB → **nyaman di 8 GB** ✅
- Expected TPS: 15–28 tok/s (lebih cepat karena ukuran lebih kecil)
- Quality: Slight degradation pada tugas nuanced reasoning

**High quality (ctx pendek): Q6_K**
- File size: ~7.0 GB
- Total VRAM (2048 ctx, KV q8_0): ~7.7 GB → **muat** ✅ (4096 ctx: terlalu ketat)
- Expected TPS: 10–18 tok/s
- Quality: Sangat dekat dengan FP16

**Tidak direkomendasikan untuk 8B:**
- Q8_0: 8.8 GB — tidak muat di VRAM 8 GB
- FP16/BF16: 16.5 GB — tidak muat

### Qwen3-27B: Rekomendasi Q3_K_M

**Constraint utama:** Total memory (RAM + VRAM) = 24 GB. Qwen3-27B Q4_K_M (~16.4 GB) **sudah melebihi 16 GB RAM** sebelum sistem operasi dan runtime dihitung.

**Primary choice: Q3_K_M**
- File size: ~13.5 GB
- RAM total (dengan overhead): ~15.5 GB → **marginal** ⚠️
- GPU layers: ~27 dari ~52 total
- CPU layers: ~25
- Expected TPS: 3–6 tok/s
- Quality: Acceptable, slight degradation pada reasoning

> ⚠️ WARNING: Q3_K_M pada sistem dengan RAM 16 GB membutuhkan semua aplikasi lain ditutup. Desktop environment bisa menambah 0.5–1 GB overhead.

**RAM-safe choice: Q2_K**
- File size: ~10.4 GB
- RAM total (dengan overhead): ~12.4 GB → **aman** ✅
- GPU layers: ~35 dari ~52 total
- CPU layers: ~17
- Expected TPS: 6–10 tok/s
- Quality: Noticeable degradation, terutama untuk instruction following kompleks

**Tidak feasible untuk 27B:**
- Q4_K_M: ~16.4 GB → RAM OOM ❌
- Q5_K_M+: Jauh di atas batas RAM

---

## KV Cache Quantization

KV cache quantization adalah optimasi tambahan yang krusial di hardware ini:

### Rekomendasi

| Skenario | KV Cache | Saving | Command |
|---|---|---|---|
| Qwen3-8B, 4096 ctx | q8_0 | -288 MB VRAM | `--cache-type-k q8_0 --cache-type-v q8_0` |
| Qwen3-8B, 2048 ctx | q8_0 | -144 MB VRAM | Sama |
| Qwen3-27B, 2048 ctx | q4_0 | -208 MB VRAM | `--cache-type-k q4_0 --cache-type-v q4_0` |

**KV q8_0 quality impact:** Minimal — KV quantization ke 8-bit memiliki impact kualitas yang sangat kecil berdasarkan literatur, berbeda dengan weight quantization.

**KV q4_0 quality impact:** Ada sedikit degradation pada long-context tasks, terutama untuk recall exact information dari early context. Acceptable untuk 27B hybrid mode dimana tradeoff diperlukan.

> ⚠️ ASSUMPTION: Quality impact KV quantization di atas belum diverifikasi empiris untuk Qwen3 specifically. Harus ditest di SPEC-006 benchmark suite.

---

## Compile Flags Wajib

llama.cpp harus dikompilasi khusus untuk Pascal SM 6.1:

```bash
# CMake build (recommended)
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=61 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

# Verifikasi build
./build/bin/llama-cli --version
# Harus ada "CUDA" dan "sm_61" di output
```

**Kenapa `CUDA_ARCHITECTURES=61` penting?**
- Tanpa flag ini, llama.cpp mungkin compile untuk target SM yang lebih modern
- Kernel yang dicompile untuk SM 7.0+ tidak akan berjalan di SM 6.1
- Explicit SM 61 menghasilkan kernel yang dioptimasi untuk Pascal

---

## Rejected Alternatives

### GPTQ (AutoGPTQ)

**Ditolak karena:** GPTQ menggunakan INT8 matrix multiplication (DP4A instruction) untuk dequantization yang tidak tersedia di Pascal SM 6.1. Di Pascal, GPTQ inference akan fallback ke FP32 path yang lebih lambat dibanding GGUF yang sudah dioptimasi llama.cpp. Tidak ada keuntungan performa, kompleksitas setup lebih tinggi.

### AWQ (Activation-aware Weight Quantization)

**Ditolak karena:** Sama dengan GPTQ — bergantung pada INT8 acceleration yang tidak ada di Pascal. AWQ tidak memberikan manfaat di atas GGUF Q4_K_M pada hardware ini.

### ExLlamaV2 (EXL2 format)

**Ditolak karena:** ExLlamaV2's performance advantage berasal dari Tensor Core path (SM ≥ 7.5). Di Pascal, ExLlamaV2 menggunakan scalar fallback yang dilaporkan community lebih lambat 20–40% dibanding llama.cpp GGUF untuk hardware yang sama. Setup lebih kompleks tanpa keuntungan.

### vLLM

**Ditolak karena:** vLLM membutuhkan CUDA SM ≥ 8.0 (Ampere) untuk PagedAttention dan Flash Attention. Tidak kompatibel dengan Pascal SM 6.1.

### TensorRT-LLM

**Ditolak karena:** Tidak ada optimasi bermakna untuk Pascal. INT8 dan FP8 paths tidak tersedia. Kompleksitas setup sangat tinggi untuk hasil yang setara atau lebih buruk dari llama.cpp.

### GGUF FP16 / BF16

**Ditolak untuk inferencing karena:** BF16 tidak punya hardware support di Pascal (GP104) — diemulasi dalam FP32, menghasilkan model 2× lebih lambat dari seharusnya. FP16 pada GP104 tidak lebih cepat dari FP32 (tidak ada FP16 fast path di GP104, hanya di GP100). Keduanya juga tidak muat di VRAM 8 GB untuk model 8B+ (membutuhkan 16.5 GB).

---

## Open Questions

- [ ] Apakah KV cache q4_0 memengaruhi kualitas output secara signifikan untuk Qwen3? → Test di SPEC-006
- [ ] Berapa TPS aktual (bukan estimasi) untuk masing-masing quant? → Measure di SPEC-002/SPEC-006
- [ ] Apakah Q3_K_M 27B benar-benar muat di 16 GB RAM tanpa swap? → Verify di SPEC-005
- [ ] Apakah ada GGUF quantization baru (Q4_K_S, Q5_K_S, dll.) yang lebih efisien? → Research ongoing

---

## References

- llama.cpp GGUF format documentation: https://github.com/ggerganov/llama.cpp/blob/master/docs/gguf.md
- NVIDIA Pascal Architecture Whitepaper: SM 6.1 compute capabilities
- llama.cpp CUDA architecture support: `ggml-cuda.cu` CUDA_ARCH_LIST
- Community benchmarks: llama.cpp GitHub discussions, r/LocalLLaMA

---

*RFC ini ditulis pada sesi bootstrap berdasarkan analisis arsitektur hardware dan software.*
*Angka-angka TPS adalah ESTIMASI. Update dengan angka terukur setelah SPEC-002 diimplementasikan.*
*Lihat `spec/handoff/SESSION-001-bootstrap.md` untuk konteks lengkap.*
