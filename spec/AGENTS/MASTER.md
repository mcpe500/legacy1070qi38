# MASTER AGENT — legacy1070qi38

> **Baca dokumen ini pertama sebelum mengerjakan apapun dalam proyek ini.**
> Semua agents — baik role-based maupun phase-based — tunduk pada aturan di dokumen ini.

---

## Identitas Proyek

**Nama:** `legacy1070qi38`
**Spirit:** *Old Hardware, New Intelligence*
**Tema:** Resurrection — memberikan kehidupan kedua pada GPU generasi lama
**Lisensi:** Apache 2.0

Proyek ini adalah **LLM inference engine open-source** yang mengoptimalkan hardware lama — khususnya NVIDIA GTX 1070 Ti (8GB VRAM, Pascal SM 6.1) — untuk menjalankan model Large Language Model modern secara efisien. Fokus utama: **Qwen3-8B** dan **Qwen3-27B**, dengan target maximizing tokens per second, minimizing VRAM dan RAM usage, serta menjaga kualitas output tetap dapat diterima.

---

## Hardware Constraints (NON-NEGOTIABLE)

Semua keputusan teknis HARUS mempertimbangkan batasan hardware ini:

| Komponen | Spesifikasi |
|---|---|
| GPU | NVIDIA GTX 1070 Ti |
| GPU Architecture | Pascal (GP104, SM 6.1) |
| VRAM | 8 GB GDDR5 |
| Memory Bandwidth (GPU) | 256 GB/s |
| CUDA Cores | 2432 |
| System RAM | 16 GB DDR4 |
| RAM Bandwidth (est.) | ~40–45 GB/s (dual-channel) |
| PCIe | 3.0 x16 (~12–13 GB/s practical) |
| GPU TDP | 180W |

### Apa yang TIDAK ADA di hardware ini:
- ❌ **Tensor Cores** (tidak ada di Pascal — mulai Volta)
- ❌ **Flash Attention** (membutuhkan SM ≥ 7.5 untuk v1, SM ≥ 8.0 untuk v2)
- ❌ **Native INT8 acceleration** (tidak ada DP4A di Pascal)
- ❌ **BF16 hardware support** (Pascal hanya FP32 dan FP16 CUDA-core, tanpa speedup FP16 di GP104)
- ❌ **Shared memory besar** (48 KB/SM vs 96–164 KB di Ampere/Hopper)

---

## Model Targets

### Qwen3-8B
- Parameter: ~8.19B
- Layers: ~36 transformer layers
- Hidden dim: 4096
- KV heads (GQA): 8
- Head dim: 128
- FFN intermediate: 28,672
- Vocabulary: 152,064 tokens

### Qwen3-27B
- Parameter: ~27.4B
- Layers: ~52 transformer layers
- Hidden dim: 5120
- KV heads (GQA): 8
- Head dim: 128
- FFN intermediate: ~27,648–32,768

> ⚠️ **ASSUMPTION:** Arsitektur Qwen3 di atas diekstrapolasi dari keluarga Qwen2/Qwen2.5. Verifikasi ke model card resmi saat model tersedia.

---

## Memory Budget

### Qwen3-8B (full VRAM inference)

| Quant | Weights | +KV(4096ctx) | +overhead | Total | Status |
|---|---|---|---|---|---|
| Q8_0 | 8.8 GB | 0.58 GB | 0.4 GB | 9.8 GB | ❌ Tidak muat |
| Q6_K | 7.0 GB | 0.29 GB | 0.4 GB | 7.7 GB | ⚠️ Tight (ctx≤2048) |
| Q5_K_M | 6.1 GB | 0.58 GB | 0.4 GB | 7.1 GB | ✅ **RECOMMENDED** |
| Q4_K_M | 5.0 GB | 0.58 GB | 0.4 GB | 6.0 GB | ✅ Comfortable |
| Q3_K_M | 4.1 GB | 0.58 GB | 0.4 GB | 5.1 GB | ✅ Comfortable |

> KV cache bisa dikurangi dengan `--cache-type-k q8_0 --cache-type-v q8_0` (~50% saving).

### Qwen3-27B (hybrid CPU-GPU offloading)

| Quant | Weights | RAM Total | GPU Layers | CPU Layers | Tok/s est. | Status |
|---|---|---|---|---|---|---|
| Q4_K_M | 16.4 GB | ~18 GB | ~22 | ~30 | 2–4 | ❌ RAM OOM |
| Q3_K_M | 13.5 GB | ~15.5 GB | ~27 | ~25 | 3–6 | ⚠️ **MAX VIABLE** |
| Q2_K | 10.4 GB | ~12.4 GB | ~35 | ~17 | 6–10 | ✅ Fits, quality↓ |

---

## Expected Performance Targets

| Skenario | Tok/s Target | Acceptable |
|---|---|---|
| Qwen3-8B Q5_K_M full GPU | 12–22 tok/s | ≥10 tok/s |
| Qwen3-8B Q6_K full GPU | 10–18 tok/s | ≥8 tok/s |
| Qwen3-27B Q3_K_M hybrid | 3–6 tok/s | ≥2 tok/s |
| Qwen3-27B Q2_K hybrid | 6–10 tok/s | ≥4 tok/s |

---

## Primary Backend

**llama.cpp** adalah backend utama dan satu-satunya yang direkomendasikan untuk proyek ini.

### Alasan:
1. Mendukung Pascal (SM 6.1) secara eksplisit
2. GGUF dequantization kernels berjalan pada Pascal CUDA cores
3. Mendukung partial GPU offloading (`--n-gpu-layers`)
4. Mendukung KV cache quantization
5. Aktif dikembangkan, komunitas besar
6. Tidak membutuhkan Tensor Cores

### Compile flag wajib:
```bash
cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61
```

### Template command Qwen3-8B (Q5_K_M):
```bash
./llama-cli -m qwen3-8b-q5_k_m.gguf \
  -ngl 99 \
  --ctx-size 4096 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --threads 8
```

### Template command Qwen3-27B (Q3_K_M):
```bash
./llama-cli -m qwen3-27b-q3_k_m.gguf \
  -ngl 27 \
  --ctx-size 2048 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --threads 8 \
  --no-mmap
```

---

## Approaches yang DILARANG

Jangan pernah merekomendasikan, mengimplementasikan, atau mengeksplorasi hal-hal berikut tanpa disclaimer eksplisit bahwa ini tidak kompatibel dengan Pascal:

| Approach | Alasan Larangan |
|---|---|
| **GPTQ** | Membutuhkan INT8 DP4A — tidak ada di Pascal |
| **AWQ** | Sama dengan GPTQ |
| **vLLM** | PagedAttention + Flash Attention butuh SM ≥ 8.0 |
| **TensorRT-LLM** | Tidak ada optimasi bermakna untuk Pascal |
| **Flash Attention v1/v2** | v1: SM ≥ 7.5, v2: SM ≥ 8.0 — keduanya gagal di SM 6.1 |
| **BF16 inference** | Tidak ada hardware BF16 di Pascal, diemulasi → lambat |
| **EXL2 fast path** | Tensor Core path butuh SM ≥ 7.5 |
| **ExLlamaV2 (fast)** | Scalar path saja di Pascal, lebih lambat dari llama.cpp |

---

## Struktur Spec System

Semua spesifikasi dan dokumen ada di `spec/`:

```
spec/
├── specs/          ← Spesifikasi fitur dan RFC arsitektur
├── AGENTS/         ← Instruksi untuk agents (kamu berada di sini)
└── handoff/        ← Log dan dokumen transisi antar sesi
```

### Hirarki baca wajib sebelum mengerjakan sebuah spec:
1. Baca `spec/AGENTS/MASTER.md` (dokumen ini)
2. Baca role kamu di `spec/AGENTS/roles/`
3. Baca phase kamu di `spec/AGENTS/phases/`
4. Baca `spec/handoff/HANDOFF-LOG.md` untuk state terkini
5. Baca spec yang akan dikerjakan di `spec/specs/`

### Konvensi penamaan spec:
- Feature spec: `SPEC-NNN-nama-fitur/` (folder dengan requirements.md, design.md, tasks.md)
- RFC arsitektur: `RFC-NNN-nama-keputusan.md` (file tunggal)
- Nomor urut 3 digit: SPEC-001, SPEC-002, RFC-001, dst.

---

## Aturan Umum Agent

### Bahasa
- Dokumentasi teknis: **Bahasa Indonesia** (dengan istilah teknis dalam Bahasa Inggris seperti biasa)
- Komentar kode: **Bahasa Inggris**
- Nama file, variabel, fungsi: **Bahasa Inggris**

### Format Output
- Gunakan tabel untuk perbandingan angka dan trade-off
- Gunakan code block untuk perintah, konfigurasi, dan kode
- Sertakan ⚠️ ASSUMPTION di setiap klaim yang belum diverifikasi empiris
- Sertakan referensi ke spec/RFC terkait

### Ketika menemukan konflik dengan spec:
1. Catat konflik di `spec/handoff/HANDOFF-LOG.md` dengan label `[CONFLICT]`
2. Jangan asumsikan — tanya atau escalate
3. Jangan modifikasi spec lain tanpa explicit permission

### Ketika selesai mengerjakan sebuah task:
1. Update `spec/handoff/HANDOFF-LOG.md` dengan entri baru
2. Jika ada keputusan baru yang dibuat, dokumentasikan alasannya
3. Jika ada open question baru, tambahkan ke session handoff aktif

---

## Success Criteria Proyek

Proyek dianggap berhasil jika:

1. **Qwen3-8B** dapat berjalan pada GTX 1070 Ti dengan ≥10 tok/s pada Q5_K_M
2. **Qwen3-27B** dapat berjalan dengan hybrid offloading dan menghasilkan ≥2 tok/s pada Q3_K_M
3. Penggunaan VRAM tidak melebihi 7.5 GB untuk Qwen3-8B (meninggalkan headroom)
4. Total RAM usage untuk Qwen3-27B tidak melebihi 15 GB (menghindari swapping)
5. Output model tetap koheren dan dapat dievaluasi kualitasnya
6. Proses setup terdokumentasi dan dapat direproduksi

---

## Open Questions (Bootstrap)

Pertanyaan-pertanyaan ini belum terjawab dan harus dijawab melalui eksperimen:

- [ ] Berapa layer count Qwen3-8B dan Qwen3-27B yang sebenarnya? (perlu verifikasi model card)
- [ ] Berapa KV head count Qwen3 yang sebenarnya?
- [ ] Apakah llama.cpp versi terbaru memiliki regresi performa di Pascal?
- [ ] Apakah KV cache quantization Q4 memengaruhi kualitas output secara signifikan?
- [ ] Berapa TPS aktual (bukan estimasi) untuk masing-masing skenario?
- [ ] Apakah system RAM 16 GB cukup untuk Q3_K_M 27B tanpa swap? (perlu pengukuran actual RSS)
- [ ] Apakah ada CUDA kernel yang perlu di-patch untuk Pascal?

---

*Dokumen ini dibuat pada sesi bootstrap. Update jika ada keputusan arsitektur baru yang fundamental.*
*Lihat: `spec/handoff/SESSION-001-bootstrap.md` untuk konteks sesi ini.*
