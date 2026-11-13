# Role Agent: Memory Manager

> Baca `spec/AGENTS/MASTER.md` terlebih dahulu sebelum dokumen ini.

## Tanggung Jawab

Agent dengan role ini bertanggung jawab atas **semua aspek pengelolaan memori** — VRAM budget, system RAM budget, KV cache, CPU-GPU offloading, dan strategi menghindari OOM.

Scope tanggung jawab:
- Kalkulasi VRAM budget per model dan konfigurasi
- Strategi CPU-GPU layer offloading (`-ngl` tuning)
- KV cache management (ukuran, quantization, eviction)
- System RAM usage monitoring dan optimasi
- Strategi mencegah dan recovery dari OOM
- Memory mapping (`mmap`) vs eager loading tradeoffs
- Tensor split strategy untuk partial offload

## Yang TIDAK Termasuk Scope

- Kecepatan inference (tok/s) → lihat `roles/inference-optimizer.md`
- Pemilihan quantization format → lihat `spec/specs/RFC-001-quantization-strategy.md`
- Pengukuran benchmark → lihat `roles/benchmarker.md`

## Memory Budget Reference

### VRAM 8 GB — Qwen3-8B Breakdown

```
Total VRAM: 8192 MB
├── CUDA runtime overhead:     ~400 MB  (driver, context, libraries)
├── Model weights (Q5_K_M):   ~6100 MB
├── KV cache (4096 ctx, FP16): ~576 MB
├── Activation buffers:         ~100 MB
└── Headroom:                   ~16 MB  ← SANGAT TIPIS

Dengan KV cache q8_0 (4096 ctx):
├── CUDA runtime:              ~400 MB
├── Model weights (Q5_K_M):   ~6100 MB
├── KV cache (4096 ctx, q8_0): ~288 MB
├── Activation buffers:         ~100 MB
└── Headroom:                  ~304 MB  ← LEBIH AMAN
```

### RAM 16 GB — Qwen3-27B Breakdown

```
Total RAM: 16384 MB
├── OS + background:          ~1500 MB  (tutup semua app dulu)
├── llama.cpp runtime:         ~200 MB
├── Model weights (Q3_K_M):  ~13500 MB
├── KV cache (2048 ctx):       ~415 MB
└── Headroom:                  ~769 MB  ← TIPIS, risiko swap
```

> ⚠️ ASSUMPTION: OS overhead ~1.5 GB adalah estimasi untuk Linux minimal. Desktop environment bisa +500 MB – 1 GB.

## Layer Offloading Strategy

### Formula Kalkulasi GPU Layers

```
layer_size_MB = model_size_MB / num_layers
available_vram_MB = total_vram_MB - cuda_overhead_MB - kv_cache_MB - activation_MB
max_gpu_layers = floor(available_vram_MB / layer_size_MB)
```

### Qwen3-27B Layer Offload Table

| Quant | Model Size | Layer Size | Available VRAM | GPU Layers | CPU Layers |
|---|---|---|---|---|---|
| Q4_K_M | 16400 MB | ~315 MB | ~7100 MB | ~22 | ~30 |
| Q3_K_M | 13500 MB | ~260 MB | ~7100 MB | ~27 | ~25 |
| Q2_K | 10400 MB | ~200 MB | ~7100 MB | ~35 | ~17 |

> ⚠️ ASSUMPTION: Embedding layer dan LM head (~1.5 GB) dihitung dalam model size dan bisa ditempatkan di GPU atau CPU tergantung strategi llama.cpp.

## KV Cache Management

### KV Cache Size Formula

```
kv_size_bytes = 2 × n_layers × n_kv_heads × head_dim × ctx_len × bytes_per_element
```

### KV Cache untuk Qwen3-8B (36 layers, 8 KV heads, 128 head dim)

| Context | FP16 | q8_0 | q4_0 |
|---|---|---|---|
| 512 | 72 MB | 36 MB | 18 MB |
| 1024 | 144 MB | 72 MB | 36 MB |
| 2048 | 288 MB | 144 MB | 72 MB |
| 4096 | 576 MB | 288 MB | 144 MB |

### KV Cache untuk Qwen3-27B (52 layers, 8 KV heads, 128 head dim)

| Context | FP16 | q8_0 | q4_0 |
|---|---|---|---|
| 512 | 104 MB | 52 MB | 26 MB |
| 1024 | 208 MB | 104 MB | 52 MB |
| 2048 | 415 MB | 208 MB | 104 MB |
| 4096 | 830 MB | 415 MB | 208 MB |

> ⚠️ ASSUMPTION: KV head count = 8 berdasarkan pola GQA Qwen2 family. Verifikasi dari model card.

## Checklist Sebelum Launch

Jalankan checklist ini sebelum meluncurkan inference untuk mencegah OOM:

```bash
# 1. Cek free VRAM
nvidia-smi --query-gpu=memory.free,memory.total --format=csv

# 2. Cek free RAM
free -h

# 3. Hitung budget (harus ada headroom ≥ 200 MB VRAM, ≥ 500 MB RAM)

# 4. Tutup proses tidak perlu
# - Browser tabs besar
# - Aplikasi desktop berat
# - Docker containers yang tidak dipakai

# 5. Set swap ke minimum (atau nonaktifkan sementara untuk perf)
# sudo swapoff -a  # HATI-HATI: hanya jika RAM cukup
```

## OOM Recovery Protocol

Jika terjadi OOM saat inference:

1. **VRAM OOM** → Kurangi `-ngl` sebesar 2–4 layers, atau kurangi `--ctx-size`
2. **RAM OOM** → Pindah ke quant lebih kecil (Q3_K_M → Q2_K), atau kurangi `--ctx-size`
3. **Both OOM** → Gunakan `--mmap` (lazy loading) + swap ke SSD (NVMe preferred)
4. **Persistent OOM** → Dokumentasikan di `spec/handoff/HANDOFF-LOG.md` dengan label `[OOM]`

## Output Format

Ketika memberikan rekomendasi memory management, selalu sertakan:

1. **Memory breakdown tabel** (VRAM dan RAM terpisah)
2. **Headroom** yang tersisa setelah konfigurasi
3. **Risk level**: 🟢 Safe | 🟡 Tight | 🔴 Likely OOM
4. **Fallback option** jika konfigurasi utama gagal

## Referensi Spec

- `spec/specs/RFC-001-quantization-strategy.md` — ukuran model per quant
- `spec/specs/SPEC-004-kv-cache-optimizer/` — optimasi KV cache lanjutan
- `spec/specs/SPEC-005-cpu-gpu-offload/` — strategi offloading lanjutan
- `spec/AGENTS/MASTER.md` — hardware constraints
