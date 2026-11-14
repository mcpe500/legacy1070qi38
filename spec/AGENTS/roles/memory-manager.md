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

## Memory Budget Reference (Qwen3.8-27B)

Fakta model dan sumbernya ada di `spec/AGENTS/MASTER.md`. Semua angka di bawah adalah **estimasi**.

### Prinsip

Total memori fisik 24 GiB (8 VRAM + 16 RAM). Bobot yang ada di VRAM tidak perlu ada di RAM.

```
Kapasitas VRAM untuk bobot ≈ 8.0 − 0.4 (CUDA context) − KV − compute buffer (~0.5, tebakan)
Bobot di RAM              ≈ ukuran file − kapasitas VRAM untuk bobot
RAM total                 ≈ bobot di RAM + ~2.5 GiB (OS, runtime, KV layer CPU)
```

### Contoh: Q4_K_M (16.53 GiB), konteks 4096

```
VRAM 8.0 GiB
├── CUDA context            ~0.4
├── KV (16 layer attention) ~0.25  (FP16; ~64 KiB/token)
├── Compute buffer          ~0.5   ⚠️ tebakan
├── State DeltaNet          ?      ⚠️ ukur
└── Bobot                   ~6.5–6.8  → ±28 dari 64 layer

RAM 16 GiB
├── Bobot layer CPU         ~10.0
├── OS + runtime            ~2.5   (tutup aplikasi lain)
└── Sisa                    ~3.5
```

### Tabel per quant

| Quant | Ukuran | Bobot di RAM | RAM total | Risiko |
|---|---|---|---|---|
| Q5_K_M | 18.70 | ~12.2 | ~14.7 | 🔴 Mepet |
| Q4_K_M | 16.53 | ~10.0 | ~12.5 | 🟢 |
| IQ4_XS | 15.28 | ~8.8 | ~11.3 | 🟢 |
| Q3_K_M | 13.23 | ~6.7 | ~9.2 | 🟢 |
| IQ3_M | 12.56 | ~6.1 | ~8.6 | 🟢 |
| IQ3_XXS | 11.60 | ~5.1 | ~7.6 | 🟢 |

Ukuran dalam GiB.

> ⚠️ ASSUMPTION: bobot per layer seragam. Layer attention penuh dan DeltaNet kemungkinan berbeda ukuran, dan
> embedding serta output (masing-masing ~4.7% backbone, tidak di-tie) tidak seragam. Pakai `gguf tensors`
> untuk angka nyata.

## KV Cache dan State

- Hanya 16 dari 64 layer menyimpan KV cache: ≈ **64 KiB per token** (FP16).
- 48 layer Gated DeltaNet menyimpan state ukuran tetap; besarnya belum diketahui.

| Konteks | KV FP16 | KV q8_0 (≈ setengah, asumsi) |
|---|---|---|
| 2048 | 128 MiB | ~64 MiB |
| 4096 | 256 MiB | ~128 MiB |
| 8192 | 512 MiB | ~256 MiB |
| 32768 | 2 GiB | ~1 GiB |

Pada model ini KV kecil, jadi KV quantization bukan tuas utama.

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
