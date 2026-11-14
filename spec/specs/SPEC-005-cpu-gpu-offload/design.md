# Design — SPEC-005: CPU-GPU Offload untuk Qwen3.8-27B

> Spec: SPEC-005 | Status: **Skeleton (direvisi)**

---

## Yang Diketahui

Perhitungan lengkap ada di `MASTER.md` (Memory Budget). Ringkas:

| Quant | Ukuran | Bobot di RAM (est.) | Batas atas TPS, RAM 20 / 40 GB/s |
|---|---|---|---|
| Q4_K_M | 16.53 GiB | ~10.0 GiB | ~1.9 / ~3.7 |
| IQ3_M | 12.56 GiB | ~6.1 GiB | ~3.1 / ~6.1 |
| IQ3_XXS | 11.60 GiB | ~5.1 GiB | ~3.6 / ~7.3 |

Bottleneck yang diperkirakan: bandwidth RAM untuk layer di CPU. PCIe tidak menjadi masalah untuk decode
satu token, karena hanya aktivasi yang melintas.

> ⚠️ ASSUMPTION: bobot per layer seragam. Layer full attention dan Gated DeltaNet kemungkinan berbeda
> ukuran, dan embedding/output masing-masing ~4.7% dari backbone. Pakai `gguf tensors` untuk angka nyata.

---

## Hal yang Dibandingkan

| Variabel | Pilihan |
|---|---|
| Quant | Q4_K_M, IQ4_XS, IQ3_M, Q3_K_M, IQ3_XXS |
| Offload | auto-fit vs `-ngl` manual |
| Backend | CUDA vs Vulkan |
| Speculative | tanpa vs MTP |
| Memory loading | mmap vs `--no-mmap` |

Gunakan `llama placement` untuk melihat layer mana di GPU atau CPU, dan `bench analyze` untuk melihat
apakah yang menahan itu bandwidth GPU, CPU, atau PCIe.

---

## Open Questions

- [ ] Apakah kernel Gated DeltaNet berjalan di SM 6.1 (diputuskan di SPEC-002)?
- [ ] Berapa RAM puncak aktual per quant?
- [ ] Apakah `--no-mmap` mempercepat dibanding mmap pada RAM 16 GB?
- [ ] Apakah MTP memberi percepatan nyata saat layer CPU menjadi penahan?
