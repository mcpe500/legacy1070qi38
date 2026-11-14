# RFC-001: Quantization dan Offload Strategy untuk Qwen3.8-27B di Pascal SM 6.1

> Status: **Proposed (direvisi)**. Sebelumnya "Accepted" dengan asumsi model yang salah.
> Berlaku untuk: semua spec yang menyentuh quantization dan offload
> Fakta model dan sumbernya: lihat `spec/AGENTS/MASTER.md`, bagian "Fakta Model"

---

## Koreksi dari Versi Awal

| Klaim lama | Masalah | Koreksi |
|---|---|---|
| Target Qwen3-8B dan Qwen3-27B | Goal proyek adalah Qwen3.8-27B. Tidak ada 8B yang terverifikasi di keluarga 3.8 | Target tunggal Qwen3.8-27B |
| 52 layer, KV cache di semua layer | Model hibrida: 64 layer, hanya 16 yang punya KV cache | KV ≈ 64 KiB/token |
| Q4_K_M 27B "tidak muat di 16 GB RAM" | Menghitung seluruh model di RAM, padahal sebagian bobot ada di VRAM | Q4_K_M muat (≈12.5 GiB RAM) |
| Hanya K-quant dibandingkan | Model card menunjukkan IQ3_M lebih kecil dari Q3_K_M dengan PPL setara | IQ-quant masuk kandidat |
| Ukuran file hasil perkiraan | Ada ukuran nyata dari repo GGUF | Pakai tabel ukuran nyata |

---

## Summary

Gunakan **GGUF via llama.cpp**. Uji empat strategi (A sampai D di MASTER) dan pilih berdasarkan pengukuran,
bukan estimasi. Urutan kandidat quant untuk hybrid: Q4_K_M, IQ4_XS, IQ3_M, Q3_K_M, IQ3_XXS.

---

## Motivation

GTX 1070 Ti tidak punya Tensor Core, INT8 cepat, atau BF16. Format yang mengandalkannya (GPTQ, AWQ, EXL2)
tidak memberi manfaat. GGUF punya kernel dequantization di CUDA yang berjalan di SM 6.1 dan mendukung
offload sebagian layer. Model ini juga hibrida (Gated DeltaNet + full attention), sehingga kebutuhan
memorinya berbeda dari transformer biasa.

---

## Kandidat Quant

| Quant | Ukuran (GiB) | PPL | RAM total (estimasi) | Catatan |
|---|---|---|---|---|
| Q5_K_M | 18.70 | 6.8010 | ~14.7 | RAM mepet |
| **Q4_K_M** | 16.53 | 6.8103 | ~12.5 | Titik awal pengukuran |
| IQ4_XS | 15.28 | 6.8529 | ~11.3 | Lebih kecil, PPL dekat |
| Q3_K_M | 13.23 | 7.1020 | ~9.2 | |
| **IQ3_M** | 12.56 | 7.0306 | ~8.6 | Kandidat utama bila perlu lebih cepat |
| IQ3_XXS | 11.60 | 7.1994 | ~7.6 | |
| Q3-DOWN-XS | 7.68 | 8.27 (protokol lain) | n/a | Strategi B |

> Sumber: [6block](https://huggingface.co/6block/Qwen3.8-27B-GGUF) dan
> [guideboardlabs](https://huggingface.co/guideboardlabs/Qwen3.8-27B-Q3-DOWN-XS-GGUF).
> PPL hanya sebanding di dalam satu setup. Angka dua repo **tidak boleh dibandingkan**.
> ⚠️ ASSUMPTION: IQ-quant berjalan di CUDA Pascal dengan kecepatan yang baik. Dequantization IQ lebih berat
> daripada K-quant. Belum diuji.

### Rumus memori

```
Bobot di RAM ≈ ukuran file − kapasitas VRAM untuk bobot (≈ 6.5 GiB)
RAM total    ≈ bobot di RAM + ~2.5 GiB
TPS (batas atas) ≈ bandwidth RAM ÷ bobot di RAM
```

---

## KV Cache dan State

- Hanya 16 layer full attention yang menyimpan KV cache: ≈ **64 KiB per token** (FP16).
- 48 layer Gated DeltaNet menyimpan state ukuran tetap. **Besarnya belum diketahui; ukur.**

| Konteks | KV FP16 | KV q8_0 (≈ setengah) |
|---|---|---|
| 2048 | 128 MiB | ~64 MiB |
| 4096 | 256 MiB | ~128 MiB |
| 8192 | 512 MiB | ~256 MiB |
| 32768 | 2 GiB | ~1 GiB |

> ⚠️ ASSUMPTION: q8_0 memakai ≈ setengah memori FP16. Pada model ini KV sudah kecil, jadi manfaat KV
> quantization jauh lebih kecil daripada yang kita kira di versi awal. Nilainya perlu diuji, termasuk
> pengaruhnya pada kualitas.

---

## Strategi yang Dibandingkan

| | Strategi | Bukti yang ada | Yang harus diukur |
|---|---|---|---|
| A | Hybrid, Q4_K_M sampai IQ3 | Hanya perhitungan bandwidth | TPS, RAM puncak, jumlah layer di GPU |
| B | Muat penuh, ~7.7 GiB | 24.48 tok/s di RX 5700 XT (Vulkan), kualitas 86.5% terhadap kontrol 3-bit | Apakah muat di 1070 Ti; TPS; kualitas absolut |
| C | MTP speculative decoding | Model card menyebut `--spec-type draft-mtp` | Dukungan di build kita; faktor percepatan |
| D | Vulkan vs CUDA | Hasil B memakai Vulkan | Selisih TPS di Pascal |

Catatan tentang B: klaim "86.5%" adalah terhadap kontrol 3-bit (UD-IQ3_XXS), **bukan** terhadap BF16.
Model card sendiri menyatakan kualitas terhadap presisi penuh tidak terukur. Hitung ulang dengan
`quality eval` pada hardware kita.

---

## Compile Flags

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

---

## Rejected Alternatives

| Alternatif | Alasan |
|---|---|
| GPTQ, AWQ | Butuh INT8 DP4A; tidak ada manfaat di Pascal |
| EXL2 / ExLlamaV2 | Jalur cepat butuh Tensor Core |
| vLLM | Butuh SM ≥ 8.0 |
| TensorRT-LLM | Tidak ada optimasi bermakna untuk Pascal |
| BF16 / FP16 penuh | Tidak ada hardware BF16; 50 GiB tidak muat |

---

## Open Questions

- [ ] Apakah kernel Gated DeltaNet di llama.cpp CUDA berjalan di SM 6.1? Jika tidak, seluruh RFC ini gugur untuk CUDA.
- [ ] Kartu resmi `Qwen/Qwen3.8-27B`: layer, KV heads, head dim, ukuran state DeltaNet?
- [ ] Apakah auto-fit layer lebih baik dari `-ngl` manual?
- [ ] Apakah IQ-quant cukup cepat di Pascal?
- [ ] Apakah MTP draft head jalan di build kita?
- [ ] Apakah KV q8_0 mengubah kualitas?

---

*Tidak ada angka TPS di dokumen ini yang terukur. Semua adalah batas atas hasil perhitungan.*
