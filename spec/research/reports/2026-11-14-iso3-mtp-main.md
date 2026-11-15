# iso3 (KV 3-bit) + MTP di llama.cpp utama

Bagian dari #31 dan #35. Patch: `patches/llama.cpp/` (dua commit di atas `ggml-org/llama.cpp` `3549c27`, branch lokal `feat/iso3-kv`). Terapkan dengan `git am patches/llama.cpp/*.patch`. Semua angka terukur di GTX 1070 Ti (sm_61), CUDA 12.6, 2026-11-14.

## Isi patch

| Bagian | Isi |
|---|---|
| Tipe | `GGML_TYPE_ISO3_0 = 44`, nama `iso3`, 128 nilai / 50 byte (0,195x FP16). Tata letak blok dan konstanta (codebook 8 nilai, 32 kuaternion) sama dengan fork `08e025c`. ID tipe beda dari fork (45), tidak masalah karena tipe KV tidak disimpan di GGUF |
| CPU | quantize/dequantize referensi, `vec_dot` iso3 x f32 (dipakai flash attention CPU) |
| CUDA | `SET_ROWS` f32 -> iso3 (penulisan KV), konversi iso3 -> f16/f32 (kontigu dan non-kontigu) |
| Flash attention CUDA | tanpa kernel vektor iso3: K/V iso3 dikonversi ke f16 sebelum kernel (jalur fallback yang sudah ada di llama.cpp utama) |
| CLI | `-ctk/-ctv/-ctkd/-ctvd iso3` |
| Tidak diporting | kuantisasi tertunda (K f16 saat prefill) dan mode layer-adaptive dari fork |

## TDD

Red (sebelum implementasi, commit `fc9437a`): build `test-iso3` gagal (`GGML_TYPE_ISO3_0 was not declared`); `tools/iso3/test_iso3_e2e.py` dengan build utama tanpa patch: 4 gagal (`Unsupported cache type: iso3`).

Green (commit `36dbad3`):

| Tes | Hasil |
|---|---|
| `test-iso3` (traits, bentuk tensor, roundtrip CPU, `SET_ROWS` CUDA vs referensi CPU, FA CUDA iso3 vs f16) | PASSED, 0 gagal |
| roundtrip gaussian (skala 0.01, 1, 50) | nmse 0.0348 (batas 0.05); kuantisasi ulang stabil 3200/3200 byte |
| byte KV yang ditulis CUDA = referensi CPU | 1600/1600 |
| FA CUDA iso3 vs f16 dari nilai yang sudah didekuantisasi (hs 128/256, n_q 1/4/32) | nmse <= 2.0e-11 |
| FA CUDA iso3 vs f16 nilai asli | nmse 0.063-0.073 (galat kuantisasi 3-bit) |
| `test-quantize-fns` | rc 0; iso3: galat 0.003857, galat dot 0.019717 |
| `test-backend-ops -b CUDA0` kasus iso3 (CUDA dibandingkan dengan CPU) | 28/28 |
| `test-backend-ops -b CUDA0 -o SET_ROWS,GET_ROWS,CPY,FLASH_ATTN_EXT` (semua tipe, regresi) | 5041/5041 |
| `ctest -L main` (kecuali tokenizer dan backend-ops) | rc 0 |
| e2e 0.8B: bentuk KV (`K (iso3): 1.17 MiB` untuk 1024 sel x 6 layer x 512), K q8_0 + V iso3 | lulus |
| e2e 0.8B: KLD iso3 vs f16 (wikitext-2 test, sha256 `173c87a5...`, ctx 512, 8 chunk, ngl 99) | mean KLD **0.0495**, same top p **87.7%** (q4_0 pada model ini: 0.0200, C-47) |
| e2e 27B MTP + iso3 | lulus (lihat bawah) |

## MTP + iso3, Qwen3.8-27B UD-IQ4_XS, `-ngl 30 -c 2048 -np 1 -fa on`, temperature 0

| KV | MTP depth | tok/s | draft | diterima | md5 (128 token) |
|---|---:|---:|---:|---:|---|
| f16 | 0 | 2.05 | - | - | 93fcac7c |
| f16 | 1 | 2.17 | 66 | 60 | ebff812c |
| f16 | 3 | 1.97 | 115 | 87 | ebff812c |
| q8_0 | 0 | 1.98 | - | - | 93fcac7c |
| q8_0 | 1 | 2.09 | 65 | 61 | 93fcac7c |
| q8_0 | 3 | 1.86 | 115 | 87 | ebff812c |
| iso3 (MTP KV f16) | 0 | 2.00 | - | - | ac406bbc |
| iso3 (MTP KV f16) | 1 | 2.02 | 69 | 58 | a6d528b1 |
| iso3 (MTP KV f16) | 3 | 1.77 | 126 | 85 | a6d528b1 |

Uji `MtpWithIso3` (96 token, `-ctkd iso3 -ctvd iso3`): depth 1: 2.08 tok/s, draft 50 diterima 45; depth 3: 1.77 tok/s, draft 91 diterima 64; tanpa MTP 2.02 tok/s. Kedua depth berbeda dari tanpa MTP mulai token 75/96; di titik itu token MTP (`' Returns'`) adalah token terbaik menurut target tanpa MTP (selisih logprob 0.000), jadi bedanya berasal dari urutan hitung float, bukan KV rusak setelah rollback.

Temuan:
- Keluaran MTP di llama.cpp utama **tidak identik bit** dengan tanpa MTP, juga dengan KV f16 (md5 93fcac7c vs ebff812c). Uji rollback memakai kriteria "token MTP = kandidat hampir seri di titik beda pertama", bukan kesamaan md5.
- **Pewarisan tipe KV**: konteks MTP punya KV sendiri untuk 1 layer MTP (`blk.64`). Tipenya dari `-ctkd/-ctvd` (default f16), **tidak** diwarisi dari `-ctk/-ctv` (log: `1 layers ... K (f16): 4.00 MiB`). Dengan `-ctkd iso3 -ctvd iso3` semua KV iso3. Mengubah default menjadi warisan adalah keputusan perilaku dan belum dilakukan.
- Di semua layer KV ditempatkan di GPU (log `layer 63: dev = CUDA0` pada ngl 30), jadi jalur FA CPU iso3 hanya diuji lewat `test-backend-ops` (CPU sebagai referensi) dan `test-iso3`.
- Pada ctx 2048 iso3 tidak mempercepat: KV 27B kecil (f16 128 MiB, iso3 25 MiB). Manfaat yang diharapkan ada pada konteks panjang; belum diukur.
- Konversi iso3 -> f16 tiap langkah memakan waktu; kernel vektor iso3 native belum dibuat.

## Konteks panjang: Qwen3.8-27B UD-Q4_K_M, ctx 16384, prompt 6011 token (wikitext-2), 64 token generasi

`tools/iso3/long_ctx.py` (llama-server 127.0.0.1, `-np 1 -fa on -t 4`, temperature 0). Untuk tiap konfigurasi dicoba ngl menurun; dicatat ngl tertinggi yang jalan. KV f16 pada 16384 sel = 1 GiB, iso3 = 0.195 GiB.

| KV | MTP | ngl maks | prompt tok/s | gen tok/s | draft / diterima | VRAM puncak MiB | md5 |
|---|---|---:|---:|---:|---|---:|---|
| f16 | - | 28 (30 OOM) | 35.30 | 1.86 | - | 7737 | 1e09276c |
| iso3 | - | 31 (32 OOM) | 37.57 | **2.23** | - | 8031 | f0d64c22 |
| f16 | depth 1 | 26 (28 OOM saat membuat konteks MTP) | 33.86 | 1.80 | 37 / 26 | 7883 | 1e09276c |
| iso3 (MTP KV iso3) | depth 1 | 28 (30, 31 OOM) | 35.42 | **2.37** | 32 / 30 | 8029 | f0d64c22 |
| iso3 (MTP KV iso3) | depth 3 | 28: **crash** CUDA OOM saat generasi (`cuMemCreate`, ggml-cuda.cu:653); 30, 31 OOM saat muat | - | - | - | - | - |

Temuan:
- Pada konteks 16k iso3 membebaskan VRAM untuk 3 layer tambahan: generasi **+20%** (2.23 vs 1.86 tok/s) tanpa MTP, dan iso3 + MTP depth 1 **+32%** (2.37 vs 1.80) dibanding f16 + MTP depth 1.
- iso3 + MTP depth 1 menghasilkan teks identik dengan iso3 tanpa MTP (md5 f0d64c22), begitu juga f16 (1e09276c) pada 64 token ini.
- Teks iso3 berbeda dari f16 (kuantisasi KV 3-bit mengubah keluaran); dampak kualitas diukur dengan KLD.
- MTP depth 3 butuh VRAM lebih banyak dari yang tersisa di ngl 28; crash OOM terjadi di tengah generasi, bukan saat muat. Belum dicoba ngl 26 untuk depth 3.
