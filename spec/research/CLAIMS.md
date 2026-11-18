# Daftar Klaim Bersumber

Aturan: klaim faktual di `MASTER.md`, RFC, dan SPEC harus punya baris di sini. Klaim tanpa baris di sini
tidak boleh masuk dokumen tersebut. Tanggal pemeriksaan: 2026-11-13.

| Jenis sumber | Arti |
|---|---|
| **Resmi** | Kartu atau dokumentasi dari pembuat (Qwen, vLLM) |
| **Kode** | Dibaca dari kode sumber yang kita pakai (fork di `~/llama-cpp-turboquant`, commit `08e025c`) |
| **Terukur** | Diukur di mesin ini dengan alat di repo |
| **Turunan** | Dihitung dari baris lain di tabel ini |
| **Komunitas** | Model card pihak ketiga, issue/diskusi GitHub; belum dicek ulang |
| **Tanpa sumber** | Belum ada sumber yang diperiksa |

## A. Model Qwen3.8-27B

Sumber utama: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (kartu resmi). Repo resmi hanya berisi
bobot safetensors untuk Transformers, vLLM, SGLang, TokenSpeed; **tidak ada GGUF resmi**.

| ID | Klaim | Sumber | Jenis | Status | OQ |
|---|---|---|---|---|---|
| C-01 | 64 layer, susunan 16 x (3 x (Gated DeltaNet -> FFN) + 1 x (Gated Attention -> FFN)) | kartu resmi, "Hidden Layout" | Resmi | ✅ Terverifikasi | OQ-010 |
| C-02 | Hidden dim 5120; FFN intermediate 17.408 | kartu resmi | Resmi | ✅ | OQ-010 |
| C-03 | Gated Attention: 24 head Q, **4 head KV**, head dim 256 | kartu resmi | Resmi | ✅ (mengganti dugaan "1024 turunan") | OQ-010 |
| C-04 | Gated DeltaNet: 48 head V, 16 head QK, head dim 128 | kartu resmi | Resmi | ✅ | OQ-010 |
| C-05 | KV cache = 64 KiB/token (FP16): 16 layer x 2 x 4 x 256 x 2 B = 65.536 B | C-01, C-03 | Turunan | ✅ | OQ-010 |
| C-06 | State rekuren DeltaNet ≈ 0.146 GiB (149,6 MiB), ukuran tetap: 48 x (48x128x128x4 B + 3x10240x4 B) | C-04 + rumus llama.cpp | Turunan | ⚠ Rumus **terverifikasi persis** pada Qwen3.5-0.8B (log `RS buffer size = 77.06 MiB` untuk 4 sekuens = hitungan 77.06). Belum diukur pada file 27B | OQ-010 |
| C-07 | Vocab 248.320 (padded); "Token Embedding" dan "LM Output" tercantum terpisah | kartu resmi | Resmi | ⚠ Kartu tidak menulis kata "untied". Cek `output.weight` terpisah dengan `gguf tensors` pada file 27B. Model 9B lokal memang punya `output.weight` terpisah (0.777 GiB) | - |
| C-08 | Konteks 262.144 native, sampai 1.000.000 dengan YaRN statis | kartu resmi | Resmi | ✅ | - |
| C-09 | MTP: dilatih dengan beberapa langkah | kartu resmi | Resmi | ✅ ada; kecepatannya lihat C-25 | OQ-012 |
| C-10 | Native vision-language (gambar, video), thinking aktif secara default | kartu resmi | Resmi | ✅ | - |
| C-11 | Sampling disarankan: thinking `temp 1.0, top_p 0.95, top_k 20`; non-thinking `temp 0.7, top_p 0.8, presence_penalty 1.5` | kartu resmi | Resmi | ✅ | - |
| C-12 | Parameter 27B teks, 27.8B dengan vision tower; `mmproj` ±0.9 GiB terpisah | model card 6block | Komunitas | ⚠ Belum dicek ke resmi | - |
| C-13 | Ukuran GGUF (GiB) dan PPL: Q5_K_M 18.70/6.8010, Q4_K_M 16.53/6.8103, IQ4_XS 15.28/6.8529, Q3_K_M 13.23/7.1020, IQ3_M 12.56/7.0306, Q3_K_S 12.16/7.2176, IQ3_XXS 11.60/7.1994 | [6block/Qwen3.8-27B-GGUF](https://huggingface.co/6block/Qwen3.8-27B-GGUF) | Komunitas | ⚠ PPL hanya sebanding di dalam tabel itu; belum diukur ulang di sini | OQ-015 |
| C-14 | Q3-DOWN-XS 7.68 GiB, 24.48 tok/s di RX 5700 XT 8 GB (Vulkan); kualitas 86.5% **terhadap kontrol 3-bit**, bukan terhadap BF16 | [guideboardlabs](https://huggingface.co/guideboardlabs/Qwen3.8-27B-Q3-DOWN-XS-GGUF) | Komunitas | ⚠ Lihat C-30: file ini tidak muat penuh di 1070 Ti | OQ-013 |

Koreksi dari MASTER versi lama: baris "Tidak ada 8B yang terverifikasi" tidak tepat. Ada model **9B** di keluarga
yang memakai arsitektur `qwen35` (lihat C-20), tetapi itu bukan Qwen3.8-27B dan bukan bobot resmi Qwen.

## B. llama.cpp, kernel, dan fork

| ID | Klaim | Sumber | Jenis | Status | OQ |
|---|---|---|---|---|---|
| C-15 | llama.cpp utama mendukung arsitektur `qwen35` termasuk Qwen3.8-27B (Gated DeltaNet + full attention, `full_attention_interval=4`) | [#28196](https://github.com/ggml-org/llama.cpp/issues/28196) (CUDA, Ada/Blackwell, kernel `gated_delta_net` dan `ssm_conv_f32` muncul di profil), [#27954](https://github.com/ggml-org/llama.cpp/discussions/27954) (ROCm lewat Ollama), [#21383](https://github.com/ggml-org/llama.cpp/issues/21383) | Komunitas | ✅ Jalan di Ada, Blackwell, RDNA2. **Pascal tidak diuji oleh mereka** | OQ-009 |
| C-16 | Fork kita punya kernel CUDA `gated_delta_net.cu`, `ssm-conv.cu`, `ssm-scan.cu` dan arsitektur `qwen35` | `ggml/src/ggml-cuda/`, `src/llama-arch.cpp` | Kode | ✅ | OQ-009 |
| C-17 | Kernel itu **jalan di GTX 1070 Ti (SM 6.1)** untuk model `qwen35` 9B | server 9B milik pengguna: `-ngl 99`, 7883 MiB VRAM, ~20.4 tok/s (rata-rata `/metrics`, 9147 token) | Terukur | ✅ untuk 9B di fork. **Belum untuk 27B dan llama.cpp utama** | OQ-009 |
| C-18 | MTP speculative decoding di llama.cpp utama: `--spec-type draft-mtp`, `--spec-draft-n-max N` | [#28196](https://github.com/ggml-org/llama.cpp/issues/28196), [#27954](https://github.com/ggml-org/llama.cpp/discussions/27954) | Komunitas | ✅ ada di utama |OQ-012 |
| C-19 | Fork `08e025c` (8 Mei 2026) **tidak punya** `draft-mtp`: `--spec-type` hanya `none, ngram-cache, ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod` | `common/arg.cpp` baris 3481 | Kode | ✅ Strategi C butuh build utama atau fork lebih baru | OQ-012 |
| C-20 | File lokal `Qwen3.8-9B-Q4_K_M.gguf`: arsitektur `qwen35`, 32 blok, `full_attention_interval=4`, `head_count_kv=4`, `key_length=256`, 5.38 GiB | `gguf inspect` | Terukur | ✅ | - |
| C-21 | "Qwen3.8-9B" di Hugging Face adalah distilasi komunitas ke arsitektur Qwen3.5-9B, bukan rilis Qwen resmi | [empero-ai/Qwen3.8-9B-Distill](https://huggingface.co/empero-ai/Qwen3.8-9B-Distill) | Komunitas | ⚠ Asal file lokal kita tidak diverifikasi. Pakai 9B sebagai pengganti **arsitektur**, bukan pengganti 27B | - |
| C-22 | `iso3` = IsoQuant: rotasi kuaternion 4D + Lloyd-Max 3-bit pada KV cache; blok 128 nilai = 50 byte (3,125 bit/nilai) | `ggml/src/ggml-iso-quant.c`, `ggml-common.h` (`block_iso3_0`) | Kode | ✅ Hanya ada di fork. Belum dicek apakah ada di llama.cpp utama | - |
| C-23 | Dampak kualitas iso3 terhadap jawaban | - | Tanpa sumber | ❌ Belum diukur. Build CPU **segfault** (kode -11) dengan `-ctk iso3`; kode fork menyebut jalur ini khusus CUDA (`src/llama-context.cpp:1689`). Butuh GPU | - |
| C-24 | Fork kita tidak membangun `llama-perplexity`/`llama-quantize` secara bawaan di `build/bin`; keduanya dibangun terpisah di `~/llama-cpu-build` (CPU) | `ls build/bin`, `llama build --target` | Terukur | ✅ | - |

## C. Kinerja

| ID | Klaim | Sumber | Jenis | Status |
|---|---|---|---|---|
| C-25 | MTP pada Qwen3.8-27B: 1.31x (depth 1), 1.50x (depth 2), 1.51x (depth 3), 1.11x (depth 6) di RTX 5090; puncak di depth 2-3 | [#28196](https://github.com/ggml-org/llama.cpp/issues/28196) | Komunitas | ⚠ GPU penuh, bukan hybrid CPU. Target kita ("+30%..+80%") belum punya dasar di hardware ini |
| C-26 | MTP di ROCm (RX 6800): 17.5 -> 19.4 tok/s (depth 2), 14.3 (depth 4) | [#27954](https://github.com/ggml-org/llama.cpp/discussions/27954) | Komunitas | ⚠ |
| C-46 | Backend Vulkan (RADV, RX 6800) pada Qwen3.8-27B: dimuat normal tetapi crash di tengah decode (`vk::DeviceLostError`); ROCm stabil | [#27954](https://github.com/ggml-org/llama.cpp/discussions/27954) | Komunitas | ⚠ Satu laporan, build lama; Pascal tidak diuji |
| C-27 | Efisiensi decode qwen35 terhadap batas bandwidth: ~86% di RTX 4090 (Linux), ~76% di RTX 5090 (Linux), ~28% di 5090 Windows; ~58% di RX 6800 (ROCm) | [#28196](https://github.com/ggml-org/llama.cpp/issues/28196), [#27954](https://github.com/ggml-org/llama.cpp/discussions/27954) | Komunitas | ⚠ |
| C-28 | 9B di 1070 Ti: bandwidth efektif ≈106 GB/s dari 256 GB/s (efisiensi ≈0.41) | `bench analyze --gen-tps 20.4 --gguf Qwen3.8-9B-Q4_K_M.gguf` | Terukur (informal) | ⚠ 20.4 tok/s adalah rata-rata `/metrics` server dengan konteks 114.688 dan KV iso3; bukan benchmark terkontrol |
| C-29 | Total VRAM 9B yang terukur 7883 MiB (7.70 GiB) vs `budget estimate` 6.99 GiB (-9%) pada konfigurasi yang sama | `nvidia-smi`, `budget estimate --kv-type iso3 --ctx 114688 ...` | Terukur | ⚠ Alat meremehkan overhead sekitar 0.7 GiB; anggap overhead nyata ≈ 1.6 GiB di luar bobot dan KV |
| C-30 | Q3-DOWN-XS 7.68 GiB **tidak muat penuh** di 1070 Ti: 7.68 + KV + state + overhead (≈1.6 GiB terukur pada 9B) melebihi 8 GiB | `budget estimate`, C-29 | Turunan | ✅ secara hitungan; jadi strategi B menjadi "hampir penuh di GPU, sebagian kecil di RAM" |

## D. Hardware

| ID | Klaim | Sumber | Jenis | Status | OQ |
|---|---|---|---|---|---|
| C-31 | CPU Ryzen 5 2400G, 4 core / 8 thread, AVX2 | `hw cpu` | Terukur | ✅ | OQ-011 |
| C-32 | RAM 15.53 GiB, swap 4 GiB | `hw ram` | Terukur | ✅ | OQ-011 |
| C-33 | Laju salin RAM 14.15 GB/s (terbaik, 2 thread, satu arah); lalu lintas nyata ≈28 GB/s | `hw membw` | Terukur | ✅ sebagai batas bawah | OQ-011 |
| C-34 | Bandwidth baca CPU **minimal 18.4 GB/s** (0.8B di CPU, 38.2 tok/s x 0.4486 GiB per token) | `bench analyze` | Terukur | ✅ batas bawah yang lebih baik dari C-33 | OQ-011 |
| C-35 | Dual channel **kemungkinan besar**: lalu lintas 28 GB/s melebihi puncak teoretis single channel DDR4-2933 (23.4 GB/s) | C-33 + spesifikasi DDR4 (hitungan: 2933 MT/s x 8 B) | Turunan | ⚠ Bukan konfirmasi. `sudo dmidecode -t memory` atau pemasangan fisik yang menentukan | OQ-011 |
| C-36 | GPU 1070 Ti: 8 GiB, 256 GB/s, 2432 CUDA core, TDP 180 W; daya puncak terukur 196 W | spesifikasi pabrik (tidak dicek ulang); `hw sample` | Tanpa sumber / Terukur | ⚠ Spesifikasi belum disumberkan; 196 W terukur |  |
| C-37 | GPU tidak punya Tensor Core, INT8 DP4A cepat, atau BF16 | - | Tanpa sumber | ⚠ Dari pengetahuan umum; belum disumberkan |  |
| C-38 | FlashAttention (Dao-AILab) tidak mendukung SM 6.1; FA bawaan llama.cpp (`-fa`) di Pascal belum diverifikasi | - | Tanpa sumber | ⚠ Server 9B berjalan dengan `-fa auto`, tetapi apakah FA aktif tidak diperiksa | OQ-017 |
| C-39 | vLLM butuh compute capability ≥ 7.0 (v0.18) dan ≥ 7.5 (v0.24); Pascal (6.1) tidak didukung | [vLLM v0.18.2](https://docs.vllm.ai/en/v0.18.2/getting_started/installation/gpu/), [v0.24.0](https://docs.vllm.ai/en/v0.24.0/getting_started/installation/gpu/) | Resmi | ✅ |  |
| C-40 | GPTQ/AWQ, TensorRT-LLM, EXL2 tidak bermanfaat di Pascal | - | Tanpa sumber | ⚠ Alasan di RFC-001 belum disumberkan |  |

## E. Format dan alat

| ID | Klaim | Sumber | Jenis | Status |
|---|---|---|---|---|
| C-41 | KV q8_0 = 34 B / 32 nilai = 0.531x FP16 (bukan "setengah"); q4_0 = 18 B / 32 = 0.281x; iso3 = 50 B / 128 = 0.195x | tabel ggml di `legacy.py` (`GGML_TYPES`), `block_iso3_0` | Kode | ✅ |
| C-42 | `llama-cli` fork mengabaikan `-no-cnv`; `-st` keluar dengan kode 0 setelah satu jawaban; timing dicetak `[ Prompt: X t/s \| Generation: Y t/s ]` | uji nyata, PR #29 | Terukur | ✅ |
| C-43 | `llama-perplexity` mencetak PPL dengan `+/-` dan KLD dengan `±` | uji nyata, PR #29 | Terukur | ✅ |
| C-44 | Berkas logit KLD ≈ 0.118 GiB per chunk (ctx 512, vocab 248.320); 100 chunk ≈ 11.8 GiB | uji nyata (0.472 GiB / 4 chunk) | Terukur | ✅ untuk vocab ini |
| C-47 | Dampak KV quant pada model `qwen35` 0.8B (IQ4_XS, wikitext-2 test, 4 chunk, ctx 512, KLD terhadap KV f16): q8_0 mean KLD 0.0038 (same-top-p 96.9%); q4_0 0.0200 (91.5%) | `quality kld --kv-type`, PR #29 | Terukur | ⚠ Model kecil, 6 layer attention, konteks pendek; **tidak berlaku otomatis untuk 27B** (OQ-004) |
| C-48 | KLD kuantisasi bobot sebagai pembanding skala: IQ4_XS -> Q3_K_S (requant, hanya uji pipeline) mean KLD 0.348, same-top-p 70.0% | `quality kld`, PR #29 | Terukur | ⚠ Requant dari quant lain; **bukan** klaim kualitas Q3_K_S |
| C-49 | Format log pemuatan: `offloaded %d/%d layers to GPU` dan `<buffer> model buffer size = X MiB` | `src/llama-model.cpp:7752` dan `:7758` (fork `08e025c`); log CPU asli dari `llama-perplexity` | Kode + Terukur | ⚠ Log dengan layer di GPU belum diuji (VRAM penuh); regex cocok dengan string sumber |
| C-50 | Qwen3.8-27B UD-IQ4_XS (unsloth) hybrid `-ngl 36` di 1070 Ti: tg32 2.29 tok/s, pp128 16.7 tok/s; UD-IQ3_XXS `-ngl 45`: tg32 1.15 tok/s. llama.cpp utama `3549c27` | `reports/2026-11-14-27b-run-001.md` | Terukur | ✅ untuk ctx kecil, KV f16 |
| C-51 | MTP (`--spec-type draft-mtp`) pada 27B IQ4_XS hybrid `-ngl 32`: 1.06x (depth 1, terima 93.8%), 1.00x (depth 2), 0.97x (depth 3); keluaran `/completion` identik di temp 0 | `reports/2026-11-14-27b-run-001.md` | Terukur | ✅ satu prompt, 128 token |
| C-52 | GGUF 27B unsloth: `block_count=65` = 64 layer + 1 layer MTP (`nextn_predict_layers=1`) | `gguf-py` pada file lokal | Terukur | ✅ |
| C-53 | Sapuan thread 27B UD-IQ4_XS `-ngl 36` (28 layer di CPU): tg 1.40 / 1.90 / 2.32 / 2.29 / 2.21 tok/s untuk `-t` 2/3/4/6/8. Naik hampir linear sampai 4 core fisik, lalu turun | `reports/2026-11-15-27b-run-002.md` | Terukur | ✅ Ryzen 5 2400G; pada 2-4 thread komputasi CPU menahan, bukan bandwidth RAM (hitungan dari angka, bukan ukur langsung) |
| C-54 | 27B Q4_K_M `-ngl 32`: 2.60 tok/s; Q4_0 `-ngl 33`: 2.39. Dengan `-ngl 99 -ot "blk\.(17..64)\.ffn_(up\|down\|gate)\.weight=CPU"`: **2.85 tok/s** (+9.6%), k=18 OOM | `reports/2026-11-15-27b-run-002.md`, `tools/bench27b/ot_sweep.sh` | Terukur | ✅ bench pendek (pp128, tg32), KV f16, 4 thread |
| C-55 | Kualitas 27B, KLD terhadap Q4_K_M KV f16 (wikitext-2, ctx 512, 20 chunk): KV q8_0 0.0005 (same-top-p 98.9%), KV q4_0 0.0024 (97.4%), KV iso3 0.0102 (95.3%), IQ4_XS 0.0162 (93.8%), Q4_0 0.0301 (92.1%), IQ3_XXS 0.0604 (88.9%). PPL: Q4_K_M 6.919, IQ4_XS 6.949, Q4_0 7.057, IQ3_XXS 7.203 | `reports/2026-11-15-27b-quality-001.md` | Terukur | ⚠ satu teks, 10 240 token; referensi Q4_K_M, bukan BF16; belum mewakili kode atau penalaran panjang |
| C-56 | ctx 16384 di 27B Q4_K_M dengan `-ot`: KV iso3 (FFN 15..64 di CPU) 2.61 tok/s, KV f16 (FFN 9..64) 2.52, KV q8_0 (FFN 12..64) 2.46. MTP depth 1 tidak mempercepat pada `-ot` di KV mana pun yang diuji (iso3+MTP 2.37); iso3+MTP depth 3 dan iso3+MTP+`-ot` FFN 12..64 crash CUDA OOM saat generasi | `HANDOFF-LOG.md`, `reports/2026-11-14-iso3-mtp-main.md` | Terukur | ⚠ satu prompt 6000 token, 64 token generasi |
| C-57 | iso3 (IsoQuant 3-bit KV) diporting ke llama.cpp utama `3549c27` (CPU + CUDA, `patches/llama.cpp/`): `test-iso3` lulus, `test-backend-ops` CUDA0 kasus iso3 28/28, regresi 5041/5041; KLD iso3 vs f16 di 0.8B 0.0495 | PR #38, `reports/2026-11-14-iso3-mtp-main.md` | Terukur | ✅ FA CUDA iso3 memakai konversi ke f16 (belum ada kernel vektor native) |
| C-58 | `-fa on` lebih lambat di Pascal pada ctx pendek: 2.59 vs 2.84 tok/s (27B Q4_K_M, `-ot` FFN 17..64); bawaan `auto` = off | `HANDOFF-LOG.md` (OQ-017) | Terukur | ⚠ satu konfigurasi, ctx pendek |
| C-59 | `IQ3_M` tidak ada di `unsloth/Qwen3.8-27B-GGUF`; pengganti diunduh dan sha256-nya cocok LFS oid: UD-IQ3_S (12 040 883 104 B), UD-Q3_K_XL (13 146 393 504 B) | API HF `tree/main`, `sha256sum` | Terukur | ✅ [CONFLICT] dengan judul #33 dan baris IQ3_M di MASTER/RFC-001 |
| C-60 | 27B ctx 4096 (`-d 4096`), 4 thread, KV f16, TPS generasi terbaik `-ngl` / `-ot`: Q4_K_M 2.04 (ngl 29) / 2.57 (FFN 11..64); IQ4_XS 1.85 (ngl 33) / 2.87 (FFN 21..64); UD-IQ3_S 1.67 (ngl 40) / 2.40 (FFN 33..64); UD-Q3_K_XL 1.85 (ngl 37) / 2.25 (FFN 26..64). VRAM puncak 7879-8051 MiB, RAM tersedia terendah ≥ 13.15 GiB | `reports/2026-11-15-27b-run-003-ctx4096.md`, `tools/bench27b/ctx_bench.py` | Terukur | ⚠ 1 run per titik, bench 16 token; IQ4_XS ngl 32/33/34 tidak monoton (1.09/1.85/0.56) |
| C-61 | OQ-014: `-fitt 512/1024 -fitc 4096` tanpa `-ngl`/`-ot` memberi 1.50-1.97 tok/s; 3-10% di bawah `-ngl` manual terbaik dan 23-41% di bawah `-ot` terbaik; tidak OOM di 6 percobaan | `reports/2026-11-15-27b-run-003-ctx4096.md` | Terukur | ✅ untuk 4 quant ini, satu mesin |
| C-62 | Pada ctx 4096 batas `-ot` Q4_K_M turun dari FFN 17..64 (ctx pendek, 2.85 tok/s, C-54) ke 11..64 (2.57); k=13..17 OOM saat alokasi buffer compute/RS | `reports/2026-11-15-27b-run-003-ctx4096.md` | Terukur | ✅ |
| C-63 | `glslc` didapat tanpa sudo dari LunarG Vulkan SDK 1.4.363.1 (`~/tools`); llama.cpp Vulkan (`36dbad3`) terbangun di `~/llama.cpp/build-vulkan`; mendeteksi GTX 1070 Ti (fp16: 0, matrix cores: none) | `reports/2026-11-15-27b-run-004-vulkan-strategy-b.md` | Terukur | ✅ |
| C-64 | Vulkan vs CUDA, 27B Q4_K_M ctx 4096: ngl 29 2.01 vs 2.04, `-ot` FFN 11..64 2.41 vs 2.57 (-1.5% sampai -6%); Vulkan memakai ±0.9 GiB lebih banyak RAM host dan menampung FFN 13..64 (2.46) di mana CUDA OOM; tidak ada crash decode | `reports/2026-11-15-27b-run-004-vulkan-strategy-b.md` | Terukur | ⚠ 1 run per titik; OOM Vulkan pada ngl besar adalah gagal alokasi, bukan crash |
| C-65 | Q3-DOWN-XS asli (guideboardlabs, 8 491 269 152 B = 7.91 GiB, sha256 cocok LFS) tidak muat 8 GiB: `-ngl 99` OOM di CUDA dan Vulkan; maks layer GPU CUDA 61/66, Vulkan 62/66. [CONFLICT] RFC-001/MASTER menulis 7.68 GiB | `reports/2026-11-15-27b-run-004-vulkan-strategy-b.md` | Terukur | ✅ |
| C-66 | Strategi B TPS ctx 4096: CUDA ngl 60 5.54 ± 0.11, Vulkan ngl 62 7.05 ± 0.07 (3 run). Prediksi 8.69 (18.4 GB/s) dan 14.96 (40 GB/s); MASTER min 8 tidak tercapai; klaim komunitas 24.48 (C-14) tidak tercapai | `reports/2026-11-15-27b-run-004-vulkan-strategy-b.md` | Terukur | ✅ 16 token generasi, satu mesin |
| C-67 | Kualitas Q3-DOWN-XS vs Q4_K_M (wikitext-2, ctx 512, 20 chunk): PPL 8.4615 ± 0.296 (Q4_K_M 6.9192, +22%), mean KLD 0.2728 ± 0.0077, same-top-p 78.2%. Melewati batas PPL 10% MASTER | `reports/2026-11-15-27b-run-004-vulkan-strategy-b.md` | Terukur | ⚠ satu teks, ngl 52 CUDA, referensi Q4_K_M bukan BF16 |
| C-68 | Q3-DOWN-XS CUDA ctx 4096: KV q4_0 memuat 63/66 layer di GPU (f16: 61) dan memberi 6.51 tok/s (f16 5.54, +17%); KV q8_0 maks ngl 62 5.62; `-fa 1` 5.64 vs 5.54 | `reports/2026-11-15-27b-run-004-vulkan-strategy-b.md` | Terukur | ⚠ 1 run, 16 token; KLD KV q4_0 pada model ini belum diukur; iso3 belum |
| C-45 | `llama-perplexity` menjalankan 4 sekuens paralel pada ctx 512 (KV 24 MiB = 4 x 6 MiB) | log nyata | Terukur | ✅ |

| C-69 | Ukuran berkas bobot terukur 2026-11-18 (`stat`, mesin ini): Ternary-Bonsai-2-27B-PTQ1_0 5,946,648,928 B (5.54 GiB), Ternary-Bonsai-2-27B-Q2_0 7,626,008,928 B (7.10 GiB), Qwen3.8-27B-UD-Q4_K_M 16,564,259,744 B (15.43 GiB) | `stat -c %s` pada berkas lokal | Terukur | ✅ ukuran berkas; **bukan** klaim kualitas atau kecepatan |
| C-70 | Batas `mlock` proses 2,035,848 KiB (1.94 GiB) tanpa root; MemTotal 15,905 MiB, MemAvailable 14,560 MiB | `ulimit -l`, `free -m` | Terukur | ✅ |
| C-71 | Batas atas tok/s decode = bandwidth / bytes bobot per token; menempatkan bobot yang **sama** di dalam executable tidak mengubah byte per token. Turunan dari C-34 (≥18.4 GB/s) dan C-54 (2.85 tok/s): FFN ±9.26 GiB di RAM → 1.6–2.0 tok/s; seluruh bobot di VRAM 256 GB/s memberi batas jauh lebih tinggi, searah dengan angka Bonsai di `bench/results/bonsai/` (baris CLAIMS Bonsai belum ada — `[OPEN]` di HANDOFF-LOG) | turunan dari C-34, C-54, MASTER (bandwidth) | Turunan | ⚠ bukan ukuran langsung; overlap dan batching membuat angka praktis sedikit lebih tinggi dari batas bawah |
| C-72 | `cmake/embed.cmake` (hex C-array) membengkakan ±5×: tiap byte menjadi 5 karakter `0xNN,`; payload 5.54 GiB → ±28 GB source C++ | turunan dari format keluaran skrip | Turunan | ⚠ rasio deterministik; dibenarkan oleh pemakaian WebUI 18 KB hari ini, belum diuji penuh pada berkas GB |

## Cara memakai

1. Sebelum menulis klaim faktual di dokumen lain, tambahkan baris di sini dengan ID baru.
2. Jangan mengubah status ke ✅ tanpa sumber atau pengukuran yang bisa diulang.
3. Bila sumber komunitas bertentangan dengan kartu resmi, kartu resmi menang dan ketidakcocokan dicatat di `HANDOFF-LOG.md` sebagai `[CONFLICT]`.
