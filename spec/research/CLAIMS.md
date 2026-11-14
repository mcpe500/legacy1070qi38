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
| C-45 | `llama-perplexity` menjalankan 4 sekuens paralel pada ctx 512 (KV 24 MiB = 4 x 6 MiB) | log nyata | Terukur | ✅ |

## Cara memakai

1. Sebelum menulis klaim faktual di dokumen lain, tambahkan baris di sini dengan ID baru.
2. Jangan mengubah status ke ✅ tanpa sumber atau pengukuran yang bisa diulang.
3. Bila sumber komunitas bertentangan dengan kartu resmi, kartu resmi menang dan ketidakcocokan dicatat di `HANDOFF-LOG.md` sebagai `[CONFLICT]`.
