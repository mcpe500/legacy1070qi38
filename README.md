# legacy1070qi38

*Old Hardware, New Intelligence.* Proyek untuk Kiroween 2026 (tema **Resurrection**): menjalankan
**Qwen3.8-27B** di GPU lama, yaitu **GTX 1070 Ti 8 GB (Pascal, SM 6.1) + RAM 16 GB DDR4**, tanpa
mengasumsikan seluruh model muat di VRAM.

Lisensi: Apache 2.0.

## Status (2026-11-15)

**Qwen3.8-27B sudah berjalan di mesin ini**, dengan sebagian layer di GPU dan sisanya di CPU. Semua angka di bawah terukur
di GTX 1070 Ti 8 GB + Ryzen 5 2400G + 16 GB RAM, dengan llama.cpp utama (`3549c27`) ditambah tipe KV `iso3` buatan kami.
Kecepatannya **pelan**: cukup untuk batch dan eksperimen, kurang nyaman untuk chat.

| Konfigurasi 27B | Token/detik | Catatan |
|---|---:|---|
| UD-IQ3_XXS, hybrid | 1.15 | lebih lambat dari IQ4_XS walau lebih kecil |
| UD-IQ4_XS, hybrid | 2.29 | |
| UD-Q4_K_M, hybrid | 2.60 | |
| **UD-Q4_K_M, semua attention di GPU, FFN layer 17..64 di CPU (`-ot`)** | **2.85** | terbaik terukur, bench pendek |
| Q4_K_M, konteks 16k, KV iso3, `-ot` | 2.61 | |
| MTP (speculative decoding) | +6% paling tinggi | tidak mempercepat bersama `-ot` |

Kualitas (KLD terhadap Q4_K_M, satu teks wikitext-2): KV q8_0 0.0005, KV q4_0 0.0024, KV iso3 0.0102, IQ4_XS 0.0162,
Q4_0 0.0301, IQ3_XXS 0.0604. Itu satu teks, belum mewakili kode atau penalaran panjang.
Rincian, sumber, dan batasnya: [`spec/research/CLAIMS.md`](spec/research/CLAIMS.md) C-50 sampai C-58 dan
[`spec/handoff/SESSION-002-27b-test.md`](spec/handoff/SESSION-002-27b-test.md).

| Sudah ada | Bukti |
|---|---|
| Spesifikasi (SPEC-000..007, RFC-001..003) dan aturan agen | `spec/` |
| Fakta model dari kartu resmi dan hasil ukur, 58 klaim bersumber | [`spec/research/CLAIMS.md`](spec/research/CLAIMS.md) |
| CLI `tools/legacy.py` (budget, hw, gguf, llama, bench, quality, spec) | 127 test unit + 3 test dengan binary llama.cpp asli |
| Agen GitHub `/kiro` (balas issue/PR dari 3 pengguna terdaftar) | `tools/agent/`, 73 test |
| iso3 (KV cache 3-bit) di llama.cpp utama, CPU dan CUDA | fork [`mcpe500/llama.cpp`](https://github.com/mcpe500/llama.cpp) sebagai submodule `engine/llama.cpp`; `test-iso3` lulus, `test-backend-ops` iso3 28/28 |
| Skrip ukur 27B | `tools/bench27b/`, `tools/iso3/` |

| Belum | Alasan |
|---|---|
| Single atau dual channel RAM | Butuh `sudo dmidecode -t memory` oleh pemilik mesin ([#32](https://github.com/mcpe500/legacy1070qi38/issues/32)) |
| Vulkan vs CUDA, Strategi B (Q3-DOWN-XS) | [#34](https://github.com/mcpe500/legacy1070qi38/issues/34): `glslc` tidak ada, Strategi B belum diuji |
| IQ3_M (tidak ada di repo GGUF unsloth), auto-fit vs `-ngl` manual | [#33](https://github.com/mcpe500/legacy1070qi38/issues/33) |
| Kernel FlashAttention iso3 native | iso3 sekarang dikonversi ke f16 sebelum kernel |
| Build dari submodule `engine/llama.cpp` | pengukuran memakai build lokal dari commit yang sama |
| Server Qwen3.8-9B lama | build baru tidak bisa memuat file 9B itu; hidupkan dari fork lama (lihat `HANDOFF-LOG.md`) |

## Mengapa sulit

Qwen3.8-27B adalah model hibrida: 64 layer = 16 x (3 Gated DeltaNet + 1 Gated Attention). Hanya 16 layer yang
punya KV cache (64 KiB per token) dan state DeltaNet tetap sekitar 0.146 GiB, jadi **konteks murah, bobot yang mahal**.
Q4_K_M berukuran 16.5 GiB, sehingga sebagian bobot harus di RAM dan kecepatan dibatasi bandwidth RAM.
Perhitungan dan sumbernya ada di [`spec/AGENTS/MASTER.md`](spec/AGENTS/MASTER.md).

## Mulai cepat

Butuh Python 3.8+ (hanya 3.10 yang diuji). Tanpa dependensi.

```bash
git clone --recurse-submodules https://github.com/mcpe500/legacy1070qi38 && cd legacy1070qi38

# Muat tidak? (tanpa GPU)
python3 tools/legacy.py budget estimate --file-gib 16.53 --vram-gib 8 --ctx 4096

# Ukur mesin Anda
python3 tools/legacy.py hw gpu
python3 tools/legacy.py hw membw

# Baca header GGUF tanpa memuat bobot
python3 tools/legacy.py gguf inspect path/ke/model.gguf

# Jalankan test
python3 tools/test_legacy.py
```

Semua subcommand punya `--json`. Daftar lengkap dan batasnya: [`tools/README.md`](tools/README.md).

## Peta repo

```
spec/AGENTS/      aturan untuk agen: MASTER.md (baca dulu), roles/, phases/, KIRO-USAGE.md
spec/specs/       SPEC-NNN (requirements, design, tasks), RFC-NNN, assets/ (SVG)
spec/research/    CLAIMS.md: setiap klaim faktual dengan sumber dan status
spec/handoff/     HANDOFF-LOG.md (append-only), catatan sesi
tools/legacy.py   CLI satu file, hanya library standar
tools/agent/      pemantau GitHub untuk komentar /kiro
engine/llama.cpp  mesin inferensi: submodule ke fork mcpe500/llama.cpp (branch feat/iso3-kv)
patches/          dua patch iso3 (riwayat); sumber utama sekarang engine/llama.cpp
.kiro/hooks/      hook Kiro (validasi spec saat disimpan)
```

## Mesin (llama.cpp dengan iso3)

Mesin yang dipakai uji 27B adalah fork [`mcpe500/llama.cpp`](https://github.com/mcpe500/llama.cpp), cabang `feat/iso3-kv`. Isinya llama.cpp utama (`ggml-org`, commit `3549c27`) ditambah tipe KV cache `iso3` (CPU + CUDA) dan tesnya. MTP sudah ada di llama.cpp utama. Fork ini masuk ke repo sebagai submodule di `engine/llama.cpp`, terkunci pada satu commit.

```bash
git clone --recurse-submodules https://github.com/mcpe500/legacy1070qi38
# sudah terlanjur clone tanpa submodule:
git submodule update --init engine/llama.cpp

cd engine/llama.cpp
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j4
# pakai dengan tools/legacy.py:
LEGACY_LLAMA_BIN_DIR=$PWD/build-cuda/bin python3 ../../tools/legacy.py llama run --help
```

Perintah `cmake` di atas belum dijalankan dari salinan submodule ini. Build yang dipakai pengukuran dibuat di `~/llama.cpp/build-iso3` dari commit yang sama (`36dbad3`).

## Menjalankan 27B (konfigurasi terbaik terukur)

Disusun dari konfigurasi yang diukur; perintah persis ini belum dijalankan utuh. Butuh GGUF `UD-Q4_K_M` dari
`unsloth/Qwen3.8-27B-GGUF` dan build dari bagian Mesin di atas.

```bash
OT="blk\.($(seq -s '|' 17 64))\.ffn_(up|down|gate)\.weight=CPU"   # FFN layer 17..64 di CPU
engine/llama.cpp/build-cuda/bin/llama-server -m Qwen3.8-27B-UD-Q4_K_M.gguf \
  -ngl 99 -ot "$OT" -t 4 -c 4096 -np 1 --host 127.0.0.1
```

Pakai `-t 4` (jumlah core fisik Ryzen 5 2400G; lebih banyak thread lebih lambat). Jangan nyalakan MTP bersama `-ot`.
Angka 17 diukur di konteks pendek. Di konteks 16k angkanya harus lebih kecil (lebih banyak FFN di CPU) agar VRAM cukup: 15 dengan KV iso3, 9 dengan KV f16.

## Cara berkontribusi

- Satu tugas, satu PR. Deskripsi PR memuat teks dan SVG, bukti test, dan daftar "yang belum".
  Aturan lengkap ada di `MASTER.md`, bagian "Aturan Pull Request".
- Klaim faktual baru wajib punya baris di `spec/research/CLAIMS.md`.
- `main` dilindungi; merge hanya oleh pemilik repo.
- Komentar `/kiro ...` di issue atau PR dibalas agen, tetapi hanya dari pengguna terdaftar.

## Penggunaan Kiro

Dicatat apa adanya di [`spec/AGENTS/KIRO-USAGE.md`](spec/AGENTS/KIRO-USAGE.md): Kiro CLI, planning agent,
sub-agent, steering langsung, spec-driven development, web search/fetch, headless mode dengan custom agent,
MCP (Sequential Thinking), hook, dan knowledge base. Fitur yang belum dipakai ditandai jelas.
