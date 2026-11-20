# qwen38-engine

Engine inferensi khusus untuk **Qwen3.8-27B** dan **Ternary Bonsai 2 27B** pada GTX 1070 Ti
8 GB (Pascal SM 6.1), Ryzen 5 2400G (Zen 1, AVX2), dan 16 GB DDR4.

Status per 2026-11-19: **Stage 1–5 selesai**, ditambah jalur Ternary Bonsai, kernel ternary v2,
mode konteks 131072, auto-fit GPU, budget memori, state save/load, serta plumbing head MTP yang
bisa disuntikkan ke Bonsai. Riwayat rinci per tahap ada di
[`docs/AFK_STATUS.md`](docs/AFK_STATUS.md).

## Yang diimplementasikan

- Loader GGUF read-only + tokenizer BPE dan chat template.
- Arsitektur hibrida Qwen3.8: 64 layer = 16 x (3 Gated DeltaNet + 1 Gated Attention).
- CPU AVX2/FP32 dan CUDA SM 6.1 dengan GEMV terkuantisasi, `__dp4a`, RMSNorm, SwiGLU, RoPE,
  Gated DeltaNet, rollback state, dan GQA dengan KV `iso3`.
- Prefill bertahap, cache prefix multi-turn, sampler llama.cpp-like, MTP + rollback DeltaNet,
  adaptive MTP, dan PLD.
- Server HTTP/SSE OpenAI-compatible dan WebUI satu berkas.
- Mode Ternary Bonsai `Q2_0` / `PTQ1_0`: dequantisasi CPU dan kernel CUDA, rotasi Hadamard,
  permutasi grouped-V, serta deteksi otomatis dari metadata `prism.hadamard`.
- Kernel ternary v2 dengan A/B byte-identical terhadap kernel awal; LM head GPU dan blok MTP GPU
  otomatis bila muat budget.
- Konteks Attention hingga **131072 posisi** pada jalur Bonsai: tiled online-softmax untuk prefill
  dan split-K 64 slice untuk decode di atas 8192 posisi.
- Auto-fit jumlah layer GPU terhadap budget VRAM, arena host yang diukur dari konfigurasi, dan
  guard budget RAM 14,2 GB.
- State model `--state-save` / `--state-load` untuk KV, state DeltaNet, token cache, dan baris
  tersembunyi MTP.
- Alat opsional `tools/bonsai_mtp_train/` untuk menyiapkan korpus, mendistilasi head MTP, dan
  menyuntikkan `blk.64` ke GGUF Bonsai.

> Batas penting: kapasitas 128K yang sudah diukur adalah **biaya decode pada posisi mendalam**, bukan
> prefill 128K penuh. `--depth-bench` mengosongkan KV sebelum probe (`-n 33` berarti satu warm-up di
> `N-33` plus 32 langkah terukur). Kualitas retrieval pada 128K juga belum diukur.

## Build

Prasyarat:

- CMake 3.20+, Ninja, CUDA Toolkit yang mendukung SM 61, g++ dengan AVX2.
- Build `llama.cpp` yang menyediakan `ggml`, `ggml-base`, `ggml-cpu`, `llama`, dan
  `llama-common`. Default CMake mencari `$HOME/llama.cpp` dan `$HOME/llama.cpp/build-iso3`.
- Fork `iso3` di `engine/llama.cpp` dapat dipakai sebagai sumber baseline.

```bash
# Dari root repo
cmake -S engine/llama.cpp -B "$HOME/llama.cpp/build-iso3" \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_BUILD_TYPE=Release
cmake --build "$HOME/llama.cpp/build-iso3" -j4

cd qwen38-engine
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_SRC="$PWD/../engine/llama.cpp" \
  -DLLAMA_BUILD="$HOME/llama.cpp/build-iso3"
ninja -C build -j4
```

Periksa lingkungan lebih dahulu dengan `scripts/setup.sh`; laporannya ditulis ke
`docs/setup-report.md`. Build engine selalu memakai `ninja -j4`.

## Menjalankan

Semua proses GPU harus memakai empat core fisik (`taskset -c 0-3`) dan tidak boleh berjalan
bersamaan dengan `llama-server` atau engine lain.

### Qwen K-quant

```bash
taskset -c 0-3 ./build/qwen38-engine \
  -m "$HOME/models/27b/Qwen3.8-27B-UD-Q4_K_M.gguf" \
  --gpu-layers auto --ctx 4096 \
  -p "The capital of France is" -n 32 --temp 0 --seed 42
```

Default engine adalah hibrida: 32 layer GPU, KV `iso3`, `__dp4a`, prefill besar di-stream melalui
GPU, dan MTP. `--gpu-layers auto` memilih prefix layer terbesar yang masih memenuhi budget.

### Ternary Bonsai

```bash
taskset -c 0-3 ./build/qwen38-engine \
  -m "$HOME/models/27b-bonsai/Ternary-Bonsai-2-27B-Q2_0.gguf" \
  --gpu-layers auto --ctx 4096 \
  -p "The capital of France is" -n 16 --temp 0 --seed 42
```

Mode ternary dideteksi otomatis dari tipe tensor. `Q2_0` dan `PTQ1_0` didukung; `PQ2_0`
(type 142) belum. GGUF Bonsai asli tidak memiliki blok MTP 64, sehingga MTP nonaktif; PLD tetap
tersedia. Alat `tools/bonsai_mtp_train/inject_mtp_gguf.py` bisa menambahkan `blk.64` uji/trained,
lalu engine memakai head tersebut (opsi `--gpu-head` dan `--gpu-mtp` mengatur penempatan GPU).

### Probe konteks 131072

```bash
taskset -c 0-3 ./build/qwen38-engine \
  -m "$HOME/models/27b-bonsai/Ternary-Bonsai-2-27B-PTQ1_0.gguf" \
  --gpu-layers auto --ctx 131072 --depth-bench 131072 -n 33
```

Perintah ini mengukur biaya decode pada kedalaman 131072 dengan KV kosong: satu warm-up di posisi
131039, lalu 32 langkah terukur di posisi 131040..131071. **Jangan** mengutipnya sebagai
kecepatan prefill konteks penuh.

### Server dan WebUI

```bash
MODEL="$HOME/models/27b/Qwen3.8-27B-UD-Q4_K_M.gguf" scripts/serve.sh --gpu-layers auto --ctx 4096
# http://127.0.0.1:8080/
```

Endpoint utama:

| Endpoint | Fungsi |
|---|---|
| `GET /` | WebUI tertanam |
| `GET /health` | status tanpa menunggu engine |
| `GET /v1/models` | model, konteks, dan vocabulary |
| `GET /v1/stats` | VRAM, RSS, cache, dan konfigurasi |
| `POST /v1/chat/completions` | OpenAI-compatible, streaming SSE atau JSON |
| `POST /v1/cache/clear` | menghapus cache prefix |

Server memproses satu permintaan pada satu waktu. Antarmuka WebUI dan parser `<think>` diuji
dengan QuickJS; belum ada uji browser sungguhan.

## Opsi CLI penting

| Opsi | Arti |
|---|---|
| `-m, --model` melalui `-m` | path GGUF (wajib) |
| `--ctx N` | panjang konteks; maksimum 8192 pada jalur Qwen K-quant, 131072 pada jalur Ternary (C-83) |
| `--gpu-layers N\|auto` | jumlah layer GPU atau auto-fit |
| `--kv iso3\|f32` | format KV GPU; CPU tetap memakai FP32 |
| `--vram-budget-mib N` | budget device memory proses; default 7373 MiB |
| `--resident-gib N` | batas bobot layer CPU yang ditahan di page cache |
| `--mtp-steps 0..3`, `--mtp-adaptive 0\|1` | draft MTP |
| `--mtp-gguf FILE` | head MTP dari sidecar `blk.64.*` (`inject_mtp_gguf.py --mtp-only-out`), di-mmap terpisah; berlaku juga untuk mode server |
| `--gen-mtp-steps 0..3` | jumlah draft untuk run ini saja; engine tetap memuat blok MTP (state file dari engine MTP bisa di-decode tanpa draft) |
| `--pld 0\|1`, `--pld-max N` | prompt lookup decoding |
| `--stream-min-t N` | minimum ukuran batch untuk prefill yang di-stream ke GPU |
| `--state-save FILE`, `--state-load FILE` | simpan/buka state model |
| `--state-test FILE --cont-file FILE` | uji round-trip state dalam satu proses |
| `--gpu-head 0\|1\|auto` | penempatan LM head; auto hanya dengan 64 layer GPU |
| `--gpu-mtp 0\|1\|auto` | penempatan blok MTP; auto mengikuti head dan sisa VRAM |
| `--depth-bench N` | probe biaya decode pada posisi N dengan KV kosong |
| `--rollback-test` | uji rollback DeltaNet bit-identik |
| `--server --host H --port P` | jalankan HTTP/SSE server; dengan `--state-load` state dipakai sebagai checkpoint tetap, dengan `--state-save` tersedia `POST /v1/state/save` dan state disimpan saat SIGINT/SIGTERM |

## Arsitektur dan memori

- `src/packer/`: pembaca GGUF, metadata array, dan ukuran baris format K-quant/ternary.
- `src/cpu/`: GEMV AVX2, Gated DeltaNet, LM head, tokenizer, sampler CPU reference, dequantisasi
  ternary, dan rotasi Hadamard.
- `src/cuda/`: GEMV `__dp4a`, operator FP32, KV `iso3`, DeltaNet GPU, attention shared-memory,
  tiled prefill, split-K decode, serta kernel ternary v2.
- `src/engine/`: arena statis, orchestrator layer, generator, sampler, state file, CLI, server.
- `tools/bonsai_mtp_train/`: korpus, distilasi head MTP, dan penyuntikan GGUF.
- `src/server/`: cpp-httplib + SSE OpenAI-compatible.
- `webui/`: satu `index.html`, tanpa CDN dan tanpa build step.

Budget default:

| Resource | Batas | Mekanisme |
|---|---:|---|
| Device memory | 7373 MiB | proyeksi sebelum alokasi + pemeriksaan setelah load |
| Host RAM | 14,2 GB desimal | arena + LM head/MTP ter-*mlock* + margin + resident layers |
| Shared memory SM 61 | 48 KiB | kernel lama sampai 8192 posisi; kernel tiled di atasnya |
| Token loop | tanpa alokasi heap | arena dibekukan setelah load; slice skor per worker |

Kernels wajib lulus SASS gate: ada `IDP.4A`, tidak ada aritmetika half. Jalankan
`scripts/sass_gate.sh` setelah perubahan `.cu`.

## Hasil terukur terbaru

ID klaim di bawah ada di [`../spec/research/CLAIMS.md`](../spec/research/CLAIMS.md).

### Attention panjang

`tests/cuda/attn_long_test.cu` membandingkan jalur cepat, tiled, dan split-K dengan referensi CPU
double-precision:

| Kasus | Hasil |
|---|---|
| FP32 dan `iso3`, n=300/4096/6000/8192 | fast vs tiled PASS |
| FP32 dan `iso3`, n=100/4096/32768/65536/131072 | split-K dan tiled vs double reference PASS |
| Galat relatif tertinggi | 2,581e-05 (batas test 1e-4; C-77) |

### Probe decode pada kedalaman

| Model | Konteks | Layer GPU | Decode probe | Puncak VRAM |
|---|---:|---:|---:|---:|
| Bonsai `PTQ1_0` | 4096 | 64 | 8,5258 tok/s | 5983 MiB |
| Bonsai `PTQ1_0` | 32768 | 64 | 8,7895 tok/s | 6351 MiB |
| Bonsai `PTQ1_0` | 65536 | 64 | 6,7641 tok/s | 6759 MiB |
| Bonsai `PTQ1_0` | 131072 | 64 | 4,5906 tok/s | 7349 MiB |
| Bonsai `Q2_0` | 4096 | 64 | 8,3228 tok/s | 7315 MiB |
| Bonsai `Q2_0` | 32768 | 62 | 1,6474 tok/s | 7297 MiB |
| Bonsai `Q2_0` | 65536 | 59 | 0,9636 tok/s | 7417 MiB |
| Bonsai `Q2_0` | 131072 | 53 | 0,4655 tok/s | 7295 MiB |

Tabel ini adalah probe **decode-only**. Detail metode dan keluaran mentah ada di
[`bench/results/bonsai128k/README.md`](bench/results/bonsai128k/README.md).

### Generasi nyata, kernel, dan rollback

- Bonsai prompt needle 4K (4032 token) tanpa MTP: `PTQ1_0` 8,1536 tok/s, `Q2_0` 8,2092 tok/s;
  keduanya menjawab `MAGENTA-FALCON-42`.
- Bonsai `PTQ1_0` + `blk.64` Qwen (plumbing): tanpa MTP 11,4932 tok/s; MTP 2 tetap 17,7078 tok/s;
  158/198 draft diterima dan 256 token identik (C-84).
- Kernel ternary v2 byte-identical pada 135 kasus A/B; PTQ1_0 microbench naik dari 15,6 ke 77,6
  GB/s pada ffn_up T=1 (C-85).
- Qwen Q4_K_M, MTP 0: 2,7191 tok/s; rollback DeltaNet PASS bit-identik untuk `keep=1..4`.
- Klaim terkait: C-79 (llama.cpp/PrismML), C-80 (engine Qwen), C-82 (SASS), C-84..C-86.
- Angka engine **tidak boleh dicampur** dengan Bonsai 17–20 tok/s dari llama.cpp/PrismML.

## Test

```bash
ninja -C build -j4
./build/attn_long_test 131072
./build/ternary_test
./build/ternary_ab_test
./build/stop_filter_test
./build/mt_dot_test
./build/foundation_test
./scripts/sass_gate.sh
python3 tests/webui_test.py
env QWEN_MT_DOT=0 python3 tests/server_test.py
python3 ../tools/test_legacy.py
python3 tools/bonsai_mtp_train/prepare_corpus.py --selftest
python3 tools/bonsai_mtp_train/inject_mtp_gguf.py --selftest
python3 tools/bonsai_mtp_train/train_bonsai_mtp.py --selftest
```

`ternary_real_test` memerlukan fixture bobot nyata dari
`bench/results/bonsai/ternary_real_ref/`; fixture tersebut tidak disertakan di repo.

## Status dan batasan

- **Sudah:** implementasi end-to-end, parity pendek dan panjang, server 69/69 (`QWEN_MT_DOT=0`),
  logika WebUI headless, rollback, SASS gate, mode ternary, kernel v2 bit-identical, plumbing MTP,
  dan kapasitas 131072.
- **Belum:** prefill penuh 131072, benchmark kualitas pada konteks panjang, portable state file,
  dukungan `PQ2_0`, head MTP Bonsai terlatih, dan uji WebUI di browser.
- **Belum dijamin:** hasil teks dari MTP/PLD pada `temperature > 0` dapat berbeda dari MTP 0
  karena jalur multi-token CPU menjumlahkan dalam urutan berbeda; greedy tetap stabil.
- Klaim faktual dan sumbernya: [`../spec/research/CLAIMS.md`](../spec/research/CLAIMS.md).
