# Design — SPEC-002: Inference Baseline

> Spec: SPEC-002 | Status: **Ready to Implement (direvisi)**
> Referensi: RFC-001, SPEC-000

---

## Alur

```
1. Build llama.cpp (CUDA, SM 61)
2. Model uji kecil, penuh di GPU  → memastikan jalur CUDA dan pengukuran bekerja
3. Baca header Qwen3.8-27B GGUF   → arsitektur, layer, tipe attention
4. Muat Qwen3.8-27B parsial       → cari tahu apakah kernel Gated DeltaNet jalan di SM 6.1
```

Langkah 4 adalah risiko terbesar proyek. Jika kernel tidak mendukung SM 6.1, jalur CUDA gugur dan
Vulkan (strategi D) menjadi jalur utama.

---

## Build

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp          # catat commit hash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
./build/bin/llama-cli --version
```

Prasyarat: `build-essential`, `cmake`, `git`, CUDA Toolkit (`nvcc --version`).

> ⚠️ ASSUMPTION: versi CUDA Toolkit terbaru masih menyertakan SM 6.1. Cek catatan rilis toolkit yang dipasang.

---

## Model Uji Kecil

Tujuan: memvalidasi build dan pengukuran, bukan mengejar angka. Pilih GGUF kecil yang lazim (misalnya
kelas 4B–8B pada Q4_K_M). Angka TPS-nya **tidak** menjadi target proyek.

```bash
./build/bin/llama-cli -m models/<model-uji>.gguf -ngl 99 -c 2048 -p "Jelaskan apa itu LLM." -n 100
```

---

## Qwen3.8-27B

Sumber GGUF: repo komunitas, misalnya [6block/Qwen3.8-27B-GGUF](https://huggingface.co/6block/Qwen3.8-27B-GGUF)
(pakai backbone saja; `mmproj` tidak perlu untuk teks).

```bash
# Dua cara offload, catat keduanya
./build/bin/llama-cli -m models/Qwen3.8-27B-Q4_K_M.gguf -c 2048 -p "Halo" -n 50
./build/bin/llama-cli -m models/Qwen3.8-27B-Q4_K_M.gguf -c 2048 -ngl 28 -p "Halo" -n 50
```

Yang dicatat: pesan saat model dimuat (jumlah layer di GPU, ukuran buffer), ada tidaknya galat kernel,
RAM dan VRAM saat berjalan, TPS.

---

## Edge Cases

| Kasus | Penanganan |
|---|---|
| Galat arsitektur tidak dikenal | Update llama.cpp ke build terbaru, catat commit |
| Galat kernel atau SM tidak didukung | Catat pesan lengkap; coba build Vulkan |
| VRAM OOM | Turunkan `-ngl` atau konteks |
| RAM habis, swap aktif | Turunkan ukuran quant |
| Output tidak koheren | Cek hash file; coba quant lain |

---

## Open Questions

- [ ] Model uji kecil mana yang dipakai?
- [ ] Apakah build llama.cpp terbaru mendukung Gated DeltaNet di CUDA SM 6.1?
- [ ] Apakah auto-fit layer lebih baik dari `-ngl` manual?
