# Design — SPEC-002: Inference Baseline

> Spec: SPEC-002 | Status: **Ready to Implement**
> Referensi: RFC-001, SPEC-000

---

## Architecture Overview

```
GTX 1070 Ti (8GB VRAM, Pascal SM 6.1)
│
└── llama.cpp (compiled: GGML_CUDA=ON, CUDA_ARCH=61)
    ├── GGUF Model Loader
    │   └── Qwen3-8B Q5_K_M (.gguf file, ~6.1 GB)
    │       └── Mmap atau eager load ke system RAM → GPU transfer
    ├── CUDA Backend (ggml-cuda)
    │   ├── Dequantization kernels (Q5_K_M → FP16)
    │   ├── CUDA core FP32 matmul (no tensor cores)
    │   └── Standard SDPA attention (no flash attention)
    ├── KV Cache (q8_0 quantized, ~288 MB at 2048 ctx)
    └── Sampler + Output
```

---

## Build Configuration

### Dependencies

```bash
# Ubuntu/Debian
sudo apt install build-essential cmake git libcurl4-openssl-dev

# CUDA Toolkit (11.x atau 12.x)
# Install dari https://developer.nvidia.com/cuda-downloads
# Verifikasi: nvcc --version
```

### Clone dan Compile

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
git checkout [stable tag atau main]  # catat commit hash untuk reproducibility

cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=61 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j$(nproc)

# Verifikasi
./build/bin/llama-cli --version
# Expected output berisi: "CUDA", "GGUF", versi
```

> ⚠️ NOTE: `-DCMAKE_CUDA_ARCHITECTURES=61` adalah flag WAJIB untuk Pascal. Tanpa ini, llama.cpp mungkin tidak menghasilkan kernel yang optimal untuk SM 6.1.

---

## Runtime Configuration

### Qwen3-8B Baseline Command

```bash
./build/bin/llama-cli \
  -m /path/to/qwen3-8b-q5_k_m.gguf \
  -ngl 99 \
  --ctx-size 2048 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --temp 0.7 \
  --top-p 0.9 \
  --threads 1 \
  -p "What is a large language model? Explain briefly."
```

### Parameter Rationale

| Parameter | Value | Alasan |
|---|---|---|
| `-ngl 99` | 99 | Load semua layer ke VRAM (Qwen3-8B Q5_K_M ~36 layers, semua muat) |
| `--ctx-size 2048` | 2048 | Safe default: hindari VRAM OOM, bisa naik ke 4096 jika headroom cukup |
| `--cache-type-k q8_0` | q8_0 | Hemat ~144 MB VRAM vs FP16 KV cache pada 2048 ctx |
| `--threads 1` | 1 | Minimal CPU threads — inference sepenuhnya di GPU |
| `--flash-attn` | tidak dipakai | Pascal tidak support — akan diabaikan atau warning |

---

## Model Download

```bash
# Option 1: HuggingFace CLI
pip install huggingface-hub
huggingface-cli download Qwen/Qwen3-8B-GGUF \
  --include "qwen3-8b-q5_k_m.gguf" \
  --local-dir ./models/

# Option 2: Direct wget (jika URL tersedia)
# wget -P ./models/ [url]

# Verifikasi integritas
# ls -lh ./models/qwen3-8b-q5_k_m.gguf
# Expected: ~6.1 GB
```

> ⚠️ NOTE: Nama file GGUF untuk Qwen3 mungkin berbeda tergantung publisher. Cek HuggingFace untuk nama file yang tepat saat model tersedia.

---

## Expected Output

Saat inference berjalan, output llama.cpp akan menampilkan:

```
llama_model_load: ...
llm_load_tensors: ...
  - GPU 0: [layers offloaded] layers
...
llama_perf_context_print:        load time =   XXX.XX ms
llama_perf_context_print:      sample time =   XXX.XX ms /   N tokens (   X.XX ms per token,   X.XX tokens per second)
llama_perf_context_print: prompt eval time =   XXX.XX ms /   N tokens (   X.XX ms per token,   X.XX tokens per second)
llama_perf_context_print:        eval time =   XXX.XX ms /   N tokens (   X.XX ms per token,   X.XX tokens per second)
```

**Yang perlu dicatat:**
- `eval time ... tokens per second` → ini adalah **decode TPS** (target ≥ 10 tok/s)
- `prompt eval time ... tokens per second` → ini adalah **prefill TPS**

---

## Edge Cases dan Error Handling

| Kasus | Penanganan |
|---|---|
| CUDA not found saat compile | Pastikan `nvcc` ada di PATH; re-install CUDA toolkit |
| VRAM OOM saat load model | Kurangi `-ngl` atau ganti ke Q4_K_M |
| Output incoherent/gibberish | Cek apakah model file corrupt (hash check); coba temp lebih rendah |
| TPS di bawah 5 tok/s | Cek GPU utilization; pastikan `-ngl 99` aktif; pastikan compile SM 61 |
| llama.cpp tidak detect GPU | `CUDA_VISIBLE_DEVICES=0`; cek `nvidia-smi` |

---

## Open Questions

- [ ] Nama exact GGUF file untuk Qwen3-8B dari HuggingFace? (bergantung publisher)
- [ ] Apakah llama.cpp versi terbaru memiliki breaking changes untuk Pascal?
- [ ] Berapa ctx-size maksimum sebelum VRAM OOM untuk Q5_K_M? (antara 2048–4096)
