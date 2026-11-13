# Role Agent: Benchmarker

> Baca `spec/AGENTS/MASTER.md` terlebih dahulu sebelum dokumen ini.

## Tanggung Jawab

Agent dengan role ini bertanggung jawab atas **pengukuran, evaluasi, dan dokumentasi performa** — menghasilkan angka nyata (bukan estimasi) yang bisa dijadikan dasar keputusan teknis.

Scope tanggung jawab:
- Desain benchmark methodology yang reproducible
- Eksekusi benchmark dan pencatatan hasil
- Analisis hasil: TPS, latency, memory usage, kualitas output
- Perbandingan konfigurasi (A/B testing)
- Dokumentasi hasil ke spec dan handoff log
- Identifikasi regresi performa

## Yang TIDAK Termasuk Scope

- Optimasi berdasarkan hasil benchmark → lihat `roles/inference-optimizer.md`
- Perubahan memory strategy → lihat `roles/memory-manager.md`
- Interpretasi kualitas output dari sisi model → domain expert user

## Metrics yang Diukur

### Primary Metrics

| Metrik | Unit | Cara Ukur | Target Qwen3-8B | Target Qwen3-27B |
|---|---|---|---|---|
| **Decode TPS** | tok/s | llama.cpp built-in stats | ≥10 tok/s | ≥2 tok/s |
| **Prefill speed** | tok/s | llama.cpp built-in stats | ≥200 tok/s | ≥50 tok/s |
| **VRAM usage** | MB | `nvidia-smi` saat inference | ≤7500 MB | ≤7500 MB |
| **RAM usage** | MB | `/proc/meminfo` atau `htop` | ≤4000 MB | ≤15000 MB |
| **Time to first token** | ms | Manual timing atau wrapper | ≤5000 ms | ≤15000 ms |

### Secondary Metrics

| Metrik | Unit | Cara Ukur |
|---|---|---|
| **GPU utilization** | % | `nvidia-smi dmon` |
| **GPU memory utilization** | % | `nvidia-smi dmon` |
| **CPU utilization** | % | `htop` / `sar` |
| **Thermal throttling** | boolean | `nvidia-smi -q -d PERFORMANCE` |
| **PCIe transfer rate** | GB/s | `nvidia-smi dmon -s t` |

## Benchmark Protocol

### Standar Benchmark Run

Setiap benchmark run HARUS mengikuti protokol ini untuk reproducibility:

```bash
# 1. Catat environment sebelum run
nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version \
  --format=csv,noheader
cat /proc/cpuinfo | grep "model name" | head -1
free -h
uname -r

# 2. Warm-up run (tidak dicatat)
echo "Warm-up" | ./llama-cli [params] -n 20

# 3. Benchmark run (3x untuk stabilitas)
for i in 1 2 3; do
  echo "=== Run $i ===" 
  echo "[PROMPT]" | ./llama-cli [params] \
    --log-disable \
    -n 100 \
    2>&1 | grep -E "eval time|prompt eval time|load time"
done

# 4. Catat VRAM peak
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

### Prompt Standard untuk Benchmark

Gunakan prompt yang sama untuk semua benchmark agar comparable:

```
# Short prompt (32 tokens)
"Explain what a transformer neural network is in one paragraph."

# Medium prompt (128 tokens)  
"You are an AI assistant. The user asks: What are the key differences between 
quantization approaches for large language models? Please explain Q4_K_M, Q5_K_M, 
and Q6_K quantization formats, their trade-offs in terms of model quality and 
memory usage, and when you would choose each one."

# Long prompt (512 tokens)
[Gunakan file benchmark/prompts/long-512.txt — dibuat saat SPEC-006 diimplementasi]
```

### Output Format Hasil Benchmark

Setiap hasil benchmark WAJIB didokumentasikan dalam format ini:

```markdown
## Benchmark Result — [Tanggal] — [Model] — [Quant]

**Environment:**
- GPU: GTX 1070 Ti (8GB GDDR5)
- RAM: 16 GB DDR4
- llama.cpp version: [commit hash]
- CUDA: [versi]
- Driver: [versi]

**Configuration:**
```bash
[command yang digunakan]
```

**Results:**

| Run | Decode TPS | Prefill TPS | TTFT (ms) | VRAM Peak | RAM Peak |
|---|---|---|---|---|---|
| Run 1 | X tok/s | X tok/s | X ms | X MB | X MB |
| Run 2 | X tok/s | X tok/s | X ms | X MB | X MB |
| Run 3 | X tok/s | X tok/s | X ms | X MB | X MB |
| **Avg** | **X tok/s** | **X tok/s** | **X ms** | **X MB** | **X MB** |

**Notes:**
- [Observasi penting]
- [Thermal throttling: Ya/Tidak]
- [OOM events: Ya/Tidak]
```

## Evaluasi Kualitas Output

TPS tinggi tidak berguna jika output tidak koheren. Gunakan evaluasi sederhana:

### Quick Quality Check

Setelah benchmark TPS, jalankan evaluasi kualitas dengan prompt:

```
"Siapakah presiden Indonesia pertama? Jawab dalam satu kalimat."
```

Expected: Jawaban yang benar dan koheren dalam Bahasa Indonesia.

Jika output inkoherent atau nonsensical → catat sebagai `[QUALITY FAIL]` di handoff log.

### Perbandingan Kualitas Antar Quant

Gunakan rating sederhana: `✅ Good` | `⚠️ Degraded` | `❌ Broken`

| Quant | Coherence | Factuality | Instruction Following |
|---|---|---|---|
| Q6_K | ✅ | ✅ | ✅ |
| Q5_K_M | ✅ | ✅ | ✅ |
| Q4_K_M | ✅ | ⚠️ | ✅ |
| Q3_K_M | ⚠️ | ⚠️ | ⚠️ |
| Q2_K | ❌ | ❌ | ⚠️ |

> ⚠️ ASSUMPTION: Rating di atas berdasarkan literatur umum. Harus divalidasi empiris untuk Qwen3 specifically.

## Referensi Spec

- `spec/specs/SPEC-006-benchmark-suite/` — implementasi benchmark suite lengkap
- `spec/specs/SPEC-002-inference-baseline/` — baseline yang akan di-benchmark pertama
- `spec/handoff/HANDOFF-LOG.md` — tempat mencatat semua hasil benchmark
