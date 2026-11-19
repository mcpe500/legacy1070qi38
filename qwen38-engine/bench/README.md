# Benchmark

Benchmark engine khusus dan perbandingan sequensial dengan llama.cpp. Dua runtime **tidak boleh
berjalan bersamaan**; GPU harus kosong sebelum tiap run.

## Instrumen utama

| Skrip | Fungsi |
|---|---|
| `compare_llamacpp.sh` | head-to-head Qwen Q4_K_M: prompt 509 token, 256 token decode, llama.cpp lalu `qwen38-engine` |
| `bonsai_ctx_run.sh` | satu run engine dengan sampler VRAM `nvidia-smi` tiap 100 ms |
| `bonsai128k_queue.sh` | antrian berurutan probe kedalaman, needle, MTP, dan state 128K |
| `make_long_prompt.py` | membuat prompt needle + source tree dengan token target yang tepat |
| `bonsai_bench.sh` | benchmark runtime llama.cpp/PrismML untuk model Bonsai |
| `bonsai_intel*.sh`, `bonsai_intel_summary.py` | benchmark kualitas/log-likelihood Bonsai |
| `bench_summary.py` | merangkum JSON run dan membuat tabel Markdown |

## Head-to-head Qwen

```bash
bench/compare_llamacpp.sh
```

Hasil utama: [`results/final_comparison.md`](results/final_comparison.md). Run standar memakai
prompt 509 token, 256 token decode, temperature 0, seed 42, KV `iso3`, MTP 2, empat core fisik,
dan VRAM maksimum 7373 MiB. Output mentah berada di `results/stage5/`.

## Konteks panjang Ternary Bonsai

Ringkasan terpusat: [`results/bonsai128k/README.md`](results/bonsai128k/README.md).

### Probe decode pada kedalaman

```bash
bench/bonsai_ctx_run.sh PTQ1_0_ctx131072_depth \
  -m "$HOME/models/27b-bonsai/Ternary-Bonsai-2-27B-PTQ1_0.gguf" \
  --gpu-layers auto --ctx 131072 --depth-bench 131072 -n 33
```

Ulangi dengan `Q2_0` dan konteks 4096, 32768, serta 65536. Probe:

1. me-reset DeltaNet;
2. mengosongkan seluruh KV GPU dan CPU;
3. menjalankan satu warm-up di posisi `depth - 33`, lalu mengukur 32 langkah decode di posisi `depth - 32 ... depth - 1` (perintah memakai `-n 33`).

Jadi angka `decode_tok_s` mengukur **biaya attention pada kedalaman tersebut**, bukan
prefill prompt sepanjang konteks. KV yang kosong membuat Attention membaca rentang penuh dengan
nilai nol.

### Generasi nyata pada 4K

```bash
bench/bonsai_ctx_run.sh Q2_0_ctx4k_short \
  -m "$HOME/models/27b-bonsai/Ternary-Bonsai-2-27B-Q2_0.gguf" \
  --gpu-layers auto --ctx 4096 -p "The capital of France is" -n 16
```

### Prompt panjang dan state

```bash
python3 bench/make_long_prompt.py "$MODEL" 4096 /tmp/bonsai-4k.txt \
  --prefix-out /tmp/bonsai-prefix.txt

./build/qwen38-engine -m "$MODEL" --parse-special -f /tmp/bonsai-prefix.txt \
  --state-save /tmp/bonsai-state.bin --ctx 4096
./build/qwen38-engine -m "$MODEL" --parse-special -f /tmp/bonsai-question.txt \
  --state-load /tmp/bonsai-state.bin --ctx 4096 -n 16
```

`make_long_prompt.py` memakai tokenizer engine sendiri dan corpus source tree. `prompt_code.txt`
adalah potongan corpus kecil untuk uji needle.

## Kernel dan MTP Bonsai

```bash
./build/ternary_ab_test      # byte-identity kernel v2 vs referensi
./build/ternary_bench        # GB/s PTQ1_0/Q2_0 ref vs v2
./build/qwen38-engine -m "$HOME/models/27b-bonsai/Ternary-Bonsai-2-27B-PTQ1_0-mtpqwen.gguf" \
  --gpu-layers auto --ctx 4096 -f bench/prompt512.txt -n 256 --mtp-adaptive 0 --mtp-steps 2
```

Alat penyuntikan/distilasi ada di `tools/bonsai_mtp_train/`. Untuk angka plumbing terukur lihat
`results/bonsai128k/README.md`.

## Bonsai: runtime lain dan kualitas

- Perbandingan runtime: [`results/ternary_bonsai_comparison.md`](results/ternary_bonsai_comparison.md).
- Log eksperimen 27 langkah: [`results/bonsai/EXPERIMENT-LOG.md`](results/bonsai/EXPERIMENT-LOG.md).
- Ledger setiap run dan VRAM: [`results/bonsai/RUNS.md`](results/bonsai/RUNS.md).
- Kualitas: [`results/bonsai/intel/INTEL.md`](results/bonsai/intel/INTEL.md).

Angka 17–20 tok/s pada tabel Bonsai berasal dari llama.cpp/PrismML, bukan `qwen38-engine`.
Jangan menggabungkannya dengan probe 128K engine.

## Aturan pengukuran

- Temperature 0, seed 42, empat core fisik (`taskset -c 0-3`).
- Hanya satu proses GPU; periksa `nvidia-smi --query-compute-apps` dan `pgrep` sebelum run.
- Puncak VRAM adalah sampel `nvidia-smi` tertinggi, bukan angka RAM GPU statis.
- Budget proses: 7373 MiB VRAM dan 14,2 GB RAM host.
- File `.vram.csv`, `.err`, `.server.log`, dan log antrian adalah artefak mentah lokal dan tidak
  boleh di-commit; yang dilacak adalah ringkasan `.txt`/Markdown dan tabel hasil.

Baca dulu `.agents/skills/webui-and-parity-bench/SKILL.md` dan
`.agents/skills/memory-budget-24gb/SKILL.md`.
