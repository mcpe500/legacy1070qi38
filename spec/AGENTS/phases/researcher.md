# Phase Agent: Researcher

> Baca `spec/AGENTS/MASTER.md` terlebih dahulu sebelum dokumen ini.

## Kapan Phase Ini Aktif

Phase Researcher aktif ketika:
- Memulai spec baru yang belum ada basis teknis
- Ada open question di handoff log yang perlu dijawab
- Ada klaim teknis yang perlu diverifikasi (terutama yang diberi ⚠️ ASSUMPTION)
- Hardware behavior baru ditemukan dan perlu dipahami

## Tanggung Jawab

Agent dalam phase ini bertugas **mengumpulkan informasi dan membangun fondasi pengetahuan** sebelum desain atau implementasi dimulai.

Output utama phase ini:
1. Jawaban terhadap open questions yang terdaftar di handoff log
2. Draft atau update RFC jika ditemukan keputusan arsitektur baru
3. Update ASSUMPTION menjadi VERIFIED (atau WRONG) berdasarkan temuan
4. Brief teknis untuk diserahkan ke phase Implementer

## Metode Riset

### 1. Verifikasi Model Architecture

Untuk memverifikasi arsitektur Qwen3 (yang masih ASSUMPTION):

```bash
# Download model card / config dari HuggingFace
# Cek config.json untuk:
# - num_hidden_layers (layer count)
# - num_key_value_heads (KV heads)
# - hidden_size (hidden dim)
# - intermediate_size (FFN dim)
# - vocab_size

# Contoh untuk model yang sudah didownload:
python3 -c "
import json
with open('path/to/config.json') as f:
    cfg = json.load(f)
print('Layers:', cfg.get('num_hidden_layers'))
print('KV heads:', cfg.get('num_key_value_heads'))
print('Hidden dim:', cfg.get('hidden_size'))
print('FFN dim:', cfg.get('intermediate_size'))
print('Vocab:', cfg.get('vocab_size'))
"
```

### 2. Verifikasi GGUF File

Untuk memverifikasi arsitektur dari file GGUF:

```bash
# Gunakan llama.cpp built-in info
./llama-cli -m model.gguf --no-warmup -n 0 2>&1 | head -50
# Output akan menampilkan: n_layer, n_head, n_head_kv, n_embd, dll.
```

### 3. Benchmark Eksplorasi

Sebelum benchmark resmi (Phase Benchmarker), lakukan quick exploration:

```bash
# Quick TPS test
echo "Hello world" | ./llama-cli -m model.gguf -ngl 99 -n 50 --log-disable 2>&1 | tail -5
```

### 4. Memory Profiling Awal

```bash
# Monitor VRAM selama loading
watch -n 0.5 'nvidia-smi --query-gpu=memory.used --format=csv,noheader'
```

## Output Format

Setelah selesai research, buat dokumen `spec/handoff/SESSION-NNN-[deskripsi].md` dengan:

```markdown
## Research Findings — [Topik]

### Questions Answered
- Q: [pertanyaan] → A: [jawaban] (VERIFIED/ESTIMATED)

### Assumptions Updated
- [ASSUMPTION lama] → [STATUS: VERIFIED CORRECT / WRONG / PARTIALLY CORRECT]
  - Evidence: [dasar verifikasi]

### New Open Questions
- [ ] [pertanyaan baru yang muncul]

### Recommended Actions
- [ ] [action yang direkomendasikan untuk phase selanjutnya]
```

## Referensi

- `spec/AGENTS/MASTER.md` — daftar open questions di bagian bawah
- `spec/handoff/HANDOFF-LOG.md` — open questions dari sesi-sesi sebelumnya
- `spec/specs/RFC-001-quantization-strategy.md` — asumsi quantization yang perlu diverifikasi
