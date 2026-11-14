# Phase Agent: Implementer

> Baca `spec/AGENTS/MASTER.md` terlebih dahulu sebelum dokumen ini.

## Kapan Phase Ini Aktif

Phase Implementer aktif ketika:
- Ada spec dengan status `Approved` yang siap diimplementasikan
- Task list di `tasks.md` sudah ada dan jelas
- Research phase sudah selesai untuk spec tersebut (tidak ada blocking open questions)
- Phase Researcher sudah memberikan brief teknis

## Tanggung Jawab

Agent dalam phase ini bertugas **mengimplementasikan spec menjadi kode yang berjalan**.

Output utama phase ini:
1. Kode yang berfungsi sesuai requirements di spec
2. Test (minimal manual verification, ideally automated test)
3. Update status spec dari `Approved` → `Implemented`
4. Demo yang bisa dijalankan dan diverifikasi

## Pre-Implementation Checklist

Sebelum mulai kode, verifikasi:

- [ ] Sudah baca `spec/AGENTS/MASTER.md`
- [ ] Sudah baca spec yang akan diimplementasikan (requirements.md, design.md, tasks.md)
- [ ] Sudah baca `spec/handoff/HANDOFF-LOG.md` untuk state terkini
- [ ] Tidak ada blocking conflict dengan spec lain
- [ ] Semua dependencies spec sudah `Implemented`
- [ ] Hardware tersedia dan berfungsi (`nvidia-smi` jalan, CUDA terdeteksi)

## Implementation Rules

### Kode harus:
1. **Berjalan di Pascal SM 6.1** — test dengan GTX 1070 Ti, bukan GPU lain
2. **Tidak menggunakan forbidden approaches** — lihat daftar di MASTER.md
3. **Menyertakan comment** yang menjelaskan trade-off dan Pascal-specific decisions
4. **Tidak melebihi memory budget** — lihat `roles/memory-manager.md`

### Komentar wajib untuk Pascal-specific code:

```python
# NOTE: Pascal (SM 6.1) does not support Flash Attention.
# Using standard SDPA kernel instead. See RFC-001 and MASTER.md.

# NOTE: BF16 is not natively supported on Pascal.
# All weights loaded as FP16 and computed in FP32 accumulators.
```

### Error handling untuk OOM:

```python
try:
    # inference call
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        # Log ke handoff, suggest kurangi -ngl atau ctx-size
        raise OOMError(f"VRAM exhausted. Try reducing -ngl or --ctx-size. Details: {e}")
```

## Task Completion Protocol

Untuk setiap task di `tasks.md`:

1. Implementasikan task
2. Jalankan verification yang disebutkan di task
3. Jika verification pass → centang task `[x]` di tasks.md
4. Jika verification fail → debug, jangan lanjut ke task berikutnya
5. Update handoff log dengan progress

## Demo Requirements

Setiap spec yang selesai diimplementasikan HARUS bisa di-demo:

```bash
# Minimal demo command yang bisa dijalankan user
# Contoh untuk SPEC-002:
python tools/legacy.py llama run \
  --model models/Qwen3.8-27B-Q4_K_M.gguf \
  --prompt "What is a large language model?" \
  --ctx 2048
```

Demo harus:
- Bisa dijalankan dengan satu command
- Menampilkan TPS di output
- Menampilkan VRAM usage
- Selesai dalam waktu wajar (<60 detik untuk 100 token)

## Referensi

- `spec/specs/SPEC-002-inference-baseline/tasks.md` — contoh task list
- `spec/AGENTS/roles/inference-optimizer.md` — parameter tuning
- `spec/AGENTS/roles/memory-manager.md` — memory constraints
- `spec/AGENTS/MASTER.md` — forbidden approaches
