# tools/legacy.py

CLI untuk proyek legacy1070qi38. Satu file, hanya library standar Python. Setiap subcommand punya `--json`.
Spec: `spec/specs/SPEC-007-cli-tooling/`. Usulan desain: `spec/specs/RFC-002-cli-tooling.md` (masih Proposed).

![Isi legacy.py](../spec/specs/assets/spec-007-cli-implementation.svg)

> Status: kode dan test ditulis oleh agen tanpa shell. Hasil test harus dikonfirmasi dengan menjalankannya (lihat bawah).
> Versi Python minimum belum diputuskan; kode diasumsikan jalan di 3.7+.

## Contoh

```bash
# Estimasi memory (tanpa GPU)
python tools/legacy.py budget estimate --file-gib 16.53 --vram-gib 8 --ctx 4096 --json
python tools/legacy.py budget suggest-ngl --file-gib 16.53 --layers 64
python tools/legacy.py budget compare --file-gib 16.53 --measured-ram-gib 12.5 --measured-tps 1.4

# Hardware
python tools/legacy.py hw gpu --json
python tools/legacy.py hw headroom --file-gib 16.53        # exit 1 bila tidak cukup
python tools/legacy.py hw membw

# Header GGUF (tidak memuat bobot)
python tools/legacy.py gguf inspect models/Qwen3.8-27B-Q4_K_M.gguf --json
python tools/legacy.py gguf tensors models/Qwen3.8-27B-Q4_K_M.gguf

# llama.cpp (tambah --dry-run untuk melihat argv saja)
python tools/legacy.py llama build --src ../llama.cpp --dry-run
python tools/legacy.py llama run --model models/Qwen3.8-27B-Q4_K_M.gguf --prompt "Halo" --ctx 2048
python tools/legacy.py llama bench --model models/Qwen3.8-27B-Q4_K_M.gguf --runs 3
python tools/legacy.py llama placement --log load.log

# Dokumen
python tools/legacy.py spec validate
python tools/legacy.py spec new nama-fitur --title "Nama Fitur"
python tools/legacy.py spec handoff --type finding --text "ringkasan satu baris"
python tools/legacy.py spec questions
```

## Exit code

| Kode | Arti |
|---|---|
| 0 | Berhasil |
| 1 | Cek gagal: no-go, validasi gagal, di luar toleransi, proses llama keluar ≠ 0 |
| 2 | Input salah (termasuk lock sedang dipegang, timeout) |
| 3 | `nvidia-smi`, binary, atau file tidak ada |

## Pengaman

- Tanpa shell: argumen proses berupa list.
- Binary hanya `llama-cli`, `llama-bench`, `llama-server` di `--bin-dir`.
- Satu inference sekaligus lewat lock file.
- `spec` menulis hanya di bawah `spec/`; `HANDOFF-LOG.md` hanya ditambah di akhir.

## Test

```bash
python tools/test_legacy.py -v
# atau
python -m unittest discover -s tools -p "test_legacy.py" -v
```

Catatan: test `llama` memakai skrip shell palsu di folder sementara (butuh folder tanpa `noexec`). Test GPU hanya
berjalan di mesin tanpa `nvidia-smi` (memeriksa exit 3). Test kernel Gated DeltaNet di Pascal (OQ-009) bukan bagian dari
alat ini.

## Yang belum ada

`llama compare`, `llama-server`, `bench analyze`, `quality eval`, status throttle GPU, pembungkus MCP.
