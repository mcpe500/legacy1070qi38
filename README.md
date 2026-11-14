# legacy1070qi38

*Old Hardware, New Intelligence.* Proyek untuk Kiroween 2026 (tema **Resurrection**): menjalankan
**Qwen3.8-27B** di GPU lama, yaitu **GTX 1070 Ti 8 GB (Pascal, SM 6.1) + RAM 16 GB DDR4**, tanpa
mengasumsikan seluruh model muat di VRAM.

Lisensi: Apache 2.0.

## Status (2026-11-13)

Proyek ini sedang pada tahap **spesifikasi dan alat ukur**. **Qwen3.8-27B belum pernah dijalankan di mesin ini**:
file GGUF-nya belum diunduh dan VRAM sedang dipakai server lain. Angka TPS 27B di dokumen adalah batas atas
hitungan, bukan hasil ukur.

| Sudah ada | Bukti |
|---|---|
| Spesifikasi (SPEC-000..007, RFC-001..003) dan aturan agen | `spec/` |
| Fakta model dari kartu resmi, 46 klaim bersumber | [`spec/research/CLAIMS.md`](spec/research/CLAIMS.md) |
| CLI `tools/legacy.py` (budget, hw, gguf, llama, bench, quality, spec) | 123 test unit + 3 test dengan binary llama.cpp asli |
| Agen GitHub `/kiro` (balas issue/PR dari 3 pengguna terdaftar) | `tools/agent/`, 73 test |
| Kernel Gated DeltaNet berjalan di GTX 1070 Ti | Model `qwen35` 9B, ±20 tok/s, di fork llama.cpp (bukan 27B) |

| Belum | Alasan |
|---|---|
| Qwen3.8-27B di GPU: TPS, RAM puncak, kualitas | Butuh GGUF 27B dan GPU kosong |
| MTP (speculative decoding) | Fork yang dipakai tidak punya `draft-mtp`; perlu llama.cpp utama |
| Single atau dual channel RAM | Butuh `sudo dmidecode -t memory` |

## Mengapa sulit

Qwen3.8-27B adalah model hibrida: 64 layer = 16 x (3 Gated DeltaNet + 1 Gated Attention). Hanya 16 layer yang
punya KV cache (64 KiB per token) dan state DeltaNet tetap sekitar 0.146 GiB, jadi **konteks murah, bobot yang mahal**.
Q4_K_M berukuran 16.5 GiB, sehingga sebagian bobot harus di RAM dan kecepatan dibatasi bandwidth RAM.
Perhitungan dan sumbernya ada di [`spec/AGENTS/MASTER.md`](spec/AGENTS/MASTER.md).

## Mulai cepat

Butuh Python 3.8+ (hanya 3.10 yang diuji). Tanpa dependensi.

```bash
git clone https://github.com/mcpe500/legacy1070qi38 && cd legacy1070qi38

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
.kiro/hooks/      hook Kiro (validasi spec saat disimpan)
```

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
