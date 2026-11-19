# RFC-004: Format Embedding Bobot ke Executable (append + mmap vs alternatif)

> Status: **Proposed** — keputusan menunggu review pemilik repo (MASTER: merge hanya oleh manusia)
> Berlaku untuk: semua spec yang menyentuh distribusi executable — dimulai dari SPEC-008
> Fakta: `spec/research/CLAIMS.md` (ID `C-nn` di bawah merujuk ke sana)

---

## Summary

Bobot Qwen3.8-27B **disambungkan (append)** ke akhir executable engine, disertai trailer 128 B, dan
dibaca saat runtime dengan `mmap` **zero-copy dari berkas executable itu sendiri**
(`/proc/self/exe`). Tidak ada ekstraksi, tidak ada kompilasi bobot, tidak ada salinan.

Payload tidak menggunakan `cmake/embed.cmake` (C-array hex) — terbukti tidak layak di skala GB (C-72).

---

## Motivation

Tema proyek adalah "resurrection": mesin lama menjalankan model besar. Distribusi hari ini =
engine ±10 MB + berkas `.gguf` 4–16 GB di sebelahnya: rawan salah path, salah versi, dan merepotkan
saat dipindah antar mesin/antar pengguna. "Satu berkas, langsung nyala" menjadikan mesin dan model
**satu artefak ter-versi-lock** (engine commit + SHA-256 payload di satu trailer).

Pertanyaan yang dijawab RFC ini: **mekanisme embedding mana yang layak di mesin ini** — bukan
"apakah embedding mempercepat" (tidak; lihat "Konsekuensi").

---

## Kandidat Mekanisme

| # | Mekanisme | Cara kerja | Ukuran berkas | Start-up | Verdict |
|---|---|---|---|---|---|
| **1** | **Append + mmap dari self-exe** | `cat engine payload trailer > exe`; runtime `mmap(fd_self, offset)` | payload + 128 B, utuh | instan (page-on-demand, sama seperti `-m`) | ✅ **Dipilih** |
| 2 | `.incbin` ke section ELF | asm `.incbin "model.bin"` di source; payload jadi section | payload + padding | instan (mapping section) | ✅ Layak, cadangan — ganti model = re-link; risiko `--gc-sections` membuang section |
| 3 | C-array via `cmake/embed.cmake` | `unsigned char w[] = {0x…}` dari hex string | ~5× payload teks C++, lalu payload di `.rodata` | instan | ❌ **Ditolak**: 5.54 GiB → ±28 GB source (C-72); compiler/linker mati |
| 4 | Berkas payload terpisah (hari ini) | `-m MODEL.gguf` | 2 berkas | instan | ⚠ Tetap didukung sebagai override `-m` |

Alasan 1 mengalahkan 2: rebuild payload **tanpa** menyentuh engine (ganti model ≠ build ulang), tidak
bergantung pada linker script/`KEEP()`, dan identik dengan perilaku `mmap` GGUF yang sudah terukur di
engine hari ini (`gguf_index.cpp`).

---

## Format Trailer (128 B, little-endian)

| Offset | Ukuran | Isi |
|---|---|---|
| 0 | 4 | magic `LWGT` (legacy1070qi38) |
| 4 | 4 | versi format (mulai 1) |
| 8 | 8 | offset payload dalam berkas (**wajib kelipatan 4096**) |
| 16 | 8 | panjang payload (byte) |
| 24 | 32 | SHA-256 payload (bukan seluruh berkas) |
| 56 | 64 | nama model (ASCII, NUL-padded) |
| 120 | 4 | panjang nama terpakai |
| 124 | 4 | reserved (0) |

Aturan:

- Trailer **selalu 128 B terakhir** berkas → pencarian tak perlu memindai isi
- Offset page-aligned adalah syarat `mmap` (POSIX); padding nol diperbolehkan di depan payload
- Digest di atas **payload saja** agar engine (yang berubah lebih sering) tidak membatalkan verifikasi model
- Payload **tidak pernah** `PROT_EXEC`; hanya `PROT_READ`

---

## Mentah vs Terkompresi

| Opsi | Ukuran artefak | Start-up | Catatan |
|---|---|---|---|
| **Mentah (default)** | payload utuh | nol (page fault biasa) | Identik dengan mode `-m` — cocok untuk parity |
| zstd payload | berkas lebih kecil | +dekompresi **ke RAM** sebelum start; bobot = RAM penuh saat jalan | Risiko: RAM 15,905 MiB (C-70); zstd rsyncable chunked bisa streaming |

Keputusan awal: **mentah**. Kompresi hanya dijalankan sebagai Task 7 SPEC-008 (ukur rasio & waktu
dekompresi di 4 core) dan masuk ke sini kalau gate lolos. Tidak boleh ditambahkan "karena kelihatan
bagus".

---

## Konsekuensi (yang WAJIB tertulis di setiap dokumen pemakainya)

1. **Embedding bukan optimasi kecepatan.** Decode terikat bandwidth:
   `tok/s ≈ bandwidth / bytes bobot per token` (C-71). Q4_K_M hybrid tetap 2.85 tok/s (C-54) baik
   dengan `-m` maupun payload — byte dan RAM-nya sama. Manfaat RFC ini: distribusi, satu artefak,
   integritas (digest), dan cerita "satu berkas" — **bukan** angka tok/s
2. Batas ukuran payload ditentukan **RAM + VRAM mesin**, bukan oleh ELF:
   kelas Bonsai (5.54–7.10 GiB, C-69) layak; Q4_K_M 15.43 GiB tidak dijamin (≈12.5 GiB RAM + overhead,
   C-29) → override `-m` tetap ada
3. Artefak multi-GiB adalah **GitHub Release**, bukan objek git/LFS (kuota)
4. Bahasa pemaket (C/C++/Goose) tidak mengubahan performa inferensi; eksperimen dibatasi kernel-level
   dengan decision gate (SPEC-008 Task 6)

---

## Risiko

| Risiko | Penanganan |
|---|---|
| Payload rusak di jalan (disk/media) | SHA-256 trailer; gagal → kode 2, tidak start |
| Versi format berubah | `version` menaikkan; executable lama menolak trailer baru dengan pesan jelas |
| `--gc-sections` / strip membuang section (opsi 2) | Tidak berlaku: opsi 1 tidak memakai section sama sekali |
| mlock gagal (limit 1.94 GiB, C-70) | Diabaikan; mmap tanpa mlock + pantau swap |
| Berkas fat ter-commit tak sengaja | Pola `.gitignore` baru di Task 4 SPEC-008 (pengaman dini) |

---

## Bukti

| Klaim | Sumber |
|---|---|
| Ukuran berkas bobot (PTQ1_0 / Q2_0 / Q4_K_M) | C-69 |
| Hex embed tidak layak di GB | C-72 |
| Batas tok/s & tidak ada speedup dari embed | C-71 |
| RAM/VRAM mesin (15,905 MiB; mlock 1.94 GiB) | C-70, `qwen38-engine/AGENTS.md` |
| Angka Q4_K_M hybrid | C-54 |