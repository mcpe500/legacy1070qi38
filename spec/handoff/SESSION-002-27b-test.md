# SESSION-002: Uji Qwen3.8-27B di GTX 1070 Ti (issue #31-#36)

> Tanggal: 2026-11-13 sampai 2026-11-15
> Status: **Complete** untuk sesi ini; issue #31 sampai #34 masih punya sisa (lihat bawah)
> Agent: Kiro (orkestrator) + wf-coder (workflow `goal`)
> Lanjutan dari: SESSION-001

---

## Tujuan Sesi Ini

Menjalankan Qwen3.8-27B di GTX 1070 Ti 8 GB + 16 GB DDR4, dengan TPS, RAM, dan kualitas yang terukur. Menggabungkan KV iso3 (dari fork) dengan MTP (di llama.cpp utama).

## Konteks dari Sesi Sebelumnya

PR #29 dan #30 sudah di-merge. Issue #1, #4, #6-#12 tertutup. Issue #31-#36 dibuat sebagai rencana tanpa dikerjakan. Fork `08e025c` tidak punya `draft-mtp`.

**Spec yang dikerjakan:** SPEC-002, SPEC-004, SPEC-005, SPEC-006 (bagian uji 27B)

---

## Pekerjaan yang Dilakukan

### Task yang Diselesaikan

- [x] Server 9B (PID 22885) dihentikan atas izin pemilik mesin; perintah start lama dicatat di HANDOFF-LOG
  - Anomali: pengaman Kiro menolak `kill` dari wf-coder, jadi dihentikan oleh orkestrator atas permintaan langsung pemilik
- [x] Fork lama diarsipkan ke `~/archive/llama-cpp-turboquant-08e025c` (tidak dihapus)
- [x] llama.cpp utama `3549c27` dibangun dengan CUDA sm_61 (`~/llama.cpp/build-cuda`, `build-iso3`)
- [x] GGUF 27B diunduh dan sha256-nya cocok dengan LFS: UD-IQ3_XXS, UD-IQ4_XS, UD-Q4_K_M, Q4_0, imatrix
- [x] iso3 diporting secara TDD (red `fc9437a`, green `36dbad3`), `patches/llama.cpp/`; PR #38
- [x] Uji hybrid, sapuan thread, `--override-tensor`, MTP, konteks panjang; PR #37
- [x] PPL dan KLD untuk 4 quant, KV q8_0 dan iso3, 8 prompt tetap

### Task yang Tidak Selesai

- [ ] Channel RAM (single atau dual) - butuh `sudo dmidecode -t memory` oleh pemilik
- [ ] Vulkan vs CUDA (OQ-016) - `glslc` tidak ada, butuh `sudo apt`
- [ ] Strategi B (Q3-DOWN-XS) - belum diunduh
- [ ] IQ3_M dan auto-fit vs `-ngl` manual (OQ-014) - belum diuji
- [ ] Kuantisasi ulang dengan imatrix - butuh BF16, sekitar 50 GB
- [ ] Kernel vektor iso3 native untuk FlashAttention
- [ ] Server 9B belum dihidupkan lagi. Build utama tidak bisa memuat `Qwen3.8-9B-Q4_K_M.gguf` (442 vs 427 tensor); gunakan fork di arsip

---

## Temuan Penting

### Measurements

Semua di llama.cpp utama `3549c27` (+ patch iso3), 4 thread, KV f16 kecuali disebut. Rincian dan sumber: `CLAIMS.md` C-50 sampai C-58.

| Konfigurasi | Decode TPS | Notes |
|---|---|---|
| IQ3_XXS `-ngl 45` | 1.15 | Lebih lambat dari IQ4_XS walau lebih kecil |
| IQ4_XS `-ngl 36` | 2.29 | |
| Q4_K_M `-ngl 32` | 2.60 | |
| Q4_K_M `-ngl 99 -ot` FFN 17..64 di CPU | **2.85** | Tercepat terukur (bench pendek) |
| Q4_K_M ctx 16384, KV iso3, `-ot` FFN 15..64 | 2.61 | Tercepat di ctx 16k |
| IQ4_XS + MTP depth 1 | 2.22 | +6%; dengan `-ot` MTP tidak mempercepat |

Kualitas (KLD terhadap Q4_K_M, satu teks): KV q8_0 0.0005, KV q4_0 0.0024, KV iso3 0.0102, IQ4_XS 0.0162, Q4_0 0.0301, IQ3_XXS 0.0604.

---

## Keputusan yang Dibuat

### DECISION-001: Rekomendasi konfigurasi 27B untuk mesin ini
- **Keputusan:** UD-Q4_K_M dengan `-ngl 99 -ot` FFN 17..64 ke CPU. Konteks panjang: KV iso3 (menghemat VRAM) atau q8_0 (kualitas hampir sama dengan f16). MTP dimatikan.
- **Alasan:** terukur paling cepat dan lolos batas kualitas MASTER. MTP tidak memberi percepatan berarti.
- **Efek:** `MASTER.md` diberi bagian hasil terukur. Keputusan ini bisa dibuka lagi bila channel RAM ternyata single.

### DECISION-002: PR di-merge oleh agen atas perintah pemilik
- **Konteks:** sebelumnya hanya pemilik yang merge. Pada 2026-11-15 pemilik menyuruh agen meng-merge PR #37 dan #38 dan menutup issue yang selesai.
- **Keputusan:** merge dengan `gh pr merge --merge` tanpa melewati proteksi `main`. Konflik `HANDOFF-LOG.md` di PR #38 diselesaikan dengan menyimpan kedua sisi.

---

## Open Questions yang Dijawab

| OQ-ID | Pertanyaan | Jawaban | Evidence |
|---|---|---|---|
| OQ-005 / OQ-015 | Apakah dequant IQ memperlambat di CPU ini | Ya. Q4_K_M lebih cepat dari IQ4_XS walau file lebih besar | C-53, C-54 |
| OQ-012 | Percepatan MTP di hybrid | +6% terbaik, tidak ada dengan `-ot` | C-51, C-56 |
| OQ-017 | Apakah `-fa` membantu | Tidak di ctx pendek (2.59 vs 2.84) | C-58 |
| OQ-009 | Gated DeltaNet 27B jalan di Pascal | Ya, di llama.cpp utama | run-001 |

## Open Questions Baru

| OQ-ID | Pertanyaan | Priority | Target |
|---|---|---|---|
| OQ-018 | Apakah RAM single atau dual channel, dan berapa percepatan bagian CPU bila dual | High | SPEC-005 |
| OQ-019 | Apakah bit campuran per layer (layer di CPU lebih rendah) menaikkan TPS tanpa turun kualitas berarti | Medium | RFC-001 |
| OQ-020 | Apakah pemangkasan satu grup layer (4 layer) layak, dan berapa kualitas hilang. Model turunan bukan lagi 27B utuh; **belum diputuskan pemilik** | Low | RFC baru |

---

## Catatan di luar kode: pilihan model Kiro

Dari `kiro.dev/docs/models` (diperbarui 2026-11-12) dan satu uji kecil di mesin ini:

| Model | Pengali kredit | Catatan |
|---|---|---|
| auto | 1.0x | Kiro memilih model per tugas |
| claude-haiku-5.5 | 0.06x sampai 100K token, 0.3x di atasnya | Disarankan Kiro untuk pekerjaan volume besar dan cepat |
| claude-opus-5.5 | 2.0x | Disarankan untuk pekerjaan agentik yang sulit |

Uji satu kalimat headless: Haiku 5.5 0.0027 kredit, auto 0.0268 kredit. Satu sampel, belum cukup untuk kesimpulan. Haiku 5.5 effort `max` sudah tersimpan di `chat.modelDefaults`. Perbandingan kualitas lima tugas belum dijalankan.

---

## Update ke HANDOFF-LOG.md

Entri sudah ditambahkan untuk: penghentian server 9B, run-001, run-002, iso3 + MTP, konteks panjang, kualitas 27B, dan blocker Vulkan serta channel RAM.

---

## Next Steps

1. Pemilik menjalankan `sudo dmidecode -t memory` untuk OQ-018.
2. Putuskan apakah server 9B dihidupkan lagi (perintah start di HANDOFF-LOG).
3. Putuskan apakah bit campuran per layer dan pemangkasan layer dibuatkan issue (OQ-019, OQ-020).
4. Uji kode dan penalaran panjang untuk kualitas, bukan hanya wikitext-2.
5. Bangun kernel vektor iso3 native bila konteks panjang menjadi kebutuhan utama.

---

## State Akhir Sesi

| Spec | Status Awal | Status Akhir |
|---|---|---|
| Uji 27B (#31-#36) | Rencana | #35 dan #36 selesai bila kriterianya terpenuhi (lihat status issue); #31-#34 punya sisa |

---

*Dibuat dari template `_TEMPLATE-session.md`.*
