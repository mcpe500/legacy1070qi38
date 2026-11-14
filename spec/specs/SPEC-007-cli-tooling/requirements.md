# Requirements — CLI Tooling (`tools/legacy.py`)

> Spec: SPEC-007 | Status: **In Review** (test lulus di sandbox; belum diuji dengan llama.cpp dan GGUF asli)
> Referensi: RFC-002 (status **Proposed**; keputusan desain masih milik pemilik repo), issue #1
> Priority: P1

---

## Problem Statement

Pertanyaan terbuka proyek ini (OQ-005, OQ-008, OQ-010, OQ-011) hanya bisa dijawab lewat pengukuran, dan
dokumen spec rawan tidak konsisten (contoh: penomoran NFR yang melompat). RFC-002 mengusulkan satu CLI
`tools/legacy.py` agar agent dan manusia memakai rumus, pembacaan hardware, pembacaan GGUF, dan perawatan dokumen
yang sama. Spec ini menetapkan apa yang harus dilakukan CLI tersebut dan bagaimana membuktikannya.

---

## Functional Requirements

- **FR-01:** `budget estimate`, `suggest-ngl`, `compare` menghitung split VRAM/RAM, jumlah layer di GPU, dan batas atas TPS dengan rumus di `spec/AGENTS/MASTER.md`, tanpa hardware
- **FR-02:** `hw gpu`, `ram`, `cpu`, `membw`, `faults`, `sample`, `headroom` membaca kondisi mesin; `headroom` keluar dengan kode 1 bila tidak cukup
- **FR-03:** `gguf inspect` dan `gguf tensors` membaca header GGUF (v2/v3) tanpa memuat bobot, dan menolak file rusak atau berbahaya dengan galat yang jelas
- **FR-04:** `llama build`, `run`, `bench`, `placement` menyusun dan menjalankan llama.cpp memakai argv list, allowlist binary, dan lock agar satu inference berjalan sekaligus
- **FR-05:** `spec new`, `session`, `handoff`, `questions`, `validate` merawat dokumen; penulisan hanya di bawah `spec/` dan `HANDOFF-LOG.md` hanya ditambah di akhir
- **FR-06:** Setiap subcommand menerima `--json`; galat dalam mode `--json` juga berbentuk JSON
- **FR-07:** Exit code: 0 ok, 1 cek gagal (no-go, validasi gagal, di luar toleransi), 2 input salah, 3 alat atau file tidak ada

---

## Non-Functional Requirements

- **NFR-01: Dependensi** — Hanya library standar Python, satu file `tools/legacy.py`
- **NFR-02: Keamanan** — Tanpa shell; binary hanya `llama-cli`, `llama-bench`, `llama-server`; nama spec dan sesi divalidasi agar tidak keluar dari `spec/`
- **NFR-03: Kejujuran angka** — Keluaran estimasi menyertakan daftar asumsi; TPS diberi label batas atas teoritis
- **NFR-04: Compatibility** — Tidak memanggil apa pun yang butuh Tensor Core, BF16, atau FlashAttention library; build CUDA memakai arsitektur 61
- **NFR-05: Versi Python** — ⚠️ ASSUMPTION: 3.7 atau lebih baru (memakai `from __future__ import annotations` dan `reconfigure`). Belum diuji di versi mana pun; keputusan versi minimum ada di RFC-002

---

## Out of Scope

- `llama compare`, `llama-server` yang tetap hidup, `bench analyze`, `quality eval` (tercantum di RFC-002, belum dikerjakan)
- Status throttle GPU di `hw gpu`
- Pembungkus MCP
- Hook Kiro yang menjalankan `spec validate` saat simpan
- Menjalankan proses panjang di latar belakang dengan polling (OQ di RFC-002 belum terjawab)

---

## Success Criteria

- [x] `python tools/test_legacy.py` lulus semua di mesin Linux dengan Python 3 (127 test OK, PR #29)
- [x] `python tools/legacy.py spec validate` keluar dengan kode 0 pada repo ini
- [ ] `python tools/legacy.py gguf inspect <Qwen3.8-27B GGUF asli> --json`: **dipindah ke issue uji 27B** (GGUF belum ada). Terverifikasi pada GGUF 9B `qwen35` (OQ-010 dijawab dari kartu resmi)
- [x] `python tools/legacy.py hw gpu --json` berjalan di mesin GTX 1070 Ti
- [x] `python tools/legacy.py llama run ...` berjalan dengan llama.cpp asli dan memuat TPS (0.8B, CPU)
- [x] RFC-002 diputuskan: Accepted, enam keputusan dengan bukti (oleh agen; dapat dibuka ulang oleh pemilik repo)

---

## Dependencies

- **Requires:** RFC-002 (keputusan desain), `spec/AGENTS/MASTER.md` (rumus)
- **Blocks:** SPEC-002 Task 4 (header GGUF), SPEC-006 (benchmark suite)
