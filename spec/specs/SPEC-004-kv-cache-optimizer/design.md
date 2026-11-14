# Design — SPEC-004: KV Cache dan State

> Spec: SPEC-004 | Status: **Skeleton (direvisi untuk Qwen3.8-27B)**

---

## Fakta yang Diketahui (sumber: kartu resmi Qwen3.8-27B, lihat `spec/research/CLAIMS.md` C-01..C-06)

- Hanya 16 dari 64 layer (full attention) punya KV cache yang tumbuh dengan konteks: 16 x 2 x 4 x 256 x 2 B = 64 KiB per token (FP16).
- 48 layer Gated DeltaNet menyimpan state ukuran tetap: ≈ 0.146 GiB (turunan; rumus cocok persis dengan log llama.cpp pada 0.8B, belum diukur pada 27B).
- Faktor tipe KV: q8_0 = 0.531x, q4_0 = 0.281x, iso3 (fork) = 0.195x FP16 (C-41).

| Konteks | KV FP16 | KV q8_0 (≈ setengah, asumsi) |
|---|---|---|
| 2048 | 128 MiB | ~64 MiB |
| 4096 | 256 MiB | ~128 MiB |
| 8192 | 512 MiB | ~256 MiB |
| 32768 | 2 GiB | ~1 GiB |

Konsekuensi: KV cache jauh lebih kecil dari yang kita kira di versi awal. Peluang optimasi bergeser ke
**state DeltaNet, compute buffer, dan konteks panjang**, bukan lagi KV quantization.

---

## Open Questions

- [x] Ukuran state DeltaNet: 0.146 GiB (turunan; ukur pada 27B). Apakah ia di GPU atau CPU saat offload parsial? Belum diketahui
- [ ] Berapa compute buffer pada konteks 4096 dan 8192?
- [ ] Apakah KV q8_0 mengubah kualitas? Apakah butuh `-fa`?
- [ ] Layer attention penuh dan layer DeltaNet tersebar tiap 4 layer. Apakah layer yang ada di GPU sebaiknya dipilih berselang (misalnya via `--override-tensor`)? ⚠️ Fitur dan dampaknya belum diverifikasi.
