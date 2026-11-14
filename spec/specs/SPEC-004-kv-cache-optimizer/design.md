# Design — SPEC-004: KV Cache dan State

> Spec: SPEC-004 | Status: **Skeleton (direvisi untuk Qwen3.8-27B)**

---

## Fakta yang Diketahui (sumber: model card 6block, lihat MASTER)

- Hanya 16 dari 64 layer (full attention) punya KV cache yang tumbuh dengan konteks: ≈ 64 KiB per token (FP16).
- 48 layer Gated DeltaNet menyimpan state ukuran tetap. **Besarnya belum diketahui.**

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

- [ ] Berapa ukuran state DeltaNet dan apakah ia di GPU atau CPU saat offload parsial?
- [ ] Berapa compute buffer pada konteks 4096 dan 8192?
- [ ] Apakah KV q8_0 mengubah kualitas? Apakah butuh `-fa`?
- [ ] Layer attention penuh dan layer DeltaNet tersebar tiap 4 layer. Apakah layer yang ada di GPU sebaiknya dipilih berselang (misalnya via `--override-tensor`)? ⚠️ Fitur dan dampaknya belum diverifikasi.
