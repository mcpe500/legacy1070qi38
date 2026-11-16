# PR #38: feat(iso3): KV iso3 di llama.cpp utama, dipakai bersama MTP

- state: MERGED
- merged: 2026-11-15T12:45:15Z
- head: feat/iso3-mtp-e2e
- base: main
- url: https://github.com/mcpe500/legacy1070qi38/pull/38

## Body

Bagian dari #31, #35, #36. KV cache iso3 (IsoQuant 3-bit dari fork `08e025c`) diporting ke llama.cpp utama `3549c27`, jadi iso3 dan `--spec-type draft-mtp` bisa dipakai bersamaan. Patch ada di `patches/llama.cpp/` (2 commit; `git am`). Laporan: `spec/research/reports/2026-11-14-iso3-mtp-main.md` dan `2026-11-15-27b-quality-001.md`. Semua angka terukur di GTX 1070 Ti (sm_61).

## TDD
- Red (`fc9437a`): `test-iso3` tidak bisa dibangun (`GGML_TYPE_ISO3_0` belum ada); e2e: 4 gagal (`Unsupported cache type: iso3`).
- Green (`36dbad3`):
  - `test-iso3` PASSED. Isinya traits dan bentuk tensor, roundtrip CPU (nmse 0.0348), byte `SET_ROWS` CUDA = referensi CPU 1600/1600, dan FA CUDA iso3 vs f16 (nmse ≤ 2e-11 pada nilai yang sudah didekuantisasi).
  - `test-quantize-fns` rc 0.
  - `test-backend-ops -b CUDA0`: kasus iso3 28/28, regresi `SET_ROWS,GET_ROWS,CPY,FLASH_ATTN_EXT` 5041/5041.
  - `ctest -L main` (tanpa tokenizer dan backend-ops) rc 0.
  - `tools/iso3/test_iso3_e2e.py`: SmallModel 3/3, MtpWithIso3 lulus.
- Tes iso3 wajib jalan di GPU. Tanpa GPU tes gagal, kecuali `ISO3_ALLOW_NO_GPU=1`.

## Batasan yang terukur
- FA CUDA tidak punya kernel vektor iso3: K/V dikonversi ke f16 sebelum kernel jalan.
- KV konteks MTP (1 layer) memakai `-ctkd/-ctvd` (default f16) dan **tidak** mewarisi `-ctk/-ctv`. Perilaku ini belum diubah karena itu keputusan perilaku.
- Keluaran MTP di llama.cpp utama tidak identik bit dengan tanpa MTP, juga pada KV f16. Uji rollback memeriksa bahwa di titik beda pertama, token MTP adalah kandidat terbaik menurut target (selisih logprob 0.000 terukur).

## Kecepatan Qwen3.8-27B UD-Q4_K_M, ctx 16384, prompt 6011, generasi 64 token
| Konfigurasi | gen tok/s |
|---|---:|
| KV iso3, `-ngl 99 -ot` FFN 15..64 ke CPU | **2.61** |
| KV f16, `-ngl 99 -ot` FFN 9..64 ke CPU | 2.52 |
| KV q8_0, `-ngl 99 -ot` FFN 12..64 | 2.46 |
| KV iso3 + MTP d1, `-ngl 28` | 2.37 |
| KV iso3 + MTP d1, `-ot` FFN 11..64 | 2.37 |
| KV iso3, `-ngl 31` | 2.23 |
| KV f16, `-ngl 28` | 1.86 |
| KV q8_0 + MTP d1, `-ot` FFN 8..64 | 2.24 |
| KV f16 + MTP d1, `-ngl 26` | 1.80 |

Crash CUDA OOM di tengah generasi terjadi pada: iso3 + MTP d3 `-ngl 28`, iso3 + MTP d1 `-ot` FFN 12..64, q8_0 `-ot` FFN 13..64, dan q8_0 + MTP d1 `-ot` FFN 9..64. MTP tidak mempercepat pada `-ot` dengan KV mana pun yang diuji.

## Kualitas (wikitext-2 test, ctx 512, 20 chunk, referensi UD-Q4_K_M KV f16)
| Model / KV | PPL | mean KLD | same top p |
|---|---:|---:|---:|
| Q4_K_M, KV f16 | 6.919 | 0 | 100% |
| Q4_K_M, KV q8_0 | - | 0.00049 | 98.9% |
| Q4_K_M, KV iso3 | - | 0.01020 | 95.3% |
| IQ4_XS | 6.949 | 0.01620 | 93.8% |
| Q4_0 | 7.057 | 0.03011 | 92.1% |
| IQ3_XXS | 7.203 | 0.06044 | 88.9% |

Dengan 8 prompt tetap, kemiripan terhadap Q4_K_M adalah IQ4_XS 0.583, Q4_0 0.571, dan IQ3_XXS 0.504. Tiga cek fakta sederhana benar di semua quant.

KLD dihitung terhadap Q4_K_M, bukan BF16, jadi angkanya bukan kualitas absolut.

## Belum
- Kernel FA vektor iso3 native.
- Pewarisan tipe KV untuk konteks MTP.
- Jalur FA CPU iso3 di model nyata (KV selalu di GPU pada uji ini).

