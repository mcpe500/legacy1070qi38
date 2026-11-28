# DFly2-style shared-KV MTP — EXPERIMENT-LOG (branch `exp/dfly2-shared-kv`)

Rantai penelitian P0-rev (issue #70, komentar revisi 2026-11-28). Format: hipotesis → aksi
(angka nyata) → keputusan. Angka hanya dari `rtk bypass` / file hasil (Hard rule 10).
Sumber desain: issue #70 + komentar revisi `issues/70#issuecomment-7241852601`.

## E-D1 · P0.a sideband block-max decode (28 Nov, mulai 15:4x)

**Hipotesis.** Skor tile-max `tm` sudah dihitung kernel decode v2 (warp-uniform, hasil 4
shuffle); menambahkan 1 CAS `atomicMax` per tile 16 posisi per grup KV ke buffer
`(blok 64 posisi × NKV + g)` memberi seleksi blok ala DFly2 dengan overhead ≈ nol.

**Aksi.**
- `src/cuda/ops.cu`: param baru `float * bmax` di `attention_kernel_decode_iso3`;
  `bmt = max(tm…)` per tile, lane 0 menulis `ATOM.E.CAS` hanya bila `bmax != null`;
  global `g_attn_blockmax` + `op_attention_set_blockmax` (null = off, default).
  Dispatch: `g_attn_blockmax` diteruskan pada jalur decode v2 saja.
- `src/cuda/ops.cuh`: deklarasi + kontrak (init −inf per posisi; ukuran
  `ceil(pos/64) × NKV` float; NKV = 4 → 102400 pos = 1600 × 4 × 4 B = 25,6 KB).
- `bench/attn_dec_bench.cu`: satu invokasi mengukur off lalu on (buffer di-reset H2D
  tiap replika di luar jendela CUDA events, agar pola retry CAS seperti langkah decode
  baru, bukan maksimum yang sudah jenuh); checksum + hash bit FNV-1a harus identik
  off vs on (sideband tidak boleh menyentuh hasil).

**Gate build (terukur, `cuobjdump`):**
- `HFMA2|HMUL2|HADD2` di `ops.cu.o` = **0** (Hard rule 8 bagian 1 lolos; IDP.4A = 0
  wajar — dp4a bukan jalur file ini, pola sama dengan E9).
- Sideband terkompilasi: 1× `ATOM.E.CAS` di SASS.
- Resource kernel decode: **REG 168 / STACK 80 — persis baseline E2**; tanpa spill
  baru; `__launch_bounds__(…, 3)` menahan 3 blok/SM → occupancy tidak berubah.

**Angka timing (attn_dec_bench, ctx 131072, 30 replika, off vs on, `rtk bypass`):**

| n | off best | on best | Δ | blok terisi | checksum / bits |
|---|---|---|---|---|---|
| 100 | 0,048 ms | 0,048 ms | +0,0 % (noise) | 2/2 | bits `666431b2…` |
| 4097 | 0,135 ms | 0,134 ms | −0,8 % (noise) | 65/65 | bits `d078ab98…` |
| 102400 | **1,589 ms** | **1,622 ms** | **+0,033 ms = +2,1 %** | 1600/1600 | checksum 0,037185, bits **`05c1dd08be724870`** |

- Bit hash ON @ 102400 = **persis bits build pair nsl=56 ter-commit** (E13: `05c1dd08…`) →
  sideband **tidak mengubah hasil attention** (gate bit-identity ON vs build lama lolos).
- Buffer terisi benar: 1600/1600 blok @ 102K, nilai masuk akal (maks 5,41 di blok 1336).
- Konteks pengukuran: run tertunda ~35 menit oleh eksperimen sesi lain (E37 `kld_decode`,
  lalu ablation `/tmp/abl/i8v_QRES`) — bukan bagian rantai ini, hanya penanda antrian GPU.

**Keputusan.** Aturan pra-daftar E-D1 bilang: overhead kernel > ~1 % ⇒ tinjau desain tulis.
Terukur +2,1 % per layer @ 102K — di atas ambang literal, **tapi** di mesin produksi sideband
hanya dipasang di **layer 63** (satu panggilan per langkah decode), bukan 16 layer: biaya
nyata ≈ 0,033 ms per token dari langkah ~82 ms ≈ **0,04 %**. Keputusan: **lanjut E-D2
(selector top-32 + sparse gather) tanpa redesain**; opsi E-D1b (flush CAS per blok 64 —
4× lebih sedikit atomik) dicatat sebagai cadangan jika kelak sideband dipasang di semua
layer atau di jalur verify T=3.

## E-D2a · P0.d zero-shot shared-KV dense (28 Nov, 16:4x–17:1x)

**Hipotesis.** Head MTP Bonsai 2.5 (blk.64, dilatih dengan KV sendiri) dipaksa membaca KV
layer 63 target tanpa training: jika ruang key-nya tak kompatibel, acceptance runtuh —
basline gagal terukur yang memisahkan "mesin benar" dari "head paham".

**Aksi.**
- Flag `QWEN_MTP_SHARED_KV=1` (default OFF), 5 file: `gpu_layers.cu` (`attention()` varian
  baca-override: lewati `wk`/`wv` gemv + `kv_store`, `op_attention` baca cache layer
  attention terakhir; `pos_att = min(pos0, pos_kv)` agar baris draft di atas `pos_kv` tak
  membaca kunci fanta), `gpu_layers.hpp`/`engine.hpp`/`engine.cpp` (param `pos_kv` +
  skip `mtp_prefill`), `generator.cpp` (3 call site kirim `pos_new`).
- Perbaikan samping: `stage_bytes` staging buffer kini menghitung tensor `blk.64.*`
  (sebelumnya "staging buffer too small for blk.64.nextn.eh_proj.weight").
- Run: PTQ1_0-mtp.gguf (blk.64 embedded; GGUF utama tanpa `-mtp` TIDAK punya head,
  `-mtp.gguf` = model penuh + head, bukan sidecar `--mtp-gguf`), prompt 5516 token
  (`head -c 16384` dari realfull102000.txt), `-n 256 --ctx 8192 --temp 0 --seed 42
  --mtp-steps 2 --mtp-adaptive 0 --pld 0 --gpu-layers 64 --gpu-mtp 1`, gpu_isolate ✓.
  Penting: split 32/32 layer membuat `mtp_gpu_` false (head CPU) → jalur GPU shared
  tak tersentuh; run pertama tanpa flag ini menghasilkan acceptance identik palsu.

**Angka (`rtk bypass`, ujung stats JSON):**

| run | acceptance | cycles | decode | mtp_prefill | VRAM |
|---|---|---|---|---|---|
| head bawaan (KV sendiri) | **22/38 = 57,9 %** | 19 | 27,2 tok/s | 0,429 s | 6554 MiB |
| zero-shot baca layer 63 | **2/78 = 2,6 %** | 39 | 14,45 tok/s | **0,015 s** | 6554 MiB |

- **ids identik di kedua run** (`diff` kosong pada `"ids"`): verify tetap lossless meski
  acceptance runtuh → mesin shared-KV benar.
- `mtp_prefill_s` 0,429 → 0,015 s (skip bekerja; @ 102K padanannya 4395 s → ~0).
- Zero-shot lebih lambat (14,45 vs 27,2) walau prefill di-skip: acceptance 2,6 % berarti
  tiap siklus nyaris tanpa token diterima — overhead draft jadi murni kerugian.

**Keputusan.** Dugaan terkonfirmasi dengan data: **head terlatih-KV-sendiri tidak paham
ruang KV teacher** (2,6 % ≈ peluang argmax acak-murah). Artinya: (1) mesin shared-KV
siap dan lossless — bagian *speed* P2 tinggal selector+sparse (E-D2b) dan penghapusan
alokasi KV MTP; (2) **P1 (training ulang head dengan K/V teacher layer 63 PTQ1_0-native,
dump 16,4 GB) adalah syarat mutlak** — tanpa itu DFly2-style hanya menghasilkan mesin
cepat yang karatan. Sparse (top-32) tidak relevan diuji sebelum head dilatih: ia hanya
akan menurunkan acceptance 2,6 % lebih jauh. Sampel 41–48 token per run kecil, tapi
selisih 57,9 % vs 2,6 % jauh dari marginal. Kode flag belum di-commit (menunggu
instruksi user, sesuai konvensi repo).
