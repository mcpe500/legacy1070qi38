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

## E-D2b · selector top-32 + sparse decode (28 Nov, CPU fase — GPU dilarang dipakai sesi lain)

**Hipotesis.** Seleksi blok ala DFly2 bisa dibangun sebagai inti portabel (host+device) yang diuji
penuh di CPU, sehingga kernel GPU tinggal konsumsi daftarnya; baris sparse memakai list per grup KV
(top-32 blok + window 128 + sink blok 0, ~2180 posisi) dengan kausalitas identik jalur dense
(entr di atas `pos0 + t` dibuang di kernel, jadi T>1 batch tetap eksak per baris).

**Aksi.**
- `src/cuda/dfly2_select.cuh`: predikat `always_selected` (sink+window), `topk_selected` (rank
  deterministik: #lebih-baik + #sama-indeks-lebih-kecil < 32), `build_list` (serial, uji CPU) dan
  `expand_list` (jalur device dari flag paralel) — satu sumber aturan untuk keduanya.
- `ops.cu`: `dfly2_select_kernel` (grid per grup KV, flag top-k dihitung paralel lalu thread-0
  mengembangkan daftar), `op_blockmax_select`, dan mode sparse di `attention_kernel_decode_iso3`
  (param `plist/plcount/pl_stride`: chunking di atas panjang daftar, alamat posisi = `plist[j]`,
  guard per posisi `ja < n` di skor dan bobot V). `op_attention_sparse` mengeksposnya; fallback
  dense bila decode-v2 tidak aktif.
- Arm/disarm sideband pindah ke **slot pointer device-side** (`g_attn_blockmax_dev`) yang diisi
  `cudaMemcpyToSymbolAsync` dari `op_attention_arm_blockmax` — capture CUDA graph mengreplay
  arm/disarm dengan benar; percobaan `extern __device__` lintas TU ditolak nvcc (warning #20044:
  "treated as a static definition" → bayangan simbol per-TU = sideband senyap mati) dan diganti.
- Kernel menulis sideband hanya saat `gridDim.x == NKV` (T=1) dan `plist == null` (decode target),
  sehingga jalur verify dan baris draft tidak mencemari skor.
- Wiring engine: `QWEN_MTP_SPARSE=1` (default OFF) di `gpu_layers.cu` — buffer bmax/list/count di
  arena (~37 KB @ 131072), sumber pinjaman = layer attention terakhir (63), selector dijalankan
  per siklus draft dari skor sideband langkah decode sebelumnya.

**Temuan uji CPU (yang menangkap bug):**
- `kStride` versi pertama 2240 salah: window 128 posisi bisa melintang tiga blok 64 (overhang
  2×64) → worst case 2303 → kStride = 2368; uji #6 kini membangun kasus disjoint eksak 2240.
- Round-trip slot: penulisan sign-bit selftest semula acak terpisah dari nibble (data sintetis
  tak konsisten — bukan bug dequant) → diperbaiki menurunkan sign dari nibble.
- Matriks merge `M`: baris r1 dua kali salah tanda (`(x,w,z,-y)` lalu `(-x,w,-z,y)`; benar
  `(-x,w,+z,-y)` dari epilog `o = M o'` kernel) — tertangkap oleh uji ortogonalitas blok 4×4.

**Status verifikasi (semua CPU, tanpa sentuh GPU):** `dfly2_select_test` PASS (top-32 vs sort
reference dengan ties 200 trial, window/sink/clamp/stride, isolasi grup, ekspansi-flag == serial);
SASS `ops.cu` dan `gpu_layers.cu`: HFMA2/HMUL2/HADD2 = 0; build ninja bersih. **Belum diukur:
waktu kernel sparse + parity vs dense pada data nyata** (butuh GPU bebas — antrean berikutnya).

**Keputusan.** Lanjut pipeline P1 (E-D3) sambil menunggu GPU; pengukuran E-D2b di GPU (timing +
acceptance sparse nyata) dijadwalkan setelah dump selesai, dengan gate yang sama.

## E-D3 · pipeline dump K/V layer 63 PTQ1_0-native + hook training (--kv63)

**Hipotesis.** Teacher K/V yang benar untuk training = byte cache iso3 engine sendiri (sudah
fake-quant, persis yang dibaca kernel saat inferensi) — 800 B/posisi, bukan 8 KB FP32; training
dengan seleksi sparse versi torch (per batch, skor nyata) adalah kembaran eksak E-D2b.

**Aksi.**
- `Engine::copy_layer_kv` + `GpuLayers::get_layer_kv`: salinan host cache layer 63 (slot 15).
- `--dump-kv63` (dump_targets.cpp): `kv63_k.npy`/`kv63_v.npy` per shard (u8 [S, L, unit]),
  meta.json + est-byte + resume kompatibel.
- `tools/bonsai_mtp_train/iso3_kv.py`: dequant vektor (LUT 64 entri + matriks merge 128×128 dari
  `tables.inc` — tabel diparse langsung dari sumber, tanpa transkripsi), loader shard, dan
  `sparse_kv_mask` (kembaran torch seleksi E-D2b, ties via bias indeks kecil).
- `train_bonsai_mtp.py --kv63`: `MTPHead.forward(kv63=, pos_abs=)` — q di-RoPE dengan posisi
  absolut sampel (fase cocok dengan K/V yang di-cache engine), k/v proyeksi tidak dipakai;
  `unrolled_loss_targets` + `target_batches` meneruskan (K, V, offs); tanpa `--kv63` jalur lama
  tak tersentuh.
- `kv63_smoke.py` (CPU): forward finite, **uji kausalitas** (mengganggu K/V posisi masa depan tak
  boleh mengubah baris awal), gradien mencapai q/o.

**Temuan uji yang menangkap bug nyata:** versi pertama memakai posisi ABSOLUT untuk mask kausal —
baris crop offset bisa melihat posisi teacher di depan relatifnya (kebocoran kausal 4,78 dari uji
smoke) → mask kini kausal RELATIF (`j <= baris`), posisi absolut hanya untuk fase RoPE q. Juga
dua bug broadcast dimensi G dan pemotongan kv63 per langkah unroll yang memangkas konteks baris
terakhir — ketiganya tertangkap sebelum ada run GPU.

**Status verifikasi:** `iso3_kv.py` selftest PASS (round-trip galat 0,00e+00), `kv63_smoke.py`
PASS, `py_compile` bersih, build engine bersih. **Belum dijalankan:** dump nyata di GPU + epoch
training di T4/Colab (user).

**Keputusan.** Pipeline siap dieksekusi begitu GPU kosong: dump (±1,6 GB tambahan) → training
4000 langkah (`--kv63 --unroll 3`) → gate = acceptance zero-shot naik dari 2,6 % mendekati 57,9 %
dense (ambang −3 poin) dengan ids tetap lossless. Prosedur lengkap: `P1_SHARED_KV_RUNBOOK.md`.
