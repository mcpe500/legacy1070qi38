# RFC-002: CLI Tooling untuk legacy1070qi38

> Status: **Accepted** (2026-11-13). Enam keputusan terbuka di issue #1 diputuskan berdasarkan bukti di bagian "Keputusan" di bawah.
> Keputusan itu diambil oleh agen berdasarkan pengukuran; pemilik proyek dapat membukanya kembali.
> Revisi: versi awal mengusulkan 4 MCP server. Diganti menjadi satu CLI karena lebih mudah dipasang dan dipakai orang lain.
> Diskusi: umbrella #1; per kelompok: A #2, B #3, C #4, D #5 (github.com/mcpe500/legacy1070qi38/issues)

![Peta subcommand](assets/rfc-002-mcp-overview.svg)

---

## Summary

Satu file Python, `tools/legacy.py`, hanya memakai library standar. Isinya subcommand untuk menghitung memory budget, memantau hardware, menjalankan dan mengukur llama.cpp, dan merawat dokumen spec. Agent memanggilnya lewat bash, manusia lewat terminal. Setiap subcommand punya opsi `--json`.

> ⚠️ ASSUMPTION: perilaku bash tool di Kiro untuk proses panjang (timeout, background) belum diverifikasi.

---

## Motivation

- OQ-001 sampai OQ-008 hanya bisa dijawab lewat **pengukuran**.
- Menjalankan `nvidia-smi` dan `llama-cli` lewat shell bebas menghasilkan output tak terstruktur dan tanpa pengaman.
- Dokumen spec rawan tidak konsisten. Contoh nyata: penomoran NFR yang melompat ke NFR-04 di sesi bootstrap.
- Proyek ini ditujukan bagi pengguna GPU lama. Alat yang perlu dipasang dan didaftarkan terlebih dulu menghambat itu. `git clone` lalu jalan lebih sesuai.

---

## Subcommand

| Kelompok | Subcommand | Fungsi | Butuh GPU | Menjawab |
|---|---|---|---|---|
| A. `budget` | `estimate`, `suggest-ngl`, `compare` | Hitung VRAM, KV cache, jumlah layer GPU; bandingkan estimasi dengan hasil ukur | Tidak | Rumus di `memory-manager.md` |
| B. `hw` | `gpu`, `sample`, `ram`, `headroom` | VRAM, suhu, power, throttle; sampling peak; RAM dan swap; go/no-go | Ya | OQ-005, OQ-008 |
| C. `gguf`, `llama` | `gguf inspect`, `llama build`, `llama run`, `llama bench`, `llama compare` | Baca metadata GGUF tanpa memuat model; build SM 61; jalankan; benchmark 3x + sampling GPU | Sebagian | OQ-001..003, OQ-006 |
| D. `spec` | `new`, `handoff`, `session`, `questions`, `validate` | Scaffold spec, tambah entri handoff, kelola OQ, cek konsistensi dokumen | Tidak | Konsistensi dokumen |

### Tambahan setelah koreksi model

| Subcommand | Fungsi | Alasan | GPU |
|---|---|---|---|
| `gguf tensors` | Ukuran tiap tensor dan layer dari header GGUF | Mengganti asumsi "layer seragam" dengan angka nyata untuk perhitungan layer di GPU | Tidak |
| `llama placement` | Baca log pemuatan: layer di GPU atau CPU, ukuran buffer | Menjawab "model masuk ke bagian mana". Format log berbeda antar versi | Ya |
| `hw cpu` | Model CPU, core, AVX2, frekuensi, throttle | Layer di CPU menentukan TPS | Tidak |
| `hw membw` | Ukur bandwidth RAM nyata | Asumsi 20/40 GB/s belum terukur | Tidak |
| `hw faults` | Page fault berat dan swap | Penyebab utama perlambatan saat RAM mepet | Tidak |
| `bench analyze` | Bandwidth efektif vs teoritis | Menunjukkan penahan: GPU, CPU, atau PCIe | Ya |
| `quality eval` | Perplexity dan KL divergence antar quant, set prompt tetap | Tujuan proyek mencakup kualitas | Ya |

Dan satu dokumen, bukan CLI: `spec/research/` berisi daftar klaim (klaim, sumber, status, OQ terkait).

> Gambar di atas menampilkan set awal; tambahan ini belum ada di gambar.

Contoh:

```bash
python tools/legacy.py budget estimate --file-gib 16.53 --vram-gib 8 --ctx 4096 --json
python tools/legacy.py gguf inspect models/Qwen3.8-27B-Q4_K_M.gguf --json
python tools/legacy.py llama bench --model models/Qwen3.8-27B-Q4_K_M.gguf --runs 3
python tools/legacy.py spec validate
```

---

## Keputusan Desain yang Diusulkan

1. **Satu file, library standar saja.** Lihat keputusan 1 untuk batas ukuran.
2. **Logika di fungsi Python, CLI hanya lapisan tipis.** Fungsi yang sama bisa dibungkus MCP nanti.
3. **Output `--json` di semua subcommand**, plus ringkasan yang enak dibaca manusia.
4. **Tanpa shell bebas.** Argumen divalidasi dengan `argparse`; binary dan path model memakai allowlist.
5. **Lock file** agar hanya satu inference berjalan sekaligus. Benchmark paralel merusak angka.
6. **Exit code bermakna**, misalnya `headroom` mengembalikan non-zero bila tidak cukup.
7. **Agent diberi tahu cara memakainya** lewat dokumen singkat `tools/README.md`, bisa dijadikan skill atau steering.
8. **Urutan:** `budget` → `gguf inspect` → `hw` → sisa `llama` → `spec`.

---

## MCP: Opsional di Akhir

MCP bukan lagi jalur utama. Pembungkus MCP (sekitar 50 baris, memanggil fungsi yang sama) ditambahkan hanya jika:
- ingin tool otomatis terdaftar dengan skema di Kiro, atau
- ingin bukti pemakaian MCP di `KIRO-USAGE.md`.

---

## Perbandingan dengan MCP Penuh

| | CLI (dipilih) | MCP server |
|---|---|---|
| Pemasangan | `git clone`, langsung jalan | Pasang dependensi, daftar di `mcp.json` |
| Pengguna | Agent, manusia, orang tanpa Kiro | Hanya agent |
| Agent mengenali tool | Lewat dokumen | Otomatis lewat skema |
| Proses panjang (27B) | Background lewat bash | Terikat timeout MCP (belum diverifikasi) |
| Bukti "pakai MCP" | Tidak ada, kecuali ada pembungkus | Ada |

---

## Keputusan

| # | Pertanyaan | Keputusan | Bukti |
|---|---|---|---|
| 1 | Satu file atau modul? | **Tetap satu file.** Ambang "1000 baris" diganti: pecah bila ada kebutuhan impor oleh kode lain, atau bila melewati ±2500 baris. Saat ini ±1900 baris | Dilewati 1000 baris sejak PR #18 (1403 baris) tanpa masalah: `python tools/legacy.py` langsung jalan, test mengimpor `legacy` apa adanya. Memecah sekarang menambah risiko tanpa manfaat terukur. **Ini perubahan atas usulan awal; mohon dikoreksi bila tidak setuju** |
| 2 | Proses panjang lewat bash tool Kiro? | **Background + polling.** Jalankan dengan `run_in_background` atau `setsid nohup ... > log &`, baca log | `llama build --target ...` (4 job, selesai dalam beberapa menit) dijalankan di latar belakang dan selesai dengan benar; `quality kld` (±70 detik) dan `quality ppl` (±32 detik) berjalan sinkron dengan `timeout` yang diatur eksplisit (satuan **milidetik** pada bash tool) |
| 3 | `llama-cli` per panggilan atau `llama-server`? | **Per panggilan untuk pengukuran** (`llama-cli -st`, `llama-bench`, `llama-perplexity`). `llama-server` tidak dibungkus sekarang | Hasil terkontrol dan punya exit code; model 0.8B termuat <5 detik. Pengguna sudah menjalankan `llama-server` sendiri di port 8080 dan alat tidak boleh menyentuhnya. Waktu muat 27B belum diukur; bila ternyata >2 menit per uji, buka ulang di issue uji 27B |
| 4 | Hook `spec validate` saat simpan? | **Ya.** `.kiro/hooks/spec-validate-on-save.json` (trigger `PostFileSave`, pola `spec/.*\.md$`) | Perintah yang sama dijalankan tangan: `ok: True`, 0 error. Hook baru aktif pada sesi berikutnya; belum diamati terpicu |
| 5 | Python minimum? | **3.8.** Hanya **3.10.12 yang diuji** | `ast.parse(..., feature_version=(3, 7))` lolos; tidak ada `:=`, `match`, `removeprefix`. Python 3.8/3.9 tidak ada di mesin ini, jadi tidak dijalankan |
| 6 | Pembungkus MCP? | **Tidak dibuat.** | Agen `/kiro` memanggil CLI lewat bash. Bukti pemakaian MCP sudah ada lewat Sequential Thinking (PR #28), yang tidak membutuhkan kode kita |
| - | Desktop yang memakai VRAM saat benchmark? | **Dicatat, bukan masalah di mesin ini.** `hw gpu` melaporkan VRAM terpakai; display NVIDIA dimatikan | `hw headroom` memeriksa VRAM bebas sebelum uji |

---

*Terkait: `spec/specs/SPEC-006-benchmark-suite/`, `spec/AGENTS/roles/memory-manager.md`, `spec/AGENTS/roles/benchmarker.md`, `spec/AGENTS/KIRO-USAGE.md`.*
