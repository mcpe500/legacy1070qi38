# HANDOFF LOG — legacy1070qi38

> File ini adalah continuous log dari semua sesi dan keputusan penting proyek.
> **Append-only** — jangan hapus atau edit entri lama; hanya tambahkan entri baru di bawah.
> Format: `[YYYY-MM-DD] [SESSION-NNN] [TYPE] Deskripsi singkat`

---

## Cara Membaca Log Ini

- **[SESSION]** — Sesi kerja baru dimulai; link ke file SESSION-NNN yang detail
- **[DECISION]** — Keputusan arsitektur atau teknis dibuat
- **[FINDING]** — Temuan baru dari eksperimen atau research
- **[CONFLICT]** — Ditemukan konflik antar spec; perlu resolusi
- **[OOM]** — Ditemukan OOM event; catat kondisi dan workaround
- **[UPDATE]** — Spec atau RFC di-update
- **[OPEN]** — Open question baru yang perlu dijawab
- **[RESOLVED]** — Open question terjawab; referensikan ke evidence

---

## Index Sesi

| Session | Tanggal | Deskripsi | Status |
|---|---|---|---|
| SESSION-001 | Bootstrap | Setup spec system, analisis hardware, keputusan teknis awal | ✅ Complete |

---

## Log Entries

### SESSION-001 — Bootstrap

```
[Bootstrap] [SESSION-001] Sesi pertama: setup spec-driven development framework
  → Lihat: spec/handoff/SESSION-001-bootstrap.md
```

```
[Bootstrap] [DECISION] Backend: llama.cpp dengan CUDA SM 61 dipilih sebagai satu-satunya backend
  → Alasan: satu-satunya backend yang support Pascal SM 6.1 secara eksplisit dengan GGUF dequant kernels
  → Lihat: spec/specs/RFC-001-quantization-strategy.md
```

```
[Bootstrap] [DECISION] Quantization Qwen3-8B: Q5_K_M sebagai primary, Q4_K_M sebagai fallback
  → Alasan: Q5_K_M ~7.1 GB total VRAM (dengan KV q8_0, 4096 ctx) — nyaman di 8 GB
  → Lihat: spec/specs/RFC-001-quantization-strategy.md
```

```
[Bootstrap] [DECISION] Quantization Qwen3-27B: Q3_K_M sebagai max viable, Q2_K sebagai safe fallback
  → Alasan: Q4_K_M ~16.4 GB melebihi 16 GB RAM; Q3_K_M ~15.5 GB total (marginal)
  → Lihat: spec/specs/RFC-001-quantization-strategy.md
```

```
[Bootstrap] [DECISION] KV cache quantization: q8_0 default untuk 8B, q4_0 untuk 27B hybrid
  → Alasan: Hemat 144–288 MB VRAM tanpa signifikan impact kualitas (perlu verifikasi empiris)
  → Lihat: spec/specs/RFC-001-quantization-strategy.md
```

```
[Bootstrap] [OPEN] Berapa layer count Qwen3-8B dan Qwen3-27B yang sebenarnya?
  → Status: UNRESOLVED
  → Akan dijawab di: SPEC-002 Task 2 dan SPEC-005 Task 2
```

```
[Bootstrap] [OPEN] Apakah KV cache q4_0 memengaruhi kualitas output secara signifikan?
  → Status: UNRESOLVED
  → Akan dijawab di: SPEC-004 (KV Cache Optimizer)
```

```
[Bootstrap] [OPEN] Apakah system RAM 16 GB cukup untuk Q3_K_M 27B tanpa swap?
  → Status: UNRESOLVED
  → Akan dijawab di: SPEC-005 Task 3
```

```
[Bootstrap] [UPDATE] Spec system bootstrap selesai:
  - spec/AGENTS/MASTER.md ✅
  - spec/AGENTS/KIRO-USAGE.md ✅
  - spec/AGENTS/roles/ (4 files) ✅
  - spec/AGENTS/phases/ (3 files) ✅
  - spec/specs/SPEC-000-project-charter.md ✅
  - spec/specs/RFC-001-quantization-strategy.md ✅
  - spec/specs/SPEC-002-inference-baseline/ (3 files) ✅
  - spec/specs/SPEC-003-memory-manager/ (skeleton) ✅
  - spec/specs/SPEC-004-kv-cache-optimizer/ (skeleton) ✅
  - spec/specs/SPEC-005-cpu-gpu-offload/ (skeleton) ✅
  - spec/specs/SPEC-006-benchmark-suite/ (skeleton) ✅
  - spec/specs/_TEMPLATE-feature-spec/ ✅
  - spec/handoff/ (dokumen ini) ✅
```

---

## Open Questions Aktif

Daftar open questions yang belum terjawab (update saat resolved):

| ID | Pertanyaan | Akan dijawab di | Status |
|---|---|---|---|
| OQ-001 | Berapa layer count Qwen3-8B yang sebenarnya? | SPEC-002 Task 2 | ❓ Open |
| OQ-002 | Berapa layer count Qwen3-27B yang sebenarnya? | SPEC-005 Task 2 | ❓ Open |
| OQ-003 | Berapa KV head count Qwen3 yang sebenarnya? | SPEC-002 Task 2 | ❓ Open |
| OQ-004 | Apakah KV q4_0 menurunkan kualitas output secara signifikan? | SPEC-004 | ❓ Open |
| OQ-005 | Apakah RAM 16 GB cukup untuk Q3_K_M 27B tanpa swap? | SPEC-005 Task 3 | ❓ Open |
| OQ-006 | Berapa TPS aktual (bukan estimasi) untuk masing-masing skenario? | SPEC-002 Task 5 | ❓ Open |
| OQ-007 | Apakah llama.cpp terbaru ada regresi di Pascal? | SPEC-002 Task 2 | ❓ Open |
| OQ-008 | Berapa ctx-size maksimum sebelum VRAM OOM untuk Q5_K_M? | SPEC-002 Task 6 | ❓ Open |

---

*Log dibuat pada sesi bootstrap. Append entri baru di bawah bagian "Log Entries" setelah header SESSION yang sesuai.*

### Post-bootstrap — Custom MCP

```
[Post-bootstrap] [OPEN] Desain MCP server kustom (4 kandidat: budget, hw, llama, spec)
  → Status: PROPOSED, belum diputuskan
  → Lihat: spec/specs/RFC-002-custom-mcp-servers.md
  → Diskusi: GitHub issues #1 (umbrella), #2 (A), #3 (B), #4 (C), #5 (D)
  → Pertanyaan kunci: timeout tool call di Kiro, llama-cli vs llama-server, 1 vs 4 server
```

```
[Post-bootstrap] [DECISION] RFC-002 direvisi: 4 MCP server diganti satu CLI (tools/legacy.py)
  → Alasan: lebih mudah dipasang dan dipakai orang lain; MCP menjadi pembungkus opsional
  → File RFC diganti nama: RFC-002-custom-mcp-servers.md → RFC-002-cli-tooling.md
  → Entri OPEN di atas yang menyebut 4 MCP server digantikan oleh entri ini
  → Status: masih PROPOSED; diskusi di GitHub issues #1..#5
```

### Koreksi model — Qwen3.8-27B

```
[Post-bootstrap] [CORRECTION] Target model dikoreksi: BUKAN "Qwen3-8B dan Qwen3-27B", melainkan Qwen3.8-27B saja
  → Entri bootstrap di atas yang menyebut Qwen3-8B/Qwen3-27B, 52/36 layer, Q5_K_M untuk 8B, dan
    "Q4_K_M tidak muat di 16 GB RAM" digantikan oleh entri ini (log append-only, entri lama tidak diubah)
  → Sebab: riset awal memakai pengetahuan model tanpa sumber. Goal user adalah "qwen 3.8 27B"
  → Sumber koreksi: model card komunitas (6block, guideboardlabs). Belum kartu resmi Qwen
  → Dokumen yang diperbarui: MASTER, RFC-001, SPEC-000, SPEC-002, SPEC-004 (design), SPEC-005, roles/*
```

```
[Post-bootstrap] [FINDING] Qwen3.8-27B: dense, hibrida. 64 layer = 16 full attention (GQA) + 48 Gated DeltaNet
  → KV cache ≈ 64 KiB/token (hanya 16 layer). State DeltaNet ukuran tetap, besarnya belum diketahui
  → Vocab 248,320, embedding dan output tidak di-tie. MTP head tersedia. mmproj terpisah dan opsional
  → Ukuran file (GiB): Q5_K_M 18.70, Q4_K_M 16.53, IQ4_XS 15.28, Q3_K_M 13.23, IQ3_M 12.56, IQ3_XXS 11.60
  → Ada quant ~7.68 GiB yang dilaporkan muat penuh di GPU 8 GB (RX 5700 XT, Vulkan) dengan 24.48 tok/s
```

```
[Post-bootstrap] [FINDING] Koreksi perhitungan memori: bobot di VRAM tidak perlu ada di RAM
  → Q4_K_M diperkirakan butuh ~12.5 GiB RAM (bukan "tidak muat"). Semua masih estimasi
```

```
[Post-bootstrap] [OPEN] OQ-009 Apakah kernel Gated DeltaNet di llama.cpp CUDA jalan di SM 6.1?  → SPEC-002 Task 5 (risiko terbesar)
[Post-bootstrap] [OPEN] OQ-010 Kartu resmi Qwen/Qwen3.8-27B: layer, KV heads, head dim, ukuran state DeltaNet?  → SPEC-002 Task 4
[Post-bootstrap] [OPEN] OQ-011 Bandwidth RAM aktual; single atau dual channel; model CPU?  → SPEC-002 Task 1
[Post-bootstrap] [OPEN] OQ-012 Apakah MTP speculative decoding jalan di build kita dan berapa percepatannya?  → SPEC-005 Task 6
[Post-bootstrap] [OPEN] OQ-013 Apakah strategi B (muat penuh ~7.7 GiB) cukup ruang di 1070 Ti dan berapa turun kualitasnya?  → SPEC-005 Task 5
[Post-bootstrap] [OPEN] OQ-014 Auto-fit layer vs -ngl manual: mana lebih baik?  → SPEC-005 Task 3
[Post-bootstrap] [OPEN] OQ-015 Apakah IQ-quant cukup cepat di Pascal?  → SPEC-005 Task 4
[Post-bootstrap] [OPEN] OQ-016 Selisih TPS Vulkan vs CUDA di Pascal?  → SPEC-005 Task 6
[Post-bootstrap] [RESOLVED] OQ-001, OQ-002, OQ-003 digantikan oleh OQ-010. OQ-004 (KV q4_0) kini rendah prioritas karena KV kecil
```

### Agen GitHub /kiro

```
[Post-bootstrap] [DECISION] Agen pemicu GitHub: skrip di laptop (tools/agent/), bukan GitHub Actions
  → Hanya ID 68177813, 73787745, 113233053; pemicu "/kiro"; 12 run/jam; PR dari fork diabaikan; hasil lewat PR
  → Lihat: spec/specs/RFC-003-github-agent-trigger.md, tools/agent/README.md
[Post-bootstrap] [FINDING] Uji nyata (issue #13..#15, PR #16..#17, semuanya ditutup):
  → balasan benar; PR dibuat; perubahan ke tools/agent/config.json ditolak; baca ~/.config/gh ditolak
[Post-bootstrap] [FINDING] bwrap (bubblewrap) belum terpasang. Mode shell dimatikan sampai terpasang dan diuji
[Post-bootstrap] [OPEN] OQ-017 Apakah --sandbox mengizinkan akses GPU (/dev/nvidia*)?  → setelah bubblewrap terpasang
```

### CLI tooling — SPEC-007

```
[Post-bootstrap] [UPDATE] SPEC-007-cli-tooling dibuat; tools/legacy.py dan tools/test_legacy.py ditulis atas permintaan di issue #1
  → Cakupan: budget, hw, gguf, llama (build/run/bench/placement), spec. Belum ada: llama compare, llama-server, bench analyze, quality eval
  → Gambar: spec/specs/assets/spec-007-cli-implementation.svg; panduan pakai: tools/README.md
  → RFC-002 tetap Proposed; keputusan desain (versi Python, llama-cli vs llama-server) belum diambil pemilik repo
```

```
[Post-bootstrap] [OPEN] Test tools/test_legacy.py ditulis oleh agen tanpa shell dan belum dijalankan
  → Perlu: jalankan `python tools/test_legacy.py -v` dan `python tools/legacy.py spec validate`, lalu catat hasilnya
  → Asumsi yang perlu dicek: tabel tipe ggml, opsi llama.cpp (-o json, -no-cnv), format baris log/timing
```

### Pengukuran nyata pertama (lewat `tools/legacy.py`, PR ini)

```
[Post-bootstrap] [FINDING] CPU: AMD Ryzen 5 2400G, 4 core / 8 thread, AVX2 ya, AVX-512 tidak (hw cpu)
[Post-bootstrap] [FINDING] RAM: 15.53 GiB total, 9.25 GiB tersedia saat diukur, swap 4 GiB (hw ram). Single/dual channel BELUM diketahui
[Post-bootstrap] [FINDING] Bandwidth salin RAM satu thread Python: 11.09 GB/s (hw membw). Ini batas bawah kasar, bukan bandwidth maksimum
[Post-bootstrap] [FINDING] VRAM saat diukur: 7875 dari 8192 MiB terpakai, 231 MiB bebas. Pemakainya sebuah llama-server dari build lokal
    "llama-cpp-turboquant" milik pemilik repo. Artinya build llama.cpp CUDA sudah berjalan di GPU Pascal ini.
    Selama proses itu hidup, tidak ada uji GPU lain yang bisa jalan. Fork "turboquant" layak dicek: kemungkinan relevan dengan quantization
[Post-bootstrap] [UPDATE] OQ-011 terjawab sebagian: CPU dan RAM diketahui; channel dan bandwidth maksimum belum
[Post-bootstrap] [FINDING] Sandbox Kiro (--sandbox): token gh tersembunyi, home read-only, jaringan terblokir, /dev/nvidia* TIDAK terlihat
    Konsekuensi: agen bersandbox tidak bisa menjalankan uji GPU. Uji CPU bisa
```

### Perbaikan: satu PR per issue, SVG di PR

```
[Post-bootstrap] [CORRECTION] PR #18 dan #19 seharusnya satu PR. Akar masalah: skrip agen membuat branch baru dari main
    untuk setiap /kiro di issue, jadi permintaan lanjutan ("tambahkan SVG di PR 18") menjadi PR terpisah
  → PR #19 digabung ke PR #18 lalu ditutup. Skrip diperbaiki: perintah lanjutan menambah ke PR terbuka milik issue yang sama
  → Lihat: spec/specs/RFC-003-github-agent-trigger.md (bagian Revisi)
[Post-bootstrap] [DECISION] Aturan PR dicatat di MASTER.md: satu tugas satu PR; deskripsi teks tapi divisualkan dengan SVG
    sebisa mungkin; bukti test dan daftar "yang belum" wajib. SVG disematkan lewat tautan SHA
```

### Perbaikan: perilaku /kiro menurut tempat komentar

```
[Post-bootstrap] [CORRECTION] Pesan awal agen selalu menulis "membuat PR baru bila ada perubahan", termasuk saat komentar ada di dalam PR
  → Di dalam PR: dibahas dan dikerjakan di PR itu. Di issue tanpa PR: analisis dulu, PR hanya bila perlu. Di issue dengan PR: tambah ke PR itu
  → Pengaman: "Keputusan: tidak perlu PR" + perubahan file = tidak di-commit
  → Uji nyata: issue pertanyaan tanpa PR; issue perubahan dengan PR; komentar di PR (pertanyaan dan perubahan) tanpa PR baru
  → Lihat: spec/specs/RFC-003-github-agent-trigger.md (Revisi 2)
[Post-bootstrap] [FINDING] Komentar pada baris kode (tab Files changed) dan isi review tidak dipantau; hanya komentar di tab Conversation
```

### Sequential Thinking MCP untuk agent /kiro

```
[Post-bootstrap] [DECISION] Agent /kiro wajib berpikir bertahap (sequentialthinking MCP) sebelum bertindak
  → Terpasang: Node v24.22.3 (SHA-256 cocok) dan @modelcontextprotocol/server-sequential-thinking 2.1.1, 0 kerentanan
  → Didaftarkan di profil agent bot, bukan di ~/.kiro/settings (Kiro melarang agent menulis ke sana)
  → Uji nyata: tool dipanggil 2 kali, jawaban memuat "Pemahaman", keputusan "tidak perlu PR" tanpa PR
  → Lihat: spec/specs/RFC-003-github-agent-trigger.md (Revisi 3), tools/agent/README.md
[Post-bootstrap] [FINDING] Kiro menolak fs_write ke ~/.kiro/settings/mcp.json (aturan bawaan). Pendaftaran global dilakukan pemilik repo
```

### Riset dan koreksi (2026-11-13): kartu resmi, llama.cpp utama, iso3, kalibrasi

```
[Post-bootstrap] [FINDING] Kartu resmi Qwen/Qwen3.8-27B dicocokkan: 64 layer = 16 x (3 DeltaNet + 1 Gated Attention), hidden 5120,
    4 head KV x head dim 256, 48 head V DeltaNet, vocab 248.320, konteks 262.144, MTP dilatih. Repo resmi tanpa GGUF
  → KV 64 KiB/token terkonfirmasi (turunan resmi); dugaan "kv_heads x head_dim = 1024 turunan" diganti angka resmi (4 x 256)
[Post-bootstrap] [RESOLVED] OQ-010 kartu resmi: layer, KV heads, head dim terjawab. State DeltaNet = 0.146 GiB (turunan, rumus cocok persis pada 0.8B)
[Post-bootstrap] [UPDATE] OQ-009: kernel Gated DeltaNet berjalan di GTX 1070 Ti untuk model qwen35 9B di fork (server pengguna, 20.4 tok/s). Belum untuk 27B dan llama.cpp utama
[Post-bootstrap] [FINDING] llama.cpp utama mendukung qwen35 (CUDA Ada/Blackwell, ROCm RDNA2) dan MTP (--spec-type draft-mtp). Fork 08e025c tidak punya draft-mtp
  → Strategi C butuh build llama.cpp utama atau fork lebih baru
[Post-bootstrap] [FINDING] iso3 = IsoQuant (rotasi kuaternion 4D + Lloyd-Max 3-bit), 50 byte per 128 nilai (0,195x FP16). Hanya di fork. Dampak kualitas belum diukur
[Post-bootstrap] [FINDING] "Qwen3.8-9B" di Hugging Face adalah distilasi komunitas ke arsitektur Qwen3.5-9B. 9B dipakai sebagai pengganti ARSITEKTUR, bukan 27B
[Post-bootstrap] [CORRECTION] MASTER.md menulis "Tidak ada 8B yang terverifikasi" dan menganggap semua fakta model dari komunitas
  → Dikoreksi: fakta model dari kartu resmi; ada GGUF 9B qwen35 lokal (terukur)
[Post-bootstrap] [FINDING] Kalibrasi budget pada 9B: estimate 6.99 GiB vs nvidia-smi 7.70 GiB. Overhead nyata ≈ 1.6 GiB di luar bobot dan KV
  → Strategi B (Q3-DOWN-XS 7.68 GiB) tidak muat penuh di 1070 Ti menurut hitungan: ±1.5 GiB tetap di RAM
[Post-bootstrap] [UPDATE] OQ-011: laju salin RAM 14.15 GB/s (2 thread); bandwidth baca CPU minimal 18.4 GB/s (0.8B di CPU)
  → Dual channel kemungkinan besar (lalu lintas 28 GB/s > puncak single channel DDR4-2933 23.4 GB/s) tetapi belum dikonfirmasi; butuh sudo dmidecode
[Post-bootstrap] [UPDATE] OQ-013: hitungan menunjukkan B tidak muat penuh; TPS dan kualitas nyata belum diukur (butuh GGUF 27B)
[Post-bootstrap] [UPDATE] OQ-012: fork tanpa draft-mtp; percepatan di hybrid tidak diketahui (pihak lain: 1.3-1.5x di GPU penuh, 1.1x di ROCm)
[Post-bootstrap] [DECISION] RFC-002 Accepted. Satu file (ambang diganti: >2500 baris atau kebutuhan impor); proses panjang = background + polling;
    llama-cli per panggilan (llama-server tidak dibungkus); hook spec validate saat simpan; Python 3.8 (hanya 3.10.12 diuji); tanpa pembungkus MCP
  → Diputuskan agen berdasarkan bukti; pemilik repo dapat membukanya kembali. Lihat RFC-002, bagian Keputusan
[Post-bootstrap] [CORRECTION] Tiga cacat CLI ditemukan saat uji binary asli: llama run macet (-no-cnv diabaikan, stdin), kv_layers_estimate salah pada hibrida,
    tidak ada test kontrak. Diperbaiki di PR #29
```

### iso3 + MTP di llama.cpp utama (2026-11-14)

```
[Post-bootstrap] [UPDATE] iso3 diporting ke llama.cpp utama 3549c27 (CPU + CUDA), patches/llama.cpp/. TDD: red lalu green
  → test-iso3 PASSED, test-backend-ops CUDA0 iso3 28/28, regresi 5041/5041, ctest -L main rc 0
[Post-bootstrap] [FINDING] KLD iso3 vs f16 pada 0.8B (8 chunk): 0.0495, same top p 87.7%
[Post-bootstrap] [FINDING] MTP + iso3 jalan bersama di 27B IQ4_XS. Keluaran MTP tidak identik bit dengan tanpa MTP, juga pada f16
[Post-bootstrap] [FINDING] KV konteks MTP (1 layer) memakai -ctkd/-ctvd (default f16), tidak mewarisi -ctk/-ctv. Belum diubah (keputusan perilaku)
  → Lihat spec/research/reports/2026-11-14-iso3-mtp-main.md
```

### iso3 konteks panjang dan kualitas 27B (2026-11-15)

```
[Post-bootstrap] [FINDING] ctx 16384, Q4_K_M: KV iso3 ngl 31 2.23 tok/s vs f16 ngl 28 1.86 (+20%); iso3+MTP d1 ngl 28 2.37 vs f16+MTP d1 ngl 26 1.80
[Post-bootstrap] [FINDING] iso3 + MTP depth 3 pada ctx 16384 ngl 28: crash CUDA OOM di tengah generasi (bukan saat muat)
[Post-bootstrap] [UPDATE] OQ-004: KLD vs Q4_K_M (wikitext-2, ctx 512, 20 chunk): KV q8_0 0.0005, KV iso3 0.0102, IQ4_XS 0.0162, Q4_0 0.0301, IQ3_XXS 0.0604
    PPL: Q4_K_M 6.919, IQ4_XS 6.949, Q4_0 7.057, IQ3_XXS 7.203
[Post-bootstrap] [FINDING] llama-perplexity Q4_K_M ngl 30 gagal (buffer compute 525 MiB); ngl 26 dipakai
  → Lihat spec/research/reports/2026-11-15-27b-quality-001.md dan 2026-11-14-iso3-mtp-main.md
```

### Konteks panjang dengan -ot dan 8 prompt tetap (2026-11-15)

```
[Post-bootstrap] [FINDING] ctx 16384 + -ot: iso3 (FFN 15..64 di CPU) 2.61 tok/s, f16 (FFN 9..64) 2.52, iso3 + MTP d1 (FFN 11..64) 2.37
  → Konfigurasi tercepat terukur di ctx 16k: Q4_K_M + KV iso3 + -ot FFN 15..64, tanpa MTP
[Post-bootstrap] [FINDING] iso3 + MTP d1 + -ot FFN 12..64: crash CUDA OOM (40 MiB) di tengah generasi
[Post-bootstrap] [UPDATE] OQ-004: 8 prompt tetap, kemiripan vs Q4_K_M: IQ4_XS 0.583, Q4_0 0.571, IQ3_XXS 0.504; 3 cek fakta benar di semua quant
  → Lihat spec/research/reports/2026-11-15-27b-quality-001.md
```

```
[Post-bootstrap] [FINDING] ctx 16384 + -ot, KV q8_0: 2.46 tok/s (FFN 12..64); + MTP d1 2.24 (FFN 8..64). MTP tidak mempercepat pada -ot di KV mana pun yang diuji
```

### Penghentian server 9B (izin pemilik mesin, 2026-11-13)

```
[Post-bootstrap] [DECISION] Server 9B (PID 22885) dihentikan dengan SIGTERM untuk uji 27B (#31-#36), atas izin pemilik mesin
  → Perintah start lama (dari /proc/22885/cmdline, jalankan dari ~):
    $HOME/llama-cpp-turboquant/build/bin/llama-server -m $HOME/models/Qwen3.8-9B-Q4_K_M.gguf -ngl 99 -c 114688 -ub 256 -b 512 --port 8080 --host 0.0.0.0 --metrics --slots -fa auto --cache-type-k iso3 --cache-type-v iso3 --reasoning-budget 0
  → Fork dipindah ke ~/archive/llama-cpp-turboquant-08e025c; ganti path binary saat menghidupkan lagi
```

### Uji 27B run-001 (2026-11-14)

```
[Post-bootstrap] [UPDATE] llama.cpp utama 3549c27 dibangun di ~/llama.cpp/build-cuda (sm_61). Gated DeltaNet 27B jalan di Pascal (OQ-009)
[Post-bootstrap] [UPDATE] OQ-005/OQ-015: IQ4_XS hybrid 2.29 tok/s (ngl 36), IQ3_XXS 1.15 tok/s (ngl 45). IQ3_XXS lebih lambat walau lebih kecil
[Post-bootstrap] [UPDATE] OQ-012: MTP di hybrid 1.06x (depth 1), tidak ada percepatan di depth 2-3
[Post-bootstrap] [CONFLICT] block_count GGUF 27B = 65 (64 + 1 layer MTP), kartu resmi 64 layer. Bukan kesalahan, layer MTP disimpan sebagai blok
[Post-bootstrap] [FINDING] Build utama gagal memuat Qwen3.8-9B-Q4_K_M.gguf pengguna (expected 442 tensors, got 427). Server 9B tetap harus pakai fork
  → Lihat spec/research/reports/2026-11-14-27b-run-001.md
```

### Uji 27B run-002 (2026-11-15): thread, Q4_K_M, Q4_0, --override-tensor

```
[Post-bootstrap] [FINDING] Sapuan -t (IQ4_XS ngl 36): 1.40/1.90/2.32/2.29/2.21 tok/s untuk t=2/3/4/6/8. Naik linear sampai 4 core fisik lalu turun
  → Pada 2-4 thread CPU (komputasi) yang menahan; IQ4_XS hanya ±15.6 GB/s dari RAM, Q4_K_M ±22.0 GB/s (hitungan)
[Post-bootstrap] [UPDATE] OQ-015: dequant IQ4_XS lebih mahal di CPU ini. Q4_K_M ngl 32: 2.60 tok/s > IQ4_XS 2.29 > Q4_0 2.39 (ngl 33)
[Post-bootstrap] [FINDING] --override-tensor FFN layer 17..64 ke CPU, sisanya -ngl 99: Q4_K_M 2.85 tok/s (+9.6% dari ngl 32). k=18 OOM
[Post-bootstrap] [DECISION] Kuantisasi ulang dengan imatrix ditunda: butuh BF16 ±50 GB; Q4_0 resmi unsloth dipakai sebagai pembanding
  → Lihat spec/research/reports/2026-11-15-27b-run-002.md
```

```
[Post-bootstrap] [UPDATE] OQ-017: -fa on lebih lambat di Pascal pada ctx pendek (2.59 vs 2.84 tok/s, -ot FFN 17..64). Bawaan auto = off
[Post-bootstrap] [FINDING] Konfigurasi terbaik: RAM tersedia terendah 14.12 GiB, swap +0.175 GiB (0.724 -> 0.899), VRAM puncak 8043 MiB
[Post-bootstrap] [BLOCKER] Vulkan: glslc tidak ada (butuh sudo apt). Channel RAM: butuh sudo dmidecode
```

### Merge PR #37 dan #38, sesi 27B dicatat (2026-11-15)

```
[Post-bootstrap] [DECISION] PR #37 dan #38 di-merge oleh agen atas perintah langsung pemilik repo (gh pr merge --merge, tanpa melewati proteksi main)
[Post-bootstrap] [UPDATE] Konflik HANDOFF-LOG di PR #38 diselesaikan dengan menyimpan kedua sisi; 127 test lulus, spec validate ok
[Post-bootstrap] [UPDATE] CLAIMS C-53..C-58 ditambah; MASTER diberi hasil terukur (target MTP +30% tidak tercapai di mesin ini)
[Post-bootstrap] [OPEN] OQ-018 channel RAM (butuh sudo dmidecode), OQ-019 bit campuran per layer, OQ-020 pemangkasan layer (belum diputuskan pemilik)
  → Lihat spec/handoff/SESSION-002-27b-test.md
```

### #33: tabel quant ctx 4096 dan OQ-014 (2026-11-15)

```
[Post-bootstrap] [CONFLICT] IQ3_M tidak ada di unsloth/Qwen3.8-27B-GGUF. Diganti UD-IQ3_S dan UD-Q3_K_XL (sha256 cocok LFS). Prediksi IQ3_M di MASTER/RFC-001 tidak diuji langsung
[Post-bootstrap] [FINDING] ctx 4096, terbaik -ngl / -ot (tok/s): Q4_K_M 2.04/2.57, IQ4_XS 1.85/2.87, IQ3_S 1.67/2.40, Q3_K_XL 1.85/2.25. Batas -ot Q4_K_M turun dari k=17 ke k=11 di ctx 4096
[Post-bootstrap] [UPDATE] OQ-014 dijawab: auto-fit (-fitt 512) 3-10% di bawah -ngl manual, 23-41% di bawah -ot; tidak OOM
[Post-bootstrap] [FINDING] Prediksi MASTER "quant lebih kecil lebih cepat" salah untuk IQ3: IQ3_S -ngl 1.67 < IQ4_XS 1.85
  → Lihat spec/research/reports/2026-11-15-27b-run-003-ctx4096.md, CLAIMS C-59..C-62
### Mesin dipindah ke fork di dalam repo (2026-11-15)

```
[Post-bootstrap] [DECISION] Kode mesin (iso3 di llama.cpp utama) dipush ke fork publik mcpe500/llama.cpp, cabang feat/iso3-kv (36dbad3), atas permintaan pemilik
[Post-bootstrap] [UPDATE] Fork masuk repo ini sebagai submodule engine/llama.cpp, terkunci di 36dbad3. patches/llama.cpp/ dipertahankan sebagai riwayat
  → Clone dengan --recurse-submodules. Perubahan mesin berikutnya: commit di fork, lalu naikkan pointer submodule lewat PR
```

### #34: Vulkan dan Strategi B (2026-11-15)

```
[Post-bootstrap] [UPDATE] BLOCKER glslc selesai: LunarG SDK 1.4.363.1 di ~/tools tanpa sudo; ~/llama.cpp/build-vulkan terbangun (36dbad3)
[Post-bootstrap] [FINDING] OQ-016: Vulkan 1-6% lebih lambat dari CUDA pada Q4_K_M (2.01 vs 2.04, 2.41 vs 2.57), tanpa crash decode; ±0.9 GiB lebih banyak RAM host
[Post-bootstrap] [FINDING] Q3-DOWN-XS asli (7.91 GiB, bukan 7.68) tidak muat 8 GiB; terbaik Vulkan ngl 62 7.05 tok/s, CUDA ngl 60 5.54. MASTER min 8 tidak tercapai
[Post-bootstrap] [FINDING] Kualitas Q3-DOWN-XS: PPL +22% vs Q4_K_M, KLD 0.273, melewati batas 10%
[Post-bootstrap] [CONFLICT] Ukuran Q3-DOWN-XS di RFC-001/MASTER 7.68 GiB vs berkas 7.91 GiB
[Post-bootstrap] [FINDING] Q3-DOWN-XS CUDA KV q4_0: ngl 63 muat, 6.51 tok/s (+17% vs f16 5.54); -fa 1 tidak berpengaruh berarti (5.64)
[Post-bootstrap] [OPEN] KV iso3 pada Q3-DOWN-XS (llama-bench menolak -ctk iso3); BF16 tidak diunduh (tidak perlu)
  → Lihat spec/research/reports/2026-11-15-27b-run-004-vulkan-strategy-b.md, CLAIMS C-63..C-68
```

### SPEC-008: single-file executable + RFC-004 embedding format (2026-11-18)

```
[Post-bootstrap] [DECISION] Ide "compile bobot LLM ke dalam executable webserver" dikerjakan sebagai SPEC-008 + RFC-004: append + mmap zero-copy dari /proc/self/exe dipilih; C-array hex (embed.cmake) ditolak (±28 GB source, C-72); .incbin dicatat sebagai cadangan
[Post-bootstrap] [DECISION] Non-goal eksplisit: embedding bukan optimasi kecepatan. Batas atas decode = bandwidth / bytes bobot per token (C-71); Q4_K_M hybrid tetap 2.85 tok/s (C-54) baik via -m maupun payload. Manfaatnya: satu artefak ter-versi-lock (engine commit + SHA-256 payload), bukan angka tok/s
[Post-bootstrap] [UPDATE] Payload yang diizinkan dibatasi kelas Bonsai (muat RAM/VRAM): PTQ1_0 5,946,648,928 B, Q2_0 7,626,008,928 B (C-69). Q4_K_M 15.43 GiB tetap lewat override -m
[Post-bootstrap] [UPDATE] Batas mesin dicatat sebagai klaim baru: mlock 1.94 GiB tanpa root, MemTotal 15,905 MiB (C-70)
[Post-bootstrap] [OPEN] Baris CLAIMS untuk eksperimen Ternary Bonsai (tok/s, PPL, KLD) belum ada; angka hanya hidup di qwen38-engine/bench/results/bonsai/. Perlu PR dokumentasi terpisah sebelum boleh dikutip sebagai klaim
[Post-bootstrap] [OPEN] Eksperimen bahasa pemaket (C/C++/Goose) masuk Task 6 SPEC-008 dengan decision gate >10%; ekspektasi ditulis eksplisit: tidak ada harness bahasa yang menembus batas bandwidth RAM
  → Lihat spec/specs/SPEC-008-single-file-executable/, spec/specs/RFC-004-embed-format.md, CLAIMS C-69..C-72
```

### Branch `exp/bonsai-128k-context`: engine Bonsai + dokumentasi 128K (2026-11-18)

```
[Post-bootstrap] [DECISION] Cabang dokumentasi/engineering `exp/bonsai-128k-context` dibuat dari `main`: SPEC-008 + RFC-004 berada di bawah `7a2ff6c`, integrasi Ternary Bonsai di `a89f467`, lalu perubahan working tree untuk attention tiled, budget memori, state file, dan probe kedalaman
[Post-bootstrap] [UPDATE] `[OPEN]` pada sesi SPEC-008 untuk baris CLAIMS Bonsai ditutup oleh klaim baru C-73..C-83; entri [OPEN] lama dibiarkan utuh sesuai aturan append-only
[Post-bootstrap] [FINDING] Attention `qwen38-engine` kini punya kernel tiled online-softmax di atas 8192 posisi: `attn_long_test 131072` lulus 16/16 kasus (f32 dan iso3), galat relatif maksimum 2,112e-05 (C-77)
[Post-bootstrap] [FINDING] Batas konteks engine: Qwen K-quant maksimum 8192; Ternary Bonsai maksimum 131072. Auto-fit memilih 64 layer GPU untuk PTQ1_0 pada 131072 dan 53 layer GPU untuk Q2_0 (C-83, C-76)
[Post-bootstrap] [FINDING] Probe `--depth-bench` pada 131072: PTQ1_0 0,4390 tok/s (7293 MiB), Q2_0 0,3786 tok/s (7285 MiB). Metodenya: warm-up satu token pada N-9 lalu delapan langkah terukur pada N-8..N-1 dengan KV nol — bukan prefill penuh, bukan uji kualitas konteks panjang (C-76)
[Post-bootstrap] [FINDING] State file engine lolos round-trip untuk Qwen Q4_K_M, Bonsai PTQ1_0, dan Bonsai Q2_0 pada prefix 957/3989 token: token identik dan baris logits pertama bit-identik (C-74)
[Post-bootstrap] [FINDING] Generasi nyata Bonsai pada engine: PTQ1_0 1,0709 tok/s dan Q2_0 2,8018 tok/s di ctx 4096. Angka ini implementasi engine, bukan angka stock llama.cpp/PrismML 17,13/19,64 tok/s (C-75, C-79)
[Post-bootstrap] [UPDATE] Dokumentasi diperbarui: README root, qwen38-engine/README.md, bench/README.md, README submodul yang masih "stage 0", docs/AFK_STATUS.md, plus ringkasan baru `qwen38-engine/bench/results/bonsai128k/README.md` dan SVG `spec/specs/assets/qwen38-engine-128k-context.svg`
[Post-bootstrap] [TEST] Build dan test pada 2026-11-18: `rtk err ninja -C build -j4` PASS; `attn_long_test 131072` 16/16 PASS; `ternary_test` PASS; `stop_filter_test` PASS; `mt_dot_test` PASS; `foundation_test` PASS; SASS gate PASS (IDP.4A x2125, half x0); `tools/test_legacy.py` 127 test OK (4 skip); `spec validate` PASS; `tools/webui_test.py` PASS
[Post-bootstrap] [OPEN] Prefill penuh 131072 belum dijalankan; yang ada hanya probe decode pada KV nol
[Post-bootstrap] [OPEN] Kualitas retrieval/needle pada konteks panjang belum diukur; `--depth-bench` tidak mengujinya
[Post-bootstrap] [OPEN] `PQ2_0` (GGML type 142) belum didukung engine; kecepatan engine Bonsai masih jauh di bawah llama.cpp/PrismML
[Post-bootstrap] [OPEN] WebUI belum dibuka di browser sungguhan; channel RAM masih butuh `sudo dmidecode -t memory`
  → Lihat qwen38-engine/bench/results/bonsai128k/, spec/research/CLAIMS.md C-73..C-83, spec/specs/SPEC-008-single-file-executable/
```

### `exp/bonsai-128k-context` lanjutan: kernel v2, split-K, MTP plumbing, perbaikan Pool (2026-11-19)

```
[Post-bootstrap] [FINDING] Depth bench diulang dengan `-n 33` (satu warm-up di N-33 + 32 langkah terukur): PTQ1_0 4,5906 tok/s dan Q2_0 0,4655 tok/s pada 131072; Q2_0 64K mencatat transient 7417 MiB (C-76)
[Post-bootstrap] [FINDING] Attention split-K 64 slice ditambahkan untuk decode/T kecil; `attn_long_test` kini 28/28 PASS, galat relatif maksimum 2,581e-05 (C-77)
[Post-bootstrap] [FINDING] Kernel ternary v2 lulus A/B byte-identical 135/135; PTQ1_0 microbench ffn_up T1 15,6→77,6 GB/s, LM head 18,7→91,1 GB/s (C-85)
[Post-bootstrap] [FINDING] Needle 4K kernel v2 tanpa MTP: PTQ1_0 8,1536 tok/s dan Q2_0 8,2092 tok/s, keduanya menjawab MAGENTA-FALCON-42
[Post-bootstrap] [FINDING] Inject `blk.64` Qwen ke Bonsai PTQ1_0 untuk uji plumbing: tanpa MTP 11,4932 tok/s, MTP 2 tetap 17,7078 tok/s, 158/198 draft diterima, 256 token identik (C-84)
[Post-bootstrap] [DECISION] Angka MTP di atas disebut plumbing saja karena head belum dilatih pada Bonsai; alat korpus/distilasi/injeksi disertakan di qwen38-engine/tools/bonsai_mtp_train/
[Post-bootstrap] [FINDING][CONFLICT] `server_test` gagal nondeterministik pada request kedua walau D=0, tanpa CUDA graph, dan dengan KV f32; binary d848269 juga gagal saat diuji ulang, sehingga log PASS lama tidak merefleksikan kondisi ini
[Post-bootstrap] [ROOT CAUSE] Caller HTTP berubah tiap request; `Pool::slot()` TLS lama melakukan fetch_add tanpa batas dan memakai fallback slot sama untuk semua caller ekstra, sehingga scratch attention score bertabrakan
[Post-bootstrap] [DECISION] `Pool::slot()` kini memberi caller aktif slot 0 dan worker persisten slot 1..n-1; caller disimpan per `parallel_for`, bukan TLS permanen (C-86)
[Post-bootstrap] [TEST] Setelah fix Pool: `server_test.py` + `QWEN_MT_DOT=0` PASS 69/69; ulangi 10 request greedy di server terpisah semuanya Paris
[Post-bootstrap] [TEST] Build `rtk err ninja -C build -j4` PASS; `attn_long_test 131072` 28/28 PASS; `ternary_test` PASS; `ternary_ab_test` 135/135 PASS; `mt_dot_test`, `stop_filter_test`, `foundation_test` PASS; SASS gate PASS (`IDP.4A x2813`, half x0); webui 0 fail; 127 CLI test OK; spec validate PASS; selftest korpus/injeksi/distilasi MTP PASS
[Post-bootstrap] [OPEN] Prefill penuh 131072 dan needle 128K belum dijalankan; antrian 128K state-test terputus sebelum hasil valid
[Post-bootstrap] [OPEN] Head MTP Bonsai terlatih belum tersedia; `PQ2_0` belum didukung engine
[Post-bootstrap] [OPEN] `docs/figures/51_bonsai_128k.svg` hasil generator masih memiliki teks <12 px; aset PR memakai `spec/specs/assets/qwen38-engine-128k-context.svg` yang lolos validator
[Post-bootstrap] [UPDATE] CLAIMS diperluas menjadi C-73..C-86; README root/engine/bench/results dan docs/AFK_STATUS.md diselaraskan dengan angka kernel v2, split-K, MTP plumbing, dan fix Pool
  → Lihat qwen38-engine/bench/results/bonsai128k/, spec/research/CLAIMS.md C-73..C-86, qwen38-engine/src/cpu/pool.hpp
```
