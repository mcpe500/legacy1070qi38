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
