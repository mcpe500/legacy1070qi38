# Penggunaan Kiro dalam Proyek legacy1070qi38

Dokumen ini mencatat apa yang **benar-benar dilakukan Kiro** dalam proyek ini, untuk
menjawab syarat submission: *"Name each product you used and what it does in your project
(Kiro tools, custom agents, hooks, specs, steering, MCP, PBT, powers)."*

> Prinsip dokumen ini: hanya mencatat fitur yang benar-benar dipakai. Fitur yang belum
> dipakai ditandai jelas pada bagian "Belum Digunakan". Update dokumen ini setiap kali
> ada fitur Kiro baru yang dipakai.

---

## Ringkasan

| Fitur Kiro | Dipakai? | Peran di proyek |
|---|---|---|
| Kiro CLI (`kiro-cli chat`) | ✅ Ya | Antarmuka utama seluruh sesi kerja |
| Knowledge base (`knowledge` tool) | ✅ Ya | Folder `spec/` diindeks (38 berkas, BM25) lalu dicari dengan pertanyaan bahasa Indonesia; lihat bagian 10 |
| Planning agent (plan → handoff ke execution) | ✅ Ya | Requirements gathering + rencana implementasi sebelum menulis file |
| Sub-agent `context-gatherer` | ✅ Ya | Riset teknis hardware/model, dan menulis file spec secara paralel |
| Built-in tools (read/list/search file) | ✅ Ya | Membaca dan memverifikasi isi workspace |
| Live steering | ✅ Ya | User menyisipkan instruksi saat sub-agent sedang berjalan |
| Web search dan web fetch | ✅ Ya | Memverifikasi model dan arsitektur; menangkap kesalahan asumsi awal |
| Headless mode (`--no-interactive`) | ✅ Ya | Menjalankan Kiro dari skrip pemantau GitHub tanpa sesi interaktif |
| Custom agent + permissions (`~/.kiro/agents/legacy-bot*.json`) | ✅ Ya | Profil agent bot dengan aturan izin per path dan tool; diuji menolak akses terlarang |
| Spec-driven development (requirements → design → tasks) | ✅ Ya (struktur) | Format `SPEC-NNN-*/` mengikuti pola requirements/design/tasks |
| Hooks | ✅ Ya | `.kiro/hooks/spec-validate-on-save.json` (`PostFileSave`, pola `spec/.*\.md$`) menjalankan `legacy.py spec validate`; dibuat lewat tool `createHook` |
| Steering files (`.kiro/steering/`) | ❌ Belum | — |
| Skills (`.kiro/skills/`) | ❌ Belum | — |
| MCP servers | ✅ Ya | Sequential Thinking MCP (`@modelcontextprotocol/server-sequential-thinking` 2.1.1) dipakai agent GitHub `/kiro` untuk menguraikan maksud permintaan sebelum bertindak |
| Workflow `/goal` (wf-coder, repeat + max iterations) | ✅ Ya | Mengeksekusi issue #31 dan directive AFK 8 jam secara berulang; dipakai untuk fase berat, lalu diabaikan setelah berulang kali "waiting input" (lihat Riwayat Sesi, fase 3–4) |
| Autonomous mode | ✅ Ya | Maraton AFK 8–10 jam tanpa bertanya; keterbatasan terukur: sandbox-nya tanpa akses GPU sehingga `ncu`/`nvprof` gagal — profiling manual oleh user, hasilnya dianalisa Kiro |
| Property-based testing (PBT) | ⚠️ Sebagian | `BudgetProperties` di `tools/test_legacy.py`: properti diuji pada 300 kasus acak (seed tetap), **tanpa library PBT** seperti Hypothesis |
| Powers | ❌ Belum | — |

---

## Detail Penggunaan

### 1. Kiro CLI (`kiro-cli chat`)
Seluruh sesi bootstrap proyek dijalankan lewat `kiro-cli chat` di terminal Linux.
Workspace: `$HOME/kiroween-2026/legacy1070qi38`.

### 2. Planning agent
Sesi dimulai dalam mode perencanaan (read-only): Kiro menanyakan klarifikasi
terstruktur (maks. 3 pertanyaan per giliran), merangkum requirement, lalu menyusun
rencana bertahap (Task 1–8). Setelah user menyetujui, rencana diserahkan ke mode
eksekusi untuk menulis file.

Keputusan yang lahir dari tahap ini (dijawab user):
- Format spec: hybrid (feature spec + RFC)
- AGENTS: hirarkis (`MASTER.md` + `roles/` + `phases/`)
- Handoff: session-based + continuous log
- Penamaan: `SPEC-NNN-nama`
- Kedalaman: dianalisis dulu, karena angka performa belum terukur

### 3. Sub-agent `context-gatherer`
Dipanggil lewat tool `invoke_sub_agent`. Dipakai untuk dua hal:
- **Riset**: analisis kelayakan awal (dengan asumsi model yang kemudian terbukti salah) pada GTX 1070 Ti (Pascal SM 6.1) + 16GB RAM —
  memory footprint, kuantisasi, estimasi TPS, bottleneck. Hasilnya menjadi dasar
  `RFC-001-quantization-strategy.md` dan tabel memory budget di `MASTER.md`.
- **Penulisan file**: membuat file di `spec/` secara paralel (roles, phases, specs, handoff).

Catatan kejujuran: sub-agent ini adalah agent bawaan Kiro, bukan custom agent yang
didefinisikan sendiri oleh proyek ini. Hasil risetnya berbasis pengetahuan model;
angka Qwen3 adalah **asumsi** yang belum diverifikasi (lihat Open Questions di `MASTER.md`).

### 4. Built-in tools
`read_file`, `list_directory`, `file_search`, `grep_search` — dipakai untuk membaca
LICENSE, memeriksa struktur workspace, dan memverifikasi file yang dibuat sub-agent.

### 5. Live steering
Saat sub-agent sedang menulis file, user mengirim pesan tambahan (ID
`steer-be120d5b-77b5-46fb-8eaf-9ae9a2d3a6cc`) agar penggunaan Kiro didokumentasikan.
Pesan itu diterima tanpa menghentikan sesi dan menghasilkan dokumen ini.

### 6. Spec-driven development
Struktur `spec/specs/SPEC-NNN-*/{requirements,design,tasks}.md` mengikuti pola spec
Kiro. Tambahan khas proyek ini (bukan bawaan Kiro): RFC untuk keputusan arsitektur,
`spec/AGENTS/` untuk instruksi agent, dan `spec/handoff/` untuk transisi antar sesi.

---

### 7. Web search dan web fetch
Dipakai setelah bootstrap untuk memeriksa nama model "Qwen 3.8 27B" yang disebut user. Hasilnya menunjukkan
bahwa riset awal sub-agent (berbasis pengetahuan model, tanpa sumber) memakai arsitektur yang salah:
bukan transformer biasa 52 layer, melainkan hibrida 64 layer (16 full attention + 48 Gated DeltaNet).
Dokumen `MASTER.md`, `RFC-001`, `SPEC-000`, `SPEC-002`, `SPEC-004`, `SPEC-005` dikoreksi dengan sumber model card.
Riwayat koreksi ada di `spec/handoff/HANDOFF-LOG.md`.

Pelajaran: klaim faktual wajib bersumber. Pencarian kedua (2026-11-13) mengambil kartu resmi `Qwen/Qwen3.8-27B`,
diskusi dan issue llama.cpp, serta dokumentasi vLLM; hasilnya ada di `spec/research/CLAIMS.md` (49 klaim, tiap baris punya jenis sumber).
Hasil penting: "Qwen3.8-9B" ternyata distilasi komunitas, dan fork kita tidak punya MTP.

### 8. Headless mode dan custom agent (agen GitHub `/kiro`)
`tools/agent/kiro_gh_agent.py` memantau issue GitHub. Perintah `/kiro` dari tiga ID yang diizinkan menjalankan
`kiro-cli chat --v3 --no-interactive --agent legacy-bot-noshell` memakai login Kiro yang ada.

- **Custom agent:** dua profil (`legacy-bot`, `legacy-bot-noshell`) ditulis oleh `install-agent` ke `~/.kiro/agents/`.
  Memuat daftar tool (`read`, `write`, `web`, dan `shell` hanya di profil bersandbox) dan `permissions.rules`.
- **Permissions:** `deny` untuk menulis ke `tools/agent`, `.github`, `.kiro`, `.git`; `deny` untuk membaca `~/.ssh`,
  `~/.config/gh`, folder status; tulis hanya di worktree.
- **Bukti uji nyata** (RFC-003): pembacaan token `gh` ditolak, perubahan `tools/agent/config.json` ditolak,
  penulisan ke `/tmp` tidak terjadi, perintah dari ID yang sah menghasilkan balasan dan PR.
- Kode dan aturan ditulis dengan bantuan Kiro (sesi ini). Uji coba memakai issue #13 sampai #15 dan PR #16, #17 (sudah ditutup).

### 9. MCP: Sequential Thinking untuk agent `/kiro`
Server MCP resmi `@modelcontextprotocol/server-sequential-thinking` (versi dikunci 2.1.1) dipasang lewat
`tools/agent/install-sequential-thinking.sh` dan didaftarkan di profil agent bot (`mcpServers`), bukan di
konfigurasi global. Agent diwajibkan memakainya sebelum bertindak, lalu menulis bagian "Pemahaman" di jawabannya.

![Alur berpikir bertahap agent /kiro](../specs/assets/rfc-003-sequential-thinking.svg)

- **Bukti uji:** agent memuat dan memanggil tool dua kali tanpa meminta izin (`thoughtHistoryLength` = 2); satu
  perintah `/kiro` nyata di issue menghasilkan jawaban dengan "Pemahaman" dan keputusan "tidak perlu PR" tanpa PR.
- **Pengaman:** izin `mcp` hanya untuk `sequential-thinking/*`; `--require-mcp-startup` membuat run gagal jelas bila
  server tidak naik, bukan diam-diam berjalan tanpa tool itu.
- **Catatan:** Kiro melarang agent menulis ke `~/.kiro/settings/`, jadi pendaftaran global untuk sesi interaktif
  dilakukan manual dengan `kiro-cli mcp add` (perintahnya dicetak oleh skrip pemasangan).

### 10. Hooks dan Knowledge base
- **Hook:** `.kiro/hooks/spec-validate-on-save.json` memicu `python3 tools/legacy.py spec validate` setiap berkas
  `spec/**/*.md` disimpan. Perintah yang sama sudah dijalankan tangan (`ok: True`). **Belum diamati terpicu**; hook
  aktif pada sesi berikutnya.
- **Knowledge base:** `knowledge add` pada `spec/` (indeks `fast`/BM25, 38 berkas) lalu `knowledge search`
  "berapa KV cache per token Qwen3.8-27B". Hasil pertama benar (`SPEC-004/design.md`), tetapi peringkat BM25
  kasar: hasil kedua dan ketiga hanya `HANDOFF-LOG.md`. Untuk tanya-jawab dokumen yang presisi, indeks
  semantik (`best`) belum dicoba.

---

## Belum Digunakan (rencana)

- **Hooks tambahan**: kandidat — auto-update `HANDOFF-LOG.md` setelah task selesai, atau
  pengingat memeriksa daftar approach terlarang saat file spec disimpan. (Satu hook sudah dipakai: lihat bagian 10.)
- **Steering files**: kandidat — memindahkan hardware constraints dari `MASTER.md`
  ke `.kiro/steering/` agar otomatis dimuat di setiap sesi.
- **PBT**: kandidat saat ada kode, mis. properti "estimasi VRAM selalu ≥ ukuran weights".
- **MCP kustom (buatan sendiri)**: opsional. Jalur utama adalah CLI `tools/legacy.py` (RFC-002; sudah ada,
  lihat PR #18). Pembungkus MCP baru dibuat jika diperlukan. Diskusi di GitHub issue #1.
- **Powers / Skills**: skills direncanakan (`vram-budget`, `run-benchmark`, `update-handoff`);
  powers belum ada kebutuhan konkret.

Hooks (`.kiro/hooks`) dan steering files di dalam repo sengaja dilarang untuk agen bot, karena menentukan perilaku agent pada run berikutnya.

Saat salah satu dipakai, pindahkan ke bagian "Detail Penggunaan" beserta buktinya.

---

*Dokumen ini ditulis ulang agar akurat. Versi sebelumnya memuat klaim tool yang tidak
benar-benar dipakai.*

---

## Riwayat Sesi Kiro (reasoning & decision log, 13–28 Nov 2026)

Dua sesi transkripnya dibaca ulang dari `~/.kiro/sessions/` dan diringkas di sini
agar history keputusan proyek lengkap. Sumber: `sess_1933586b-…` (sesi utama, 685
entri user, 2.530 tool call, 35 MB) dan `sess_e36bda6e-…` (sesi lanjutan 28 Nov).

### Sesi utama — `sess_1933586b-60a4-426b-bb33-e8de4f643e88` (13 Nov 17:25 → 28 Nov 05:10)

**Fase 1 — Bootstrap spec-driven framework (13 Nov).** Planning mode mengajukan
klarifikasi terstruktur; keputusan user: spec hybrid (feature spec + RFC), AGENTS
hirarkis, handoff session-based + continuous log, kedalaman dokumen dianalisis dulu
(karena angka performa belum terukur). Live steering menghasilkan dokumen ini
(KIRO-USAGE). Riset awal sub-agent memakai arsitektur model yang salah; web search
mengoreksinya (hibrida 64 layer: 16 attention + 48 Gated DeltaNet) — pelajaran
"klaim faktual wajib bersumber" melahirkan `spec/research/CLAIMS.md`.

**Fase 2 — Agen GitHub `/kiro` (14 Nov).** Keputusan arah: kebutuhan tooling dicatat
sebagai issue + SVG visual (bukan teks panjang), tool dibuat sebagai **CLI Python
satu file, bukan MCP daemon** (hemat RAM — nanti dikunci jadi aturan Zero-MCP di
`setup.sh`), Sequential Thinking MCP hanya untuk bot, `/kiro` hanya dari 3 username
(mcpe500, ibrahimmuh26, nicostap), batas 12 run/jam, branch protection main aktif.
PR #18/#19 diperbaiki jadi satu (menjadi PR #22) dengan konvensi baru: **deskripsi
PR/issue sebisa mungkin divisualkan SVG** — konvensi ini dipakai semua PR sesudahnya.

**Fase 3 — Issue #31 sebagai workflow `/goal` (14–15 Nov).** Qwen3.8-27B pertama
kali jalan di mesin: server 9B dimatikan (blokir pengaman Kiro terhadap `kill` —
user mengeksekusi sendiri), GGUF 27B diunduh, llama.cpp baru dipasang. Hasil
terukur: **2,85 tok/s** (Q4_K_M + `-ot` ffn 17..64), ctx 16K + KV iso3 = 2,61;
MTP komunitas = 2,2 tok/s. `iso3` di-port ke llama.cpp utama secara TDD (regresi
5041/5041, backend-ops 28/28) dan digabung dengan MTP — jadi PR #37/#38 (merged,
landasan `docs/ref_pr38_iso3_mtp.diff`). Workflow terhenti berulang di "waiting
input"; pelajaran operasional: `/goal` + subagent (`wf-coder`) rapuh untuk kerja
berjam-jam, screen/tmux tidak menolong karena workflow memang butuh sesi hidup.

**Fase 4 — Workspace engine + 8 jam AFK Stage 2→5 (15–16 Nov).** `prompt1.md`
dieksekusi: `scripts/setup.sh` (zero-MCP, nvcc sm_61, znver1/AVX2, ninja -j4,
taskset 0-3), **5 SKILL.md** domain (pascal-sm61-cuda, qwen38-hybrid-arch,
iso3-kv-and-mtp, memory-budget-24gb, webui-and-parity-bench), `AGENTS.md` +
roadmap 5 stage. Keputusan tetap yang dikunci user (kini menjadi konflik-dokumen
di AGENTS.md): IsoQuant quaternion untuk iso3, FP32 murni (jangan tiru jalur FP16
llama.cpp), cap VRAM 7,2 GB, mmap tanpa mlock penuh, text-only (vision diabaikan),
cpp-httplib, tanpa Node/Playwright. Saat dua `wf-coder` saling menulis folder
`qwen38-engine`, **keputusan kanonik**: pohon kode milik sesi utama adalah satu-satunya
kanonik, folder dikunci dari sesi lain, sisa file sesi penantang dibersihkan; sisa
Stage dikerjakan sebagai chat biasa **tanpa `/goal` dan tanpa subagent**. Stage 3
mengaudit tipe kuantisasi per layer dulu (Unsloth Dynamic = campuran) baru menulis
kernel; SASS gate (`IDP.4A` hadir, `HFMA2`/`HMUL2` nol) jadi gerbang wajib. Baseline
di-push dengan `.gitignore` biner + catatan gaya "coba A (alasan) → output A".

**Fase 5 — Serving, RTK, dan pergantian arah (17 Nov).** RTK dipasang dan dijadikan
**aturan WAJIB** semua shell (tercatat di AGENTS.md). Eksperimen knapsack layer +
ctx 16K di-revert (keputusan: kembali contiguous 32 GPU layer, ctx 4096) — analisa
menunjukkan keputusan tadi terlalu berisiko untuk kernel yang sudah stabil. Branch
`exp/ternary-bonsai-27b-eval`: evaluasi Ternary Bonsai 2 27B (PTQ1_0/Q2_0), 27
langkah eksperimen termasuk yang gagal beserta root cause, lalu PR #50 diperbaiki
(conflict + SVG + kedalaman analisa).

**Fase 6 — Context 128K, Colab, maraton serving (18–20 Nov).** `exp/bonsai-128k-context`
(pra-kondisi PR #51 yang sudah merged), lalu Colab notebook `tools/bonsai_mtp_train/`
(PR #53 dst.), maraton 10 jam `exp/bonsai-local-10h-opt-serving`: decode 17,4 tok/s
tanpa MTP → **26,7 tok/s** dengan r3 MTP (lossless), prefill 128K terukur, state
save/load round-trip PASS. PR-PR ini yang kemudian menjadi fondasi jalur T4
(PR #64/#65 dikerjakan OpenCode, bukan sesi Kiro).

**Fase 7 — Riset, Bonsai 2.5 vs 2, dan perburuan 100 TPS @ 100K (25–28 Nov).**
Issue #67 (riset engine ekstrem: Strata/Maya/peers) dan issue serving multi-user
à la vLLM dibuat. `exp/bonsai-2.5-vs-2` (PR #69): benchmark adil Bonsai 2.5 vs 2 —
kesimpulan jujur: **B2.5 tidak lebih cepat** (128K decode 3,3 tok/s vs 8,9 karena LM
head jatuh ke CPU; chat-tuned jadi raw prompt langsung berhenti), keunggulannya
di tuning chat/inteligensi, bukan kecepatan; tabel kecerdasan (PPL/KLD/HellaSwag/
Winogrande/ARC/MMLU) direkam. Analisa llm.asm ditinjau (tidak diadopsi). Issue
DFly2 (speculative decoding reuse sparse KV) + target user **100 TPS @ 100K ctx**;
Kiro menjawab jujur tidak realistis tanpa ubah hardware; user memaksa jalur
"matematika lowest-level di CUDA" → tiga optimisasi diterapkan dan diukur:
**lane-per-group PTQ1_0 decode GEMV (17,9 → 26,7 tok/s; MTP 29,1 → 44,1 @ 4K)**,
**LPG prefill GEMV (prefill 69,7 → 112,6 tok/s @ 4K)**, **pair-code table attention
decode (-20% @ 102K)**. `ncu`/`nvprof` gagal dijalankan dari mode Autonomous
(sandbox tanpa GPU); user menjalankan profiler manual dan hasilnya dianalisa Kiro
(prompt8–prompt12). Workflow `/goal` terakhir di-abort user 25 Nov; sesi berakhir
28 Nov 05:10 dalam keadaan sehat/idle.

### Sesi lanjutan — `sess_e36bda6e-4482-442a-81a1-4573e80befdc` (28 Nov 05:15 → 06:08)

**Analisa resume gagal.** Kiro mengaudit `~/.kiro/sessions/5b1192a7…`: `messages.jsonl`
35 MB / 14.434 baris semuanya JSON valid, turn terakhir idle normal, tidak ada lease
sisa, disk/RAM lega; uji coba `--resume-id` selama 20 detik **berhasil memuat sesi**.
Penyebab kegagalan awal user tidak tertinggal di log; dugaan: resume workspace-scoped
(harus dari `cwd` yang sama) atau timeout telemetry — kalau terjadi lagi, jalankan
dari `/home/ivan/kiroween-2026/legacy1070qi38`.

**Melanjutkan isi sesi lama — rantai penelitian penuh (perf attention decode iso3 @ 102K, 28 Nov 05:20–06:11).**
Di bawah ini log eksperimen utuh: hipotesis → aksi → angka → keputusan. Metodologi dasar:
microbenchmark standalone di luar CMake + harness timing `op_attention` (T=1, CUDA events,
checksum float + hash bit-level), dan ablation dengan mem-patch salinan `ops.cu` lalu mem-link
object pengganti ke sisa objek `libgpu_cuda` (percobaan yang gagal karena izin/mount `/tmp`
dan profiler CUDA tidak dicatat — itu keterbatasan device, bukan hasil penelitian).

- **E1 · pola load iso3 (05:21).** `bench/attn_load_bench.cu` (nvcc standalone). n=102400,
  nsl=100, useful bytes 78,1 MiB/layer: v1 loop consume-at-once **0,567 ms = 144,4 GB/s**;
  v2 16 posisi in-flight **0,502 ms = 163,3 GB/s**; v3 32 posisi 0,695 ms = 117,8 GB/s;
  v4 uint4 contiguous (plafon) **0,374 ms = 219,2 GB/s**; nsl=112 v2 175,1 GB/s.
  → Pola load kernel sendiri 144–175 GB/s, jauh di atas throughput efektif kernel
  (29 GB/s pada 2,77 ms/layer) → **bukan memori**.
- **E2 · audit SASS (05:22).** `cuobjdump -sass`: REG:168 STACK:80 (spill), 6.211 baris.
  Estimasi ~3.000 instruksi per 16 posisi → utilisasi scheduler GP104 hanya ~23 % →
  **stall-bound, bukan issue-bound**.
- **E3 · harness ablation (05:23–05:24).** `bench/attn_dec_bench.cu`; varian: base
  **2,784 ms**, noV (compute V jadi konstanta) **1,618 ms**, nomem **1,412 ms**
  (checksum base 0,037185). → Separuh waktu compute murni; memori ~1,4 ms dan
  **tidak tumpang-tindih** dengan compute.
- **E4 · sweep jumlah slice (05:25).** kVerifySlices 112/228/456 vs 100:
  2,773 / 2,818 / 3,114 ms → jumlah slice tak memengaruhi waktu → throughput-bound
  per warp pada konfigurasi lama, bukan wave scheduling.
- **E5 · sweep `__launch_bounds__` minBlocks (05:25–05:26).** minBlocks 1 → 0 spill,
  **3,928 ms**; 2 → 0 spill, 3,918 ms; base 3 (REG 168, STACK 80) → **2,78 ms**.
  → Occupancy mengalahkan biaya spill: **latency/occupancy-bound**, bukan spill-bound.
- **E6 · instrumentasi clock64 (05:27).** Per warp: K+scores 1,509M siklus, softmax
  0,182M, V 0,916M (total ~2,6M ≈ 1,6 ms @ ~1,6–1,7 GHz); 400 warp per launch, 228
  resident (19 SM × 12 warp) ≈ 1,75 gelombang → cocok dengan hasil terukur. Fase
  **K+scores = 58 %**, softmax 7 %, V 35 %.
- **E7 · pair-code table float2 LDS (05:28).** Ganti 2 LDS scalar per elemen dengan 1
  LDS.64: 2,334 ms, varian terbaik **2,190 ms (-21 %)**, checksum identik 0,037185.
  Varian minBlocks lain lebih buruk → tetap 3 blok/SM. Inilah fix yang di-commit.
- **E8 · unroll 8 posisi (05:29).** Spill naik (STACK 88, spill 148/172), 2,310–2,996 ms
  → ~156+ live values melampaui register → **dibuang**.
- **E9 · patch ke `ops.cu` + verifikasi penuh (05:30–05:57).** ninja -j4 OK;
  `attn_long_test 131072` semua PASS vs FP64 (max_abs ≤ 4e-6, needle identik);
  `verify_t17_test` 32/32; SASS gate HFMA2=0 (IDP=0 wajar — dp4a bukan jalur kernel ini).
  Timing + **hash bit-level identik** `9e2d2cc1…` base vs pair pada nsl=100: n=102400
  **2,346 vs 3,155 ms**; n=4097 0,218 vs 0,255 ms; n=100 0,071 vs 0,072 ms. Decode step
  82 ms hemat ~10 ms → **12,17 → 13,5 tok/s @ 102K**. **Commit `f6590f0`** (ops.cu +
  `bench/attn_load_bench.cu` + `bench/attn_dec_bench.cu`) — eksekusi pesan "commit dulu saja".
- **E10 · V prefetch dalam warp depth 1/2/3 (06:01).** 2,294 / 2,314 / ~2,3 ms; angka
  spill identik di semua depth → compiler mendedup pola; prefetch tak menolong →
  hipotesis latensi melemah.
- **E11 · hitung instruksi SASS dinamis (06:01–06:02).** ~3.500 instruksi/warp-iterasi
  (~250 di antaranya SHFL) vs siklus terukur → **IPC ≈ 0,26** → masalah bukan jumlah
  instruksi melainkan IPC rendah.
- **E12 · split 3 kepala/warp, minBlocks 4/5 (06:02).** 96 register, spill turun
  (STACK 64), **2,942 / 2,804 ms** — bit-identical tapi kalah dari 2,2 ms 6-kepala →
  duplikasi decode menelan untung occupancy. **Dibuang.**
- **E13 · sweep nsl dengan patch (06:04).** nsl 20 (80 warp) **5,511 ms**; 40 (160 warp)
  2,664 ms; **56 (224 warp) 1,926 ms**; 76 (304 warp) 2,996 ms (tail gelombang kedua
  timpang). Model "SM tersibuk" konsisten (W ≈ 35–37) → **latency-bound**; wave
  quantization faktor kedua. Catatan jujur: bits hash berubah lintas nsl
  (`237df3…`/`2395ef…`/`05c1dd…`/`ed9e10…`) — urutan akumulasi ikut berubah; bits
  nsl=56 `05c1dd08…` konsisten dengan build pair nsl=56.
- **E14 · split-heads + nsl dicocokkan (06:06).** minBlocks 4 nsl 36/72 → 2,749/2,751 ms;
  mb5 nsl44 → 2,406 ms; mb6 nsl56 → 2,760 ms. Semua kalah dari baseline 1,926 ms →
  **split-heads final dibuang**; perubahan murah `kVerifySlices`/`kDecSlices = 56`
  (buffer alokasi tetap 100; nsl di-clamp terhadap ctx; rencana portabilitas: derive
  dari jumlah SM runtime, 3×SM dibulatkan ke kelipatan 4 — di T4 nilainya beda) —
  **belum di-commit**, perlu regresi lintas n/ctx.
- **E15 · set ablation di nsl=56 (06:07).** base **2,101 ms** (noise 1,9–2,1); nomem
  **1,077 ms**; varian V/shuffle/reduce berturut-turut 1,068 / 0,988 / 0,702 / 0,736 ms →
  compute-only ≈ 1,08 ms (V ≈ 0,38 ms, K-dot FMA ≈ 0,34 ms, shuffle nyaris nol biaya).
- **E16 · `prefetch.global.L2` PD=1..4 tile (06:09).** 2,262–2,270 ms — **nol efek** →
  latensi DRAM/L2 dibantah.
- **E17 · `prefetch.global.L1` PD=1/2 (06:10).** 2,243 / 2,269 ms — **nol efek**; grep
  SASS tak menemukan opcode prefetch di sm_61 (mungkin disembunyikan di mnemonic
  cache-control seperti CCTL) → instruksinya mungkin tidak diekmit Pascal.
- **E18 · ablation uniform-index (06:11).** Load asli dipertahankan, indeks tabel LDS
  dipaksa lane-uniform: **1,878 ms** vs base ~2,1 ms → **bank conflict ≈ 0,22 ms**,
  paparan memori tersisa ≈ 0,8 ms. Keputusan: keduanya nyata, prefetch tidak
  menyelesaikannya → dugaan berikutnya: throughput L1 / biaya sektor LDG (load kecil
  per lane bisa melintang dua sektor 32B) dan branching berat `BRK/SSY` di loop.
  Rencana terakhir yang tercatat: ekstrak body loop jadi lambda dengan cabang
  full-unroll compile-time untuk memisahkan efek unroll vs runtime-loop — **sesi
  terputus di sini, belum dieksekusi**.
- **Alternatif yang ditimbang dan ditolak tanpa eksekusi:** decode ALU/polinomial
  pengganti LDS, PRMT byte-select (biaya per-elemen setara), replikasi tabel lintas
  bank (matematika bank tak menyelesaikan), prefetch ke register (96 register untuk 32
  posisi — meledakkan budget), staging shared memory tanpa cp.async/TMA (Pascal tidak
  punya jalur satu-warp), dan **producer–consumer warp specialization** (1 warp loader +
  4 warp compute, ring buffer smem + named barrier; conflict-free terverifikasi untuk
  pola baca 100 B/row) — kandidat serius, belum dieksekusi.

**Status akhir rantai:** fix ter-commit hanya `f6590f0` (pair-table, E7/E9, -20 %,
bit-identical pada nsl=100). Temuan `nsl=56` (E13, 1,93 ms, -30 % lagi) terukur tapi
belum di-commit — hash berubah lintas nsl karena urutan akumulasi (masih dalam
kebijakan toleransi; greedy ids wajib dicek ulang). Hipotesis terbuka: throughput/
biaya sektor L1 pada LDG kecil per lane, bank conflict 0,22 ms, dan load×compute yang
tidak overlap (kandidat besar: producer–consumer warp specialization atau pipeline
dalam warp — dua-duanya belum dieksekusi). Langkah instrumen berikutnya: ablation
nomem yang menjaga struktur alamat (anti-broadcast), hitung instruksi dinamis per
fase, dan uji full-unroll loop body.

---

*Kisah di atas disusun ulang pada 28 Nov 2026 dengan membaca transkrip kedua sesi
(`~/.kiro/sessions/5b1192a7a7c672ca/…`) — tidak ada klaim yang ditulis dari ingatan.*
