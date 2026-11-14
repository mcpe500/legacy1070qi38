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
| Planning agent (plan → handoff ke execution) | ✅ Ya | Requirements gathering + rencana implementasi sebelum menulis file |
| Sub-agent `context-gatherer` | ✅ Ya | Riset teknis hardware/model, dan menulis file spec secara paralel |
| Built-in tools (read/list/search file) | ✅ Ya | Membaca dan memverifikasi isi workspace |
| Live steering | ✅ Ya | User menyisipkan instruksi saat sub-agent sedang berjalan |
| Spec-driven development (requirements → design → tasks) | ✅ Ya (struktur) | Format `SPEC-NNN-*/` mengikuti pola requirements/design/tasks |
| Hooks | ❌ Belum | — |
| Steering files (`.kiro/steering/`) | ❌ Belum | — |
| Skills (`.kiro/skills/`) | ❌ Belum | — |
| MCP servers | ❌ Belum | — |
| Property-based testing (PBT) | ❌ Belum | Belum ada kode yang diuji |
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
- **Riset**: analisis kelayakan Qwen3-8B/27B pada GTX 1070 Ti (Pascal SM 6.1) + 16GB RAM —
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

## Belum Digunakan (rencana)

- **Hooks**: kandidat — auto-update `HANDOFF-LOG.md` setelah task selesai, atau
  pengingat memeriksa daftar approach terlarang saat file spec disimpan.
- **Steering files**: kandidat — memindahkan hardware constraints dari `MASTER.md`
  ke `.kiro/steering/` agar otomatis dimuat di setiap sesi.
- **PBT**: kandidat saat ada kode, mis. properti "estimasi VRAM selalu ≥ ukuran weights".
- **MCP kustom**: sekarang opsional. Jalur utama adalah CLI `tools/legacy.py` (RFC-002,
  status Proposed, belum dibuat). Pembungkus MCP baru dibuat jika diperlukan. Lihat
  `spec/specs/RFC-002-cli-tooling.md` dan diskusi di GitHub issue #1.
- **Powers / Skills**: skills direncanakan (`vram-budget`, `run-benchmark`, `update-handoff`);
  powers belum ada kebutuhan konkret.

Saat salah satu dipakai, pindahkan ke bagian "Detail Penggunaan" beserta buktinya.

---

*Dokumen ini ditulis ulang agar akurat. Versi sebelumnya memuat klaim tool yang tidak
benar-benar dipakai.*
