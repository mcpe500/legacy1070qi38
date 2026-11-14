# RFC-002: Custom MCP Servers untuk legacy1070qi38

> Status: **Proposed** (untuk diskusi, belum diputuskan)
> Penulis: Kiro (planning), atas permintaan user
> Diskusi: umbrella #1; per server: A #2, B #3, C #4, D #5 (github.com/mcpe500/legacy1070qi38/issues)

---

## Summary

Membuat beberapa MCP server kustom (lokal, transport stdio) agar agent Kiro bisa **mengukur hardware,
menjalankan llama.cpp, menghitung memory budget, dan merawat dokumen spec/handoff** lewat tool
terstruktur, bukan lewat shell bebas. Tujuannya: pengukuran reproducible, aman, dan tercatat.

> ⚠️ ASSUMPTION: Detail konfigurasi MCP di Kiro (lokasi `mcp.json`, timeout tool call, dukungan
> progress untuk tool yang lama) belum diverifikasi. Lihat Open Questions.

---

## Motivation

- Banyak open question (OQ-001 s/d OQ-008) hanya bisa dijawab dengan **pengukuran**, bukan analisis.
- Agent yang menjalankan `nvidia-smi` / `llama-cli` lewat bash bebas: hasilnya tidak terstruktur,
  mudah salah parse, dan tidak ada pengaman (timeout, allowlist, satu-inference-sekali-jalan).
- Dokumen spec rawan inkonsisten. Contoh nyata di sesi bootstrap: penomoran NFR melompat ke NFR-04.
- Untuk submission: MCP kustom adalah bukti penggunaan Kiro yang spesifik pada proyek ini.

---

## Daftar Kandidat

### A. `legacy-budget` — kalkulator memory budget (tanpa hardware)

| Tool | Fungsi |
|---|---|
| `estimate_vram` | Input: layers, kv_heads, head_dim, bits-per-weight, ctx, tipe KV. Output: rincian weights/KV/overhead + risiko 🟢🟡🔴 |
| `suggest_ngl` | Input: VRAM tersedia + model. Output: jumlah layer GPU maksimum |
| `compare_estimate_vs_measured` | Bandingkan estimasi vs hasil ukur, untuk kalibrasi rumus |

Memformalkan rumus di `spec/AGENTS/roles/memory-manager.md`. Fungsi murni, cocok untuk **PBT**
(mis. total VRAM tidak pernah lebih kecil dari ukuran weights; KV cache monoton naik terhadap ctx).
Bisa dibuat dan diuji **sekarang**, tanpa GPU.

### B. `legacy-hw` — monitor hardware (read-only)

| Tool | Fungsi |
|---|---|
| `gpu_status` | VRAM used/free, suhu, power, clock, throttle reason (dari `nvidia-smi`) |
| `sample_gpu` | Sampling selama N detik, kembalikan peak VRAM, rata-rata util, suhu maks |
| `ram_status` | RAM bebas, swap terpakai, RSS proses llama |
| `check_headroom` | Input: kebutuhan MB. Output: go / no-go beserta alasan |

Menjawab OQ-005 (RAM 16 GB cukup untuk 27B tanpa swap?) dan OQ-008 (ctx maksimum sebelum OOM).
Read-only, risiko rendah.

### C. `legacy-llama` — runner dan benchmark (mengeksekusi proses)

| Tool | Fungsi |
|---|---|
| `inspect_gguf` | Baca metadata GGUF (layer, KV heads, hidden dim) **tanpa memuat model** → OQ-001..003 |
| `build_llamacpp` | Build SM 61, catat commit hash dan versi CUDA |
| `run_inference` | Jalankan satu prompt dengan parameter terbatas, parse tok/s |
| `run_benchmark` | 3x run + sampling GPU bersamaan, tulis hasil ke `benchmarks/results/` sesuai format `benchmarker.md` |
| `compare_results` | Bandingkan dua hasil, tandai regresi |

Pengaman: allowlist binary dan path model, timeout, **lock** agar hanya satu inference berjalan
(benchmark paralel merusak angka), tidak ada argumen shell bebas.

### D. `legacy-spec` — perawat dokumen spec dan handoff

| Tool | Fungsi |
|---|---|
| `new_spec` | Scaffold `SPEC-NNN-nama/` dari `_TEMPLATE-feature-spec/` dengan nomor berikutnya |
| `append_handoff` | Tambah entri ke `HANDOFF-LOG.md` (menjaga sifat append-only) |
| `new_session` | Buat `SESSION-NNN` dari template + update index |
| `list_open_questions` / `resolve_question` | Kelola OQ beserta evidence |
| `validate_specs` | Cek: penomoran FR/NFR berurutan, label ASSUMPTION, tidak ada approach terlarang tanpa disclaimer, link antar spec valid |

Mencegah kelas bug yang sudah terjadi di sesi bootstrap.

---

## Keputusan Desain yang Diusulkan (perlu didiskusikan)

1. **Beberapa server kecil, bukan satu besar.** Dipisah menurut hak akses: A dan B tidak punya efek
   samping, C mengeksekusi proses, D menulis file docs. Alasan lain: tiap tool menambah konteks agent.
2. **Python + SDK MCP resmi**, transport **stdio**, didaftarkan di `.kiro/settings/mcp.json`.
3. **Output terstruktur (JSON)** plus ringkasan Markdown.
4. **Tidak ada shell bebas**: parameter divalidasi dan dibatasi.
5. **Urutan pengerjaan:** A → (C: `inspect_gguf`) → B → sisa C → D.
6. **Kode** disimpan di `tools/mcp/` (belum ada; repo saat ini hanya dokumen).

---

## Open Questions

- [ ] Berapa timeout tool call di Kiro, dan apakah ada progress untuk tool lama? Memuat 27B bisa menit-an.
      Jika timeout pendek: pakai pola `start_job` + `get_job`.
- [ ] Lebih baik `run_inference` memanggil `llama-cli` per panggilan, atau mengelola `llama-server`
      yang tetap hidup (menghindari reload model 27B tiap kali)?
- [ ] Satu server atau empat? (usulan: empat)
- [ ] Apakah `validate_specs` lebih cocok sebagai hook Kiro daripada tool MCP, atau keduanya?
- [ ] Bagaimana mengunci GPU bila desktop environment ikut memakai VRAM?
- [ ] Apakah MCP kustom ini layak dirilis sebagai bagian proyek open-source (Apache 2.0)?

---

## Rejected / Tidak Diperlukan

- MCP filesystem, fetch, dan web search: Kiro sudah punya tool bawaan setara.
- MCP yang mengizinkan eksekusi shell bebas: melemahkan reproducibility dan keamanan.

---

*Terkait: `spec/specs/SPEC-006-benchmark-suite/`, `spec/AGENTS/roles/memory-manager.md`,
`spec/AGENTS/roles/benchmarker.md`, `spec/AGENTS/KIRO-USAGE.md`.*
