# Tasks — Single-File Executable

> Spec: SPEC-008 | Status: **Not Started**
> Dependencies: RFC-004 harus **Accepted** sebelum Task 1 dimulai
> Blocks: SPEC-009 (rilis artefak), PR #52 (eksperimen bahasa), PR #53 (forward Bonsai sebagai prasyarat
> model cepat di dalam executable)

---

## Task List

- [ ] **Task 1: Payload trailer + tester**
  - Implementation: `src/engine/payload.{hpp,cpp}` — struct 128 B little-endian, `trailer_read`,
    `payload_verify` (SHA-256 header-only apa adanya di `third_party/`); `static_assert` ukuran
  - Test: `tests/payload_test.cpp` — tulis+baca trailer pada berkas temp; hash acak cocok; 1 bit dibalik
    → `payload_verify` false; magic/versi asing → tolak dengan pesan
  - Done when: test lulus di `ninja -j4` tanpa GPU, `ninja -j4` target test hijau

- [ ] **Task 2: mmap diri-sendiri + integrasi gguf_index**
  - Implementation: konstruktor `gguf_index(fd, offset, length)`; `open_self()` (`/proc/self/exe`);
    pembulatan halaman (offset align 4096 — syarat `mmap`), sisa < 1 halaman via pread; nol salinan
  - Test: `tests/payload_mmap_test.cpp` — bangun GGUF kecil (9B arsitektur, `tools/gguf inspect`),
    embed sebagai payload di berkas uji, muat, bandingkan `gguf_index` hasil payload vs berkas biasa:
    jumlah tensor, offset relatif, dan checksum tiap tensor identik
  - Done when: dua pembacaan menghasilkan peta tensor bit-identik

- [ ] **Task 3: CLI + server default + `--version`/`--list-payload`**
  - Implementation: `main.cpp` — resolusi `-m` > payload > usage(2); `--version` dengan digest;
    `--list-payload` tanpa arena
  - Test: `curl -N` SSE (protocol B webui-and-parity-bench): `./legacy1070qi38` start, GET `/` 200,
    POST `/v1/chat/completions` identik dengan mode `-m`
  - Done when: ketiga subcommand lulus; usage tanpa model = kode 2

- [ ] **Task 4: `tools/embed_payload.py` + target CMake + `.gitignore`**
  - Implementation: skrip stdlib-only (validasi header GGUF, pad 4096, SHA-256, trailer, concat,
    `chmod +x`, `--verify` dari berkas jadi); `cmake --build --target embed_payload`;
    pola baru `.gitignore` (binary fat + `*.bin` payload) — **pengaman dini**, artefak tetap Release
  - Test: `embed_payload.py --verify` cocok dengan `sha256sum` payload pembangun; sentuh file model →
    `ninja` no-op untuk engine (FR-06 "tanpa compile ulang"); tanpa model → galat jelas, kode 2
  - Done when: build fat jalan 2× (utama dan manual `cat` FR-06b) dengan hasil `--verify` sama

- [ ] **Task 5: Ukur dan laporkan (bukan mengklaim)**
  - Implementation: `bench/payload_vs_file.sh` — mode payload vs mode `-m`, model Bonsai PTQ1_0:
    waktu start, pp/tg (protocol C, 512 prompt/256 decode), RAM puncak, VRAM puncak, SwapUsed;
    jumlah byte payload diperiksa ulang dengan stat
  - Test: minimal 3 run berurutan (engine dan llama-server **tidak pernah** bersamaan, rule #5);
    hasil → `bench/results/single-file/` + laporan di `spec/research/reports/`
  - Done when: tabel lengkap; tidak ada perbedaan tok/s melebihi 5% tanpa penjelasan

- [ ] **Task 6: Eksperimen bahasa (C/C++/Goose) di level kernel — decision gate**
  - Implementation: satu kernel ternary/AVX2 dot + CLI pembaca payload diimplementasikan di C, dan di
    Goose (launcher-level: baca self-exe + dot kernel); C++/`mt_dot` sebagai referensi. Semua
    `-march=znver1`, `taskset -c 0,1,2,3`. Goose diuji sebagai bahasa pemaket, **bukan** mesin CUDA
  - Test: parity — greedy token identik, cosine ≥ 0.998 (protocol A) pada 3 prompt; throughput GB/s
    per bahasa; tabel + SVG
  - Done when: keputusan tertulis: kalau C/Goose tidak >10% mengalahkan C++ → **tetap C++**, hasil
    tetap dicatat di CLAIMS. Bahasa tidak diperdebatkan ulang setelah gate ini

- [ ] **Task 7: zstd vs mentah (opsional, jangan mengembang)**
  - Implementation: bila Task 1-5 selesai dulu, ukur `zstd -3 -T4` pada payload PTQ1_0: rasio dan waktu
    `zstd -d -T4` di 4 core; keputusan masuk RFC-004 (default tetap **mentah**)
  - Test: rasio < 1.25x → tutup dengan "tidak worth it"; ≥ 1.25x → rancang payload terkompresi opsi flag
  - Done when: satu baris keputusan di RFC-004 dengan angka; tidak ada implementasi kompresi kecuali
    gate lolos

- [ ] **Task 8: Tutup dokumentasi**
  - Implementation: status spec → **Implemented** (atau sesuainya), HANDOFF-LOG append, `AFK_STATUS.md`
    di baris atas, README "Status" + "Mulai cepat" (perintah `./legacy1070qi38`), SVG permanen (SHA
    commit) di PR
  - Test: `python3 tools/legacy.py spec validate` lulus; `python3 tools/test_legacy.py` tidak regresi
  - Done when: semua success criteria di `requirements.md` terceklis, PR siap direview manusia

---

## Notes

- **Jebakan yang sudah diketahui**: jangan pakai `cmake/embed.cmake` (hex C-array) untuk payload —
  ~5× pembengkakan teks → ±28 GB source untuk 5.54 GiB (C-72). Append+mmap sudah diputuskan di RFC-004
- Model yang layak di-embed dibatasi kelas **Bonsai** (muat RAM/VRAM): PTQ1_0 5,946,648,928 B,
  Q2_0 7,626,008,928 B (C-69). Q4_K_M 16,564,259,744 B = 15.43 GiB tetap lewat `-m` (tanpa payload
  yang mungkin dijamin pada 16 GB RAM; lihat C-29 overhead)
- `spec validate` menolak penomoran FR/NFR yang melompat — tambahkan di akhir, jangan di tengah
- Perubahan di `qwen38-engine` mengikuti aturan lokalnya: `ninja -j4`, SASS gate, taskset, tanpa
  MCP/daemon, dan SEMUA perintah shell lewat `rtk` (`rtk err ninja -C build -j4`, `rtk run`, ...)
- Angka tok/s Bonsai yang beredar (19.64 dst.) hidup di `bench/results/bonsai/`; baris CLAIMS-nya
  belum ada — pekerjaan dokumen terpisah, **jangan** dikutip sebagai klaim di PR ini