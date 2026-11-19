# Design — Single-File Executable

> Spec: SPEC-008 | Status: **Draft**
> Referensi: RFC-004 (keputusan format), `qwen38-engine/AGENTS.md`, skill `memory-budget-24gb`
> Ilustrasi: [`assets/spec-008-executable-layout.svg`](../assets/spec-008-executable-layout.svg)

---

## Architecture Overview

Satu berkas, tiga bagian. Tidak ada kompilasi bobot — payload **disambung** (append), bukan dikompilasi;
digest melindungi integritas, versi format melindungi kompatibilitas.

```
legacy1070qi38            (satu berkas, ±5.6 GiB untuk Bonsai PTQ1_0, C-69)
│
├─[ ELF   .text/.rodata ] engine C++/CUDA + WebUI + cpp-httplib             ±10 MB  ← mmap biasa oleh loader
├─[ PAYLOAD bobot       ] GGUF/Bonsai, WAJIB dimulai pada kelipatan 4096 B   ±5.5 GiB ← mmap zero-copy oleh engine
└─[ TRAILER 128 B       ] magic + versi + offset + panjang + SHA-256(json)   128 B   ← dibaca dengan pread dari ujung

startup (tanpa -m):
  1. readlink("/proc/self/exe")            → fd berkas sendiri
  2. fstat(fd) → size; pread(fd, size-128) → verifikasi magic + versi format
  3. sha256(payload, offset, panjang)      → bandingkan dengan trailer; gagal → kode 2
  4. mmap(nullptr, panjang, PROT_READ, MAP_PRIVATE, fd, offset)
  5. arena + residency streaming seperti biasa (base pointer = hasil mmap, bukan halaman file)
```

Kunci desain: **payload GGUF adalah byte-for-byte berkas GGUF yang sama**. Yang berubah hanya
*dari mana* bytenya datang (mmap diri-sendiri vs mmap berkas sebelah). Semua pengukuran tok/s lama
tetap berlaku — karena memang jalan datanya sama.

---

## Components

### 1. Payload trailer (`src/engine/payload.{hpp,cpp}`)

- **Tanggung jawab:** mendefinisikan, menulis, membaca, dan memverifikasi trailer; satu-satunya modul baru
  yang wajib untuk SPEC-008
- **Interface:**
  - `struct PayloadInfo { uint32_t magic; uint32_t version; uint64_t offset; uint64_t length;
    unsigned char sha256[32]; char model_name[64]; char model_version[32]; }` — **128 B tetap**
    (little-endian, `static_assert`), sehingga pencarian trailer tidak butuh pemindaian
  - `PayloadInfo trailer_read(int fd)` — `pread` dari `size − 128`; menolak magic/versi asing
  - `bool payload_verify(int fd, const PayloadInfo&)` — SHA-256 berjalan (`SHA256` sendiri, lib standalone,
    tanpa dependensi eksternal; engine hari ini sudah header-only di `third_party/`)
- **Dependencies:** libc (`pread`, `fstat`, `mmap`), SHA-256
- **Pascal considerations:** tidak ada isi CUDA; trailer murni host agar bisa diuji tanpa GPU

### 2. Self-path resolution

- **Tanggung jawab:** menemukan berkas executable yang sedang berjalan
- **Interface:** `int open_self()` → fd, atau `-1` + galat
- **Detail:** Linux: `readlink("/proc/self/exe")`. Bila hasilnya bukan berkas reguler (mis. interpreter
  eksperimen), jatuh ke `argv[0]` + `realpath`. Pengujian memakai berkas uji kecil, bukan binary fat 5.6 GB
- **Non-goal:** portabilitas macOS/Windows (di luar scope, `Out of Scope` requirements)

### 3. Integrasi mmap (`src/packer/gguf_index.{hpp,cpp}` — perubahan kecil)

- **Tanggung jawab:** membaca GGUF dari payload, bukan dari berkas terpisah
- **Caranya:** konstruktor baru `gguf_index(int fd, uint64_t payload_offset, uint64_t payload_length)`.
  Catatan penting implementasi: `mmap` menuntut `offset` kelipatan halaman (4096) — **trailer menyatakan
  offset yang sudah page-aligned**, dan `data_` base = hasil mmap (bukan awal file). Panjang payload
  membulat ke bawah ke halaman terdekat; sisa < 1 halaman dibaca dengan `pread` ke akhir arena — sekecil
  4096 B, dibayar sekali saat start
- **Invariant yang harus dipertahankan:** tensor tetap pointer ke mapping read-only; **nol salinan**
  (`gguf_index.hpp` hari ini: "Tensor bytes stay in the page cache; nothing is copied here")
- **Alternatif yang ditolak di sini:** memetakan seluruh file termasuk kode ELF (menyia-nyiakan
  page-table entries dan membingungkan gc-sections)

### 4. Build embedding (`tools/embed_payload.py` + target CMake)

- **Tanggung jawab:** menyusun `engine.bin + payload.bin + trailer` → executable fat
- **Alur:**
  1. engine sudah ter-build (`ninja -j4` biasa, `AGENTS.md` rule #3)
  2. `cmake --build build --target embed_payload` menjalankan
     `python3 tools/embed_payload.py --engine build/qwen38-engine --model MODEL.gguf --out legacy1070qi38`
  3. skrip memvalidasi GGUF header, membulatkan awal payload ke 4096 B (pad dengan nol), menghitung
     SHA-256 payload, menulis trailer, menyambung ketiganya, `chmod +x`
  4. `--verify` dari berkas jadi: parse trailer, verifikasi digest, cetak ukuran total
- **Hanya library standar** (konsisten SPEC-007), tanpa writing ke luar direktori build kecuali `--out`
- **Batasan tegas:** ganti model → jalankan ulang `embed_payload` saja; **tidak boleh** memicu
  re-kompilasi engine (uji: sentuh file model, `ninja` harus no-op untuk target engine)

### 5. CLI & server default (`src/engine/main.cpp`)

- **Tanggung jawab:** mode payload menjadi default; `--server` tetap eksplisit opsional
- **Urutan resolusi model:** `-m` (menang) → payload internal → galat usage, kode 2
- **`--version` output:** `engine <commit> sm_61 znver1 | payload <name> v<ver> <bytes> B sha256:<hex>`
- **`--list-payload`:** info trailer + verifikasi digest **tanpa** memuat arena inferensi

### 6. Model matrix dan jatuhan-dan-gagal

| Kondisi | Perilaku ditargetkan |
|---|---|
| Payload valid, tanpa `-m` | Server langsung (default FR-01) |
| Payload valid + `-m lain.gguf` | Mode `-m` (override), payload diabaikan, tercatat di log start |
| Payload rusak 1 byte | `payload: sha256 mismatch` + offset, kode 2, **tidak** start |
| Trailer magic asing (mis. build format baru) | `payload: unknown trailer version` + yang didukung, kode 2 |
| Offset tidak page-aligned | Ditolak saat **build** (`embed_payload.py`), bukan saat runtime |
| Payload + `-m` dua-duanya absen | Pesan usage, kode 2 (unchanged NFR-05) |
| Payload ter-set besar → `mlockall` gagal | Abaikan senyap; lanjut mmap tanpa mlock (limit 1.94 GiB, C-70) |

### 7. Memori dan residensi

- Page cache di-share dengan berkas; menghapus berkas setelah start **tidak** melepas halaman terpakai
- Streaming residency/`pin_stream` yang sudah ada tidak berubah (payload sama sekali tidak disalin)
- `mlock` sebagian hanya untuk LM head + MTP head (total ≤ 1.8 GiB, di bawah batas C-70); swappa
  dipantau, bukan dihalangi
- Karena Bonsai PTQ1_0 = 5.54 GiB (C-69) muat RAM dan **hampir** muat VRAM, mode payload memakai
  konfigurasi offload terukur untuk Bonsai; Q4_K_M 15.43 GiB **tetap** lewat `-m` eksternal
  (tidak bisa dijamin hidup sebagai payload di 16 GB RAM — lihat RFC-001 kalibrasi overhead 1.6 GiB, C-29)

---

## Configuration

Tidak ada berkas konfigurasi baru. Semua status lewat CLI, sesuai SPEC-007/SPEC-005:

```bash
# build engine, lalu bentulkan executable fat (FR-06a)
cmake --build build --target embed_payload

# cara manual (FR-06b) — trailer disiapkan skrip, dikonkatenasi
python3 tools/embed_payload.py --engine build/qwen38-engine --model MODEL.gguf --out legacy1070qi38

# verifikasi tanpa inferensi
./legacy1070qi38 --list-payload
```

---

## Consequences

1. **Bukan optimasi.** Semua angka tok/s yang berlaku untuk payload sama dengan mode `-m`; kalau ada
   yang mengklaim sebaliknya, itu regresi atau kesalahan ukur (C-71 adalah batas atasnya)
2. **Binary multi-GiB adalah artefak rilis**, bukan objek git: pola `.gitignore` baru + GitHub Release
   (NFR-07); alat ukur (`legacy.py`) tidak pernah membaca binary itu
3. **Parity WAJIB** melindungi klaim: identitas byte payload menjadikan ini murni regresi-test, bukan
   fitur baru de-facto — kalau parity gagal, yang salah adalah pembacaan payload, bukan modelnya