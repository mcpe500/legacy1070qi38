# RFC-003: Memicu Kiro CLI dari GitHub Issues

> Status: **Implemented (mode tanpa shell)**. Mode shell menunggu `bubblewrap` terpasang dan diuji.
> Kode: `tools/agent/` (lihat `tools/agent/README.md`)

## Summary

Skrip Python di laptop memantau issue dan komentar repo. Bila pengirimnya salah satu dari tiga ID yang diizinkan
dan teksnya diawali `/kiro`, skrip menjalankan `kiro-cli chat --v3 --no-interactive` memakai login Kiro yang sudah ada.
Perubahan file dikirim sebagai PR, tidak pernah langsung ke `main`.

## Keputusan

| Hal | Keputusan | Alasan |
|---|---|---|
| Tempat jalan | Laptop, bukan GitHub Actions | Memakai login yang ada (tanpa API key, yang butuh paket Kiro tertentu); GPU tersedia; tidak ada kode dari luar yang berjalan di mesin |
| Pemicu | Polling tiap 60 detik | Sederhana; tidak butuh port terbuka |
| Identitas | ID numerik: 68177813 (mcpe500), 73787745 (ibrahimmuh26), 113233053 (nicostap) | Username bisa berubah atau didaftarkan ulang |
| Perintah | Awalan `/kiro` | Komentar biasa tidak memakai kuota |
| Batas | 12 run per jam | Keputusan pemilik repo |
| PR dari fork | Diabaikan | Keputusan pemilik repo |
| Hasil | Balasan + PR dari branch `kiro/...` | Merge tetap oleh manusia |

## Model ancaman

| Ancaman | Penanganan | Diuji? |
|---|---|---|
| Orang luar menulis `/kiro` | Pengirim difilter dengan ID; tidak dibalas sama sekali | Unit test |
| Orang berusername sama tapi ID beda | Dicocokkan lewat ID | Unit test |
| Akun bot dengan ID terdaftar | Tipe harus `User` | Unit test |
| Orang luar mengedit komentar milik anggota | Komentar yang sudah diedit diabaikan | Unit test |
| Agent membalas dirinya sendiri (loop) | Penanda tersembunyi di semua balasan agen, dilewati | Unit test + uji nyata (dua komentar agen dilewati) |
| Teks orang luar di thread dipakai sebagai perintah | Hanya isi event pengirim yang sah yang menjadi instruksi. Isi issue milik orang luar tidak diberikan ke agent | Dirancang; belum diuji dengan akun luar |
| Agent membaca token atau kunci | Aturan `deny fs_read` untuk `~/.ssh`, `~/.config/gh`, folder status; tanpa shell | Uji nyata: pembacaan token `gh` ditolak |
| Agent mengubah skripnya sendiri | `deny fs_write` untuk `tools/agent`, `.github`, `.kiro`, `.git` + pemeriksaan ulang sebelum commit | Uji nyata: perubahan `config.json` ditolak |
| Agent menulis di luar worktree | Izin menulis hanya `./**`; selain itu "ask", yang ditolak pada mode non-interaktif | Uji nyata: `/tmp` tidak dibuat |
| Push ke `main` | Skrip hanya push ke branch `kiro/...`; branch protection di `main` | Branch protection: lihat catatan |
| Menyalahgunakan kuota | 12 run/jam, timeout 30 menit | Unit test untuk pembatas |
| Mention massal lewat keluaran agent | `@` dinetralkan di semua komentar dan PR | Unit test |

## Yang belum terjamin

- **Prompt injection lewat web**: agent boleh `web_fetch`. Halaman web berbahaya bisa membujuknya menulis isi tertentu ke file atau balasan. Mitigasi: tanpa shell, PR perlu review, repo ini publik.
- **Mode shell** belum diaktifkan. Tanpa sandbox ia setara akses penuh ke akun Anda.
- **GPU di dalam sandbox** belum diuji.
- **Konsistensi isi balasan**: agent bisa keliru. Ringkasannya harus dibaca, bukan dipercaya begitu saja.
- **Tidak ada pemantauan biaya**: setiap run memakai kredit Kiro Anda.

## Open Questions

- [ ] Apakah `--sandbox` mengizinkan akses `/dev/nvidia*`?
- [ ] Apakah perlu `sandbox.json` global agar `--sandbox` aktif?
- [ ] Perlu layanan systemd agar skrip hidup kembali setelah reboot?
- [ ] Apakah agent perlu membaca thread issue (komentar anggota lain yang diizinkan) sebagai konteks?
