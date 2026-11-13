# Design — SPEC-003: Memory Manager

> Spec: SPEC-003 | Status: **Skeleton**

---

> 📋 **Status: Skeleton — Belum diisi sepenuhnya**
> Isi setelah SPEC-002 selesai. Lihat `roles/memory-manager.md` untuk referensi
> memory budget dan formula kalkulasi.

---

## Architecture Overview

[TODO: Diagram setelah SPEC-002 baseline tersedia]

---

## Open Questions

- [ ] Berapa actual VRAM overhead CUDA runtime pada hardware ini? (target: ukur di SPEC-002)
- [ ] Apakah perlu custom memory allocator atau cukup lewat llama.cpp flags?
- [ ] Bagaimana cara graceful degrade saat mendekati VRAM limit?
