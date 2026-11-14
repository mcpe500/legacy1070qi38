# Tasks — SPEC-005: CPU-GPU Offload untuk Qwen3.8-27B

> Spec: SPEC-005 | Status: **Blocked** (menunggu SPEC-002 Task 5)
> Dependencies: SPEC-002
> Blocks: tidak ada

---

## Task List

- [ ] **Task 1: Ukur dasar hardware**
  - Implementation: bandwidth RAM, single atau dual channel, CPU (lihat SPEC-002 Task 1)
  - Done when: angka terukur menggantikan asumsi 20/40 GB/s

- [ ] **Task 2: Unduh quant kandidat**
  - Implementation: Q4_K_M, IQ3_M, IQ3_XXS dari repo GGUF; catat sumber dan hash
  - Done when: file tersedia dan terbaca

- [ ] **Task 3: Pengukuran baseline Q4_K_M**
  - Implementation: auto-fit dan `-ngl` manual; catat TPS, RAM, VRAM, layer di GPU
  - Done when: tabel terisi

- [ ] **Task 4: Sapuan quant dan jumlah layer**
  - Implementation: ulangi Task 3 untuk quant lain dan beberapa nilai `-ngl`
  - Done when: ditemukan konfigurasi dengan TPS tertinggi tanpa swap

- [ ] **Task 5: Strategi B (muat penuh)**
  - Implementation: quant ~7.7 GiB (misalnya Q3-DOWN-XS); catat apakah muat, TPS, kualitas
  - Done when: hasil, termasuk jika gagal, tercatat

- [ ] **Task 6: Strategi C (MTP) dan D (Vulkan)**
  - Implementation: sesuai model card dan build Vulkan
  - Done when: ada angka atau alasan gagal

- [ ] **Task 7: Ukur kualitas konfigurasi terpilih**
  - Implementation: perplexity atau KL divergence terhadap quant lebih tinggi
  - Done when: kualitas dan kecepatan dapat dibandingkan

- [ ] **Task N: Update status dan handoff**
