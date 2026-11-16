# Laporan setup.sh

- waktu (UTC): 2026-11-15T15:35:23Z
- workspace: /home/ivan/kiroween-2026/legacy1070qi38/qwen38-engine

## 1. Sistem dan kunci perangkat keras
  [info] Linux 5.15.0-203-generic x86_64
  [info] CPU: AMD Ryzen 5 2400G with Radeon Vega Graphics
  [OK]   gcc -march=native = znver1 (sesuai kunci)
  [OK]   CPU punya AVX2
  [OK]   CPU punya FMA
  [info] core fisik: 4. Satu CPU logis per core: 0,1,2,3
  [OK]   taskset -c 0-3 memakai 4 core fisik yang berbeda

## 2. Memori
  [info] RAM total 15905 MiB, tersedia 13092 MiB. Batas brief 14.2 GB = 13.2 GiB
  [OK]   RAM total sesuai 16 GB DDR4
  [info] swap: /swap.img   4G 581.1M
  [WARN] memlock 2035848 KiB (1.94 GiB). mlock model penuh gagal tanpa root. Brief 'mmap + mlock' perlu jalur lain
  [info] vm.swappiness = 60

## 3. GPU
  [info] GPU: NVIDIA GeForce GTX 1070 Ti, 6.1, 8192 MiB, 580.178.04
  [OK]   GTX 1070 Ti terdeteksi
  [OK]   compute capability 6.1 (sm_61)
  [WARN] ada proses GPU aktif. Jangan jalankan engine bersamaan:
        402229, /home/ivan/llama.cpp/build-vulkan/bin/llama-bench, 7601 MiB
  [OK]   tidak ada llama-server yang berjalan

## 4. Toolchain CUDA dan sm_61
  [OK]   nvcc 12.6 (/usr/local/cuda-12.6/bin/nvcc)
  [OK]   CUDA 12.x masih menerima -arch=sm_61
  [OK]   nvcc -arch=sm_61 -cubin berhasil untuk kedua probe
  [OK]   kontrol positif: IDP.4A muncul 1 kali untuk __dp4a
  [OK]   kontrol negatif: HFMA2 muncul 1 kali untuk __hfma2 (grep terbukti sensitif)
  [OK]   cuobjdump: /usr/local/cuda-12.6/bin/cuobjdump
  [OK]   nvdisasm: /usr/local/cuda-12.6/bin/nvdisasm
  [OK]   ncu: /usr/local/cuda-12.6/bin/ncu

## 5. Compiler dan build
  [OK]   g++ 11 (/usr/bin/g++)
  [OK]   g++ -O3 -march=znver1 -mavx2 -mfma: vfmadd x1, register ymm x4
  [WARN] clang++ tidak terpasang. Opsional. Build memakai g++
  [OK]   cmake 4.4.4
  [OK]   ninja 1.13.2.git.kitware.jobserver-pipe-1. Kebijakan: selalu ninja -j4
  [OK]   python3 3.10.12

## 6. CLI pendukung dan headless browser
  [OK]   taskset: /usr/bin/taskset
  [OK]   /usr/bin/time (GNU time) ada
  [OK]   jq: /usr/bin/jq
  [OK]   curl: /usr/bin/curl
  [OK]   gh: /usr/bin/gh
  [OK]   git: /usr/bin/git
  [OK]   hyperfine: /home/ivan/.local/bin/hyperfine
  [OK]   rg: /home/ivan/.local/bin/rg
  [WARN] node tidak ada. Dibutuhkan untuk Playwright. Pasang Node LTS (nvm tanpa sudo, atau paket resmi)
  [WARN] npx tidak ada. Ikut Node. Playwright dijalankan one-shot dengan npx setelah Node ada
  [WARN] tidak ada chromium/google-chrome. Tes UI memakai Playwright setelah Node ada, atau pasang chromium (tanpa snap jika bisa)

## 7. Referensi PR #38 (iso3 + MTP)
  [OK]   gh login: mcpe500
  [WARN] gh pr view polos gagal (field Projects classic di GraphQL). Ringkasan dibuat dari gh pr view --json
  [OK]   docs/ref_pr38_iso3_mtp.diff: 1565 baris
  [OK]   diff PR #38 memuat GGML_TYPE_ISO3_0 (42 kali)
  [info] status PR #38: MERGED merged=2026-11-15T12:45:15Z

## 8. Kebijakan: tanpa MCP, tanpa kode engine
  [OK]   tidak ada konfigurasi MCP di workspace ini
  [OK]   src/ belum berisi kode sumber (tahap scaffold)

## 9. Skill dan symlink
  [OK]   .agents/skills/iso3-kv-and-mtp/SKILL.md: frontmatter name dan description valid
  [OK]   .agents/skills/memory-budget-24gb/SKILL.md: frontmatter name dan description valid
  [OK]   .agents/skills/pascal-sm61-cuda/SKILL.md: frontmatter name dan description valid
  [OK]   .agents/skills/qwen38-hybrid-arch/SKILL.md: frontmatter name dan description valid
  [OK]   .agents/skills/webui-and-parity-bench/SKILL.md: frontmatter name dan description valid
  [OK]   5 dari 5 SKILL.md valid
  [OK]   .opencode/skills -> ../.agents/skills
  [OK]   .claude/skills -> ../.agents/skills
  [OK]   CLAUDE.md -> AGENTS.md

## 10. Model dan binary (untuk Tahap 1)
  [OK]   model 27B: /home/ivan/models/27b/Qwen3.8-27B-UD-Q4_K_M.gguf (16G)
  [info] varian lokal: Qwen3.8-27B-Q3-DOWN-XS.gguf (7.9G)
  [info] varian lokal: Qwen3.8-27B-Q4_0.gguf (15G)
  [info] varian lokal: Qwen3.8-27B-UD-IQ3_S.gguf (594M)
  [info] varian lokal: Qwen3.8-27B-UD-IQ3_XXS.gguf (11G)
  [info] varian lokal: Qwen3.8-27B-UD-IQ4_XS.gguf (14G)
  [info] varian lokal: Qwen3.8-27B-UD-Q4_K_M.gguf (16G)
  [OK]   /home/ivan/llama.cpp/build-iso3/bin/llama-server
  [OK]   /home/ivan/llama.cpp/build-iso3/bin/llama-bench
  [OK]   /home/ivan/llama.cpp/build-iso3/bin/llama-perplexity
  [WARN] submodule legacy engine/llama.cpp tidak ditemukan

## Ringkasan
  OK=47  WARN=8  FAIL=0
  laporan: docs/setup-report.md
