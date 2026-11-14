# Tasks — CLI Tooling (`tools/legacy.py`)

> Spec: SPEC-007 | Status: **Done untuk alat** (123 test unit + 3 test kontrak binary asli lulus). Uji pada GGUF 27B dan GPU dipindah ke issue uji 27B
> Dependencies: RFC-002 (Accepted)
> Blocks: SPEC-002 Task 4, SPEC-006

---

## Task List

- [x] **Task 1: Spec dan gambar**
  - Implementation: requirements, design, dan SVG `spec/specs/assets/spec-007-cli-implementation.svg`
  - Test: `python tools/legacy.py spec validate` (tautan SVG valid, penomoran FR/NFR berurutan)
  - Done when: validate keluar dengan kode 0 (⚠️ belum dikonfirmasi)

- [x] **Task 2: `budget`**
  - Implementation: `estimate`, `suggest_ngl`, `compare_measured` sebagai fungsi murni
  - Test: `BudgetKnownValues`, `BudgetInvalidInput`, `BudgetProperties` di `tools/test_legacy.py`
  - Done when: test lulus (⚠️ belum dikonfirmasi)

- [x] **Task 3: `gguf inspect` dan `gguf tensors`**
  - Implementation: parser header v2/v3 dengan batas pengaman
  - Test: `GgufTests` memakai file GGUF sintetis, file terpotong, dan panjang berbahaya
  - Done when: test lulus (⚠️ belum dikonfirmasi)

- [x] **Task 4: `hw`**
  - Implementation: parser murni + pembaca `/proc` dan `nvidia-smi`
  - Test: `HwParsers`, `HwRealSystem`, `HwGpuMissing`
  - Done when: test lulus (⚠️ belum dikonfirmasi); `hw gpu`, `hw cpu`, `hw ram`, `hw membw` dijalankan di GTX 1070 Ti ✅ (hasil di HANDOFF-LOG)

- [x] **Task 5: `llama`**
  - Implementation: argv list, allowlist, `InferenceLock`, timeout, parser timing/bench/placement
  - Test: `LlamaPure`, `LockTests`, `LlamaEndToEnd` (skrip llama palsu)
  - Done when: test lulus (⚠️ belum dikonfirmasi); uji dengan llama.cpp asli (belum)

- [x] **Task 6: `spec`**
  - Implementation: `new`, `session`, `handoff`, `questions`, `validate`
  - Test: `SpecScaffold`, `SpecHandoff`, `SpecQuestions`, `SpecValidate` (termasuk `test_real_repo_passes`)
  - Done when: test lulus (⚠️ belum dikonfirmasi)

- [x] **Task 7: Konfirmasi hasil test**
  - Implementation: jalankan `python tools/test_legacy.py -v` dan tempel keluaran di PR
  - Test: semua test lulus; kegagalan diperbaiki
  - Done when: keluaran test terlampir ✅ (87 test OK, 1 dilewati karena khusus mesin tanpa GPU; lihat deskripsi PR)

- [x] **Task 8: Uji pada data asli** (model `qwen35` 9B dan 0.8B, fork `08e025c`, CPU)
  - Implementation: `gguf inspect`/`tensors` pada 9B, `hw gpu`/`cpu`/`ram`/`membw` di mesin target, `llama run`/`bench` pada binary asli
  - Test: `RealBinaryContract` (3 test) + hasil di HANDOFF-LOG dan `spec/research/CLAIMS.md`
  - Done when: OQ-009 (9B), OQ-010, OQ-011 (sebagian) ditandai dengan bukti ✅. GGUF 27B belum ada di mesin: lihat issue uji 27B

- [x] **Task 9: Subcommand tambahan** (PR #29)
  - Implementation: `llama compare`, `bench analyze`, `quality ppl|kld|prompts`, clock + throttle GPU, `hw membw` multi-thread
  - Test: `FixedDefectsTests`, `GpuClockTests`, `MembwTests`, `AnalyzeAndCompareTests`, `QualityTests`, `RealBinaryContract`
  - Done when: lulus ✅. `llama-server` sengaja tidak dibungkus; pembungkus MCP tidak dibuat (RFC-002, keputusan 3 dan 6)

---

## Notes

- Perintah test: `python tools/test_legacy.py -v` atau `python -m unittest discover -s tools -p "test_legacy.py" -v` dari root repo.
- Test `llama` memakai skrip shell palsu di folder sementara; gagal bila folder sementara dipasang `noexec`.
- Test kontrak: `LEGACY_LLAMA_BIN_DIR=... LEGACY_TEST_MODEL=... python tools/test_legacy.py RealBinaryContract -v` (dilewati bila kosong; dipaksa CPU).
- Test `HwGpuMissing` hanya berjalan di mesin tanpa `nvidia-smi`.
- Tidak ada file di `tools/agent/` yang diubah.
