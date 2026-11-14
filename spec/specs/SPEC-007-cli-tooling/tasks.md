# Tasks — CLI Tooling (`tools/legacy.py`)

> Spec: SPEC-007 | Status: **In Progress** (kode dan test ditulis; test belum dijalankan oleh agen)
> Dependencies: RFC-002 (Proposed)
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
  - Done when: test lulus (⚠️ belum dikonfirmasi); `hw gpu` diuji di GTX 1070 Ti (belum)

- [x] **Task 5: `llama`**
  - Implementation: argv list, allowlist, `InferenceLock`, timeout, parser timing/bench/placement
  - Test: `LlamaPure`, `LockTests`, `LlamaEndToEnd` (skrip llama palsu)
  - Done when: test lulus (⚠️ belum dikonfirmasi); uji dengan llama.cpp asli (belum)

- [x] **Task 6: `spec`**
  - Implementation: `new`, `session`, `handoff`, `questions`, `validate`
  - Test: `SpecScaffold`, `SpecHandoff`, `SpecQuestions`, `SpecValidate` (termasuk `test_real_repo_passes`)
  - Done when: test lulus (⚠️ belum dikonfirmasi)

- [ ] **Task 7: Konfirmasi hasil test**
  - Implementation: jalankan `python tools/test_legacy.py -v` dan tempel keluaran di PR
  - Test: semua test lulus; kegagalan diperbaiki
  - Done when: keluaran test terlampir

- [ ] **Task 8: Uji pada data asli**
  - Implementation: `gguf inspect` pada Qwen3.8-27B GGUF, `hw gpu`/`hw membw` pada mesin target, `llama run` pada llama.cpp asli
  - Test: hasil dicatat di HANDOFF-LOG (menjawab OQ-010, OQ-011)
  - Done when: OQ terkait ditandai terjawab dengan bukti

- [ ] **Task 9: Subcommand yang belum ada**
  - Implementation: `llama compare`, `llama-server`, `bench analyze`, `quality eval`, status throttle GPU
  - Test: ikuti pola task di atas
  - Done when: diputuskan pemilik repo (RFC-002)

---

## Notes

- Perintah test: `python tools/test_legacy.py -v` atau `python -m unittest discover -s tools -p "test_legacy.py" -v` dari root repo.
- Test `llama` memakai skrip shell palsu di folder sementara; gagal bila folder sementara dipasang `noexec`.
- Test `HwGpuMissing` hanya berjalan di mesin tanpa `nvidia-smi`.
- Tidak ada file di `tools/agent/` yang diubah.
