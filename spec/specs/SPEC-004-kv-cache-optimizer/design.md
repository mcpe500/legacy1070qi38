# Design — SPEC-004: KV Cache Optimizer

> Spec: SPEC-004 | Status: **Skeleton**

---

> 📋 **Status: Skeleton**

## Known Facts (dari RFC-001)

KV cache size untuk Qwen3-8B (36 layers, 8 KV heads, 128 head dim):

| Context | FP16 | q8_0 | q4_0 |
|---|---|---|---|
| 2048 | 288 MB | 144 MB | 72 MB |
| 4096 | 576 MB | 288 MB | 144 MB |

Llama.cpp sudah support: `--cache-type-k q8_0` dan `--cache-type-v q8_0`

> ⚠️ ASSUMPTION: Angka di atas berdasarkan formula teoritis; harus diverifikasi empiris di SPEC-002.

## Open Questions

- [ ] Apakah KV q4_0 menurunkan kualitas output secara signifikan untuk Qwen3?
- [ ] Berapa ctx maksimum yang bisa dicapai dengan kombinasi KV q4_0?
- [ ] Apakah ada benefit menggunakan KV q8_0 untuk K dan q4_0 untuk V (atau sebaliknya)?
