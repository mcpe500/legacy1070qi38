# Design — SPEC-006: Benchmark Suite

> Spec: SPEC-006 | Status: **Skeleton**

---

## Planned Components

```
benchmarks/
├── run_benchmark.sh          ← Main benchmark script
├── prompts/
│   ├── short-32.txt          ← 32 token prompt
│   ├── medium-128.txt        ← 128 token prompt
│   └── long-512.txt          ← 512 token prompt
├── results/
│   └── [timestamp]-[model]-[quant].md  ← Auto-generated results
└── compare.py                ← Compare multiple benchmark results
```

## Standard Prompts

Lihat `roles/benchmarker.md` untuk prompt standar yang sudah didefinisikan.

## Output Format

Lihat `roles/benchmarker.md` bagian "Output Format Hasil Benchmark" untuk format standar.

## Open Questions

- [ ] Apakah perlu Python untuk parsing llama.cpp output, atau cukup dengan bash + grep/awk?
- [ ] Format penyimpanan: JSON (mudah di-parse) atau Markdown (mudah dibaca manusia)?
- [ ] Apakah benchmark perlu mengukur quality metrics secara otomatis?
