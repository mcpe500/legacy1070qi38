# src/engine

Orkestrator inferensi: arena statis, pembagian layer CPU/GPU, generator chunked-prefill, sampler,
state DeltaNet/rollback, penyimpanan state, dan CLI.

Status: **implementasi lengkap dan aktif**. Jalur Qwen hibrida dan mode Ternary Bonsai berjalan dari
binary yang sama. Batas konteks: 8192 pada jalur Qwen K-quant, 131072 pada jalur Ternary.

Baca dulu `.agents/skills/memory-budget-24gb/SKILL.md`, `qwen38-hybrid-arch`, dan
`iso3-kv-and-mtp`.
