# src/cuda

Kernel SM 61 untuk GEMV `__dp4a`, operator FP32, DeltaNet, KV `iso3`, Attention GQA, dan format
ternary Bonsai.

Status: **implementasi lengkap dan aktif**. Attention memakai kernel shared-memory sampai 8192
posisi dan kernel tiled online-softmax di atasnya, hingga 131072 posisi pada jalur Bonsai.

Baca dulu `.agents/skills/pascal-sm61-cuda/SKILL.md`, `iso3-kv-and-mtp`, dan
`memory-budget-24gb`. Setiap `.cu` wajib lulus `scripts/sass_gate.sh`.
