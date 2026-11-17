### Weight footprint (from the GGUF headers)

| Packing | file GB | 64 blocks MiB | output.weight MiB | token_embd MiB (host) | GPU weights MiB | left under 7373 MiB before KV, state and runtime |
|---|---:|---:|---:|---:|---:|---:|
| Q2_0 | 7.63 | 6580.1 | 341.0 | 341.0 | 6921.1 | 451.9 |
| PQ2_0 | 7.21 | 6217.6 | 322.1 | 322.1 | 6539.7 | 833.3 |
| PTQ1_0 | 5.95 | 5130.1 | 265.2 | 265.2 | 5395.3 | 1977.7 |

### KV cache size (16 attention layers; DeltaNet state is constant 149.6 MiB)

| KV type | bytes / token (16 layers) | 4K | 8K | 16K | 32K | 64K | 128K | 256K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| f16 | 65536 | 256 MiB | 512 MiB | 1024 MiB | 2048 MiB | 4096 MiB | 8192 MiB | 16384 MiB |
| q8_0 | 34816 | 136 MiB | 272 MiB | 544 MiB | 1088 MiB | 2176 MiB | 4352 MiB | 8704 MiB |
| q4_0 | 18432 | 72 MiB | 144 MiB | 288 MiB | 576 MiB | 1152 MiB | 2304 MiB | 4608 MiB |
| iso3 | 12800 | 50 MiB | 100 MiB | 200 MiB | 400 MiB | 800 MiB | 1600 MiB | 3200 MiB |

### Largest context with every layer on the GPU (runtime overhead from the measured model: 295 MiB + 6.1 KiB per token)

| Packing | KV type | VRAM left at 0 tokens MiB | max context (tokens) | 4K | 8K | 16K | 32K | 64K | 128K | 256K |
|---|---|---:|---:|---|---|---|---|---|---|---|
| Q2_0 | f16 | 7 | 106 | no | no | no | no | no | no | no |
| Q2_0 | q8_0 | 7 | 185 | no | no | no | no | no | no | no |
| Q2_0 | q4_0 | 7 | 308 | no | no | no | no | no | no | no |
| Q2_0 | iso3 | 7 | 400 | no | no | no | no | no | no | no |
| PQ2_0 | f16 | 389 | 5678 | yes | no | no | no | no | no | no |
| PQ2_0 | q8_0 | 389 | 9926 | yes | yes | no | no | no | no | no |
| PQ2_0 | q4_0 | 389 | 16516 | yes | yes | yes | no | no | no | no |
| PQ2_0 | iso3 | 389 | 21400 | yes | yes | yes | no | no | no | no |
| PTQ1_0 | f16 | 1533 | 22394 | yes | yes | yes | no | no | no | no |
| PTQ1_0 | q8_0 | 1533 | 39148 | yes | yes | yes | yes | no | no | no |
| PTQ1_0 | q4_0 | 1533 | 65138 | yes | yes | yes | yes | no | no | no |
| PTQ1_0 | iso3 | 1533 | 84400 | yes | yes | yes | yes | yes | no | no |

### Shared memory (48 KiB per block on sm_61)

- qwen38-engine keeps the attention scores of one query row in dynamic shared memory (4 bytes per position plus about 1.1 KiB): the kernel can address at most 12000 positions; the engine is limited to 4096 on purpose.
- llama.cpp's flash-attention kernels tile the KV sequence, so shared memory does not limit the context there.

### RAM (cap 14.2 GB = 13.22 GiB)

- Q2_0: the mapped file is 7.10 GiB; with every layer on the GPU only token_embd (341 MiB) and host buffers stay resident, far under the cap.
- PQ2_0: the mapped file is 6.71 GiB; with every layer on the GPU only token_embd (322 MiB) and host buffers stay resident, far under the cap.
- PTQ1_0: the mapped file is 5.54 GiB; with every layer on the GPU only token_embd (265 MiB) and host buffers stay resident, far under the cap.
