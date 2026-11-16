---
name: pascal-sm61-cuda
description: "Rules and verification gates for CUDA code on the GTX 1070 Ti (GP104, compute capability 6.1, sm_61). Read before writing or changing any .cu or .cuh file, GEMV, attention, DeltaNet, or dequantization kernel. Covers the FP32-only arithmetic rule, __dp4a integer dot products, memory coalescing, and the cuobjdump SASS gate."
---

# Pascal sm_61 CUDA

## Target facts
- GPU: GeForce GTX 1070 Ti, GP104, compute capability 6.1, 8192 MiB GDDR5, driver 580.178.04.
  Checked with nvidia-smi by scripts/setup.sh (section 3).
- Toolchain: nvcc 12.6.85 at /usr/local/cuda-12.6. `nvcc -arch=sm_61 -cubin` compiles
  (setup.sh section 4).
- 256 GB/s bandwidth and no Tensor Cores: from the brief and the public GP104 spec. Not measured here.
- FP16 arithmetic at 1/64 of FP32 rate: from the brief. Measure it with a micro-benchmark before
  citing it in a design document.

## Hard rules
1. **FP32 arithmetic only.** Activations, RMSNorm, SiLU and SwiGLU, RoPE, softmax, and DeltaNet
   recurrent state run in FP32. Do not use `half`, `__half`, `__half2`, `__hfma2`, `__hmul2`, `hexp`,
   `hsqrt`, or any other half intrinsic in arithmetic. Unpack packed formats to FP32 in registers.
2. **Integer dot products use `__dp4a`.** Quantize each input vector once per matmul to Q8_1-style
   blocks (int8 values plus one FP32 scale per block) in shared memory. Unpack 3-bit or 4-bit weights
   to int8 in registers. Accumulate `__dp4a` results in INT32 per block, then multiply by the FP32
   scales. The accumulator that leaves the kernel is FP32.
3. **Coalesced weight layout.** The offline packer interleaves GPU weight tiles so that one
   32-thread warp reads contiguous 128-byte segments. Use 16-byte vector loads (`int4`) where
   alignment allows. Never read weights with a stride equal to the row length.
4. **Launch bounds.** Every kernel declares `__launch_bounds__(threads, minBlocks)`. Record the values
   next to the ncu result.
5. **Release flags:** `-gencode arch=compute_61,code=sm_61 -O3`. Do not ship PTX-only objects.
   PTX JIT hides a missing SASS build.
6. **Build only with `ninja -j4`.** Never call ninja without `-j4` in this workspace.

## SASS gate (mandatory after every .cu compile)
```bash
cuobjdump -sass <obj-or-cubin> > sass.txt
grep -c 'IDP\.4A' sass.txt        # must be > 0 for every kernel that calls __dp4a
grep -cE 'HFMA2|HMUL2' sass.txt   # must be 0 for engine kernels (FP32-only rule)
```
- scripts/setup.sh checks the method on two probes: `__dp4a` must produce IDP.4A (positive control),
  and `__hfma2` must produce HFMA2 (negative control). A gate result counts only while both controls pass.
- A kernel that fails the gate is not merged, even if its output matches the CPU reference.

## Procedure for a new kernel
1. Write the CPU reference first, in src/cpu/ or tests/.
2. Write the kernel under the rules above.
3. Compile with `ninja -j4`.
4. Run the SASS gate.
5. Run the parity test against the CPU reference (tests/).
6. Profile with `ncu --set basic <binary>`, or targeted `--metrics`. Record occupancy, DRAM
   throughput, and registers per thread.

## Anti-patterns
- `cudaMalloc`, `cudaFree`, or `cudaMallocAsync` inside the token loop. See memory-budget-24gb.
- FP16 math in a fast path, even when the stored format is FP16.
- `atomicAdd` on FP32 inside the GEMV inner loop. Reduce per block in shared memory instead.
- Timing a kernel while another process uses the GPU. Check `nvidia-smi --query-compute-apps` first.
- Copying llama.cpp's iso3 CUDA path. It converts KV to FP16 before flash attention (PR #38, patch 2
  description, "Flash attention CUDA"). That breaks the FP32-only rule.

## Measured context on this card (llama.cpp, same GPU)
- Hybrid offload, Q4_K_M, FFN layers 17..64 on CPU, all attention and DeltaNet on GPU:
  2.85 tok/s in llama-bench (pp128, tg32). Source: legacy repo
  `spec/research/reports/2026-11-15-27b-run-002.md`.
- Peak VRAM 8043 to 8057 MiB at 30 to 36 GPU layers (run-001). That is above the 7.2 GB cap
  in the brief. See memory-budget-24gb.
- In the 4-thread CPU sweep, CPU compute limited throughput, not the GPU (run-002, C-53).

## Merge checklist
- [ ] Compiles with `ninja -j4` for sm_61.
- [ ] SASS gate passes.
- [ ] No half-precision arithmetic (grep `__half`, `__hfma`, `__hmul`).
- [ ] Parity test against the CPU reference passes.
- [ ] ncu profile recorded next to the kernel.
