---
name: memory-budget-24gb
description: "Hard VRAM and RAM budgets for the 8 GB GTX 1070 Ti and 16 GB DDR4 machine, with the measured numbers that test them. Read before changing any allocation, arena, pinned buffer, mmap or mlock use, layer placement, thread pinning, or any benchmark that reads memory."
---

# Memory budget (8 GB VRAM, 16 GB DDR4)

## Ceilings (from the brief) and what the machine really gives
| Resource | Brief | Measured on this machine | Status |
|---|---|---|---|
| VRAM | <= 7.2 GB | 8192 MiB card. llama.cpp peaked at 8043 to 8057 MiB with 30 to 36 GPU layers (run-001) | **conflict**: 7.2 GB = 6866 MiB, about 1.1 GiB below what llama.cpp used |
| System RAM | <= 14.2 GB | 15905 MiB total, 14499 MiB available at audit | 14.2 GB = 13.2 GiB, fits |
| Swap | no growth during decode | 4 GiB file `/swap.img`, swappiness 60 | monitor it; llama bench grew swap by 0.175 GiB without an explanation |
| mlock | "mmap + mlock so Linux never swaps" | memlock limit 2035848 KiB = 1.94 GiB (setup.sh section 2) | **conflict**: cannot lock 13 GiB without root |

Units matter. 1 GB = 1000^3 bytes, 1 GiB = 1024^3 bytes. Write both units in every budget line.

## Derived sizes (exact formulas, verify with real allocations)
- KV iso3, 8192 tokens: 12.5 KiB per token x 8192 = 100 MiB. The brief's 100 to 130 MB fits.
- KV f16, 8192 tokens: 512 MiB.
- DeltaNet matrix state, f32: 144 MiB. Snapshots multiply this (see iso3-kv-and-mtp).
- Scratch: two FP32 ping-pong buffers, 5120 floats (20 KiB) and 17408 floats (68 KiB).
- Layer placement: not a constant. Compute the GPU layer count from the real GGUF tensor sizes
  (`gguf inspect`), then confirm with a measured peak. Do not use a number from this document.

## Enforcement rules
1. **Static arena.** Allocate at load time, then never allocate in the token loop:
   - one VRAM arena (`cudaMalloc` once),
   - one pinned host arena (`cudaHostAlloc` once; pinned memory counts against RAM),
   - two FP32 ping-pong scratch buffers.
   No `malloc`, `new`, `cudaMalloc`, `cudaMallocAsync`, or `std::vector` growth after warm-up.
2. **Debug counter.** In debug builds, wrap every allocator and count calls. The count must stay at
   zero after the first warm-up token.
3. **mmap without mlock on this machine** unless the memlock limit is raised by root. Keep the
   model file in the page cache, and verify that swap-in and swap-out stay at zero during decode.
4. **Thread pinning.** `taskset -c 0,1,2,3` for CPU work. setup.sh confirmed that CPUs 0 to 3 are four
   different physical cores on this Ryzen 5 2400G (SMT pairs are 0/4, 1/5, 2/6, 3/7).
5. **One engine at a time.** Never run llama-server and the custom engine together. The GPU and
   the RAM budget are both shared.

## Measurement commands
```bash
# VRAM peak during a run (100 ms sampling)
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 100 > vram.log &
# RSS peak of the engine process
/usr/bin/time -v taskset -c 0,1,2,3 ./engine ... 2> time.log
# Swap activity, must stay at 0 in si and so columns during decode
vmstat 1 > vmstat.log &
```
After each run, confirm that `nvidia-smi --query-compute-apps` is empty before the next run.

## Known numbers to beat (llama.cpp, same card, measured)
- Q4_K_M, `-ngl 99 -ot` FFN 17..64 on CPU, `-t 4`: 2.85 tok/s (bench pendek).
- RAM available minimum during that run: 14.12 GiB. Swap rose from 0.724 to 0.899 GiB (run-002).

## Checks before merge
- [ ] No allocation calls in the token loop (debug counter = 0 after warm-up).
- [ ] Peak VRAM from the sampler is at or below the cap in the unit the cap is written in.
- [ ] Peak RSS and swap activity recorded for the benchmark run.
- [ ] Thread pinning list matches setup.sh output.
