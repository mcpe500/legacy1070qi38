| run | model | runtime | tool | KV | context | -ngl | -ub | prompt tok | pp tok/s | tg tok/s | peak VRAM MiB | vs 7373 MiB | note |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| prism_q2_q4_d0_ngl99_ub128 | Q2_0 | prism fork | llama-bench | q4_0 | depth 0 | 99 | 128 | 512 | 182.47 | 18.67 | 7365 | OK |  |
| ptq1_f16_d0_ngl99_ub512 | PTQ1_0 | prism fork | llama-bench | f16 | depth 0 | 99 | 512 | 512 | 98.94 | 19.64 | 6579 | OK |  |
| ptq1_q4_d16384_ngl99_ub512 | PTQ1_0 | prism fork | llama-bench | q4_0 | depth 16384 | 99 | 512 | 512 | 77.19 | 11.70 | 6861 | OK |  |
| ptq1_q4_d32768_ngl99_ub512 | PTQ1_0 | prism fork | llama-bench | q4_0 | depth 32768 | 99 | 512 | 512 | 66.00 | 7.51 | 7165 | OK |  |
| ptq1_q4_d4096_ngl99_ub512 | PTQ1_0 | prism fork | llama-bench | q4_0 | depth 4096 | 99 | 512 | 512 | 92.72 | 15.19 | 6633 | OK |  |
| q2_f16_d0_ngl60 | Q2_0 | stock | llama-bench | f16 | depth 0 | 60 | 512 | 512 | 156.33 | 8.48 | 7263 | OK |  |
| q2_f16_d0_ngl61 | Q2_0 | stock | llama-bench | f16 | depth 0 | 61 | 512 | 512 | 166.75 | 9.00 | 7369 | OK |  |
| q2_f16_d0_ngl62 | Q2_0 | stock | llama-bench | f16 | depth 0 | 62 | 512 | 512 | 172.72 | 9.42 | 7471 | OVER CAP |  |
| q2_f16_d0_ngl63 | Q2_0 | stock | llama-bench | f16 | depth 0 | 63 | 512 | 512 | 183.53 | 11.10 | 7577 | OVER CAP |  |
| q2_f16_d0_ngl64 | Q2_0 | stock | llama-bench | f16 | depth 0 | 64 | 512 | 512 | 196.87 | 13.33 | 7685 | OVER CAP |  |
| q2_f16_d0_ngl99 | Q2_0 | stock | llama-bench | f16 | depth 0 | 99 | 512 | 512 | 209.98 | 17.53 | 7791 | OVER CAP |  |
| q2_q4_d0_ngl99_ub128 | Q2_0 | stock | llama-bench | q4_0 | depth 0 | 99 | 128 | 512 | 182.74 | 17.13 | 7371 | OK |  |
| q2_q4_d0_ngl99_ub256 | Q2_0 | stock | llama-bench | q4_0 | depth 0 | 99 | 256 | 512 | 202.74 | 17.23 | 7513 | OVER CAP |  |
| q2_q4_d16384_ngl62_ub128 | Q2_0 | stock | llama-bench | q4_0 | depth 16384 | 62 | 128 | 512 | 89.11 | 7.20 | 7339 | OK |  |
| q2_q4_d32768_ngl59_ub128 | Q2_0 | stock | llama-bench | q4_0 | depth 32768 | 59 | 128 | 512 | 59.11 | 3.57 | 7355 | OK |  |
| q2_q4_d4096_ngl64_ub128 | Q2_0 | stock | llama-bench | q4_0 | depth 4096 | 64 | 128 | 512 | 139.17 | 12.43 | 7337 | OK |  |
| qwen_q4_d0_ngl25 | Qwen Q4_K_M | stock | llama-bench | q4_0 | depth 0 | 25 | 512 | 512 | 30.00 | 2.26 | 6945 | OK |  |
| qwen_q4_d16384_ngl24 | Qwen Q4_K_M | stock | llama-bench | q4_0 | depth 16384 | 24 | 512 | 512 | 32.02 | 1.35 | 6835 | OK |  |
| qwen_q4_d4096_ngl25 | Qwen Q4_K_M | stock | llama-bench | q4_0 | depth 4096 | 25 | 512 | 512 | 34.83 | 2.00 | 6973 | OK |  |
| srv_ptq1_f16_c16384_ngl99_ub128 | PTQ1_0 | prism fork | server | f16 | ctx 16384 | 99 | 128 | 487 | 43.51 | 19.18 | 7089 | OK |  |
| srv_ptq1_q4_0_c131072_ngl48_ub512 | PTQ1_0 | prism fork | server | q4_0 | ctx 131072 | 48 | 512 | 487 | 40.95 | 2.24 | 7303 | OK |  |
| srv_ptq1_q4_0_c65536_ngl62_full | PTQ1_0 | prism fork | server | q4_0 | ctx 65536 | 62 | 512 | 64988 | 59.11 | 3.36 | 7361 | OK |  |
| srv_ptq1_q4_0_c65536_ngl62_ub512 | PTQ1_0 | prism fork | server | q4_0 | ctx 65536 | 62 | 512 | 487 | 74.25 | 6.77 | 7361 | OK |  |
| srv_ptq1_q4_0_c65536_ngl64_ub128 | PTQ1_0 | prism fork | server | q4_0 | ctx 65536 | 64 | 128 | 487 | 40.38 | 11.80 | 7405 | OVER CAP |  |
| srv_ptq1_q4_0_c65536_ngl99_ub128 | PTQ1_0 | prism fork | server | q4_0 | ctx 65536 | 99 | 128 | 487 | 43.10 | 18.77 | 7471 | OVER CAP |  |
| srv_ptq1_q4_0_c65536_ngl99_ub512 | PTQ1_0 | prism fork | server | q4_0 | ctx 65536 | 99 | 512 | 487 | 89.94 | 18.80 | 7595 | OVER CAP |  |
| srv_ptq1_q8_0_c32768_ngl99_ub128 | PTQ1_0 | prism fork | server | q8_0 | ctx 32768 | 99 | 128 | 487 | 43.16 | 18.91 | 7271 | OK |  |
| srv_q2_f16_c32768_ngl59 | Q2_0 | stock | server | f16 | ctx 32768 | 59 | 128 |  | - | - | 6417 | FAILED | load failed: allocating 1920.00 MiB on device 0: cudaMalloc failed: out of memory |
| srv_q2_f16_c4096_ngl99 | Q2_0 | stock | server | f16 | ctx 4096 | 99 | 512 | 1231 | 179.82 | 17.08 | 7647 | OVER CAP |  |
| srv_q2_iso3_c131072_ngl46 | Q2_0 | stock | server | iso3 | ctx 131072 | 46 | 128 | 487 | 32.40 | 3.65 | 7113 | OK |  |
| srv_q2_iso3_c131072_ngl47 | Q2_0 | stock | server | iso3 | ctx 131072 | 47 | 128 | 487 | 33.79 | 3.82 | 7219 | OK |  |
| srv_q2_iso3_c131072_ngl48 | Q2_0 | stock | server | iso3 | ctx 131072 | 48 | 128 | 487 | 35.30 | 3.98 | 7327 | OK |  |
| srv_q2_iso3_c131072_ngl51 | Q2_0 | stock | server | iso3 | ctx 131072 | 51 | 128 | 487 | 40.53 | 4.57 | 7739 | OVER CAP |  |
| srv_q2_iso3_c131072_ngl52 | Q2_0 | stock | server | iso3 | ctx 131072 | 52 | 128 | 487 | 42.79 | 4.80 | 7847 | OVER CAP |  |
| srv_q2_iso3_c32768_ngl59 | Q2_0 | stock | server | iso3 | ctx 32768 | 59 | 128 | 515 | 71.68 | 7.81 | 7167 | OK |  |
| srv_q2_iso3_c65536_ngl56_full | Q2_0 | stock | server | iso3 | ctx 65536 | 56 | 128 | 64988 | 49.05 | 2.04 | 7351 | OK |  |
| srv_q2_iso3_c65536_ngl56_full_eos | Q2_0 | stock | server | iso3 | ctx 65536 | 56 | 128 | 64988 | 49.07 | - | - | superseded | 1 token generated (greedy EOS); rerun with ignore_eos; VRAM samples overwritten by the rerun |
| srv_q2_q4_0_c131072_ngl43 | Q2_0 | stock | server | q4_0 | ctx 131072 | 43 | 128 | 487 | 28.95 | 3.24 | 7227 | OK |  |
| srv_q2_q4_0_c131072_ngl48 | Q2_0 | stock | server | q4_0 | ctx 131072 | 48 | 128 | 487 | 35.23 | 3.94 | 7899 | OVER CAP |  |
| srv_q2_q4_0_c32768_ngl59 | Q2_0 | stock | server | q4_0 | ctx 32768 | 59 | 128 | 515 | 71.43 | 7.80 | 7343 | OK |  |
| srv_q2_q8_0_c32768_ngl59 | Q2_0 | stock | server | q8_0 | ctx 32768 | 59 | 128 | 515 | 71.58 | 7.74 | 7855 | OVER CAP |  |
