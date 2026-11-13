# Design — SPEC-005: CPU-GPU Offload Strategy

> Spec: SPEC-005 | Status: **Skeleton**

---

## Known Architecture (dari RFC-001 dan MASTER.md)

### Qwen3-27B Layer Distribution

| Quant | Total Size | Layer Size | GPU Layers | CPU Layers | Tok/s est. |
|---|---|---|---|---|---|
| Q4_K_M | 16.4 GB | ~315 MB | ~22 | ~30 | 2–4 ❌ RAM OOM |
| Q3_K_M | 13.5 GB | ~260 MB | ~27 | ~25 | 3–6 ⚠️ Viable |
| Q2_K | 10.4 GB | ~200 MB | ~35 | ~17 | 6–10 ✅ |

> ⚠️ ASSUMPTION: Layer count Qwen3-27B = 52. Harus diverifikasi dari model card.
> ⚠️ ASSUMPTION: Layer size estimasi dari total_size/num_layers (uniform). Aktual mungkin bervariasi.

### Bottleneck Analysis

```
Decode bottleneck untuk hybrid mode:
- GPU layers: dibatasi GDDR5 bandwidth (256 GB/s)
- CPU layers: dibatasi DDR4 bandwidth (~40 GB/s)
- PCIe transfer (activation per boundary): ~10 KB × 2 = 20 KB per token → negligible
- BOTTLENECK UTAMA: DDR4 bandwidth untuk CPU layers
```

## Open Questions

- [ ] Berapa layer count Qwen3-27B yang sebenarnya? → Verifikasi model card
- [ ] Berapa RAM peak usage aktual untuk Q3_K_M? → Ukur empiris
- [ ] Apakah `--no-mmap` memberikan speedup yang signifikan vs default mmap?
- [ ] Apakah ada nilai -ngl optimal selain 27? (mungkin 24 atau 30 lebih baik?)
