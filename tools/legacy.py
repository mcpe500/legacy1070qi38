#!/usr/bin/env python3
"""legacy.py - CLI tooling for the legacy1070qi38 project (SPEC-007, RFC-002).

One file, standard library only. Every subcommand supports --json.
Logic lives in plain functions; the argparse layer at the bottom is thin so
the same functions can later be wrapped by MCP if that is ever decided.

Groups:
  budget  estimate | suggest-ngl | compare      (pure math, no hardware)
  hw      gpu | sample | ram | cpu | membw | faults | headroom
  gguf    inspect | tensors                     (reads the GGUF header only)
  llama   build | run | bench | placement
  spec    new | session | handoff | questions | validate

Exit codes: 0 ok, 1 check failed (no-go, validation errors, out of tolerance),
2 bad input, 3 required external tool or file missing.

NOTE: Pascal (SM 6.1) has no Tensor Cores, no BF16 and no usable Flash
Attention library. This tool never launches anything that depends on them;
it only builds llama.cpp with CUDA arch 61. See spec/AGENTS/MASTER.md.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
from datetime import date

GIB = 1024 ** 3
MIB = 1024 ** 2

EXIT_OK = 0
EXIT_CHECK_FAILED = 1
EXIT_BAD_INPUT = 2
EXIT_ENV = 3


class CliError(Exception):
    """Bad input or unusable state. Maps to exit code 2."""

    exit_code = EXIT_BAD_INPUT


class EnvError(CliError):
    """Missing external tool or file. Maps to exit code 3."""

    exit_code = EXIT_ENV


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------

def _positive(name, value):
    if value is None or not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CliError(f"{name} harus berupa angka")
    if not math.isfinite(value) or value <= 0:
        raise CliError(f"{name} harus > 0 (dapat: {value})")
    return value


def _non_negative(name, value):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise CliError(f"{name} harus >= 0 (dapat: {value})")
    return value


# --------------------------------------------------------------------------
# A. budget  (pure functions)
# --------------------------------------------------------------------------

# Bytes per element relative to FP16, from ggml block layouts:
# q8_0 = 34 bytes / 32 values, q4_0 = 18 bytes / 32 values, FP16 = 64 bytes / 32 values.
# memory-manager.md says "about half" for q8_0; the exact block ratio is 0.531.
# iso3 (fork llama-cpp-turboquant, IsoQuant): block of 128 values = 2 (norm) + 32 + 16 = 50 bytes
# (ggml-common.h block_iso3_0), i.e. 3.125 bits/value.
KV_TYPE_FACTOR = {"f16": 1.0, "q8_0": 34.0 / 64.0, "q4_0": 18.0 / 64.0, "iso3": 50.0 / 256.0}

# State rekuren Gated DeltaNet (ukuran tetap, tidak tumbuh dengan konteks), Qwen3.8-27B:
# 48 layer x (48 head V x 128 x 128 x 4 B + konv 3 x 10240 x 4 B) = 149.6 MiB.
# Rumus diverifikasi persis pada Qwen3.5-0.8B (log "RS buffer size = 77.06 MiB" untuk 4 sekuens).
DEFAULT_RS_GIB = 0.146

# Qwen3.8-27B, kartu resmi Qwen/Qwen3.8-27B: 16 layer Gated Attention, 4 head KV, head dim 256
# -> 16 x 2 x 4 x 256 x 2 B = 64 KiB per token (FP16).
DEFAULT_KV_LAYERS = 16
DEFAULT_KV_HEADS = 4
DEFAULT_HEAD_DIM = 256
DEFAULT_TOTAL_LAYERS = 64

RISK_LABEL = {"green": "🟢 Safe", "yellow": "🟡 Tight", "red": "🔴 Likely OOM"}


def kv_bytes_per_token(kv_layers, kv_heads, head_dim, kv_type="f16"):
    """K and V, 2 bytes per FP16 element, scaled by cache type."""
    _positive("kv_layers", kv_layers)
    _positive("kv_heads", kv_heads)
    _positive("head_dim", head_dim)
    if kv_type not in KV_TYPE_FACTOR:
        raise CliError(f"kv_type tidak dikenal: {kv_type} (pilihan: {', '.join(KV_TYPE_FACTOR)})")
    return 2 * kv_layers * kv_heads * head_dim * 2 * KV_TYPE_FACTOR[kv_type]


def classify_risk(ram_headroom_gib, vram_room_gib):
    """Thresholds follow memory-manager.md: RAM headroom >= 0.5 GiB is required.

    The 2.0 GiB 'tight' threshold is a project choice, not a measured value.
    """
    if ram_headroom_gib < 0.5 or vram_room_gib <= 0:
        return "red"
    if ram_headroom_gib < 2.0:
        return "yellow"
    return "green"


def estimate(file_gib, vram_gib=8.0, ram_gib=16.0, ctx=4096,
             kv_layers=DEFAULT_KV_LAYERS, kv_heads=DEFAULT_KV_HEADS,
             head_dim=DEFAULT_HEAD_DIM, kv_type="f16",
             cuda_ctx_gib=0.4, compute_gib=0.5, os_gib=2.5,
             ram_bw_gbs=20.0, gpu_bw_gbs=256.0, rs_gib=DEFAULT_RS_GIB):
    """Memory split and TPS upper bound, using the formula in MASTER.md.

    Weights that fit in VRAM are not duplicated in RAM. The TPS figure is a
    theoretical upper bound: time per token = RAM weights / RAM bandwidth
    + GPU weights / GPU bandwidth.
    """
    _positive("file_gib", file_gib)
    _positive("vram_gib", vram_gib)
    _positive("ram_gib", ram_gib)
    _positive("ctx", ctx)
    _non_negative("cuda_ctx_gib", cuda_ctx_gib)
    _non_negative("compute_gib", compute_gib)
    _non_negative("os_gib", os_gib)
    _non_negative("rs_gib", rs_gib)
    _positive("ram_bw_gbs", ram_bw_gbs)
    _positive("gpu_bw_gbs", gpu_bw_gbs)

    kv_gib = kv_bytes_per_token(kv_layers, kv_heads, head_dim, kv_type) * ctx / GIB
    fixed = cuda_ctx_gib + kv_gib + compute_gib + rs_gib
    capacity = max(0.0, vram_gib - fixed)
    weights_gpu = min(file_gib, capacity)
    weights_ram = file_gib - weights_gpu

    vram_used = fixed + weights_gpu
    ram_total = weights_ram + os_gib
    ram_headroom = ram_gib - ram_total
    vram_room = vram_gib - fixed  # room left for weights; <= 0 means nothing fits

    seconds = (weights_ram * GIB) / (ram_bw_gbs * 1e9) + (weights_gpu * GIB) / (gpu_bw_gbs * 1e9)
    tps_max = 1.0 / seconds if seconds > 0 else None
    risk = classify_risk(ram_headroom, vram_room)

    return {
        "inputs": {
            "file_gib": file_gib, "vram_gib": vram_gib, "ram_gib": ram_gib, "ctx": ctx,
            "kv_layers": kv_layers, "kv_heads": kv_heads, "head_dim": head_dim,
            "kv_type": kv_type, "ram_bw_gbs": ram_bw_gbs, "gpu_bw_gbs": gpu_bw_gbs,
        },
        "kv_bytes_per_token": kv_bytes_per_token(kv_layers, kv_heads, head_dim, kv_type),
        "kv_gib": round(kv_gib, 4),
        "vram": {
            "cuda_context_gib": cuda_ctx_gib,
            "kv_gib": round(kv_gib, 4),
            "compute_buffer_gib": compute_gib,
            "recurrent_state_gib": rs_gib,
            "weights_capacity_gib": round(capacity, 4),
            "weights_gib": round(weights_gpu, 4),
            "used_gib": round(vram_used, 4),
            "free_gib": round(vram_gib - vram_used, 4),
        },
        "ram": {
            "weights_gib": round(weights_ram, 4),
            "os_runtime_gib": os_gib,
            "total_gib": round(ram_total, 4),
            "headroom_gib": round(ram_headroom, 4),
        },
        "tps_upper_bound": None if tps_max is None else round(tps_max, 3),
        "risk": risk,
        "risk_label": RISK_LABEL[risk],
        "assumptions": [
            "ASSUMPTION: compute buffer, CUDA context dan OS/runtime adalah tebakan, belum terukur",
            "kv_layers/kv_heads/head_dim default = kartu resmi Qwen3.8-27B (16 x 4 x 256)",
            "ASSUMPTION: KV dan state Gated DeltaNet semua di GPU; state 27B = 0.146 GiB (turunan "
            "kartu resmi, rumus terverifikasi pada 0.8B; ukur di 27B)",
            "TPS adalah batas atas teoritis berbasis bandwidth, bukan hasil ukur",
        ],
    }


def suggest_ngl(file_gib, layers=DEFAULT_TOTAL_LAYERS, **kwargs):
    """Number of layers that fit in VRAM, assuming uniform layer size.

    ASSUMPTION: uniform layers. Embedding/output tensors and the two layer
    kinds (attention vs DeltaNet) differ in size; use `gguf tensors` for real data.
    """
    _positive("layers", layers)
    est = estimate(file_gib, **kwargs)
    per_layer = file_gib / layers
    ngl = int(min(layers, math.floor(est["vram"]["weights_capacity_gib"] / per_layer)))
    ngl = max(0, ngl)
    return {
        "ngl": ngl,
        "layers": int(layers),
        "per_layer_gib": round(per_layer, 4),
        "weights_capacity_gib": est["vram"]["weights_capacity_gib"],
        "estimate": est,
        "assumptions": ["ASSUMPTION: ukuran layer seragam; cek `gguf tensors` untuk angka nyata"],
    }


def compare_measured(est, measured_vram_gib=None, measured_ram_gib=None,
                     measured_tps=None, tolerance_pct=15.0):
    """Compare an estimate() result against measurements.

    measured_ram_gib is the peak system RAM used, comparable to est['ram']['total_gib'].
    measured_tps is compared against the upper bound; a measurement above it
    means the model of the system is wrong, so it is flagged.
    """
    _positive("tolerance_pct", tolerance_pct)
    pairs = []
    if measured_vram_gib is not None:
        pairs.append(("vram_used_gib", est["vram"]["used_gib"], measured_vram_gib))
    if measured_ram_gib is not None:
        pairs.append(("ram_total_gib", est["ram"]["total_gib"], measured_ram_gib))
    if measured_tps is not None:
        pairs.append(("tps", est["tps_upper_bound"], measured_tps))
    if not pairs:
        raise CliError("beri minimal satu --measured-vram-gib / --measured-ram-gib / --measured-tps")
    rows = []
    ok = True
    for name, e, m in pairs:
        _non_negative("measured " + name, m)
        diff_pct = None if not e else (m - e) / e * 100.0
        if name == "tps":
            # Upper bound: measurement should be <= estimate. Far below is normal.
            within = e is None or m <= e * (1 + tolerance_pct / 100.0)
            verdict = "ok (di bawah batas atas)" if within else "model salah (hasil ukur melampaui batas atas)"
        else:
            within = diff_pct is None or abs(diff_pct) <= tolerance_pct
            verdict = "ok" if within else "di luar toleransi"
        ok = ok and within
        rows.append({"metric": name, "estimate": e, "measured": m,
                     "diff_pct": None if diff_pct is None else round(diff_pct, 2),
                     "verdict": verdict})
    return {"tolerance_pct": tolerance_pct, "all_within": ok, "rows": rows}


# --------------------------------------------------------------------------
# B. hw  (parsers are pure; thin wrappers call the OS)
# --------------------------------------------------------------------------

def run_capture(argv, timeout=15):
    """Run a fixed argv list (never a shell string) and return stdout."""
    try:
        proc = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, universal_newlines=True, timeout=timeout,
                              shell=False)
    except FileNotFoundError:
        raise EnvError(f"program tidak ditemukan: {argv[0]}")
    except subprocess.TimeoutExpired:
        raise CliError(f"{argv[0]} melewati batas waktu {timeout}s")
    if proc.returncode != 0:
        raise CliError(f"{argv[0]} gagal (kode {proc.returncode}): {proc.stderr.strip()[:300]}")
    return proc.stdout


def _num(text):
    text = text.strip()
    if text in ("", "N/A", "[N/A]", "[Not Supported]", "Not Supported"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


NVIDIA_QUERY = ("name,memory.total,memory.used,memory.free,temperature.gpu,power.draw,"
                "utilization.gpu,clocks.sm,clocks.mem,clocks_throttle_reasons.active")

# Bit topeng dari nvidia-smi (NVML clocksThrottleReason*).
THROTTLE_BITS = {
    0x1: "gpu_idle", 0x2: "applications_clocks_setting", 0x4: "sw_power_cap",
    0x8: "hw_slowdown", 0x10: "sync_boost", 0x20: "sw_thermal_slowdown",
    0x40: "hw_thermal_slowdown", 0x80: "hw_power_brake_slowdown", 0x100: "display_clock_setting",
}
# Hanya alasan ini yang berarti GPU benar-benar ditahan (idle dan setelan aplikasi bukan masalah).
THROTTLE_LIMITING = 0x4 | 0x8 | 0x20 | 0x40 | 0x80


def decode_throttle(value):
    """Hex/int bitmask -> (daftar nama, apakah ada yang membatasi). None bila tak terbaca."""
    if value is None:
        return None, None
    try:
        mask = int(str(value).strip(), 16) if isinstance(value, str) else int(value)
    except ValueError:
        return None, None
    names = [n for bit, n in sorted(THROTTLE_BITS.items()) if mask & bit]
    return names, bool(mask & THROTTLE_LIMITING)


def parse_nvidia_csv(text):
    """Parse `nvidia-smi --query-gpu=<NVIDIA_QUERY> --format=csv,noheader,nounits`."""
    gpus = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) not in (7, 10):
            raise CliError(f"keluaran nvidia-smi tidak dikenali: {line!r}")
        total, used, free = _num(parts[1]), _num(parts[2]), _num(parts[3])
        gpus.append({
            "name": parts[0],
            "vram_total_mib": total, "vram_used_mib": used, "vram_free_mib": free,
            "vram_free_gib": None if free is None else round(free / 1024.0, 3),
            "temp_c": _num(parts[4]), "power_w": _num(parts[5]), "util_pct": _num(parts[6]),
        })
        if len(parts) == 10:
            names, limiting = decode_throttle(parts[9] if parts[9] not in ("", "N/A", "[N/A]",
                                                                           "[Not Supported]") else None)
            gpus[-1].update({"sm_clock_mhz": _num(parts[7]), "mem_clock_mhz": _num(parts[8]),
                             "throttle_reasons": names, "throttle_limiting": limiting})
    if not gpus:
        raise CliError("nvidia-smi tidak mengembalikan GPU")
    return gpus


def read_gpu():
    out = run_capture(["nvidia-smi", "--query-gpu=" + NVIDIA_QUERY, "--format=csv,noheader,nounits"])
    return parse_nvidia_csv(out)


def parse_meminfo(text):
    kb = {}
    for line in text.splitlines():
        m = re.match(r"^(\w+)(?:\([\w]+\))?:\s+(\d+)\s*kB", line)
        if m:
            kb[m.group(1)] = int(m.group(2))
    for key in ("MemTotal", "MemAvailable"):
        if key not in kb:
            raise CliError(f"/proc/meminfo tidak memuat {key}")

    def gib(k):
        return round(kb.get(k, 0) * 1024 / GIB, 3)

    swap_total = gib("SwapTotal")
    swap_free = gib("SwapFree")
    return {
        "total_gib": gib("MemTotal"), "available_gib": gib("MemAvailable"),
        "free_gib": gib("MemFree"), "cached_gib": gib("Cached"),
        "swap_total_gib": swap_total, "swap_used_gib": round(swap_total - swap_free, 3),
    }


def read_ram():
    try:
        with open("/proc/meminfo", "r", encoding="ascii", errors="replace") as fh:
            return parse_meminfo(fh.read())
    except OSError as exc:
        raise EnvError(f"tidak bisa membaca /proc/meminfo: {exc}")


def parse_cpuinfo(text):
    model = None
    threads = 0
    cores = set()
    flags = set()
    mhz = []
    phys = core = None
    for line in text.splitlines() + [""]:
        if not line.strip():
            if phys is not None and core is not None:
                cores.add((phys, core))
            phys = core = None
            continue
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()
        if key == "processor":
            threads += 1
        elif key == "model name" and model is None:
            model = val
        elif key == "physical id":
            phys = val
        elif key == "core id":
            core = val
        elif key == "flags" and not flags:
            flags = set(val.split())
        elif key == "cpu MHz":
            try:
                mhz.append(float(val))
            except ValueError:
                pass
    return {
        "model": model, "threads": threads,
        "cores": len(cores) if cores else None,
        "avx2": "avx2" in flags, "avx512f": "avx512f" in flags, "fma": "fma" in flags,
        "mhz_max_now": max(mhz) if mhz else None,
    }


def read_cpu():
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as fh:
            return parse_cpuinfo(fh.read())
    except OSError as exc:
        raise EnvError(f"tidak bisa membaca /proc/cpuinfo: {exc}")


def parse_vmstat(text):
    wanted = ("pgmajfault", "pswpin", "pswpout")
    out = {k: 0 for k in wanted}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in out:
            out[parts[0]] = int(parts[1])
    return out


def read_vmstat():
    try:
        with open("/proc/vmstat", "r", encoding="ascii", errors="replace") as fh:
            return parse_vmstat(fh.read())
    except OSError as exc:
        raise EnvError(f"tidak bisa membaca /proc/vmstat: {exc}")


def vmstat_delta(before, after, seconds):
    _positive("seconds", seconds)
    delta = {k: max(0, after[k] - before[k]) for k in before}
    swapping = delta["pswpin"] + delta["pswpout"] > 0
    return {
        "seconds": seconds,
        "major_faults": delta["pgmajfault"],
        "major_faults_per_s": round(delta["pgmajfault"] / seconds, 2),
        "swap_in_pages": delta["pswpin"], "swap_out_pages": delta["pswpout"],
        "swapping": swapping,
    }


def summarize_samples(samples):
    """samples: list of dicts with vram_used_mib, temp_c, power_w, util_pct (any may be None)."""
    if not samples:
        raise CliError("tidak ada sampel")

    def peak(key):
        vals = [s[key] for s in samples if s.get(key) is not None]
        return max(vals) if vals else None

    utils = [s["util_pct"] for s in samples if s.get("util_pct") is not None]
    busy_clocks = [s["sm_clock_mhz"] for s in samples
                   if s.get("sm_clock_mhz") is not None and (s.get("util_pct") or 0) >= 50]
    reasons = sorted({r for s in samples for r in (s.get("throttle_reasons") or [])})
    limited = sum(1 for s in samples if s.get("throttle_limiting"))
    return {
        "samples": len(samples),
        "peak_sm_clock_mhz": peak("sm_clock_mhz"),
        "min_sm_clock_mhz_when_busy": min(busy_clocks) if busy_clocks else None,
        "throttle_reasons_seen": reasons,
        "samples_throttled": limited,
        "peak_vram_used_mib": peak("vram_used_mib"),
        "peak_temp_c": peak("temp_c"),
        "peak_power_w": peak("power_w"),
        "mean_util_pct": round(sum(utils) / len(utils), 1) if utils else None,
    }


def headroom_check(free_vram_gib, avail_ram_gib, need_vram_gib, need_ram_gib,
                   vram_margin_gib=0.2, ram_margin_gib=0.5):
    """Go/no-go. Margins follow the checklist in memory-manager.md (200 MB VRAM, 500 MB RAM)."""
    vram_left = None if free_vram_gib is None else free_vram_gib - need_vram_gib
    ram_left = avail_ram_gib - need_ram_gib
    vram_ok = True if vram_left is None else vram_left >= vram_margin_gib
    ram_ok = ram_left >= ram_margin_gib
    return {
        "go": bool(vram_ok and ram_ok),
        "vram": {"free_gib": free_vram_gib, "need_gib": need_vram_gib,
                 "left_gib": None if vram_left is None else round(vram_left, 3),
                 "margin_gib": vram_margin_gib, "ok": vram_ok},
        "ram": {"available_gib": avail_ram_gib, "need_gib": need_ram_gib,
                "left_gib": round(ram_left, 3), "margin_gib": ram_margin_gib, "ok": ram_ok},
    }


def _copy_once(size_bytes, threads, src, dst):
    """Satu salinan memori dibagi ke `threads` thread. ctypes.memmove melepas GIL."""
    import ctypes
    import threading
    src_addr = ctypes.addressof((ctypes.c_char * size_bytes).from_buffer(src))
    dst_addr = ctypes.addressof((ctypes.c_char * size_bytes).from_buffer(dst))
    chunk = size_bytes // threads
    workers = []
    for i in range(threads):
        off = i * chunk
        n = chunk if i < threads - 1 else size_bytes - off
        workers.append(threading.Thread(target=ctypes.memmove,
                                        args=(dst_addr + off, src_addr + off, n)))
    t0 = time.perf_counter()
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    return time.perf_counter() - t0


def measure_membw(size_mib=256, repeats=5, threads=1):
    """Bandwidth salin memori (memmove) dengan `threads` thread; ambil waktu terbaik.

    Mengukur laju salin satu arah (byte yang dipindah per detik). Lalu lintas memori sebenarnya
    sekitar dua kali lipat (baca + tulis), jadi angka ini batas bawah konservatif untuk laju baca.
    Tidak membedakan single/dual channel; bandingkan dengan spesifikasi RAM untuk itu.
    """
    if not 16 <= size_mib <= 2048:
        raise CliError("--size-mib harus antara 16 dan 2048")
    if not 1 <= repeats <= 50:
        raise CliError("--repeats harus antara 1 dan 50")
    if not 1 <= threads <= 64:
        raise CliError("--threads harus antara 1 dan 64")
    size = size_mib * MIB
    src = bytearray(b"\xa5") * size   # nilai bukan nol: memaksa halaman benar-benar dipetakan
    dst = bytearray(b"\x5a") * size
    best = None
    for _ in range(repeats):
        dt = _copy_once(size, threads, src, dst)
        best = dt if best is None else min(best, dt)
    return {"size_mib": size_mib, "repeats": repeats, "threads": threads,
            "copy_gb_per_s": round(size / best / 1e9, 2),
            "note": "laju salin satu arah (memmove); lalu lintas memori nyata ~2x; batas bawah laju baca"}


def measure_membw_sweep(size_mib=256, repeats=5, max_threads=None):
    """Ulangi untuk 1,2,4,.. thread; hasil terbaik dilaporkan sebagai perkiraan bandwidth."""
    cap = max(1, min(max_threads or (os.cpu_count() or 1), 16))
    counts, n = [], 1
    while n < cap:
        counts.append(n)
        n *= 2
    counts.append(cap)
    runs = [measure_membw(size_mib, repeats, t) for t in sorted(set(counts))]
    best = max(runs, key=lambda r: r["copy_gb_per_s"])
    return {"size_mib": size_mib, "repeats": repeats,
            "by_threads": {str(r["threads"]): r["copy_gb_per_s"] for r in runs},
            "best_threads": best["threads"], "copy_gb_per_s": best["copy_gb_per_s"],
            "note": best["note"]}


# --------------------------------------------------------------------------
# C1. gguf  (reads the header only; never loads tensor data)
# --------------------------------------------------------------------------

GGUF_MAGIC = b"GGUF"
MAX_STR = 1 << 26
MAX_ARRAY = 1 << 24
MAX_KV = 1 << 20
MAX_TENSORS = 1 << 22
ARRAY_KEEP = 512

_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
            6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

# ggml type id -> (name, values per block, bytes per block).
# ASSUMPTION: transcribed from the ggml type table; unknown ids give nbytes=None.
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1),
    25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2),
}


def _read(fh, n):
    data = fh.read(n)
    if len(data) != n:
        raise CliError("file GGUF terpotong atau rusak")
    return data


def _u32(fh):
    return struct.unpack("<I", _read(fh, 4))[0]


def _u64(fh):
    return struct.unpack("<Q", _read(fh, 8))[0]


def _gstr(fh):
    n = _u64(fh)
    if n > MAX_STR:
        raise CliError(f"string GGUF terlalu panjang ({n})")
    return _read(fh, n).decode("utf-8", "replace")


def _gvalue(fh, vtype, depth=0):
    if vtype in _SCALARS:
        fmt = _SCALARS[vtype]
        return struct.unpack(fmt, _read(fh, struct.calcsize(fmt)))[0]
    if vtype == 8:
        return _gstr(fh)
    if vtype == 9:
        if depth >= 3:
            raise CliError("array GGUF bersarang terlalu dalam")
        etype = _u32(fh)
        n = _u64(fh)
        if n > MAX_ARRAY:
            raise CliError(f"array GGUF terlalu besar ({n})")
        keep = n <= ARRAY_KEEP
        items = []
        for _ in range(n):
            v = _gvalue(fh, etype, depth + 1)
            if keep:
                items.append(v)
        return items if keep else {"array_len": n, "elem_type": etype}
    raise CliError(f"tipe nilai GGUF tidak dikenal: {vtype}")


def tensor_nbytes(dims, type_id):
    info = GGML_TYPES.get(type_id)
    if info is None:
        return None
    _, block, size = info
    n = 1
    for d in dims:
        n *= d
    if n % block:
        return None
    return n // block * size


def parse_gguf(path):
    """Parse the GGUF header and tensor table (v2/v3, little endian)."""
    if not os.path.isfile(path):
        raise EnvError(f"file tidak ditemukan: {path}")
    with open(path, "rb") as fh:
        if _read(fh, 4) != GGUF_MAGIC:
            raise CliError("bukan file GGUF (magic salah)")
        version = _u32(fh)
        if version not in (2, 3):
            raise CliError(f"versi GGUF {version} tidak didukung (hanya 2 dan 3)")
        n_tensors = _u64(fh)
        n_kv = _u64(fh)
        if n_kv > MAX_KV or n_tensors > MAX_TENSORS:
            raise CliError("jumlah kv/tensor GGUF tidak masuk akal")
        metadata = {}
        for _ in range(n_kv):
            key = _gstr(fh)
            vtype = _u32(fh)
            metadata[key] = _gvalue(fh, vtype)
        tensors = []
        for _ in range(n_tensors):
            name = _gstr(fh)
            n_dims = _u32(fh)
            if not 1 <= n_dims <= 8:
                raise CliError(f"jumlah dimensi tensor tidak valid: {n_dims}")
            dims = [_u64(fh) for _ in range(n_dims)]
            type_id = _u32(fh)
            offset = _u64(fh)
            tinfo = GGML_TYPES.get(type_id)
            tensors.append({"name": name, "dims": dims, "type_id": type_id,
                            "type": tinfo[0] if tinfo else f"unknown({type_id})",
                            "offset": offset, "nbytes": tensor_nbytes(dims, type_id)})
    return {"version": version, "tensor_count": n_tensors, "metadata": metadata,
            "tensors": tensors, "file_size": os.path.getsize(path)}


def summarize_gguf(g):
    md = g["metadata"]
    arch = md.get("general.architecture")

    def a(key):
        return md.get(f"{arch}.{key}") if arch else None

    kv_heads = a("attention.head_count_kv")
    block_count = a("block_count")
    kv_layers = None
    kv_source = None
    interval = a("full_attention_interval")
    if isinstance(kv_heads, list):
        kv_layers = sum(1 for h in kv_heads if h)
        kv_source = "head_count_kv array"
    elif isinstance(interval, int) and interval > 0 and isinstance(block_count, int):
        # Hibrida (Gated DeltaNet): hanya tiap layer ke-`interval` memakai full attention + KV cache.
        kv_layers = block_count // interval
        kv_source = "block_count // full_attention_interval"
    elif isinstance(block_count, int):
        kv_layers = block_count
        kv_source = "block_count"
    total = sum(t["nbytes"] for t in g["tensors"] if t["nbytes"] is not None)
    return {
        "gguf_version": g["version"],
        "architecture": arch,
        "name": md.get("general.name"),
        "file_type": md.get("general.file_type"),
        "context_length": a("context_length"),
        "block_count": block_count,
        "embedding_length": a("embedding_length"),
        "head_count": a("attention.head_count"),
        "head_count_kv": kv_heads,
        "key_length": a("attention.key_length"),
        "value_length": a("attention.value_length"),
        "kv_layers_estimate": kv_layers,
        "kv_layers_source": kv_source,
        "full_attention_interval": interval,
        "tensor_count": g["tensor_count"],
        "metadata_keys": len(md),
        "tensor_bytes_gib": round(total / GIB, 4),
        "file_size_gib": round(g["file_size"] / GIB, 4),
        "notes": [
            "kv_layers_estimate: entri head_count_kv > 0 bila array; bila ada "
            "<arch>.full_attention_interval maka block_count // interval (terverifikasi pada "
            "Qwen3.8-9B arsitektur qwen35: 32 // 4 = 8); selain itu block_count.",
        ],
    }


def summarize_tensors(g):
    layers = {}
    other = {}
    by_type = {}
    unknown = 0
    for t in g["tensors"]:
        nb = t["nbytes"]
        if nb is None:
            unknown += 1
            continue
        by_type[t["type"]] = by_type.get(t["type"], 0) + nb
        m = re.match(r"^blk\.(\d+)\.", t["name"])
        if m:
            idx = int(m.group(1))
            layers[idx] = layers.get(idx, 0) + nb
        else:
            other[t["name"]] = nb
    layer_rows = [{"layer": i, "gib": round(layers[i] / GIB, 4)} for i in sorted(layers)]
    layer_total = sum(layers.values())
    other_total = sum(other.values())
    return {
        "layers": layer_rows,
        "layer_count": len(layers),
        "layers_total_gib": round(layer_total / GIB, 4),
        "non_layer": {k: round(v / GIB, 4) for k, v in sorted(other.items())},
        "non_layer_total_gib": round(other_total / GIB, 4),
        "total_gib": round((layer_total + other_total) / GIB, 4),
        "by_type_gib": {k: round(v / GIB, 4) for k, v in sorted(by_type.items())},
        "tensors_unknown_size": unknown,
    }


# --------------------------------------------------------------------------
# C2. llama  (argv lists only, binary allowlist, one inference at a time)
# --------------------------------------------------------------------------

ALLOWED_BINS = ("llama-cli", "llama-bench", "llama-server", "llama-perplexity")
DEFAULT_LOCK = os.path.join(tempfile.gettempdir(), "legacy1070qi38.lock")


class InferenceLock:
    """Exclusive lock file so two benchmarks never run at once (RFC-002 item 5)."""

    def __init__(self, path=DEFAULT_LOCK):
        self.path = path
        self._held = False

    def _stale(self):
        try:
            with open(self.path, "r", encoding="ascii") as fh:
                pid = int(fh.read().strip())
        except (OSError, ValueError):
            return True
        if pid <= 0:
            return True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        return False

    def __enter__(self):
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                if self._stale():
                    with contextlib.suppress(FileNotFoundError):
                        os.remove(self.path)
                    continue
                raise CliError(f"inference lain sedang berjalan (lock: {self.path})")
            with os.fdopen(fd, "w", encoding="ascii") as fh:
                fh.write(str(os.getpid()))
            self._held = True
            return self
        raise CliError(f"tidak bisa mengambil lock: {self.path}")

    def __exit__(self, *exc):
        if self._held:
            with contextlib.suppress(FileNotFoundError):
                os.remove(self.path)
            self._held = False
        return False


def resolve_bin(bin_dir, name):
    if name not in ALLOWED_BINS:
        raise CliError(f"binary tidak diizinkan: {name} (allowlist: {', '.join(ALLOWED_BINS)})")
    path = os.path.join(bin_dir, name)
    if not (os.path.isfile(path) and os.access(path, os.X_OK)):
        raise EnvError(f"binary tidak ditemukan atau tidak executable: {path}")
    return path


def check_model(path):
    if not path.lower().endswith(".gguf"):
        raise CliError("model harus berekstensi .gguf")
    if not os.path.isfile(path):
        raise EnvError(f"model tidak ditemukan: {path}")
    return os.path.abspath(path)


def build_run_argv(bin_path, model, ctx=4096, ngl=None, prompt="", n_predict=64,
                   kv_type=None, single_turn=True):
    """argv for llama-cli. ngl=None leaves offload to llama.cpp auto-fit (OQ-014).

    single_turn menambah -st. Terbukti di build fork 08e025c: -no-cnv diabaikan dan llama-cli
    masuk mode percakapan tanpa akhir; -st keluar dengan kode 0 setelah satu jawaban.
    """
    _positive("ctx", ctx)
    _positive("n_predict", n_predict)
    argv = [bin_path, "-m", model, "-c", str(int(ctx)), "-n", str(int(n_predict)), "-p", prompt]
    if ngl is not None:
        _non_negative("ngl", ngl)
        argv += ["-ngl", str(int(ngl))]
    if kv_type:
        if kv_type not in KV_TYPE_FACTOR:
            raise CliError(f"kv_type tidak dikenal: {kv_type}")
        argv += ["-ctk", kv_type, "-ctv", kv_type]
    if single_turn:
        argv.append("-st")
    return argv


def build_bench_argv(bin_path, model, ngl=None, prompt_tokens=512, gen_tokens=128, reps=3):
    _positive("prompt_tokens", prompt_tokens)
    _positive("gen_tokens", gen_tokens)
    _positive("reps", reps)
    argv = [bin_path, "-m", model, "-p", str(int(prompt_tokens)), "-n", str(int(gen_tokens)),
            "-r", str(int(reps)), "-o", "json"]
    if ngl is not None:
        _non_negative("ngl", ngl)
        argv += ["-ngl", str(int(ngl))]
    return argv


def build_cmake_argv(src, build_dir="build", backend="cuda", jobs=None, targets=None):
    """Configure and build commands. CUDA arch 61 is Pascal (GTX 1070 Ti)."""
    flag = {"cuda": ["-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=61"],
            "vulkan": ["-DGGML_VULKAN=ON"],
            "cpu": ["-DGGML_CUDA=OFF", "-DGGML_VULKAN=OFF"]}.get(backend)
    if flag is None:
        raise CliError("backend harus cuda, vulkan, atau cpu")
    jobs = jobs or os.cpu_count() or 1
    _positive("jobs", jobs)
    configure = ["cmake", "-B", build_dir] + flag + ["-DCMAKE_BUILD_TYPE=Release"]
    build = ["cmake", "--build", build_dir, "--config", "Release", "-j", str(int(jobs))]
    if targets:
        for t in targets:
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", t):
                raise CliError(f"nama target tidak valid: {t!r}")
        build += ["--target"] + list(targets)
    return [configure, build]


_RE_EVAL = re.compile(r"^(?P<label>.*?eval time)\s*=.*?([\d.]+)\s+tokens per second", re.M)


_RE_BRACKET = re.compile(r"\[\s*Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]")


def parse_llama_timing(text):
    """Pull tokens/s from llama.cpp timing output. The format varies by version.

    Dua bentuk dikenali: baris `... eval time = ... tokens per second` (llama_perf) dan
    `[ Prompt: X t/s | Generation: Y t/s ]` (llama-cli -st di build fork, terukur).
    """
    out = {"prompt_tps": None, "gen_tps": None}
    brackets = _RE_BRACKET.findall(text)
    if brackets:
        out["prompt_tps"], out["gen_tps"] = float(brackets[-1][0]), float(brackets[-1][1])
        return out
    for m in _RE_EVAL.finditer(text):
        label = m.group("label")
        value = float(m.group(2))
        if "prompt eval time" in label:
            out["prompt_tps"] = value
        elif "eval time" in label:
            out["gen_tps"] = value
    return out


def parse_bench_json(text):
    """llama-bench -o json: a JSON list; each entry has avg_ts (tokens/s).

    ASSUMPTION: field names (n_prompt, n_gen, avg_ts, stddev_ts) match current llama-bench.
    """
    try:
        rows = json.loads(text)
    except ValueError:
        raise CliError("keluaran llama-bench bukan JSON valid")
    if not isinstance(rows, list):
        raise CliError("keluaran llama-bench tidak berbentuk list")
    result = []
    for r in rows:
        if not isinstance(r, dict) or "avg_ts" not in r:
            continue
        result.append({"n_prompt": r.get("n_prompt"), "n_gen": r.get("n_gen"),
                       "avg_ts": r.get("avg_ts"), "stddev_ts": r.get("stddev_ts")})
    return result


_RE_OFFLOAD = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers to GPU")
_RE_BUFFER = re.compile(r"([A-Za-z][\w.:-]*?)\s+model buffer size\s*=\s*([\d.]+)\s*MiB")


def parse_placement(text):
    """Read layer placement from a llama.cpp load log. Format differs per version."""
    offload = _RE_OFFLOAD.search(text)
    buffers = {}
    for m in _RE_BUFFER.finditer(text):
        buffers[m.group(1)] = buffers.get(m.group(1), 0.0) + float(m.group(2))
    if not offload and not buffers:
        raise CliError("tidak ada baris penempatan layer yang dikenali di log")
    return {
        "layers_on_gpu": int(offload.group(1)) if offload else None,
        "layers_total_reported": int(offload.group(2)) if offload else None,
        "buffers_mib": {k: round(v, 2) for k, v in buffers.items()},
        "note": "ASSUMPTION: pola baris log berdasarkan keluaran llama.cpp yang umum; "
                "total yang dilaporkan bisa menghitung output layer",
    }


def _run_logged(argv, timeout, cwd=None):
    try:
        # stdin ditutup: llama-cli tanpa -st menunggu input selamanya (terbukti di build fork).
        proc = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, universal_newlines=True, timeout=timeout,
                              shell=False, cwd=cwd)
    except subprocess.TimeoutExpired:
        raise CliError(f"proses melewati batas waktu {timeout}s dan dihentikan")
    except OSError as exc:
        raise EnvError(f"gagal menjalankan {argv[0]}: {exc}")
    return proc


# --------------------------------------------------------------------------
# C3. bench analyze / llama compare  (murni: hanya membaca hasil yang sudah ada)
# --------------------------------------------------------------------------

GTX_1070TI_BW_GBS = 256.0   # spesifikasi pabrik, bukan hasil ukur


def load_bench_rows(path):
    """Baca keluaran `llama bench --json` (dict dengan results) atau list mentah llama-bench."""
    if not os.path.isfile(path):
        raise EnvError(f"berkas tidak ditemukan: {path}")
    if os.path.getsize(path) > 10 * MIB:
        raise CliError(f"{path} lebih dari 10 MiB")
    with open(path, "r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except ValueError:
            raise CliError(f"{path} bukan JSON valid")
    rows = data.get("results") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise CliError(f"{path}: tidak ada daftar hasil (results)")
    return [r for r in rows if isinstance(r, dict) and isinstance(r.get("avg_ts"), (int, float))]


def decode_rows(rows):
    """Pilih tokens/s generasi dan prompt dari baris llama-bench."""
    gen = [r for r in rows if (r.get("n_gen") or 0) > 0 and not r.get("n_prompt")]
    prompt = [r for r in rows if (r.get("n_prompt") or 0) > 0 and not r.get("n_gen")]
    return {"gen_tps": gen[0]["avg_ts"] if gen else None,
            "prompt_tps": prompt[0]["avg_ts"] if prompt else None,
            "model_size_bytes": next((r["model_size"] for r in rows if r.get("model_size")), None),
            "n_gpu_layers": next((r["n_gpu_layers"] for r in rows
                                  if r.get("n_gpu_layers") is not None), None)}


def weights_read_per_token(g):
    """Byte bobot yang dibaca per token decode. token_embd hanya dibaca satu baris per token,
    jadi dikeluarkan bila model punya output.weight terpisah (bila di-tie, dipakai sebagai output)."""
    total = sum(t["nbytes"] for t in g["tensors"] if t["nbytes"] is not None)
    names = {t["name"]: t["nbytes"] for t in g["tensors"] if t["nbytes"] is not None}
    if "output.weight" in names and "token_embd.weight" in names:
        return total - names["token_embd.weight"], "total - token_embd (output.weight terpisah)"
    return total, "total (token_embd di-tie sebagai output)"


def analyze_bandwidth(gen_tps, read_bytes, gpu_fraction=1.0, gpu_bw_gbs=GTX_1070TI_BW_GBS,
                      ram_bw_gbs=None):
    """Bandingkan TPS terukur dengan batas atas dari bandwidth.

    Model: waktu per token = byte_GPU/bw_GPU + byte_CPU/bw_RAM (sisi GPU dan CPU berjalan
    berurutan per token pada layer yang berbeda). Itu batas atas; efisiensi = terukur / batas.
    """
    if not isinstance(gen_tps, (int, float)) or gen_tps <= 0:
        raise CliError("gen_tps harus > 0")
    if not read_bytes or read_bytes <= 0:
        raise CliError("jumlah byte bobot harus > 0")
    if not 0.0 <= gpu_fraction <= 1.0:
        raise CliError("--gpu-fraction harus 0..1")
    _positive("gpu_bw_gbs", gpu_bw_gbs)
    gpu_bytes, cpu_bytes = read_bytes * gpu_fraction, read_bytes * (1 - gpu_fraction)
    if cpu_bytes > 0 and not ram_bw_gbs:
        raise CliError("ada bobot di CPU: beri --ram-bw-gbs (ukur dengan `hw membw`)")
    t_token = gpu_bytes / (gpu_bw_gbs * 1e9) + (cpu_bytes / (ram_bw_gbs * 1e9) if cpu_bytes else 0.0)
    ceiling = 1.0 / t_token
    eff = gen_tps / ceiling
    if eff > 1.05:
        verdict = ("lewat batas yang diasumsikan: bandwidth yang diberikan terlalu rendah "
                   f"(bandwidth efektif minimal {read_bytes * gen_tps / 1e9:.1f} GB/s); "
                   "ukur ulang atau naikkan asumsi")
    elif eff >= 0.7:
        verdict = "dekat batas bandwidth: kurangi byte yang dibaca per token (quant lebih kecil, MTP)"
    elif eff >= 0.4:
        verdict = "sedang: ada ruang, tetapi bukan sekadar bandwidth; cek kernel dan transfer"
    else:
        verdict = ("jauh di bawah batas: kernel (dequant/DeltaNet), transfer PCIe, sinkronisasi, "
                   "throttle, atau KV/konteks besar menahan")
    return {
        "read_gib_per_token": round(read_bytes / GIB, 4),
        "gpu_fraction": gpu_fraction,
        "measured_gen_tps": gen_tps,
        "effective_bandwidth_gb_per_s": round(read_bytes * gen_tps / 1e9, 2),
        "ceiling_gen_tps": round(ceiling, 2),
        "efficiency": round(eff, 3),
        "gpu_bw_gb_per_s_assumed": gpu_bw_gbs,
        "ram_bw_gb_per_s_assumed": ram_bw_gbs,
        "verdict": verdict,
        "note": "efisiensi > 1 berarti asumsi bandwidth terlalu rendah atau bobot di-cache "
                "(L2/cache CPU); cek input sebelum percaya",
    }


def compare_bench(base_rows, cand_rows, tolerance_pct=5.0):
    """Cocokkan baris menurut (n_prompt, n_gen); beri rasio dan putusan dengan memperhitungkan stddev."""
    if tolerance_pct < 0:
        raise CliError("--tolerance-pct tidak boleh negatif")
    key = lambda r: (r.get("n_prompt") or 0, r.get("n_gen") or 0)  # noqa: E731
    base = {key(r): r for r in base_rows}
    out = []
    for r in cand_rows:
        k = key(r)
        if k not in base:
            continue
        b, c = base[k], r
        ratio = c["avg_ts"] / b["avg_ts"] if b["avg_ts"] else None
        noise = ((b.get("stddev_ts") or 0) ** 2 + (c.get("stddev_ts") or 0) ** 2) ** 0.5
        delta = c["avg_ts"] - b["avg_ts"]
        pct = (ratio - 1) * 100 if ratio is not None else None
        if pct is None or (abs(pct) <= tolerance_pct) or abs(delta) <= noise:
            verdict = "sama (dalam toleransi/noise)"
        else:
            verdict = "lebih cepat" if delta > 0 else "lebih lambat"
        out.append({"n_prompt": k[0], "n_gen": k[1], "base_tps": b["avg_ts"], "cand_tps": c["avg_ts"],
                    "ratio": None if ratio is None else round(ratio, 3),
                    "change_pct": None if pct is None else round(pct, 1), "verdict": verdict})
    if not out:
        raise CliError("tidak ada baris (n_prompt, n_gen) yang sama di kedua berkas")
    return out


# --------------------------------------------------------------------------
# C4. quality  (membungkus llama-perplexity dan llama-cli; protokol dicatat di keluaran)
# --------------------------------------------------------------------------

_PM = r"(?:\+/-|±)"   # llama-perplexity memakai "+/-" untuk PPL dan "±" untuk KLD (terukur di fork)
_RE_PPL = re.compile(r"Final estimate:\s*PPL\s*=\s*([\d.]+)\s*" + _PM + r"\s*([\d.]+)")
_RE_KLD = re.compile(r"Mean\s+KLD:\s*(-?[\d.]+)\s*" + _PM + r"\s*([\d.]+)")
_RE_KLD99 = re.compile(r"^99\.0%\s+KLD:\s*(-?[\d.]+)", re.M)
_RE_SAMETOP = re.compile(r"Same top p:\s*([\d.]+)\s*" + _PM + r"\s*([\d.]+)\s*%")


def parse_ppl(text):
    m = _RE_PPL.search(text)
    if not m:
        raise CliError("baris 'Final estimate: PPL = ...' tidak ditemukan di keluaran llama-perplexity")
    return {"ppl": float(m.group(1)), "ppl_stderr": float(m.group(2))}


def parse_kld(text):
    m = _RE_KLD.search(text)
    if not m:
        raise CliError("baris 'Mean KLD' tidak ditemukan di keluaran llama-perplexity")
    out = {"mean_kld": float(m.group(1)), "mean_kld_stderr": float(m.group(2))}
    t = _RE_SAMETOP.search(text)
    out["same_top_p_pct"] = float(t.group(1)) if t else None
    out["same_top_p_stderr"] = float(t.group(2)) if t else None
    q = _RE_KLD99.search(text)
    out["kld_p99"] = float(q.group(1)) if q else None
    return out


def build_ppl_argv(bin_path, model, text_file, ctx=512, chunks=None, ngl=None, threads=None,
                   extra=None, kv_type=None):
    _positive("ctx", ctx)
    argv = [bin_path, "-m", model, "-f", text_file, "-c", str(int(ctx))]
    if chunks is not None:
        _positive("chunks", chunks)
        argv += ["--chunks", str(int(chunks))]
    if ngl is not None:
        _non_negative("ngl", ngl)
        argv += ["-ngl", str(int(ngl))]
    if threads is not None:
        _positive("threads", threads)
        argv += ["-t", str(int(threads))]
    if kv_type:
        if kv_type not in KV_TYPE_FACTOR:
            raise CliError(f"kv_type tidak dikenal: {kv_type}")
        argv += ["-ctk", kv_type, "-ctv", kv_type]
    return argv + list(extra or [])


def text_similarity(a, b):
    import difflib
    return round(difflib.SequenceMatcher(None, a, b).ratio(), 4)


def compare_prompt_runs(base, cand):
    """base/cand: hasil `quality prompts`. Cocokkan menurut teks prompt."""
    cmap = {r["prompt"]: r for r in cand.get("results", [])}
    rows = []
    for r in base.get("results", []):
        c = cmap.get(r["prompt"])
        if c is None:
            continue
        rows.append({"prompt": r["prompt"][:60], "identical": r["output"] == c["output"],
                     "similarity": text_similarity(r["output"], c["output"])})
    if not rows:
        raise CliError("tidak ada prompt yang sama di kedua berkas")
    return {"prompts": len(rows), "identical": sum(1 for x in rows if x["identical"]),
            "mean_similarity": round(sum(x["similarity"] for x in rows) / len(rows), 4), "rows": rows}


def load_prompts(path):
    if not os.path.isfile(path):
        raise EnvError(f"berkas prompt tidak ditemukan: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        prompts = [ln.rstrip("\n") for ln in fh if ln.strip() and not ln.startswith("#")]
    if not prompts:
        raise CliError("berkas prompt kosong")
    if len(prompts) > 200:
        raise CliError("maksimum 200 prompt")
    return prompts


def strip_cli_noise(stdout):
    """Ambil jawaban dari keluaran llama-cli -st: buang banner, echo prompt, dan baris timing."""
    text = re.sub(r".\x08", "", stdout)       # animasi spinner llama-cli: "|\b-\b\\\b"
    text = text.replace("\x08", "")
    text = _RE_BRACKET.sub("", text)
    text = text.replace("Exiting...", "")
    if "\n> " in text:
        text = text.split("\n> ", 1)[1]
        text = text.split("\n", 1)[1] if "\n" in text else ""
    return text.strip()


# --------------------------------------------------------------------------
# D. spec  (writes only below <root>/spec)
# --------------------------------------------------------------------------

SPEC_DIR_RE = re.compile(r"^SPEC-(\d{3})-([a-z0-9][a-z0-9-]*)$")
SESSION_RE = re.compile(r"^SESSION-(\d{3})-([a-z0-9][a-z0-9-]*)\.md$")
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,60}$")
HANDOFF_TYPES = ("DECISION", "FINDING", "CONFLICT", "OOM", "UPDATE", "OPEN", "RESOLVED", "CORRECTION")
SPEC_FILES = ("requirements.md", "design.md", "tasks.md")


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _spec_dir(root):
    d = os.path.join(root, "spec")
    if not os.path.isdir(d):
        raise EnvError(f"folder spec/ tidak ada di {root}")
    return d


def _check_slug(slug):
    if not SLUG_RE.match(slug):
        raise CliError("nama harus huruf kecil, angka, dan tanda hubung (contoh: cli-tooling)")


def _next_number(names, regex):
    nums = [int(m.group(1)) for m in (regex.match(n) for n in names) if m]
    return (max(nums) + 1) if nums else 0


def scaffold_spec(root, slug, title=None):
    """Create spec/specs/SPEC-NNN-slug/ from the template. Never overwrites."""
    _check_slug(slug)
    specs = os.path.join(_spec_dir(root), "specs")
    tmpl = os.path.join(specs, "_TEMPLATE-feature-spec")
    if not os.path.isdir(tmpl):
        raise EnvError("template _TEMPLATE-feature-spec tidak ditemukan")
    number = _next_number(os.listdir(specs), re.compile(r"^SPEC-(\d{3})-"))
    spec_id = f"SPEC-{number:03d}"
    target = os.path.join(specs, f"{spec_id}-{slug}")
    title = title or slug.replace("-", " ").title()
    os.makedirs(target)  # raises FileExistsError if it somehow exists; never clobbers
    created = []
    for fname in SPEC_FILES:
        with open(os.path.join(tmpl, fname), "r", encoding="utf-8") as fh:
            text = fh.read()
        text = text.replace("SPEC-NNN", spec_id).replace("[NAMA FITUR]", title)
        with open(os.path.join(target, fname), "x", encoding="utf-8") as out:
            out.write(text)
        created.append(os.path.relpath(os.path.join(target, fname), root))
    return {"spec_id": spec_id, "dir": os.path.relpath(target, root), "files": created}


def scaffold_session(root, slug, title=None, on=None):
    _check_slug(slug)
    handoff = os.path.join(_spec_dir(root), "handoff")
    tmpl = os.path.join(handoff, "_TEMPLATE-session.md")
    if not os.path.isfile(tmpl):
        raise EnvError("template _TEMPLATE-session.md tidak ditemukan")
    number = _next_number(os.listdir(handoff), re.compile(r"^SESSION-(\d{3})-"))
    sid = f"SESSION-{number:03d}"
    with open(tmpl, "r", encoding="utf-8") as fh:
        text = fh.read()
    text = text.replace("SESSION-NNN", sid, 1)
    text = text.replace("[Deskripsi Singkat Sesi]", title or slug.replace("-", " "), 1)
    text = text.replace("> Tanggal: [YYYY-MM-DD]", f"> Tanggal: {on or date.today().isoformat()}", 1)
    path = os.path.join(handoff, f"{sid}-{slug}.md")
    with open(path, "x", encoding="utf-8") as out:
        out.write(text)
    return {"session_id": sid, "file": os.path.relpath(path, root)}


def append_handoff(root, entry_type, text, session=None, on=None):
    """Append one entry to HANDOFF-LOG.md. The log is append-only: mode 'a' only."""
    entry_type = entry_type.upper()
    if entry_type not in HANDOFF_TYPES:
        raise CliError(f"tipe tidak dikenal: {entry_type} (pilihan: {', '.join(HANDOFF_TYPES)})")
    text = text.strip()
    if not text or len(text) > 2000:
        raise CliError("teks harus 1-2000 karakter")
    if "\n" in text or "\r" in text or "```" in text:
        raise CliError("teks harus satu baris dan tanpa ``` (merusak blok kode log)")
    if session is not None and not re.match(r"^SESSION-\d{3}$", session):
        raise CliError("--session harus berformat SESSION-NNN")
    log = os.path.join(_spec_dir(root), "handoff", "HANDOFF-LOG.md")
    if not os.path.isfile(log):
        raise EnvError("HANDOFF-LOG.md tidak ditemukan")
    label = f"[{on or date.today().isoformat()}]" + (f" [{session}]" if session else "") + f" [{entry_type}]"
    block = f"\n```\n{label} {text}\n```\n"
    with open(log, "rb") as fh:
        content = fh.read()
    prefix = "" if content.endswith(b"\n") else "\n"
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(prefix + block)
    return {"file": os.path.relpath(log, root), "entry": f"{label} {text}"}


_RE_OQ_ID = re.compile(r"OQ-\d{3}")
_RE_RESOLVED_LEAD = re.compile(r"\[RESOLVED\]\s*((?:OQ-\d{3}(?:\s*,\s*|\s+dan\s+|\s+)?)+)")


def parse_questions(text):
    """List open questions from HANDOFF-LOG.md.

    Sources: the 'Open Questions Aktif' table, `[OPEN] OQ-NNN ...` lines, and
    `[RESOLVED] OQ-NNN, OQ-NNN ...` lines (only the leading id list counts).
    """
    questions = {}
    for line in text.splitlines():
        row = re.match(r"^\|\s*(OQ-\d{3})\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*$", line)
        if row:
            questions[row.group(1)] = {"id": row.group(1), "question": row.group(2),
                                       "status": "open", "source": "table"}
            continue
        op = re.search(r"\[OPEN\]\s*(OQ-\d{3})\s+(.*)$", line)
        if op:
            questions.setdefault(op.group(1), {"id": op.group(1), "question": op.group(2).strip(),
                                               "status": "open", "source": "log"})
    resolved = set()
    for line in text.splitlines():
        m = _RE_RESOLVED_LEAD.search(line)
        if m:
            resolved.update(_RE_OQ_ID.findall(m.group(1)))
    for oq in resolved:
        if oq in questions:
            questions[oq]["status"] = "resolved"
    return [questions[k] for k in sorted(questions)]


_RE_FENCE = re.compile(r"^\s*(```|~~~)")
_RE_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
_RE_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def check_links(md_path):
    """Return a list of broken relative links in one markdown file."""
    problems = []
    in_fence = False
    with open(md_path, "r", encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            if _RE_FENCE.match(line):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            clean = re.sub(r"`[^`]*`", "", line)
            for m in _RE_LINK.finditer(clean):
                pieces = m.group(1).strip().split()
                target = pieces[0].strip("<>") if pieces else ""
                if not target or target.startswith(("#", "/")) or _RE_SCHEME.match(target):
                    continue
                target = re.split(r"[#?]", target, 1)[0]
                if not target:
                    continue
                full = os.path.normpath(os.path.join(os.path.dirname(md_path), target))
                if not os.path.exists(full):
                    problems.append({"file": md_path, "line": lineno, "target": target})
    return problems


def check_numbering(text, prefix):
    """FR-NN / NFR-NN list items must run 01, 02, 03... without gaps or repeats."""
    found = [int(n) for n in re.findall(r"^\s*-\s+\*\*" + prefix + r"-(\d+)", text, re.M)]
    expected = list(range(1, len(found) + 1))
    return None if found == expected else {"prefix": prefix, "found": found, "expected": expected}


def validate_spec(root):
    spec = _spec_dir(root)
    errors = []
    warnings = []
    specs = os.path.join(spec, "specs")
    if os.path.isdir(specs):
        for name in sorted(os.listdir(specs)):
            d = os.path.join(specs, name)
            if not os.path.isdir(d) or not SPEC_DIR_RE.match(name):
                continue
            for fname in SPEC_FILES:
                if not os.path.isfile(os.path.join(d, fname)):
                    errors.append(f"{name}: file {fname} tidak ada")
            req = os.path.join(d, "requirements.md")
            if os.path.isfile(req):
                with open(req, "r", encoding="utf-8", errors="replace") as fh:
                    body = fh.read()
                for prefix in ("FR", "NFR"):
                    bad = check_numbering(body, prefix)
                    if bad:
                        errors.append(f"{name}/requirements.md: penomoran {prefix} tidak berurutan "
                                      f"(ada {bad['found']}, seharusnya {bad['expected']})")
    for dirpath, _dirs, files in sorted(os.walk(spec)):
        for fname in sorted(files):
            if fname.endswith(".md"):
                for p in check_links(os.path.join(dirpath, fname)):
                    rel = os.path.relpath(p["file"], root)
                    errors.append(f"{rel}:{p['line']}: tautan rusak -> {p['target']}")
    log = os.path.join(spec, "handoff", "HANDOFF-LOG.md")
    if os.path.isfile(log):
        with open(log, "r", encoding="utf-8", errors="replace") as fh:
            seen = {}
            for line in fh:
                row = re.match(r"^\|\s*(OQ-\d{3})\s*\|", line)
                if row:
                    seen[row.group(1)] = seen.get(row.group(1), 0) + 1
        for oq, n in sorted(seen.items()):
            if n > 1:
                warnings.append(f"HANDOFF-LOG.md: {oq} muncul {n}x di tabel")
    else:
        warnings.append("HANDOFF-LOG.md tidak ditemukan")
    return {"ok": not errors, "errors": errors, "warnings": warnings}


# --------------------------------------------------------------------------
# CLI layer (thin)
# --------------------------------------------------------------------------

def render(data, indent=0):
    pad = "  " * indent
    out = []
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (dict, list)) and v:
                out.append(f"{pad}{k}:")
                out.append(render(v, indent + 1))
            else:
                out.append(f"{pad}{k}: {'-' if v is None or v == [] or v == {} else v}")
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                out.append(f"{pad}-")
                out.append(render(item, indent + 1))
            else:
                out.append(f"{pad}- {item}")
    else:
        out.append(f"{pad}{data}")
    return "\n".join(out)


def _budget_kwargs(a):
    return dict(vram_gib=a.vram_gib, ram_gib=a.ram_gib, ctx=a.ctx, kv_layers=a.kv_layers,
                kv_heads=a.kv_heads, head_dim=a.head_dim, kv_type=a.kv_type,
                cuda_ctx_gib=a.cuda_ctx_gib, compute_gib=a.compute_gib, os_gib=a.os_gib,
                ram_bw_gbs=a.ram_bw_gbs, gpu_bw_gbs=a.gpu_bw_gbs, rs_gib=a.rs_gib)


def cmd_budget_estimate(a):
    return estimate(a.file_gib, **_budget_kwargs(a)), EXIT_OK


def cmd_budget_ngl(a):
    return suggest_ngl(a.file_gib, layers=a.layers, **_budget_kwargs(a)), EXIT_OK


def cmd_budget_compare(a):
    est = estimate(a.file_gib, **_budget_kwargs(a))
    res = compare_measured(est, a.measured_vram_gib, a.measured_ram_gib, a.measured_tps, a.tolerance_pct)
    return res, EXIT_OK if res["all_within"] else EXIT_CHECK_FAILED


def cmd_hw_gpu(a):
    return {"gpus": read_gpu()}, EXIT_OK


def cmd_hw_ram(a):
    return read_ram(), EXIT_OK


def cmd_hw_cpu(a):
    return read_cpu(), EXIT_OK


def cmd_hw_membw(a):
    if a.threads == 0:
        return measure_membw_sweep(a.size_mib, a.repeats), EXIT_OK
    return measure_membw(a.size_mib, a.repeats, a.threads), EXIT_OK


def cmd_hw_faults(a):
    _positive("seconds", a.seconds)
    if a.seconds > 3600:
        raise CliError("--seconds maksimum 3600")
    before = read_vmstat()
    time.sleep(a.seconds)
    return vmstat_delta(before, read_vmstat(), a.seconds), EXIT_OK


def cmd_hw_sample(a):
    if not 1 <= a.count <= 3600:
        raise CliError("--count harus 1..3600")
    if not 0.1 <= a.interval <= 60:
        raise CliError("--interval harus 0.1..60")
    samples = []
    for i in range(a.count):
        g = read_gpu()[0]
        samples.append({k: g.get(k) for k in ("vram_used_mib", "temp_c", "power_w", "util_pct",
                                              "sm_clock_mhz", "throttle_reasons",
                                              "throttle_limiting")})
        if i + 1 < a.count:
            time.sleep(a.interval)
    res = summarize_samples(samples)
    res["gpu_index"] = 0
    return res, EXIT_OK


def cmd_hw_headroom(a):
    if a.need_vram_gib is None or a.need_ram_gib is None:
        if a.file_gib is None:
            raise CliError("beri --file-gib, atau keduanya: --need-vram-gib dan --need-ram-gib")
        est = estimate(a.file_gib, ctx=a.ctx)
        need_vram = est["vram"]["used_gib"] if a.need_vram_gib is None else a.need_vram_gib
        # Available RAM already excludes the OS, so only model weights count here.
        need_ram = est["ram"]["weights_gib"] if a.need_ram_gib is None else a.need_ram_gib
    else:
        need_vram, need_ram = a.need_vram_gib, a.need_ram_gib
    free_vram = None
    if not a.no_gpu:
        free_vram = read_gpu()[0]["vram_free_gib"]
    res = headroom_check(free_vram, read_ram()["available_gib"], need_vram, need_ram,
                         a.vram_margin_gib, a.ram_margin_gib)
    return res, EXIT_OK if res["go"] else EXIT_CHECK_FAILED


def cmd_gguf_inspect(a):
    return summarize_gguf(parse_gguf(a.path)), EXIT_OK


def cmd_gguf_tensors(a):
    return summarize_tensors(parse_gguf(a.path)), EXIT_OK


def cmd_llama_build(a):
    src = os.path.abspath(a.src)
    if not os.path.isfile(os.path.join(src, "CMakeLists.txt")):
        raise EnvError(f"CMakeLists.txt tidak ada di {src}")
    cmds = build_cmake_argv(src, a.build_dir, a.backend, a.jobs, a.target)
    if a.dry_run:
        return {"dry_run": True, "cwd": src, "commands": cmds}, EXIT_OK
    for argv in cmds:
        proc = _run_logged(argv, a.timeout, cwd=src)
        if proc.returncode != 0:
            return {"failed": argv, "returncode": proc.returncode,
                    "stderr_tail": proc.stderr[-1500:]}, EXIT_CHECK_FAILED
    return {"built": True, "cwd": src, "commands": cmds}, EXIT_OK


def cmd_llama_run(a):
    model = check_model(a.model)
    argv = build_run_argv(resolve_bin(a.bin_dir, "llama-cli") if not a.dry_run else
                          os.path.join(a.bin_dir, "llama-cli"),
                          model, a.ctx, a.ngl, a.prompt, a.n_predict, a.kv_type, not a.conversation)
    if a.dry_run:
        return {"dry_run": True, "argv": argv}, EXIT_OK
    with InferenceLock(a.lock_file):
        proc = _run_logged(argv, a.timeout)
    combined = proc.stdout + "\n" + proc.stderr
    res = {"returncode": proc.returncode, "argv": argv, **parse_llama_timing(combined),
           "stdout_tail": proc.stdout[-800:], "stderr_tail": proc.stderr[-800:]}
    try:
        res["placement"] = parse_placement(combined)
    except CliError:
        res["placement"] = None
    return res, EXIT_OK if proc.returncode == 0 else EXIT_CHECK_FAILED


def cmd_llama_bench(a):
    model = check_model(a.model)
    binp = os.path.join(a.bin_dir, "llama-bench") if a.dry_run else resolve_bin(a.bin_dir, "llama-bench")
    argv = build_bench_argv(binp, model, a.ngl, a.prompt_tokens, a.gen_tokens, a.runs)
    if a.dry_run:
        return {"dry_run": True, "argv": argv}, EXIT_OK
    with InferenceLock(a.lock_file):
        proc = _run_logged(argv, a.timeout)
    if proc.returncode != 0:
        return {"returncode": proc.returncode, "stderr_tail": proc.stderr[-1500:]}, EXIT_CHECK_FAILED
    return {"argv": argv, "results": parse_bench_json(proc.stdout)}, EXIT_OK


def cmd_llama_compare(a):
    base = load_bench_rows(a.base)
    cand = load_bench_rows(a.cand)
    return {"base": a.base, "cand": a.cand, "tolerance_pct": a.tolerance_pct,
            "rows": compare_bench(base, cand, a.tolerance_pct)}, EXIT_OK


def cmd_bench_analyze(a):
    rows = decode_rows(load_bench_rows(a.bench_json)) if a.bench_json else {}
    gen_tps = a.gen_tps if a.gen_tps is not None else rows.get("gen_tps")
    if gen_tps is None:
        raise CliError("beri --bench-json yang berisi baris generasi atau --gen-tps")
    if a.gguf:
        g = parse_gguf(a.gguf)
        read_bytes, how = weights_read_per_token(g)
    elif a.weights_gib:
        read_bytes, how = a.weights_gib * GIB, "diberikan lewat --weights-gib"
    elif rows.get("model_size_bytes"):
        read_bytes, how = rows["model_size_bytes"], "model_size dari llama-bench"
    else:
        raise CliError("beri --gguf, --weights-gib, atau bench-json yang memuat model_size")
    res = analyze_bandwidth(gen_tps, read_bytes, a.gpu_fraction, a.gpu_bw_gbs, a.ram_bw_gbs)
    res["bytes_source"] = how
    if a.sample_json:
        with open(a.sample_json, "r", encoding="utf-8") as fh:
            smp = json.load(fh)
        res["gpu_throttle"] = {"reasons_seen": smp.get("throttle_reasons_seen"),
                               "samples_throttled": smp.get("samples_throttled"),
                               "min_sm_clock_mhz_when_busy": smp.get("min_sm_clock_mhz_when_busy")}
        if smp.get("samples_throttled"):
            res["verdict"] += "; GPU di-throttle selama sampel, perbaiki itu dulu"
    return res, EXIT_OK


def _quality_bin(a, name):
    return os.path.join(a.bin_dir, name) if a.dry_run else resolve_bin(a.bin_dir, name)


def cmd_quality_ppl(a):
    model = check_model(a.model)
    if not os.path.isfile(a.file):
        raise EnvError(f"berkas teks tidak ditemukan: {a.file}")
    argv = build_ppl_argv(_quality_bin(a, "llama-perplexity"), model, os.path.abspath(a.file),
                          a.ctx, a.chunks, a.ngl, a.threads, kv_type=a.kv_type)
    protocol = {"file": os.path.abspath(a.file), "ctx": a.ctx, "chunks": a.chunks, "ngl": a.ngl,
                "kv_type": a.kv_type}
    if a.dry_run:
        return {"dry_run": True, "argv": argv, "protocol": protocol}, EXIT_OK
    with InferenceLock(a.lock_file):
        proc = _run_logged(argv, a.timeout)
    combined = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        return {"returncode": proc.returncode, "stderr_tail": proc.stderr[-1500:]}, EXIT_CHECK_FAILED
    return {"model": model, "protocol": protocol, **parse_ppl(combined)}, EXIT_OK


def cmd_quality_kld(a):
    base = check_model(a.base_model)
    cand = check_model(a.model)
    if not os.path.isfile(a.file):
        raise EnvError(f"berkas teks tidak ditemukan: {a.file}")
    binp = _quality_bin(a, "llama-perplexity")
    text = os.path.abspath(a.file)
    # --base-kv-type: KV cache model referensi; --kv-type: KV cache kandidat (mengukur dampak KV quant).
    step1 = build_ppl_argv(binp, base, text, a.ctx, a.chunks, a.ngl, a.threads,
                           ["--kl-divergence-base", a.kld_file], kv_type=a.base_kv_type)
    step2 = build_ppl_argv(binp, cand, text, a.ctx, a.chunks, a.ngl, a.threads,
                           ["--kl-divergence", "--kl-divergence-base", a.kld_file], kv_type=a.kv_type)
    protocol = {"file": text, "ctx": a.ctx, "chunks": a.chunks, "ngl": a.ngl, "kld_file": a.kld_file,
                "base_kv_type": a.base_kv_type, "kv_type": a.kv_type}
    if a.dry_run:
        return {"dry_run": True, "argv": [step1, step2], "protocol": protocol}, EXIT_OK
    with InferenceLock(a.lock_file):
        if not (a.reuse_base and os.path.isfile(a.kld_file)):
            p1 = _run_logged(step1, a.timeout)
            if p1.returncode != 0:
                return {"step": "base", "returncode": p1.returncode,
                        "stderr_tail": p1.stderr[-1500:]}, EXIT_CHECK_FAILED
        p2 = _run_logged(step2, a.timeout)
    if p2.returncode != 0:
        return {"step": "candidate", "returncode": p2.returncode,
                "stderr_tail": p2.stderr[-1500:]}, EXIT_CHECK_FAILED
    res = parse_kld(p2.stdout + "\n" + p2.stderr)
    with contextlib.suppress(CliError):
        res.update(parse_ppl(p2.stdout + "\n" + p2.stderr))
    kld_gib = round(os.path.getsize(a.kld_file) / GIB, 3) if os.path.isfile(a.kld_file) else None
    return {"base_model": base, "model": cand, "protocol": protocol, "kld_file_gib": kld_gib,
            **res}, EXIT_OK


def cmd_quality_prompts(a):
    if a.compare:
        loaded = []
        for path in a.compare:
            if not os.path.isfile(path):
                raise EnvError(f"berkas tidak ditemukan: {path}")
            with open(path, "r", encoding="utf-8") as fh:
                loaded.append(json.load(fh))
        return compare_prompt_runs(loaded[0], loaded[1]), EXIT_OK
    if not (a.model and a.prompts and a.out):
        raise CliError("jalankan dengan --model, --prompts, --out; atau --compare A.json B.json")
    model = check_model(a.model)
    prompts = load_prompts(a.prompts)
    binp = _quality_bin(a, "llama-cli")
    fixed = ["--temp", "0", "--seed", "1"]
    argvs = [build_run_argv(binp, model, a.ctx, a.ngl, p, a.n_predict) + fixed for p in prompts]
    if a.dry_run:
        return {"dry_run": True, "prompts": len(prompts), "argv_first": argvs[0]}, EXIT_OK
    results = []
    with InferenceLock(a.lock_file):
        for p, argv in zip(prompts, argvs):
            proc = _run_logged(argv, a.timeout)
            if proc.returncode != 0:
                return {"prompt": p[:60], "returncode": proc.returncode,
                        "stderr_tail": proc.stderr[-800:]}, EXIT_CHECK_FAILED
            results.append({"prompt": p, "output": strip_cli_noise(proc.stdout),
                            **parse_llama_timing(proc.stdout + proc.stderr)})
    doc = {"model": model, "protocol": {"temp": 0, "seed": 1, "ctx": a.ctx, "n_predict": a.n_predict,
                                         "ngl": a.ngl}, "results": results}
    out = os.path.abspath(a.out)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    return {"saved": out, "prompts": len(results)}, EXIT_OK


def cmd_llama_placement(a):
    if not os.path.isfile(a.log):
        raise EnvError(f"log tidak ditemukan: {a.log}")
    if os.path.getsize(a.log) > 50 * MIB:
        raise CliError("log lebih dari 50 MiB")
    with open(a.log, "r", encoding="utf-8", errors="replace") as fh:
        return parse_placement(fh.read()), EXIT_OK


def cmd_spec_new(a):
    return scaffold_spec(a.root, a.name, a.title), EXIT_OK


def cmd_spec_session(a):
    return scaffold_session(a.root, a.name, a.title, a.date), EXIT_OK


def cmd_spec_handoff(a):
    return append_handoff(a.root, a.type, a.text, a.session, a.date), EXIT_OK


def cmd_spec_questions(a):
    log = os.path.join(_spec_dir(a.root), "handoff", "HANDOFF-LOG.md")
    if not os.path.isfile(log):
        raise EnvError("HANDOFF-LOG.md tidak ditemukan")
    with open(log, "r", encoding="utf-8", errors="replace") as fh:
        qs = parse_questions(fh.read())
    if not a.all:
        qs = [q for q in qs if q["status"] == "open"]
    return {"count": len(qs), "questions": qs}, EXIT_OK


def cmd_spec_validate(a):
    res = validate_spec(a.root)
    return res, EXIT_OK if res["ok"] else EXIT_CHECK_FAILED


def _add_budget_args(p):
    p.add_argument("--file-gib", type=float, required=True, help="ukuran file GGUF (GiB)")
    p.add_argument("--vram-gib", type=float, default=8.0)
    p.add_argument("--ram-gib", type=float, default=16.0)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--kv-layers", type=int, default=DEFAULT_KV_LAYERS, help="layer dengan KV cache")
    p.add_argument("--kv-heads", type=int, default=DEFAULT_KV_HEADS)
    p.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    p.add_argument("--kv-type", choices=sorted(KV_TYPE_FACTOR), default="f16")
    p.add_argument("--cuda-ctx-gib", type=float, default=0.4)
    p.add_argument("--compute-gib", type=float, default=0.5)
    p.add_argument("--os-gib", type=float, default=2.5)
    p.add_argument("--rs-gib", type=float, default=DEFAULT_RS_GIB,
                   help="state rekuren Gated DeltaNet (0 untuk model non-hibrida)")
    p.add_argument("--ram-bw-gbs", type=float, default=20.0, help="bandwidth RAM; ukur dengan hw membw")
    p.add_argument("--gpu-bw-gbs", type=float, default=256.0)


def _add_llama_common(p):
    p.add_argument("--model", required=True)
    p.add_argument("--bin-dir", default=os.path.join("build", "bin"))
    p.add_argument("--ngl", type=int, default=None, help="kosongkan agar llama.cpp auto-fit")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--lock-file", default=DEFAULT_LOCK)
    p.add_argument("--dry-run", action="store_true", help="tampilkan argv tanpa menjalankan")


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="keluaran JSON")
    spec_common = argparse.ArgumentParser(add_help=False, parents=[common])
    spec_common.add_argument("--root", default=repo_root(), help="root repo (default: otomatis)")

    parser = argparse.ArgumentParser(prog="legacy.py", description="CLI tooling legacy1070qi38")
    groups = parser.add_subparsers(dest="group", required=True)

    def leaf(sub, name, func, help_, parents=(common,)):
        p = sub.add_parser(name, help=help_, parents=list(parents))
        p.set_defaults(func=func)
        return p

    g = groups.add_parser("budget", help="hitung memory budget (tanpa hardware)").add_subparsers(
        dest="cmd", required=True)
    _add_budget_args(leaf(g, "estimate", cmd_budget_estimate, "estimasi VRAM, RAM, TPS"))
    p = leaf(g, "suggest-ngl", cmd_budget_ngl, "jumlah layer di GPU")
    _add_budget_args(p)
    p.add_argument("--layers", type=int, default=DEFAULT_TOTAL_LAYERS)
    p = leaf(g, "compare", cmd_budget_compare, "bandingkan estimasi dengan hasil ukur")
    _add_budget_args(p)
    p.add_argument("--measured-vram-gib", type=float)
    p.add_argument("--measured-ram-gib", type=float)
    p.add_argument("--measured-tps", type=float)
    p.add_argument("--tolerance-pct", type=float, default=15.0)

    g = groups.add_parser("hw", help="pantau hardware").add_subparsers(dest="cmd", required=True)
    leaf(g, "gpu", cmd_hw_gpu, "VRAM, suhu, power (nvidia-smi)")
    leaf(g, "ram", cmd_hw_ram, "RAM dan swap")
    leaf(g, "cpu", cmd_hw_cpu, "model CPU, core, AVX2")
    p = leaf(g, "membw", cmd_hw_membw, "bandwidth RAM kasar")
    p.add_argument("--size-mib", type=int, default=256)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--threads", type=int, default=0, help="0 = coba 1,2,4,.. thread dan ambil terbaik")
    p = leaf(g, "faults", cmd_hw_faults, "page fault berat dan swap")
    p.add_argument("--seconds", type=float, default=5.0)
    p = leaf(g, "sample", cmd_hw_sample, "sampling puncak VRAM/suhu/power")
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--interval", type=float, default=1.0)
    p = leaf(g, "headroom", cmd_hw_headroom, "go/no-go (exit 1 bila tidak cukup)")
    p.add_argument("--file-gib", type=float)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--need-vram-gib", type=float)
    p.add_argument("--need-ram-gib", type=float)
    p.add_argument("--vram-margin-gib", type=float, default=0.2)
    p.add_argument("--ram-margin-gib", type=float, default=0.5)
    p.add_argument("--no-gpu", action="store_true", help="lewati cek VRAM")

    g = groups.add_parser("gguf", help="baca metadata GGUF").add_subparsers(dest="cmd", required=True)
    leaf(g, "inspect", cmd_gguf_inspect, "metadata ringkas").add_argument("path")
    leaf(g, "tensors", cmd_gguf_tensors, "ukuran tensor per layer").add_argument("path")

    g = groups.add_parser("llama", help="build/jalankan llama.cpp").add_subparsers(dest="cmd", required=True)
    p = leaf(g, "build", cmd_llama_build, "build llama.cpp (CUDA SM 61)")
    p.add_argument("--src", required=True)
    p.add_argument("--build-dir", default="build")
    p.add_argument("--backend", choices=("cuda", "vulkan", "cpu"), default="cuda")
    p.add_argument("--target", action="append", help="target cmake (boleh diulang), mis. llama-perplexity")
    p.add_argument("--jobs", type=int)
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--dry-run", action="store_true")
    p = leaf(g, "run", cmd_llama_run, "jalankan llama-cli satu kali")
    _add_llama_common(p)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--prompt", required=True)
    p.add_argument("--n-predict", type=int, default=64)
    p.add_argument("--kv-type", choices=sorted(KV_TYPE_FACTOR))
    p.add_argument("--conversation", action="store_true",
                   help="jangan tambah -st (hanya untuk build llama-cli lama yang tidak punya -st)")
    p = leaf(g, "bench", cmd_llama_bench, "llama-bench dengan beberapa ulangan")
    _add_llama_common(p)
    p.add_argument("--prompt-tokens", type=int, default=512)
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument("--runs", type=int, default=3)
    leaf(g, "placement", cmd_llama_placement, "baca log: layer di GPU/CPU").add_argument("--log", required=True)
    p = leaf(g, "compare", cmd_llama_compare, "bandingkan dua hasil `llama bench --json`")
    p.add_argument("base")
    p.add_argument("cand")
    p.add_argument("--tolerance-pct", type=float, default=5.0)

    g = groups.add_parser("bench", help="analisis hasil benchmark").add_subparsers(dest="cmd", required=True)
    p = leaf(g, "analyze", cmd_bench_analyze, "efisiensi bandwidth: GPU, CPU, atau lainnya yang menahan")
    p.add_argument("--bench-json", help="keluaran `llama bench --json` (berisi baris generasi)")
    p.add_argument("--gen-tps", type=float, help="TPS generasi terukur, bila tanpa bench-json")
    p.add_argument("--gguf", help="hitung byte bobot per token dari header GGUF")
    p.add_argument("--weights-gib", type=float)
    p.add_argument("--gpu-fraction", type=float, default=1.0, help="bagian bobot di GPU, 0..1")
    p.add_argument("--gpu-bw-gbs", type=float, default=GTX_1070TI_BW_GBS)
    p.add_argument("--ram-bw-gbs", type=float, help="dari `hw membw` (wajib bila ada bobot di CPU)")
    p.add_argument("--sample-json", help="keluaran `hw sample --json` untuk cek throttle")

    g = groups.add_parser("quality", help="ukur kualitas, bukan hanya kecepatan").add_subparsers(
        dest="cmd", required=True)

    def _quality_common(p):
        p.add_argument("--bin-dir", default=os.path.join("build", "bin"))
        p.add_argument("--ngl", type=int, default=None)
        p.add_argument("--ctx", type=int, default=512)
        p.add_argument("--timeout", type=int, default=3600)
        p.add_argument("--lock-file", default=DEFAULT_LOCK)
        p.add_argument("--dry-run", action="store_true")

    p = leaf(g, "ppl", cmd_quality_ppl, "perplexity pada teks tetap (llama-perplexity)")
    _quality_common(p)
    p.add_argument("--model", required=True)
    p.add_argument("--file", required=True, help="teks evaluasi, mis. wikitext-2 raw test")
    p.add_argument("--chunks", type=int, default=None, help="batasi jumlah chunk agar cepat")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--kv-type", choices=sorted(KV_TYPE_FACTOR), default=None, help="tipe KV cache (-ctk/-ctv)")
    p = leaf(g, "kld", cmd_quality_kld, "KL divergence kandidat terhadap model referensi")
    _quality_common(p)
    p.add_argument("--base-model", required=True, help="quant referensi (tertinggi yang muat)")
    p.add_argument("--model", required=True, help="quant yang diuji")
    p.add_argument("--file", required=True)
    p.add_argument("--chunks", type=int, default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--kv-type", choices=sorted(KV_TYPE_FACTOR), default=None,
                   help="tipe KV cache kandidat (-ctk/-ctv)")
    p.add_argument("--base-kv-type", choices=sorted(KV_TYPE_FACTOR), default=None,
                   help="tipe KV cache referensi")
    p.add_argument("--kld-file", required=True, help="berkas logit referensi (BESAR; lihat docs)")
    p.add_argument("--reuse-base", action="store_true", help="pakai ulang --kld-file yang sudah ada")
    p = leaf(g, "prompts", cmd_quality_prompts, "jalankan set prompt tetap, simpan/bandingkan jawaban")
    _quality_common(p)
    p.add_argument("--model")
    p.add_argument("--prompts", help="satu prompt per baris; # = komentar")
    p.add_argument("--out", help="berkas JSON hasil (tidak menimpa)")
    p.add_argument("--n-predict", type=int, default=128)
    p.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))

    g = groups.add_parser("spec", help="rawat dokumen spec").add_subparsers(dest="cmd", required=True)
    p = leaf(g, "new", cmd_spec_new, "scaffold SPEC-NNN dari template", (spec_common,))
    p.add_argument("name")
    p.add_argument("--title")
    p = leaf(g, "session", cmd_spec_session, "buat SESSION-NNN dari template", (spec_common,))
    p.add_argument("name")
    p.add_argument("--title")
    p.add_argument("--date")
    p = leaf(g, "handoff", cmd_spec_handoff, "tambah entri ke HANDOFF-LOG (append-only)", (spec_common,))
    p.add_argument("--type", required=True, choices=[t.lower() for t in HANDOFF_TYPES] + list(HANDOFF_TYPES))
    p.add_argument("--text", required=True)
    p.add_argument("--session")
    p.add_argument("--date")
    p = leaf(g, "questions", cmd_spec_questions, "daftar open question", (spec_common,))
    p.add_argument("--all", action="store_true")
    leaf(g, "validate", cmd_spec_validate, "cek konsistensi dokumen (exit 1 bila ada error)", (spec_common,))
    return parser


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        data, code = args.func(args)
    except CliError as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    except FileExistsError as exc:
        msg = f"file sudah ada, tidak ditimpa: {exc.filename}"
        if args.json:
            print(json.dumps({"ok": False, "error": msg}, ensure_ascii=False))
        print(f"error: {msg}", file=sys.stderr)
        return EXIT_BAD_INPUT
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(render(data))
    return code


if __name__ == "__main__":
    sys.exit(main())
