#!/usr/bin/env python3
"""Satu uji llama-bench pada kedalaman konteks 4096 dengan sampel RAM/VRAM tiap detik.

Contoh:
  ctx_bench.py --model M.gguf --tag q4km-ngl28 --ngl 28
  ctx_bench.py --model M.gguf --tag q4km-ot17 --ngl 99 --ot-from 17
  ctx_bench.py --model M.gguf --tag q4km-fit --fit-target 1024 --fit-ctx 4096   # auto-fit
Keluaran: JSON satu baris di stdout, dan berkas <out-dir>/c4096-<tag>.json
"""
import argparse
import json
import os
import re
import subprocess
import threading
import time

HOME = os.path.expanduser("~")


def meminfo_avail_gib():
    with open("/proc/meminfo") as f:
        d = {l.split(":")[0]: int(l.split()[1]) for l in f}
    return d["MemAvailable"] / 1048576, (d["SwapTotal"] - d["SwapFree"]) / 1048576


def vram_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return float(out.splitlines()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bin-dir", default=os.path.join(HOME, "llama.cpp/build-iso3/bin"))
    ap.add_argument("--out-dir", default=os.path.join(HOME, "27b-results"))
    ap.add_argument("--ngl", type=int, default=None, help="kosongkan untuk auto-fit atau default")
    ap.add_argument("--ot-from", type=int, default=None, help="FFN layer k..64 ke CPU")
    ap.add_argument("--fit-target", type=int, default=None, help="-fitt MiB")
    ap.add_argument("--fit-ctx", type=int, default=None, help="-fitc")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--depth", type=int, default=4096)
    ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1500)
    a = ap.parse_args()

    cmd = [os.path.join(a.bin_dir, "llama-bench"), "-m", a.model, "-p", "0", "-n", str(a.gen),
           "-d", str(a.depth), "-r", str(a.runs), "-t", str(a.threads), "-v", "-o", "json"]
    if a.ngl is not None:
        cmd += ["-ngl", str(a.ngl)]
    if a.ot_from is not None:
        ls = "|".join(str(i) for i in range(a.ot_from, 65))
        cmd += ["-ot", r"blk\.(%s)\.ffn_(up|down|gate)\.weight=CPU" % ls]
    if a.fit_target is not None:
        cmd += ["-fitt", str(a.fit_target)]
    if a.fit_ctx is not None:
        cmd += ["-fitc", str(a.fit_ctx)]

    stat = {"min_avail": 1e9, "peak_swap": 0.0, "peak_vram": 0.0}
    base_vram = vram_used_mib()
    swap0 = meminfo_avail_gib()[1]
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            av, sw = meminfo_avail_gib()
            stat["min_avail"] = min(stat["min_avail"], av)
            stat["peak_swap"] = max(stat["peak_swap"], sw)
            stat["peak_vram"] = max(stat["peak_vram"], vram_used_mib())
            time.sleep(1)

    avail0 = meminfo_avail_gib()[0]
    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
        out, err, rc = p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        out, err, rc = "", "timeout", -1
    wall = time.time() - t0
    stop.set()
    th.join()

    res = {"tag": a.tag, "model": os.path.basename(a.model), "cmd": cmd[1:], "rc": rc, "wall_s": round(wall, 1),
           "base_vram_mib": base_vram, "avail0_gib": round(avail0, 2)}
    try:
        rows = json.loads(out)
        res["tg_tps"] = rows[0]["avg_ts"]
        res["tg_std"] = rows[0]["stddev_ts"]
    except Exception:
        res["tg_tps"] = None
        res["oom"] = "out of memory" in err.lower()
    ms = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", err)  # auto-fit mencoba beberapa kali; pakai yang terakhir
    m = None
    if ms:
        m = re.match(r"(\d+) (\d+)", "%s %s" % ms[-1])
    if m:
        res["layers_gpu"], res["layers_total"] = int(m.group(1)), int(m.group(2))
    for key, pat in (("cuda_model_mib", r"CUDA0 model buffer size =\s+([\d.]+)"),
                     ("cpu_model_mib", r"CPU_Mapped model buffer size =\s+([\d.]+)"),
                     ("cuda_kv_mib", r"CUDA0 KV buffer size =\s+([\d.]+)"),
                     ("n_ctx", r"llama_context: n_ctx\s+=\s+(\d+)")):
        ms = re.findall(pat, err)
        if ms:
            res[key] = float(ms[-1])
    fit = [l for l in err.splitlines() if "fit" in l.lower() and ("llama_params_fit" in l or "fitting" in l.lower())]
    if fit:
        res["fit_log"] = fit[:6]
    res["ram_min_avail_gib"] = round(stat["min_avail"], 2)
    res["ram_peak_used_gib_est"] = round(avail0 - stat["min_avail"], 2)
    res["swap_delta_gib"] = round(stat["peak_swap"] - swap0, 3)
    res["vram_peak_mib"] = stat["peak_vram"]
    res["err_tail"] = err.strip().splitlines()[-3:] if rc != 0 else []
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "c4096-%s.err" % a.tag), "w") as f:
        f.write(err)
    with open(os.path.join(a.out_dir, "c4096-%s.json" % a.tag), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
