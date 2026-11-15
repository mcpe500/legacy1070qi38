#!/usr/bin/env python3
"""Konteks panjang: prompt ±N token dari wikitext, lalu generate 64 token, via llama-server (127.0.0.1).
Pemakaian: long_ctx.py <tag> <ctx> <prompt_tokens> <ngl> [argumen llama-server tambahan...]
Mencetak satu baris JSON: tps prompt/gen, draft, VRAM puncak (nvidia-smi), md5 keluaran."""
import hashlib, json, os, subprocess, sys, threading, time, urllib.request
from pathlib import Path

tag, ctx, n_prompt, ngl = sys.argv[1:5]
extra = sys.argv[5:]
B = Path(os.environ.get("B", str(Path.home() / "llama.cpp/build-iso3/bin")))
M = os.environ.get("M", str(Path.home() / "models/27b/Qwen3.8-27B-UD-Q4_K_M.gguf"))
PORT = 8094
text = Path.home().joinpath("datasets/wikitext-2-raw/wiki.test.raw").read_text()

log = open(f"long-{tag}.server.log", "w")
p = subprocess.Popen([str(B / "llama-server"), "-m", M, "-ngl", ngl, "-c", ctx, "-np", "1", "-fa", "on", "-t", "4",
                      "--host", "127.0.0.1", "--port", str(PORT)] + extra,
                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
peak = [0]
stop = False


def vram():
    while not stop:
        try:
            v = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])
            peak[0] = max(peak[0], v)
        except Exception:
            pass
        time.sleep(1)


threading.Thread(target=vram, daemon=True).start()
res = None
try:
    for _ in range(900):
        if p.poll() is not None:
            break
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            break
        except Exception:
            time.sleep(1)
    if p.poll() is None:
        def post(path, body):
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=7200).read())
        toks = post("/tokenize", {"content": text[:int(n_prompt) * 6]})["tokens"][:int(n_prompt)]
        prompt = toks + post("/tokenize", {"content": "\n\nSummarize the text above in three sentences:"})["tokens"]
        res = post("/completion", {"prompt": prompt, "n_predict": 64, "temperature": 0, "seed": 1, "cache_prompt": False})
finally:
    stop = True
    if p.poll() is None:
        p.terminate()
        p.wait(120)
    log.close()

out = {"tag": tag, "ctx": int(ctx), "ngl": ngl, "extra": " ".join(extra), "peak_vram_mib": peak[0]}
if res is None:
    tail = Path(f"long-{tag}.server.log").read_text(errors="replace").splitlines()[-3:]
    out["error"] = " | ".join(tail)[-300:]
else:
    t = res["timings"]
    out.update(prompt_n=t["prompt_n"], prompt_tps=round(t["prompt_per_second"], 2), gen_n=t["predicted_n"],
               gen_tps=round(t["predicted_per_second"], 2), draft_n=t.get("draft_n"), draft_accepted=t.get("draft_n_accepted"),
               md5=hashlib.md5(res["content"].encode()).hexdigest()[:8])
print(json.dumps(out), flush=True)
