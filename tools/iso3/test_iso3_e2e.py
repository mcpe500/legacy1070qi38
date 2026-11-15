"""Test end-to-end KV cache iso3 + MTP di llama.cpp utama (CUDA).

Butuh GPU, binary llama.cpp, dan model lokal, jadi tidak ikut `tools/test_legacy.py`.
Jalankan:

    LLAMA_BIN=~/llama.cpp/build-iso3/bin python3 -m unittest tools/iso3/test_iso3_e2e.py -v

Variabel lingkungan (semua punya default untuk mesin proyek):
    LLAMA_BIN        folder binary (llama-completion, llama-perplexity, llama-server)
    ISO3_SMALL_MODEL model kecil qwen35 yang muat penuh di GPU (default Qwen3.5-0.8B-IQ4_XS)
    ISO3_MTP_MODEL   model dengan tensor MTP (default Qwen3.8-27B-UD-IQ4_XS)
    ISO3_MTP_NGL     -ngl untuk model MTP (default 30)
    ISO3_WIKI        wiki.test.raw (wikitext-2)
    ISO3_SKIP_MTP=1  lewati uji MTP (27B butuh ±10 menit)
"""

import hashlib
import json
import os
import re
import signal
import subprocess
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

HOME = Path.home()
BIN = Path(os.environ.get("LLAMA_BIN", str(HOME / "llama.cpp/build-iso3/bin"))).expanduser()
SMALL = Path(os.environ.get("ISO3_SMALL_MODEL", str(HOME / "models/Qwen3.5-0.8B-IQ4_XS.gguf"))).expanduser()
MTP_MODEL = Path(os.environ.get("ISO3_MTP_MODEL", str(HOME / "models/27b/Qwen3.8-27B-UD-IQ4_XS.gguf"))).expanduser()
MTP_NGL = os.environ.get("ISO3_MTP_NGL", "30")
WIKI = Path(os.environ.get("ISO3_WIKI", str(HOME / "datasets/wikitext-2-raw/wiki.test.raw"))).expanduser()
PORT = 8093

# Batas lulus. KLD q4_0 pada model yang sama = 0.0200 (CLAIMS C-47); iso3 3-bit boleh lebih buruk, tetapi tidak rusak.
MAX_MEAN_KLD = 0.10
MIN_SAME_TOP_P = 80.0


def run(args, timeout=900):
    p = subprocess.run([str(a) for a in args], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def kv_line(log):
    m = re.search(r"llama_kv_cache: size = .*?K \((\w+)\):\s+([\d.]+) MiB, V \((\w+)\):\s+([\d.]+) MiB", log)
    return m.groups() if m else None


class SmallModel(unittest.TestCase):
    """Model 0.8B (6 layer attention x 2 head KV x 256) penuh di GPU."""

    def setUp(self):
        for f in (BIN / "llama-completion", SMALL):
            if not f.exists():
                self.fail(f"tidak ada: {f}")

    def complete(self, ctk, ctv, n=48):
        return run([BIN / "llama-completion", "-m", SMALL, "-ngl", "99", "-c", "1024", "-n", str(n), "-fa", "on",
                    "-ctk", ctk, "-ctv", ctv, "--temp", "0", "--seed", "1", "-no-cnv", "-v",
                    "-p", "The capital of France is"])

    def test_kv_shape_iso3(self):
        rc, log = self.complete("iso3", "iso3")
        self.assertEqual(rc, 0, log[-2000:])
        kv = kv_line(log)
        self.assertIsNotNone(kv, "baris llama_kv_cache tidak ditemukan")
        # 1024 sel x 6 layer x (2 head x 256) nilai, 50 byte per 128 nilai
        want_mib = 1024 * 6 * (2 * 256) / 128 * 50 / 2**20
        self.assertEqual(kv[0], "iso3")
        self.assertEqual(kv[2], "iso3")
        self.assertAlmostEqual(float(kv[1]), want_mib, delta=0.01)
        self.assertAlmostEqual(float(kv[3]), want_mib, delta=0.01)

    def test_mixed_k_q8_v_iso3(self):
        rc, log = self.complete("q8_0", "iso3")
        self.assertEqual(rc, 0, log[-2000:])
        self.assertEqual(kv_line(log)[2], "iso3")

    def test_kld_vs_f16(self):
        if not WIKI.exists():
            self.fail(f"tidak ada: {WIKI}")
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "f16.kld"
            common = [BIN / "llama-perplexity", "-m", SMALL, "-f", WIKI, "-c", "512", "--chunks", "8", "-ngl", "99", "-fa", "on"]
            rc, log = run(common + ["--kl-divergence-base", base])
            self.assertEqual(rc, 0, log[-2000:])
            rc, log = run(common + ["-ctk", "iso3", "-ctv", "iso3", "--kl-divergence-base", base, "--kl-divergence"])
            self.assertEqual(rc, 0, log[-2000:])
        mean = re.search(r"Mean\s+KLD:\s+([-\d.]+)", log)
        same = re.search(r"Same top p:\s+([\d.]+)", log)
        self.assertIsNotNone(mean, log[-2000:])
        print(f"\n[iso3 vs f16, 0.8B, 8 chunk] mean KLD {mean.group(1)}, same top p {same.group(1) if same else '?'} %")
        self.assertLess(float(mean.group(1)), MAX_MEAN_KLD)
        if same:
            self.assertGreater(float(same.group(1)), MIN_SAME_TOP_P)


class Server:
    """llama-server lokal untuk satu set argumen; dipakai dengan `with`."""

    def __init__(self, model, ngl, extra):
        self.args = [BIN / "llama-server", "-m", model, "-ngl", ngl, "-c", "2048", "-np", "1", "-fa", "on",
                     "--host", "127.0.0.1", "--port", str(PORT), "-v"] + extra
        self.log_path = None
        self.p = None

    def __enter__(self):
        log = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
        self.log_path = Path(log.name)
        self.p = subprocess.Popen([str(a) for a in self.args], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        for _ in range(600):
            if self.p.poll() is not None:
                break
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except Exception:
                time.sleep(1)
        return self

    def alive(self):
        return self.p.poll() is None

    def log(self):
        return self.log_path.read_text(errors="replace")

    def complete(self, prompt, n_predict, **kw):
        body = dict(prompt=prompt, n_predict=n_predict, temperature=0, seed=1, cache_prompt=False, return_tokens=True)
        body.update(kw)
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=1800).read())

    def __exit__(self, *exc):
        if self.p.poll() is None:
            self.p.send_signal(signal.SIGTERM)
            try:
                self.p.wait(60)
            except subprocess.TimeoutExpired:
                self.p.kill()
        self.log_path.unlink()


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


# Selisih logprob maksimum (nats) antara token terbaik dan token pilihan MTP pada titik beda pertama.
# Pada temperature 0, beda karena urutan penjumlahan float (ukuran batch verifikasi lain) hanya terjadi di token yang hampir seri.
MAX_TIE_GAP = 0.3
N_PREDICT = 96


@unittest.skipIf(os.environ.get("ISO3_SKIP_MTP") == "1", "ISO3_SKIP_MTP=1")
class MtpWithIso3(unittest.TestCase):
    """Konteks MTP membaca memori target (ctx_other) dan punya KV 1 layer sendiri untuk kepala MTP.
    Tipe KV target diatur -ctk/-ctv, tipe KV MTP diatur -ctkd/-ctvd (default f16, tidak diwarisi;
    terukur: tanpa -ctkd log menunjukkan "1 layers ... K (f16)"). Draft yang ditolak dibuang dari KV iso3 (rollback).

    Keluaran MTP di llama.cpp utama tidak selalu identik bit dengan tanpa MTP, juga dengan KV f16
    (terukur: 27B IQ4_XS ngl 30 -fa on, f16: md5 tanpa MTP 93fcac7c, MTP ebff812c). Maka kriteria rollback:
    sampai titik beda pertama token sama, dan di titik itu token MTP adalah kandidat hampir seri
    (selisih logprob < MAX_TIE_GAP) menurut target tanpa MTP. KV yang rusak setelah rollback tidak lolos uji ini."""

    PROMPT = ("<|im_start|>user\nTulis fungsi Python untuk menghitung bilangan prima sampai n, lalu jelaskan."
              "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")

    def test_mtp_rollback_with_iso3(self):
        for f in (BIN / "llama-server", MTP_MODEL):
            if not f.exists():
                self.fail(f"tidak ada: {f}")
        kv = ["-ctk", "iso3", "-ctv", "iso3"]

        runs = {}
        for depth in ("1", "3"):
            # konteks MTP punya KV sendiri untuk layer MTP (blk.64); tipenya dari -ctkd/-ctvd (default f16), bukan dari -ctk/-ctv
            mtp = ["--spec-type", "draft-mtp", "--spec-draft-n-max", depth, "-ctkd", "iso3", "-ctvd", "iso3"]
            with Server(MTP_MODEL, MTP_NGL, kv + mtp) as srv:
                self.assertTrue(srv.alive(), srv.log()[-3000:])
                res = srv.complete(self.PROMPT, N_PREDICT)
                log = srv.log()
            t = res["timings"]
            print(f"\n[iso3, MTP depth {depth}] {t['predicted_per_second']:.2f} tok/s draft {t.get('draft_n')} "
                  f"accepted {t.get('draft_n_accepted')} md5 {hashlib.md5(res['content'].encode()).hexdigest()[:8]}")
            # semua KV bertipe iso3: 16 layer attention target + 1 layer MTP
            kv_types = re.findall(r"llama_kv_cache: size = .*?K \((\w+)\):.*?V \((\w+)\)", log)
            self.assertTrue(kv_types, "baris llama_kv_cache tidak ditemukan")
            self.assertTrue(all(k == "iso3" and v == "iso3" for k, v in kv_types), f"tipe KV: {kv_types}")
            self.assertRegex(log, r"llama_kv_cache: size = .*\b1 layers.*K \(iso3\)", "KV konteks MTP bukan iso3")
            self.assertIn("creating MTP draft context", log)
            self.assertGreater(t.get("draft_n", 0), 0, "MTP tidak membuat draft")
            if depth == "3":
                self.assertLess(t["draft_n_accepted"], t["draft_n"], "tidak ada draft ditolak: rollback tidak teruji")
            runs[depth] = res

        with Server(MTP_MODEL, MTP_NGL, kv) as srv:
            self.assertTrue(srv.alive(), srv.log()[-3000:])
            ref = srv.complete(self.PROMPT, N_PREDICT)
            print(f"[iso3, tanpa MTP] {ref['timings']['predicted_per_second']:.2f} tok/s "
                  f"md5 {hashlib.md5(ref['content'].encode()).hexdigest()[:8]}")
            for depth, res in runs.items():
                i = first_diff(ref["tokens"], res["tokens"])
                if i is None:
                    print(f"[depth {depth}] identik dengan tanpa MTP ({len(res['tokens'])} token)")
                    continue
                # logprob target tanpa MTP di posisi i, dengan awalan yang sama
                probe = srv.complete([self.PROMPT] + res["tokens"][:i], 1, n_probs=10)
                top = probe["completion_probabilities"][0]["top_logprobs"]
                lp = {e["id"]: e["logprob"] for e in top}
                mtp_tok = res["tokens"][i]
                gap = top[0]["logprob"] - lp[mtp_tok] if mtp_tok in lp else float("inf")
                names = {e["id"]: e["token"] for e in top}
                ref_tok = ref["tokens"][i]
                print(f"[depth {depth}] beda pertama di token {i}/{len(res['tokens'])}: tanpa MTP {names.get(ref_tok, ref_tok)!r} "
                      f"(logprob {lp.get(ref_tok, float('-inf')):.3f}), MTP {names.get(mtp_tok, mtp_tok)!r} "
                      f"(logprob {lp.get(mtp_tok, float('-inf')):.3f}), selisih ke terbaik {gap:.3f}")
                self.assertLess(gap, MAX_TIE_GAP, f"depth {depth}: token MTP bukan kandidat hampir seri (selisih {gap:.3f})")


if __name__ == "__main__":
    unittest.main()
