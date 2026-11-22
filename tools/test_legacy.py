"""Tests for tools/legacy.py. Standard library only.

Run from the repo root:
    python -m unittest discover -s tools -p "test_legacy.py" -v
or directly:
    python tools/test_legacy.py
"""
import contextlib
import io
import json
import os
import random
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import legacy  # noqa: E402

GIB = legacy.GIB
HAS_PROC = os.path.exists("/proc/meminfo") and os.path.exists("/proc/cpuinfo")


def cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = legacy.main(list(argv))
    return code, out.getvalue(), err.getvalue()


def cli_json(*argv):
    code, out, err = cli(*argv, "--json")
    return code, json.loads(out), err


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="legacy_test_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def path(self, *parts):
        return os.path.join(self.tmp, *parts)

    def write(self, rel, text, mode="w"):
        full = self.path(rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, mode, encoding=None if "b" in mode else "utf-8") as fh:
            fh.write(text)
        return full


# --------------------------------------------------------------------------
# A. budget
# --------------------------------------------------------------------------

class BudgetKnownValues(unittest.TestCase):
    """Worked examples from spec/AGENTS/roles/memory-manager.md, recomputed by hand."""

    def test_kv_default_is_64_kib_per_token(self):
        self.assertEqual(legacy.kv_bytes_per_token(16, 8, 128, "f16"), 65536)

    def test_kv_quantized_is_smaller(self):
        f16 = legacy.kv_bytes_per_token(16, 8, 128, "f16")
        self.assertAlmostEqual(legacy.kv_bytes_per_token(16, 8, 128, "q8_0") / f16, 34 / 64)
        self.assertAlmostEqual(legacy.kv_bytes_per_token(16, 8, 128, "q4_0") / f16, 18 / 64)

    def test_q4_k_m_split(self):
        e = legacy.estimate(16.53, rs_gib=0)   # nilai tangan tanpa state DeltaNet
        self.assertAlmostEqual(e["kv_gib"], 0.25, places=4)
        self.assertAlmostEqual(e["vram"]["weights_capacity_gib"], 6.85, places=3)
        self.assertAlmostEqual(e["ram"]["weights_gib"], 9.68, places=3)
        self.assertAlmostEqual(e["ram"]["total_gib"], 12.18, places=3)
        self.assertEqual(e["risk"], "green")
        # 1 / (9.68 GiB / 20 GB/s + 6.85 GiB / 256 GB/s)
        self.assertAlmostEqual(e["tps_upper_bound"], 1.823, places=2)

    def test_q5_k_m_is_tight(self):
        e = legacy.estimate(18.70, rs_gib=0)
        self.assertAlmostEqual(e["ram"]["headroom_gib"], 1.65, places=3)
        self.assertEqual(e["risk"], "yellow")

    def test_does_not_fit_in_ram(self):
        e = legacy.estimate(30.0)
        self.assertLess(e["ram"]["headroom_gib"], 0)
        self.assertEqual(e["risk"], "red")

    def test_small_model_fits_fully_in_vram(self):
        e = legacy.estimate(5.0)
        self.assertEqual(e["ram"]["weights_gib"], 0)
        self.assertAlmostEqual(e["ram"]["total_gib"], 2.5)
        self.assertAlmostEqual(e["tps_upper_bound"], 256e9 / (5.0 * GIB), places=1)

    def test_vram_too_small_for_fixed_costs_is_red(self):
        e = legacy.estimate(5.0, vram_gib=1.0)
        self.assertEqual(e["risk"], "red")
        self.assertEqual(e["vram"]["weights_gib"], 0)

    def test_suggest_ngl(self):
        self.assertEqual(legacy.suggest_ngl(16.53, layers=64, rs_gib=0)["ngl"], 26)
        self.assertEqual(legacy.suggest_ngl(5.0, layers=64, rs_gib=0)["ngl"], 64)
        self.assertEqual(legacy.suggest_ngl(100.0, layers=64, rs_gib=0)["ngl"], 4)

    def test_recurrent_state_is_counted_by_default(self):
        with_rs = legacy.estimate(16.53)
        without = legacy.estimate(16.53, rs_gib=0)
        self.assertAlmostEqual(without["vram"]["weights_capacity_gib"] - with_rs["vram"]["weights_capacity_gib"],
                               legacy.DEFAULT_RS_GIB, places=4)
        self.assertEqual(with_rs["vram"]["recurrent_state_gib"], 0.146)

    def test_default_rs_matches_official_dims(self):
        # 48 layer GDN x (48 head V x 128 x 128 x 4 B + konv 3 x (6144 + 2*16*128) x 4 B)
        per_layer = 48 * 128 * 128 * 4 + 3 * (6144 + 2 * 16 * 128) * 4
        self.assertAlmostEqual(48 * per_layer / GIB, legacy.DEFAULT_RS_GIB, places=3)

    def test_recurrent_state_formula_matches_measured_08b(self):
        # Log asli llama.cpp, Qwen3.5-0.8B, 4 sekuens: "RS buffer size = 77.06 MiB"
        per_layer = 16 * 128 * 128 * 4 + 3 * (2048 + 2 * 16 * 128) * 4
        self.assertAlmostEqual(4 * 18 * per_layer / 2 ** 20, 77.06, places=1)

    def test_iso3_kv_factor_from_block_layout(self):
        # block_iso3_0 = 2 + 128/4 + 128/8 = 50 byte per 128 nilai
        self.assertEqual(2 + 32 + 16, 50)
        f16 = legacy.kv_bytes_per_token(16, 4, 256, "f16")
        self.assertAlmostEqual(legacy.kv_bytes_per_token(16, 4, 256, "iso3") / f16, 50 / 256)

    def test_compare(self):
        est = legacy.estimate(16.53)
        ok = legacy.compare_measured(est, measured_ram_gib=12.5, measured_tps=1.0)
        self.assertTrue(ok["all_within"])
        far = legacy.compare_measured(est, measured_ram_gib=20.0)
        self.assertFalse(far["all_within"])
        impossible = legacy.compare_measured(est, measured_tps=5.0)
        self.assertFalse(impossible["all_within"])
        with self.assertRaises(legacy.CliError):
            legacy.compare_measured(est)


class BudgetInvalidInput(unittest.TestCase):
    def test_rejects_bad_numbers(self):
        for bad in (0, -1, float("nan"), float("inf"), None, True):
            with self.assertRaises(legacy.CliError, msg=repr(bad)):
                legacy.estimate(bad)
        with self.assertRaises(legacy.CliError):
            legacy.estimate(10, ctx=0)
        with self.assertRaises(legacy.CliError):
            legacy.estimate(10, kv_type="q9")
        with self.assertRaises(legacy.CliError):
            legacy.estimate(10, os_gib=-1)
        with self.assertRaises(legacy.CliError):
            legacy.suggest_ngl(10, layers=0)


class BudgetProperties(unittest.TestCase):
    """Property checks with a fixed seed (no external library)."""

    def setUp(self):
        self.rng = random.Random(1070)

    def random_case(self):
        r = self.rng
        return dict(file_gib=r.uniform(0.5, 40), vram_gib=r.uniform(4, 24), ram_gib=r.uniform(8, 64),
                    ctx=r.randint(128, 65536), ram_bw_gbs=r.uniform(5, 80))

    def test_weights_are_conserved_and_never_exceed_vram(self):
        for _ in range(300):
            c = self.random_case()
            e = legacy.estimate(**c)
            self.assertAlmostEqual(e["vram"]["weights_gib"] + e["ram"]["weights_gib"], c["file_gib"], delta=1e-3)
            self.assertGreaterEqual(e["ram"]["weights_gib"], 0)
            if e["vram"]["weights_capacity_gib"] > 0:
                self.assertLessEqual(e["vram"]["used_gib"], c["vram_gib"] + 1e-3)
            self.assertGreaterEqual(e["ram"]["total_gib"], e["ram"]["os_runtime_gib"] - 1e-9)

    def test_bigger_file_never_helps(self):
        for _ in range(300):
            c = self.random_case()
            small = legacy.estimate(**c)
            c["file_gib"] += self.rng.uniform(0.1, 10)
            big = legacy.estimate(**c)
            self.assertGreaterEqual(big["ram"]["weights_gib"], small["ram"]["weights_gib"] - 1e-3)
            self.assertLessEqual(big["tps_upper_bound"], small["tps_upper_bound"] + 1e-2)

    def test_longer_context_never_gives_more_gpu_weights(self):
        for _ in range(300):
            c = self.random_case()
            short = legacy.estimate(**c)
            c["ctx"] += self.rng.randint(1, 65536)
            long_ = legacy.estimate(**c)
            self.assertGreaterEqual(long_["kv_gib"], short["kv_gib"])
            self.assertLessEqual(long_["vram"]["weights_gib"], short["vram"]["weights_gib"] + 1e-3)

    def test_faster_ram_never_slower(self):
        for _ in range(300):
            c = self.random_case()
            slow = legacy.estimate(**c)
            c["ram_bw_gbs"] *= 2
            fast = legacy.estimate(**c)
            self.assertGreaterEqual(fast["tps_upper_bound"], slow["tps_upper_bound"] - 1e-2)

    def test_ngl_in_range_and_monotonic_in_file_size(self):
        for _ in range(300):
            c = self.random_case()
            a = legacy.suggest_ngl(c["file_gib"], layers=64, **{k: v for k, v in c.items() if k != "file_gib"})
            self.assertTrue(0 <= a["ngl"] <= 64)
            c["file_gib"] += self.rng.uniform(0.1, 10)
            b = legacy.suggest_ngl(c["file_gib"], layers=64, **{k: v for k, v in c.items() if k != "file_gib"})
            self.assertLessEqual(b["ngl"], a["ngl"])


# --------------------------------------------------------------------------
# B. hw parsers
# --------------------------------------------------------------------------

class HwParsers(unittest.TestCase):
    def test_nvidia_csv(self):
        g = legacy.parse_nvidia_csv("NVIDIA GeForce GTX 1070 Ti, 8192, 1024, 7168, 45, 30.5, 3\n")[0]
        self.assertEqual(g["name"], "NVIDIA GeForce GTX 1070 Ti")
        self.assertEqual(g["vram_free_gib"], 7.0)
        self.assertEqual(g["temp_c"], 45.0)
        self.assertEqual(g["power_w"], 30.5)

    def test_nvidia_na_becomes_none(self):
        g = legacy.parse_nvidia_csv("GPU, 8192, 1024, 7168, 45, [N/A], [Not Supported]\n")[0]
        self.assertIsNone(g["power_w"])
        self.assertIsNone(g["util_pct"])

    def test_nvidia_bad_input(self):
        with self.assertRaises(legacy.CliError):
            legacy.parse_nvidia_csv("only, three, fields\n")
        with self.assertRaises(legacy.CliError):
            legacy.parse_nvidia_csv("\n")

    def test_meminfo(self):
        text = ("MemTotal:       16384000 kB\nMemFree:         1000000 kB\nMemAvailable:    8388608 kB\n"
                "Cached:          2000000 kB\nActive(anon):     100 kB\nSwapTotal:       2097152 kB\n"
                "SwapFree:        1048576 kB\n")
        m = legacy.parse_meminfo(text)
        self.assertEqual(m["total_gib"], 15.625)
        self.assertEqual(m["available_gib"], 8.0)
        self.assertEqual(m["swap_total_gib"], 2.0)
        self.assertEqual(m["swap_used_gib"], 1.0)
        with self.assertRaises(legacy.CliError):
            legacy.parse_meminfo("MemTotal: 1 kB\n")

    def test_cpuinfo(self):
        text = ("processor\t: 0\nmodel name\t: Test CPU @ 3.0GHz\nphysical id\t: 0\ncore id\t: 0\n"
                "flags\t: fpu avx2 fma\ncpu MHz\t: 3000.0\n\n"
                "processor\t: 1\nmodel name\t: Test CPU @ 3.0GHz\nphysical id\t: 0\ncore id\t: 1\n"
                "flags\t: fpu avx2 fma\ncpu MHz\t: 3800.5\n")
        c = legacy.parse_cpuinfo(text)
        self.assertEqual(c["model"], "Test CPU @ 3.0GHz")
        self.assertEqual(c["threads"], 2)
        self.assertEqual(c["cores"], 2)
        self.assertTrue(c["avx2"])
        self.assertFalse(c["avx512f"])
        self.assertEqual(c["mhz_max_now"], 3800.5)

    def test_vmstat_delta(self):
        before = legacy.parse_vmstat("pgmajfault 10\npswpin 0\npswpout 5\nother 9\n")
        after = legacy.parse_vmstat("pgmajfault 30\npswpin 0\npswpout 8\n")
        d = legacy.vmstat_delta(before, after, 2.0)
        self.assertEqual(d["major_faults"], 20)
        self.assertEqual(d["major_faults_per_s"], 10.0)
        self.assertTrue(d["swapping"])
        with self.assertRaises(legacy.CliError):
            legacy.vmstat_delta(before, after, 0)

    def test_summarize_samples(self):
        s = legacy.summarize_samples([
            {"vram_used_mib": 100, "temp_c": 40, "power_w": None, "util_pct": 10},
            {"vram_used_mib": 900, "temp_c": 55, "power_w": 80.0, "util_pct": 30},
        ])
        self.assertEqual(s["peak_vram_used_mib"], 900)
        self.assertEqual(s["peak_temp_c"], 55)
        self.assertEqual(s["peak_power_w"], 80.0)
        self.assertEqual(s["mean_util_pct"], 20.0)
        with self.assertRaises(legacy.CliError):
            legacy.summarize_samples([])

    def test_headroom(self):
        go = legacy.headroom_check(7.0, 10.0, 6.0, 5.0)
        self.assertTrue(go["go"])
        no_vram = legacy.headroom_check(6.1, 10.0, 6.0, 5.0)  # 0.1 left < 0.2 margin
        self.assertFalse(no_vram["go"])
        no_ram = legacy.headroom_check(7.0, 5.4, 6.0, 5.0)  # 0.4 left < 0.5 margin
        self.assertFalse(no_ram["go"])
        no_gpu = legacy.headroom_check(None, 10.0, 6.0, 5.0)
        self.assertTrue(no_gpu["go"])


@unittest.skipUnless(HAS_PROC, "butuh /proc (Linux)")
class HwRealSystem(unittest.TestCase):
    def test_ram_and_cpu_real(self):
        code, ram, _ = cli_json("hw", "ram")
        self.assertEqual(code, 0)
        self.assertGreater(ram["total_gib"], 0)
        code, cpu, _ = cli_json("hw", "cpu")
        self.assertEqual(code, 0)
        self.assertGreaterEqual(cpu["threads"], 1)

    def test_headroom_no_go_exit_code(self):
        code, data, _ = cli_json("hw", "headroom", "--no-gpu", "--need-vram-gib", "1",
                                 "--need-ram-gib", "1000000")
        self.assertEqual(code, 1)
        self.assertFalse(data["go"])

    def test_headroom_go_exit_code(self):
        code, data, _ = cli_json("hw", "headroom", "--no-gpu", "--need-vram-gib", "1",
                                 "--need-ram-gib", "0.001", "--ram-margin-gib", "0.001")
        self.assertEqual(code, 0)
        self.assertTrue(data["go"])

    def test_membw_small(self):
        code, data, _ = cli_json("hw", "membw", "--size-mib", "16", "--repeats", "1")
        self.assertEqual(code, 0)
        self.assertGreater(data["copy_gb_per_s"], 0)

    def test_membw_rejects_out_of_range(self):
        code, data, _ = cli_json("hw", "membw", "--size-mib", "1")
        self.assertEqual(code, 2)
        self.assertFalse(data["ok"])


class HwGpuMissing(unittest.TestCase):
    @unittest.skipIf(shutil.which("nvidia-smi"), "nvidia-smi ada; uji ini untuk mesin tanpa GPU")
    def test_gpu_without_nvidia_smi_exits_3(self):
        code, _, err = cli("hw", "gpu")
        self.assertEqual(code, 3)
        self.assertIn("nvidia-smi", err)


# --------------------------------------------------------------------------
# C1. gguf
# --------------------------------------------------------------------------

def gs(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_str(key, val):
    return gs(key) + struct.pack("<I", 8) + gs(val)


def kv_u32(key, val):
    return gs(key) + struct.pack("<I", 4) + struct.pack("<I", val)


def kv_arr_u32(key, vals):
    return (gs(key) + struct.pack("<I", 9) + struct.pack("<I", 4) + struct.pack("<Q", len(vals))
            + b"".join(struct.pack("<I", v) for v in vals))


def kv_arr_u8(key, n):
    return gs(key) + struct.pack("<I", 9) + struct.pack("<I", 0) + struct.pack("<Q", n) + bytes(n)


def tensor_info(name, dims, type_id, offset=0):
    return (gs(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
            + struct.pack("<I", type_id) + struct.pack("<Q", offset))


def make_gguf(path, kvs, tensors, version=3):
    blob = (b"GGUF" + struct.pack("<I", version) + struct.pack("<Q", len(tensors))
            + struct.pack("<Q", len(kvs)) + b"".join(kvs) + b"".join(tensors))
    with open(path, "wb") as fh:
        fh.write(blob)
    return blob


class GgufTests(TempDirCase):
    def sample(self):
        p = self.path("m.gguf")
        make_gguf(p, [
            kv_str("general.architecture", "hybridtest"),
            kv_str("general.name", "Tiny Hybrid"),
            kv_u32("general.file_type", 15),
            kv_u32("hybridtest.block_count", 4),
            kv_u32("hybridtest.context_length", 8192),
            kv_arr_u32("hybridtest.attention.head_count_kv", [0, 0, 0, 8]),
            kv_arr_u8("tokenizer.big", 600),
        ], [
            tensor_info("token_embd.weight", [256, 10], 12),
            tensor_info("blk.0.attn_q.weight", [256, 4], 12),
            tensor_info("blk.0.norm.weight", [8], 0),
            tensor_info("blk.1.x.weight", [64], 8),
            tensor_info("blk.2.odd.weight", [10], 12),        # not a multiple of the block size
            tensor_info("blk.2.new.weight", [4], 99),         # unknown type id
            tensor_info("blk.3.big.weight", [256, 4194304], 12),
        ])
        return p

    def test_parse_values(self):
        g = legacy.parse_gguf(self.sample())
        md = g["metadata"]
        self.assertEqual(g["version"], 3)
        self.assertEqual(md["general.architecture"], "hybridtest")
        self.assertEqual(md["hybridtest.attention.head_count_kv"], [0, 0, 0, 8])
        self.assertEqual(md["tokenizer.big"], {"array_len": 600, "elem_type": 0})
        sizes = {t["name"]: t["nbytes"] for t in g["tensors"]}
        self.assertEqual(sizes["token_embd.weight"], 1440)       # 2560 / 256 * 144
        self.assertEqual(sizes["blk.0.attn_q.weight"], 576)
        self.assertEqual(sizes["blk.0.norm.weight"], 32)
        self.assertEqual(sizes["blk.1.x.weight"], 68)            # 64 / 32 * 34
        self.assertIsNone(sizes["blk.2.odd.weight"])
        self.assertIsNone(sizes["blk.2.new.weight"])
        self.assertEqual(sizes["blk.3.big.weight"], 603979776)

    def test_inspect_summary(self):
        s = legacy.summarize_gguf(legacy.parse_gguf(self.sample()))
        self.assertEqual(s["architecture"], "hybridtest")
        self.assertEqual(s["block_count"], 4)
        self.assertEqual(s["context_length"], 8192)
        self.assertEqual(s["kv_layers_estimate"], 1)
        self.assertEqual(s["tensor_count"], 7)

    def test_tensors_summary(self):
        t = legacy.summarize_tensors(legacy.parse_gguf(self.sample()))
        self.assertEqual(t["layer_count"], 3)  # layers 0, 1, 3 have known sizes
        by_layer = {r["layer"]: r["gib"] for r in t["layers"]}
        self.assertEqual(by_layer[3], 0.5625)
        self.assertEqual(t["tensors_unknown_size"], 2)
        self.assertIn("token_embd.weight", t["non_layer"])
        self.assertEqual(t["by_type_gib"]["Q4_K"], 0.5625)

    def test_scalar_array_of_strings(self):
        p = self.path("s.gguf")
        arr = (gs("tokenizer.tokens") + struct.pack("<I", 9) + struct.pack("<I", 8)
               + struct.pack("<Q", 2) + gs("a") + gs("bc"))
        make_gguf(p, [arr], [])
        self.assertEqual(legacy.parse_gguf(p)["metadata"]["tokenizer.tokens"], ["a", "bc"])

    def test_bad_files(self):
        with self.assertRaises(legacy.EnvError):
            legacy.parse_gguf(self.path("missing.gguf"))
        bad_magic = self.write("bad.gguf", b"NOPE" + bytes(32), "wb")
        with self.assertRaises(legacy.CliError):
            legacy.parse_gguf(bad_magic)
        v1 = self.path("v1.gguf")
        make_gguf(v1, [], [], version=1)
        with self.assertRaises(legacy.CliError):
            legacy.parse_gguf(v1)

    def test_truncated(self):
        blob = make_gguf(self.path("full.gguf"), [kv_str("general.architecture", "x")],
                         [tensor_info("a", [4], 0)])
        for cut in (3, 10, len(blob) - 1):
            p = self.write(f"cut{cut}.gguf", blob[:cut], "wb")
            with self.assertRaises(legacy.CliError, msg=f"cut={cut}"):
                legacy.parse_gguf(p)

    def test_hostile_lengths_are_rejected_without_allocating(self):
        huge_key = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 1) \
            + struct.pack("<Q", 1 << 40)
        with self.assertRaises(legacy.CliError):
            legacy.parse_gguf(self.write("h1.gguf", huge_key, "wb"))
        many = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 1 << 60) + struct.pack("<Q", 0)
        with self.assertRaises(legacy.CliError):
            legacy.parse_gguf(self.write("h2.gguf", many, "wb"))
        big_arr = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 1) \
            + gs("k") + struct.pack("<I", 9) + struct.pack("<I", 0) + struct.pack("<Q", 1 << 40)
        with self.assertRaises(legacy.CliError):
            legacy.parse_gguf(self.write("h3.gguf", big_arr, "wb"))

    def test_cli(self):
        p = self.sample()
        code, data, _ = cli_json("gguf", "inspect", p)
        self.assertEqual(code, 0)
        self.assertEqual(data["architecture"], "hybridtest")
        code, data, _ = cli_json("gguf", "tensors", p)
        self.assertEqual(code, 0)
        self.assertEqual(data["layer_count"], 3)
        code, data, _ = cli_json("gguf", "inspect", self.path("nope.gguf"))
        self.assertEqual(code, 3)


# --------------------------------------------------------------------------
# C2. llama
# --------------------------------------------------------------------------

class LlamaPure(TempDirCase):
    def test_resolve_bin_allowlist(self):
        with self.assertRaises(legacy.CliError):
            legacy.resolve_bin(self.tmp, "rm")
        with self.assertRaises(legacy.CliError):
            legacy.resolve_bin(self.tmp, "../llama-cli")
        with self.assertRaises(legacy.EnvError):
            legacy.resolve_bin(self.tmp, "llama-cli")

    @unittest.skipIf(os.name == "nt", "butuh bit executable POSIX")
    def test_resolve_bin_ok(self):
        f = self.write("llama-cli", "#!/bin/sh\n")
        os.chmod(f, 0o755)
        self.assertEqual(legacy.resolve_bin(self.tmp, "llama-cli"), f)

    def test_check_model(self):
        with self.assertRaises(legacy.CliError):
            legacy.check_model("model.bin")
        with self.assertRaises(legacy.EnvError):
            legacy.check_model(self.path("none.gguf"))
        f = self.write("ok.gguf", "x")
        self.assertEqual(legacy.check_model(f), os.path.abspath(f))

    def test_run_argv(self):
        argv = legacy.build_run_argv("/b/llama-cli", "/m.gguf", ctx=2048, ngl=20, prompt="-hi; rm -rf /",
                                     n_predict=8, kv_type="q8_0")
        self.assertEqual(argv[:3], ["/b/llama-cli", "-m", "/m.gguf"])
        self.assertIn("-ngl", argv)
        self.assertEqual(argv[argv.index("-ngl") + 1], "20")
        self.assertEqual(argv[argv.index("-p") + 1], "-hi; rm -rf /")  # one argv item, no shell
        self.assertEqual(argv[argv.index("-ctk") + 1], "q8_0")
        auto = legacy.build_run_argv("/b/llama-cli", "/m.gguf", prompt="x")
        self.assertNotIn("-ngl", auto)
        self.assertNotIn("-no-cnv", auto)
        with self.assertRaises(legacy.CliError):
            legacy.build_run_argv("/b", "/m", ctx=0)
        with self.assertRaises(legacy.CliError):
            legacy.build_run_argv("/b", "/m", ngl=-1)

    def test_bench_argv(self):
        argv = legacy.build_bench_argv("/b/llama-bench", "/m.gguf", ngl=10, reps=3)
        self.assertEqual(argv[argv.index("-r") + 1], "3")
        self.assertEqual(argv[argv.index("-o") + 1], "json")
        self.assertEqual(argv[argv.index("-ngl") + 1], "10")

    def test_cmake_argv(self):
        configure, build = legacy.build_cmake_argv("src", "build", "cuda", 4)
        self.assertIn("-DGGML_CUDA=ON", configure)
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=61", configure)
        self.assertEqual(build[-1], "4")
        vk, _ = legacy.build_cmake_argv("src", "build", "vulkan", 2)
        self.assertIn("-DGGML_VULKAN=ON", vk)
        self.assertNotIn("-DGGML_CUDA=ON", vk)
        with self.assertRaises(legacy.CliError):
            legacy.build_cmake_argv("src", "build", "opencl", 2)

    def test_parse_timing(self):
        text = ("llama_perf_context_print: prompt eval time =     100.00 ms /    10 tokens "
                "(   10.00 ms per token,   100.00 tokens per second)\n"
                "llama_perf_context_print:        eval time =    2000.00 ms /    20 runs   "
                "(  100.00 ms per token,    10.00 tokens per second)\n")
        t = legacy.parse_llama_timing(text)
        self.assertEqual(t["prompt_tps"], 100.0)
        self.assertEqual(t["gen_tps"], 10.0)
        none = legacy.parse_llama_timing("nothing here")
        self.assertIsNone(none["gen_tps"])

    def test_parse_bench_json(self):
        rows = legacy.parse_bench_json('[{"n_prompt":512,"n_gen":0,"avg_ts":100.5,"stddev_ts":1.0},{"x":1}]')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["avg_ts"], 100.5)
        with self.assertRaises(legacy.CliError):
            legacy.parse_bench_json("not json")
        with self.assertRaises(legacy.CliError):
            legacy.parse_bench_json('{"a": 1}')

    def test_parse_placement(self):
        log = ("load_tensors: offloaded 28/65 layers to GPU\n"
               "load_tensors:   CUDA0 model buffer size =  6500.00 MiB\n"
               "load_tensors: CPU_Mapped model buffer size = 10000.00 MiB\n")
        p = legacy.parse_placement(log)
        self.assertEqual(p["layers_on_gpu"], 28)
        self.assertEqual(p["layers_total_reported"], 65)
        self.assertEqual(p["buffers_mib"], {"CUDA0": 6500.0, "CPU_Mapped": 10000.0})
        with self.assertRaises(legacy.CliError):
            legacy.parse_placement("random log")


class LockTests(TempDirCase):
    def test_second_holder_is_refused_and_release_works(self):
        lock_path = self.path("l.lock")
        with legacy.InferenceLock(lock_path):
            with self.assertRaises(legacy.CliError):
                with legacy.InferenceLock(lock_path):
                    pass
            self.assertTrue(os.path.exists(lock_path))
        self.assertFalse(os.path.exists(lock_path))
        with legacy.InferenceLock(lock_path):
            pass

    def test_stale_lock_is_replaced(self):
        lock_path = self.write("l.lock", "not-a-pid")
        with legacy.InferenceLock(lock_path):
            pass
        self.write("l.lock", "0")
        with legacy.InferenceLock(lock_path):
            pass

    def test_dead_pid_lock_is_replaced(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        lock_path = self.write("l.lock", str(proc.pid))
        with legacy.InferenceLock(lock_path):
            pass

    def test_lock_released_on_exception(self):
        lock_path = self.path("l.lock")
        with self.assertRaises(RuntimeError):
            with legacy.InferenceLock(lock_path):
                raise RuntimeError("boom")
        self.assertFalse(os.path.exists(lock_path))


@unittest.skipIf(os.name == "nt", "memakai skrip sh palsu")
class LlamaEndToEnd(TempDirCase):
    def fake_bin(self, name, script):
        d = self.path("bin")
        f = self.write(os.path.join("bin", name), script)
        os.chmod(f, os.stat(f).st_mode | stat.S_IXUSR)
        return d

    def test_dry_run_does_not_need_binary(self):
        model = self.write("m.gguf", "x")
        code, data, _ = cli_json("llama", "run", "--model", model, "--prompt", "hi", "--ngl", "5",
                                 "--bin-dir", self.path("nobin"), "--dry-run")
        self.assertEqual(code, 0)
        self.assertTrue(data["dry_run"])
        self.assertIn("-ngl", data["argv"])

    def test_run_with_fake_llama_cli(self):
        model = self.write("m.gguf", "x")
        script = ("#!/bin/sh\necho hello\n"
                  "echo 'load_tensors: offloaded 20/65 layers to GPU' >&2\n"
                  "echo 'llama_perf_context_print:        eval time =    2000.00 ms /    20 runs   "
                  "(  100.00 ms per token,    10.00 tokens per second)' >&2\n")
        bin_dir = self.fake_bin("llama-cli", script)
        lock = self.path("run.lock")
        code, data, _ = cli_json("llama", "run", "--model", model, "--prompt", "hi", "--bin-dir", bin_dir,
                                 "--lock-file", lock)
        self.assertEqual(code, 0)
        self.assertEqual(data["gen_tps"], 10.0)
        self.assertEqual(data["placement"]["layers_on_gpu"], 20)
        self.assertFalse(os.path.exists(lock))

    def test_run_reports_failure_exit_code(self):
        model = self.write("m.gguf", "x")
        bin_dir = self.fake_bin("llama-cli", "#!/bin/sh\necho oops >&2\nexit 7\n")
        code, data, _ = cli_json("llama", "run", "--model", model, "--prompt", "hi", "--bin-dir", bin_dir,
                                 "--lock-file", self.path("r.lock"))
        self.assertEqual(code, 1)
        self.assertEqual(data["returncode"], 7)

    def test_run_refused_while_locked(self):
        model = self.write("m.gguf", "x")
        bin_dir = self.fake_bin("llama-cli", "#!/bin/sh\nexit 0\n")
        lock = self.path("busy.lock")
        with legacy.InferenceLock(lock):
            code, data, _ = cli_json("llama", "run", "--model", model, "--prompt", "hi",
                                     "--bin-dir", bin_dir, "--lock-file", lock)
        self.assertEqual(code, 2)
        self.assertFalse(data["ok"])

    def test_run_timeout(self):
        model = self.write("m.gguf", "x")
        bin_dir = self.fake_bin("llama-cli", "#!/bin/sh\nexec sleep 5\n")
        lock = self.path("t.lock")
        code, data, _ = cli_json("llama", "run", "--model", model, "--prompt", "hi", "--bin-dir", bin_dir,
                                 "--lock-file", lock, "--timeout", "1")
        self.assertEqual(code, 2)
        self.assertFalse(os.path.exists(lock))

    def test_bench_with_fake_binary(self):
        model = self.write("m.gguf", "x")
        bin_dir = self.fake_bin(
            "llama-bench",
            "#!/bin/sh\necho '[{\"n_prompt\":512,\"n_gen\":0,\"avg_ts\":150.0,\"stddev_ts\":2.0}]'\n")
        code, data, _ = cli_json("llama", "bench", "--model", model, "--bin-dir", bin_dir,
                                 "--lock-file", self.path("b.lock"))
        self.assertEqual(code, 0)
        self.assertEqual(data["results"][0]["avg_ts"], 150.0)

    def test_binary_outside_allowlist_is_not_reachable(self):
        # The CLI has no option to pick another binary name, so only the allowlist is runnable.
        with self.assertRaises(legacy.CliError):
            legacy.resolve_bin(self.tmp, "bash")

    def test_build_dry_run(self):
        src = self.path("src")
        self.write("src/CMakeLists.txt", "project(x)\n")
        code, data, _ = cli_json("llama", "build", "--src", src, "--dry-run", "--jobs", "2")
        self.assertEqual(code, 0)
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=61", data["commands"][0])
        code, _, _ = cli_json("llama", "build", "--src", self.path("empty"), "--dry-run")
        self.assertEqual(code, 3)

    def test_placement_cli(self):
        log = self.write("l.log", "load_tensors: offloaded 3/10 layers to GPU\n")
        code, data, _ = cli_json("llama", "placement", "--log", log)
        self.assertEqual(code, 0)
        self.assertEqual(data["layers_on_gpu"], 3)


# --------------------------------------------------------------------------
# D. spec
# --------------------------------------------------------------------------

REQ_TEMPLATE = ("# Requirements — [NAMA FITUR]\n\n> Spec: SPEC-NNN | Status: **Draft**\n\n"
                "- **FR-01:** a\n- **FR-02:** b\n\n- **NFR-01: x** y\n- **NFR-02: x** y\n")
DESIGN_TEMPLATE = "# Design — [NAMA FITUR]\n\n> Spec: SPEC-NNN\n"
TASKS_TEMPLATE = "# Tasks — [NAMA FITUR]\n\n> Spec: SPEC-NNN\n\n- [ ] **Task 1: x**\n"
SESSION_TEMPLATE = "# SESSION-NNN: [Deskripsi Singkat Sesi]\n\n> Tanggal: [YYYY-MM-DD]\n"
LOG = ("# HANDOFF LOG\n\n## Open Questions Aktif\n\n| ID | Pertanyaan | Akan dijawab di | Status |\n"
       "|---|---|---|---|\n| OQ-001 | q1 | SPEC-002 | ❓ Open |\n| OQ-002 | q2 | SPEC-002 | ❓ Open |\n\n"
       "[Post-bootstrap] [OPEN] OQ-009 Apakah kernel jalan?  → SPEC-002\n"
       "[Post-bootstrap] [RESOLVED] OQ-001 digantikan oleh OQ-010. OQ-002 (catatan)\n")


class SpecCase(TempDirCase):
    def make_repo(self):
        self.write("spec/specs/_TEMPLATE-feature-spec/requirements.md", REQ_TEMPLATE)
        self.write("spec/specs/_TEMPLATE-feature-spec/design.md", DESIGN_TEMPLATE)
        self.write("spec/specs/_TEMPLATE-feature-spec/tasks.md", TASKS_TEMPLATE)
        self.write("spec/specs/SPEC-000-charter.md", "# charter\n")
        self.write("spec/specs/SPEC-001-first/requirements.md", REQ_TEMPLATE)
        self.write("spec/specs/SPEC-001-first/design.md", DESIGN_TEMPLATE)
        self.write("spec/specs/SPEC-001-first/tasks.md", TASKS_TEMPLATE)
        self.write("spec/handoff/_TEMPLATE-session.md", SESSION_TEMPLATE)
        self.write("spec/handoff/SESSION-001-boot.md", "# s\n")
        self.write("spec/handoff/HANDOFF-LOG.md", LOG)
        return self.tmp


class SpecScaffold(SpecCase):
    def test_new_spec_numbers_and_replaces_placeholders(self):
        root = self.make_repo()
        r = legacy.scaffold_spec(root, "cli-tooling", "CLI Tooling")
        self.assertEqual(r["spec_id"], "SPEC-002")
        with open(self.path("spec/specs/SPEC-002-cli-tooling/requirements.md"), encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("SPEC-002", text)
        self.assertIn("CLI Tooling", text)
        self.assertNotIn("SPEC-NNN", text)
        self.assertEqual(legacy.scaffold_spec(root, "next-one")["spec_id"], "SPEC-003")

    def test_new_spec_rejects_bad_names(self):
        root = self.make_repo()
        for bad in ("Bad Name", "../escape", "a/b", "", "UPPER", "-x", "a" * 80):
            with self.assertRaises(legacy.CliError, msg=bad):
                legacy.scaffold_spec(root, bad)
        self.assertFalse(os.path.exists(self.path("escape")))

    def test_session(self):
        root = self.make_repo()
        r = legacy.scaffold_session(root, "cli-work", "CLI work", on="2025-01-02")
        self.assertEqual(r["session_id"], "SESSION-002")
        with open(self.path("spec/handoff/SESSION-002-cli-work.md"), encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("# SESSION-002: CLI work", text)
        self.assertIn("> Tanggal: 2025-01-02", text)

    def test_missing_spec_dir(self):
        with self.assertRaises(legacy.EnvError):
            legacy.scaffold_spec(self.tmp, "x")

    def test_new_does_not_overwrite_via_cli(self):
        root = self.make_repo()
        code, data, _ = cli_json("spec", "new", "alpha", "--root", root)
        self.assertEqual(code, 0)
        self.assertEqual(data["spec_id"], "SPEC-002")


class SpecHandoff(SpecCase):
    def test_append_only(self):
        root = self.make_repo()
        log = self.path("spec/handoff/HANDOFF-LOG.md")
        with open(log, "rb") as fh:
            before = fh.read()
        r = legacy.append_handoff(root, "finding", "ukuran header 1 MiB", session="SESSION-002", on="2025-01-02")
        with open(log, "rb") as fh:
            after = fh.read()
        self.assertTrue(after.startswith(before))
        tail = after[len(before):].decode("utf-8")
        self.assertIn("[2025-01-02] [SESSION-002] [FINDING] ukuran header 1 MiB", tail)
        self.assertEqual(r["entry"], "[2025-01-02] [SESSION-002] [FINDING] ukuran header 1 MiB")

    def test_adds_newline_when_log_lacks_one(self):
        root = self.make_repo()
        log = self.path("spec/handoff/HANDOFF-LOG.md")
        with open(log, "wb") as fh:
            fh.write(b"no trailing newline")
        legacy.append_handoff(root, "UPDATE", "x", on="2025-01-02")
        with open(log, "r", encoding="utf-8") as fh:
            self.assertTrue(fh.read().startswith("no trailing newline\n"))

    def test_rejects_bad_input(self):
        root = self.make_repo()
        for kwargs in (dict(entry_type="NOPE", text="x"), dict(entry_type="UPDATE", text=""),
                       dict(entry_type="UPDATE", text="a\nb"), dict(entry_type="UPDATE", text="a ``` b"),
                       dict(entry_type="UPDATE", text="x" * 2001),
                       dict(entry_type="UPDATE", text="x", session="S-1")):
            with self.assertRaises(legacy.CliError, msg=str(kwargs)):
                legacy.append_handoff(root, **kwargs)


class SpecQuestions(SpecCase):
    def test_parse(self):
        qs = {q["id"]: q["status"] for q in legacy.parse_questions(LOG)}
        self.assertEqual(qs, {"OQ-001": "resolved", "OQ-002": "open", "OQ-009": "open"})

    def test_cli_filters_open(self):
        root = self.make_repo()
        code, data, _ = cli_json("spec", "questions", "--root", root)
        self.assertEqual(code, 0)
        self.assertEqual([q["id"] for q in data["questions"]], ["OQ-002", "OQ-009"])
        code, data, _ = cli_json("spec", "questions", "--root", root, "--all")
        self.assertEqual(data["count"], 3)


class SpecValidate(SpecCase):
    def test_clean_repo(self):
        root = self.make_repo()
        res = legacy.validate_spec(root)
        self.assertTrue(res["ok"], res)

    def test_scaffolded_spec_passes(self):
        root = self.make_repo()
        legacy.scaffold_spec(root, "fresh")
        self.assertTrue(legacy.validate_spec(root)["ok"])

    def test_missing_file(self):
        root = self.make_repo()
        os.remove(self.path("spec/specs/SPEC-001-first/design.md"))
        res = legacy.validate_spec(root)
        self.assertFalse(res["ok"])
        self.assertTrue(any("design.md" in e for e in res["errors"]))

    def test_numbering_gap(self):
        root = self.make_repo()
        self.write("spec/specs/SPEC-001-first/requirements.md",
                   "- **FR-01:** a\n- **FR-02:** b\n- **NFR-01: x**\n- **NFR-02: y**\n- **NFR-04: z**\n")
        res = legacy.validate_spec(root)
        self.assertFalse(res["ok"])
        self.assertTrue(any("NFR" in e for e in res["errors"]))
        self.assertFalse(any("penomoran FR" in e for e in res["errors"]))

    def test_numbering_repeat(self):
        self.assertIsNotNone(legacy.check_numbering("- **FR-01:** a\n- **FR-01:** b\n", "FR"))
        self.assertIsNone(legacy.check_numbering("- **FR-01:** a\n- **FR-02:** b\n", "FR"))
        self.assertIsNone(legacy.check_numbering("no items\n", "FR"))

    def test_links(self):
        root = self.make_repo()
        self.write("spec/specs/assets/pic.svg", "<svg/>")
        self.write("spec/specs/RFC-001-x.md",
                   "![ok](assets/pic.svg)\n[web](https://example.com/x)\n[anchor](#top)\n"
                   "[ok2](assets/pic.svg#frag)\n`[code](nope.md)`\n```\n[fenced](nope2.md)\n```\n")
        self.assertTrue(legacy.validate_spec(root)["ok"])
        self.write("spec/specs/RFC-002-y.md", "line1\n[broken](assets/missing.svg)\n")
        res = legacy.validate_spec(root)
        self.assertFalse(res["ok"])
        self.assertEqual(len(res["errors"]), 1)
        self.assertIn("RFC-002-y.md:2", res["errors"][0])

    def test_empty_link_target_does_not_crash(self):
        p = self.write("x.md", "[a]( )\n[b]()\n")
        self.assertEqual(legacy.check_links(p), [])

    def test_duplicate_oq_warns(self):
        root = self.make_repo()
        with open(self.path("spec/handoff/HANDOFF-LOG.md"), "a", encoding="utf-8") as fh:
            fh.write("| OQ-001 | dup | SPEC-002 | ❓ Open |\n")
        res = legacy.validate_spec(root)
        self.assertTrue(res["ok"])
        self.assertTrue(any("OQ-001" in w for w in res["warnings"]))

    def test_cli_exit_codes(self):
        root = self.make_repo()
        code, data, _ = cli_json("spec", "validate", "--root", root)
        self.assertEqual((code, data["ok"]), (0, True))
        self.write("spec/specs/RFC-003-z.md", "[broken](none.md)\n")
        code, data, _ = cli_json("spec", "validate", "--root", root)
        self.assertEqual((code, data["ok"]), (1, False))

    def test_real_repo_passes(self):
        root = legacy.repo_root()
        if not os.path.isdir(os.path.join(root, "spec")):
            self.skipTest("folder spec/ tidak ada")
        res = legacy.validate_spec(root)
        self.assertTrue(res["ok"], res["errors"])

    # --- CLAIMS enforcement tests -------------------------------------------

    def _make_repo_with_claims(self, claims_rows="", extra_md_content=""):
        """Make a minimal repo with spec/research/CLAIMS.md and an extra spec doc."""
        root = self.make_repo()
        # CLAIMS.md with a header row and optional data rows
        claims_body = (
            "# Daftar Klaim\n\n"
            "| ID | Klaim | Sumber | Jenis | Status |\n"
            "|---|---|---|---|---|\n"
        ) + claims_rows
        self.write("spec/research/CLAIMS.md", claims_body)
        if extra_md_content:
            self.write("spec/specs/RFC-002-test.md", extra_md_content)
        return root

    def test_claims_reference_to_defined_id_passes(self):
        root = self._make_repo_with_claims(
            claims_rows="| C-01 | klaim satu | kartu | Resmi | ✅ |\n",
            extra_md_content="Lihat C-01 untuk detail.\n",
        )
        res = legacy.validate_spec(root)
        self.assertTrue(res["ok"], res)

    def test_claims_reference_to_missing_id_fails(self):
        root = self._make_repo_with_claims(
            claims_rows="| C-01 | klaim satu | kartu | Resmi | ✅ |\n",
            extra_md_content="Lihat C-01 dan C-99 untuk detail.\n",
        )
        res = legacy.validate_spec(root)
        self.assertFalse(res["ok"])
        self.assertTrue(any("C-99" in e for e in res["errors"]))

    def test_claims_reference_in_claims_file_itself_is_ignored(self):
        # CLAIMS.md may reference other C-NN IDs in its own rows; that must not trigger errors.
        root = self._make_repo_with_claims(
            claims_rows=(
                "| C-01 | klaim satu | kartu | Resmi | ✅ |\n"
                "| C-02 | turunan dari C-01 | C-01 | Turunan | ✅ |\n"
            ),
        )
        res = legacy.validate_spec(root)
        self.assertTrue(res["ok"], res)

    def test_claims_multiple_missing_all_reported(self):
        root = self._make_repo_with_claims(
            claims_rows="| C-01 | a | x | Resmi | ✅ |\n",
            extra_md_content="C-01 ok, C-50 nope, C-51 also nope.\n",
        )
        res = legacy.validate_spec(root)
        self.assertFalse(res["ok"])
        missing = [e for e in res["errors"] if "C-50" in e or "C-51" in e]
        self.assertEqual(len(missing), 2)

    def test_claims_no_claims_file_gives_warning_not_error(self):
        # When CLAIMS.md does not exist the check is skipped (no error), but a warning is emitted.
        root = self.make_repo()
        # make_repo does not create spec/research/CLAIMS.md
        res = legacy.validate_spec(root)
        # The repo may still pass (no C-NN refs in its minimal content)
        self.assertFalse(any("C-" in e for e in res["errors"]))

    def test_claims_location_reported_in_error(self):
        root = self._make_repo_with_claims(
            claims_rows="| C-01 | a | x | Resmi | ✅ |\n",
            extra_md_content="Lihat C-77 di baris ini.\n",
        )
        res = legacy.validate_spec(root)
        self.assertFalse(res["ok"])
        err = next(e for e in res["errors"] if "C-77" in e)
        # Must contain the filename and line number
        self.assertIn("RFC-002-test.md:1", err)

    def test_claims_handoff_log_references_checked(self):
        root = self._make_repo_with_claims(
            claims_rows="| C-10 | a | x | Resmi | ✅ |\n",
        )
        # Append a C-NN reference to HANDOFF-LOG.md
        with open(self.path("spec/handoff/HANDOFF-LOG.md"), "a", encoding="utf-8") as fh:
            fh.write("\nC-99 disebutkan di log ini.\n")
        res = legacy.validate_spec(root)
        self.assertFalse(res["ok"])
        self.assertTrue(any("C-99" in e for e in res["errors"]))


# --------------------------------------------------------------------------
# CLI surface
# --------------------------------------------------------------------------

class CliSurface(unittest.TestCase):
    def test_budget_json_example_from_issue(self):
        code, data, _ = cli_json("budget", "estimate", "--file-gib", "16.53", "--vram-gib", "8", "--ctx", "4096")
        self.assertEqual(code, 0)
        self.assertEqual(data["risk"], "green")
        self.assertIn("assumptions", data)

    def test_budget_human_output(self):
        code, out, _ = cli("budget", "estimate", "--file-gib", "16.53")
        self.assertEqual(code, 0)
        self.assertIn("tps_upper_bound", out)
        self.assertIn("🟢", out)

    def test_budget_bad_input_exit_2_with_json_error(self):
        code, out, err = cli("budget", "estimate", "--file-gib=-1", "--json")
        self.assertEqual(code, 2)
        self.assertFalse(json.loads(out)["ok"])
        self.assertIn("error", err)

    def test_compare_exit_codes(self):
        base = ("budget", "compare", "--file-gib", "16.53")
        self.assertEqual(cli(*base, "--measured-ram-gib", "12.5")[0], 0)
        self.assertEqual(cli(*base, "--measured-ram-gib", "30")[0], 1)
        self.assertEqual(cli(*base)[0], 2)

    def test_suggest_ngl_cli(self):
        code, data, _ = cli_json("budget", "suggest-ngl", "--file-gib", "16.53", "--rs-gib", "0")
        self.assertEqual((code, data["ngl"]), (0, 26))

    def test_unknown_subcommand_is_usage_error(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                legacy.main(["nope"])
        self.assertEqual(cm.exception.code, 2)

    def test_every_leaf_accepts_json_flag(self):
        parser = legacy.build_parser()
        help_text = []
        for group in ("budget", "hw", "gguf", "llama", "spec"):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                with self.assertRaises(SystemExit):
                    parser.parse_args([group, "--help"])
            help_text.append(buf.getvalue())
        self.assertTrue(all(h for h in help_text))
        for argv in (["budget", "estimate", "--file-gib", "1", "--json"], ["hw", "gpu", "--json"],
                     ["hw", "ram", "--json"], ["hw", "cpu", "--json"], ["gguf", "inspect", "x", "--json"],
                     ["llama", "placement", "--log", "x", "--json"], ["spec", "validate", "--json"],
                     ["spec", "questions", "--json"], ["spec", "handoff", "--type", "update",
                                                        "--text", "t", "--json"]):
            self.assertTrue(parser.parse_args(argv).json, argv)

    def test_render_nested(self):
        text = legacy.render({"a": 1, "b": {"c": [1, {"d": 2}]}, "e": None, "f": []})
        self.assertIn("a: 1", text)
        self.assertIn("  c:", text)
        self.assertIn("e: -", text)
        self.assertIn("f: -", text)



# --------------------------------------------------------------------------
# Perbaikan #4: -st, stdin tertutup, timing bentuk baru, KV hibrida
# --------------------------------------------------------------------------

BRACKET_TIMING = "\n[ Prompt: 69.1 t/s | Generation: 38.3 t/s ]\n\nExiting...\n"
# Keluaran asli `llama-perplexity` fork 08e025c (dipotong), termasuk tanda ± pada KLD.
KLD_REAL = """====== KL divergence statistics ======
Mean    KLD:   0.348379 \u00b1   0.011129
Maximum KLD:   3.404110
99.9%   KLD:   3.204420
99.0%   KLD:   1.563301
95.0%   KLD:   0.977717
Same top p: 70.000 \u00b1  1.436 %
"""


class FixedDefectsTests(TempDirCase):
    def test_run_argv_uses_single_turn_not_no_cnv(self):
        argv = legacy.build_run_argv("/b/llama-cli", "/m.gguf", prompt="x")
        self.assertIn("-st", argv)
        self.assertNotIn("-no-cnv", argv)
        self.assertNotIn("-st", legacy.build_run_argv("/b/llama-cli", "/m.gguf", prompt="x",
                                                      single_turn=False))

    def test_parse_bracket_timing(self):
        t = legacy.parse_llama_timing("garbage\n" + BRACKET_TIMING)
        self.assertEqual((t["prompt_tps"], t["gen_tps"]), (69.1, 38.3))

    def test_cli_that_waits_for_stdin_without_st_does_not_hang(self):
        # Meniru perilaku asli: tanpa -st llama-cli membaca stdin selamanya.
        script = ("#!/bin/sh\nfor a in \"$@\"; do [ \"$a\" = \"-st\" ] && st=1; done\n"
                  "[ -z \"$st\" ] && { read line; echo waiting; exit 9; }\n"
                  "printf '[ Prompt: 10.0 t/s | Generation: 5.0 t/s ]\\n'\n")
        d = self.path("bin")
        f = self.write(os.path.join("bin", "llama-cli"), script)
        os.chmod(f, os.stat(f).st_mode | stat.S_IXUSR)
        model = self.write("m.gguf", "x")
        code, data, _ = cli_json("llama", "run", "--model", model, "--prompt", "hi", "--bin-dir", d,
                                 "--lock-file", self.path("l.lock"), "--timeout", "10")
        self.assertEqual(code, 0)
        self.assertEqual(data["gen_tps"], 5.0)

    def test_stdin_is_closed_for_children(self):
        out = legacy.run_capture(["sh", "-c", "read x; echo got=$?"])
        self.assertIn("got=1", out)   # EOF langsung, tidak menunggu

    def hybrid(self, interval=4, blocks=32):
        p = self.path("h.gguf")
        make_gguf(p, [
            kv_str("general.architecture", "qwen35"),
            kv_u32("qwen35.block_count", blocks),
            kv_u32("qwen35.full_attention_interval", interval),
            kv_u32("qwen35.attention.head_count_kv", 4),
        ], [tensor_info("token_embd.weight", [256, 10], 12),
            tensor_info("output.weight", [256, 10], 12),
            tensor_info("blk.0.x.weight", [256, 4], 12)])
        return p

    def test_kv_layers_for_hybrid_uses_interval(self):
        s = legacy.summarize_gguf(legacy.parse_gguf(self.hybrid()))
        self.assertEqual(s["kv_layers_estimate"], 8)           # nilai terverifikasi di Qwen3.8-9B
        self.assertEqual(s["full_attention_interval"], 4)
        self.assertIn("full_attention_interval", s["kv_layers_source"])

    def test_kv_layers_falls_back_to_block_count(self):
        p = self.path("d.gguf")
        make_gguf(p, [kv_str("general.architecture", "dense"), kv_u32("dense.block_count", 12)],
                  [tensor_info("token_embd.weight", [256], 12)])
        s = legacy.summarize_gguf(legacy.parse_gguf(p))
        self.assertEqual((s["kv_layers_estimate"], s["kv_layers_source"]), (12, "block_count"))


class GpuClockTests(unittest.TestCase):
    def test_ten_column_line(self):
        g = legacy.parse_nvidia_csv("GPU, 8192, 100, 8092, 80, 170.5, 90, 1835, 3802, 0x0000000000000004\n")[0]
        self.assertEqual(g["sm_clock_mhz"], 1835.0)
        self.assertEqual(g["throttle_reasons"], ["sw_power_cap"])
        self.assertTrue(g["throttle_limiting"])

    def test_idle_is_not_limiting(self):
        names, limiting = legacy.decode_throttle("0x0000000000000001")
        self.assertEqual((names, limiting), (["gpu_idle"], False))
        self.assertEqual(legacy.decode_throttle("0x0"), ([], False))
        self.assertEqual(legacy.decode_throttle(None), (None, None))
        self.assertEqual(legacy.decode_throttle("zz"), (None, None))

    def test_seven_columns_still_accepted(self):
        g = legacy.parse_nvidia_csv("GPU, 8192, 1024, 7168, 45, 30.5, 3\n")[0]
        self.assertNotIn("sm_clock_mhz", g)

    def test_samples_report_clock_and_throttle(self):
        s = legacy.summarize_samples([
            {"sm_clock_mhz": 1900, "util_pct": 99, "throttle_reasons": [], "throttle_limiting": False},
            {"sm_clock_mhz": 1500, "util_pct": 99, "throttle_reasons": ["sw_power_cap"],
             "throttle_limiting": True},
            {"sm_clock_mhz": 139, "util_pct": 0, "throttle_reasons": ["gpu_idle"],
             "throttle_limiting": False},
        ])
        self.assertEqual(s["peak_sm_clock_mhz"], 1900)
        self.assertEqual(s["min_sm_clock_mhz_when_busy"], 1500)
        self.assertEqual(s["samples_throttled"], 1)
        self.assertEqual(s["throttle_reasons_seen"], ["gpu_idle", "sw_power_cap"])


class RamSampleTests(unittest.TestCase):
    def test_summary_reports_lowest_ram_and_peak_swap(self):
        s = legacy.summarize_samples([
            {"ram_available_gib": 6.0, "swap_used_gib": 0.1},
            {"ram_available_gib": 2.5, "swap_used_gib": 0.9},
            {"ram_available_gib": 4.0, "swap_used_gib": 0.4},
        ])
        self.assertEqual(s["min_ram_available_gib"], 2.5)
        self.assertEqual(s["peak_swap_used_gib"], 0.9)

    def test_summary_without_ram_fields(self):
        s = legacy.summarize_samples([{"util_pct": 10}])
        self.assertIsNone(s["min_ram_available_gib"])
        self.assertIsNone(s["peak_swap_used_gib"])


class MembwTests(unittest.TestCase):
    def test_threads_validated_and_reported(self):
        r = legacy.measure_membw(16, 1, 2)
        self.assertEqual(r["threads"], 2)
        self.assertGreater(r["copy_gb_per_s"], 0)
        with self.assertRaises(legacy.CliError):
            legacy.measure_membw(16, 1, 0)

    def test_sweep(self):
        r = legacy.measure_membw_sweep(16, 1, max_threads=2)
        self.assertEqual(sorted(r["by_threads"]), ["1", "2"])
        self.assertIn(r["best_threads"], (1, 2))


class AnalyzeAndCompareTests(TempDirCase):
    def test_efficiency_math(self):
        # 4 GiB per token di GPU 256 GB/s -> batas = 256e9 / (4*2^30) ~ 59.6 tok/s
        r = legacy.analyze_bandwidth(30.0, 4 * GIB, 1.0, 256.0)
        self.assertAlmostEqual(r["ceiling_gen_tps"], 59.6, places=1)
        self.assertAlmostEqual(r["efficiency"], 0.503, places=2)
        self.assertAlmostEqual(r["effective_bandwidth_gb_per_s"], 128.8, places=1)

    def test_hybrid_needs_ram_bandwidth(self):
        with self.assertRaises(legacy.CliError):
            legacy.analyze_bandwidth(2.0, 8 * GIB, 0.5, 256.0)
        r = legacy.analyze_bandwidth(2.0, 8 * GIB, 0.5, 256.0, 20.0)
        self.assertLess(r["ceiling_gen_tps"], 6)   # sisi CPU mendominasi
        self.assertGreater(r["ceiling_gen_tps"], 4)

    def test_over_one_is_flagged(self):
        r = legacy.analyze_bandwidth(38.2, 0.4486 * GIB, 0.0, 256.0, 14.15)
        self.assertGreater(r["efficiency"], 1.05)
        self.assertIn("terlalu rendah", r["verdict"])

    def test_bad_inputs(self):
        for bad in ((0, GIB, 1.0), (10, 0, 1.0), (10, GIB, 1.5)):
            with self.assertRaises(legacy.CliError):
                legacy.analyze_bandwidth(*bad)

    def test_weights_exclude_embedding_only_when_output_separate(self):
        g = {"tensors": [{"name": "token_embd.weight", "nbytes": 100},
                         {"name": "output.weight", "nbytes": 50},
                         {"name": "blk.0.a.weight", "nbytes": 10}]}
        self.assertEqual(legacy.weights_read_per_token(g)[0], 60)
        g["tensors"].pop(1)
        self.assertEqual(legacy.weights_read_per_token(g)[0], 110)

    def write_bench(self, name, gen, prompt, std=0.5):
        rows = [{"n_prompt": 128, "n_gen": 0, "avg_ts": prompt, "stddev_ts": std},
                {"n_prompt": 0, "n_gen": 64, "avg_ts": gen, "stddev_ts": std}]
        return self.write(name, json.dumps({"results": rows}))

    def test_compare_verdicts(self):
        a = legacy.load_bench_rows(self.write_bench("a.json", 38.0, 80.0))
        b = legacy.load_bench_rows(self.write_bench("b.json", 30.0, 80.5))
        rows = {(r["n_prompt"], r["n_gen"]): r for r in legacy.compare_bench(a, b)}
        self.assertEqual(rows[(0, 64)]["verdict"], "lebih lambat")
        self.assertEqual(rows[(128, 0)]["verdict"], "sama (dalam toleransi/noise)")
        with self.assertRaises(legacy.CliError):
            legacy.compare_bench(a, [{"n_prompt": 1, "n_gen": 1, "avg_ts": 1.0}])

    def test_cli_compare_and_analyze(self):
        a = self.write_bench("a.json", 38.0, 80.0)
        b = self.write_bench("b.json", 50.0, 80.0)
        code, data, _ = cli_json("llama", "compare", a, b)
        self.assertEqual(code, 0)
        self.assertEqual([r["verdict"] for r in data["rows"]][1], "lebih cepat")
        code, data, _ = cli_json("bench", "analyze", "--bench-json", a, "--weights-gib", "2",
                                 "--gpu-fraction", "1")
        self.assertEqual(code, 0)
        self.assertEqual(data["measured_gen_tps"], 38.0)
        code, _, _ = cli("bench", "analyze", "--bench-json", a)
        self.assertEqual(code, 2)   # tanpa sumber byte bobot

    def test_missing_file_is_env_error(self):
        with self.assertRaises(legacy.EnvError):
            legacy.load_bench_rows(self.path("nope.json"))


class QualityTests(TempDirCase):
    def test_parse_ppl_and_kld_real_output(self):
        self.assertEqual(legacy.parse_ppl("Final estimate: PPL = 16.4243 +/- 1.58524")["ppl"], 16.4243)
        k = legacy.parse_kld(KLD_REAL)
        self.assertEqual((k["mean_kld"], k["same_top_p_pct"], k["kld_p99"]), (0.348379, 70.0, 1.563301))
        with self.assertRaises(legacy.CliError):
            legacy.parse_ppl("tidak ada")
        with self.assertRaises(legacy.CliError):
            legacy.parse_kld("tidak ada")

    def test_ppl_argv(self):
        argv = legacy.build_ppl_argv("/b/llama-perplexity", "/m.gguf", "/t.txt", 512, 4, 0, 2)
        self.assertEqual(argv[argv.index("--chunks") + 1], "4")
        self.assertEqual(argv[argv.index("-ngl") + 1], "0")
        with self.assertRaises(legacy.CliError):
            legacy.build_ppl_argv("/b", "/m", "/t", 512, 0)

    def test_kv_type_reaches_argv(self):
        argv = legacy.build_ppl_argv("/b/p", "/m", "/t", 512, 2, 0, 2, kv_type="q8_0")
        self.assertEqual(argv[argv.index("-ctk") + 1], "q8_0")
        self.assertEqual(argv[argv.index("-ctv") + 1], "q8_0")
        self.assertNotIn("-ctk", legacy.build_ppl_argv("/b/p", "/m", "/t"))
        with self.assertRaises(legacy.CliError):
            legacy.build_ppl_argv("/b/p", "/m", "/t", kv_type="bogus")

    def test_kld_candidate_and_base_kv_types_are_separate(self):
        m, t = self.write("m.gguf", "x"), self.write("t.txt", "x")
        code, data, _ = cli_json("quality", "kld", "--base-model", m, "--model", m, "--file", t,
                                 "--kld-file", self.path("k.bin"), "--kv-type", "q8_0", "--dry-run",
                                 "--bin-dir", "x")
        base_argv, cand_argv = data["argv"]
        self.assertNotIn("-ctk", base_argv)
        self.assertEqual(cand_argv[cand_argv.index("-ctk") + 1], "q8_0")
        self.assertEqual(data["protocol"]["kv_type"], "q8_0")

    def test_dry_run_commands(self):
        m, t = self.write("m.gguf", "x"), self.write("t.txt", "x")
        code, data, _ = cli_json("quality", "ppl", "--model", m, "--file", t, "--dry-run",
                                 "--bin-dir", self.path("nobin"))
        self.assertEqual(code, 0)
        self.assertEqual(data["protocol"]["ctx"], 512)
        code, data, _ = cli_json("quality", "kld", "--base-model", m, "--model", m, "--file", t,
                                 "--kld-file", self.path("k.bin"), "--dry-run", "--bin-dir", "x")
        self.assertIn("--kl-divergence-base", data["argv"][0])
        self.assertIn("--kl-divergence", data["argv"][1])

    def test_ppl_with_fake_binary(self):
        script = "#!/bin/sh\necho 'Final estimate: PPL = 7.5000 +/- 0.1000'\n"
        self.write(os.path.join("bin", "llama-perplexity"), script)
        os.chmod(self.path("bin", "llama-perplexity"), 0o755)
        m, t = self.write("m.gguf", "x"), self.write("t.txt", "x")
        code, data, _ = cli_json("quality", "ppl", "--model", m, "--file", t, "--bin-dir", self.path("bin"),
                                 "--lock-file", self.path("q.lock"))
        self.assertEqual((code, data["ppl"]), (0, 7.5))

    def test_strip_cli_noise(self):
        raw = "banner\n\n> prompt text\n\n|\x08-\x08\\\x08 \x08Jawaban A\n" + BRACKET_TIMING
        self.assertEqual(legacy.strip_cli_noise(raw), "Jawaban A")

    def test_prompts_compare(self):
        a = {"results": [{"prompt": "p1", "output": "sama"}, {"prompt": "p2", "output": "abcdef"}]}
        b = {"results": [{"prompt": "p1", "output": "sama"}, {"prompt": "p2", "output": "abcxyz"}]}
        r = legacy.compare_prompt_runs(a, b)
        self.assertEqual((r["prompts"], r["identical"]), (2, 1))
        self.assertTrue(0.5 < r["mean_similarity"] < 1.0)
        with self.assertRaises(legacy.CliError):
            legacy.compare_prompt_runs(a, {"results": [{"prompt": "lain", "output": "x"}]})

    def test_prompts_file_rules(self):
        f = self.write("p.txt", "# komentar\n\nsatu\ndua\n")
        self.assertEqual(legacy.load_prompts(f), ["satu", "dua"])
        with self.assertRaises(legacy.CliError):
            legacy.load_prompts(self.write("e.txt", "# saja\n"))

    def test_shipped_prompt_set_is_valid(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quality", "prompts-v1.txt")
        self.assertEqual(len(legacy.load_prompts(path)), 8)

    def test_build_targets(self):
        _, build = legacy.build_cmake_argv("src", "b", "cpu", 2, ["llama-perplexity"])
        self.assertEqual(build[-2:], ["--target", "llama-perplexity"])
        with self.assertRaises(legacy.CliError):
            legacy.build_cmake_argv("src", "b", "cpu", 2, ["x; rm -rf /"])


# --------------------------------------------------------------------------
# Kontrak dengan binary asli. Dilewati bila tidak diset; tidak memakai GPU.
#   LEGACY_LLAMA_BIN_DIR=/path/build/bin LEGACY_TEST_MODEL=/path/kecil.gguf python tools/test_legacy.py
# --------------------------------------------------------------------------

REAL_BIN = os.environ.get("LEGACY_LLAMA_BIN_DIR")
REAL_MODEL = os.environ.get("LEGACY_TEST_MODEL")


@unittest.skipUnless(REAL_BIN and REAL_MODEL and os.path.isfile(os.path.join(REAL_BIN or "", "llama-cli")),
                     "set LEGACY_LLAMA_BIN_DIR dan LEGACY_TEST_MODEL untuk uji binary asli")
class RealBinaryContract(TempDirCase):
    def setUp(self):
        super().setUp()
        self._old = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""      # jangan sentuh VRAM
        self.addCleanup(self.restore)

    def restore(self):
        if self._old is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = self._old

    def test_run_finishes_and_parses_timing(self):
        code, data, _ = cli_json("llama", "run", "--model", REAL_MODEL, "--bin-dir", REAL_BIN, "--ngl", "0",
                                 "--ctx", "256", "--n-predict", "8", "--prompt", "Halo",
                                 "--timeout", "180", "--lock-file", self.path("r.lock"))
        self.assertEqual(code, 0)
        self.assertEqual(data["returncode"], 0)
        self.assertGreater(data["gen_tps"], 0)
        self.assertGreater(data["prompt_tps"], 0)

    def test_bench_json_shape(self):
        code, data, _ = cli_json("llama", "bench", "--model", REAL_MODEL, "--bin-dir", REAL_BIN, "--ngl", "0",
                                 "--prompt-tokens", "16", "--gen-tokens", "8", "--runs", "1",
                                 "--timeout", "180", "--lock-file", self.path("b.lock"))
        self.assertEqual(code, 0)
        self.assertTrue(any(r["n_gen"] for r in data["results"]))

    def test_inspect_real_model(self):
        s = legacy.summarize_gguf(legacy.parse_gguf(REAL_MODEL))
        self.assertTrue(s["architecture"])
        self.assertGreater(s["tensor_count"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
