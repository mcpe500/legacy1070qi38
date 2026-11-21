#!/usr/bin/env python3
"""Writes train_bonsai_mtp_colab.ipynb (Google Colab, RTX PRO 6000 96 GB) from the cell sources below.
Edit the cells here, then run:  python3 make_colab_notebook.py && python3 check_colab_notebook.py
Every code cell is plain Python (no ! or % magics), so check_colab_notebook.py can parse it and compare each CLI call
with the argparse options of the scripts in this folder."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "train_bonsai_mtp_colab.ipynb")
COLAB_URL = ("https://colab.research.google.com/github/mcpe500/legacy1070qi38/blob/main/"
             "qwen38-engine/tools/bonsai_mtp_train/train_bonsai_mtp_colab.ipynb")

INTRO = f"""# Ternary Bonsai 2 27B: train the MTP draft head (blk.64) on Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_URL})

This notebook trains a multi-token-prediction head (`blk.64`, one decoder layer + `nextn` projection, 424.7M parameters) for
**Ternary Bonsai 2 27B** by live online distillation, so that `qwen38-engine` can draft with it. The output is a small GGUF
with only the `blk.64` tensors (Q4_K_M about 273 MiB, Q8_0 about 450 MiB), which you inject into your Bonsai GGUF at home.

**Runtime:** *Runtime → Change runtime type →* a GPU with at least 80 GB (RTX PRO 6000 96 GB). The teacher alone is 50.1 GiB in BF16.

| Cell | What it does | Where data goes |
|---|---|---|
| 1 | preflight (GPU, RAM, free disk on `/content` ≥ 70 GB), mount Google Drive, helpers | `DRIVE_OUT/logs/` |
| 2 | clone / pull `mcpe500/legacy1070qi38`, install Python packages, script self-tests | `/content/legacy1070qi38` |
| 3 | download the Bonsai GGUF (PTQ1_0 or Q2_0), optional warm start from Qwen3.8-27B's own `blk.64` | `/content/models/` (local SSD), init on Drive |
| 4 | GGUF → Hugging Face BF16 teacher (`convert_gguf_to_hf_bf16.py`, about 35 min) | `/content/models/bonsai-hf-bf16` (local SSD, not Drive) |
| 5 | streamed multilingual corpus (`prepare_corpus.py`), then training (`train_bonsai_mtp.py`) | corpus and checkpoints on Drive |
| 6 | export the final `blk.64` (safetensors + GGUF Q4_K_M / Q8_0) and print the commands for your PC | `DRIVE_OUT/final/` |

**After a disconnect:** run cells 1 to 5 again. The Bonsai GGUF is downloaded again and the teacher converted again (local disk
is wiped). The corpus, the warm-start file and the training state are on Drive, so training resumes from the last saved step
(`--resume-from auto`).

**Drive space:** about 5 GB for the training state (`train_state.pt` 1.7 GB + the newest `--keep-last` checkpoints of 1.7 GB each),
1.7 GB for the warm-start file, 1 to 3 GB for the corpus, under 1 GB for the final export.

Gated datasets (`bigcode/starcoderdata`, `uonlp/CulturaX`) need a Hugging Face token whose account accepted their terms. Put it in
Colab *Secrets* as `HF_TOKEN` (or in the field of cell 5). Without it those sources are skipped and the mix shifts.
"""

CELL1 = r'''#@title 1. Preflight: GPU, RAM, disk, Google Drive, helpers
DRIVE_OUT = "/content/drive/MyDrive/bonsai_mtp"  #@param {type:"string"}
MOUNT_DRIVE = True  #@param {type:"boolean"}
MIN_FREE_GB = 70  #@param {type:"integer"}
MIN_VRAM_GB = 80  #@param {type:"integer"}
MIN_RAM_GB = 24  #@param {type:"integer"}

import collections
import datetime
import json
import os
import shutil
import subprocess
import sys
import time

PY = sys.executable
WORK = "/content"
MODELS = "/content/models"
REPO_DIR = "/content/legacy1070qi38"
TOOLS = f"{REPO_DIR}/qwen38-engine/tools/bonsai_mtp_train"


def run(cmd, name, cwd=None, env=None, ok_codes=(0,)):
    """Runs cmd (a list), streams its output here and into DRIVE_OUT/logs/<name>.log, raises on a bad exit code."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log = os.path.join(LOG_DIR, f"{name}.log")
    full_env = dict(os.environ, PYTHONUNBUFFERED="1", **(env or {}))
    tail = collections.deque(maxlen=40)
    t0 = time.time()
    print(f"$ {' '.join(map(str, cmd))}\n  log: {log}", flush=True)
    with open(log, "a", encoding="utf-8") as lf:
        lf.write(f"\n===== {datetime.datetime.now().isoformat()} $ {' '.join(map(str, cmd))}\n")
        p = subprocess.Popen([str(c) for c in cmd], cwd=cwd, env=full_env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace")
        for line in p.stdout:
            print(line, end="", flush=True)
            lf.write(line)
            tail.append(line)
        rc = p.wait()
        lf.write(f"===== exit code {rc} after {time.time() - t0:.0f} s\n")
    if rc not in ok_codes:
        raise RuntimeError(f"[{name}] failed with exit code {rc} after {time.time() - t0:.0f} s. Last lines:\n"
                           + "".join(tail) + f"\nFull log: {log}")
    print(f"[{name}] OK in {time.time() - t0:.0f} s", flush=True)
    return rc


def free_gb(path=WORK):
    return shutil.disk_usage(path).free / 1e9


# --- GPU
try:
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True).stdout.strip().splitlines()
except (OSError, subprocess.CalledProcessError) as e:
    raise RuntimeError("No NVIDIA GPU visible. Runtime -> Change runtime type -> select the RTX PRO 6000 (96 GB) GPU.") from e
gpu_name, gpu_mib, driver = [x.strip() for x in q[0].split(",")]
gpu_gb = float(gpu_mib) / 1024
print(f"GPU: {gpu_name}, {gpu_gb:.1f} GiB, driver {driver} ({len(q)} GPU(s))")
if gpu_gb < MIN_VRAM_GB:
    raise RuntimeError(f"{gpu_name} has {gpu_gb:.1f} GiB; the BF16 teacher (50.1 GiB) plus training needs >= {MIN_VRAM_GB} GiB.")

# --- RAM
with open("/proc/meminfo") as f:
    mem = {l.split(":")[0]: int(l.split()[1]) for l in f}
ram_gb = mem["MemTotal"] / 1e6
print(f"RAM: {ram_gb:.1f} GB total, {mem['MemAvailable'] / 1e6:.1f} GB available")
if ram_gb < MIN_RAM_GB:
    raise RuntimeError(f"{ram_gb:.1f} GB RAM; the converter and the teacher load need >= {MIN_RAM_GB} GB.")

# --- disk (GGUF 6-7.6 GB + BF16 teacher 53.8 GB + optional Qwen GGUF 16.6 GB, deleted after use)
disk_gb = free_gb()
print(f"Disk /content: {disk_gb:.1f} GB free")
if disk_gb < MIN_FREE_GB and not os.path.exists(f"{MODELS}/bonsai-hf-bf16/conversion.json"):
    raise RuntimeError(f"/content has {disk_gb:.1f} GB free, needs >= {MIN_FREE_GB} GB (teacher 53.8 GB + GGUF + temp).")

# --- Google Drive
if MOUNT_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    if not os.path.isdir("/content/drive/MyDrive"):
        raise RuntimeError("Google Drive did not mount at /content/drive/MyDrive.")
elif DRIVE_OUT.startswith("/content/drive"):
    raise RuntimeError("MOUNT_DRIVE is off but DRIVE_OUT points into /content/drive; set a local path or mount Drive.")
for sub in ("logs", "run", "init", "final", "corpus"):
    os.makedirs(os.path.join(DRIVE_OUT, sub), exist_ok=True)
LOG_DIR = os.path.join(DRIVE_OUT, "logs")
os.makedirs(MODELS, exist_ok=True)
print(f"DRIVE_OUT = {DRIVE_OUT} ({shutil.disk_usage(DRIVE_OUT).free / 1e9:.1f} GB free there)")
print("preflight OK")
'''

CELL2 = r'''#@title 2. Repository and Python packages
REPO_URL = "https://github.com/mcpe500/legacy1070qi38.git"  #@param {type:"string"}
BRANCH = "main"  #@param {type:"string"}
INSTALL_FLA = True  #@param {type:"boolean"}
INSTALL_FLASH_ATTN = False  #@param {type:"boolean"}
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"  #@param {type:"string"}
RUN_SELFTESTS = True  #@param {type:"boolean"}

# --- repository
if os.path.isdir(os.path.join(REPO_DIR, ".git")):
    run(["git", "-C", REPO_DIR, "fetch", "--depth", "1", "origin", BRANCH], "git_fetch")
    run(["git", "-C", REPO_DIR, "checkout", "-B", BRANCH, f"origin/{BRANCH}"], "git_checkout")
else:
    run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, REPO_DIR], "git_clone")
run(["git", "-C", REPO_DIR, "log", "-1", "--format=%h %s"], "git_head")
for f in ("convert_gguf_to_hf_bf16.py", "prepare_corpus.py", "train_bonsai_mtp.py", "inject_mtp_gguf.py"):
    if not os.path.exists(os.path.join(TOOLS, f)):
        raise RuntimeError(f"{TOOLS}/{f} is missing: wrong BRANCH or REPO_URL?")

# --- torch: keep Colab's build when it runs on this GPU, else install the CUDA 12.8 wheels (Blackwell = sm_120)
TORCH_CHECK = ("import torch; assert torch.cuda.is_available(); x = torch.randn(256, 256, device='cuda', dtype=torch.bfloat16); "
               "y = (x @ x).float().sum().item(); cap = torch.cuda.get_device_capability(); "
               "print('torch', torch.__version__, 'cuda', torch.version.cuda, 'capability', cap, 'bf16 matmul ok')")
try:
    run([PY, "-c", TORCH_CHECK], "torch_check")
except RuntimeError:
    print("the installed torch does not run on this GPU: installing the CUDA 12.8 build")
    run([PY, "-m", "pip", "install", "-q", "--upgrade", "torch", "--index-url", TORCH_INDEX_URL], "pip_torch")
    run([PY, "-c", TORCH_CHECK], "torch_check")

# --- libraries (transformers >= 5 has Qwen3_5ForCausalLM, the teacher architecture)
run([PY, "-m", "pip", "install", "-q", "--upgrade", "transformers>=5.0", "datasets", "safetensors", "huggingface_hub",
     "accelerate", "gguf", "numpy", "tokenizers"], "pip_libs")
run([PY, "-c", "import transformers; from transformers import Qwen3_5ForCausalLM; print('transformers', transformers.__version__)"],
    "check_transformers")

# --- optional kernels. flash-linear-attention: Triton kernels for the 48 Gated DeltaNet layers of the teacher (much faster
# than the PyTorch fallback). flash-attn: only for the 16 full-attention layers; SDPA is used when it is missing.
if INSTALL_FLA:
    try:
        run([PY, "-m", "pip", "install", "-q", "flash-linear-attention"], "pip_fla")
        run([PY, "-c", "import fla; print('flash-linear-attention', getattr(fla, '__version__', '?'))"], "check_fla")
    except RuntimeError as e:
        print(f"WARNING: flash-linear-attention not available, the teacher uses the slower PyTorch path.\n{e}")
ATTN_IMPL = "sdpa"
if INSTALL_FLASH_ATTN:
    try:
        run([PY, "-m", "pip", "install", "-q", "flash-attn", "--no-build-isolation"], "pip_flash_attn")
        run([PY, "-c", "import flash_attn; print('flash-attn', flash_attn.__version__)"], "check_flash_attn")
        ATTN_IMPL = "flash_attention_2"
    except RuntimeError as e:
        print(f"WARNING: flash-attn not available, falling back to SDPA.\n{e}")
print(f"teacher attention implementation: {ATTN_IMPL}")

# --- CPU self-tests of the four scripts (shapes, quantizers, rotation inverse, corpus formatters)
if RUN_SELFTESTS:
    cpu = {"CUDA_VISIBLE_DEVICES": ""}
    run([PY, f"{TOOLS}/train_bonsai_mtp.py", "--selftest"], "selftest_train", env=cpu)
    run([PY, f"{TOOLS}/convert_gguf_to_hf_bf16.py", "--selftest"], "selftest_convert", env=cpu)
    run([PY, f"{TOOLS}/inject_mtp_gguf.py", "--selftest"], "selftest_inject", env=cpu)
    run([PY, f"{TOOLS}/prepare_corpus.py", "--selftest"], "selftest_corpus", env=cpu)
print("setup OK")
'''

CELL3 = r'''#@title 3. Download the Bonsai GGUF (and the optional warm-start head) to local disk
QUANT = "PTQ1_0"  #@param ["PTQ1_0", "Q2_0"]
INIT_MODE = "qwen_blk64"  #@param ["random", "qwen_blk64", "drive_safetensors"]
INIT_SAFETENSORS = ""  #@param {type:"string"}

BONSAI_REPO = "prism-ml/Ternary-Bonsai-2-27B-gguf"
GGUF_NAME = f"Ternary-Bonsai-2-27B-{QUANT}.gguf"
GGUF_PATH = f"{MODELS}/{GGUF_NAME}"
QWEN_REPO, QWEN_NAME = "unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-Q4_K_M.gguf"
INIT_DRIVE = f"{DRIVE_OUT}/init/mtp_init_qwen_blk64.safetensors"

DOWNLOAD = r"""
import os, sys
from huggingface_hub import HfApi, hf_hub_download
repo, name, dst = sys.argv[1], sys.argv[2], sys.argv[3]
info = HfApi().get_paths_info(repo, [name])
if not info:
    sys.exit(f"{name} not found in {repo}")
size = info[0].size
path = os.path.join(dst, name)
if os.path.exists(path) and os.path.getsize(path) == size:
    print(f"{path}: already complete ({size / 1e9:.2f} GB), skipping")
    sys.exit(0)
print(f"downloading {repo}/{name} ({size / 1e9:.2f} GB) to {dst}")
p = hf_hub_download(repo, name, local_dir=dst)
got = os.path.getsize(p)
if got != size:
    sys.exit(f"size mismatch: {got} != {size}")
print(f"{p}: {got / 1e9:.2f} GB OK")
"""

run([PY, "-c", DOWNLOAD, BONSAI_REPO, GGUF_NAME, MODELS], f"download_{QUANT}")

INIT_ARGS = []
if INIT_MODE == "qwen_blk64":
    # Qwen3.8-27B ships a trained blk.64 with the same shapes; it is a better start than random weights. The 16.6 GB GGUF is
    # only needed once: the dequantized head (1.7 GB) is kept on Drive and the GGUF is deleted.
    if not os.path.exists(INIT_DRIVE):
        if free_gb() < 20:
            raise RuntimeError(f"{free_gb():.1f} GB free on /content, the Qwen GGUF needs 17 GB (or set INIT_MODE = random)")
        run([PY, "-c", DOWNLOAD, QWEN_REPO, QWEN_NAME, MODELS], "download_qwen")
        tmp = f"{MODELS}/mtp_init_qwen_blk64.safetensors"
        run([PY, f"{TOOLS}/train_bonsai_mtp.py", "--export-init", f"gguf:{MODELS}/{QWEN_NAME}", "--out", tmp],
            "export_init", env={"CUDA_VISIBLE_DEVICES": ""})
        shutil.copy(tmp, INIT_DRIVE + ".tmp")
        os.replace(INIT_DRIVE + ".tmp", INIT_DRIVE)
        os.remove(tmp)
        os.remove(f"{MODELS}/{QWEN_NAME}")
    else:
        print(f"warm start: {INIT_DRIVE} already on Drive")
    INIT_ARGS = ["--init-mtp-safetensors", INIT_DRIVE]
elif INIT_MODE == "drive_safetensors":
    if not os.path.exists(INIT_SAFETENSORS):
        raise RuntimeError(f"INIT_SAFETENSORS = {INIT_SAFETENSORS!r} does not exist")
    INIT_ARGS = ["--init-mtp-safetensors", INIT_SAFETENSORS]
print(f"GGUF: {GGUF_PATH} ({os.path.getsize(GGUF_PATH) / 1e9:.2f} GB); init: {INIT_ARGS or 'random'}; {free_gb():.1f} GB free")
'''

CELL4 = r'''#@title 4. Convert the GGUF into the BF16 Hugging Face teacher (local SSD, about 35 min)
TEACHER_DIR = "/content/models/bonsai-hf-bf16"  #@param {type:"string"}
CHECK_INVERSE = True  #@param {type:"boolean"}
DELETE_GGUF_AFTER = False  #@param {type:"boolean"}

if TEACHER_DIR.startswith("/content/drive"):
    raise RuntimeError("TEACHER_DIR must be on the local disk (/content/...): 50.1 GiB would eat the Drive quota.")
cpu = {"CUDA_VISIBLE_DEVICES": ""}
if CHECK_INVERSE:   # numeric check of the inverse transform on real tensors (W_plain x == W' H(s x)), about 1 min
    run([PY, f"{TOOLS}/convert_gguf_to_hf_bf16.py", "--check", "--gguf", GGUF_PATH], "convert_check", env=cpu)
done = os.path.exists(os.path.join(TEACHER_DIR, "conversion.json"))
if not done and free_gb() < 56:
    raise RuntimeError(f"{free_gb():.1f} GB free on /content; the BF16 teacher needs 54 GB")
# skips by itself when TEACHER_DIR already holds a complete conversion of this GGUF
run([PY, f"{TOOLS}/convert_gguf_to_hf_bf16.py", "--gguf", GGUF_PATH, "--out", TEACHER_DIR, "--dtype", "bf16"], "convert",
    env=cpu)
with open(os.path.join(TEACHER_DIR, "conversion.json")) as f:
    conv = json.load(f)
if conv["layers"] != 64 or conv["bytes"] < 50e9:
    raise RuntimeError(f"unexpected conversion result: {conv}")
print(f"teacher: {TEACHER_DIR}, {conv['bytes'] / 2**30:.1f} GiB, {conv['layers']} layers, vocab {conv['vocab']}, "
      f"tokenizer: {conv['tokenizer']}")
if DELETE_GGUF_AFTER:
    os.remove(GGUF_PATH)
print(f"{free_gb():.1f} GB free on /content")
'''

CELL5 = r'''#@title 5. Corpus (streamed, on Drive) and training (checkpoints on Drive, auto-resume)
HF_TOKEN = ""  #@param {type:"string"}
CORPUS_TOTAL = 200000  #@param {type:"integer"}
CORPUS_DIR = ""  #@param {type:"string"}
STEPS = 12000  #@param {type:"integer"}
SEQ_LEN = 2048  #@param {type:"integer"}
BATCH = 2  #@param {type:"integer"}
GRAD_ACCUM = 8  #@param {type:"integer"}
LR = 2e-4  #@param {type:"number"}
WARMUP = 300  #@param {type:"integer"}
UNROLL = 4  #@param {type:"integer"}    # unrolled self-rollout steps of the `mtp` drafter, 1..8
DRAFTER = "mtp"  #@param {type:"string"}    # mtp (single-layer nextn head, --unroll) | df3 (block schedule)
DF3_BLOCK = 16  #@param {type:"integer"}    # draft block of the `df3` schedule = the 17-row verify batch of the engine
DF3_LAYERS = 2  #@param {type:"integer"}    # decoder layers of the target DFlash head; the engine loader knows one
KL_CHUNK = 256  #@param {type:"integer"}
ONPOLICY_FRAC = 0.25  #@param {type:"number"}
SAVE_EVERY = 250  #@param {type:"integer"}
KEEP_LAST = 2  #@param {type:"integer"}
TIME_LIMIT_MIN = 0  #@param {type:"number"}
RESUME_FROM = "auto"  #@param ["auto", "none"]

token = HF_TOKEN
if not token:
    try:
        from google.colab import userdata
        token = userdata.get("HF_TOKEN") or ""
    except Exception:   # no secret set or not on Colab
        token = ""
env = {"HF_TOKEN": token} if token else {}
if not token:
    print("WARNING: no HF_TOKEN: the gated sources (bigcode/starcoderdata, uonlp/CulturaX) will be skipped")
CORPUS = CORPUS_DIR or f"{DRIVE_OUT}/corpus"
RUN_DIR = f"{DRIVE_OUT}/run"

# --- corpus: resumable (manifest.json); finished sources are not streamed again
man_p = os.path.join(CORPUS, "manifest.json")
have = 0
if os.path.exists(man_p):
    with open(man_p) as f:
        have = json.load(f).get("total", 0)
if have >= 0.9 * CORPUS_TOTAL and any(n.startswith("shard-") for n in os.listdir(CORPUS)):
    print(f"corpus: {have} samples already in {CORPUS}, skipping")
else:
    run([PY, f"{TOOLS}/prepare_corpus.py", "--out", CORPUS, "--total", str(CORPUS_TOTAL), "--on-error", "skip",
         "--min-fraction", "0.5"], "prepare_corpus", env=dict(env, HF_DATASETS_DISABLE_PROGRESS_BARS="1"))
with open(man_p) as f:
    man = json.load(f)
print(f"corpus: {man['total']} samples; per source: {man['counts']}")
for name, err in man.get("errors", {}).items():
    if not man["done"].get(name):
        print(f"  missing {name}: {err}")

# --- training. BF16 autocast + TF32 matmuls (Tensor Cores), fused AdamW, gradient checkpointing per unroll step.
# train_state.pt and the newest KEEP_LAST blk.64 checkpoints are written to Drive every SAVE_EVERY steps; running this cell
# again (also after a disconnect, once cells 1-4 have run) continues from the last saved step.
train_cmd = [PY, f"{TOOLS}/train_bonsai_mtp.py", "--teacher", TEACHER_DIR, "--corpus", CORPUS, "--out", RUN_DIR,
             "--steps", str(STEPS), "--seq-len", str(SEQ_LEN), "--batch", str(BATCH), "--grad-accum", str(GRAD_ACCUM),
             "--lr", str(LR), "--warmup", str(WARMUP), "--unroll", str(UNROLL), "--kl-chunk", str(KL_CHUNK),
             "--drafter", str(DRAFTER), "--df3-block", str(DF3_BLOCK), "--df3-layers", str(DF3_LAYERS),
             "--onpolicy-frac", str(ONPOLICY_FRAC), "--save-every", str(SAVE_EVERY), "--keep-last", str(KEEP_LAST),
             "--time-limit-min", str(TIME_LIMIT_MIN), "--resume-from", RESUME_FROM, "--attn-impl", ATTN_IMPL,
             "--device", "cuda"] + INIT_ARGS
run(train_cmd, "train", env=dict(env, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TOKENIZERS_PARALLELISM="false"))
with open(os.path.join(RUN_DIR, "train_log.jsonl")) as f:
    last = json.loads(f.readlines()[-1])
print(f"last logged step {last['step']}: top-1 agreement per unroll step {[round(x, 3) for x in last['top1_agree']]}")
'''

CELL6 = r'''#@title 6. Export the final blk.64 (safetensors + GGUF) to Drive and print the commands for your PC
EXPORT_QUANTS = "q4_k_m,q8_0"  #@param {type:"string"}

import hashlib

FINAL = f"{DRIVE_OUT}/final"
latest = subprocess.run([PY, f"{TOOLS}/train_bonsai_mtp.py", "--latest", RUN_DIR], capture_output=True, text=True)
if latest.returncode != 0:
    raise RuntimeError(f"no checkpoint in {RUN_DIR}: run cell 5 first.\n{latest.stderr}")
ckpt = latest.stdout.strip()
print(f"checkpoint: {ckpt}")
dst = os.path.join(FINAL, "mtp_blk64.safetensors")
shutil.copy(ckpt, dst + ".tmp")
os.replace(dst + ".tmp", dst)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


lines = [f"source checkpoint: {os.path.basename(ckpt)}", f"mtp_blk64.safetensors  {os.path.getsize(dst)} B  sha256 {sha256(dst)}"]
for qt in [q.strip() for q in EXPORT_QUANTS.split(",") if q.strip()]:
    out = os.path.join(FINAL, f"mtp_blk64_{qt}.gguf")
    run([PY, f"{TOOLS}/inject_mtp_gguf.py", "--from-checkpoint", dst, "--quant", qt, "--mtp-only-out", out],
        f"export_{qt}", env={"CUDA_VISIBLE_DEVICES": ""})
    lines.append(f"mtp_blk64_{qt}.gguf  {os.path.getsize(out)} B ({os.path.getsize(out) / 2**20:.1f} MiB)  sha256 {sha256(out)}")
with open(os.path.join(FINAL, "MANIFEST.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))

qt0 = [q.strip() for q in EXPORT_QUANTS.split(",") if q.strip()][0]
print(f"""
Files on Drive: {FINAL}
On your PC (repo checkout, no GPU needed for this step; numpy is enough):

  python3 qwen38-engine/tools/bonsai_mtp_train/inject_mtp_gguf.py \\
      --base ~/models/27b-bonsai/Ternary-Bonsai-2-27B-{QUANT}.gguf \\
      --from-gguf mtp_blk64_{qt0}.gguf \\
      --out ~/models/27b-bonsai/Ternary-Bonsai-2-27B-{QUANT}-mtp.gguf

Then run the engine with drafting (the MTP block is placed on the GPU when it fits):

  taskset -c 0-3 ./build/qwen38-engine -m ~/models/27b-bonsai/Ternary-Bonsai-2-27B-{QUANT}-mtp.gguf \\
      --gpu-layers auto --ctx 4096 --mtp-adaptive 0 --mtp-steps 2 -f bench/prompt512.txt -n 256

Compare decode_tok_s and draft_n_accepted / draft_n with --mtp-steps 0 (the output tokens must be identical).
""")
'''

TROUBLE = """## Troubleshooting

- **A cell failed:** the error shows the exit code and the last 40 lines; the full log is in `DRIVE_OUT/logs/<step>.log`.
- **CUDA out of memory in cell 5:** lower `KL_CHUNK` (256 → 128), then `SEQ_LEN` (2048 → 1536) or `BATCH`. The resumed run keeps
  its step count.
- **Corpus has fewer samples than requested:** a gated source was skipped. Set `HF_TOKEN` (account must have accepted the dataset
  terms), delete `DRIVE_OUT/corpus/manifest.json` entries you want to redo (or the folder), and run cell 5 again.
- **Session limit:** set `TIME_LIMIT_MIN` a bit below the remaining session time: training saves and stops cleanly, the next session
  continues from there.
- **Start over:** delete `DRIVE_OUT/run` (keeps the corpus and the warm-start file), or run with `RESUME_FROM = none`.
"""


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {"cellView": "form"}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def main():
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "machine_shape": "hm", "name": "train_bonsai_mtp_colab.ipynb"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": [md(INTRO), code(CELL1), code(CELL2), code(CELL3), code(CELL4), code(CELL5), code(CELL6), md(TROUBLE)],
    }
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"cell-{i}"
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
