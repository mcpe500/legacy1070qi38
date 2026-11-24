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
COLAB_URL_T4 = ("https://colab.research.google.com/github/mcpe500/legacy1070qi38/blob/main/"
                "qwen38-engine/tools/bonsai_mtp_train/train_bonsai_t4_student_colab.ipynb")

INTRO = f"""# Ternary Bonsai 2 27B: train the MTP draft head (blk.64) on Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_URL})

This notebook trains a multi-token-prediction head (`blk.64`, one decoder layer + `nextn` projection, 424.7M parameters) for
**Ternary Bonsai 2 27B** by live online distillation, so that `qwen38-engine` can draft with it. The output is a small GGUF
with only the `blk.64` tensors (Q4_K_M about 273 MiB, Q8_0 about 450 MiB), which you use at home as a `--mtp-gguf` sidecar
or inject into your Bonsai GGUF (cell 6 prints both).

**Runtime:** *Runtime → Change runtime type →* a GPU with at least 80 GB (RTX PRO 6000 96 GB). The teacher alone is 50.1 GiB in BF16.

**Speed, measured on a real RTX PRO 6000 run (steps 0-40, warm start from `qwen_blk64`):**

- **Install `causal-conv1d` (cell 2, on by default).** Without it the teacher conv falls back to the PyTorch reference
  and one step takes **71-76 s**; the first checkpoint (`SAVE_EVERY = 250`) would arrive only after 5+ hours. With the
  fused kernels the same step is several times faster. The training log prints which kernels are active
  (`accelerator: causal-conv1d: ACTIVE / FALLBACK`), and `--save-every-min 20` saves every 20 minutes of wall clock
  regardless of the step count, so a slow run never loses more than 20 minutes.
- **`ONPOLICY_FRAC = 0.0` by default.** The on-policy path runs a teacher `generate()` per batch, which is the second
  big cost. 0.0 gives roughly 3-5x the throughput of the old default (0.25); raise it to 0.1-0.25 later for a polish
  run once `causal-conv1d` is active.
- **Warm start converges fast:** with `qwen_blk64` the step-1 top-1 agreement went from 75.1 % to 83.5 % and all four
  unroll steps improved, within 30-40 steps (about one hour). `STEPS = 2000` is a solid first target; the plateau of
  `top1_agree` in `train_log.jsonl` tells you when to stop, not the step count.
- **Warm-start options (cell 3):** `qwen_blk64` (Qwen3.8-27B's own head, needs the 16.6 GB GGUF once) or
  `procreations_r3` (only an 850 MB `model_mtp.safetensors` download: the head ProCreations trained for Bonsai, the
  same one the engine uses as the `r3_mtp_blk64_q4_k_m.gguf` sidecar — the closest starting point for a Bonsai MTP).

| Cell | What it does | Where data goes |
|---|---|---|
| 1 | preflight (GPU, RAM, free disk on `/content` ≥ 70 GB), mount Google Drive, helpers | `DRIVE_OUT/logs/` |
| 2 | clone / pull `mcpe500/legacy1070qi38`, install Python packages incl. `causal-conv1d`, script self-tests | `/content/legacy1070qi38` |
| 3 | download the Bonsai GGUF (PTQ1_0 or Q2_0), warm start from `qwen_blk64` / `procreations_r3` / Drive / random | `/content/models/` (local SSD), init on Drive |
| 4 | GGUF → Hugging Face BF16 teacher (`convert_gguf_to_hf_bf16.py`, about 35 min) | `/content/models/bonsai-hf-bf16` (local SSD, not Drive) |
| 5 | streamed multilingual corpus (`prepare_corpus.py`), then training (`train_bonsai_mtp.py`) | corpus and checkpoints on Drive |
| 6 | export the final `blk.64` (safetensors + GGUF Q4_K_M / Q8_0) and print the commands for your PC | `DRIVE_OUT/final/` |

**After a disconnect:** run cells 1 to 5 again. The Bonsai GGUF is downloaded again and the teacher converted again (local disk
is wiped). The corpus, the warm-start file and the training state are on Drive, so training resumes from the last saved step
(`--resume-from auto`; the corpus is not re-streamed, and gated sources that failed without a token are not retried without one).

**Drive space:** about 5 GB for the training state (`train_state.pt` 1.7 GB + the newest `--keep-last` checkpoints of 1.7 GB each),
1.7 GB for the warm-start file (or 850 MB for `procreations_r3`), 1 to 3 GB for the corpus, under 1 GB for the final export.

Gated datasets (`bigcode/starcoderdata`, `uonlp/CulturaX`) need a Hugging Face token whose account accepted their terms. Put it in
Colab *Secrets* as `HF_TOKEN` (or in the field of cell 5). Without it those sources are skipped once and then not retried on
reruns; the `aya` multilingual source is open (CohereLabs/aya_collection) and needs no token.
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
INSTALL_CAUSAL_CONV1D = True  #@param {type:"boolean"}
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
# than the PyTorch fallback). causal-conv1d: the fused conv1d of the same layers; without it the step time is 71-76 s
# instead of seconds (the conv falls back to the PyTorch reference). flash-attn: only for the 16 full-attention layers;
# SDPA is used when it is missing.
if INSTALL_FLA:
    try:
        run([PY, "-m", "pip", "install", "-q", "flash-linear-attention"], "pip_fla")
        run([PY, "-c", "import fla; print('flash-linear-attention', getattr(fla, '__version__', '?'))"], "check_fla")
    except RuntimeError as e:
        print(f"WARNING: flash-linear-attention not available, the teacher uses the slower PyTorch path.\n{e}")
if INSTALL_CAUSAL_CONV1D:
    # No torch-matched wheel on PyPI (sdist only): build with --no-build-isolation against the installed torch.
    # When no nvcc / the sm_120 build fails, this prints one warning and training continues on the PyTorch fallback.
    try:
        run([PY, "-m", "pip", "install", "-q", "--no-build-isolation", "causal-conv1d>=1.5"], "pip_causal_conv1d")
        run([PY, "-c", "import causal_conv1d; assert hasattr(causal_conv1d, 'causal_conv1d_fn'); "
                       "print('causal-conv1d', getattr(causal_conv1d, '__version__', '?'), 'causal_conv1d_fn ok')"],
            "check_causal_conv1d")
    except RuntimeError as e:
        print(f"WARNING: causal-conv1d could not be built for this runtime (sm_120 needs nvcc); the teacher convs "
              f"use the PyTorch reference and one step takes 71-76 s instead of seconds. Training still works and "
              f"--save-every-min protects the progress.\n{e}")
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
INIT_MODE = "qwen_blk64"  #@param ["qwen_blk64", "procreations_r3", "random", "drive_safetensors"]
INIT_SAFETENSORS = ""  #@param {type:"string"}

BONSAI_REPO = "prism-ml/Ternary-Bonsai-2-27B-gguf"
GGUF_NAME = f"Ternary-Bonsai-2-27B-{QUANT}.gguf"
GGUF_PATH = f"{MODELS}/{GGUF_NAME}"
QWEN_REPO, QWEN_NAME = "unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-Q4_K_M.gguf"
# ProCreations/Ternary-Bonsai-2-27B-MTP: the single-layer head already trained for Bonsai (draft_tokens 2), the same
# weights the engine uses as the r3_mtp_blk64_q4_k_m.gguf sidecar. Only the 850 MB safetensors is downloaded.
PROC_REPO, PROC_NAME = "ProCreations/Ternary-Bonsai-2-27B-MTP", "model_mtp.safetensors"
INIT_DRIVE = f"{DRIVE_OUT}/init/mtp_init_qwen_blk64.safetensors"
INIT_DRIVE_PROC = f"{DRIVE_OUT}/init/model_mtp.safetensors"

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
elif INIT_MODE == "procreations_r3":
    # The r3 head as trained by ProCreations (mtp.* HF nextn naming); train_bonsai_mtp.py auto-maps the 15 keys to the
    # blk.64 names and rejects any shape mismatch. 850 MB instead of the 16.6 GB Qwen GGUF, and the closest start for Bonsai.
    if not os.path.exists(INIT_DRIVE_PROC):
        run([PY, "-c", DOWNLOAD, PROC_REPO, PROC_NAME, str(os.path.dirname(INIT_DRIVE_PROC))], "download_procreations")
    else:
        print(f"warm start: {INIT_DRIVE_PROC} already on Drive")
    INIT_ARGS = ["--init-mtp-safetensors", INIT_DRIVE_PROC]
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
SKIP_CORPUS_IF_READY = True  #@param {type:"boolean"}
STEPS = 2000  #@param {type:"integer"}
SEQ_LEN = 2048  #@param {type:"integer"}
BATCH = 2  #@param {type:"integer"}
GRAD_ACCUM = 8  #@param {type:"integer"}
LR = 2e-4  #@param {type:"number"}
WARMUP = 100  #@param {type:"integer"}
UNROLL = 4  #@param {type:"integer"}    # unrolled self-rollout steps of the `mtp` drafter, 1..8
DRAFTER = "mtp"  #@param {type:"string"}    # mtp (single-layer nextn head, --unroll) | df3 (block schedule)
DF3_BLOCK = 16  #@param {type:"integer"}    # draft block of the `df3` schedule = the 17-row verify batch of the engine
DF3_LAYERS = 2  #@param {type:"integer"}    # decoder layers of the target DFlash head; the engine loader knows one
KL_CHUNK = 256  #@param {type:"integer"}
ONPOLICY_FRAC = 0.0  #@param {type:"number"}
SAVE_EVERY = 25  #@param {type:"integer"}
SAVE_EVERY_MIN = 20  #@param {type:"number"}
KEEP_LAST = 2  #@param {type:"integer"}
TIME_LIMIT_MIN = 0  #@param {type:"number"}
RESUME_FROM = "auto"  #@param ["auto", "none"]

# ONPOLICY_FRAC = 0.0: the teacher generate() per batch is the second big cost after a missing causal-conv1d; 0.0 gives
# about 3-5x the throughput of 0.25. Raise it to 0.1-0.25 for a short polish run once cell 2 reported causal-conv1d ACTIVE.
# TIME_LIMIT_MIN = 0 keeps the cell running until Colab itself stops it; SAVE_EVERY_MIN = 20 writes an atomic checkpoint
# every 20 wall-clock minutes, and SIGINT/SIGTERM (cell interrupt) also save before stopping, so nothing is lost.
# SAVE_EVERY = 25 keeps the step-granular checkpoints cheap; KEEP_LAST = 2 prunes the old ones on Drive.

token = HF_TOKEN
if not token:
    try:
        from google.colab import userdata
        token = userdata.get("HF_TOKEN") or ""
    except Exception:   # no secret set or not on Colab
        token = ""
env = {"HF_TOKEN": token} if token else {}
if not token:
    print("WARNING: no HF_TOKEN: the gated sources (bigcode/starcoderdata, uonlp/CulturaX) are skipped once and not "
          "retried on reruns; the aya multilingual source is open and needs no token")
CORPUS = CORPUS_DIR or f"{DRIVE_OUT}/corpus"
RUN_DIR = f"{DRIVE_OUT}/run"

# --- corpus: resumable (manifest.json); finished sources are not streamed again, and gated sources that failed
# without a token are not re-attempted without one (SKIP_CORPUS_IF_READY short-circuits the whole step)
man_p = os.path.join(CORPUS, "manifest.json")
have, corpus_ok, has_shards = 0, False, False
if os.path.exists(man_p):
    with open(man_p) as f:
        man_old = json.load(f)
    have = man_old.get("total", 0)
    has_shards = os.path.isdir(CORPUS) and any(n.startswith("shard-") for n in os.listdir(CORPUS))
    corpus_ok = has_shards and have >= 0.9 * CORPUS_TOTAL
if SKIP_CORPUS_IF_READY and has_shards and not env:
    # no token now: rerunning would re-skip the gated sources anyway; train on what Drive already holds
    print(f"SKIP_CORPUS_IF_READY: training on the {have} samples already in {CORPUS} (no HF_TOKEN: nothing new to stream)")
elif corpus_ok:
    print(f"corpus: {have} samples already in {CORPUS}, skipping")
else:
    cmd = [PY, f"{TOOLS}/prepare_corpus.py", "--out", CORPUS, "--total", str(CORPUS_TOTAL), "--on-error", "skip",
           "--min-fraction", "0.5"] + (["--retry-errors"] if env else [])
    run(cmd, "prepare_corpus", env=dict(env, HF_DATASETS_DISABLE_PROGRESS_BARS="1"))
with open(man_p) as f:
    man = json.load(f)
print(f"corpus: {man['total']} samples; per source: {man['counts']}")
for name, err in man.get("errors", {}).items():
    if not man["done"].get(name):
        print(f"  missing {name}: {err}")

# --- training. BF16 autocast + TF32 matmuls (Tensor Cores), fused AdamW, gradient checkpointing per unroll step.
# train_state.pt and the newest KEEP_LAST blk.64 checkpoints are written to Drive every SAVE_EVERY steps or every
# SAVE_EVERY_MIN minutes (whichever comes first); running this cell again (also after a disconnect, once cells 1-4
# have run) continues from the last saved step.
train_cmd = [PY, f"{TOOLS}/train_bonsai_mtp.py", "--teacher", TEACHER_DIR, "--corpus", CORPUS, "--out", RUN_DIR,
             "--steps", str(STEPS), "--seq-len", str(SEQ_LEN), "--batch", str(BATCH), "--grad-accum", str(GRAD_ACCUM),
             "--lr", str(LR), "--warmup", str(WARMUP), "--unroll", str(UNROLL), "--kl-chunk", str(KL_CHUNK),
             "--drafter", str(DRAFTER), "--df3-block", str(DF3_BLOCK), "--df3-layers", str(DF3_LAYERS),
             "--onpolicy-frac", str(ONPOLICY_FRAC), "--save-every", str(SAVE_EVERY),
             "--save-every-min", str(SAVE_EVERY_MIN), "--keep-last", str(KEEP_LAST),
             "--time-limit-min", str(TIME_LIMIT_MIN), "--resume-from", RESUME_FROM, "--attn-impl", ATTN_IMPL,
             "--device", "cuda"] + INIT_ARGS
run(train_cmd, "train", env=dict(env, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TOKENIZERS_PARALLELISM="false"))
with open(os.path.join(RUN_DIR, "train_log.jsonl")) as f:
    last = json.loads(f.readlines()[-1])
print(f"last logged step {last['step']}: top-1 agreement per unroll step {[round(x, 3) for x in last['top1_agree']]} "
      f"({last.get('sec_per_step', '?')} s/step, eta {last.get('eta_h', '?')} h)")
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
(download the sidecar GGUF(s) from there to your PC first, e.g. into ~/models/27b-bonsai/)
On your PC (repo checkout of this branch, engine built with ./scripts/setup.sh):

OPTION 1 (recommended, nothing duplicated): run the sidecar directly with --mtp-gguf. The 5.4 GB base GGUF stays untouched:

  taskset -c 0-3 ./build/qwen38-engine -m ~/models/27b-bonsai/Ternary-Bonsai-2-27B-{QUANT}.gguf \\
      --mtp-gguf ~/models/27b-bonsai/mtp_blk64_{qt0}.gguf \\
      --gpu-layers auto --ctx 4096 --mtp-adaptive 1 --gen-mtp-steps 3 -f bench/prompt512.txt -n 256

Race it against the r3 head (the ProCreations sidecar your warm start began from), same flags with a different sidecar:

  taskset -c 0-3 ./build/qwen38-engine -m ~/models/27b-bonsai/Ternary-Bonsai-2-27B-{QUANT}.gguf \\
      --mtp-gguf ~/models/27b-bonsai/r3_mtp_blk64_{qt0}.gguf \\
      --gpu-layers auto --ctx 4096 --mtp-adaptive 1 --gen-mtp-steps 3 -f bench/prompt512.txt -n 256

Compare decode_tok_s and draft_n_accepted / draft_n between the two (greedy output tokens must be identical to
--gen-mtp-steps 0; the drafts only change the speed).

OPTION 2 (optional, permanent): copy the head into a duplicated GGUF:

  python3 qwen38-engine/tools/bonsai_mtp_train/inject_mtp_gguf.py \\
      --base ~/models/27b-bonsai/Ternary-Bonsai-2-27B-{QUANT}.gguf \\
      --from-gguf mtp_blk64_{qt0}.gguf \\
      --out ~/models/27b-bonsai/Ternary-Bonsai-2-27B-{QUANT}-mtp.gguf

  taskset -c 0-3 ./build/qwen38-engine -m ~/models/27b-bonsai/Ternary-Bonsai-2-27B-{QUANT}-mtp.gguf \\
      --gpu-layers auto --ctx 4096 --mtp-adaptive 0 --mtp-steps 2 -f bench/prompt512.txt -n 256
""")
'''

TROUBLE = """## Troubleshooting

- **A cell failed:** the error shows the exit code and the last 40 lines; the full log is in `DRIVE_OUT/logs/<step>.log`.
- **Steps take 71-76 s:** `causal-conv1d` did not install (see the WARNING at the end of cell 2). The training log says
  `accelerator: causal-conv1d: FALLBACK`. Retrying the pip line sometimes helps after a Colab restart; otherwise the run
  is just slower, `--save-every-min 20` protects every 20 minutes, and `--steps 2000` is sized so the warm start still
  finishes in one session.
- **CUDA out of memory in cell 5:** lower `KL_CHUNK` (256 → 128), then `SEQ_LEN` (2048 → 1536) or `BATCH`. The resumed run keeps
  its step count.
- **Corpus has fewer samples than requested:** a gated source was skipped (cell 5 prints `missing <source>: ...`).
  Set `HF_TOKEN` (account must have accepted the dataset terms) and run cell 5 again: with a token the gated sources
  are retried. Without a token they are skipped once and not retried again (`--retry-errors` would force it).
  `SKIP_CORPUS_IF_READY` trains directly on what Drive already holds when no token is set.
- **Warm start failed with `no tensor for blk....` or a shape message:** the init file is not a single-layer blk.64 head
  (the loader auto-maps the ProCreations `mtp.*` names and rejects shape mismatches). Pick `INIT_MODE = qwen_blk64` or
  `procreations_r3` instead of a hand-made safetensors.
- **Session limit:** set `TIME_LIMIT_MIN` a bit below the remaining session time: training saves and stops cleanly, the next session
  continues from there. An interrupted cell (stop button) also saves once before exiting.
- **Start over:** delete `DRIVE_OUT/run` (keeps the corpus and the warm-start file), or run with `RESUME_FROM = none`.
"""


INTRO_T4 = f"""# Ternary Bonsai 2 27B: native PTQ1_0 target precompute + teacher-free head training (Tesla T4)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_URL_T4})

Companion to the live-training notebook — but this one is **T4-only**: there is no BF16 conversion (no
50.1 GiB teacher folder), no A100 session, and no layer-by-layer CPU<->GPU streaming. The engine itself
(`qwen38-engine`, built here for sm_75) loads the whole `Ternary-Bonsai-2-27B-PTQ1_0.gguf` (~5.54 GiB,
all 64 layers + the LM head resident, ~5.7 GiB VRAM) and runs Phase A directly:

| Phase | What | Cost |
|---|---|---|
| A (once) | `./build/qwen38-engine --dump-targets`: per sample a clean DeltaNet/KV state, one full prefill, the output-normed hidden of every position captured, top-64 logits + FP16 logsumexp computed chunked on the GPU, hidden stored INT4 group-64 — the exact math the home engine runs, so parity is by construction. Plus `export_frozen_head.py`: only 3 tensors (embed, output_norm, lm_head — unrotated) -> `lm_head_embed_frozen.safetensors` in ~2-4 min, never touching the other ~850 tensors | ~5.8 GB Drive per 2000 samples (4000 samples ~11.6 GB); speed recorded by the first run |
| B (any number of times) | `train_bonsai_mtp.py --from-targets`: the same blk.64 student (mtp unroll-4 or df3 block-16), **renormalized top-K KL (mathematically >= 0)** + cosine loss, FP16 + GradScaler on pre-Ampere cards, 8-bit AdamW, warm start from your r3 sidecar | T4; ~2-6 s/step at 16K tokens/step is the estimate until the first run measures it |

Notes:
- Phase A reads `tokens.bin` + `plan.json` written by `tokenize_for_engine.py`: the corpus is tokenized with the
  engine's own BPE (the exact tokenizer the dump uses), balanced over the sources of the `prepare_corpus.py`
  shards (OpenCodeInstruct / OpenThoughts / UltraChat / Wikipedia-ID / Aya).
- On sm_75+ the dump dequantizes one weight matrix at a time into a small FP16 VRAM scratch (~170-335 MiB) and
  multiplies through `cublasGemmEx` (FP16 in, FP32 accumulate); sm_61 (the local GTX 1070 Ti) keeps the
  FP32/dp4a path untouched (SASS gate: zero half arithmetic).
- Shards are plain `.npy` directories (the Python tool writes `.npz`) — `--from-targets` reads both. Every shard
  is written atomically and auto-resumed; `--max-gb` stops with a consistent manifest.
- Phase B warms from `procreations_r3` by default (the `INIT_MODE` dropdown in cell 4).
"""

T4CELL1 = r'''#@title 1. Preflight (Tesla T4 15 GiB) and helpers
DRIVE_OUT = "/content/drive/MyDrive/bonsai_mtp"  #@param {type:"string"}
MOUNT_DRIVE = True  #@param {type:"boolean"}
MIN_VRAM_GB = 14  #@param {type:"number"}
MIN_RAM_GB = 12  #@param {type:"number"}

import collections
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

PY = sys.executable
REPO_DIR = "/content/legacy1070qi38"
TOOLS = f"{REPO_DIR}/qwen38-engine/tools/bonsai_mtp_train"
ENGINE = f"{REPO_DIR}/qwen38-engine/build/qwen38-engine"
TARGETS_DIR = f"{DRIVE_OUT}/targets"
CORPUS_DIR = f"{DRIVE_OUT}/corpus"
LOG_DIR = f"{DRIVE_OUT}/logs"


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


try:
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True).stdout.strip().splitlines()
except (OSError, subprocess.CalledProcessError) as e:
    raise RuntimeError("No NVIDIA GPU visible. Runtime -> Change runtime type -> select a Tesla T4 (or better).") from e
gpu_name, gpu_mib, driver = [x.strip() for x in q[0].split(",")]
gpu_gb = float(gpu_mib) / 1024
print(f"GPU: {gpu_name}, {gpu_gb:.1f} GiB, driver {driver}")
if gpu_gb < MIN_VRAM_GB:
    raise RuntimeError(f"{gpu_name} has {gpu_gb:.1f} GiB; this notebook wants >= {MIN_VRAM_GB} GiB (Tesla T4 = 15 GiB).")
with open("/proc/meminfo") as f:
    mem = {l.split(":")[0]: int(l.split()[1]) for l in f}
ram_gb = mem["MemTotal"] / 1e6
print(f"RAM: {ram_gb:.1f} GB total; CPU threads {os.cpu_count()}")
if ram_gb < MIN_RAM_GB:
    raise RuntimeError(f"{ram_gb:.1f} GB RAM; the engine build + shard staging want >= {MIN_RAM_GB} GB.")
if MOUNT_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    if not os.path.isdir("/content/drive/MyDrive"):
        raise RuntimeError("Google Drive did not mount at /content/drive/MyDrive.")
os.makedirs(LOG_DIR, exist_ok=True)
print(f"DRIVE_OUT = {DRIVE_OUT} ({shutil.disk_usage(DRIVE_OUT).free / 1e9:.1f} GB free there)")
print("preflight OK")
'''

T4CELL2 = r'''#@title 2. Clone, build qwen38-engine for sm_75, light pip packages, CPU self-tests
REPO_URL = "https://github.com/mcpe500/legacy1070qi38.git"  #@param {type:"string"}
BRANCH = "main"  #@param {type:"string"}
LLAMA_REPO = "https://github.com/mcpe500/llama.cpp.git"  #@param {type:"string"}
LLAMA_BRANCH = "feat/iso3-kv"  #@param {type:"string"}
INSTALL_BITSANDBYTES = True  #@param {type:"boolean"}
RUN_SELFTESTS = True  #@param {type:"boolean"}

# The engine links the ggml CPU libraries of the mcpe500 fork (iso3/PTQ1_0 row sizes); no CUDA ggml needed —
# the engine carries its own CUDA kernels (this build targets sm_75 for the T4; sm_61 stays untouched).
LLAMA_DIR = "/content/llama.cpp"
LLAMA_BUILD = f"{LLAMA_DIR}/build-iso3"

if not os.path.isdir(REPO_DIR):
    run(["git", "clone", "--filter=blob:none", REPO_URL, REPO_DIR], "clone_repo")
run(["git", "-C", REPO_DIR, "fetch", "origin", BRANCH], "fetch_repo")
run(["git", "-C", REPO_DIR, "checkout", BRANCH], "checkout_repo")
run(["git", "-C", REPO_DIR, "pull", "--ff-only"], "pull_repo")
if not os.path.isdir(LLAMA_DIR):
    run(["git", "clone", "--filter=blob:none", "-b", LLAMA_BRANCH, LLAMA_REPO, LLAMA_DIR], "clone_llama")
if not os.path.exists(f"{LLAMA_BUILD}/bin/libggml-cpu.so"):
    run(["cmake", "-S", LLAMA_DIR, "-B", LLAMA_BUILD, "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=OFF",
         "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_SERVER=OFF"], "cmake_llama")
    run(["cmake", "--build", LLAMA_BUILD, "--target", "ggml", "-j", str(os.cpu_count() or 2)], "build_ggml")
run(["cmake", "-S", f"{REPO_DIR}/qwen38-engine", "-B", f"{REPO_DIR}/qwen38-engine/build",
     "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=75"], "cmake_engine",
    env={"LLAMA_SRC": LLAMA_DIR, "LLAMA_BUILD": LLAMA_BUILD})
run(["ninja", "-C", f"{REPO_DIR}/qwen38-engine/build", "-j", str(os.cpu_count() or 2), "qwen38-engine"], "build_engine",
    env={"LLAMA_SRC": LLAMA_DIR, "LLAMA_BUILD": LLAMA_BUILD})
run([ENGINE], "engine_link_check", ok_codes=(2,))   # prints its usage line (exit 2 = expected): proves the binary links and runs

if INSTALL_BITSANDBYTES:
    try:
        run([PY, "-m", "pip", "install", "-q", "bitsandbytes"], "pip_bitsandbytes")
        run([PY, "-c", "import bitsandbytes; print('bitsandbytes', bitsandbytes.__version__)"], "check_bnb")
    except RuntimeError as e:
        print(f"WARNING: bitsandbytes unavailable, --optim adamw8bit falls back to Adafactor.\n{e}")
if RUN_SELFTESTS:
    for s in ("precompute_teacher_targets", "train_bonsai_mtp", "prepare_corpus", "convert_gguf_to_hf_bf16",
              "inject_mtp_gguf", "export_frozen_head", "tokenize_for_engine"):
        run([PY, f"{TOOLS}/{s}.py", "--selftest"], f"selftest_{s}", env={"CUDA_VISIBLE_DEVICES": ""})
print("cell 2 OK (engine built for sm_75)")
'''

T4CELL3 = r'''#@title 3. Phase A (once): native PTQ1_0 dump on the T4 -> target package on Drive
RUN_PHASE_A = True  #@param {type:"boolean"}
HF_REPO = "mcpe500/bonsai-ptq10"  #@param {type:"string"}
GGUF_NAME = "Ternary-Bonsai-2-27B-PTQ1_0.gguf"  #@param {type:"string"}
SAMPLES = 4000  #@param {type:"integer"}
SEQ_LEN = 1024  #@param {type:"integer"}
TOP_K = 64  #@param {type:"integer"}
HIDDEN_DTYPE = "int4g64"  #@param ["int4g64", "int8"]
SHARD_SAMPLES = 128  #@param {type:"integer"}
MAX_GB = 40  #@param {type:"number"}
VRAM_BUDGET_MIB = 14000  #@param {type:"integer"}
DUMP_F32_HIDDEN = False  #@param {type:"boolean"}

# All-resident native dump: ~5.7 GiB VRAM for the quantized 27B + a small FP16 dequant scratch, NO BF16 copy,
# NO layer streaming. Auto-resume per shard; re-running with the same TARGETS_DIR continues where it stopped.
import json as _json

def manifest_done():
    p = os.path.join(TARGETS_DIR, "manifest.json")
    if not os.path.exists(p):
        return False
    try:
        return _json.load(open(p)).get("total_samples", 0) >= SAMPLES
    except (ValueError, OSError):
        return False

if not RUN_PHASE_A:
    print(f"RUN_PHASE_A = False: skipped (frozen-head file still exported below if missing)")
else:
    if manifest_done():
        print(f"manifest at {TARGETS_DIR} already covers {SAMPLES} samples - Phase A skipped")
    else:
        if not os.path.isdir(CORPUS_DIR) or not os.listdir(CORPUS_DIR):
            raise RuntimeError(f"no corpus at {CORPUS_DIR}: run the main notebook's corpus step once "
                               f"(prepare_corpus.py) or point CORPUS_DIR at your prepared corpus")
        gguf = f"/content/models/{GGUF_NAME}"
        os.makedirs("/content/models", exist_ok=True)
        if not os.path.exists(gguf):
            run([PY, "-m", "huggingface_hub.commands.huggingface_cli", "download", HF_REPO, GGUF_NAME,
                 "--local-dir", "/content/models"], "hf_download_gguf")
        if not os.path.getsize(gguf) == 5946648928:
            print(f"WARNING: {gguf} is {os.path.getsize(gguf)} bytes, expected 5946648928")
        run([PY, f"{TOOLS}/tokenize_for_engine.py", "--corpus", CORPUS_DIR, "--gguf", gguf, "--engine", ENGINE,
             "--out", "/content/tok", "--samples", str(SAMPLES), "--seq-len", str(SEQ_LEN)], "tokenize")
        cmd = [ENGINE, "-m", gguf, "--dump-targets", TARGETS_DIR, "--tokens-bin", "/content/tok/tokens.bin",
               "--dump-samples", str(SAMPLES), "--dump-seq-len", str(SEQ_LEN), "--dump-top-k", str(TOP_K),
               "--dump-hidden-dtype", HIDDEN_DTYPE, "--shard-samples", str(SHARD_SAMPLES),
               "--max-gb", str(MAX_GB), "--vram-budget-mib", str(VRAM_BUDGET_MIB)]
        if DUMP_F32_HIDDEN:
            cmd.append("--dump-f32-hidden")
        run(cmd, "dump_targets")
frozen = os.path.join(TARGETS_DIR, "lm_head_embed_frozen.safetensors")
if not os.path.exists(frozen):
    run([PY, f"{TOOLS}/export_frozen_head.py", "--gguf", f"/content/models/{GGUF_NAME}", "--out", frozen],
        "export_frozen_head", env={"CUDA_VISIBLE_DEVICES": ""})
man_p = os.path.join(TARGETS_DIR, "manifest.json")
if os.path.exists(man_p):
    man = _json.load(open(man_p))
    print(f"package: {man['total_tokens'] / 1e6:.2f}M tokens, {man['total_bytes'] / 1e9:.1f} GB, "
          f"top_k {man.get('top_k')}, hidden {man.get('hidden_dtype')}")
else:
    print(f"no manifest at {TARGETS_DIR} yet - Phase B needs it (run Phase A first)")
'''

T4CELL4 = r'''#@title 4. Phase B: teacher-free student training from the package (T4, FP16 + GradScaler)
RUN_PHASE_B = True  #@param {type:"boolean"}
DRAFTER = "mtp"  #@param ["mtp", "df3"]
INIT_MODE = "procreations_r3"  #@param ["procreations_r3", "drive_safetensors", "qwen_blk64", "random"]
INIT_HF_REPO = "mcpe500/bonsai-mtp"  #@param {type:"string"}
INIT_HF_FILE = "model_mtp.safetensors"  #@param {type:"string"}
INIT_DRIVE_PATH = "/content/drive/MyDrive/bonsai_mtp/final_t4/mtp_blk64.safetensors"  #@param {type:"string"}
STEPS = 2000  #@param {type:"integer"}
BATCH = 2  #@param {type:"integer"}
GRAD_ACCUM = 8  #@param {type:"integer"}
LR = 2e-4  #@param {type:"number"}
WARMUP = 100  #@param {type:"integer"}
EPOCHS = 3  #@param {type:"integer"}
OPTIM = "adamw8bit"  #@param ["adamw8bit", "adafactor", "adamw"]
UNROLL = 4  #@param {type:"integer"}
DF3_BLOCK = 16  #@param {type:"integer"}
SAVE_EVERY = 25  #@param {type:"integer"}
SAVE_EVERY_MIN = 20  #@param {type:"number"}
STAGE_DIR = "/content/target_cache"  #@param {type:"string"}
RUN_DIR = "/content/drive/MyDrive/bonsai_mtp/run_t4"  #@param {type:"string"}
RESUME_FROM = "auto"  #@param ["auto", "none"]

# TOP_K / HIDDEN_DTYPE come from the package manifest (never from a skipped cell), so this cell is safe to run
# on its own after Phase A produced the package once.
import json as _json

man_p = os.path.join(TARGETS_DIR, "manifest.json")
if not os.path.exists(man_p):
    raise RuntimeError(f"no target package at {TARGETS_DIR}: run cell 3 (Phase A) first - it writes the manifest")
man = _json.load(open(man_p))
TOP_K = int(man.get("top_k", 64))
HIDDEN_DTYPE = str(man.get("hidden_dtype", "int4g64"))
print(f"package: {man['total_tokens'] / 1e6:.2f}M tokens, {man['total_bytes'] / 1e9:.1f} GB; "
      f"loss width top_k={TOP_K}, hidden {HIDDEN_DTYPE}")
if not RUN_PHASE_B:
    print("RUN_PHASE_B = False: skipped")
else:
    init_args = []
    if INIT_MODE == "procreations_r3":
        init_st = f"/content/models/{INIT_HF_FILE}"
        if not os.path.exists(init_st):
            os.makedirs("/content/models", exist_ok=True)
            run([PY, "-m", "huggingface_hub.commands.huggingface_cli", "download", INIT_HF_REPO, INIT_HF_FILE,
                 "--local-dir", "/content/models"], "hf_download_init")
        init_args = ["--init-mtp-safetensors", init_st]
    elif INIT_MODE == "drive_safetensors":
        if not os.path.exists(INIT_DRIVE_PATH):
            raise RuntimeError(f"--init safetensors not found: {INIT_DRIVE_PATH}")
        init_args = ["--init-mtp-safetensors", INIT_DRIVE_PATH]
    elif INIT_MODE == "qwen_blk64":
        init_args = ["--init", f"gguf:/content/models/{GGUF_NAME}"]
    run([PY, f"{TOOLS}/train_bonsai_mtp.py", "--from-targets", TARGETS_DIR,
         "--lm-head-file", os.path.join(TARGETS_DIR, "lm_head_embed_frozen.safetensors"),
         "--out", RUN_DIR, "--drafter", str(DRAFTER), "--unroll", str(UNROLL), "--df3-block", str(DF3_BLOCK),
         "--steps", str(STEPS), "--seq-len", "1024", "--batch", str(BATCH), "--grad-accum", str(GRAD_ACCUM),
         "--lr", str(LR), "--warmup", str(WARMUP), "--epochs", str(EPOCHS), "--optim", str(OPTIM),
         "--top-k-kl", str(TOP_K), "--hidden-dtype", HIDDEN_DTYPE, "--save-every", str(SAVE_EVERY),
         "--save-every-min", str(SAVE_EVERY_MIN), "--keep-last", "2", "--resume-from", str(RESUME_FROM),
         "--log-every", "10", "--device", "cuda", "--stage-dir", STAGE_DIR] + init_args, "train_t4")
    with open(os.path.join(RUN_DIR, "train_log.jsonl")) as f:
        last = _json.loads(f.readlines()[-1])
    print(f"last step {last['step']}: top1_agree {[round(x, 3) for x in last['top1_agree']]} "
          f"({last.get('sec_per_step', '?')} s/step, eta {last.get('eta_h', '?')} h)")
'''

T4CELL5 = r'''#@title 5. Export the trained head to sidecar GGUFs (same plug-and-play file as always)
EXPORT_QUANTS = "q4_k_m,q8_0"  #@param {type:"string"}
FINAL = f"{DRIVE_OUT}/final_t4"

latest = subprocess.run([PY, f"{TOOLS}/train_bonsai_mtp.py", "--latest", RUN_DIR], capture_output=True, text=True)
if latest.returncode != 0:
    raise RuntimeError(f"no checkpoint in {RUN_DIR}: run cell 4 first.\n{latest.stderr}")
ckpt = latest.stdout.strip()
print(f"checkpoint: {ckpt}")
os.makedirs(FINAL, exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


lines = []
for qt in [q.strip() for q in EXPORT_QUANTS.split(",") if q.strip()]:
    out = os.path.join(FINAL, f"mtp_blk64_{qt}.gguf")
    run([PY, f"{TOOLS}/inject_mtp_gguf.py", "--from-checkpoint", ckpt, "--quant", qt, "--mtp-only-out", out],
        f"export_{qt}", env={"CUDA_VISIBLE_DEVICES": ""})
    lines.append(f"mtp_blk64_{qt}.gguf  {os.path.getsize(out)} B  sha256 {sha256(out)}")
qt0 = [q.strip() for q in EXPORT_QUANTS.split(",") if q.strip()][0]
print("\n".join(lines))
print(f"""
Sidecar GGFs on Drive: {FINAL}. On your PC (engine built with ./scripts/setup.sh):

  taskset -c 0,1,2,3 ./build/qwen38-engine -m ~/models/27b-bonsai/Ternary-Bonsai-2-27B-PTQ1_0.gguf \\
      --mtp-gguf ~/models/27b-bonsai/mtp_blk64_{qt0}.gguf --gpu-layers auto --ctx 4096 \\
      --mtp-adaptive 1 --gen-mtp-steps 2 -f bench/prompt512.txt -n 256
""")
'''

TROUBLE_T4 = """## Troubleshooting (T4 notebook)

- **Cell 2 build fails:** the engine needs the ggml CPU libraries of the `mcpe500/llama.cpp` fork
  (`feat/iso3-kv`); the cell clones and builds them automatically (~3-8 min on the T4's 2 vCPUs, once per
  session). The engine build itself targets `CMAKE_CUDA_ARCHITECTURES=75`; `LLAMA_SRC`/`LLAMA_BUILD` are passed
  through the environment. A "GGML_LIB not found" means the llama.cpp build failed — open its log in Drive.
- **Phase A says "no corpus":** Phase A tokenizes from the `prepare_corpus.py` shards on Drive (balanced over
  OpenCodeInstruct / OpenThoughts / UltraChat / Wikipedia-ID / Aya). Run the main notebook's corpus step once;
  it lands in `DRIVE_OUT/corpus`.
- **HF download fails:** set `HF_REPO` / `INIT_HF_REPO` to the repos that hold your
  `Ternary-Bonsai-2-27B-PTQ1_0.gguf` (5,946,648,928 bytes) and your warm-start `model_mtp.safetensors`.
- **CUDA out of memory in cell 3:** keep `VRAM_BUDGET_MIB >= 14000` (the resident 27B is ~5.7 GiB plus the FP16
  scratch); lower `SEQ_LEN` (1024 -> 512) if the runtime is smaller than a full T4.
- **Slow steps in cell 4 with Drive shards:** keep `STAGE_DIR` at local SSD (`/content/target_cache`), not Drive.
- **`bitsandbytes` WARNING in cell 2:** `--optim adamw8bit` falls back to Adafactor automatically.
- **KL is >= 0:** the sparse top-K loss is the renormalized teacher distribution
  (`softmax(topk_vals)` vs the student's renormalized K logits) — a true KL divergence. `top1_agree` is the
  number to watch (the engine's acceptance proxy), then the real engine race.
"""


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {"cellView": "form"}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def main():
    def write_nb(path, name, cells):
        nb = {
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {
                "accelerator": "GPU",
                "colab": {"provenance": [], "machine_shape": "hm", "name": name},
                "kernelspec": {"display_name": "Python 3", "name": "python3"},
                "language_info": {"name": "python"},
            },
            "cells": cells,
        }
        for i, c in enumerate(nb["cells"]):
            c["id"] = f"cell-{i}"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"wrote {path}")

    write_nb(OUT, "train_bonsai_mtp_colab.ipynb",
             [md(INTRO), code(CELL1), code(CELL2), code(CELL3), code(CELL4), code(CELL5), code(CELL6), md(TROUBLE)])
    out_t4 = os.path.join(HERE, "train_bonsai_t4_student_colab.ipynb")
    write_nb(out_t4, "train_bonsai_t4_student_colab.ipynb",
             [md(INTRO_T4), code(T4CELL1), code(T4CELL2), code(T4CELL3), code(T4CELL4), code(T4CELL5), md(TROUBLE_T4)])


if __name__ == "__main__":
    main()
