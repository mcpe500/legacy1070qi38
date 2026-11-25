#!/usr/bin/env bash
# Ternary Bonsai 2.5 27B vs Ternary Bonsai 2 27B on this machine (GTX 1070 Ti, Ryzen 5 2400G): download, load check, parity gates,
# speed/VRAM/RSS per context, 128K needle + state round trip, MTP race, intelligence (PPL, KLD, task subsets), report, branch, PR.
#
# Run it in YOUR OWN terminal (tmux/ssh, NOT inside the opencode session: stage "gpu" SIGSTOPs every opencode process).
# The machine needs the GPU, network (huggingface.co, github.com) and a logged-in `gh`.
#
#   bench/bonsai25_vs_2.sh                      # every stage, in order (several hours; 128K prefill is ~60 min per model)
#   STAGES="download probe build" bench/bonsai25_vs_2.sh
#   STAGES="depth mtp report" bench/bonsai25_vs_2.sh
#   SKIP_PR=1 bench/bonsai25_vs_2.sh            # everything except push + gh pr create
#
# Stages (in this order): branch download probe build gates load depth needle mtp intel report pr
#   branch    git fetch, then git checkout -b exp/bonsai-2.5-vs-2 origin/main (never touches main); the script and the summarizer are carried over
#   download  hf download both Bonsai 2.5 GGUFs into ~/models/27b-bonsai-2.5, verify size and LFS sha256
#   probe     gguf tensor/metadata probe (blk.64 / nextn tensors, block_count, tokenizer hash), try to extract the 2.5 head as a sidecar (CPU only)
#   build     taskset -c 0-3 ninja -C build -j4 (qwen38-engine, ternary_ab_test) and freeze a copy of the engine for the whole run
#   gates     scripts/sass_gate.sh (must PASS), ternary_ab_test (must be bit-identical)
#   load      does the engine load Bonsai 2.5 and give sane greedy output (versus Bonsai 2, same prompt)
#   depth     decode tok/s, peak VRAM, RSS at 512..131072 (depth probe, protocol of bonsai_overnight_v2/depth_matrix.md) + real 4K generation
#   needle    128K-class needle prompts + state round trip (prefill tok/s, answer, VRAM); Bonsai 2 only up to 32K unless B2_LONG=1
#   mtp       MTP race (protocol of bonsai_overnight_v2/mtp_race): 2.5 built-in head, r3 sidecar, user head, extracted 2.5 head; lossless check
#   intel     llama-perplexity: WikiText-2 PPL, KLD vs Qwen Q4_K_M, HellaSwag/Winogrande/ARC/MMLU subsets (same commands as bench/results/bonsai/intel)
#   report    bench/bonsai25_vs_2_summary.py -> REPORT.md, figure
#   pr        git add (script, summarizer, results dir only), commit, push -u origin, gh pr create --base main
#
# Every stage is resumable: a run whose result file already exists is skipped (FORCE=1 reruns everything; delete a raw/<tag>.txt to rerun one).
# A failed run (for example 128K not fitting) is recorded in raw/<tag>.fail.txt and the script goes on; nothing here aborts the whole run
# except: no GPU, a failed download/sha256, a failed SASS gate, or a dirty tree at the branch step.
#
# Env: STAGES, SKIP_PR=1, FORCE=1, B2_LONG=1 (also run Bonsai 2 at 64K/128K needle+state, +2 h), NEEDLE_CTXS="4096 32768 65536 131072",
#      DEPTH_CTXS, MTP_STEPS="2" (or "2 3"), B2_INTEL_FILL=0 (skip the Bonsai 2 PTQ1_0 ARC/MMLU fill-in runs), PRISM_DIR, WORK (scratch, default ~/kiro-tmp/bonsai25),
#      BRANCH, PAUSE_OPENCODE=0 (do not SIGSTOP opencode), HF_TOKEN.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 2
BR="${BRANCH:-exp/bonsai-2.5-vs-2}"
RES="$ROOT/bench/results/bonsai25_vs_2"
RAW="$RES/raw"
WORK="${WORK:-$HOME/kiro-tmp/bonsai25}"
M2DIR="$HOME/models/27b-bonsai"
M25DIR="$HOME/models/27b-bonsai-2.5"
HF_REPO="prism-ml/Ternary-Bonsai-2.5-27B-GGUF"
M2="$M2DIR/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
F25="Ternary-Bonsai-2.5-27B-PTQ1_0.gguf";     S25=5993834848; H25=929560ec9d37cf3cadd4b00cc87b51c570082a628dd830b4692dc0b2ed18dbbe
F25M="Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf"; S25M=6232818944; H25M=e0fc5f3d47b9c926d52677d303b19df356a2376614b3e1cb9973865fc04ca431
M25="$M25DIR/$F25"
M25MTP="$M25DIR/$F25M"
R3="$M2DIR/procreations-mtp/r3_mtp_blk64_q4_k_m.gguf"
UQ4="$M2DIR/mtp_blk64_q4_k_m.gguf"
QWEN="$HOME/models/27b/Qwen3.8-27B-UD-Q4_K_M.gguf"
GGUFPY="$HOME/llama.cpp/gguf-py"
STOCK="$HOME/llama.cpp/build-iso3/bin"
PRISM="${PRISM_DIR:-$HOME/llama-prism/llama-prism-b12128-d7b2896}"
KLD_BASE="${KLD_BASE:-$HOME/kiro-tmp/qwen_q4km_wiki40.kld}"
ENG="$WORK/qwen38-engine.frozen"
ALL_STAGES="branch download probe build gates load depth needle mtp intel report pr"
STAGES="${STAGES:-$ALL_STAGES}"
DEPTH_CTXS="${DEPTH_CTXS:-512 4096 16384 32768 65536 98304 114688 131072}"
NEEDLE_CTXS="${NEEDLE_CTXS:-4096 32768 65536 131072}"
MTP_STEPS="${MTP_STEPS:-2}"
export PATH="$HOME/.local/bin:$PATH"

now() { date '+%F %T'; }
log() { echo "[$(now)] $*"; }
want() { [[ " $STAGES " == *" $1 "* ]]; }
die() { log "FATAL: $*"; exit 1; }
banner() { echo; echo "=================================================================="; log "STAGE $1"; echo "=================================================================="; }
status() { mkdir -p "$RES"; printf '%s\t%s\t%s\n' "$(now)" "$1" "$2" >> "$RES/status.tsv"; log "$1: $2"; }

# ---------------------------------------------------------------------------------------------------------------- GPU isolation
GPU_HELD=0
gpu_begin() {
  [ "$GPU_HELD" = 1 ] && return 0
  command -v nvidia-smi > /dev/null 2>&1 || die "nvidia-smi not found: this stage needs the GPU"
  nvidia-smi > /dev/null 2>&1 || die "nvidia-smi fails: no usable GPU"
  GPU_HELD=1
  trap 'gpu_end' EXIT INT TERM       # SIGCONT must always happen
  if [ "${PAUSE_OPENCODE:-1}" = 1 ]; then "$ROOT/bench/gpu_isolate.sh" pause; fi
  "$ROOT/scripts/gpu_isolate.sh" || die "scripts/gpu_isolate.sh refuses to start (GPU busy or over budget); free the GPU and rerun"
  pgrep -x llama-server > /dev/null && die "llama-server is running; stop it first (never run it next to the engine)"
}
gpu_end() {
  [ "$GPU_HELD" = 1 ] || return 0
  GPU_HELD=0
  "$ROOT/bench/gpu_isolate.sh" resume
}

# ---------------------------------------------------------------------------------------------------------------- engine runs
b25_ok() { grep -qs '"rc":0' "$RAW/b25_sanity.txt"; }

run() {   # run TAG engine-args... : one engine run through bench/bonsai_ctx_run.sh; skipped when raw/TAG.txt already has its result line
  local tag="$1"; shift
  if [ -z "${FORCE:-}" ] && grep -qs "\"tag\":\"$tag\"" "$RAW/$tag.txt"; then log "skip $tag (done)"; return "$(grep -o '"rc":[0-9]*' "$RAW/$tag.txt" | tail -1 | cut -d: -f2)"; fi
  local rc try
  for try in 1 2 3 4 5 6; do
    log "run $tag"
    OUT_DIR="$RAW" ENGINE_BIN="$ENG" "$ROOT/bench/bonsai_ctx_run.sh" "$tag" "$@"
    rc=$?
    [ "$rc" = 3 ] || break
    log "GPU busy, retry in 30 s ($try/6)"; sleep 30
  done
  log "done $tag rc=$rc: $(tail -1 "$RAW/$tag.txt" 2> /dev/null)"
  if [ "$rc" != 0 ]; then
    { echo "tag=$tag rc=$rc"; echo "args: $*"; tail -8 "$RAW/$tag.err" 2> /dev/null; } > "$RAW/$tag.fail.txt"
    status "run-failed" "$tag rc=$rc (see raw/$tag.fail.txt)"
  else
    rm -f "$RAW/$tag.fail.txt"
  fi
  return "$rc"
}

COMMON=(--gpu-layers auto)

# ================================================================================================================ main
main() {
  cd "$ROOT" || exit 2

  # ------------------------------------------------------------------------------------------------------------ branch
  if want branch; then
    banner "branch: $BR from origin/main"
    git fetch origin || die "git fetch origin failed (network?)"
    cur="$(git branch --show-current)"
    if [ "$cur" = "$BR" ]; then
      log "already on $BR"
    else
      if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
        git status --short --untracked-files=no
        die "tracked changes in the work tree; commit or stash them first (untracked files are fine and are carried over)"
      fi
      if git show-ref --verify --quiet "refs/heads/$BR"; then
        git checkout "$BR" || die "git checkout $BR failed"
      else
        git checkout -b "$BR" origin/main || die "git checkout -b $BR origin/main failed (an untracked file would be overwritten?)"
      fi
    fi
    [ -f "$ROOT/bench/bonsai25_vs_2.sh" ] && [ -f "$ROOT/bench/bonsai25_vs_2_summary.py" ] || die "script or summarizer missing after checkout"
    log "now on $(git branch --show-current) at $(git rev-parse --short HEAD)"
  fi

  mkdir -p "$RAW" "$WORK" "$RES/intel" "$RES/prompts"
  exec > >(tee -a "$RES/run.log") 2>&1
  log "bonsai25_vs_2 started, STAGES=[$STAGES] ROOT=$ROOT WORK=$WORK"

  # ------------------------------------------------------------------------------------------------------------ download
  if want download; then
    banner "download Bonsai 2.5 into $M25DIR"
    command -v hf > /dev/null || die "hf CLI not found (~/.local/bin/hf)"
    mkdir -p "$M25DIR"
    avail=$(df -BM --output=avail "$M25DIR" | tail -1 | tr -dc 0-9)
    log "free disk: ${avail} MiB"
    verify() {   # verify FILE SIZE SHA: 0 ok, 1 missing or wrong size, 2 sha mismatch (a marker file caches a good sha256)
      local f="$1" sz="$2" sha="$3" mk="$1.sha256ok" got
      [ -f "$f" ] && [ "$(stat -c %s "$f")" = "$sz" ] || return 1
      [ -f "$mk" ] && [ "$(cat "$mk")" = "$sha $(stat -c %Y "$f")" ] && return 0
      log "sha256sum $(basename "$f") ..."
      got="$(sha256sum "$f" | awk '{print $1}')"
      [ "$got" = "$sha" ] || { log "sha256 mismatch: got $got want $sha"; return 2; }
      echo "$sha $(stat -c %Y "$f")" > "$mk"
    }
    for spec in "$F25:$S25:$H25" "$F25M:$S25M:$H25M"; do
      IFS=: read -r f sz sha <<< "$spec"
      verify "$M25DIR/$f" "$sz" "$sha"
      rc=$?
      if [ "$rc" = 0 ]; then log "$f present and verified"; continue; fi
      [ "$rc" = 2 ] && { mv "$M25DIR/$f" "$M25DIR/$f.bad" && log "moved the bad file to $f.bad"; }
      [ "${avail:-0}" -gt 7000 ] || die "less than 7 GB free in $M25DIR"
      log "hf download $HF_REPO $f"
      hf download "$HF_REPO" "$f" --local-dir "$M25DIR" || die "hf download $f failed (network? gated repo? set HF_TOKEN)"
      verify "$M25DIR/$f" "$sz" "$sha" || die "$f failed size/sha256 verification after download"
      log "$f verified: $sz bytes, sha256 $sha"
    done
    status "download" "both files verified"
  fi

  for f in "$M2" "$R3" "$UQ4"; do [ -f "$f" ] || log "WARNING: missing $f (the stages that use it will fail and be recorded)"; done

  # ------------------------------------------------------------------------------------------------------------ probe
  if want probe; then
    banner "probe: GGUF metadata and tensor names (blk.64 / nextn)"
    [ -f "$M25" ] || die "$M25 missing: run the download stage"
    PYTHONPATH="$GGUFPY" python3 - "$RES" "$M2" "$M25" "$M25MTP" <<'PY'
import hashlib, json, os, sys
import gguf
res = sys.argv[1]
def scalar(f):
    try:
        t = f.types[0]
        if t == gguf.GGUFValueType.STRING:
            return bytes(f.parts[f.data[0]]).decode("utf-8", "replace")
        if t == gguf.GGUFValueType.ARRAY:
            return "array[%d]" % len(f.data)
        v = f.parts[f.data[0]].tolist()
        return v[0] if isinstance(v, list) and len(v) == 1 else v
    except Exception as e:
        return "?%s" % e
def probe(path):
    out = {"file": os.path.basename(path), "size": os.path.getsize(path)}
    if not os.path.exists(path):
        out["error"] = "missing"; return out
    try:
        r = gguf.GGUFReader(path)
    except Exception as e:
        out["error"] = "GGUFReader: %s" % e; return out
    keys = ("architecture", "name", "file_type", "block_count", "nextn", "mtp", "context_length", "embedding_length", "vocab", "quantization", "expert", "ssm", "full_attention")
    out["meta"] = {k: scalar(f) for k, f in r.fields.items() if any(w in k for w in keys) and not k.startswith("tokenizer.ggml.merges")}
    tf = r.fields.get("tokenizer.ggml.tokens")
    if tf is not None:
        h = hashlib.sha256()
        for i in tf.data:
            h.update(bytes(tf.parts[i])); h.update(b"\0")
        out["tokenizer_tokens"] = len(tf.data); out["tokenizer_sha256"] = h.hexdigest()
    hist, tensors, names = {}, [], set()
    maxblk = -1
    for t in r.tensors:
        names.add(t.name)
        ty = t.tensor_type.name
        h = hist.setdefault(ty, [0, 0]); h[0] += 1; h[1] += int(t.n_bytes)
        if t.name.startswith("blk."):
            try: maxblk = max(maxblk, int(t.name.split(".")[1]))
            except ValueError: pass
        if t.name.startswith("blk.64.") or "nextn" in t.name or "mtp" in t.name.lower():
            tensors.append({"name": t.name, "type": ty, "shape": [int(x) for x in t.shape], "bytes": int(t.n_bytes)})
    out.update(n_tensors=len(names), max_block_index=maxblk, type_hist={k: {"n": v[0], "bytes": v[1]} for k, v in hist.items()}, head_tensors=tensors, _names=sorted(names))
    return out
paths = sys.argv[2:]
P = [probe(p) for p in paths]
b2 = set(P[0].get("_names", []))
for p in P[1:]:
    n = set(p.get("_names", []))
    p["only_in_this_vs_bonsai2"] = sorted(n - b2)[:60]
    p["only_in_bonsai2_vs_this"] = sorted(b2 - n)[:60]
for p in P: p.pop("_names", None)
json.dump(P, open(os.path.join(res, "probe.json"), "w"), indent=1)
with open(os.path.join(res, "probe.txt"), "w") as fh:
    for p in P:
        fh.write("== %s (%d bytes)\n" % (p["file"], p["size"]))
        if "error" in p: fh.write("  %s\n" % p["error"]); continue
        for k, v in sorted(p["meta"].items()): fh.write("  %s = %s\n" % (k, v))
        fh.write("  tensors: %d, highest blk index %d, tokenizer %s tokens sha256 %s\n" % (p["n_tensors"], p["max_block_index"], p.get("tokenizer_tokens"), p.get("tokenizer_sha256")))
        fh.write("  types: %s\n" % json.dumps(p["type_hist"]))
        fh.write("  blk.64 / nextn / mtp tensors (%d):\n" % len(p["head_tensors"]))
        for t in p["head_tensors"]: fh.write("    %-48s %-8s %s %d B\n" % (t["name"], t["type"], t["shape"], t["bytes"]))
        for k in ("only_in_this_vs_bonsai2", "only_in_bonsai2_vs_this"):
            if k in p: fh.write("  %s: %s\n" % (k, p[k]))
print(open(os.path.join(res, "probe.txt")).read())
PY
    [ $? = 0 ] && status "probe" "probe.json / probe.txt written" || status "probe-failed" "python gguf probe failed (is $GGUFPY present?)"
    if [ -f "$M25MTP" ]; then
      log "extracting the 2.5 head as a sidecar (CPU only)"
      rm -f "$WORK/b25_head_blk64.gguf"
      python3 tools/bonsai_mtp_train/inject_mtp_gguf.py --from-gguf "$M25MTP" --mtp-only-out "$WORK/b25_head_blk64.gguf" > "$RES/head_extract.log" 2>&1 \
        && status "head-extract" "ok $(stat -c %s "$WORK/b25_head_blk64.gguf") bytes" || status "head-extract-failed" "see head_extract.log"
    fi
  fi

  # ------------------------------------------------------------------------------------------------------------ build
  if want build; then
    banner "build: taskset -c 0-3 ninja -C build -j4"
    taskset -c 0-3 ninja -C build -j4 qwen38-engine ternary_ab_test || die "ninja build failed"
    cp -f build/qwen38-engine "$ENG" && log "frozen engine copy: $ENG ($(git rev-parse --short HEAD))"
    echo "engine from $(git rev-parse HEAD) on $(git branch --show-current)" > "$RES/engine_build.txt"
    status "build" "ok"
  fi
  [ -x "$ENG" ] || { [ -x build/qwen38-engine ] && cp -f build/qwen38-engine "$ENG"; }
  [ -x "$ENG" ] || die "no engine binary ($ENG); run the build stage"

  # ------------------------------------------------------------------------------------------------------------ gates
  if want gates || want load || want depth || want needle || want mtp || want intel; then gpu_begin; fi

  if want gates; then
    banner "gates: SASS gate and ternary_ab_test"
    bash scripts/sass_gate.sh > "$RES/sass_gate.txt" 2>&1; rc=$?
    tail -4 "$RES/sass_gate.txt"
    [ "$rc" = 0 ] || { status "SASS-GATE-FAIL" "rc=$rc"; die "SASS gate failed (bench/results/bonsai25_vs_2/sass_gate.txt)"; }
    status "sass_gate" "PASS"
    scripts/gpu_isolate.sh -- taskset -c 0-3 build/ternary_ab_test > "$RES/ternary_ab_test.txt" 2>&1; rc=$?
    tail -2 "$RES/ternary_ab_test.txt"
    [ "$rc" = 0 ] && status "ternary_ab_test" "$(tail -1 "$RES/ternary_ab_test.txt")" || status "ternary_ab_test-FAIL" "rc=$rc"
  fi

  # ------------------------------------------------------------------------------------------------------------ load
  if want load; then
    banner "load: does the engine load Bonsai 2.5 (greedy sanity versus Bonsai 2)"
    run b2_sanity  -m "$M2"  "${COMMON[@]}" --ctx 4096 -f bench/prompt5.txt -n 48 --temp 0 --seed 42 --mtp-steps 0
    run b25_sanity -m "$M25" "${COMMON[@]}" --ctx 4096 -f bench/prompt5.txt -n 48 --temp 0 --seed 42 --mtp-steps 0
    python3 - "$RAW" > "$RES/sanity.md" <<'PY'
import json, sys, os
raw = sys.argv[1]
def get(t):
    try:
        for l in open(os.path.join(raw, t + ".txt"), encoding="utf-8"):
            if l.startswith("{") and '"ids"' in l: return json.loads(l)
    except Exception: pass
a, b = get("b2_sanity"), get("b25_sanity")
print("# Greedy sanity (prompt `The capital of France is`, 48 tokens, temp 0)\n")
for n, d in (("Bonsai 2", a), ("Bonsai 2.5", b)):
    if not d: print("- %s: **no result** (engine failed to load or run, see raw/*.fail.txt)" % n); continue
    ids = d["ids"]; print("- %s: %d tokens, %d distinct ids, decode %.2f tok/s, text: `%s`" % (n, len(ids), len(set(ids)), d["decode_tok_s"], d["text"][:160].replace("\n", " ")))
if a and b:
    n = next((i for i, (x, y) in enumerate(zip(a["ids"], b["ids"])) if x != y), min(len(a["ids"]), len(b["ids"])))
    print("\nFirst %d of %d generated ids are identical between the two models (different weights, so a divergence is expected)." % (n, len(a["ids"])))
    ok = len(set(b["ids"])) > 5 and len(b["text"].strip()) > 10
    print("\nVerdict: %s" % ("sane (varied, non-empty text)" if ok else "SUSPICIOUS output (degenerate or empty)"))
PY
    cat "$RES/sanity.md"
    b25_ok && status "load" "Bonsai 2.5 loads and generates" || status "load-FAILED" "Bonsai 2.5 did not load; dependent 2.5 stages are skipped, see raw/b25_sanity.fail.txt"
  fi

  # ------------------------------------------------------------------------------------------------------------ depth
  if want depth; then
    banner "depth: decode tok/s, VRAM, RSS versus context"
    depth_run() {   # depth_run LABEL MODEL CTX extra-args... ; 131072 falls back to 122880 when it fails
      local lab="$1" m="$2" c="$3"; shift 3
      local cx=$(( c < 4096 ? 4096 : c ))
      run "${lab}_d${c}_depth" -m "$m" "${COMMON[@]}" --ctx "$cx" --depth-bench "$c" -n 33 "$@"
      if [ $? != 0 ] && [ "$c" = 131072 ]; then
        log "ctx 131072 refused for $lab, trying 122880"
        run "${lab}_d122880_depth" -m "$m" "${COMMON[@]}" --ctx 122880 --depth-bench 122880 -n 33 "$@"
      fi
    }
    for c in $DEPTH_CTXS; do
      depth_run b2 "$M2" "$c"
      b25_ok && depth_run b25 "$M25" "$c"
    done
    if b25_ok && [ -f "$M25MTP" ]; then   # the file with the head: without MTP (mtp-steps 0) and with it, as in depth_matrix.md
      for c in 4096 32768 65536 114688 131072; do
        depth_run b25mtp "$M25MTP" "$c" --mtp-steps 0
        depth_run b25mtpon "$M25MTP" "$c" --mtp-adaptive 0 --mtp-steps 2
      done
    fi
    log "real 4K prompt, no MTP"
    run b2_p4k_s0 -m "$M2" "${COMMON[@]}" --ctx 8192 -f bench/prompt4k.txt -n 128 --temp 0 --seed 42 --mtp-steps 0
    b25_ok && run b25_p4k_s0 -m "$M25" "${COMMON[@]}" --ctx 8192 -f bench/prompt4k.txt -n 128 --temp 0 --seed 42 --mtp-steps 0
  fi

  # ------------------------------------------------------------------------------------------------------------ needle
  if want needle; then
    banner "needle: long prompts and state round trip"
    printf '%s' "$(python3 -c "import sys; sys.path.insert(0,'bench'); import make_long_prompt as m; sys.stdout.write(m.QUESTION)")" > "$WORK/question.txt"
    needle_run() {   # needle_run LABEL MODEL CTX
      local lab="$1" m="$2" c="$3" n=$(( $3 - 64 ))
      mkdir -p "$WORK/p"
      if [ ! -s "$WORK/p/pre$n.txt" ]; then
        log "make_long_prompt $n tokens (tokenizer of Bonsai 2, shared by both models)"
        python3 bench/make_long_prompt.py "$M2" "$n" "$WORK/p/p$n.txt" --prefix-out "$WORK/p/pre$n.txt" || { status "prompt-failed" "$n"; return 1; }
      fi
      run "${lab}_c${c}_statetest" -m "$m" "${COMMON[@]}" --ctx "$c" --parse-special -f "$WORK/p/pre$n.txt" --state-test "$WORK/${lab}_$c.state" --cont-file "$WORK/question.txt" -n 24
      local rc=$?
      rm -f "$WORK/${lab}_$c.state"
      return $rc
    }
    for c in $NEEDLE_CTXS; do
      if [ "$c" -le 32768 ] || [ "${B2_LONG:-0}" = 1 ]; then needle_run b2 "$M2" "$c"
      else log "Bonsai 2 at ctx $c: reusing the recorded run in bench/results/bonsai128k_v2 (B2_LONG=1 to rerun, about 1 h per 128K)"; fi
      if b25_ok; then
        needle_run b25 "$M25" "$c"
        if [ $? != 0 ] && [ "$c" = 131072 ]; then log "131072 failed for 2.5, trying 122880"; needle_run b25 "$M25" 122880; fi
      fi
    done
  fi

  # ------------------------------------------------------------------------------------------------------------ mtp
  if want mtp; then
    banner "mtp: race (lossless: same ids as the no-draft baseline of the same model)"
    cp -f bench/prompt512.txt "$RES/prompts/p512.txt"
    printf '%s\n' 'Implement a thread-safe LRU cache in Python with O(1) get and put. Provide the full class `LRUCache` with `__init__(self, capacity: int)`, `get(self, key: str) -> object`, and `put(self, key: str, value: object) -> None`, using `collections.OrderedDict` and a `threading.Lock`. Then explain the eviction policy in one sentence and state the time complexity of each method.' > "$RES/prompts/coding.txt"
    printf '%s\n' 'Jelaskan perbedaan antara list, tuple, dan set dalam bahasa Python: kapan masing-masing sebaiknya dipakai, bagaimana perilaku duplikat dan urutan pada setiap struktur, dan berikan satu contoh kode singkat untuk masing-masing.' > "$RES/prompts/id.txt"
    race() {   # race CFG MODEL [sidecar-or-empty] : base (--gen-mtp-steps 0) and the MTP runs for every workload
      local cfg="$1" m="$2" sc="${3:-}" w s
      for w in p512 coding id; do
        if [ "${cfg##*_}" = base ]; then
          run "race_${cfg}_${w}" -m "$m" "${COMMON[@]}" --ctx 4096 -f "$RES/prompts/$w.txt" -n 256 --temp 0 --seed 42 --gen-mtp-steps 0
        else
          for s in $MTP_STEPS; do
            if [ -n "$sc" ]; then
              run "race_${cfg}_s${s}_${w}" -m "$m" "${COMMON[@]}" --ctx 4096 -f "$RES/prompts/$w.txt" -n 256 --temp 0 --seed 42 --mtp-gguf "$sc" --mtp-adaptive 1 --gen-mtp-steps "$s"
            else
              run "race_${cfg}_s${s}_${w}" -m "$m" "${COMMON[@]}" --ctx 4096 -f "$RES/prompts/$w.txt" -n 256 --temp 0 --seed 42 --mtp-adaptive 1 --gen-mtp-steps "$s"
            fi
          done
        fi
      done
    }
    race b2_base "$M2"
    race b2_r3  "$M2" "$R3"
    race b2_uq4 "$M2" "$UQ4"
    if b25_ok; then
      race b25_base "$M25"
      race b25_r3  "$M25" "$R3"        # the Bonsai 2 heads on the 2.5 trunk: lossless either way, only acceptance changes
      race b25_uq4 "$M25" "$UQ4"
      if [ -f "$M25MTP" ]; then
        race b25mtp_base "$M25MTP"
        race b25mtp_builtin "$M25MTP"   # built-in head (blk.64 of the same file); if the engine rejects it the failure is recorded in raw/*.fail.txt
      fi
      [ -s "$WORK/b25_head_blk64.gguf" ] && race b25head "$M25" "$WORK/b25_head_blk64.gguf"   # the 2.5 head extracted as a sidecar
    fi
  fi

  # ------------------------------------------------------------------------------------------------------------ intel
  if want intel; then
    banner "intel: llama-perplexity (never together with the engine)"
    IDIR="$ROOT/bench/results/bonsai/intel"      # bench/bonsai_intel.sh writes here; the new logs are moved into $RES/intel
    RL="$RES/intel/run_lines.txt"; touch "$RL"
    ibench() {   # ibench TAG MODEL BINDIR NGL TASK N UB   (env KLD_BASE, KV, LD_LIBRARY_PATH from the caller)
      local tag="$1" model="$2" bin="$3" ngl="$4" task="$5" n="$6" ub="$7"
      if [ -z "${FORCE:-}" ] && grep "^$tag: " "$RL" | tail -1 | grep -qE ' OK( OVER CAP)?$'; then log "skip $tag (done)"; return 0; fi
      [ -x "$bin/llama-perplexity" ] || { echo "$tag: task=$task n=$n ngl=$ngl score=[none] no llama-perplexity in $bin FAILED rc=127" >> "$RL"; return 1; }
      log "intel $tag ($task n=$n ngl=$ngl bin=$bin)"
      local line
      line="$(UB="$ub" KV=q4_0 KLD_BASE="$KLD_BASE" "$ROOT/bench/bonsai_intel.sh" "$model" "$bin" "$tag" "$ngl" "$task" "$n")"
      echo "$line" | tee -a "$RL"
      for ext in log vram.csv; do [ -f "$IDIR/$tag.$ext" ] && mv -f "$IDIR/$tag.$ext" "$RES/intel/$tag.$ext"; done
      echo "$line" | grep -qE ' OK( OVER CAP)?$' && ! echo "$line" | grep -q 'score=\[none\]'
    }
    prism_env() { export LD_LIBRARY_PATH="$PRISM" CUDA_CACHE_MAXSIZE=4294967296; }
    if [ ! -s "$KLD_BASE" ]; then
      log "Qwen reference logits missing, regenerating (about 15 min)"
      ibench qwen_kldbase "$QWEN" "$STOCK" 25 kldbase 40 512
    fi
    # which llama.cpp loads Bonsai 2.5? first the PrismML fork used for Bonsai 2 PTQ1_0, then the stock iso3 build
    CHOICE="$RES/intel/bin_choice.txt"
    if [ ! -s "$CHOICE" ] || [ -n "${FORCE:-}" ]; then
      ( prism_env; ibench b25_ppl "$M25" "$PRISM" 99 ppl 40 8 ) && echo prism > "$CHOICE"
      if [ ! -s "$CHOICE" ]; then
        [ -f "$RES/intel/b25_ppl.log" ] && cp -f "$RES/intel/b25_ppl.log" "$RES/intel/b25_ppl_prism_failed.log"
        status "intel-prism-failed" "PrismML fork $PRISM cannot run Bonsai 2.5 (see intel/b25_ppl_prism_failed.log); trying the stock iso3 build"
        ( unset LD_LIBRARY_PATH; ibench b25_ppl "$M25" "$STOCK" 99 ppl 40 128 ) && echo stock > "$CHOICE"
      fi
    fi
    if [ -s "$CHOICE" ]; then
      if [ "$(cat "$CHOICE")" = prism ]; then B="$PRISM"; UBX=8; prism_env; else B="$STOCK"; UBX=128; unset LD_LIBRARY_PATH; fi
      status "intel-bin" "Bonsai 2.5 evaluated with $B"
      ibench b25_ppl "$M25" "$B" 99 ppl 40 "$UBX"
      ibench b25_kld "$M25" "$B" 99 kld 40 "$UBX"
      ibench b25_hellaswag "$M25" "$B" 99 hellaswag 100 "$UBX"
      ibench b25_winogrande "$M25" "$B" 99 winogrande 200 "$UBX"
      ibench b25_arc "$M25" "$B" 99 arc 100 "$UBX" || ibench b25_arc "$M25" "$B" 80 arc 100 "$UBX"
      ibench b25_mmlu "$M25" "$B" 99 mmlu 150 "$UBX" || ibench b25_mmlu "$M25" "$B" 80 mmlu 150 "$UBX"
    else
      status "intel-FAILED" "neither llama.cpp build loads Bonsai 2.5; the PrismML fork probably needs updating (intel/b25_ppl*.log)"
    fi
    if [ "${B2_INTEL_FILL:-1}" = 1 ] && [ -x "$PRISM/llama-perplexity" ]; then
      # Bonsai 2 PTQ1_0 had no ARC/MMLU run in the earlier suite: same subsets as Qwen/Bonsai 2.5 so every column compares like with like
      ( prism_env; ibench b2_ptq1_arc "$M2" "$PRISM" 99 arc 100 8; ibench b2_ptq1_mmlu "$M2" "$PRISM" 99 mmlu 150 8 )
    fi
  fi
  gpu_end

  # ------------------------------------------------------------------------------------------------------------ report
  if want report; then
    banner "report"
    python3 bench/bonsai25_vs_2_summary.py "$RES" > "$RES/REPORT.md" && status "report" "REPORT.md and figure written" || status "report-FAILED" "summarizer error"
    sed -n 1,60p "$RES/REPORT.md"
  fi

  # ------------------------------------------------------------------------------------------------------------ pr
  if want pr && [ "${SKIP_PR:-0}" != 1 ]; then
    banner "pr: commit, push, gh pr create"
    [ "$(git branch --show-current)" = "$BR" ] || die "not on $BR; refusing to commit"
    [ -s "$RES/REPORT.md" ] || die "no REPORT.md (run the report stage)"
    git add bench/bonsai25_vs_2.sh bench/bonsai25_vs_2_summary.py bench/results/bonsai25_vs_2 || die "git add failed"
    if git diff --cached --quiet; then log "nothing new to commit"
    else
      git commit -m "exp: Ternary Bonsai 2.5 27B vs Bonsai 2 on the GTX 1070 Ti (speed, VRAM, RAM, MTP, quality)" || die "git commit failed"
    fi
    git push -u origin "$BR" || die "git push failed"
    if gh pr view "$BR" --json url -q .url 2> /dev/null; then log "PR already exists"
    else
      gh pr create --base main --head "$BR" --title "exp: Ternary Bonsai 2.5 27B vs Bonsai 2 (speed, VRAM, RAM, MTP, quality)" --body-file "$RES/REPORT.md" || die "gh pr create failed (gh auth login?)"
    fi
  elif want pr; then log "SKIP_PR=1: not pushing"; fi

  log "finished. results: $RES"
}

main "$@"
exit $?
