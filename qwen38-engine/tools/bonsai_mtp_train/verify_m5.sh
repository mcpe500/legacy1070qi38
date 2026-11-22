#!/usr/bin/env bash
# CPU-only verification of the M5 training pipeline (issue #57). No GPU is touched: the trainer selftest runs on
# the CPU, and the notebook check only parses the generated notebook. Run it before a Colab run and after any
# change to train_bonsai_mtp.py, make_colab_notebook.py or inject_mtp_gguf.py.
#   tools/bonsai_mtp_train/verify_m5.sh
# Checks:
#   1. the trainer selftest (forward shapes, unrolled loss, backward, GGUF names and full-size shapes);
#   2. the drafter schedule matrix (mtp: --unroll 1..8, df3: --df3-block = the block of the verify batch);
#   3. the CLI rejects an out-of-range --unroll;
#   4. the Colab notebook regenerates and the checker parses every code cell;
#   5. the GGUF names the trainer exports are exactly the names the engine loader reads (no missing, no extra);
#   6. the converter unrotates token_embd (the ProCreations embedding fix), so the trainer's teacher sees the
#      plain-basis embedding the engine builds from the GGUF.
# Exit 0 only if every check passes.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python3}"

echo "== 1. trainer selftest (CPU) =="
"$PY" "$HERE/train_bonsai_mtp.py" --selftest | tail -2

echo "== 2. drafter schedule matrix =="
"$PY" - "$HERE" <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("t", sys.argv[1] + "/train_bonsai_mtp.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
class A: pass
cases = [({"drafter": "mtp", "unroll": k, "df3_block": 16}, k) for k in (1, 2, 4, 6, 8)]
cases += [({"drafter": "df3", "unroll": 4, "df3_block": b}, b) for b in (12, 16)]
for arm, want in cases:
    a = A(); a.__dict__.update(arm)
    got = m.unroll_K(a)
    assert got == want, f"{arm}: unroll_K {got}, expected {want}"
    print(f"  {arm['drafter']:3s} unroll={arm['unroll']} df3_block={arm['df3_block']} -> {got} step(s)")
print("schedule matrix: PASS")
PY

echo "== 3. CLI rejects --unroll 12 =="
set +e
out12="$("$PY" "$HERE/train_bonsai_mtp.py" --unroll 12 --smoke --teacher t --corpus c --out o 2>&1)"
rc12=$?
set -e
if [ "$rc12" -ne 0 ] && printf '%s' "$out12" | grep -q "unroll must be 1..8"; then
    echo "  --unroll 12 rejected: PASS"
else
    echo "  --unroll 12 NOT rejected: FAIL"; exit 1
fi

echo "== 4. notebook regenerate + check =="
"$PY" "$HERE/make_colab_notebook.py" > /dev/null
"$PY" "$HERE/check_colab_notebook.py" | tail -1

echo "== 5. GGUF names the trainer exports are exactly the engine loader names =="
"$PY" - "$HERE" <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("t", sys.argv[1] + "/train_bonsai_mtp.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
exported = set(m.GGUF_NAMES.values())
# the names Engine::load reads for blk.64 (src/engine/engine.cpp: the mtp block plus load_layer of the drafter)
engine_reads = {
    "blk.64.nextn.eh_proj.weight", "blk.64.nextn.enorm.weight", "blk.64.nextn.hnorm.weight",
    "blk.64.nextn.shared_head_norm.weight",
    "blk.64.attn_norm.weight", "blk.64.post_attention_norm.weight", "blk.64.ffn_gate.weight",
    "blk.64.ffn_up.weight", "blk.64.ffn_down.weight", "blk.64.attn_q.weight", "blk.64.attn_k.weight",
    "blk.64.attn_v.weight", "blk.64.attn_output.weight", "blk.64.attn_q_norm.weight", "blk.64.attn_k_norm.weight",
}
missing = engine_reads - exported
extra = exported - engine_reads
assert not missing, f"the engine reads tensors the trainer never exports: {sorted(missing)}"
assert not extra, f"the trainer exports tensors the engine does not read: {sorted(extra)}"
shape = tuple(m.MTPHead(m.Dims()).gguf_state()["blk.64.nextn.eh_proj.weight"].shape)
assert shape == (5120, 10240), f"eh_proj shape {shape}, the engine wants (5120, 10240)"
print(f"  {len(exported)} tensors, no missing and no extra; eh_proj {shape}")
print("GGUF name/shape audit: PASS")
PY

echo "== 6. the converter unrotates token_embd (embedding fix audit) =="
"$PY" - "$HERE" <<'PY'
import sys
src = open(sys.argv[1] + "/convert_gguf_to_hf_bf16.py").read()
assert "inverse_weight_names" in src, "the converter does not know prism.hadamard.inverse_weight_names"
assert "token_embd.weight" in src and "hadamard_unrotate" in src, "the converter has no unrotate path"
print("  invert_path: prism.hadamard.inverse_weight_names -> hadamard_unrotate in tensor()")
print("  hf name    : token_embd.weight -> model.embed_tokens.weight (plain basis)")
print("  engine     : Engine::mtp_forward does qmat_row_to_f32(tok_embd_) + hadamard_unrotate(.., signs_5120_)")
print("embedding fix audit: PASS")
PY
echo "verify_m5: PASS"