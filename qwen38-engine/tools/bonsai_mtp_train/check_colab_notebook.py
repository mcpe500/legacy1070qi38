#!/usr/bin/env python3
"""Static check of train_bonsai_mtp_colab.ipynb, CPU only (no GPU, no network, no Colab):

  1. the file is valid JSON and a valid nbformat v4 notebook (nbformat.validate when nbformat is installed);
  2. every code cell parses as Python (no shell magics), every `#@param` line is a plain assignment;
  3. every name a cell reads is defined by an earlier cell, by the same cell, or is a builtin;
  4. every CLI call of a script in this folder (a list literal that contains ".../<script>.py") uses only options that the
     script's argparse knows (from `python3 <script> --help`), and constant values of options with choices are allowed;
     lists assigned to INIT_ARGS are checked against train_bonsai_mtp.py;
  5. the four scripts compile (py_compile) and their --help exits with code 0.
Exit code 0 only when everything passes.  Usage: python3 check_colab_notebook.py [NOTEBOOK]"""
import ast
import builtins
import json
import os
import py_compile
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["convert_gguf_to_hf_bf16.py", "prepare_corpus.py", "train_bonsai_mtp.py", "inject_mtp_gguf.py"]


def script_options(script):
    r = subprocess.run([sys.executable, os.path.join(HERE, script), "--help"], capture_output=True, text=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    if r.returncode != 0:
        return None, None, r.stderr[-500:]
    usage = r.stdout.split("\n\n", 1)[0] + "\n" + r.stdout   # usage block + option list
    opts = set(re.findall(r"(?<![\w-])(--[a-z0-9][a-z0-9-]*)", usage))
    choices = {m.group(1): set(m.group(2).split(",")) for m in re.finditer(r"(--[a-z0-9-]+) \{([^}]+)\}", r.stdout)}
    return opts, choices, None


def const_str(node):
    """The string of a Constant, or the literal tail of an f-string (f"{TOOLS}/x.py" -> "/x.py")."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str))
    return None


def defined_names(tree):
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            names.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
            for a in n.args.args + n.args.kwonlyargs + n.args.posonlyargs:
                names.add(a.arg)
            for a in (n.args.vararg, n.args.kwarg):
                if a:
                    names.add(a.arg)
        elif isinstance(n, ast.Lambda):
            for a in n.args.args:
                names.add(a.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            names.add(n.name)
    return names


def main():
    nb_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "train_bonsai_mtp_colab.ipynb")
    fails = []
    raw = open(nb_path, encoding="utf-8").read()
    try:
        nb = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[FAIL] JSON: {e}")
        return 1
    print(f"[PASS] JSON: {len(nb['cells'])} cells, nbformat {nb.get('nbformat')}.{nb.get('nbformat_minor')}")
    try:
        import nbformat
        nbformat.validate(nbformat.reads(raw, as_version=4))
        print(f"[PASS] nbformat {nbformat.__version__} schema validation")
    except ImportError:
        print("[SKIP] nbformat not installed: schema validation skipped")
    except Exception as e:   # nbformat.ValidationError
        fails.append(f"nbformat: {e}")
        print(f"[FAIL] nbformat: {e}")

    # scripts
    opts, choices = {}, {}
    for s in SCRIPTS:
        try:
            py_compile.compile(os.path.join(HERE, s), doraise=True)
        except py_compile.PyCompileError as e:
            fails.append(f"{s}: {e}")
            print(f"[FAIL] py_compile {s}: {e}")
            continue
        o, c, err = script_options(s)
        if o is None:
            fails.append(f"{s} --help: {err}")
            print(f"[FAIL] {s} --help: {err}")
            continue
        opts[s], choices[s] = o, c
        print(f"[PASS] {s}: compiles, --help OK ({len(o)} options)")

    known = set(dir(builtins)) | {"__file__", "__name__"}
    used_flags = {s: set() for s in SCRIPTS}
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    for i, cell in enumerate(code_cells, 1):
        src = "".join(cell["source"])
        bad = [l for l in src.splitlines() if l.lstrip().startswith(("!", "%"))]
        if bad:
            fails.append(f"cell {i}: shell magic {bad[0]!r}")
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            fails.append(f"cell {i}: {e}")
            print(f"[FAIL] cell {i}: syntax {e}")
            continue
        for l in src.splitlines():
            if "#@param" in l and not re.match(r"^[A-Z_][A-Z0-9_]* = .+#@param", l):
                fails.append(f"cell {i}: malformed #@param line {l!r}")
        own = defined_names(tree)
        undefined = sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
                           - own - known)
        if undefined:
            fails.append(f"cell {i}: undefined names {undefined}")
        known |= own
        # CLI calls
        init_lists = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "INIT_ARGS" for t in n.targets) \
                    and isinstance(n.value, ast.List):
                init_lists.add(id(n.value))
        n_calls = 0
        for n in ast.walk(tree):
            if not isinstance(n, ast.List):
                continue
            strs = [const_str(e) for e in n.elts]
            script = next((s for s in SCRIPTS for v in strs if v and v.endswith("/" + s)), None)
            if script is None and id(n) in init_lists:
                script = "train_bonsai_mtp.py"
            if script is None or script not in opts:
                continue
            n_calls += 1
            for k, v in enumerate(strs):
                if not (v and v.startswith("--")) or v.endswith(".py"):
                    continue
                used_flags[script].add(v)
                if v not in opts[script]:
                    fails.append(f"cell {i}: {script} has no option {v}")
                    print(f"[FAIL] cell {i}: {script} {v} is not an option of the script")
                nxt = strs[k + 1] if k + 1 < len(strs) else None
                if v in choices[script] and nxt is not None and isinstance(n.elts[k + 1], ast.Constant) \
                        and nxt not in choices[script][v]:
                    fails.append(f"cell {i}: {script} {v} {nxt!r} not in {sorted(choices[script][v])}")
        print(f"[{'PASS' if not [f for f in fails if f.startswith(f'cell {i}:')] else 'FAIL'}] code cell {i}: parses, "
              f"names defined, {n_calls} script call(s) checked")
    for s, fl in used_flags.items():
        if fl:
            print(f"       {s}: {' '.join(sorted(fl))}")
    # the Colab link in the README / notebook points at a file that exists in the repo
    rel = "qwen38-engine/tools/bonsai_mtp_train/train_bonsai_mtp_colab.ipynb"
    if not os.path.exists(os.path.join(HERE, "..", "..", "..", rel)):
        fails.append(f"{rel} not found relative to the repository root")
    for f in fails:
        print(f"  - {f}")
    print("check_colab_notebook:", "PASS" if not fails else f"FAIL ({len(fails)} problems)")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
