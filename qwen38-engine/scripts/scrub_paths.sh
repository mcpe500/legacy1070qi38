#!/usr/bin/env bash
# Replaces the absolute home directory in text artefacts (logs, JSON, Markdown) with "~" before they are committed,
# so no user name ends up in the repository. Usage: scripts/scrub_paths.sh [DIR...]   (default: bench docs tests)
set -eu
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
H="$HOME"
[ "$#" -gt 0 ] || set -- bench docs tests .agents AGENTS.md
grep -rIl --exclude-dir=third_party -- "$H" "$@" 2>/dev/null | while read -r f; do
  sed -i "s#${H}#~#g" "$f"
  echo "scrubbed $f"
done
