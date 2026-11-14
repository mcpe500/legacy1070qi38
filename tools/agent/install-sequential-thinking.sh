#!/usr/bin/env bash
# Pasang Node.js 24 (folder home, checksum diverifikasi) dan server Sequential Thinking MCP (versi dikunci).
#
# - Tanpa sudo. Tidak mengubah sistem. Semuanya di ~/.local.
# - Tidak menulis ke ~/.kiro/settings (dilindungi Kiro). Untuk sesi interaktif, daftarkan sendiri dengan
#   `kiro-cli mcp add` (perintahnya dicetak di akhir). Agent GitHub /kiro memakai profil sendiri:
#   jalankan `python3 tools/agent/kiro_gh_agent.py install-agent --force` setelah skrip ini.
# - Aman dijalankan ulang (idempoten).
set -euo pipefail

NODE_VERSION="${NODE_VERSION:-v24.22.3}"
MCP_VERSION="${MCP_VERSION:-2.1.1}"
OPT="${HOME}/.local/opt"
MCP_DIR="${HOME}/.local/share/kiro-mcp/sequential-thinking"

case "$(uname -m)" in
  x86_64)  ARCH=x64 ;;
  aarch64) ARCH=arm64 ;;
  *) echo "arsitektur tidak didukung: $(uname -m)" >&2; exit 1 ;;
esac
[ "$(uname -s)" = "Linux" ] || { echo "skrip ini hanya untuk Linux" >&2; exit 1; }

NODE_DIR="${OPT}/node-${NODE_VERSION}-linux-${ARCH}"
if [ ! -x "${NODE_DIR}/bin/node" ]; then
  tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
  f="node-${NODE_VERSION}-linux-${ARCH}.tar.xz"
  echo "mengunduh ${f} ..."
  curl -fsSL -o "${tmp}/${f}" "https://nodejs.org/dist/${NODE_VERSION}/${f}"
  curl -fsSL -o "${tmp}/SHASUMS256.txt" "https://nodejs.org/dist/${NODE_VERSION}/SHASUMS256.txt"
  # Checksum dan berkas berasal dari server yang sama: melindungi dari unduhan rusak, bukan dari server yang dibobol.
  (cd "$tmp" && grep " ${f}\$" SHASUMS256.txt | sha256sum -c -)
  mkdir -p "$OPT"
  tar -xJf "${tmp}/${f}" -C "$OPT"
fi
ln -sfn "$NODE_DIR" "${OPT}/node-v24"
export PATH="${OPT}/node-v24/bin:${PATH}"
echo "node $(node --version), npm $(npm --version)"

mkdir -p "$MCP_DIR"
cd "$MCP_DIR"
[ -f package.json ] || npm init -y >/dev/null
# --ignore-scripts: tidak ada skrip paket yang dijalankan saat pemasangan.
npm install --save-exact --ignore-scripts --no-audit --no-fund \
  "@modelcontextprotocol/server-sequential-thinking@${MCP_VERSION}"

ENTRY="${MCP_DIR}/node_modules/@modelcontextprotocol/server-sequential-thinking/dist/index.js"
NODE_BIN="${OPT}/node-v24/bin/node"
[ -f "$ENTRY" ] || { echo "entry tidak ditemukan: $ENTRY" >&2; exit 1; }

# Uji asap lewat protokol MCP: initialize, daftar tool, satu panggilan.
NODE_BIN="$NODE_BIN" ENTRY="$ENTRY" python3 - <<'PY'
import json, os, subprocess, sys
p = subprocess.Popen([os.environ["NODE_BIN"], os.environ["ENTRY"]], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True, env={"PATH": "/usr/bin:/bin", "DISABLE_THOUGHT_LOGGING": "true"})
def rpc(o):
    p.stdin.write(json.dumps(o) + "\n"); p.stdin.flush()
    return json.loads(p.stdout.readline()) if "id" in o else None
rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {},
     "clientInfo": {"name": "install-check", "version": "0"}}})
rpc({"jsonrpc": "2.0", "method": "notifications/initialized"})
names = [t["name"] for t in rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]]
r = rpc({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "sequentialthinking", "arguments": {
    "thought": "uji", "nextThoughtNeeded": False, "thoughtNumber": 1, "totalThoughts": 1}}})
p.stdin.close(); p.wait(timeout=10)
ok = names == ["sequentialthinking"] and not r["result"].get("isError")
print("uji asap:", "LULUS" if ok else "GAGAL", names)
sys.exit(0 if ok else 1)
PY

cat <<EOF

Selesai. Berikutnya:

1) Agent GitHub /kiro (profil bot):
   python3 tools/agent/kiro_gh_agent.py install-agent --force

2) Sesi Kiro interaktif Anda (opsional; Kiro melarang agent menulis ke ~/.kiro/settings, jadi jalankan sendiri):
   kiro-cli mcp add --name sequential-thinking --scope global \\
     --command ${NODE_BIN} \\
     --args '["${ENTRY}"]' \\
     --env DISABLE_THOUGHT_LOGGING=true --force
EOF
