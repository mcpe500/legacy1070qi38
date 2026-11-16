#!/usr/bin/env bash
# scripts/setup.sh: verifikasi lingkungan dan laporkan statusnya (Tahap 0).
# Tanpa sudo. Tanpa MCP. Tidak menjalankan llama-server. Tidak menulis kode engine.
# Aman dijalankan berulang: hanya membaca sistem, probe di direktori sementara,
# dan mengganti docs/ref_pr38_*.
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 2
cd "$ROOT" || exit 2
mkdir -p docs
REPORT="docs/setup-report.md"
TMP="$(mktemp -d)" || exit 2
trap 'rm -rf "$TMP"' EXIT

PASS=0; FAIL=0; WARN=0
ok()   { PASS=$((PASS+1)); printf '  [OK]   %s\n' "$*"; }
bad()  { FAIL=$((FAIL+1)); printf '  [FAIL] %s\n' "$*"; }
warn() { WARN=$((WARN+1)); printf '  [WARN] %s\n' "$*"; }
info() { printf '  [info] %s\n' "$*"; }
sec()  { printf '\n## %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

# Semua keluaran tampil di layar dan disalin ke $REPORT.
exec > >(tee "$REPORT") 2>&1

printf '# Laporan setup.sh\n\n'
printf -- '- waktu (UTC): %s\n' "$(date -u +%FT%TZ)"
printf -- '- workspace: %s\n' "$ROOT"

# ---------------------------------------------------------------- 1. Sistem
sec "1. Sistem dan kunci perangkat keras"
info "$(uname -srm)"
info "CPU: $(lscpu | awk -F: '/Model name/{sub(/^ +/,"",$2); print $2; exit}')"
ARCH="$(gcc -march=native -Q --help=target 2>/dev/null | awk '/-march=/{print $2; exit}')"
if [ "$ARCH" = "znver1" ]; then ok "gcc -march=native = znver1 (sesuai kunci)"; else bad "gcc -march=native = '${ARCH:-kosong}', harus znver1"; fi
if grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -qx avx2; then ok "CPU punya AVX2"; else bad "CPU tanpa AVX2"; fi
if grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -qx fma; then ok "CPU punya FMA"; else bad "CPU tanpa FMA"; fi

NPHYS="$(lscpu -p=CORE,SOCKET --online | grep -v '^#' | sort -u | wc -l)"
PHYS="$(lscpu -p=CPU,CORE,SOCKET --online | grep -v '^#' | awk -F, '!seen[$2","$3]++{printf "%s%s", (n++?",":""), $1}')"
FIRST4="$(lscpu -p=CPU,CORE,SOCKET --online | grep -v '^#' | awk -F, '$1<=3{print $2","$3}' | sort -u | wc -l)"
info "core fisik: $NPHYS. Satu CPU logis per core: $PHYS"
if [ "$FIRST4" -eq 4 ]; then
  ok "taskset -c 0-3 memakai 4 core fisik yang berbeda"
else
  warn "taskset -c 0-3 hanya memakai $FIRST4 core fisik. Gunakan: taskset -c $PHYS"
fi

# ---------------------------------------------------------------- 2. Memori
sec "2. Memori"
MEM_TOTAL="$(free -m | awk '/^Mem:/{print $2}')"
MEM_AVAIL="$(free -m | awk '/^Mem:/{print $7}')"
info "RAM total ${MEM_TOTAL} MiB, tersedia ${MEM_AVAIL} MiB. Batas brief 14.2 GB = 13.2 GiB"
if [ "${MEM_TOTAL:-0}" -ge 15000 ]; then ok "RAM total sesuai 16 GB DDR4"; else bad "RAM total ${MEM_TOTAL} MiB di bawah 15000 MiB"; fi
swapon --show=NAME,SIZE,USED --noheadings 2>/dev/null | sed 's/^/  [info] swap: /'
ML="$(ulimit -l)"
if [ "$ML" = "unlimited" ]; then
  ok "memlock tanpa batas"
else
  GIB="$(awk -v k="$ML" 'BEGIN{printf "%.2f", k/1048576}')"
  warn "memlock ${ML} KiB (${GIB} GiB). mlock model penuh gagal tanpa root. Brief 'mmap + mlock' perlu jalur lain"
fi
info "vm.swappiness = $(cat /proc/sys/vm/swappiness 2>/dev/null)"

# ---------------------------------------------------------------- 3. GPU
sec "3. GPU"
if have nvidia-smi; then
  GPU_LINE="$(nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -1)"
  if [ -n "$GPU_LINE" ]; then
    info "GPU: $GPU_LINE"
    case "$GPU_LINE" in *"1070 Ti"*) ok "GTX 1070 Ti terdeteksi";; *) bad "GPU bukan GTX 1070 Ti";; esac
    case "$GPU_LINE" in *", 6.1,"*) ok "compute capability 6.1 (sm_61)";; *) bad "compute capability bukan 6.1";; esac
  else
    bad "nvidia-smi tidak bisa membaca GPU"
  fi
  APPS="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)"
  if [ -z "$APPS" ]; then
    ok "tidak ada proses GPU yang aktif"
  else
    warn "ada proses GPU aktif. Jangan jalankan engine bersamaan:"
    printf '%s\n' "$APPS" | sed 's/^/        /'
  fi
else
  bad "nvidia-smi tidak ada"
fi
if pgrep -x llama-server >/dev/null 2>&1; then
  warn "llama-server sedang berjalan (PID $(pgrep -x llama-server | head -1)). Jangan jalankan engine bersamaan"
else
  ok "tidak ada llama-server yang berjalan"
fi

# ---------------------------------------------------------------- 4. Toolchain CUDA
sec "4. Toolchain CUDA dan sm_61"
if have nvcc; then
  NVCC_VER="$(nvcc --version | grep -o 'release [0-9.]*' | awk '{print $2}')"
  ok "nvcc $NVCC_VER ($(command -v nvcc))"
  case "$NVCC_VER" in
    12.*) ok "CUDA 12.x masih menerima -arch=sm_61";;
    1[3-9].*) bad "CUDA $NVCC_VER: sm_61 kemungkinan tidak lagi didukung. Cek release notes sebelum dipakai";;
    *) warn "versi CUDA tidak dikenali: $NVCC_VER";;
  esac
  mkdir -p "$TMP/cuda" || exit 2
  cat > "$TMP/cuda/dp4a.cu" <<'EOF'
// Probe toolchain, bukan kode engine. __dp4a harus menghasilkan IDP.4A pada sm_61.
__global__ void probe_dp4a(const int* a, const int* b, int* c) {
    int i = threadIdx.x;
    c[i] = __dp4a(a[i], b[i], c[i]);
}
EOF
  cat > "$TMP/cuda/hfma2.cu" <<'EOF'
// Probe kontrol negatif, bukan kode engine. __hfma2 harus menghasilkan HFMA2 pada sm_61.
#include <cuda_fp16.h>
__global__ void probe_hfma2(const __half2* a, const __half2* b, __half2* c) {
    int i = threadIdx.x;
    c[i] = __hfma2(a[i], b[i], c[i]);
}
EOF
  if (cd "$TMP/cuda" && nvcc -arch=sm_61 -cubin -O3 -o dp4a.cubin dp4a.cu 2> dp4a.err && nvcc -arch=sm_61 -cubin -O3 -o hfma2.cubin hfma2.cu 2> hfma2.err); then
    ok "nvcc -arch=sm_61 -cubin berhasil untuk kedua probe"
    N_DP="$(cuobjdump -sass "$TMP/cuda/dp4a.cubin" 2>/dev/null | grep -c 'IDP\.4A')"
    N_HF="$(cuobjdump -sass "$TMP/cuda/hfma2.cubin" 2>/dev/null | grep -c 'HFMA2')"
    if [ "$N_DP" -gt 0 ]; then ok "kontrol positif: IDP.4A muncul $N_DP kali untuk __dp4a"; else bad "IDP.4A tidak muncul untuk __dp4a"; fi
    if [ "$N_HF" -gt 0 ]; then ok "kontrol negatif: HFMA2 muncul $N_HF kali untuk __hfma2 (grep terbukti sensitif)"; else bad "HFMA2 tidak muncul untuk kontrol negatif. Grep SASS tidak bisa dipercaya"; fi
  else
    bad "nvcc -arch=sm_61 gagal: $(head -c 240 "$TMP/cuda/dp4a.err" "$TMP/cuda/hfma2.err" 2>/dev/null | tr '\n' ' ')"
  fi
else
  bad "nvcc tidak ditemukan"
fi
for t in cuobjdump nvdisasm ncu; do
  if have "$t"; then ok "$t: $(command -v "$t")"; else bad "$t tidak ada di PATH"; fi
done

# ---------------------------------------------------------------- 5. Compiler dan build
sec "5. Compiler dan build"
cat > "$TMP/avx.cpp" <<'EOF'
// Probe toolchain, bukan kode engine. FMA 256-bit harus menjadi vfmadd dengan register ymm.
#include <immintrin.h>
extern "C" void probe_fma8(const float* x, const float* y, float* z) {
    _mm256_storeu_ps(z, _mm256_fmadd_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(y), _mm256_loadu_ps(z)));
}
EOF
check_cxx() {
  local cc="$1" tag
  tag="$(echo "$cc" | tr -d '+')"
  if "$cc" -O3 -march=znver1 -mavx2 -mfma -c "$TMP/avx.cpp" -o "$TMP/avx-$tag.o" 2> "$TMP/$tag.err"; then
    local nf ny
    nf="$(objdump -d "$TMP/avx-$tag.o" | grep -c 'vfmadd')"
    ny="$(objdump -d "$TMP/avx-$tag.o" | grep -c 'ymm')"
    if [ "$nf" -gt 0 ] && [ "$ny" -gt 0 ]; then
      ok "$cc -O3 -march=znver1 -mavx2 -mfma: vfmadd x$nf, register ymm x$ny"
    else
      bad "$cc: objek tidak memuat vfmadd atau ymm"
    fi
  else
    bad "$cc gagal: $(head -c 200 "$TMP/$tag.err" | tr '\n' ' ')"
  fi
}
if have g++; then ok "g++ $(g++ -dumpversion) ($(command -v g++))"; check_cxx g++; else bad "g++ tidak ada"; fi
if have clang++; then check_cxx clang++; else warn "clang++ tidak terpasang. Opsional. Build memakai g++"; fi
if have cmake; then ok "cmake $(cmake --version | head -1 | awk '{print $3}')"; else bad "cmake tidak ada"; fi
if have ninja; then ok "ninja $(ninja --version). Kebijakan: selalu ninja -j4"; else bad "ninja tidak ada"; fi
ok "python3 $(python3 --version 2>&1 | awk '{print $2}')"

# ---------------------------------------------------------------- 6. CLI pendukung
sec "6. CLI pendukung dan headless browser"
check_tool() {
  local t="$1" level="$2" hint="$3"
  if have "$t"; then
    ok "$t: $(command -v "$t")"
  elif [ "$level" = req ]; then
    bad "$t tidak ada. $hint"
  else
    warn "$t tidak ada. $hint"
  fi
}
check_tool taskset req "Paket util-linux"
if [ -x /usr/bin/time ]; then ok "/usr/bin/time (GNU time) ada"; else warn "/usr/bin/time tidak ada. Pasang paket time: sudo apt install time"; fi
check_tool jq req "sudo apt install jq"
check_tool curl req "sudo apt install curl"
check_tool gh req "https://cli.github.com"
check_tool git req "sudo apt install git"
check_tool hyperfine opt "Dibutuhkan Tahap 5. Pasang: sudo apt install hyperfine, atau biner resmi dari github.com/sharkdp/hyperfine ke ~/.local/bin"
check_tool rg opt "Dibutuhkan untuk pencarian. Pasang: sudo apt install ripgrep"
check_tool node opt "Dibutuhkan untuk Playwright. Pasang Node LTS (nvm tanpa sudo, atau paket resmi)"
check_tool npx opt "Ikut Node. Playwright dijalankan one-shot dengan npx setelah Node ada"
HB=""
for b in chromium chromium-browser google-chrome; do have "$b" && HB="$b" && break; done
if [ -n "$HB" ]; then ok "headless browser: $HB"; else warn "tidak ada chromium/google-chrome. Tes UI memakai Playwright setelah Node ada, atau pasang chromium (tanpa snap jika bisa)"; fi

# ---------------------------------------------------------------- 7. Referensi PR #38
sec "7. Referensi PR #38 (iso3 + MTP)"
PLACEHOLDER_HEAD="# PLACEHOLDER"
write_placeholder() {
  cat > "$1" <<EOF
$PLACEHOLDER_HEAD: referensi belum tersedia

Alasan: $2

Unduh ulang saat gh sudah login dan online:

    gh pr view 38 --repo mcpe500/legacy1070qi38 > docs/ref_pr38_summary.md
    gh pr diff 38 --repo mcpe500/legacy1070qi38 > docs/ref_pr38_iso3_mtp.diff

Jangan dipakai sebagai referensi implementasi sebelum diganti dengan isi asli.
EOF
}
is_real() { [ -s "$1" ] && ! head -1 "$1" | grep -q "^$PLACEHOLDER_HEAD"; }

if have gh && gh auth status >/dev/null 2>&1; then
  ok "gh login: $(gh api user -q .login 2>/dev/null)"
  # Perintah persis dari brief. Di gh versi ini 'pr view' gagal karena field Projects (classic),
  # jadi jika gagal, ringkasan dibuat dari --json yang tidak meminta field itu.
  if gh pr view 38 --repo mcpe500/legacy1070qi38 > "$TMP/summary.md" 2> "$TMP/view.err" && [ -s "$TMP/summary.md" ]; then
    mv "$TMP/summary.md" docs/ref_pr38_summary.md
    ok "docs/ref_pr38_summary.md: gh pr view berhasil"
  elif gh pr view 38 --repo mcpe500/legacy1070qi38 --json number,title,state,mergedAt,headRefName,baseRefName,url,body > "$TMP/pr.json" 2>/dev/null; then
    python3 - "$TMP/pr.json" > docs/ref_pr38_summary.md <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"# PR #{d['number']}: {d['title']}\n")
print(f"- state: {d['state']}")
print(f"- merged: {d.get('mergedAt')}")
print(f"- head: {d['headRefName']}")
print(f"- base: {d['baseRefName']}")
print(f"- url: {d['url']}\n")
print("## Body\n")
print(d.get('body') or '')
PY
    warn "gh pr view polos gagal (field Projects classic di GraphQL). Ringkasan dibuat dari gh pr view --json"
  else
    if is_real docs/ref_pr38_summary.md; then
      warn "gagal mengambil ringkasan PR #38. Memakai salinan lokal yang sudah ada"
    else
      write_placeholder docs/ref_pr38_summary.md "gh pr view gagal"
      warn "gagal mengambil ringkasan PR #38. Placeholder ditulis"
    fi
  fi
  if gh pr diff 38 --repo mcpe500/legacy1070qi38 > "$TMP/diff.patch" 2> "$TMP/diff.err" && [ -s "$TMP/diff.patch" ]; then
    mv "$TMP/diff.patch" docs/ref_pr38_iso3_mtp.diff
    ok "docs/ref_pr38_iso3_mtp.diff: $(wc -l < docs/ref_pr38_iso3_mtp.diff) baris"
  else
    if is_real docs/ref_pr38_iso3_mtp.diff; then
      warn "gagal mengambil diff PR #38. Memakai salinan lokal yang sudah ada"
    else
      write_placeholder docs/ref_pr38_iso3_mtp.diff "gh pr diff gagal: $(head -c 160 "$TMP/diff.err" | tr '\n' ' ')"
      warn "gagal mengambil diff PR #38. Placeholder ditulis"
    fi
  fi
else
  warn "gh belum login atau offline. Membuat placeholder, skrip tetap lanjut"
  is_real docs/ref_pr38_summary.md || write_placeholder docs/ref_pr38_summary.md "gh tidak login atau offline"
  is_real docs/ref_pr38_iso3_mtp.diff || write_placeholder docs/ref_pr38_iso3_mtp.diff "gh tidak login atau offline"
fi
if is_real docs/ref_pr38_iso3_mtp.diff; then
  N_ISO="$(grep -c 'GGML_TYPE_ISO3_0' docs/ref_pr38_iso3_mtp.diff)"
  if [ "$N_ISO" -gt 0 ]; then ok "diff PR #38 memuat GGML_TYPE_ISO3_0 ($N_ISO kali)"; else bad "diff PR #38 tidak memuat GGML_TYPE_ISO3_0"; fi
  info "status PR #38: $(gh pr view 38 --repo mcpe500/legacy1070qi38 --json state,mergedAt -q '.state + " merged=" + (.mergedAt // "-")' 2>/dev/null || echo tidak diketahui)"
else
  bad "referensi PR #38 belum tersedia (placeholder). Lihat warning di atas"
fi

# ---------------------------------------------------------------- 8. Kebijakan
sec "8. Kebijakan: tanpa MCP, tanpa kode engine"
if [ -e .mcp.json ] || [ -e .kiro/settings/mcp.json ]; then
  bad "ada konfigurasi MCP di workspace ini. Hapus. Workspace ini tanpa MCP"
else
  ok "tidak ada konfigurasi MCP di workspace ini"
fi
ENG="$(find src -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.hpp' \) 2>/dev/null | wc -l)"
if [ "$ENG" -eq 0 ]; then ok "src/ belum berisi kode sumber (tahap scaffold)"; else warn "src/ berisi $ENG berkas sumber. Tahap ini hanya scaffold"; fi

# ---------------------------------------------------------------- 9. Skill dan symlink
sec "9. Skill dan symlink"
SK_OK=0
for f in .agents/skills/*/SKILL.md; do
  [ -f "$f" ] || continue
  if python3 - "$f" <<'PY'
import re, sys
path = sys.argv[1]
text = open(path, encoding="utf-8").read()
m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
if not m:
    sys.exit(1)
fm = m.group(1)
try:
    import yaml
    meta = yaml.safe_load(fm)
except ImportError:
    meta = {}
    for key in ("name", "description"):
        hit = re.search(r"^" + key + r":\s*(.+)$", fm, re.M)
        if hit:
            meta[key] = hit.group(1).strip().strip('"')
except Exception:
    sys.exit(2)
if not isinstance(meta, dict) or not meta.get("name") or not meta.get("description"):
    sys.exit(3)
folder = path.split("/")[-2]
if str(meta["name"]).strip() != folder:
    sys.exit(4)
PY
  then
    ok "$f: frontmatter name dan description valid"
    SK_OK=$((SK_OK+1))
  else
    bad "$f: frontmatter tidak valid (cek exit code python: 1 tanpa frontmatter, 2 YAML rusak, 3 field kosong, 4 name beda dengan folder)"
  fi
done
if [ "$SK_OK" -eq 5 ]; then ok "5 dari 5 SKILL.md valid"; else bad "hanya $SK_OK dari 5 SKILL.md yang valid"; fi
check_link() {
  if [ -L "$1" ] && [ "$(readlink "$1")" = "$2" ] && [ -e "$1" ]; then
    ok "$1 -> $2"
  else
    bad "$1 harus symlink relatif ke $2"
  fi
}
check_link .opencode/skills ../.agents/skills
check_link .claude/skills ../.agents/skills
check_link CLAUDE.md AGENTS.md

# ---------------------------------------------------------------- 10. Model dan binary
sec "10. Model dan binary (untuk Tahap 1)"
GGUF="$HOME/models/27b/Qwen3.8-27B-UD-Q4_K_M.gguf"
if [ -s "$GGUF" ]; then ok "model 27B: $GGUF ($(du -h "$GGUF" | cut -f1))"; else bad "model 27B tidak ditemukan di $GGUF"; fi
for f in "$HOME"/models/27b/Qwen3.8-27B-*.gguf; do
  [ -s "$f" ] && info "varian lokal: $(basename "$f") ($(du -h "$f" | cut -f1))"
done
LB="$HOME/llama.cpp/build-iso3/bin"
for b in llama-server llama-bench llama-perplexity; do
  if [ -x "$LB/$b" ]; then ok "$LB/$b"; else warn "$b tidak ada di $LB"; fi
done
SUB="$HOME/kiroween-2026/legacy1070qi38/engine/llama.cpp"
if [ -d "$SUB" ]; then info "submodule legacy engine/llama.cpp: $(git -C "$SUB" rev-parse --short HEAD 2>/dev/null)"; else warn "submodule legacy engine/llama.cpp tidak ditemukan"; fi

# ---------------------------------------------------------------- Ringkasan
sec "Ringkasan"
printf '  OK=%d  WARN=%d  FAIL=%d\n' "$PASS" "$WARN" "$FAIL"
printf '  laporan: %s\n' "$REPORT"
[ "$FAIL" -eq 0 ]
