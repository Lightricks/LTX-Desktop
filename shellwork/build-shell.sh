#!/usr/bin/env bash
# Build the patched LTX UI and assemble the Windows x64 Electron shell on Linux.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
app_dir="$(cd "${1:-$here/../app}" && pwd)"
mkdir -p "${LTX_SHELL_WORK:-$here}"
export LTX_SHELL_WORK="$(cd "${LTX_SHELL_WORK:-$here}" && pwd)"
export ELECTRON_SKIP_BINARY_DOWNLOAD=1
export GOMAXPROCS="${GOMAXPROCS:-2}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-2}"
export UV_THREADPOOL_SIZE="${UV_THREADPOOL_SIZE:-2}"

cd "$app_dir"
corepack pnpm install --frozen-lockfile
corepack pnpm exec vite build
corepack pnpm exec vite build --config vite.remote.config.ts

if [[ "$LTX_SHELL_WORK" != "$here" ]]; then
  mkdir -p "$LTX_SHELL_WORK/asar-tool"
  cp "$here/asar-tool/"package{,-lock}.json "$here/asar-tool/brand-launcher.mjs" "$LTX_SHELL_WORK/asar-tool/"
fi
npm ci --prefix "$LTX_SHELL_WORK/asar-tool" --ignore-scripts --no-audit --no-fund
cd "$LTX_SHELL_WORK"

fetch_archive() {
  local url="$1" filename="$2" checksum="$3" hash_command="$4"
  if [[ -f "$filename" ]] && printf '%s  %s\n' "$checksum" "$filename" | "$hash_command" -c --status; then
    return
  fi
  curl --fail --show-error --silent --location --retry 3 --output "$filename.part" "$url"
  printf '%s  %s\n' "$checksum" "$filename.part" | "$hash_command" -c
  mv -f "$filename.part" "$filename"
}
fetch_archive \
  'https://github.com/electron/electron/releases/download/v41.0.2/electron-v41.0.2-win32-x64.zip' \
  electron.zip dcd36396a606a5ae2f5651b4ee6bb463a624dbf15f786eda57cee2cc361c138c sha256sum
fetch_archive \
  'https://registry.npmjs.org/@koromix/koffi-win32-x64/-/koffi-win32-x64-3.1.4.tgz' \
  koffi-win32-x64.tgz 04f7905dcd5b45dd1004e925bec3fe023a119d4cca6cf344eab7f1ed5371ade6e1874f4c2a577689bb2d80a5a7e9e8d02c471f284a1427e580588857e003ac22 sha512sum
python3 "$here/assemble.py" "$app_dir"
