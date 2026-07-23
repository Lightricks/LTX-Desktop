#!/usr/bin/env bash
# prebuild-mps-sdpa-ext.sh
# Ahead-of-time compile mps-sdpa's zero-copy `mpsgraph_zc` attention extension so
# it can be BUNDLED and loaded on end-user Macs that have no compiler.
#
# Why: mps-sdpa JIT-compiles this extension on first import via torch's
# cpp_extension.load(), which needs Xcode Command Line Tools (clang + Metal SDK).
# End users don't have those, so mps-sdpa silently falls back to a Metal-memory-
# leaking backend that OOMs longer generations. A prebuilt cache loads in ~1.7s
# with only the `ninja` binary present (no clang). See
# docs/mps-attention-memory-leak.md.
#
# Build against the SHIPPED interpreter (python-embed) so the cache matches the
# bundled torch/python/arch exactly. Run AFTER prepare-python.sh, on a macOS
# Apple-Silicon build machine WITH Xcode Command Line Tools.
#
# Output: <output>/mps_sdpa_zc_ext/{mps_sdpa_zc_ext.so, mpsgraph_zc.o, build.ninja,...}
# prepare-python.sh calls this with <output> = python-embed/mps-ext-prebuilt so the
# cache rides the CI python-embed cache and is bundled to resources/python/mps-ext-prebuilt
# — where the runtime env LTX_MPS_EXT_PREBUILT_DIR points (see electron/python-backend.ts).

set -euo pipefail

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "prebuild-mps-sdpa-ext: skipping (Apple Silicon macOS only)"; exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Prefer the bundled interpreter so the cache matches what ships; fall back to the
# dev venv for local testing.
PY_EMBED="$PROJECT_DIR/python-embed/bin/python3"
PY_VENV="$PROJECT_DIR/backend/.venv/bin/python"
PYTHON="${PYTHON:-$([[ -x "$PY_EMBED" ]] && echo "$PY_EMBED" || echo "$PY_VENV")}"

OUTPUT_DIR="${1:-$PROJECT_DIR/python-embed/mps-ext-prebuilt}"
mkdir -p "$OUTPUT_DIR"
# Note: TORCH_EXTENSIONS_DIR is set to OUTPUT_DIR below, so the cache lands at
# OUTPUT_DIR/mps_sdpa_zc_ext — bundled with python-embed and staged at runtime.

# ninja must be importable/on PATH for torch's extension loader.
PY_BIN_DIR="$(dirname "$PYTHON")"
export PATH="$PY_BIN_DIR:$PATH"
export TORCH_EXTENSIONS_DIR="$OUTPUT_DIR"

echo "prebuild-mps-sdpa-ext: python=$PYTHON out=$OUTPUT_DIR"
"$PYTHON" - <<'PY'
import os, sys
from mps_sdpa._cpp import get_ext, load_error
ext = get_ext()
so = os.path.join(os.environ["TORCH_EXTENSIONS_DIR"], "mps_sdpa_zc_ext", "mps_sdpa_zc_ext.so")
if ext is None or not os.path.exists(so):
    print("BUILD FAILED:", load_error(), file=sys.stderr); sys.exit(1)
print("OK: prebuilt", so)
PY

echo "prebuild-mps-sdpa-ext: cache at $OUTPUT_DIR/mps_sdpa_zc_ext (bundled with python-embed)"
