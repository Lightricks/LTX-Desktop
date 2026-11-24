#!/usr/bin/env bash
# local-build.sh
# All-in-one local build script for creating the LTX Desktop installer.
# Prepares the Python environment, installs pnpm deps, builds the frontend,
# then packages with electron-builder via create-installer.sh.
#
# Usage:
#   bash scripts/local-build.sh [options]
#
# Options:
#   --platform mac|win   Target platform (auto-detected if omitted)
#   --skip-python        Use existing python-embed/ directory
#   --clean              Remove build artifacts before starting
#   --unpack             Build unpacked app only (faster, no installer/dmg)
#   --publish <mode>     Publish mode for electron-builder (always|never|onTag)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON_EMBED_DIR="$PROJECT_DIR/python-embed"
RELEASE_DIR="$PROJECT_DIR/release"

cat << 'BANNER'

  _   _______  __  ____            _    _
 | | |_   _\ \/ / |  _ \  ___  ___| | _| |_ ___  _ __
 | |   | |  \  /  | | | |/ _ \/ __| |/ / __/ _ \| '_ \
 | |___| |  /  \  | |_| |  __/\__ \   <| || (_) | |_) |
 |_____|_| /_/\_\ |____/ \___||___/_|\_\\__\___/| .__/
                                                 |_|
  Local Build Script

BANNER

# ============================================================
# Parse arguments
# ============================================================
SKIP_PYTHON=false
CLEAN=false
PLATFORM=""
PKG_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-python) SKIP_PYTHON=true ;;
    --clean)       CLEAN=true ;;
    --unpack)
      PKG_ARGS+=("$1")
      ;;
    --platform)
      PLATFORM="$2"
      PKG_ARGS+=("$1" "$2")
      shift
      ;;
    --publish)
      PKG_ARGS+=("$1" "$2")
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--platform mac|win] [--skip-python] [--clean] [--unpack] [--publish always|never|onTag]"
      exit 1
      ;;
  esac
  shift
done

# Auto-detect platform if not specified
if [ -z "$PLATFORM" ]; then
  case "$(uname -s)" in
    Darwin)          PLATFORM="mac" ;;
    MINGW*|MSYS*|CYGWIN*) PLATFORM="win" ;;
    Linux)           PLATFORM="linux" ;;
    *)               echo "ERROR: Could not detect platform. Use --platform mac|win"; exit 1 ;;
  esac
fi

cd "$PROJECT_DIR"

echo "Platform: $PLATFORM"
echo ""

# ============================================================
# Step 0: Clean if requested
# ============================================================
if [ "$CLEAN" = true ]; then
  echo "Cleaning previous build artifacts..."
  rm -rf "$PYTHON_EMBED_DIR" "$RELEASE_DIR" dist dist-electron
  echo "Clean complete."
  echo ""
fi

# ============================================================
# Step 1: Prepare Python environment
# ============================================================
if [ "$SKIP_PYTHON" = false ]; then
  echo "[1/3] Preparing Python environment..."

  if [ -d "$PYTHON_EMBED_DIR" ]; then
    echo "  Python environment already exists. Use --clean to rebuild."
  else
    case "$PLATFORM" in
      mac|linux)
        bash "$SCRIPT_DIR/prepare-python.sh"
        ;;
      win)
        powershell.exe -ExecutionPolicy Bypass -File "$SCRIPT_DIR/prepare-python.ps1"
        ;;
    esac
  fi
else
  echo "[1/3] Skipping Python preparation (using existing)..."
fi

if [ ! -d "$PYTHON_EMBED_DIR" ]; then
  echo "ERROR: Python environment not found at $PYTHON_EMBED_DIR"
  echo "Run without --skip-python to create it."
  exit 1
fi
echo ""

# ============================================================
# Step 2: Install pnpm dependencies
# ============================================================
echo "[2/3] Installing pnpm dependencies..."
pnpm install
echo ""

# ============================================================
# Step 3: Build frontend
# ============================================================
echo "[3/3] Building frontend and Electron app..."
pnpm run build:frontend
echo ""

# ============================================================
# Step 4: Create installer
# ============================================================
bash "$SCRIPT_DIR/create-installer.sh" "${PKG_ARGS[@]+"${PKG_ARGS[@]}"}"
