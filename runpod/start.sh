#!/usr/bin/env bash
set -euo pipefail

export LTX_APP_DATA_DIR="${LTX_APP_DATA_DIR:-/workspace/ltx-data}"
export LTX_PORT="${LTX_PORT:-8000}"

mkdir -p "${LTX_APP_DATA_DIR}"

if [[ -z "${RUNPOD_PRIVATE_API_TOKEN:-${LTX_AUTH_TOKEN:-}}" ]]; then
  echo "RUNPOD_PRIVATE_API_TOKEN or LTX_AUTH_TOKEN must be set." >&2
  exit 1
fi

exec uv run python runpod_server.py
