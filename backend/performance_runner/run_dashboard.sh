#!/usr/bin/env bash
# Launch the perf dashboard (macOS / Linux). Opens a browser + prints the URL.
# Uses uv if present so fastapi/uvicorn/torch resolve from the backend env.
set -euo pipefail
cd "$(dirname "$0")"
if command -v uv >/dev/null 2>&1; then
  exec uv run python run_dashboard.py "$@"
else
  exec python3 run_dashboard.py "$@"
fi
