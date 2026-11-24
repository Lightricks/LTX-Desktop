#!/usr/bin/env python3
"""Cross-platform launcher for the perf dashboard (Windows / macOS / Linux).

Starts the dashboard control server and opens a browser (Chrome if found, else
the default browser); the clickable URL is always printed so you can open it
manually. Ctrl+C stops the server and, via its shutdown hook, terminates any
still-running spawned perf runs.

Run it with the backend env so fastapi/uvicorn/torch resolve, e.g.:
    uv run python run_dashboard.py
    uv run python run_dashboard.py --port 8750

Binds 127.0.0.1 only — never expose it.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import threading
import time
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent            # .../perf
DASH_DIR = HERE / "dashboard"


def _find_chrome() -> str | None:
    for name in ("google-chrome", "google-chrome-stable", "chrome", "chromium", "chromium-browser"):
        p = shutil.which(name)
        if p:
            return p
    if sys.platform == "darwin":
        candidates = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
    elif os.name == "nt":
        candidates = [
            os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
        ]
    else:
        candidates = []
    for c in candidates:
        if c and Path(c).exists():
            return c
    return None


def _open_browser(url: str) -> None:
    time.sleep(1.5)  # let uvicorn bind first
    chrome = _find_chrome()
    try:
        if chrome:
            webbrowser.register("chrome", None, webbrowser.BackgroundBrowser(chrome))
            webbrowser.get("chrome").open(url)
        else:
            webbrowser.open(url)
    except Exception:  # noqa: BLE001
        pass  # the URL is printed regardless


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PERF_DASH_PORT", "8750")))
    ap.add_argument("--no-browser", action="store_true", help="don't auto-open a browser")
    args = ap.parse_args()

    if not (DASH_DIR / "server.py").exists():
        sys.exit(f"dashboard server not found at {DASH_DIR / 'server.py'}")

    os.environ["PERF_DASH_PORT"] = str(args.port)
    url = f"http://127.0.0.1:{args.port}"

    try:
        sys.path.insert(0, str(DASH_DIR))
        import server as dashboard_server  # dashboard/server.py — defines `app`
        import uvicorn
    except ModuleNotFoundError as exc:
        sys.exit(
            f"missing dependency ({exc.name}). Launch with the backend env, e.g.:\n"
            f"    uv run python {Path(__file__).name}"
        )

    print("=" * 60)
    print(f"  LTX Perf Dashboard  ->  {url}")
    print("  localhost only · internal · Ctrl+C to stop")
    print("=" * 60, flush=True)

    if not args.no_browser:
        threading.Thread(target=_open_browser, args=(url,), daemon=True).start()

    uvicorn.run(dashboard_server.app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
