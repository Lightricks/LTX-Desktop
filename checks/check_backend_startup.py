r"""Start the actual LTX server from a fresh profile and require a healthy HTTP response.

From a source checkout in Windows PowerShell:
  & "$env:LOCALAPPDATA\Programs\LTXDesktopAMD\application\resources\python\python.exe" .\checks\check_backend_startup.py

A build host with the backend dependencies may run: python checks/check_backend_startup.py --cpu-host
The default checks real ROCm execution during startup. No model downloads or account are required.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener


def run_server(path):
    # Embedded Windows Python uses an isolated ._pth file. Match the Electron
    # launcher, which explicitly adds the packaged backend to module search.
    if os.name == "nt":
        import ctypes
        ctypes.WinDLL("kernel32", use_last_error=True).SetDllDirectoryW("")
    sys.path.insert(0, str(path.parent))
    runpy.run_path(str(path), run_name="__main__")


def check(backend, cpu_host, timeout):
    entry = (backend / "ltx2_server.py").resolve(strict=True)
    token = secrets.token_urlsafe(32)
    with tempfile.TemporaryDirectory(prefix="ltx-backend-check-") as scratch:
        log = Path(scratch) / "startup.log"
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        env = {
            **os.environ,
            "LTX_APP_DATA_DIR": scratch,
            "LTX_ASSETS_DIR": "",
            "LTX_PORT": str(port),
            "LTX_REMOTE_PORT": "0",
            "LTX_AUTH_TOKEN": token,
            "LTX_ADMIN_TOKEN": secrets.token_urlsafe(32),
            "LTX_REQUIRE_ROCM": "0" if cpu_host else "1",
            "LTX_DEV_MODE": "0",
            "HF_HUB_OFFLINE": "1",
        }
        if cpu_host:
            env.update(CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="")
        url = f"http://127.0.0.1:{port}/health"
        request = Request(url, headers={"Authorization": f"Bearer {token}"})
        opener = build_opener(ProxyHandler({}))
        with log.open("wb") as stream:
            process = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()), "--serve", str(entry)],
                cwd=entry.parent, env=env, stdout=stream, stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline and process.poll() is None:
                    try:
                        with opener.open(request, timeout=2) as response:
                            if response.status == 200 and json.load(response).get("status") == "ok":
                                target = "CPU host" if cpu_host else "ROCm"
                                print(f"PASS: {target} backend started with a fresh profile and returned HTTP 200 ok.")
                                return
                    except (URLError, TimeoutError):
                        pass
                    time.sleep(0.5)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        raise RuntimeError("Backend failed to become healthy. Startup log:\n" + log.read_text(encoding="utf-8", errors="replace")[-16000:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--backend", type=Path, help="Backend directory to verify (defaults to packaged backend)")
    parser.add_argument("--cpu-host", action="store_true", help="Run on a build host without requiring a Radeon")
    parser.add_argument("--timeout", type=int, default=240, help="Maximum seconds to wait for startup (default 240)")
    args = parser.parse_args()
    if args.serve:
        run_server(args.serve)
        return
    packaged = Path(sys.executable).resolve().parent.parent / "backend"
    checkout = Path(__file__).resolve().parents[1] / "app/backend"
    check(args.backend or (packaged if packaged.is_dir() else checkout), args.cpu_host, args.timeout)


if __name__ == "__main__":
    main()
