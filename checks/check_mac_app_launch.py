"""Launch an installed community Mac app and verify its renderer, preload and local backend."""

import argparse
import os
from pathlib import Path
import platform
import plistlib
import signal
import socket
import subprocess
import tempfile
import time


def check(app: Path, timeout: int = 240) -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("This check needs a native Apple silicon macOS session.")
    contents = app.resolve(strict=True) / "Contents"
    with (contents / "Info.plist").open("rb") as stream:
        executable = contents / "MacOS" / plistlib.load(stream)["CFBundleExecutable"]
    subprocess.run(["/usr/bin/codesign", "--verify", "--deep", "--strict", str(contents.parent)], check=True)

    with tempfile.TemporaryDirectory(prefix="ltx-mac-launch-") as directory:
        home = Path(directory)
        profile = home / "Library/Application Support/LTXDesktopSpaceSpeak"
        # Give the app a clean profile, with model/network providers offline.
        # Its renderer requests backend startup and reports success only after
        # Electron's authenticated HTTP health probe has returned successfully.
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        env = dict(os.environ, HOME=directory, LTX_PORT=str(port), LTX_REMOTE_PORT="0",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        log = home / "electron-output.log"
        with log.open("wb") as stream:
            proc = subprocess.Popen([str(executable)], cwd=directory, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline and proc.poll() is None:
                    for session in (profile / "logs").glob("session_*.log"):
                        if "[Renderer] Python backend started successfully" in session.read_text(errors="replace"):
                            print("Community Electron renderer and packaged backend completed authenticated startup.")
                            return
                    time.sleep(1)
                raise RuntimeError("The native desktop did not report a successful backend start.\n" +
                                   log.read_text(errors="replace")[-8000:])
            finally:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=10)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app", type=Path)
    check(parser.parse_args().app)
