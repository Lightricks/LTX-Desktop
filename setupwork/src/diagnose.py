"""Exercise the same AMD startup path as the desktop app, using temporary app data."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback


def probe(application):
    backend = application / "resources/backend"
    sys.path.insert(0, str(backend))
    # Match the Electron launcher and let the backend perform its actual GPU checks.
    print("Starting the backend's Radeon GPU check and importing the video pipeline...", flush=True)
    import ltx2_server
    from transformers import PreTrainedModel, ProcessorMixin
    from ltx_core.text_encoders.gemma import LTXGemmaTextEncoder
    import ltx_pipelines.ic_lora
    import diffusers.hooks.context_parallel
    import diffusers.models._modeling_parallel
    import av
    import torchaudio
    import torchvision

    if not callable(ltx2_server.app):
        raise RuntimeError("The backend did not create its FastAPI application.")
    print("PASS: Radeon GPU checks, Transformers, Gemma, Diffusers, video pipeline and backend initialization.")


def diagnose(application):
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetConsoleTitleW("LTX Desktop - AMD 395 MAX Diagnostics")
    local = Path(os.environ["LOCALAPPDATA"])
    logs = local / "LTXDesktopAMD/logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / "radeon-diagnostic.txt"
    print(f"Diagnostic log: {log}", flush=True)
    with log.open("w", encoding="utf-8", buffering=1) as stream:
        def output(message):
            print(message, file=stream)
            print(message, flush=True)
        try:
            output("LTX Desktop - AMD 395 MAX Diagnostics")
            output(f"{datetime.now().isoformat(timespec='seconds')} | {sys.version} | {sys.executable}")
            # A fresh process both exercises real import order and releases SQLite/driver
            # handles before we remove the temporary data directory on Windows.
            with tempfile.TemporaryDirectory(prefix="ltx-amd-check-") as scratch:
                env = os.environ.copy()
                env.update(LTX_REQUIRE_ROCM="1", LTX_APP_DATA_DIR=scratch, LTX_ASSETS_DIR="",
                           LTX_DEV_MODE="0", LTX_ANALYTICS_SINK_URL="", LTX_ANALYTICS_TOKEN="",
                           BACKEND_DEBUG="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                           HF_HUB_DISABLE_TELEMETRY="1", LTX_PORT="41956", LTX_REMOTE_PORT="41957")
                command = [sys.executable, "-u", "-X", "utf8", str(Path(__file__).resolve()),
                           "--probe", "--application", str(application)]
                with subprocess.Popen(command, env=env, cwd=application / "resources/backend",
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, encoding="utf-8", errors="replace") as child:
                    for line in child.stdout:
                        output(line.rstrip())
                    code = child.wait()
                if code:
                    output(f"FAIL: Backend diagnostic process exited with code {code}.")
                return int(code != 0)
        except Exception:
            output(traceback.format_exc())
            return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--application", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.probe:
        probe(args.application.resolve())
    else:
        sys.exit(diagnose(args.application.resolve()))
