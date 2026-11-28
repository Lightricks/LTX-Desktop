"""Entry point for ComfyUI under the LTX Desktop embeddable Python interpreter."""

from __future__ import annotations

import os
import runpy
import site
import sys
import threading
import time
from pathlib import Path


def _watch_parent(parent: int) -> None:
    # A macOS/Linux worker can outlive a crashed Electron backend. Its original
    # parent PID changes on reparenting, even if the old PID is later reused.
    while os.getppid() == parent:
        time.sleep(2)
    os._exit(0)


def _verify_gpu(device: str) -> None:
    import torch

    if device == "mps":
        available = torch.backends.mps.is_available()
    elif device == "cuda":
        available = torch.cuda.is_available() and bool(torch.version.cuda)
    else:
        raise RuntimeError(f"Unsupported private engine GPU: {device}")
    if not available:
        raise RuntimeError(f"The bundled PyTorch runtime cannot access the {device} GPU.")
    with torch.inference_mode():
        sample = torch.ones((4, 4), device=device, dtype=torch.float16)
        output = sample @ sample
        if output.device.type != device or output[0, 0].item() != 4:
            raise RuntimeError(f"The private engine calculation failed on {device}.")


def main() -> None:
    core, packages = (Path(arg).resolve(strict=True) for arg in sys.argv[1:3])
    # The Windows embedded interpreter ignores PYTHONPATH. Its site initialization
    # has already registered the installed torch runtime's DLL directories. Enable the private
    # wheels' .pth initialization and give Comfy its own PyAV before any imports.
    package_path = str(packages)
    sys.path[:0] = [str(core), package_path]
    site.addsitedir(package_path)
    if parent := os.environ.get("LTX_COMFY_PARENT_PID"):
        threading.Thread(target=_watch_parent, args=(int(parent),), daemon=True).start()
    if device := os.environ.get("LTX_COMFY_GPU_PROBE"):
        _verify_gpu(device)
    entry = str(core / "main.py")
    sys.argv = [entry, *sys.argv[3:]]
    runpy.run_path(entry, run_name="__main__")


if __name__ == "__main__":
    main()
