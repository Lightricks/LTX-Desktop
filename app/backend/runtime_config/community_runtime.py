"""Select the managed video backend for the installed desktop and visible hardware."""

import os
import platform
import sys
from functools import lru_cache
from typing import Literal

HOST_PLATFORM = sys.platform
LocalVideoBackend = Literal["cuda", "rocm", "mps"]


@lru_cache(maxsize=2)
def _native_backend(host: str) -> LocalVideoBackend | None:
    import torch

    if host == "darwin" and platform.machine() == "arm64" and torch.backends.mps.is_available():
        return "mps"
    if host == "linux" and torch.version.cuda and torch.cuda.is_available():
        return "cuda"
    return None


def local_video_backend() -> LocalVideoBackend | None:
    if HOST_PLATFORM != "win32":
        return _native_backend(HOST_PLATFORM)
    selected = os.environ.get("LTX_GPU_RUNTIME")
    if selected == "cuda":
        return "cuda"
    if selected == "rocm":
        return "rocm"
    # Legacy AMD installs/source launches predate the explicit runtime selection.
    if selected is None and os.environ.get("LTX_REQUIRE_ROCM") == "1":
        return "rocm"
    return None


def local_video_renderer_available() -> bool:
    return local_video_backend() is not None


def windows_video_renderer_available() -> bool:
    return HOST_PLATFORM == "win32" and local_video_renderer_available()
