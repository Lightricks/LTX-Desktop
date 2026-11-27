"""GPU runtime selected by the Windows community installer and verified at startup."""

import os
import sys
from typing import Literal

HOST_PLATFORM = sys.platform
LocalVideoBackend = Literal["cuda", "rocm"]


def local_video_backend() -> LocalVideoBackend | None:
    if HOST_PLATFORM != "win32":
        return None
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
