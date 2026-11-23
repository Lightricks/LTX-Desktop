"""Desktop ffmpeg adapter: map pathless inference errors onto ``MediaError``.

Desktop imports the constants, the errors, and ``run_ffmpeg`` from here.
Inference callers keep ``inference.ffmpeg``, which cannot import ``services``.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable

from inference.ffmpeg import (
    FFMPEG_PROTOCOL_WHITELIST as FFMPEG_PROTOCOL_WHITELIST,
    FFMPEG_TIMEOUT_SECONDS as FFMPEG_TIMEOUT_SECONDS,
    PIXEL_TIMEOUT_CAP_SECONDS as PIXEL_TIMEOUT_CAP_SECONDS,
    STITCH_TIMEOUT_CAP_SECONDS as STITCH_TIMEOUT_CAP_SECONDS,
    STITCH_TIMEOUT_PER_SOURCE_SECOND as STITCH_TIMEOUT_PER_SOURCE_SECOND,
    FfmpegCancelledError as FfmpegCancelledError,
    FfmpegTimeoutError as FfmpegTimeoutError,
    FfmpegUnavailableError as FfmpegUnavailableError,
    ffmpeg_seconds as ffmpeg_seconds,
    run_ffmpeg as _run_ffmpeg,
    timeout_for_pixels as timeout_for_pixels,
    timeout_for_source_duration as timeout_for_source_duration,
)
from services.records import MediaError


def run_ffmpeg(
    args: list[str],
    *,
    ffmpeg_exe: str | list[str] | None = None,
    timeout: float = FFMPEG_TIMEOUT_SECONDS,
    check: bool = True,
    should_cancel: Callable[[], bool] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return _run_ffmpeg(
            args,
            ffmpeg_exe=ffmpeg_exe,
            timeout=timeout,
            check=check,
            should_cancel=should_cancel,
        )
    except FfmpegUnavailableError as exc:
        raise MediaError("UNREADABLE_MEDIA", "ffmpeg is not available") from exc
    except FfmpegTimeoutError as exc:
        raise MediaError("UNREADABLE_MEDIA", "ffmpeg timed out") from exc
