"""ffmpeg invocation shared by audio trim and video stitch."""

from __future__ import annotations

import logging
import subprocess
import time
from collections.abc import Callable

import imageio_ffmpeg

from services.records import MediaError

logger = logging.getLogger(__name__)

# Local file outputs/inputs only (plus pipe for stdio-style transports).
FFMPEG_PROTOCOL_WHITELIST = "file,pipe"
# Re-encoding a long import can legitimately take tens of seconds.
FFMPEG_TIMEOUT_SECONDS = 120.0
# Wing re-encode is slower than realtime on CPU. Floor at the shared timeout
# and cap so one long source cannot hold the worker all day.
STITCH_TIMEOUT_PER_SOURCE_SECOND = 8.0
STITCH_TIMEOUT_CAP_SECONDS = 1800.0


class FfmpegCancelledError(Exception):
    """``should_cancel`` turned true and ffmpeg was killed."""


_STDERR_LOG_CHARS = 800
# How often a cancellable run checks its cancel flag.
_CANCEL_POLL_SECONDS = 0.2


def timeout_for_source_duration(duration_seconds: float) -> float:
    """How long a stitch of a source this long may run ffmpeg."""
    scaled = max(0.0, duration_seconds) * STITCH_TIMEOUT_PER_SOURCE_SECOND
    return min(STITCH_TIMEOUT_CAP_SECONDS, max(FFMPEG_TIMEOUT_SECONDS, scaled))


def ffmpeg_seconds(value: float) -> str:
    """``str(1e-05)`` is scientific notation, which ``-ss``/``-t`` reject."""
    return f"{value:.6f}"


def _ffmpeg() -> str:
    return str(imageio_ffmpeg.get_ffmpeg_exe())


def _run_until_cancelled(
    command: list[str], timeout: float, should_cancel: Callable[[], bool]
) -> subprocess.CompletedProcess[bytes]:
    """Like ``subprocess.run``, but kills ffmpeg when ``should_cancel`` turns true."""
    deadline = time.monotonic() + timeout
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as process:
        while True:
            try:
                stdout, stderr = process.communicate(timeout=_CANCEL_POLL_SECONDS)
            except subprocess.TimeoutExpired:
                if should_cancel():
                    process.kill()
                    process.communicate()
                    raise FfmpegCancelledError() from None
                if time.monotonic() >= deadline:
                    process.kill()
                    process.communicate()
                    raise subprocess.TimeoutExpired(command, timeout) from None
                continue
            return subprocess.CompletedProcess(
                process.args, process.returncode, stdout, stderr
            )


def run_ffmpeg(
    args: list[str],
    *,
    ffmpeg_exe: str | list[str] | None = None,
    timeout: float = FFMPEG_TIMEOUT_SECONDS,
    check: bool = True,
    should_cancel: Callable[[], bool] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    if ffmpeg_exe is None:
        command = [_ffmpeg()]
    elif isinstance(ffmpeg_exe, str):
        command = [ffmpeg_exe]
    else:
        command = ffmpeg_exe
    full_command = [*command, "-nostdin", *args]
    try:
        if should_cancel is None:
            completed = subprocess.run(
                full_command,
                check=False,
                capture_output=True,
                timeout=timeout,
            )
        else:
            completed = _run_until_cancelled(full_command, timeout, should_cancel)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", "replace") if exc.stderr else ""
        # Logged only: stderr carries local paths that must not reach responses.
        logger.warning(
            "ffmpeg exited %s: %s", exc.returncode, stderr[-_STDERR_LOG_CHARS:]
        )
        raise
    except FileNotFoundError as exc:
        raise MediaError("UNREADABLE_MEDIA", "ffmpeg is not available") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaError("UNREADABLE_MEDIA", "ffmpeg timed out") from exc
    if check and completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode, completed.args, completed.stdout, completed.stderr
        )
    return completed
