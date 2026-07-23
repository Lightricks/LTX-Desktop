"""Local source-video prep shared by the retake and extend local paths.

The local pipeline needs width/height divisible by 32 and a frame count of the form
``8k+1``. Rather than rejecting inputs that don't comply (standard 1080p isn't ÷32 in
height; arbitrary clips rarely land on 8k+1), we correct both: snap resolution down to a
÷32 size (never above source) and trim the frame count down to the nearest ``8k+1``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from _routes._errors import HTTPError

if TYPE_CHECKING:
    from api_types import TargetResolution

_SPATIAL_FACTOR = 32
# VAE temporal downscale: valid frame counts are 8k+1. Single source of truth for the
# 8-frame rule (extend's snap-up and video generation's frame math reuse this).
TIME_FACTOR = 8
_MIN_FRAMES = 9  # smallest usable clip (2 latent frames).

# The source is user-selected from anywhere on disk (file dialog / drag-drop), so it can't
# be confined to the app dirs. We require a real regular file with a recognized video
# extension instead — enough to keep the localhost endpoint from being coerced into
# reading/uploading an arbitrary file (e.g. ~/.ssh/id_rsa) on the cloud path.
_ALLOWED_SOURCE_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".webm", ".mkv"})


def validate_source_video_path(video_path: str) -> Path:
    """Resolve and sanity-check a caller-supplied source video path before use."""
    if not video_path:
        raise HTTPError(400, "Missing video_path parameter")
    resolved = Path(video_path).resolve()
    if not resolved.is_file():
        raise HTTPError(400, f"Video file not found: {video_path}")
    if resolved.suffix.lower() not in _ALLOWED_SOURCE_SUFFIXES:
        raise HTTPError(400, "Unsupported video file type")
    return resolved


def read_source_metadata(video_path: str) -> tuple[float, int, int, int]:
    """Return (fps, width, height, frames). Frame count and resolution are corrected
    downstream (not rejected)."""
    from ltx_pipelines.utils.media_io import get_videostream_metadata

    try:
        meta = get_videostream_metadata(video_path)
    except Exception as exc:  # av.open & friends raise lib-specific errors on bad input
        raise HTTPError(400, "Couldn't read video file; it may be corrupt or unsupported.") from exc
    return meta.fps, meta.width, meta.height, meta.frames


def correct_frame_count(frames: int) -> int:
    """Trim the frame count down to the nearest valid ``8k+1`` (never fabricating frames)."""
    corrected = ((frames - 1) // TIME_FACTOR) * TIME_FACTOR + 1
    if corrected < _MIN_FRAMES:
        raise HTTPError(400, f"Video is too short: it must have at least {_MIN_FRAMES} frames.")
    return corrected


def correct_resolution(
    target_width: int,
    target_height: int,
    *,
    source_width: int,
    source_height: int,
) -> tuple[int, int]:
    # Never upscale, then snap each edge down to a multiple of 32 (min one tile).
    width = min(target_width, source_width)
    height = min(target_height, source_height)
    width = max(_SPATIAL_FACTOR, (width // _SPATIAL_FACTOR) * _SPATIAL_FACTOR)
    height = max(_SPATIAL_FACTOR, (height // _SPATIAL_FACTOR) * _SPATIAL_FACTOR)
    return width, height


def resolve_target_resolution(
    resolution: TargetResolution | None,
    source_width: int,
    source_height: int,
) -> tuple[int, int]:
    """Resolve the effective output size: the requested resolution if given, else the
    source — always corrected to a valid (÷32, not-upscaled) size."""
    target_width = resolution.width if resolution else source_width
    target_height = resolution.height if resolution else source_height
    return correct_resolution(
        target_width, target_height, source_width=source_width, source_height=source_height
    )
