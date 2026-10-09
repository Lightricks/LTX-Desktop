"""Local source-video prep shared by the retake and extend local paths.

The local pipeline needs width/height divisible by 32 and a frame count of the form
``8k+1``. GenSpace snaps a requested size down to that grid. Home retake and extend
keep the picture size and letterbox up to the next multiple of 32, then crop the
bars off after generation. Frame count is always trimmed down to the nearest ``8k+1``.
"""

from __future__ import annotations

from dataclasses import dataclass
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


# Home/queued Retake and Extend fit inside the local 1080p cell (1920×1088).
# 1088 (not 1080) is the box so a clip that is already on the ÷32 grid is not shrunk.
# A 1920×1080 picture is letterboxed up to 1920×1088, then cropped back.
_1080P_LONG_EDGE = 1920
_1080P_SHORT_EDGE = 1088


@dataclass(frozen=True)
class SpatialLetterbox:
    """Picture rect centered on the ÷32 canvas the model actually encodes."""

    content_width: int
    content_height: int
    canvas_width: int
    canvas_height: int
    left: int
    top: int

    @property
    def right(self) -> int:
        return self.canvas_width - self.content_width - self.left

    @property
    def bottom(self) -> int:
        return self.canvas_height - self.content_height - self.top


def generation_letterbox(width: int, height: int) -> SpatialLetterbox:
    """Pad ``width``×``height`` up to the next multiple of 32, split evenly.

    An odd remainder gives the extra pixel to the right or the bottom.
    """
    canvas_width = ((width + _SPATIAL_FACTOR - 1) // _SPATIAL_FACTOR) * _SPATIAL_FACTOR
    canvas_height = ((height + _SPATIAL_FACTOR - 1) // _SPATIAL_FACTOR) * _SPATIAL_FACTOR
    pad_x = canvas_width - width
    pad_y = canvas_height - height
    return SpatialLetterbox(
        content_width=width,
        content_height=height,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        left=pad_x // 2,
        top=pad_y // 2,
    )


def cap_to_1080p(width: int, height: int) -> tuple[int, int]:
    """Fit inside local 1080p without upscaling or changing aspect.

    Home/queued Retake and Extend opt into this. GenSpace ``resolve_target_resolution``
    does not — it keeps the requested (or source) size after the ÷32 snap.
    """
    max_w, max_h = (
        (_1080P_LONG_EDGE, _1080P_SHORT_EDGE)
        if width >= height
        else (_1080P_SHORT_EDGE, _1080P_LONG_EDGE)
    )
    scale = min(max_w / width, max_h / height, 1.0)
    if scale >= 1.0:
        return width, height
    return max(1, round(width * scale)), max(1, round(height * scale))


def resolve_home_edit_size(
    requested_width: int | None,
    requested_height: int | None,
    source_width: int,
    source_height: int,
) -> tuple[int, int]:
    """Home Retake/Extend picture size, before the ÷32 letterbox.

    A missing request keeps the source. The 1080p cap still applies, and the
    result is never larger than the source. Callers pad this with
    ``generation_letterbox`` for the model and crop back to this size.
    """
    width = source_width if requested_width is None or requested_height is None else requested_width
    height = source_height if requested_width is None or requested_height is None else requested_height
    width, height = cap_to_1080p(width, height)
    return min(width, source_width), min(height, source_height)


def resolve_target_resolution(
    resolution: TargetResolution | None,
    source_width: int,
    source_height: int,
) -> tuple[int, int]:
    """Resolve the effective output size: the requested resolution if given, else the
    source — always corrected to a valid (÷32, not-upscaled) size.
    """
    target_width = resolution.width if resolution else source_width
    target_height = resolution.height if resolution else source_height
    return correct_resolution(
        target_width,
        target_height,
        source_width=source_width,
        source_height=source_height,
    )
