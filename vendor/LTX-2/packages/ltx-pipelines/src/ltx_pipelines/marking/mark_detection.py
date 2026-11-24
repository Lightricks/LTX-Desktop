"""File-level mark detect: decode stamp-domain media, then call the marker."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch
from torch._prims_common import DeviceLikeType

from ltx_core.devices import get_preferred_device
from ltx_pipelines.marking.types import AudioMarker, MarkDetectResult, VideoMarker
from ltx_pipelines.utils.media_io.decode import decode_audio_from_file, decode_video_by_frame


def detect_video_mark(marker: VideoMarker, path: str | Path) -> MarkDetectResult:
    """Detects a marked file at ``path`` if it was marked by marker."""
    return marker.detect(_sdr_detect_frames(path, get_preferred_device()))


def detect_audio_mark(marker: AudioMarker, path: str | Path) -> MarkDetectResult | None:
    """Detects a marked file at ``path`` if it was marked by marker."""
    decoded = decode_audio_from_file(str(path), get_preferred_device())
    if decoded is None:
        return None
    return marker.detect(decoded)


def _sdr_detect_frames(path: str | Path, device: DeviceLikeType) -> Iterator[torch.Tensor]:
    """Yield SDR RGB frames ``[H, W, C]`` in ``[0, 1]``."""
    for frame in decode_video_by_frame(str(path), device):
        yield frame.squeeze(0).to(dtype=torch.float32).div_(255.0)
