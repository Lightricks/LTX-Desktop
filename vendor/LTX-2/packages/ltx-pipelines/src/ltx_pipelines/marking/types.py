"""Public video and audio marker protocols and detect result."""

from __future__ import annotations

from collections.abc import Iterator
from typing import NamedTuple, Protocol

import torch

from ltx_core.types import Audio


class MarkDetectResult(NamedTuple):
    """Clip-level detector output.
    ``bit_accuracy`` and ``p_value`` are the binomial test of decoded bits vs the baked
    payload. ``detected`` is this marker's own call (not that test).
    """

    detected: bool
    bit_accuracy: float
    p_value: float
    payload: torch.Tensor | None


class VideoMarker(Protocol):
    """Embeds a watermark into RGB frames and detects it in a frame stream."""

    @property
    def name(self) -> str:
        """Short identifier."""
        ...

    def embed(self, frames: torch.Tensor) -> torch.Tensor:
        """Stamp one ``[F, H, W, C]`` RGB tensor in ``[0, 1]``; return the same layout."""
        ...

    def detect(self, frames: Iterator[torch.Tensor]) -> MarkDetectResult:
        """Detect the watermark in a stream of ``[H, W, C]`` RGB frames in ``[0, 1]``."""
        ...


class AudioMarker(Protocol):
    """Embeds a watermark into stereo PCM and detects it in an ``Audio`` track."""

    @property
    def name(self) -> str:
        """Short identifier."""
        ...

    def embed(self, audio: Audio) -> Audio:
        """Stamp one yielded stereo ``Audio`` track."""
        ...

    def detect(self, audio: Audio) -> MarkDetectResult:
        """Detect the audio watermark in one stereo ``Audio`` track."""
        ...


__all__ = ["AudioMarker", "MarkDetectResult", "VideoMarker"]
