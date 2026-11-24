"""Post-hoc video and audio marking (Videoseal / AudioSeal opt-in).
Importing this package does not require the ``videoseal`` or ``audioseal`` extras.
A pack directory is loaded with :func:`create_markers`. Seal classes live in
``ltx_pipelines.marking.video_seal`` and ``ltx_pipelines.marking.audio_seal``.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ltx_pipelines.marking.mark_detection import detect_audio_mark, detect_video_mark
from ltx_pipelines.marking.types import AudioMarker, MarkDetectResult, VideoMarker

PIXELSEAL_CHECKPOINT_NAME = "pixelseal.pth"
PIXELSEAL_CARD_NAME = "pixelseal.yaml"
PIXELSEAL_ATTENUATION_NAME = "videoseal_attenuation.yaml"
AUDIOSEAL_GENERATOR_NAME = "audioseal_generator.pth"
AUDIOSEAL_DETECTOR_NAME = "audioseal_detector.pth"

_PACK_FILES = (
    PIXELSEAL_CHECKPOINT_NAME,
    PIXELSEAL_CARD_NAME,
    PIXELSEAL_ATTENUATION_NAME,
    AUDIOSEAL_GENERATOR_NAME,
    AUDIOSEAL_DETECTOR_NAME,
)


def _require_marking_pack(root: Path | str) -> Path:
    pack = Path(root)
    missing = [name for name in _PACK_FILES if not (pack / name).is_file()]
    if missing:
        raise FileNotFoundError(f"marking pack {pack} requires {', '.join(missing)}")
    return pack


def create_markers(root: Path | str, device: torch.device) -> tuple[VideoMarker, AudioMarker]:
    """Load VideoSeal and AudioSeal markers from a marking pack directory.
    All pack files must be present. Both extras are imported (fail fast).
    """
    pack = _require_marking_pack(root)
    try:
        from ltx_pipelines.marking.audio_seal import AudioSealMarker  # noqa: PLC0415
        from ltx_pipelines.marking.video_seal import VideoSealMarker  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise ImportError("videoseal and audioseal packages are required") from exc
    return (
        VideoSealMarker.create(
            pack / PIXELSEAL_CHECKPOINT_NAME,
            card=pack / PIXELSEAL_CARD_NAME,
            attenuation_config=pack / PIXELSEAL_ATTENUATION_NAME,
            device=device,
        ),
        AudioSealMarker.create(
            pack / AUDIOSEAL_GENERATOR_NAME,
            pack / AUDIOSEAL_DETECTOR_NAME,
            device=device,
        ),
    )


__all__ = [
    "AUDIOSEAL_DETECTOR_NAME",
    "AUDIOSEAL_GENERATOR_NAME",
    "PIXELSEAL_ATTENUATION_NAME",
    "PIXELSEAL_CARD_NAME",
    "PIXELSEAL_CHECKPOINT_NAME",
    "AudioMarker",
    "MarkDetectResult",
    "VideoMarker",
    "create_markers",
    "detect_audio_mark",
    "detect_video_mark",
]
