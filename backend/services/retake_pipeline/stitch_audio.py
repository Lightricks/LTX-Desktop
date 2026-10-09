"""Hard-cut audio for retake and extend stitches. Video fades; audio does not."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from services.ffmpeg import ffmpeg_seconds


@dataclass(frozen=True)
class AudioPiece:
    """``label`` between ``start`` and ``end`` seconds. ``None`` runs to that edge."""

    label: str
    start: float | None = None
    end: float | None = None


def audio_cut_graph(pieces: Sequence[AudioPiece]) -> str:
    """Concatenate ``pieces`` in order into ``[a]``."""
    parts: list[str] = []
    outputs: list[str] = []
    for index, piece in enumerate(pieces):
        bounds: list[str] = []
        if piece.start is not None:
            bounds.append(f"start={ffmpeg_seconds(piece.start)}")
        if piece.end is not None:
            bounds.append(f"end={ffmpeg_seconds(piece.end)}")
        output = f"[p{index}]"
        parts.append(f"{piece.label}atrim={':'.join(bounds)},asetpts=PTS-STARTPTS{output}")
        outputs.append(output)
    parts.append(f"{''.join(outputs)}concat=n={len(outputs)}:v=0:a=1[a]")
    return ";".join(parts)
