"""Audio cuts for retake and extend stitches. Video fades over many frames. Audio is cut at one
point with a very short fade on each side, so the join does not click."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from inference.ffmpeg import ffmpeg_seconds


# Length of the fade on each side of a join. A hard join leaves a step in the waveform, which
# is heard as a click. A longer fade would dip the level in speech.
JOIN_FADE_SECONDS = 0.015


@dataclass(frozen=True)
class AudioPiece:
    """``label`` between ``start`` and ``end`` seconds. ``None`` runs to that edge.

    ``gain_db`` changes the level. A peak limiter stops clipping after a boost; a cut cannot
    clip, so it gets none."""

    label: str
    start: float | None = None
    end: float | None = None
    gain_db: float = 0.0


def _join_fades(piece: AudioPiece, *, after_join: bool, before_join: bool) -> str:
    """Fade in after a join and fade out before one. Pieces keep their length, so audio and
    video stay in sync. A piece that is too short for both fades gets shorter fades, and an
    empty piece gets none (a zero-length fade is not a valid filter).

    The fade-out sits at the requested end of the piece. A track that stops before that end
    simply ends where it ends, which leaves no cut to fade."""
    length = None if piece.end is None else piece.end - (piece.start or 0.0)
    fade = JOIN_FADE_SECONDS if length is None else min(JOIN_FADE_SECONDS, length / 2)
    if fade <= 0:
        return ""
    fades = ""
    if after_join:
        fades += f",afade=t=in:d={ffmpeg_seconds(fade)}"
    if before_join and length is not None:
        fades += f",afade=t=out:st={ffmpeg_seconds(length - fade)}:d={ffmpeg_seconds(fade)}"
    return fades


def audio_cut_graph(pieces: Sequence[AudioPiece]) -> str:
    """Concatenate ``pieces`` in order into ``[a]``, with a short fade at each join."""
    parts: list[str] = []
    outputs: list[str] = []
    for index, piece in enumerate(pieces):
        bounds: list[str] = []
        if piece.start is not None:
            bounds.append(f"start={ffmpeg_seconds(piece.start)}")
        if piece.end is not None:
            bounds.append(f"end={ffmpeg_seconds(piece.end)}")
        output = f"[p{index}]"
        gain = ""
        if piece.gain_db:
            gain = f",volume={piece.gain_db:.2f}dB"
            if piece.gain_db > 0:
                # ``level=0``: the limiter's auto level would add about +0.26 dB of its own.
                gain += ",alimiter=limit=0.97:level=0"
        fades = _join_fades(piece, after_join=index > 0, before_join=index < len(pieces) - 1)
        parts.append(f"{piece.label}atrim={':'.join(bounds)},asetpts=PTS-STARTPTS{gain}{fades}{output}")
        outputs.append(output)
    parts.append(f"{''.join(outputs)}concat=n={len(outputs)}:v=0:a=1[a]")
    return ";".join(parts)
