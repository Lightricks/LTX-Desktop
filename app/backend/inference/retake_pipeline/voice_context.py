"""Keep a voice in the extend context when the kept source tail is silent.

The Short context mode of Extend encodes about 2s of source. The model copies the voice from the
audio in that window. When the speaker stopped before the window starts, the window holds no
voice and the new speech comes out in another voice.

The fix builds a short bridge clip for the model to read. It joins a piece of the last speech
(picture and sound together, so the lips match the voice) with the source frames at the seam
that the model regenerates. The bridge has as many frames as the window, so the run costs
the same. The stitch reads the original file for the source part of the output, so the bridge
never reaches the output.

``speech_spans`` is an energy check, not a voice detector. Music or noise in the window
counts as sound, so the bridge is skipped there.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

from inference.ffmpeg import (
    FFMPEG_PROTOCOL_WHITELIST,
    ffmpeg_seconds,
    run_ffmpeg,
    timeout_for_source_duration,
)
from inference.media import media_has_audio
from inference.retake_pipeline.speech_analysis import (
    ANALYSIS_SAMPLE_RATE,
    AUDIO_ERRORS,
    Span,
    decode_mono,
    speech_spans,
)
from inference.retake_pipeline.window import MASK_DELTA_SECONDS, ExtendEncodeWindow
from inference.types import ExtendMode

logger = logging.getLogger(__name__)

BRIDGE_SAMPLE_RATE = 48_000
# Shortest span that counts as a voice sample, and as speech already in the window.
MIN_SPEECH_SECONDS = 0.4
# Real frames kept before the part the model regenerates. A longer run lets the model carry
# the pose across the seam. A short one gives a visible jump. The rest of the window is the
# speech piece, which must stay above ``MIN_SPEECH_SECONDS``.
KEPT_SEAM_SECONDS = 0.8


@dataclass(frozen=True)
class VoiceBridge:
    """Two source frame ranges. The bridge plays the speech piece next to the seam piece."""

    speech_start_frame: int
    speech_frames: int
    seam_start_frame: int
    seam_frames: int
    mode: ExtendMode

    @property
    def total_frames(self) -> int:
        return self.speech_frames + self.seam_frames


def _speech_inside(spans: list[Span], start: float, end: float) -> float:
    return sum(max(0.0, min(span.end, end) - max(span.start, start)) for span in spans)


def plan_voice_bridge(
    spans: list[Span],
    *,
    window: ExtendEncodeWindow,
    fps: float,
    source_frames: int,
    mode: ExtendMode,
) -> VoiceBridge | None:
    """Pick the speech for the bridge, or ``None`` when the window has a voice or there is none.

    The model regenerates the ``MASK_DELTA_SECONDS`` next to the new content. The seam piece
    keeps that part and ``KEPT_SEAM_SECONDS`` of real frames before it. The speech piece fills the rest of
    the window.
    """
    seam_frames = math.ceil((MASK_DELTA_SECONDS + KEPT_SEAM_SECONDS) * fps)
    speech_frames = window.context_frames - seam_frames
    if speech_frames / fps < MIN_SPEECH_SECONDS:
        logger.info(
            "Voice bridge skipped: a %d-frame window at %.2f fps leaves %.2fs for speech (needs %.2fs)",
            window.context_frames, fps, max(speech_frames, 0) / fps, MIN_SPEECH_SECONDS,
        )
        return None
    window_start = window.start_time(fps)
    window_end = window_start + window.context_frames / fps
    seam_seconds = seam_frames / fps
    if mode == "end":
        zone = (window_start, window_end - seam_seconds)
        seam_start_frame = window.encode_start_frame + speech_frames
    else:
        zone = (window_start + seam_seconds, window_end)
        seam_start_frame = window.encode_start_frame
    if _speech_inside(spans, *zone) >= MIN_SPEECH_SECONDS:
        return None
    # A span that runs into the zone counts up to the zone edge: that part is speech next to
    # the window, and the piece must not reach into the window.
    if mode == "end":
        ends = [min(s.end, zone[0]) for s in spans if min(s.end, zone[0]) - s.start >= MIN_SPEECH_SECONDS]
        if not ends:
            return None
        piece_end = min(source_frames, round(ends[-1] * fps))
        speech_start_frame = max(0, piece_end - speech_frames)
    else:
        starts = [max(s.start, zone[1]) for s in spans if s.end - max(s.start, zone[1]) >= MIN_SPEECH_SECONDS]
        if not starts:
            return None
        speech_start_frame = max(0, min(round(starts[0] * fps), source_frames - speech_frames))
    return VoiceBridge(speech_start_frame, speech_frames, seam_start_frame, seam_frames, mode)


def _write_bridge_clip(source_path: str, dest_path: str, bridge: VoiceBridge, fps: float) -> None:
    def video(label: str, start: int, count: int, out: str) -> str:
        return f"[{label}]trim=start_frame={start}:end_frame={start + count},setpts=PTS-STARTPTS[{out}]"

    def audio(label: str, start: int, count: int, out: str) -> str:
        return (
            f"[{label}]atrim=start={ffmpeg_seconds(start / fps)}:end={ffmpeg_seconds((start + count) / fps)},"
            f"asetpts=PTS-STARTPTS,aformat=sample_rates={BRIDGE_SAMPLE_RATE}:channel_layouts=stereo[{out}]"
        )

    speech = (bridge.speech_start_frame, bridge.speech_frames)
    seam = (bridge.seam_start_frame, bridge.seam_frames)
    order = ("vs", "as", "vt", "at") if bridge.mode == "end" else ("vt", "at", "vs", "as")
    graph = ";".join(
        [
            "[0:v]split=2[v0][v1]",
            # Same alignment as decode_mono: a late track is padded, so each piece's sound
            # sits under its own frames.
            "[0:a]aresample=async=1:first_pts=0,asplit=2[a0][a1]",
            video("v0", *speech, "vs"),
            video("v1", *seam, "vt"),
            audio("a0", *speech, "as"),
            audio("a1", *seam, "at"),
            f"[{order[0]}][{order[1]}][{order[2]}][{order[3]}]concat=n=2:v=1:a=1[v][a]",
        ]
    )
    run_ffmpeg(
        [
            "-y", "-protocol_whitelist", FFMPEG_PROTOCOL_WHITELIST, "-i", source_path,
            "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
            "-r", f"{fps:.6f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "12",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", dest_path,
        ],
        timeout=timeout_for_source_duration(bridge.total_frames / fps),
    )


def find_voice_bridge(
    source_path: str,
    *,
    window: ExtendEncodeWindow,
    fps: float,
    source_frames: int,
    mode: ExtendMode,
) -> VoiceBridge | None:
    """Plan the bridge from the sound of the source. Writes no file. Best effort."""
    try:
        if not media_has_audio(source_path):
            return None
        spans = speech_spans(decode_mono(source_path, source_frames / fps), ANALYSIS_SAMPLE_RATE)
    except AUDIO_ERRORS:
        logger.warning("Extend voice bridge skipped: could not read the audio", exc_info=True)
        return None
    return plan_voice_bridge(spans, window=window, fps=fps, source_frames=source_frames, mode=mode)


def write_voice_bridge(source_path: str, dest_path: str, bridge: VoiceBridge, fps: float) -> bool:
    """Write the bridge clip to ``dest_path``. ``False``: nothing written, the extend reads the
    plain source. Best effort."""
    try:
        _write_bridge_clip(source_path, dest_path, bridge, fps)
    except AUDIO_ERRORS:
        logger.warning("Extend voice bridge skipped: could not write the clip", exc_info=True)
        Path(dest_path).unlink(missing_ok=True)
        return False
    return True
