"""Read the sound of a clip and find where someone speaks.

Shared by the voice bridge (``voice_context``) and the level match (``audio_level``).
``speech_spans`` is an energy check, not a voice detector. Music or noise counts as sound.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

import av
import numpy as np

from inference.ffmpeg import (
    FFMPEG_PROTOCOL_WHITELIST,
    FfmpegTimeoutError,
    FfmpegUnavailableError,
    run_ffmpeg,
    timeout_for_source_duration,
)

# What reading or writing the sound of a clip can raise. The callers are best effort: they log
# it and carry on without the audio feature.
AUDIO_ERRORS = (
    subprocess.CalledProcessError,
    FfmpegTimeoutError,
    FfmpegUnavailableError,
    av.FFmpegError,
)

ANALYSIS_SAMPLE_RATE = 16_000
ACTIVITY_WINDOW_SECONDS = 0.05
# Sound counts as speech above the floor and within this many dB of the loud level.
SPEECH_FLOOR_DBFS = -50.0
SPEECH_BELOW_LOUD_DB = 30.0
# Pauses shorter than this stay inside one speech span.
MERGE_GAP_SECONDS = 0.2


@dataclass(frozen=True)
class Span:
    start: float
    end: float


def speech_spans(samples: np.ndarray, rate: int) -> list[Span]:
    hop = max(1, int(rate * ACTIVITY_WINDOW_SECONDS))
    count = len(samples) // hop
    if count == 0:
        return []
    frames = samples[: count * hop].reshape(count, hop)
    level_db = 20.0 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-9)
    threshold = max(SPEECH_FLOOR_DBFS, float(np.percentile(level_db, 95)) - SPEECH_BELOW_LOUD_DB)
    seconds_per_hop = hop / rate
    spans: list[Span] = []
    for index in np.flatnonzero(level_db > threshold):
        start, end = float(index) * seconds_per_hop, float(index + 1) * seconds_per_hop
        if spans and start - spans[-1].end <= MERGE_GAP_SECONDS:
            spans[-1] = Span(spans[-1].start, end)
        else:
            spans.append(Span(start, end))
    return spans


def decode_mono(
    source_path: str, duration: float, *, start: float | None = None, end: float | None = None
) -> np.ndarray:
    """Mono samples on the clip's own timeline: sample 0 is time 0, the first video frame.

    A track that starts late (common in phone and screen recordings) is padded with silence,
    and one that starts early is trimmed, so a speech span in seconds is the same moment as
    that second of video. Only the start is aligned; the audio is never stretched.

    ``start`` and ``end`` (seconds on that timeline) keep only that stretch, so a short
    measurement does not hold the whole track in memory. Sample 0 is then ``start``.
    ``duration`` is the length of the file, for the ffmpeg time limit."""
    chain = "aresample=async=1:first_pts=0"
    if start is not None or end is not None:
        bounds = [f"start={max(start, 0.0):.6f}" if start is not None else "", f"end={end:.6f}" if end is not None else ""]
        chain += ",atrim=" + ":".join(b for b in bounds if b)
    completed = run_ffmpeg(
        [
            "-protocol_whitelist", FFMPEG_PROTOCOL_WHITELIST, "-i", source_path, "-vn",
            "-af", chain,
            "-ac", "1", "-ar", str(ANALYSIS_SAMPLE_RATE), "-f", "f32le", "-",
        ],
        timeout=timeout_for_source_duration(duration),
    )
    return np.frombuffer(completed.stdout, dtype=np.float32)
