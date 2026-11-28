"""Match the loudness of generated speech to the speech in the source.

The model often returns speech that is much louder than the source. The stitch joins both
in one track, so the level jumps at the seam. Extend has one seam, Retake has one on each side
of the new part.
"""

from __future__ import annotations

import logging

import numpy as np

from inference.media import media_has_audio
from inference.retake_pipeline.speech_analysis import (
    ANALYSIS_SAMPLE_RATE,
    AUDIO_ERRORS,
    decode_mono,
    speech_spans,
)
from inference.types import ExtendMode

logger = logging.getLogger(__name__)

# Largest correction. A bigger gap is more likely a wrong measurement than a real level.
MAX_CUT_DB = 18.0
MAX_BOOST_DB = 6.0
# The source level is read within this many seconds of the seam: the stretch a listener hears
# next to the generated speech. A clip with a loud intro and a quiet end would otherwise be
# matched to its average. A clip shorter than this is read whole.
SEAM_LEVEL_SECONDS = 10.0


def _voiced(samples: np.ndarray) -> np.ndarray | None:
    """The speech parts of ``samples`` joined, or ``None`` when there is no speech."""
    rate = ANALYSIS_SAMPLE_RATE
    parts = [samples[int(span.start * rate) : int(span.end * rate)] for span in speech_spans(samples, rate)]
    return np.concatenate(parts) if parts else None


def _rms_db(voiced: np.ndarray) -> float:
    return float(20.0 * np.log10(np.sqrt((voiced**2).mean()) + 1e-9))


def speech_level_db(samples: np.ndarray) -> float | None:
    """RMS level in dBFS over the speech parts, or ``None`` when there is no speech."""
    voiced = _voiced(samples)
    return None if voiced is None else _rms_db(voiced)


def generated_gain_db(
    *,
    source_path: str,
    generated_path: str,
    mode: ExtendMode,
    generated_cut: float,
    source_seconds: float,
    generated_seconds: float,
) -> float:
    """Gain in dB for the generated part, from the cut to the end (``end``) or from the start to
    the cut (``start``). ``0.0`` when a level cannot be measured. Best effort."""
    try:
        if not media_has_audio(source_path):
            return 0.0
        # Only the stretch next to the seam is decoded, not the whole track. The source
        # stretch is on the source timeline, so it uses the source length, not the length of
        # the stitched output.
        if mode == "end":
            source_samples = decode_mono(
                source_path, source_seconds, start=max(source_seconds - SEAM_LEVEL_SECONDS, 0.0)
            )
            generated_samples = decode_mono(generated_path, generated_seconds, start=generated_cut)
        else:
            source_samples = decode_mono(source_path, source_seconds, end=SEAM_LEVEL_SECONDS)
            generated_samples = decode_mono(generated_path, generated_seconds, end=generated_cut)
        source = speech_level_db(source_samples)
        piece = speech_level_db(generated_samples)
    except AUDIO_ERRORS:
        logger.warning("Extend level match skipped: could not measure the audio", exc_info=True)
        return 0.0
    if source is None or piece is None:
        return 0.0
    gain = min(MAX_BOOST_DB, max(-MAX_CUT_DB, source - piece))
    logger.info("Extend level match: source %.1f dB, generated %.1f dB, gain %.1f dB", source, piece, gain)
    return gain


def retake_gain_db(
    *,
    source_path: str,
    generated_path: str,
    mask_start: float,
    mask_end: float,
    generated_start: float,
    source_seconds: float,
    generated_seconds: float,
) -> float:
    """Gain in dB for the new part of a retake. ``mask_start`` and ``mask_end`` are the selection
    on the source timeline, ``generated_start`` is where the selection starts in the generated
    file. The new speech is matched to the speech on both sides of the selection, read within
    ``SEAM_LEVEL_SECONDS`` of it, since a listener hears both joins. ``0.0`` when a level cannot be
    measured. Best effort."""
    try:
        if not media_has_audio(source_path):
            return 0.0
        neighbours: list[np.ndarray] = []
        if mask_start > 0:
            before = decode_mono(
                source_path, source_seconds, start=max(mask_start - SEAM_LEVEL_SECONDS, 0.0), end=mask_start
            )
            neighbours.append(before)
        if mask_end < source_seconds:
            after = decode_mono(
                source_path, source_seconds, start=mask_end, end=min(mask_end + SEAM_LEVEL_SECONDS, source_seconds)
            )
            neighbours.append(after)
        generated_samples = decode_mono(
            generated_path,
            generated_seconds,
            start=generated_start,
            end=generated_start + (mask_end - mask_start),
        )
    except AUDIO_ERRORS:
        logger.warning("Retake level match skipped: could not measure the audio", exc_info=True)
        return 0.0
    voiced = [part for part in map(_voiced, neighbours) if part is not None]
    piece = speech_level_db(generated_samples)
    if not voiced or piece is None:
        return 0.0
    source = _rms_db(np.concatenate(voiced))
    gain = min(MAX_BOOST_DB, max(-MAX_CUT_DB, source - piece))
    logger.info("Retake level match: source %.1f dB, generated %.1f dB, gain %.1f dB", source, piece, gain)
    return gain
