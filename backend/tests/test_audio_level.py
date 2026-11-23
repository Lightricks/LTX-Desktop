"""Level match for generated extend speech: measure the speech, clamp the gain, fail soft."""

from __future__ import annotations

from pathlib import Path

import pytest

from inference.retake_pipeline.audio_level import (
    MAX_BOOST_DB,
    MAX_CUT_DB,
    generated_gain_db,
    retake_gain_db,
)
from tests.video_fixtures import write_clip_with_tone_spans, write_mp4

DURATION = 4.0
FRAMES = 96  # 4 s at 24 fps


def _gain(source: Path, generated: Path, *, mode: str = "end", cut: float = 2.0) -> float:
    return generated_gain_db(
        source_path=str(source),
        generated_path=str(generated),
        mode=mode,  # type: ignore[arg-type]
        generated_cut=cut,
        source_seconds=DURATION,
        generated_seconds=DURATION,
    )


@pytest.mark.parametrize(
    ("source_amplitude", "generated_amplitude", "expected"),
    [
        (0.1, 0.2, -6.0),
        (0.5, 0.1, MAX_BOOST_DB),
        (0.07, 0.8, -MAX_CUT_DB),
    ],
)
def test_gain_matches_the_source_speech_and_stays_inside_the_clamp(
    tmp_path: Path, source_amplitude: float, generated_amplitude: float, expected: float
) -> None:
    # Bug: a near-silent or very loud measurement gives a runaway gain.
    source = write_clip_with_tone_spans(
        tmp_path / "source.mp4", frames=FRAMES, spans=[(0.0, 2.0)], amplitude=source_amplitude
    )
    generated = write_clip_with_tone_spans(
        tmp_path / "generated.mp4", frames=FRAMES, spans=[(2.0, 4.0)], amplitude=generated_amplitude
    )
    assert _gain(source, generated) == pytest.approx(expected, abs=0.7)


def test_only_the_generated_side_of_the_cut_is_measured(tmp_path: Path) -> None:
    # Bug: the audio the stitch throws away (before the cut in end mode, after it in start mode)
    # sets the gain, so a loud source-side tone changes a quiet new part.
    source = write_clip_with_tone_spans(tmp_path / "source.mp4", frames=FRAMES, spans=[(0.0, 2.0)])
    loud_before_cut = write_clip_with_tone_spans(
        tmp_path / "before.mp4", frames=FRAMES, spans=[(0.0, 2.0)], amplitude=0.8
    )
    loud_after_cut = write_clip_with_tone_spans(
        tmp_path / "after.mp4", frames=FRAMES, spans=[(2.0, 4.0)], amplitude=0.8
    )
    assert _gain(source, loud_before_cut, mode="end", cut=2.5) == 0.0
    assert _gain(source, loud_after_cut, mode="start", cut=1.5) == 0.0


def test_no_gain_when_there_is_nothing_to_compare(tmp_path: Path) -> None:
    # Bug: the stitch fails or applies a gain computed from silence.
    generated = write_clip_with_tone_spans(tmp_path / "generated.mp4", frames=FRAMES, spans=[(2.0, 4.0)])
    no_audio = write_mp4(tmp_path / "no_audio.mp4", seconds=DURATION, fps=24)
    silent = write_clip_with_tone_spans(tmp_path / "silent.mp4", frames=FRAMES, spans=[])
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    for source in (no_audio, silent, broken):
        assert _gain(source, generated) == 0.0


def test_the_source_level_is_read_near_the_seam_only(tmp_path: Path) -> None:
    # Bug: a long clip was matched to the speech of its whole length, not to the stretch beside the
    # new part, so a loud intro set the level for a quiet end.
    frames = 20 * 24
    source = write_clip_with_tone_spans(tmp_path / "source.mp4", frames=frames, spans=[(0.0, 5.0)], amplitude=0.5)
    generated = write_clip_with_tone_spans(
        tmp_path / "generated.mp4", frames=frames, spans=[(0.0, 5.0), (10.0, 20.0)], amplitude=0.1
    )
    common = {
        "source_path": str(source),
        "generated_path": str(generated),
        "source_seconds": 20.0,
        "generated_seconds": 20.0,
    }
    # End mode: the last 10 s of the source hold no speech, so there is nothing to match to.
    assert generated_gain_db(**common, mode="end", generated_cut=10.0) == 0.0
    # Start mode: the first 10 s hold the loud speech, so the quiet new part is boosted (clamped).
    assert generated_gain_db(**common, mode="start", generated_cut=10.0) == pytest.approx(MAX_BOOST_DB, abs=0.7)


def test_the_source_stretch_is_counted_on_the_source_timeline_not_the_output_timeline(tmp_path: Path) -> None:
    # Bug: the stretch was counted from the length of the stitched output (source plus extension),
    # so a long extension moved it past the end of the source and the level match did nothing.
    early_speech = write_clip_with_tone_spans(
        tmp_path / "early.mp4", frames=12 * 24, spans=[(0.0, 1.5)], amplitude=0.5
    )
    late_speech = write_clip_with_tone_spans(
        tmp_path / "late.mp4", frames=12 * 24, spans=[(10.5, 12.0)], amplitude=0.1
    )
    generated = write_clip_with_tone_spans(tmp_path / "generated.mp4", frames=8 * 24, spans=[(2.0, 8.0)], amplitude=0.2)
    common = {"generated_path": str(generated), "mode": "end", "generated_cut": 2.0, "generated_seconds": 8.0}
    # The speech in the last 10 s of the 12 s source sets the level.
    gain = generated_gain_db(source_path=str(late_speech), source_seconds=12.0, **common)  # type: ignore[arg-type]
    assert gain == pytest.approx(-6.0, abs=1.0)
    # Speech only in the first 1.5 s is outside the last 10 s: nothing to match to.
    assert generated_gain_db(source_path=str(early_speech), source_seconds=12.0, **common) == 0.0  # type: ignore[arg-type]


# Retake: the selection is 2.0-4.0 s of an 8 s clip. The generated file is the whole clip, so the
# selection starts at 2.0 s in it too.
RETAKE_SECONDS = 8.0
RETAKE_FRAMES = 192


def _retake_gain(source: Path, generated: Path) -> float:
    return retake_gain_db(
        source_path=str(source),
        generated_path=str(generated),
        mask_start=2.0,
        mask_end=4.0,
        generated_start=2.0,
        source_seconds=RETAKE_SECONDS,
        generated_seconds=RETAKE_SECONDS,
    )


def test_retake_gain_matches_the_speech_on_both_sides_of_the_selection(tmp_path: Path) -> None:
    # Bug: the regenerated speech of a retake came out several dB louder than the speech around it.
    source = write_clip_with_tone_spans(
        tmp_path / "source.mp4", frames=RETAKE_FRAMES, spans=[(0.0, 2.0), (4.0, 6.0)], amplitude=0.1
    )
    generated = write_clip_with_tone_spans(
        tmp_path / "generated.mp4", frames=RETAKE_FRAMES, spans=[(2.0, 4.0)], amplitude=0.2
    )
    assert _retake_gain(source, generated) == pytest.approx(-6.0, abs=0.7)


def test_retake_gain_uses_the_side_that_has_speech(tmp_path: Path) -> None:
    # A selection next to silence on one side still matches the speech on the other.
    source = write_clip_with_tone_spans(
        tmp_path / "source.mp4", frames=RETAKE_FRAMES, spans=[(4.0, 6.0)], amplitude=0.1
    )
    generated = write_clip_with_tone_spans(
        tmp_path / "generated.mp4", frames=RETAKE_FRAMES, spans=[(2.0, 4.0)], amplitude=0.2
    )
    assert _retake_gain(source, generated) == pytest.approx(-6.0, abs=0.7)


def test_retake_gain_measures_only_the_selection_of_the_generated_file(tmp_path: Path) -> None:
    # Bug: the context around the selection in the generated file set the gain.
    source = write_clip_with_tone_spans(
        tmp_path / "source.mp4", frames=RETAKE_FRAMES, spans=[(0.0, 2.0), (4.0, 6.0)], amplitude=0.1
    )
    loud_context = write_clip_with_tone_spans(
        tmp_path / "loud_context.mp4", frames=RETAKE_FRAMES, spans=[(0.0, 1.8), (4.2, 6.0)], amplitude=0.8
    )
    assert _retake_gain(source, loud_context) == 0.0


def test_retake_gain_is_zero_when_there_is_nothing_to_compare(tmp_path: Path) -> None:
    generated = write_clip_with_tone_spans(
        tmp_path / "generated.mp4", frames=RETAKE_FRAMES, spans=[(2.0, 4.0)], amplitude=0.2
    )
    no_audio = write_mp4(tmp_path / "no_audio.mp4", seconds=RETAKE_SECONDS, fps=24)
    silent = write_clip_with_tone_spans(tmp_path / "silent.mp4", frames=RETAKE_FRAMES, spans=[])
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    for source in (no_audio, silent, broken):
        assert _retake_gain(source, generated) == 0.0


def test_retake_gain_stays_inside_the_clamp(tmp_path: Path) -> None:
    source = write_clip_with_tone_spans(
        tmp_path / "source.mp4", frames=RETAKE_FRAMES, spans=[(0.0, 2.0), (4.0, 6.0)], amplitude=0.07
    )
    generated = write_clip_with_tone_spans(
        tmp_path / "generated.mp4", frames=RETAKE_FRAMES, spans=[(2.0, 4.0)], amplitude=0.8
    )
    assert _retake_gain(source, generated) == pytest.approx(-MAX_CUT_DB, abs=0.7)
