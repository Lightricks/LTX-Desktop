"""Home/Remote retake encode-window math (ltxv-api 505-frame worker cap)."""

from __future__ import annotations

from pathlib import Path

import pytest

from inference.retake_pipeline.window import (
    PreparedRetakeWindow,
    MAX_INPUT_FRAMES,
    RETAKE_BLEND_SECONDS,
    SELECTION_OUT_OF_RANGE_MESSAGE,
    continuation_context_frames,
    prepare_retake_window,
    retake_mask_times,
    window_for_retake,
)


def test_retake_out_of_range_matches_home_advisory_copy() -> None:
    """The Home advisory hardcodes its own out-of-range line: a drift means
    the picker and create-time enforcement disagree on the same error."""
    advisor = (
        Path(__file__).resolve().parents[2]
        / "frontend"
        / "ltx-io"
        / "screens"
        / "Feature"
        / "definitions"
        / "retake.ts"
    )
    assert SELECTION_OUT_OF_RANGE_MESSAGE in advisor.read_text(encoding="utf-8")


def test_short_clip_keeps_the_full_source_and_splits_context() -> None:
    window = window_for_retake(
        source_frames=241,
        mask_start_frame=96,
        mask_end_frame=144,
        max_input_frames=MAX_INPUT_FRAMES,
    )
    assert window.encode_start_frame == 0
    assert window.encode_frames == 241
    assert window.context_before_frames == 96
    assert window.context_after_frames == 97
    assert window.mask_start_frame == 96
    assert window.mask_end_frame == 144
    assert (window.encode_frames - 1) % 8 == 0


def test_long_clip_slices_505_around_the_mask() -> None:
    window = window_for_retake(
        source_frames=1001,
        mask_start_frame=400,
        mask_end_frame=496,
        max_input_frames=MAX_INPUT_FRAMES,
        fps=25.0,
    )
    assert window.encode_frames <= MAX_INPUT_FRAMES
    assert (window.encode_frames - 1) % 8 == 0
    assert window.encode_start_frame + window.encode_frames <= 1001
    assert window.mask_start_frame - window.encode_start_frame == window.context_before_frames
    assert (
        window.encode_start_frame + window.encode_frames - window.mask_end_frame
        == window.context_after_frames
    )
    assert window.context_before_frames > 0
    assert window.context_after_frames > 0


def test_mask_at_the_start_has_no_prefix_context() -> None:
    window = window_for_retake(
        source_frames=241,
        mask_start_frame=0,
        mask_end_frame=48,
        max_input_frames=MAX_INPUT_FRAMES,
    )
    assert window.encode_start_frame == 0
    assert window.context_before_frames == 0
    assert window.context_after_frames > 0


def test_mask_larger_than_the_cap_is_rejected() -> None:
    with pytest.raises(ValueError, match="too large"):
        window_for_retake(
            source_frames=1001,
            mask_start_frame=0,
            mask_end_frame=MAX_INPUT_FRAMES + 8,
            max_input_frames=MAX_INPUT_FRAMES,
        )


def test_full_clip_longer_than_the_cap_is_rejected() -> None:
    with pytest.raises(ValueError, match="too large"):
        window_for_retake(
            source_frames=1001,
            mask_start_frame=0,
            mask_end_frame=1001,
            max_input_frames=MAX_INPUT_FRAMES,
        )


def test_exact_cap_mask_with_leftover_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="too large"):
        window_for_retake(
            source_frames=1001,
            mask_start_frame=0,
            mask_end_frame=MAX_INPUT_FRAMES,
            max_input_frames=MAX_INPUT_FRAMES,
        )


def test_short_full_clip_at_the_cap_keeps_zero_context() -> None:
    window = window_for_retake(
        source_frames=MAX_INPUT_FRAMES,
        mask_start_frame=0,
        mask_end_frame=MAX_INPUT_FRAMES,
        max_input_frames=MAX_INPUT_FRAMES,
    )
    assert window.encode_start_frame == 0
    assert window.encode_frames == MAX_INPUT_FRAMES
    assert window.context_before_frames == 0
    assert window.context_after_frames == 0


def test_min_blend_does_not_push_the_window_over_the_cap() -> None:
    window = window_for_retake(
        source_frames=1001,
        mask_start_frame=100,
        mask_end_frame=596,
        max_input_frames=MAX_INPUT_FRAMES,
        fps=25.0,
    )
    assert window.encode_frames <= MAX_INPUT_FRAMES
    assert (window.encode_frames - 1) % 8 == 0
    assert window.mask_end_frame - window.mask_start_frame == 496


def test_denoise_mask_is_exactly_the_selection() -> None:
    """Mask == stitch edit span. A wider mask put the 16-frame lerp over
    regenerated content, which ghosted at the seam."""
    window = window_for_retake(
        source_frames=241,
        mask_start_frame=96,
        mask_end_frame=144,
        max_input_frames=MAX_INPUT_FRAMES,
    )
    fps = 24.0
    start, end = retake_mask_times(window, fps)
    assert start == pytest.approx(window.window_mask_start(fps))
    assert end == pytest.approx(window.window_mask_end(fps))
    assert end - start == pytest.approx(48 / fps)


def test_denoise_mask_at_clip_start_begins_at_zero() -> None:
    window = window_for_retake(
        source_frames=241,
        mask_start_frame=0,
        mask_end_frame=48,
        max_input_frames=MAX_INPUT_FRAMES,
    )
    start, end = retake_mask_times(window, 24.0)
    assert start == 0.0
    assert end == pytest.approx(48 / 24.0)
    assert RETAKE_BLEND_SECONDS == 0.3


@pytest.mark.parametrize("fps", [24.0, 30.0, 60.0])
def test_short_context_keeps_about_two_seconds_on_each_side(fps: float) -> None:
    mask = round(3 * fps)
    cap_frames = continuation_context_frames(fps)
    window = window_for_retake(
        source_frames=int(60 * fps),
        mask_start_frame=int(20 * fps),
        mask_end_frame=int(20 * fps) + mask,
        max_input_frames=MAX_INPUT_FRAMES,
        fps=fps,
        max_context_frames=cap_frames,
    )
    # Alignment to 8k + 1 may add a few frames to one side.
    assert cap_frames - 8 <= window.context_before_frames <= cap_frames + 8
    assert cap_frames - 8 <= window.context_after_frames <= cap_frames + 8
    assert (window.encode_frames - 1) % 8 == 0
    assert window.encode_frames < MAX_INPUT_FRAMES


def test_long_context_still_fills_the_window_to_the_cap() -> None:
    kwargs = dict(
        source_frames=1440,
        mask_start_frame=480,
        mask_end_frame=552,
        max_input_frames=MAX_INPUT_FRAMES,
        fps=24.0,
    )
    long = window_for_retake(**kwargs)
    short = window_for_retake(**kwargs, max_context_frames=continuation_context_frames(24.0))
    assert long.encode_frames >= MAX_INPUT_FRAMES - 8
    assert short.encode_frames < long.encode_frames // 2


def test_short_context_is_clamped_by_the_source_edges() -> None:
    window = window_for_retake(
        source_frames=300,
        mask_start_frame=10,
        mask_end_frame=82,
        max_input_frames=MAX_INPUT_FRAMES,
        fps=24.0,
        max_context_frames=continuation_context_frames(24.0),
    )
    assert window.context_before_frames <= 10
    assert window.encode_start_frame >= 0


def test_prepare_retake_window_short_mode_shrinks_the_window() -> None:
    kwargs = dict(
        fps=24.0,
        target_frames=721,
        target_width=1920,
        target_height=1080,
        frame_cap=MAX_INPUT_FRAMES,
        start_time=12.0,
        duration=3.0,
    )
    long = prepare_retake_window(**kwargs)
    short = prepare_retake_window(**kwargs, context_mode="short")
    assert isinstance(long, PreparedRetakeWindow) and isinstance(short, PreparedRetakeWindow)
    assert short.window.encode_frames < long.window.encode_frames
    assert (short.window.mask_start_frame, short.window.mask_end_frame) == (
        long.window.mask_start_frame,
        long.window.mask_end_frame,
    )
