"""Home/Remote retake encode-window math (ltxv-api 505-frame worker cap)."""

from __future__ import annotations

import pytest

from services.retake_pipeline.window import (
    MASK_DELTA_SECONDS,
    MAX_INPUT_FRAMES,
    RETAKE_BLEND_SECONDS,
    feathered_mask_times,
    window_for_retake,
)


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


def test_feather_pulls_the_denoise_mask_into_kept_context() -> None:
    window = window_for_retake(
        source_frames=241,
        mask_start_frame=96,
        mask_end_frame=144,
        max_input_frames=MAX_INPUT_FRAMES,
    )
    fps = 24.0
    start, end = feathered_mask_times(window, fps)
    assert start == pytest.approx(96 / fps - MASK_DELTA_SECONDS)
    assert end == pytest.approx(144 / fps + MASK_DELTA_SECONDS)
    assert start >= 0
    assert end <= window.encode_duration(fps)


def test_feather_does_not_cross_a_missing_prefix() -> None:
    window = window_for_retake(
        source_frames=241,
        mask_start_frame=0,
        mask_end_frame=48,
        max_input_frames=MAX_INPUT_FRAMES,
    )
    start, end = feathered_mask_times(window, 24.0)
    assert start == 0.0
    assert end > 48 / 24.0
    assert RETAKE_BLEND_SECONDS == 0.3
