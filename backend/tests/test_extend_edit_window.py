"""CPU-only checks for optional Home/Remote extend encode-window math.

GenSpace keeps the uncapped full-source path. Home/Remote passes
``max_input_frames=505`` so the GPU only encodes a context window.
"""

from __future__ import annotations

import pytest

from services.retake_pipeline.window import (
    MAX_INPUT_FRAMES,
    duration_to_extend_frames,
    window_for_extend,
)


def test_uncapped_window_encodes_the_whole_source() -> None:
    window = window_for_extend(
        source_frames=241,
        extend_frames=96,
        mode="end",
    )
    assert window.context_frames == 241
    assert window.encode_start_frame == 0
    assert window.dropped_prefix_frames == 0
    assert window.dropped_suffix_frames == 0


def test_end_extend_drops_a_prefix_when_source_plus_pad_exceeds_cap() -> None:
    window = window_for_extend(
        source_frames=1001,
        extend_frames=96,
        mode="end",
        max_input_frames=MAX_INPUT_FRAMES,
    )
    assert window.context_frames + 96 <= MAX_INPUT_FRAMES
    assert (window.context_frames - 1) % 8 == 0
    assert window.dropped_prefix_frames == 1001 - window.context_frames
    assert window.encode_start_frame == window.dropped_prefix_frames
    assert window.dropped_suffix_frames == 0


def test_start_extend_drops_a_suffix_when_capped() -> None:
    window = window_for_extend(
        source_frames=1001,
        extend_frames=96,
        mode="start",
        max_input_frames=MAX_INPUT_FRAMES,
    )
    assert window.encode_start_frame == 0
    assert window.dropped_prefix_frames == 0
    assert window.dropped_suffix_frames == 1001 - window.context_frames
    assert window.context_frames + 96 <= MAX_INPUT_FRAMES


def test_short_source_under_the_cap_is_not_trimmed() -> None:
    window = window_for_extend(
        source_frames=97,
        extend_frames=96,
        mode="end",
        max_input_frames=MAX_INPUT_FRAMES,
    )
    assert window.context_frames == 97
    assert window.dropped_prefix_frames == 0


def test_end_extend_keeps_non_8k1_remainder_in_the_dropped_prefix() -> None:
    window = window_for_extend(
        source_frames=32,
        extend_frames=16,
        mode="end",
        max_input_frames=MAX_INPUT_FRAMES,
    )
    assert window.context_frames == 25
    assert window.dropped_prefix_frames == 7
    assert window.encode_start_frame == 7


def test_extend_that_fills_the_cap_is_rejected() -> None:
    with pytest.raises(ValueError, match="exceeds limit"):
        window_for_extend(
            source_frames=97,
            extend_frames=MAX_INPUT_FRAMES,
            mode="end",
            max_input_frames=MAX_INPUT_FRAMES,
        )


def test_duration_to_extend_frames_snaps_up_to_multiple_of_8() -> None:
    assert duration_to_extend_frames(3.0, 24.0) == 72
    assert duration_to_extend_frames(3.1, 24.0) == 80
    assert duration_to_extend_frames(4.0, 24.0) == 96
    assert duration_to_extend_frames(4.0, 24.0) % 8 == 0
