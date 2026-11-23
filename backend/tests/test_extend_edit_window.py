"""CPU-only checks for optional Home/Remote extend encode-window math.

GenSpace keeps the uncapped full-source path. Home/Remote passes
``max_input_frames=505`` so the GPU only encodes a context window.
"""

from __future__ import annotations

import pytest

from inference.retake_pipeline.window import (
    CONTINUATION_CONTEXT_FRAMES,
    VIDEO_JOB_TOO_LARGE,
    EditRefusal,
    MAX_INPUT_FRAMES,
    PreparedExtendWindow,
    context_cap_frames,
    continuation_context_frames,
    extend_context_floor,
    prepare_extend_window,
    MIN_EXTEND_CONTEXT_FRAMES,
    duration_to_extend_frames,
    extend_exceeds_cap,
    extend_too_long_message,
    max_extend_frames,
    window_for_extend,
)


def test_min_context_is_8k1_aligned() -> None:
    """121 survives the VAE snap exactly, so the floor is exact, not ~."""
    assert (MIN_EXTEND_CONTEXT_FRAMES - 1) % 8 == 0


def test_max_extend_leaves_exactly_min_context() -> None:
    assert max_extend_frames(505) == 384
    assert not extend_exceeds_cap(384, 505)
    assert extend_exceeds_cap(385, 505)


def test_extend_at_max_keeps_full_min_context() -> None:
    window = window_for_extend(
        source_frames=1001,
        extend_frames=max_extend_frames(MAX_INPUT_FRAMES),
        mode="end",
        max_input_frames=MAX_INPUT_FRAMES,
    )
    assert window.context_frames == MIN_EXTEND_CONTEXT_FRAMES


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


def test_extend_too_long_at_hard_cap_is_machine_neutral() -> None:
    """The 121-context rule binds on every machine; the copy must not blame this one."""
    message = extend_too_long_message(
        duration_s=17.0, target_width=1280, target_height=720,
        frame_cap=MAX_INPUT_FRAMES, fps=24.0,
    )
    assert message == "Extend of 17s is too long for 1280x720 (at most 16s)."


def test_extend_too_long_at_shortened_cap_names_the_machine() -> None:
    message = extend_too_long_message(
        duration_s=10.0, target_width=1920, target_height=1080,
        frame_cap=300, fps=24.0,
    )
    assert message == "Extend of 10s is too long for 1920x1080 on this machine (at most 7s)."


def _prepare(context_mode: str, *, frame_cap: int, source_frames: int = 144, mode: str = "end"):
    return prepare_extend_window(
        fps=24.0,
        source_frames=source_frames,
        target_width=1280,
        target_height=720,
        frame_cap=frame_cap,
        duration=2.0,
        mode=mode,  # type: ignore[arg-type]
        context_mode=context_mode,  # type: ignore[arg-type]
    )


def test_short_context_is_admitted_where_long_context_is_refused() -> None:
    # Bug: Short context keeps the 121 frame admission floor, so the shorter window it needs
    # never gets a chance on a machine with a small frame cap.
    refusal = _prepare("long", frame_cap=100)
    assert isinstance(refusal, EditRefusal)
    assert refusal.code == VIDEO_JOB_TOO_LARGE
    prepared = _prepare("short", frame_cap=100)
    assert isinstance(prepared, PreparedExtendWindow)


@pytest.mark.parametrize("mode", ["end", "start"])
def test_short_context_is_capped_and_long_context_is_not(mode: str) -> None:
    short = _prepare("short", frame_cap=MAX_INPUT_FRAMES, mode=mode)
    long = _prepare("long", frame_cap=MAX_INPUT_FRAMES, mode=mode)
    assert isinstance(short, PreparedExtendWindow) and isinstance(long, PreparedExtendWindow)
    assert short.window.context_frames == CONTINUATION_CONTEXT_FRAMES
    # 144 source frames snap down to the 8k+1 length 137. Long mode keeps all of it.
    assert long.window.context_frames == 137


@pytest.mark.parametrize(
    ("fps", "expected"), [(24.0, 49), (25.0, 49), (30.0, 57), (50.0, 97), (60.0, 121)]
)
def test_continuation_context_is_about_two_seconds_at_any_frame_rate(fps: float, expected: int) -> None:
    # Bug: a fixed 49 frames is 1.6 s at 30 fps and 0.8 s at 60 fps, too short for the voice bridge.
    frames = continuation_context_frames(fps)
    assert frames == expected
    assert (frames - 1) % 8 == 0
    assert 1.85 <= frames / fps <= 2.3


def test_short_context_window_follows_the_frame_rate() -> None:
    prepared = prepare_extend_window(
        fps=30.0,
        source_frames=180,
        target_width=1920,
        target_height=1088,
        frame_cap=MAX_INPUT_FRAMES,
        duration=2.0,
        mode="end",
        context_mode="short",
    )
    assert isinstance(prepared, PreparedExtendWindow)
    assert prepared.window.context_frames == continuation_context_frames(30.0) == 57


def test_the_context_mode_sets_the_cap_and_the_admission_floor() -> None:
    assert context_cap_frames("long", 24.0) is None
    assert context_cap_frames("short", 24.0) == CONTINUATION_CONTEXT_FRAMES
    assert extend_context_floor("long", 24.0) == MIN_EXTEND_CONTEXT_FRAMES
    # Short keeps less than the Long floor, so its floor is lowered to what it keeps.
    assert extend_context_floor("short", 24.0) == CONTINUATION_CONTEXT_FRAMES
    # An explicit floor (the shrink retry passes 1) is never raised.
    assert extend_context_floor("short", 24.0, 1) == 1
