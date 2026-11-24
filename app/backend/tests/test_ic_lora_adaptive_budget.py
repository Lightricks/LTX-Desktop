"""The job budget chooses the window of an adaptive LoRA (Refine Details) before the job starts."""

from __future__ import annotations

import pytest

from runtime_config.ic_lora_high_res import high_res_tiling
from runtime_config.ic_lora_job_budget import (
    WINDOW_OVER_CAP_MESSAGE,
    IcLoraJobLoad,
    IcLoraJobReject,
    decide_ic_lora_job,
    dropped_duration_message_and_code,
)
from runtime_config.ic_lora_stage_mode import IcLoraStageMode
from runtime_config.ic_lora_tiling import IcLoraTiling, with_tiling_overrides
from tests.test_video_job_budget import _windows_measured_filter

_BASE = IcLoraTiling(long_side=1024, short_side=576, adaptive_window=True)
_270P = (480, 256)
_720P = (1280, 704)
_1080P = (1920, 1088)
_4K = (3840, 2112)


def _decide(size: tuple[int, int], frames: int, vram_gib: float, tiling: IcLoraTiling = _BASE):
    return decide_ic_lora_job(
        *size,
        frames,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(vram_gib, 64),
        tiling=tiling,
    ).priced.decision


def _window(decision: object) -> int | None:
    assert isinstance(decision, IcLoraJobLoad), decision
    return decision.chunk_pixel_frames


@pytest.mark.parametrize("size", [_270P, _720P])
def test_a_low_resolution_clip_runs_in_one_pass_when_the_budget_admits_it(size: tuple[int, int]) -> None:
    # The former 73-frame window chunked a 270p clip longer than 73 frames.
    assert _window(_decide(size, 241, 24)) is None


def test_a_low_resolution_clip_takes_a_window_above_97_when_one_pass_does_not_fit() -> None:
    window = _window(_decide(_720P, 241, 14))
    assert window is not None
    assert 97 < window < 241
    assert (window - 1) % 8 == 0


def test_1080p_runs_a_short_clip_in_one_pass_and_a_long_one_in_97_frame_windows() -> None:
    assert _window(_decide(_1080P, 49, 24)) is None
    assert _window(_decide(_1080P, 241, 24)) == 97


def test_4k_never_passes_73_frames_and_the_decode_check_can_go_lower() -> None:
    assert _window(_decide(_4K, 137, 48, high_res_tiling(_BASE))) == 73
    assert _window(_decide(_4K, 137, 31.84, high_res_tiling(_BASE))) == 49


def test_a_user_maximum_is_the_longest_window_the_walk_tries() -> None:
    capped = with_tiling_overrides(_BASE, overlap_fraction=None, carry_frames=None, max_window_frames=57)
    assert capped.max_window_frames == 57
    assert capped.window_frames is None  # a fixed window would chunk every clip, so it is not set
    assert _window(_decide(_1080P, 241, 24, capped)) == 57


def test_a_maximum_above_the_cap_of_the_resolution_is_refused_not_run_shorter() -> None:
    # The run's overrides come after the high-cell plan, as in ``_resolve_settings``.
    tiling = with_tiling_overrides(
        high_res_tiling(_BASE), overlap_fraction=None, carry_frames=None, max_window_frames=97
    )
    decision = _decide(_4K, 137, 48, tiling)
    assert isinstance(decision, IcLoraJobReject)
    assert decision.reason == "window_over_cap"
    message, _ = dropped_duration_message_and_code(decision, _windows_measured_filter(48, 64))
    assert message == WINDOW_OVER_CAP_MESSAGE
    # The same maximum runs at 1080p.
    assert _window(_decide(_1080P, 137, 24, tiling)) == 97


def test_an_adaptive_high_cell_stays_adaptive_and_gets_no_fixed_window() -> None:
    tiling = high_res_tiling(_BASE)
    assert tiling.adaptive_window
    assert tiling.window_frames is None
    assert tiling.carry_frames == 17


def test_a_maximum_must_be_more_than_twice_the_carry() -> None:
    with pytest.raises(ValueError, match="more than twice the carry"):
        with_tiling_overrides(_BASE, overlap_fraction=None, carry_frames=33, max_window_frames=57)


def test_a_maximum_above_97_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot exceed 97"):
        with_tiling_overrides(_BASE, overlap_fraction=None, carry_frames=None, max_window_frames=105)
