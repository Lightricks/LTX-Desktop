"""Window choice of ``decide_video_job`` for 1440p and 4K (Windows 5090-class budget).

Each test names the bug it guards. The VRAM values sit between the measured-law asks:
a 4K window of 49 frames asks 21.0 GiB and a window of 41 frames asks 18.7 GiB. Two
windows add 6.5e-5 GiB per window token and 3 or more add 1.05e-4.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from frame_math import compute_num_frames
from api_model_specs import get_local_video_generation_model_specs
from runtime_config.budget_admission import Admitted, RiskAllowance
from runtime_config.budget_decision import VideoJobLoad, VideoJobReject
from runtime_config.fast_chunking import is_chunked_canvas, last_window_frames, window_count
from runtime_config.ltx_capabilities import budget_shape, effective_local_caps
from runtime_config.measured_cells import MEASURED_HIGH_RES_CELLS, MEASURED_MAX_DECODE_TOKENS
from runtime_config.video_job_budget import (
    advertised_fast_durations,
    decide_edit_job,
    decide_video_job,
    fast_job_stream_gib,
    max_edit_frames_that_load,
    video_tokens,
)
from tests.test_video_job_budget import _windows_measured_filter

_4K = (3840, 2176)
_1440P = (2560, 1472)
_DURATIONS = [2, 3, 4, 5, 6, 8, 10, 20]


def _decide(size: tuple[int, int], frames: int, vram_gib: float):
    admission = decide_video_job(
        *size,
        frames,
        job_filter=_windows_measured_filter(vram_gib, 64),
        allowance=RiskAllowance.none(),
    )
    return admission.load if isinstance(admission, Admitted) else admission.reject


def test_clip_that_fits_one_window_is_not_chunked() -> None:
    decision = _decide(_4K, 49, 32)
    assert isinstance(decision, VideoJobLoad)
    assert decision.chunk_pixel_frames is None


def test_clip_over_one_window_takes_the_longest_window_not_the_shortest() -> None:
    decision = _decide(_4K, 73, 32)
    assert isinstance(decision, VideoJobLoad)
    assert decision.chunk_pixel_frames == 49


def test_card_too_small_for_the_long_window_falls_back_to_a_shorter_one() -> None:
    # 81 frames has a full 49-frame second window (24.7 GiB). A 24 GiB card rejects
    # the 49-frame windows and loads the 41-frame windows (23.8 GiB).
    decision = _decide(_4K, 81, 24)
    assert isinstance(decision, VideoJobLoad)
    assert decision.chunk_pixel_frames == 41


def test_short_second_window_prices_last_window_tokens_and_covers_measured_peaks() -> None:
    # G2a (4K 73f: 49f then 41f) peaked at 21.80 GiB and G2b (1440p 145f: 129f then 33f)
    # peaked at 21.88 GiB, while G1 (1440p 241f: 129f then 129f) peaked at 25.58 GiB.
    # Pricing every 2-window job as two full windows over-asks G2a (24.71) and G2b (26.58)
    # past a 24 GiB card; omitting max(base_law(t1), ...) under-asks G2b (14.22) below window 1.
    g2a = _decide(_4K, 73, 24)
    assert isinstance(g2a, VideoJobLoad)
    assert g2a.chunk_pixel_frames == 49
    assert 21.80 <= g2a.stream_gib < 24.0

    g2b = _decide(_1440P, 145, 24)
    assert isinstance(g2b, VideoJobLoad)
    assert g2b.chunk_pixel_frames == 129
    assert 21.88 <= g2b.stream_gib < 24.0

    g1 = _decide(_1440P, 241, 32)
    assert isinstance(g1, VideoJobLoad)
    assert g1.chunk_pixel_frames == 129
    assert last_window_frames(241, 129) == 129
    assert g1.stream_gib == pytest.approx(26.5832)


def test_three_windows_pay_a_larger_extra_than_two_windows() -> None:
    tokens = video_tokens(_4K[1], _4K[0], 49)
    one = fast_job_stream_gib(tokens, windows=1, darwin=False)
    two = fast_job_stream_gib(tokens, windows=2, darwin=False)
    three = fast_job_stream_gib(tokens, windows=3, darwin=False)
    assert one < two < three


def test_two_window_job_with_full_last_window_asks_the_second_window_step() -> None:
    # 1440p at 10 s and 24 fps is 129 frames then 112 more: window 2 peaked 25.6 GiB.
    # A +0 two-window price asked 22.5 and let a 24 GiB card admit a job that pages.
    tokens = video_tokens(_1440P[1], _1440P[0], 129)
    assert window_count(241, 129) == 2
    assert fast_job_stream_gib(tokens, windows=2, darwin=False) >= 25.6


def test_plateau_extra_scales_with_window_tokens_not_a_fixed_constant() -> None:
    # A fixed 5.7 asked 16.9 of 41-frame windows that peaked 13.04, refusing a 16 GB card's
    # job, and asked 28.2 of 1440p's 129-frame windows that peaked 28.31.
    small = fast_job_stream_gib(video_tokens(_1440P[1], _1440P[0], 41), windows=20, darwin=False)
    assert 13.04 <= small < 16.88
    large = fast_job_stream_gib(video_tokens(_1440P[1], _1440P[0], 129), windows=5, darwin=False)
    assert large >= 28.31


def test_extra_does_not_grow_with_window_count_past_three() -> None:
    # The peak is flat from window 3 to 20; a per-window term would price long jobs out.
    tokens = video_tokens(_4K[1], _4K[0], 49)
    three = fast_job_stream_gib(tokens, windows=3, darwin=False)
    assert fast_job_stream_gib(tokens, windows=20, darwin=False) == three


def test_fast_4k_row_loads_at_15_windows_and_refuses_at_16() -> None:
    # 497 frames is 15 windows of 49; 529 is 16. The row moved from 7 to 15 on the Fast key only.
    assert MEASURED_HIGH_RES_CELLS[("fast", "windows_measured", "2160p")] == 15
    at_row = _decide(_4K, 497, 64)
    assert isinstance(at_row, VideoJobLoad)
    assert window_count(497, 49) == 15
    past_row = _decide(_4K, 529, 64)
    assert isinstance(past_row, VideoJobReject)
    assert past_row.reason == "unmeasured_cell"


def test_fast_1440p_row_is_the_129_frame_window_count_not_the_41_frame_one() -> None:
    # 961 frames is 9 windows of 129 (measured); 1033 is 10. Opening the row to the 41-frame
    # window's 20 would admit 129-frame jobs of 10 to 20 windows that no run covers.
    assert MEASURED_HIGH_RES_CELLS[("fast", "windows_measured", "1440p")] == 9
    at_row = _decide(_1440P, 961, 64)
    assert isinstance(at_row, VideoJobLoad)
    assert at_row.chunk_pixel_frames == 129
    past_row = _decide(_1440P, 1033, 64)
    assert isinstance(past_row, VideoJobReject)
    assert past_row.reason == "unmeasured_cell"


def test_windows_edit_1440p_and_4k_rows_load_up_to_65_frames_and_mark_73_unmeasured() -> None:
    # 1440p Retake and Extend sit inside the 1080p reserved law through 193 frames (29.50 GiB
    # vs 29.93 ask), while 4K VAE decode is priced via _EDIT_DECODE_STREAM (21.57 GiB at
    # 49f, 26.83 GiB at 65f) and capped at 73,440 decode tokens (65f at 4K) where WDDM paging stays zero.
    assert MEASURED_HIGH_RES_CELLS[("edit", "windows_measured", "1440p")] == 1
    assert MEASURED_HIGH_RES_CELLS[("edit", "windows_measured", "2160p")] == 1
    assert MEASURED_MAX_DECODE_TOKENS[("edit", "windows_measured")] == 73_440.0
    job_filter = _windows_measured_filter(32, 64)
    edit_1440p = decide_edit_job(
        2560, 1440, 193, job_filter=job_filter, allowance=RiskAllowance.none()
    )
    assert isinstance(edit_1440p, Admitted)
    edit_4k_65 = decide_edit_job(
        *_4K, 65, job_filter=job_filter, allowance=RiskAllowance.none()
    )
    assert isinstance(edit_4k_65, Admitted)
    assert edit_4k_65.load.stream_gib == pytest.approx(3.15 + 3.222e-4 * 73_440)
    assert max_edit_frames_that_load(*_4K, frame_cap=505, job_filter=job_filter) == 65
    edit_4k_73 = decide_edit_job(
        *_4K, 73, job_filter=job_filter, allowance=RiskAllowance.none()
    )
    assert not isinstance(edit_4k_73, Admitted)
    assert edit_4k_73.reject.reason == "unmeasured_cell"


def test_windows_4k_edit_frame_cap_stops_at_measured_decode_ceiling() -> None:
    # A short 4K retake in Brave mode must not pad optional context past the 65-frame
    # (73,440-token) decode ceiling into the 73-frame WDDM paging zone, while an explicit
    # >65-frame selection in Brave mode can still reach the VRAM ceiling as an unmeasured bet.
    cautious_filter = replace(_windows_measured_filter(32, 64), budget_mode="cautious")
    brave_filter = replace(cautious_filter, budget_mode="brave")

    assert max_edit_frames_that_load(*_4K, frame_cap=505, job_filter=cautious_filter) == 65
    assert max_edit_frames_that_load(*_4K, frame_cap=505, job_filter=brave_filter) == 65
    assert (
        max_edit_frames_that_load(*_4K, frame_cap=505, min_frames=73, job_filter=cautious_filter)
        == 65
    )
    assert (
        max_edit_frames_that_load(*_4K, frame_cap=505, min_frames=73, job_filter=brave_filter)
        == 73
    )
    brave_73 = decide_edit_job(*_4K, 73, job_filter=brave_filter)
    assert isinstance(brave_73, Admitted)
    assert brave_73.risk is not None and brave_73.risk.tier == "unmeasured"


def test_narrow_1440p_canvases_use_chunked_budget_law_instead_of_single_pass_1080p_law() -> None:
    # 1440p 1:1 (1472x1472 = 2.17 Mpx), 4:5 (1472x1856 = 2.73 Mpx), and 4:3 (1984x1472 = 2.92 Mpx)
    # sit under 3.20 Mpx; gating is_chunked_canvas on pixel area instead of short side drops them
    # into the single-pass <=1080p path and refuses 20 s on a 24 GiB card.
    caps = effective_local_caps("ltx-2.5-22b-distilled", duration_head_ready=True)
    for aspect in ("1:1", "4:5", "4:3"):
        width, height = budget_shape(caps, "1440p", aspect, mode="stage2")
        assert is_chunked_canvas(width, height), aspect
        decision = _decide((width, height), compute_num_frames(20, 24), 24)
        assert isinstance(decision, VideoJobLoad), aspect
        assert decision.chunk_pixel_frames is not None, aspect


@pytest.mark.usefixtures("single_pass_only")
def test_narrower_aspect_ratios_unlock_longer_durations_than_16_9_anchor_on_same_vram() -> None:
    # At 1080p/24fps, 10 s at 1:1 (1088x1088, 35,880t -> 15.05 GiB) fits a 16 GiB card while
    # 16:9 (1920x1088, 63,240t -> 22.71 GiB) only fits up to 5 s (32,640t -> 14.14 GiB).
    specs = get_local_video_generation_model_specs(
        duration_head_ready=True,
        job_filter=_windows_measured_filter(16, 64),
    )
    fast_spec = specs[0].spec
    assert fast_spec.aspect_specs is not None
    durations_16_9 = fast_spec.aspect_specs["16:9"].supported_resolutions_durations["1080p"].fps_to_durations[24]
    durations_1_1 = fast_spec.aspect_specs["1:1"].supported_resolutions_durations["1080p"].fps_to_durations[24]
    assert 10 not in durations_16_9
    assert 10 in durations_1_1


def test_reject_names_the_ask_of_the_shortest_window() -> None:
    frames = 241
    decision = _decide(_4K, frames, 14)
    assert isinstance(decision, VideoJobReject)
    shortest = 41
    expected = fast_job_stream_gib(
        video_tokens(_4K[1], _4K[0], shortest),
        windows=window_count(frames, shortest),
        darwin=False,
    )
    assert decision.stream_gib == expected


def test_offered_durations_match_the_create_verdict_at_1440p_and_4k() -> None:
    for size, vram in ((_1440P, 16), (_4K, 24), (_4K, 28)):
        job_filter = _windows_measured_filter(vram, 64)
        offered = advertised_fast_durations(*size, 24, _DURATIONS, job_filter=job_filter)
        loads = [
            seconds
            for seconds in _DURATIONS
            if isinstance(
                decide_video_job(
                    *size, compute_num_frames(seconds, 24), job_filter=job_filter
                ),
                Admitted,
            )
        ]
        assert offered == loads
