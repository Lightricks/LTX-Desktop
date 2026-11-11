"""Mac Fast windowed price at 1440p and 4K (``decide_video_job`` on the Darwin route).

Each test names the bug it guards. The Mac law prices the smallest allocator cap that still
passes, so the reference points are capped runs: at 4K a cap of 22.7 GiB failed and 23.4
passed for two windows (73f), and 28.4 failed and 29.2 passed for seven (241f).
"""

from __future__ import annotations

import pytest

from runtime_config.budget_admission import Admitted, RiskAllowance
from runtime_config.fast_chunking import window_candidates
from runtime_config.measured_cells import MEASURED_HIGH_RES_CELLS
from runtime_config.video_job_budget import (
    NOMINAL_MPS_POOL_RATIO,
    VideoJobLoad,
    VideoJobReject,
    _mac_fast_stream_required_gib,
    _windows_fast_reserved_required_gib,
    decide_video_job,
    fast_job_stream_gib,
    video_tokens,
)
from tests.test_video_job_budget import _MAC_HOST_GIB, _cap, _job_filter, _mac_profile

_4K = (3840, 2176)
_1440P = (2560, 1472)
_1440P_9_16 = (1472, 2624)
_1440P_21_9 = (3456, 1472)
_4K_WINDOW_TOKENS = video_tokens(_4K[1], _4K[0], 49)
_1440P_WINDOW_TOKENS = video_tokens(_1440P[1], _1440P[0], 129)


def _decide(size: tuple[int, int], frames: int, pool_gib: float):
    job_filter = _job_filter(
        process_mode="streaming_models_loading", memory_gb=None, **_mac_profile(pool_gib=pool_gib)
    )
    admission = decide_video_job(
        *size, frames, job_filter=job_filter, allowance=RiskAllowance.none()
    )
    return admission.load if isinstance(admission, Admitted) else admission.reject


def _mac_ask(tokens: float, windows: int) -> float:
    return fast_job_stream_gib(tokens, windows=windows, darwin=True)


def test_mac_plateau_ask_covers_the_capped_4k_edge() -> None:
    # A cap of 28.4 GiB failed a 7-window 4K job and 29.2 passed (29.07 peak, capbound=n).
    # The fixed 5.7 asked 28.39, and the Windows coefficient put on the Mac asks 28.69:
    # both admit a pool that fails or under-asks the 29.2 passing cap.
    assert _mac_ask(_4K_WINDOW_TOKENS, 7) >= 29.2
    assert _mac_ask(_4K_WINDOW_TOKENS, 7) == _mac_ask(_4K_WINDOW_TOKENS, 3)
    assert isinstance(_decide(_4K, 241, _cap(29.3)), VideoJobLoad)


def test_mac_two_window_ask_pays_the_second_window_step() -> None:
    # A cap of 22.7 GiB failed the 73-frame 4K job (2 windows: 49f then 41f) and 24.0 passed.
    # Pricing 2 windows from last_window_frames must keep 73f at or above the 24.0 passing cap
    # while fitting a 24.96 GiB MPS budget, and keep the full-last-window 1440p 241f job at
    # 29.52 GiB (>= 29.25 passing cap).
    short_4k = _decide(_4K, 73, _cap(24.96))
    assert isinstance(short_4k, VideoJobLoad)
    assert short_4k.chunk_pixel_frames == 49
    assert short_4k.stream_gib - _MAC_HOST_GIB >= 24.0
    refused = _decide(_4K, 73, _cap(22.7))
    assert isinstance(refused, VideoJobReject)
    assert refused.reason == "stream_over"

    short_1440p = _decide(_1440P, 145, _cap(24.96))
    assert isinstance(short_1440p, VideoJobLoad)
    assert short_1440p.chunk_pixel_frames == 129
    assert short_1440p.stream_gib == pytest.approx(_mac_ask(_1440P_WINDOW_TOKENS, 1) + _MAC_HOST_GIB)
    assert _mac_ask(_1440P_WINDOW_TOKENS, 2) >= _mac_ask(_1440P_WINDOW_TOKENS, 1) + 3.72


def test_mac_windowed_extra_scales_with_window_tokens() -> None:
    # A fixed Mac extra asks the same for a 41-frame window as for a 129-frame one, and
    # under-asks the larger 1440p window against the 4K one.
    small = video_tokens(_1440P[1], _1440P[0], 41)

    def extra(tokens: float) -> float:
        return _mac_ask(tokens, 3) - _mac_ask(tokens, 1)

    assert extra(small) < extra(_4K_WINDOW_TOKENS) < extra(_1440P_WINDOW_TOKENS)
    assert extra(_1440P_WINDOW_TOKENS) / _1440P_WINDOW_TOKENS == pytest.approx(
        extra(_4K_WINDOW_TOKENS) / _4K_WINDOW_TOKENS
    )


def test_mac_single_window_ask_is_the_base_law() -> None:
    # The extra starts at the second window. Applied at one window it re-prices every
    # single-pass cell on the Mac.
    for tokens in (_4K_WINDOW_TOKENS, _1440P_WINDOW_TOKENS):
        assert _mac_ask(tokens, 1) == _mac_fast_stream_required_gib(tokens)


def test_windows_and_mac_windowed_coefficients_stay_separate() -> None:
    # One shared constant or branch would move both routes when either is refit.
    tokens = _4K_WINDOW_TOKENS
    for windows, mac_coefficient, windows_coefficient in ((2, 8.0e-5, 6.5e-5), (3, 1.15e-4, 1.05e-4)):
        mac_extra = _mac_ask(tokens, windows) - _mac_fast_stream_required_gib(tokens)
        assert mac_extra == pytest.approx(mac_coefficient * tokens)
        windows_extra = fast_job_stream_gib(
            tokens, windows=windows, darwin=False
        ) - _windows_fast_reserved_required_gib(tokens)
        assert windows_extra == pytest.approx(windows_coefficient * tokens)


def test_mac_fast_4k_row_loads_at_7_windows_and_refuses_at_8() -> None:
    # 241 frames is 7 windows of 49, 249 is 8. The row is the longest completed run: a row
    # on the wrong tier, feature or route, or off by one, moves this boundary.
    assert MEASURED_HIGH_RES_CELLS[("fast", "mac_measured", "2160p")] == 7
    at_row = _decide(_4K, 241, 107.52)
    assert isinstance(at_row, VideoJobLoad)
    assert at_row.chunk_pixel_frames == 49
    past_row = _decide(_4K, 249, 107.52)
    assert isinstance(past_row, VideoJobReject)
    assert past_row.reason == "unmeasured_cell"
    assert ("iclora", "mac_measured", "2160p") not in MEASURED_HIGH_RES_CELLS


def test_mac_fast_1440p_row_loads_at_5_windows_and_refuses_at_6() -> None:
    # 481 frames is 5 windows of 129, 601 (12 s at 50 fps) is 6. A row on the wrong
    # tier, feature or route, or off by one, moves this boundary.
    assert MEASURED_HIGH_RES_CELLS[("fast", "mac_measured", "1440p")] == 5
    at_row = _decide(_1440P, 481, 107.52)
    assert isinstance(at_row, VideoJobLoad)
    assert at_row.chunk_pixel_frames == 129
    past_row = _decide(_1440P, 601, 107.52)
    assert isinstance(past_row, VideoJobReject)
    assert past_row.reason == "unmeasured_cell"


def test_mac_1440p_asks_cover_their_lowest_passing_caps() -> None:
    # Capped 1440p runs (129-frame windows) passed at 24.3 (1w), 29.25 (2w) and 31.7 GiB
    # (5w) and failed at 23.3, 28.0 and 30.7. A coefficient drop (1.15e-4 -> 1.10e-4 or
    # 8.0e-5 -> 7.0e-5) puts an ask under a measured passing cap.
    assert _mac_ask(_1440P_WINDOW_TOKENS, 1) >= 24.3
    assert _mac_ask(_1440P_WINDOW_TOKENS, 2) >= 29.25
    assert _mac_ask(_1440P_WINDOW_TOKENS, 5) >= 31.7


def test_mac_1440p_parts_shorter_than_129_frames_are_offered_in_cautious() -> None:
    # Every Mac 1440p run used 129-frame parts. A 32 GB Mac steps 1440p 10 s (241 frames)
    # down to shorter parts, and the longest 9:16 and 21:9 parts are shorter than 129 on
    # every Mac. A 129-frame floor stops the 32 GB Mac at 8 s of 1440p, 9:16 at 5 s and
    # 21:9 at 3 s (one pass) even on a 48 GB Mac.
    mac_32_gb = _decide(_1440P, 241, 32 * NOMINAL_MPS_POOL_RATIO)
    assert isinstance(mac_32_gb, VideoJobLoad)
    assert mac_32_gb.chunk_pixel_frames is not None and mac_32_gb.chunk_pixel_frames < 129
    for size, frames in ((_1440P_9_16, 481), (_1440P_21_9, 241)):
        loaded = _decide(size, frames, 48 * NOMINAL_MPS_POOL_RATIO)
        assert isinstance(loaded, VideoJobLoad)
        assert loaded.chunk_pixel_frames == window_candidates(*size)[0] < 129


def test_mac_1440p_row_stays_off_other_features() -> None:
    # Adding the Mac 1440p row under the wrong feature key would open unmeasured IC-LoRA
    # or edit cells on Darwin.
    assert ("iclora", "mac_measured", "1440p") not in MEASURED_HIGH_RES_CELLS
    assert ("edit", "mac_measured", "1440p") not in MEASURED_HIGH_RES_CELLS
