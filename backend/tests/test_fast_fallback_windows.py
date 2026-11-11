"""Fast runs in windows when one pass cannot fit the machine, at any canvas."""

from __future__ import annotations

from dataclasses import replace
from itertools import product

import pytest

import runtime_config.video_job_budget as budget
from runtime_config.budget_admission import Admission, Admitted, Refused, RiskAllowance
from runtime_config.budget_evidence import Extrapolated, Fitted
from runtime_config.device_calibration import DeviceCalibration, FrontierPoint, envelope_for
from runtime_config.fast_chunking import (
    CARRY_FRAMES,
    MIN_WINDOW_FRAMES,
    is_chunked_canvas,
    job_window_count,
    window_candidates,
)
from runtime_config.measured_cells import (
    MEASURED_WINDOWED_BELOW_1440P_CELLS,
    is_unmeasured_windowed_below_1440p,
)
from runtime_config.video_job_budget import (
    NOMINAL_MPS_POOL_RATIO,
    LocalJobFilter,
    VideoJobDecision,
    VideoJobLoad,
    VideoJobReject,
    decide_video_job,
    price_video_job,
    video_job_reject,
)
from tests.test_video_job_budget import _job_filter, _mac_profile, _windows_measured_filter

_1080P = (1920, 1088)
_1080P_21_9 = (2560, 1088)
_3_2 = (1632, 1088)  # 1.78 Mpx: between 720p and 1080p, so no pixel band names it
_720P = (1280, 704)
_540P = (1024, 576)
_1440P = (2560, 1472)
_RTX_5090_GIB = 31.84
_MAC_32_GB_POOL_GIB = 32 * NOMINAL_MPS_POOL_RATIO
_BRAVE = RiskAllowance.for_mode("brave")


def _admission(
    size: tuple[int, int],
    seconds: int,
    job_filter: LocalJobFilter,
    *,
    fps: int = 24,
    allowance: RiskAllowance | None = None,
) -> Admission:
    return decide_video_job(*size, seconds * fps + 1, job_filter=job_filter, allowance=allowance)


def _decision(admission: Admission) -> VideoJobDecision:
    return admission.load if isinstance(admission, Admitted) else video_job_reject(admission)


def _decide(
    size: tuple[int, int], seconds: int, vram_gib: float, fps: int = 24, free_gib: float | None = None
) -> VideoJobDecision:
    jf = _windows_measured_filter(vram_gib, 64, free_gib=free_gib)
    return _decision(_admission(size, seconds, jf, fps=fps))


def _single_pass(size: tuple[int, int], seconds: int, vram_gib: float) -> VideoJobDecision:
    """The verdict with the fallback off: what the job got before windows were tried."""
    return budget._decide_fast_pass(  # pyright: ignore[reportPrivateUsage]
        *size, seconds * 24 + 1, windows=1, job_filter=_windows_measured_filter(vram_gib, 64)
    )


def _mac_filter(pool_gib: float) -> LocalJobFilter:
    return _job_filter(memory_gb=pool_gib * 2, process_mode="full_models_loading", **_mac_profile(pool_gib=pool_gib))


def test_1080p_20s_runs_in_three_windows_on_the_measured_card() -> None:
    decision = _decide(_1080P, 20, _RTX_5090_GIB)
    assert isinstance(decision, VideoJobLoad)
    assert decision.chunk_pixel_frames == window_candidates(*_1080P)[0] == 241
    assert job_window_count(20 * 24 + 1, decision.chunk_pixel_frames) == 3


def test_the_rule_names_no_canvas() -> None:
    """A canvas no pixel band named runs in windows once one pass cannot fit the card."""
    one_pass = _single_pass(_3_2, 20, _RTX_5090_GIB)
    assert isinstance(one_pass, VideoJobReject) and one_pass.stream_gib > _RTX_5090_GIB
    decision = _decide(_3_2, 20, _RTX_5090_GIB)
    assert isinstance(decision, VideoJobLoad)
    assert decision.chunk_pixel_frames is not None
    assert not is_chunked_canvas(*_3_2)


@pytest.mark.parametrize("vram_gib", [16, 24, 28])
def test_a_card_below_the_measured_one_runs_its_windows_without_a_card_floor(vram_gib: float) -> None:
    """The row is route-wide: a 16 GiB card's 6 windows of 105 frames are as covered as the 5090's 3."""
    admitted = _admission(_1080P, 20, _windows_measured_filter(vram_gib, 64))
    assert isinstance(admitted, Admitted)
    assert admitted.load.chunk_pixel_frames is not None
    assert isinstance(admitted.priced.evidence, Fitted)


def test_a_card_whose_windows_are_shorter_runs_more_of_them_and_passes_the_row() -> None:
    """A 12 GiB card windows 1080p 20 s at 57 frames: 12 windows, past the row's 9."""
    jf = _windows_measured_filter(12, 64)
    refused = _admission(_1080P, 20, jf)
    assert isinstance(refused, Refused)
    assert refused.reject.unmeasured_basis == "windowed_range"
    window = refused.reject.chunk_pixel_frames
    assert window is not None
    assert job_window_count(20 * 24 + 1, window) > MEASURED_WINDOWED_BELOW_1440P_CELLS[("fast", "windows_measured")].max_windows

    admitted = _admission(_1080P, 20, jf, allowance=_BRAVE)
    assert isinstance(admitted, Admitted)
    assert admitted.load.chunk_pixel_frames == window
    assert isinstance(admitted.priced.evidence, Extrapolated)


@pytest.mark.parametrize("free_gib", [22.0, 20.0, 16.0])  # one pass of 1080p 10 s asks 22.7 GiB
def test_a_busy_card_keeps_the_single_pass_refusal(free_gib: float) -> None:
    """One pass fits an empty card, so a busy or warm card refuses it. Evict-retry and the
    close-other-apps message stay in charge: windows must not hide a shortfall that frees up."""
    decision = _decide(_1080P, 10, _RTX_5090_GIB, free_gib=free_gib)
    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "stream_over"
    assert decision.chunk_pixel_frames is None


def test_a_job_inside_the_measured_window_count_is_fitted_not_pinned_to_three_windows() -> None:
    """20 s at 48 fps is 5 windows of 241. A row that stayed at the 1080p run's 3 would refuse it."""
    jf = _windows_measured_filter(_RTX_5090_GIB, 64)
    admitted = _admission(_1080P, 20, jf, fps=48)
    assert isinstance(admitted, Admitted)
    assert job_window_count(20 * 48 + 1, admitted.load.chunk_pixel_frames) == 5
    assert isinstance(admitted.priced.evidence, Fitted)


def test_a_20_second_540p_48_fps_clip_is_one_pass_only_where_the_mode_takes_the_bet() -> None:
    """69.7k tokens is past the 63,240-token fit. Brave runs the one pass with an untested note,
    and Cautious windows it at the ceiling (2 windows) instead of refusing."""
    jf = _windows_measured_filter(_RTX_5090_GIB, 64)
    brave = _admission(_540P, 20, jf, fps=48, allowance=_BRAVE)
    assert isinstance(brave, Admitted)
    assert brave.load.chunk_pixel_frames is None
    assert isinstance(brave.priced.evidence, Extrapolated)
    assert brave.priced.evidence.basis == "pass_tokens"

    cautious = _admission(_540P, 20, jf, fps=48)
    assert isinstance(cautious, Admitted)
    assert cautious.load.chunk_pixel_frames == window_candidates(*_540P)[0]
    assert job_window_count(20 * 48 + 1, cautious.load.chunk_pixel_frames) == 2
    assert isinstance(cautious.priced.evidence, Fitted)


def test_a_clean_brave_one_pass_run_opens_that_pass_in_cautious_on_this_card() -> None:
    """The device frontier is what proves a pass past the fit: after one clean run, Cautious runs it."""
    jf = _windows_measured_filter(_RTX_5090_GIB, 64)
    ran = price_video_job(*_540P, 20 * 48 + 1, job_filter=jf, allowance=_BRAVE)
    assert ran.chunk_pixel_frames is None
    envelope = envelope_for(ran.shape, ran.decision.mode, ran.stream_gib)
    proven = replace(
        jf,
        calibration=DeviceCalibration(
            identity=None,
            frontier={envelope.bucket: (FrontierPoint(point=envelope.point, peak_growth_bytes=20 * 1024**3),)},
        ),
    )

    admitted = _admission(_540P, 20, proven, fps=48)
    assert isinstance(admitted, Admitted)
    assert admitted.load.chunk_pixel_frames is None
    assert admitted.priced.covered

    longer = _admission(_540P, 20, proven, fps=50)
    assert isinstance(longer, Admitted)
    assert longer.load.chunk_pixel_frames == window_candidates(*_540P)[0]


def test_a_one_pass_job_inside_the_fit_stays_fitted_and_one_pass() -> None:
    """1080p 10 s is 63,240 tokens, the fit's own point: no note, no windows, in any mode."""
    jf = _windows_measured_filter(_RTX_5090_GIB, 64)
    admitted = _admission(_1080P, 10, jf)
    assert isinstance(admitted, Admitted)
    assert admitted.load.chunk_pixel_frames is None
    assert isinstance(admitted.priced.evidence, Fitted)


def test_the_window_count_does_not_raise_the_ask_past_the_plateau() -> None:
    """20 s at 48 fps runs 5 windows against 3 at 24 fps, and asks the same: the window is priced, not the clip."""
    jf = _windows_measured_filter(_RTX_5090_GIB, 64)
    three = _decision(_admission(_1080P, 20, jf))
    five = _decision(_admission(_1080P, 20, jf, fps=48))
    assert job_window_count(20 * 24 + 1, three.chunk_pixel_frames) == 3
    assert job_window_count(20 * 48 + 1, five.chunk_pixel_frames) == 5
    assert three.stream_gib == five.stream_gib
    assert three.chunk_pixel_frames == five.chunk_pixel_frames


def test_a_card_no_window_fits_refuses_with_the_shortest_window_not_the_whole_clip() -> None:
    """A Brave downgrade runs this window, so it must be the smallest ask and not one whole-clip pass."""
    decision = _decide(_540P, 20, 6.0)
    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "stream_over"
    assert decision.chunk_pixel_frames == MIN_WINDOW_FRAMES
    assert decision.stream_gib < _single_pass(_540P, 20, 6.0).stream_gib


@pytest.mark.parametrize("pool_gib", [16, 32, 64, 96])
def test_mac_runs_a_1080p_48_fps_clip_in_windows_past_the_sequence_fence(pool_gib: float) -> None:
    """246k tokens are far past the 116k-token MPS fence in one pass, and every window sits under it."""
    jf = _mac_filter(pool_gib)
    one_pass = budget._decide_fast_pass(  # pyright: ignore[reportPrivateUsage]
        *_1080P, 20 * 48 + 1, windows=1, job_filter=jf
    )
    assert isinstance(one_pass, VideoJobReject) and one_pass.reason == "mps_sequence_over"

    admitted = _admission(_1080P, 20, jf, fps=48, allowance=_BRAVE)
    assert isinstance(admitted, Admitted)
    assert admitted.load.chunk_pixel_frames is not None

    cautious = _admission(_1080P, 20, jf, fps=48)
    windows = job_window_count(20 * 48 + 1, admitted.load.chunk_pixel_frames)
    if windows <= MEASURED_WINDOWED_BELOW_1440P_CELLS[("fast", "mac_measured")].max_windows:
        assert isinstance(cautious, Admitted) and isinstance(cautious.priced.evidence, Fitted)
    else:
        assert isinstance(cautious, Refused) and cautious.reject.reason == "unmeasured_cell"


@pytest.mark.parametrize(("pool_gib", "seconds"), [(16, 10), (32, 20)])  # asks 24.7 and 49 GiB in one pass
def test_mac_windows_a_job_whose_ask_is_over_the_pool(pool_gib: float, seconds: int) -> None:
    admission = _admission(_1080P, seconds, _mac_filter(pool_gib), allowance=_BRAVE)
    assert isinstance(admission, Admitted)
    assert admission.load.chunk_pixel_frames is not None


def test_mac_parts_smaller_than_the_measured_runs_are_offered_in_cautious() -> None:
    """A 32 GB Mac steps 1080p 20 s down to parts shorter than the longest 1080p part, the size
    the Mac runs used. A floor at the measured part size stops that Mac at 10 s of 1080p."""
    admitted = _admission(_1080P, 20, _mac_filter(_MAC_32_GB_POOL_GIB))
    assert isinstance(admitted, Admitted)
    assert admitted.load.chunk_pixel_frames is not None
    assert admitted.load.chunk_pixel_frames < window_candidates(*_1080P)[0]
    assert isinstance(admitted.priced.evidence, Fitted)


def test_mac_jobs_past_five_parts_are_still_withheld_in_cautious() -> None:
    """The same part on the same Mac: five of them run, and one more is past the longest Mac run."""
    jf = _mac_filter(_MAC_32_GB_POOL_GIB)
    window = _decision(_admission(_1080P, 20, jf)).chunk_pixel_frames
    assert window is not None
    five_frames = window + 4 * (window - CARRY_FRAMES)
    six_frames = five_frames + 8
    five = decide_video_job(*_1080P, five_frames, job_filter=jf)
    assert isinstance(five, Admitted)
    assert five.load.chunk_pixel_frames == window
    assert job_window_count(five_frames, window) == 5

    six = decide_video_job(*_1080P, six_frames, job_filter=jf)
    assert isinstance(six, Refused)
    assert six.reject.unmeasured_basis == "windowed_range"
    assert six.reject.chunk_pixel_frames == window
    assert job_window_count(six_frames, window) == 6

    brave = decide_video_job(*_1080P, six_frames, job_filter=jf, allowance=_BRAVE)
    assert isinstance(brave, Admitted)
    assert isinstance(brave.priced.evidence, Extrapolated)


def test_the_unmeasured_window_cover_follows_the_route_row() -> None:
    windows_cover = MEASURED_WINDOWED_BELOW_1440P_CELLS[("fast", "windows_measured")]
    assert windows_cover.max_windows == 9
    kwargs = {"window_tokens": 20_000.0}
    assert not is_unmeasured_windowed_below_1440p("fast", "windows_measured", *_720P, windows=9, **kwargs)
    assert is_unmeasured_windowed_below_1440p("fast", "windows_measured", *_720P, windows=10, **kwargs)
    mac_cover = MEASURED_WINDOWED_BELOW_1440P_CELLS[("fast", "mac_measured")]
    assert not is_unmeasured_windowed_below_1440p(
        "fast", "mac_measured", *_1080P, windows=mac_cover.max_windows, window_tokens=63_240.0
    )
    assert is_unmeasured_windowed_below_1440p(
        "fast", "mac_measured", *_1080P, windows=mac_cover.max_windows + 1, window_tokens=63_240.0
    )
    # One pass prices on the fitted rows, and 1440p and 4K have their own rows.
    assert not is_unmeasured_windowed_below_1440p("fast", "mac_measured", *_1080P, windows=1, **kwargs)
    assert not is_unmeasured_windowed_below_1440p("fast", "windows_measured", *_1440P, windows=9, **kwargs)
    assert is_unmeasured_windowed_below_1440p("fast", "legacy", *_1080P, windows=2, **kwargs)


def test_no_job_that_loads_or_is_refused_for_another_reason_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fallback only turns a refusal no empty card could pass into a windowed verdict."""
    sizes = [_540P, _720P, _3_2, _1080P, _1080P_21_9, _1440P]
    cards = [(12, None), (24, None), (_RTX_5090_GIB, None), (48, None), (_RTX_5090_GIB, 20.0), (48, 30.0)]
    cases = list(product(sizes, (5, 8, 10, 20), (24, 48), cards))

    def run() -> dict[tuple, VideoJobDecision]:
        return {c: _decide(c[0], c[1], c[3][0], fps=c[2], free_gib=c[3][1]) for c in cases}

    after = run()
    monkeypatch.setattr(budget, "_windows_help", lambda *_a, **_k: False)
    before = run()

    changed = [c for c in cases if before[c] != after[c]]
    assert any(isinstance(after[c], VideoJobLoad) for c in changed), "a refused job should open"
    for c in changed:
        was, now = before[c], after[c]
        total = c[3][0]
        assert isinstance(was, VideoJobReject), c
        assert was.reason == "stream_over", c
        assert was.stream_gib > total, c  # no empty card of this size runs it in one pass
        assert not is_chunked_canvas(*c[0]), c
        assert now.chunk_pixel_frames is not None, c
        if isinstance(now, VideoJobReject):
            assert now.reason in {"unmeasured_cell", "stream_over"}, c


def test_a_refused_job_for_ram_keeps_its_refusal_instead_of_windowing() -> None:
    """A window does not shorten the pinned-RAM floor, so the RAM refusal stands."""
    jf = _windows_measured_filter(_RTX_5090_GIB, 8)
    decision = _decision(_admission(_540P, 20, jf))
    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "ram_shortfall"
    assert decision.chunk_pixel_frames is None
