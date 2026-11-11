"""The specs response carries what the Home memory meter draws: one ask per cell and the capacity."""

from __future__ import annotations

from collections import Counter

from api_model_specs import build_generate_video_model_specs_response
from frame_math import compute_num_frames
from runtime_config.budget_admission import PricedJob
from runtime_config.budget_decision import VideoJobLoad
from runtime_config.budget_evidence import Extrapolated, JobShape
from runtime_config.ltx_capabilities import budget_size, local_canvas
from runtime_config.memory_meter import estimate_band_gib, meter_confidence
from runtime_config.model_download_specs import get_ltx_model_spec
from runtime_config.video_job_budget import LocalJobFilter, decide_video_job
from tests.test_video_job_budget import _job_filter, _mac_profile, _windows_measured_filter

_MODEL = "ltx-2.5-22b-distilled"


def _spec(job_filter: LocalJobFilter):
    return build_generate_video_model_specs_response(_MODEL, job_filter=job_filter).local_models[0].spec


def _ask(job_filter: LocalJobFilter, resolution: str, fps: int, duration: int):
    asks = _spec(job_filter).cell_asks or []
    return next(
        a
        for a in asks
        if (a.resolution, a.fps, a.duration, a.audio) == (resolution, fps, duration, False)
    )


def test_a_legacy_route_1440p_cell_tagged_untested_is_rough_not_solid() -> None:
    # The legacy curve tags 1440p unmeasured before it checks the route; its number is still rough.
    legacy = _job_filter(memory_gb=24, process_mode="streaming_models_loading")
    asks = _spec(legacy).cell_asks or []
    high_res = [a for a in asks if a.resolution in ("1440p", "2160p")]
    assert high_res
    assert {a.confidence for a in high_res} == {"rough"}


def test_an_untested_job_in_parts_is_solid_not_an_estimate() -> None:
    # Mac 1440p past 5 windows is an unmeasured cell, but its peak plateaus from window 3.
    mac = _job_filter(memory_gb=96, process_mode="full_models_loading", **_mac_profile(pool_gib=48))
    ask = _ask(mac, "1440p", 48, 20)
    assert ask.windows == 9
    assert ask.confidence == "solid"
    assert ask.band_low_gib is None and ask.band_high_gib is None


def test_an_untested_single_pass_is_an_estimate_with_a_band_around_its_ask() -> None:
    priced = PricedJob(
        feature="fast",
        decision=VideoJobLoad(mode="streaming_models_loading", seq=75_480, full_gib=0, stream_gib=28.84),
        evidence=Extrapolated(basis="high_res_cell"),
        shape=JobShape(
            feature="fast",
            model_id=_MODEL,
            adapter_id="",
            stage_mode="",
            tiled=False,
            chunked=False,
            seq=75_480,
            canvas_px=1920 * 1088,
            windows=1,
        ),
    )
    for route in ("windows_measured", "mac_measured"):
        assert meter_confidence(priced, route=route) == "estimate"
        band = estimate_band_gib(28.84, route=route)
        assert band is not None
        low, high = band
        assert low < 28.84 < high


def test_an_ask_is_never_rounded_below_the_figure_the_gate_compared() -> None:
    win16 = _windows_measured_filter(16, 64)
    width, height = budget_size(local_canvas("1080p", mode="video"))
    gate = decide_video_job(width, height, compute_num_frames(6, 24), job_filter=win16)
    stream_gib = gate.priced.decision.stream_gib
    ask = _ask(win16, "1080p", 24, 6)
    assert stream_gib <= ask.ask_gib < stream_gib + 0.001


def test_every_static_cell_has_exactly_one_ask_including_withheld_cells() -> None:
    # A 16 GB card withholds most long cells; a hover preview still needs their asks.
    win16 = _windows_measured_filter(16, 64)
    static = get_ltx_model_spec(_MODEL).supported_pipelines[0][1]
    expected = Counter(
        (resolution, int(fps), duration, audio)
        for audio, grid in (
            (False, static.supported_resolutions_durations),
            (True, static.a2v_supported_resolutions_durations or {}),
        )
        for resolution, cell in grid.items()
        for fps, durations in cell.fps_to_durations.items()
        for duration in durations
    )
    spec = _spec(win16)
    assert spec.dropped_cells, "the card must withhold cells for this to test anything"
    for asks in (spec.cell_asks or [], (spec.aspect_specs or {})["1:1"].cell_asks):
        listed = Counter((a.resolution, int(a.fps), a.duration, a.audio) for a in asks)
        assert listed == expected


def test_mac_capacity_is_the_mps_pool_not_total_ram() -> None:
    mac = _job_filter(memory_gb=96, process_mode="full_models_loading", **_mac_profile(pool_gib=48))
    memory = build_generate_video_model_specs_response(_MODEL, job_filter=mac).memory
    assert memory is not None
    assert memory.device == "mac"
    assert memory.capacity_gib == 48


def test_uncalibrated_non_5090_gpu_omits_cell_estimated_seconds() -> None:
    win16 = _windows_measured_filter(16, 64)
    ask = _ask(win16, "540p", 24, 5)
    assert ask.estimated_seconds is None

