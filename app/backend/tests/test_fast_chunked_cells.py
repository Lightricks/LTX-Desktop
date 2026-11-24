"""The Fast spec tells the pickers which kept cells run in overlapping windows."""

from __future__ import annotations

from api_model_specs import _filter_local_fast_resolution_map  # pyright: ignore[reportPrivateUsage]
from frame_math import compute_num_frames
from runtime_config.fast_chunking import job_window_count
from runtime_config.ltx_capabilities import local_caps
from runtime_config.model_download_specs import get_ltx_model_spec
from tests.test_video_job_budget import _job_filter, _mac_profile, _windows_measured_filter

_CAPS = local_caps("ltx-2.5-22b-distilled")
_SPEC = get_ltx_model_spec("ltx-2.5-22b-distilled").supported_pipelines[0][1]


def _plain(vram_gib: float):
    assert _SPEC.supported_resolutions_durations is not None
    return _filter_local_fast_resolution_map(
        _SPEC.supported_resolutions_durations, _CAPS, job_filter=_windows_measured_filter(vram_gib, 64)
    )


def test_1080p_20s_is_a_chunked_kept_cell_on_the_measured_card() -> None:
    kept, _dropped, _aspect_dropped, _warned, _aspect_warned, chunked = _plain(31.84)
    assert kept is not None
    cells = [c for c in chunked if (c.resolution, c.fps, c.duration) == ("1080p", 24, 20)]
    assert len(cells) == 1
    assert (cells[0].windows, cells[0].window_frames, cells[0].audio) == (3, 241, False)
    assert 20 in kept["1080p"].fps_to_durations[24]


def test_every_chunked_cell_is_kept_and_counts_its_windows() -> None:
    kept, *_rest, chunked = _plain(31.84)
    assert kept is not None
    assert chunked
    for cell in chunked:
        assert cell.duration in kept[cell.resolution].fps_to_durations[cell.fps], cell
        frames = compute_num_frames(cell.duration, int(cell.fps))
        assert cell.windows == job_window_count(frames, cell.window_frames), cell
        assert cell.windows >= 2, cell


def test_a_single_pass_cell_is_never_listed() -> None:
    *_rest, chunked = _plain(31.84)
    listed = {(c.resolution, c.fps, c.duration) for c in chunked}
    assert ("1080p", 24, 10) not in listed
    # Below 1080p only the cells one pass cannot take run in windows. No resolution is named: the
    # rule is the same one that windows 1080p. A 20 s 540p clip at 24 fps is 35k tokens, inside
    # the fit. At 48 fps it is 70k, past it, so Cautious takes it in windows.
    assert ("540p", 24, 20) not in listed
    assert ("540p", 48, 20) in listed
    assert {(c.fps, c.duration) for c in chunked if c.resolution == "720p" and c.duration <= 20} == {
        (48, 20),
        (50, 20),
    }


def test_a_card_below_the_measured_one_lists_its_1080p_windows_without_a_card_floor() -> None:
    kept, *_rest, chunked = _plain(24)
    assert kept is not None
    cell = next(c for c in chunked if (c.resolution, c.fps, c.duration) == ("1080p", 24, 20))
    assert (cell.windows, cell.window_frames) == (3, 185)
    assert 20 in kept["1080p"].fps_to_durations[24]


def test_the_a2v_grid_lists_its_own_cells_as_audio() -> None:
    assert _SPEC.a2v_supported_resolutions_durations is not None
    *_rest, chunked = _filter_local_fast_resolution_map(
        _SPEC.a2v_supported_resolutions_durations,
        _CAPS,
        job_filter=_windows_measured_filter(31.84, 64),
        mode="a2v",
        audio=True,
    )
    assert all(cell.audio for cell in chunked)


def test_a_refused_mac_1440p_cell_carries_the_window_a_run_would_use() -> None:
    # Mac 1440p past five windows is an unmeasured cell, so Cautious refuses it and Brave can
    # offer it. The refusal knows the windows that run would take, so a picker can say so.
    assert _SPEC.supported_resolutions_durations is not None
    jf = _job_filter(memory_gb=96, process_mode="full_models_loading", **_mac_profile(pool_gib=48))
    _kept, dropped, *_rest = _filter_local_fast_resolution_map(
        _SPEC.supported_resolutions_durations, _CAPS, job_filter=jf
    )
    cell = next(c for c in dropped if (c.resolution, c.fps, c.duration) == ("1440p", 48, 20))
    assert cell.reason == "unmeasured_cell"
    assert (cell.windows, cell.window_frames) == (9, 129)


def test_a_busy_card_refusal_carries_no_window() -> None:
    """A shortfall that frees up keeps its one-pass refusal, so no window is promised."""
    assert _SPEC.supported_resolutions_durations is not None
    _kept, dropped, *_rest = _filter_local_fast_resolution_map(
        _SPEC.supported_resolutions_durations, _CAPS, job_filter=_windows_measured_filter(31.84, 64, free_gib=16)
    )
    single_pass = [c for c in dropped if c.resolution in ("540p", "720p", "1080p") and c.fps == 24 and c.duration <= 10]
    assert single_pass
    assert all(c.windows is None and c.window_frames is None for c in single_pass)
