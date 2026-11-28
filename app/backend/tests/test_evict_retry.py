"""Queued execute's one-shot evict-and-resample (W-band rejects)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from _routes._errors import HTTPError
from api_types import EstimateVideoRequest, GenerateVideoCompleteResponse, GenerateVideoRequest
from runtime_config.evict_retry import evict_retry_shortfall_gib
from runtime_config.video_job_budget import (
    VIDEO_JOB_TOO_LARGE,
    HardwareProfile,
    LocalJobFilter,
    MemoryBudget,
)

_GIB = 1024**3
_FAST_2_5 = "ltx-2.5-22b-distilled"


def _windows_filter(*, total_gib: float, free_gib: float | None) -> LocalJobFilter:
    return LocalJobFilter(
        profile=HardwareProfile(
            system="Windows",
            model_id=_FAST_2_5,
            process_mode="streaming_models_loading",
            drive_type=None,
        ),
        budget=MemoryBudget(
            total_vram_bytes=int(total_gib * _GIB),
            free_vram_bytes=None if free_gib is None else int(free_gib * _GIB),
            total_ram_bytes=64 * _GIB,
            memory_gb=total_gib,
            mps_pool_bytes=None,
        ),
    )


def _shortfall(
    *,
    route: str = "windows_measured",
    is_vram_occupancy_miss: bool = True,
    stream_gib: float = 20.5,
    total_gib: float = 24.0,
    free_gib: float | None = 19.0,
    live_gib: float | None = 4.5,
) -> float | None:
    return evict_retry_shortfall_gib(
        route=route,  # type: ignore[arg-type]
        is_vram_occupancy_miss=is_vram_occupancy_miss,
        stream_gib=stream_gib,
        job_filter=_windows_filter(total_gib=total_gib, free_gib=free_gib),
        live_allocated_bytes=None if live_gib is None else int(live_gib * _GIB),
    )


def test_evict_retry_declines_on_legacy_route() -> None:
    # Legacy curves compare a frozen integer, so a fresh sample cannot change the verdict.
    assert _shortfall(route="windows_measured") == pytest.approx(1.5)
    assert _shortfall(route="legacy") is None


def test_evict_retry_declines_when_ask_exceeds_total() -> None:
    assert _shortfall(stream_gib=24.0, live_gib=6.0) == pytest.approx(5.0)
    assert _shortfall(stream_gib=24.01, live_gib=6.0) is None


def test_evict_retry_boundary_at_live_weights() -> None:
    assert _shortfall(live_gib=1.5) == pytest.approx(1.5)
    assert _shortfall(live_gib=1.5 - 1 / 1024) is None


def test_evict_retry_fails_closed_without_probes() -> None:
    assert _shortfall(free_gib=None) is None
    assert _shortfall(live_gib=None) is None


def test_evict_retry_declines_non_occupancy_rejects() -> None:
    # IC-LoRA spatial/fence refusals still carry a stream ask over free; evicting cannot fix them.
    assert _shortfall(is_vram_occupancy_miss=True) == pytest.approx(1.5)
    assert _shortfall(is_vram_occupancy_miss=False) is None


_T2V = {"prompt": "test", "model": "fast", "fps": 24}


def _configure_windows_24gb(test_state) -> None:
    test_state.config.system = "Windows"
    test_state.config.vram_gb = 24
    test_state.config.vram_bytes = 24 * _GIB
    test_state.config.total_ram_bytes = 48 * _GIB
    test_state.config.local_generations_mode = "streaming_models_loading"
    test_state.state.app_settings.use_local_text_encoder = True


def _queued_t2v(test_state, out: Path, *, resolution: str, duration: int):
    shutdown = threading.Event()
    with test_state.generation.wait_for_generation_slot(shutdown):
        return test_state.video_generation.generate_local_reserved(
            GenerateVideoRequest.model_validate(
                {**_T2V, "resolution": resolution, "duration": duration}
            ),
            generation_id=f"{resolution}-{duration}",
            output_path=out,
            local_model_id=_FAST_2_5,
        )


def _warm_pipeline(test_state, fake_services, tmp_path: Path) -> None:
    fake_services.gpu_info.free_vram_bytes = 23 * _GIB
    warm = _queued_t2v(test_state, tmp_path / "warm.mp4", resolution="540p", duration=5)
    assert isinstance(warm, GenerateVideoCompleteResponse)
    assert test_state.state.gpu_slot is not None


def _resident_weights_probe(test_state) -> Callable[[], int]:
    return lambda: int(4.5 * _GIB) if test_state.state.gpu_slot is not None else 0


def test_queued_execute_evict_retry_admits_w_band_job(
    test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    # 720p/20s asks 20.03; warm weights leave 19.0 free, evicting them leaves 23.5.
    create_fake_model_files()
    _configure_windows_24gb(test_state)
    _warm_pipeline(test_state, fake_services, tmp_path)
    fake_services.gpu_info.free_vram_bytes_fn = (
        lambda: int((19.0 if test_state.state.gpu_slot is not None else 23.5) * _GIB)
    )
    fake_services.gpu_info.live_allocated_bytes_fn = _resident_weights_probe(test_state)

    result = _queued_t2v(test_state, tmp_path / "w-band.mp4", resolution="720p", duration=20)

    assert isinstance(result, GenerateVideoCompleteResponse)
    assert len(fake_services.fast_video_pipeline.generate_calls) == 2


def test_queued_execute_retries_at_most_once(
    test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    # Other apps still hold the memory after the evict; a second retry would find room.
    create_fake_model_files()
    _configure_windows_24gb(test_state)
    _warm_pipeline(test_state, fake_services, tmp_path)
    samples = iter([19.0, 19.0])
    fake_services.gpu_info.free_vram_bytes_fn = lambda: int(next(samples, 31.0) * _GIB)
    fake_services.gpu_info.live_allocated_bytes = int(4.5 * _GIB)

    with pytest.raises(HTTPError) as refused:
        _queued_t2v(test_state, tmp_path / "short.mp4", resolution="720p", duration=20)

    assert refused.value.status_code == 422
    assert refused.value.code == VIDEO_JOB_TOO_LARGE
    assert len(fake_services.fast_video_pipeline.generate_calls) == 1


def test_direct_generate_evict_retry_admits_w_band_job(
    test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    # Same W-band as queued: GenSpace must evict once, not 422 on its own weights.
    create_fake_model_files()
    _configure_windows_24gb(test_state)
    _warm_pipeline(test_state, fake_services, tmp_path)
    fake_services.gpu_info.free_vram_bytes_fn = (
        lambda: int((19.0 if test_state.state.gpu_slot is not None else 23.5) * _GIB)
    )
    fake_services.gpu_info.live_allocated_bytes_fn = _resident_weights_probe(test_state)

    result = test_state.video_generation.generate(
        GenerateVideoRequest.model_validate({**_T2V, "resolution": "720p", "duration": 20})
    )

    assert isinstance(result, GenerateVideoCompleteResponse)
    assert len(fake_services.fast_video_pipeline.generate_calls) == 2


def test_estimate_leaves_resident_pipeline_untouched(
    test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    # Estimate is read-only: a W-band 422 must not evict the resident pipeline.
    create_fake_model_files()
    _configure_windows_24gb(test_state)
    _warm_pipeline(test_state, fake_services, tmp_path)
    resident = test_state.state.gpu_slot
    assert resident is not None
    fake_services.gpu_info.free_vram_bytes = int(19.0 * _GIB)
    fake_services.gpu_info.live_allocated_bytes = int(4.5 * _GIB)

    with pytest.raises(HTTPError) as refused:
        test_state.video_generation.estimate_local_fast(
            EstimateVideoRequest.model_validate(
                {"resolution": "720p", "duration": 20, "fps": 24}
            )
        )

    assert refused.value.status_code == 422
    assert test_state.state.gpu_slot is resident


def test_auto_duration_static_fallback_gates_on_post_evict_free(
    test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    # 720p asks 9.9-20.03; occupied free 9.8 drops every duration, then the
    # static probe evicts 10.5 of weights. The gate must price 20.3, not 9.8.
    create_fake_model_files()
    _configure_windows_24gb(test_state)
    _warm_pipeline(test_state, fake_services, tmp_path)
    fake_services.gpu_info.free_vram_bytes_fn = (
        lambda: int((9.8 if test_state.state.gpu_slot is not None else 20.3) * _GIB)
    )
    fake_services.gpu_info.live_allocated_bytes_fn = (
        lambda: int(10.5 * _GIB) if test_state.state.gpu_slot is not None else 0
    )

    result = test_state.video_generation.generate(
        GenerateVideoRequest.model_validate(
            {**_T2V, "resolution": "720p", "duration": None}
        )
    )

    assert isinstance(result, GenerateVideoCompleteResponse)
