"""Shrink-then-run: a free-refused total-fitting edit window re-windows smaller.

On an idle-desktop 5090 (31.84 total, 30.1 free) every 1080p retake/extend
pads to a 369f window that asks 31.6 GiB and deterministically refuses.
Prepare must shrink to the free-gated cap (348 -> 345f, asks 29.9) and run,
refusing only when the selection itself cannot fit.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from _routes._errors import HTTPError
from api_types import ExtendParams, LTXLocalModelId, RetakeParams
from runtime_config.budget_admission import RiskAllowance
from runtime_config.device_calibration import JobEnvelope
from runtime_config.video_job_budget import (
    LOCAL_GENERATION_RAM_SHORTFALL,
    VIDEO_JOB_TOO_LARGE,
)
from services.features.video.extend import ExtendExecutor
from services.features.video.retake import RetakeExecutor
from services.generation_queue.types import OutputAllocation
from tests.fakes.generation import FakeJobSession
from services.records import CapabilityFailedError
from services.sqlite_store import SqliteStore
from tests.video_fixtures import write_mp4

_GIB = 1024**3
_LOCAL_2_5: LTXLocalModelId = "ltx-2.5-22b-distilled"


def _configure_5090_box(test_state, fake_services) -> None:
    test_state.config.system = "Windows"
    test_state.config.vram_gb = 31
    test_state.config.vram_bytes = int(31.84 * _GIB)
    test_state.config.total_ram_bytes = 64 * _GIB
    test_state.config.local_generations_mode = "streaming_models_loading"
    fake_services.gpu_info.free_vram_bytes = int(30.1 * _GIB)


def _install_local(test_state, create_fake_model_files) -> None:
    create_fake_model_files(model_id=_LOCAL_2_5)
    test_state.state.app_settings.use_local_text_encoder = True


def _clip_1080p(path: Path) -> Path:
    return write_mp4(path, seconds=481 / 24, fps=24, width=1920, height=1080)


def _store(test_state) -> SqliteStore:
    return SqliteStore(test_state.config.app_data_dir)


def _retake_executor(test_state, store) -> RetakeExecutor:
    return RetakeExecutor(
        test_state.pipelines,
        store,
        test_state.text,
        lambda: test_state.video_generation.models_dir,
        "negative",
        test_state.generation,
        test_state.prompt_enhancement,
    )


def _extend_executor(test_state, store) -> ExtendExecutor:
    return ExtendExecutor(
        test_state.pipelines,
        store,
        test_state.text,
        lambda: test_state.video_generation.models_dir,
        "negative",
        test_state.generation,
        test_state.prompt_enhancement,
    )


def test_home_edit_free_short_refusal_advises_lower_resolution(
    test_state, fake_services, create_fake_model_files
) -> None:
    """Queued Home must not tell the user to close apps: the re-pad makes retry futile."""
    _install_local(test_state, create_fake_model_files)
    _configure_5090_box(test_state, fake_services)

    with pytest.raises(HTTPError) as exc_info:
        test_state.pipelines.load_retake_pipeline(
            distilled=True,
            ltx_model_id=_LOCAL_2_5,
            width=1920,
            height=1088,
            frames=369,
            queued_home=True,
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.code == VIDEO_JOB_TOO_LARGE
    assert exc_info.value.detail == (
        "This job needs 31.6 GB of GPU memory but only 30.1 GB is free "
        "right now (of 31.8 GB total). Lower the resolution."
    )


def test_edit_budget_lines_carry_vram_triple(
    test_state, fake_services, create_fake_model_files, caplog
) -> None:
    """Refuse and load lines must carry allocated/reserved/free for the next triage."""
    _install_local(test_state, create_fake_model_files)
    _configure_5090_box(test_state, fake_services)
    fake_services.gpu_info.live_allocated_bytes = int(0.5 * _GIB)
    fake_services.gpu_info.reserved_bytes = int(1.0 * _GIB)
    caplog.set_level(logging.INFO)

    with pytest.raises(HTTPError):
        test_state.pipelines.load_retake_pipeline(
            distilled=True,
            ltx_model_id=_LOCAL_2_5,
            width=1920,
            height=1088,
            frames=369,
            queued_home=True,
        )
    state, _ = test_state.pipelines.load_retake_pipeline(
        distilled=True,
        ltx_model_id=_LOCAL_2_5,
        width=1920,
        height=1088,
        frames=345,
        queued_home=True,
    )

    assert state.loading_mode == "streaming_models_loading"
    lines = [
        record.getMessage()
        for record in caplog.records
        if "[video-edit] job budget" in record.getMessage()
    ]
    assert len(lines) == 2
    for line in lines:
        assert line.endswith(" (vram alloc=0.5GiB reserved=1.0GiB free=30.1GiB)")


@pytest.mark.parametrize("allow_risk", [False, True])
def test_retake_prepare_shrinks_free_refused_window_instead_of_raising(
    test_state, fake_services, create_fake_model_files, tmp_path, caplog, allow_risk: bool
) -> None:
    _install_local(test_state, create_fake_model_files)
    _configure_5090_box(test_state, fake_services)
    store = _store(test_state)
    video = store.ingest_upload(str(_clip_1080p(tmp_path / "clip.mp4")))
    generation = store.insert_generation(
        "retake",
        {
            "params": {"prompt": "the character turns"},
            "inputs": {"video": {"assetId": video.id}},
        },
    )
    caplog.set_level(logging.INFO)

    prepared = _retake_executor(test_state, store).prepare(
        generation,
        RetakeParams(
            prompt="the character turns",
            model="ltx-2.5-fast",
            startTime=10.8,
            duration=5.3,
            mode="replace_video",
        ),
        video,
        _LOCAL_2_5,
        7,
        tmp_path / "out.mp4",
        str(tmp_path / "out.window.mp4"),
        allowance=RiskAllowance(over=allow_risk, unmeasured=False),
    )

    assert prepared.window.encode_frames == 345
    assert (prepared.window.mask_start_frame, prepared.window.mask_end_frame) == (
        259,
        386,
    )
    assert [
        record.getMessage()
        for record in caplog.records
        if "shrinking to free-gated cap" in record.getMessage()
    ] == [
        f"Retake {generation.id} free gate refused the total-fitting 369-frame "
        "window; shrinking to free-gated cap 348 (encode 345)"
    ]


@pytest.mark.parametrize("allow_risk", [False, True])
def test_extend_prepare_shrinks_free_refused_window_instead_of_raising(
    test_state, fake_services, create_fake_model_files, tmp_path, caplog, allow_risk: bool
) -> None:
    _install_local(test_state, create_fake_model_files)
    _configure_5090_box(test_state, fake_services)
    store = _store(test_state)
    video = store.ingest_upload(str(_clip_1080p(tmp_path / "clip.mp4")))
    generation = store.insert_generation(
        "extend",
        {
            "params": {"prompt": "the scene continues"},
            "inputs": {"video": {"assetId": video.id}},
        },
    )
    caplog.set_level(logging.INFO)

    prepared = _extend_executor(test_state, store).prepare(
        generation,
        ExtendParams(
            prompt="the scene continues",
            model="ltx-2.5-fast",
            duration=4,
            mode="end",
        ),
        video,
        _LOCAL_2_5,
        7,
        tmp_path / "out.mp4",
        str(tmp_path / "out.window.mp4"),
        allowance=RiskAllowance(over=allow_risk, unmeasured=False),
    )

    assert prepared.extend_frames == 96
    assert prepared.window.context_frames + prepared.extend_frames == 345
    assert [
        record.getMessage()
        for record in caplog.records
        if "shrinking to free-gated cap" in record.getMessage()
    ] == [
        f"Extend {generation.id} free gate refused the total-fitting 369-frame "
        "window; shrinking to free-gated cap 348 (encode 345)"
    ]


@pytest.mark.parametrize("allow_risk", [False, True])
def test_retake_prepare_refuses_when_selection_exceeds_free_fitting_window(
    test_state, fake_services, create_fake_model_files, tmp_path, allow_risk: bool
) -> None:
    """A 360f mask fits the 372 total cap but not the 348 free cap: refuse, naming the max."""
    _install_local(test_state, create_fake_model_files)
    _configure_5090_box(test_state, fake_services)
    store = _store(test_state)
    video = store.ingest_upload(str(_clip_1080p(tmp_path / "clip.mp4")))
    generation = store.insert_generation(
        "retake",
        {
            "params": {"prompt": "the character turns"},
            "inputs": {"video": {"assetId": video.id}},
        },
    )

    with pytest.raises(CapabilityFailedError) as exc_info:
        _retake_executor(test_state, store).prepare(
            generation,
            RetakeParams(
                prompt="the character turns",
                model="ltx-2.5-fast",
                startTime=2.0,
            duration=15,
            mode="replace_video",
        ),
        video,
        _LOCAL_2_5,
        7,
        tmp_path / "out.mp4",
        str(tmp_path / "out.window.mp4"),
        allowance=RiskAllowance(over=allow_risk, unmeasured=False),
    )

    assert exc_info.value.code == VIDEO_JOB_TOO_LARGE
    assert exc_info.value.detail == (
        "Retake selection of 15s is too long for 1920x1080 on this machine "
        "(at most 14s)."
    )


def test_extend_prepare_shrinks_context_below_floor_instead_of_refusing(
    test_state, fake_services, create_fake_model_files, tmp_path, caplog
) -> None:
    """A 10s extend needs 362f (240 + 121 floor) but free fits 348: run with 105 context, not fail."""
    _install_local(test_state, create_fake_model_files)
    _configure_5090_box(test_state, fake_services)
    store = _store(test_state)
    video = store.ingest_upload(str(_clip_1080p(tmp_path / "clip.mp4")))
    generation = store.insert_generation(
        "extend",
        {
            "params": {"prompt": "the scene continues"},
            "inputs": {"video": {"assetId": video.id}},
        },
    )
    caplog.set_level(logging.INFO)

    prepared = _extend_executor(test_state, store).prepare(
        generation,
        ExtendParams(
            prompt="the scene continues",
            model="ltx-2.5-fast",
            duration=10,
            mode="end",
        ),
        video,
        _LOCAL_2_5,
        7,
        tmp_path / "out.mp4",
        str(tmp_path / "out.window.mp4"),
        allowance=RiskAllowance.none(),
    )

    assert prepared.extend_frames == 240
    assert prepared.window.context_frames == 105
    assert prepared.window.context_frames + prepared.extend_frames == 345
    assert [
        record.getMessage()
        for record in caplog.records
        if "below the 121-frame coherence floor" in record.getMessage()
    ] == [
        f"Extend {generation.id} running with 105 context frames, "
        "below the 121-frame coherence floor (unmeasured regime)"
    ]


def test_extend_prepare_from_start_shrink_trims_tail_context(
    test_state, fake_services, create_fake_model_files, tmp_path
) -> None:
    """Shrinking a start-extend must drop the latest frames (far from the seam), not the head."""
    _install_local(test_state, create_fake_model_files)
    _configure_5090_box(test_state, fake_services)
    store = _store(test_state)
    video = store.ingest_upload(str(_clip_1080p(tmp_path / "clip.mp4")))
    generation = store.insert_generation(
        "extend",
        {
            "params": {"prompt": "the scene continues"},
            "inputs": {"video": {"assetId": video.id}},
        },
    )

    prepared = _extend_executor(test_state, store).prepare(
        generation,
        ExtendParams(
            prompt="the scene continues",
            model="ltx-2.5-fast",
            duration=10,
            mode="start",
        ),
        video,
        _LOCAL_2_5,
        7,
        tmp_path / "out.mp4",
        str(tmp_path / "out.window.mp4"),
        allowance=RiskAllowance.none(),
    )

    assert prepared.extend_frames == 240
    assert prepared.window.context_frames == 105
    assert prepared.window.encode_start_frame == 0
    assert prepared.window.dropped_suffix_frames == 376


@pytest.mark.parametrize("allow_risk", [False, True])
def test_extend_prepare_refuses_when_selection_exceeds_total_cap(
    test_state, fake_services, create_fake_model_files, tmp_path, caplog, allow_risk: bool
) -> None:
    """A 360f extension exceeds 372 - 121 at any free level: refuse before any load, naming the max."""
    _install_local(test_state, create_fake_model_files)
    _configure_5090_box(test_state, fake_services)
    store = _store(test_state)
    video = store.ingest_upload(str(_clip_1080p(tmp_path / "clip.mp4")))
    generation = store.insert_generation(
        "extend",
        {
            "params": {"prompt": "the scene continues"},
            "inputs": {"video": {"assetId": video.id}},
        },
    )
    caplog.set_level(logging.INFO)

    with pytest.raises(CapabilityFailedError) as exc_info:
        _extend_executor(test_state, store).prepare(
            generation,
            ExtendParams(
                prompt="the scene continues",
                model="ltx-2.5-fast",
            duration=15,
            mode="end",
        ),
        video,
        _LOCAL_2_5,
        7,
        tmp_path / "out.mp4",
        str(tmp_path / "out.window.mp4"),
        allowance=RiskAllowance(over=allow_risk, unmeasured=False),
    )

    assert exc_info.value.code == VIDEO_JOB_TOO_LARGE
    assert exc_info.value.detail == (
        "Extend of 15s is too long for 1920x1080 on this machine "
        "(at most 10s)."
    )
    assert not [
        record.getMessage()
        for record in caplog.records
        if "shrinking to free-gated cap" in record.getMessage()
    ]


@pytest.mark.parametrize("allow_risk", [False, True])
def test_retake_prepare_refuses_when_free_gap_exceeds_shrink_limit(
    test_state, fake_services, create_fake_model_files, tmp_path, caplog, allow_risk: bool
) -> None:
    """Free 24GB leaves a 111f gap past the 48 limit: keep the load refusal, don't run the mask on slivers."""
    _install_local(test_state, create_fake_model_files)
    _configure_5090_box(test_state, fake_services)
    fake_services.gpu_info.free_vram_bytes = 24 * _GIB
    store = _store(test_state)
    video = store.ingest_upload(str(_clip_1080p(tmp_path / "clip.mp4")))
    generation = store.insert_generation(
        "retake",
        {
            "params": {"prompt": "the character turns"},
            "inputs": {"video": {"assetId": video.id}},
        },
    )
    caplog.set_level(logging.INFO)

    with pytest.raises(HTTPError) as exc_info:
        _retake_executor(test_state, store).prepare(
            generation,
            RetakeParams(
                prompt="the character turns",
                model="ltx-2.5-fast",
                startTime=10.8,
                duration=5.3,
                mode="replace_video",
            ),
            video,
            _LOCAL_2_5,
            7,
            tmp_path / "out.mp4",
            str(tmp_path / "out.window.mp4"),
            allowance=RiskAllowance(over=allow_risk, unmeasured=False),
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.code == VIDEO_JOB_TOO_LARGE
    assert exc_info.value.detail == (
        "This job needs 31.6 GB of GPU memory but only 24.0 GB is free "
        "right now (of 31.8 GB total). Lower the resolution."
    )
    assert [
        record.getMessage()
        for record in caplog.records
        if "too far below total cap" in record.getMessage()
    ] == [
        "Edit shrink skipped: free cap 261 too far below total cap 372 "
        "for 1920x1088 (gap 111 > 48); failing instead of running on slivers"
    ]


@pytest.mark.parametrize("allow_risk", [False, True])
def test_extend_prepare_refuses_when_free_gap_exceeds_shrink_limit(
    test_state, fake_services, create_fake_model_files, tmp_path, allow_risk: bool
) -> None:
    """Free 24GB leaves a 111f gap past the 48 limit: keep the load refusal, don't extend from slivers."""
    _install_local(test_state, create_fake_model_files)
    _configure_5090_box(test_state, fake_services)
    fake_services.gpu_info.free_vram_bytes = 24 * _GIB
    store = _store(test_state)
    video = store.ingest_upload(str(_clip_1080p(tmp_path / "clip.mp4")))
    generation = store.insert_generation(
        "extend",
        {
            "params": {"prompt": "the scene continues"},
            "inputs": {"video": {"assetId": video.id}},
        },
    )

    with pytest.raises(HTTPError) as exc_info:
        _extend_executor(test_state, store).prepare(
            generation,
            ExtendParams(
                prompt="the scene continues",
                model="ltx-2.5-fast",
                duration=4,
                mode="end",
            ),
            video,
            _LOCAL_2_5,
            7,
            tmp_path / "out.mp4",
            str(tmp_path / "out.window.mp4"),
            allowance=RiskAllowance(over=allow_risk, unmeasured=False),
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.code == VIDEO_JOB_TOO_LARGE
    assert exc_info.value.detail == (
        "This job needs 31.6 GB of GPU memory but only 24.0 GB is free "
        "right now (of 31.8 GB total). Lower the resolution."
    )


def _configure_ram_short_box(test_state, fake_services) -> None:
    test_state.config.system = "Windows"
    test_state.config.vram_gb = 48
    test_state.config.vram_bytes = 48 * _GIB
    test_state.config.total_ram_bytes = 16 * _GIB
    test_state.config.local_generations_mode = "streaming_models_loading"
    fake_services.gpu_info.free_vram_bytes = 48 * _GIB


def test_stored_over_still_runs_retake_ram_shortfall_shrink_cannot_salvage(
    test_state, fake_services, create_fake_model_files, tmp_path, caplog
) -> None:
    # Shrink ignores a RAM refusal. The stored over bet has to run that window.
    _install_local(test_state, create_fake_model_files)
    _configure_ram_short_box(test_state, fake_services)
    store = _store(test_state)
    video = store.ingest_upload(str(_clip_1080p(tmp_path / "clip.mp4")))
    generation = store.insert_generation(
        "retake",
        {
            "params": {"prompt": "the character turns"},
            "inputs": {"video": {"assetId": video.id}},
        },
    )
    params = RetakeParams(
        prompt="the character turns",
        model="ltx-2.5-fast",
        startTime=10.8,
        duration=5.3,
        mode="replace_video",
    )
    executor = _retake_executor(test_state, store)
    with pytest.raises(HTTPError) as refused:
        executor.prepare(
            generation,
            params,
            video,
            _LOCAL_2_5,
            7,
            tmp_path / "out.mp4",
            str(tmp_path / "out.window.mp4"),
            allowance=RiskAllowance.none(),
        )
    assert refused.value.code == LOCAL_GENERATION_RAM_SHORTFALL
    caplog.set_level(logging.INFO)
    prepared = executor.prepare(
        generation,
        params,
        video,
        _LOCAL_2_5,
        7,
        tmp_path / "out.mp4",
        str(tmp_path / "out.window.mp4"),
        allowance=RiskAllowance(over=True, unmeasured=False),
    )
    assert prepared.window.encode_frames == 481
    assert not any(
        "shrinking to free-gated cap" in record.getMessage() for record in caplog.records
    )


def test_stored_over_still_runs_extend_ram_shortfall_shrink_cannot_salvage(
    test_state, fake_services, create_fake_model_files, tmp_path, caplog
) -> None:
    # Shrink ignores a RAM refusal. The stored over bet has to run that window.
    _install_local(test_state, create_fake_model_files)
    _configure_ram_short_box(test_state, fake_services)
    store = _store(test_state)
    video = store.ingest_upload(str(_clip_1080p(tmp_path / "clip.mp4")))
    generation = store.insert_generation(
        "extend",
        {
            "params": {"prompt": "the scene continues"},
            "inputs": {"video": {"assetId": video.id}},
        },
    )
    params = ExtendParams(
        prompt="the scene continues",
        model="ltx-2.5-fast",
        duration=4,
        mode="end",
    )
    executor = _extend_executor(test_state, store)
    with pytest.raises(HTTPError) as refused:
        executor.prepare(
            generation,
            params,
            video,
            _LOCAL_2_5,
            7,
            tmp_path / "out.mp4",
            str(tmp_path / "out.window.mp4"),
            allowance=RiskAllowance.none(),
        )
    assert refused.value.code == LOCAL_GENERATION_RAM_SHORTFALL
    caplog.set_level(logging.INFO)
    prepared = executor.prepare(
        generation,
        params,
        video,
        _LOCAL_2_5,
        7,
        tmp_path / "out.mp4",
        str(tmp_path / "out.window.mp4"),
        allowance=RiskAllowance(over=True, unmeasured=False),
    )
    assert prepared.extend_frames == 96
    assert prepared.window.context_frames + prepared.extend_frames == 505
    assert not any(
        "shrinking to free-gated cap" in record.getMessage() for record in caplog.records
    )


class _Slot:
    def start_generation(
        self, generation_id: str, *, ran: JobEnvelope | None = None
    ) -> None:
        del generation_id, ran

    def complete_generation(
        self, result: str | list[str] | None = None, *, ran: JobEnvelope | None = None
    ) -> None:
        del result, ran

    def fail_generation(self, error: str) -> None:
        del error


class _RecordingRetake(RetakeExecutor):
    encode_frames: int | None = None

    def generate(self, prepared: object) -> None:
        self.encode_frames = prepared.window.encode_frames  # type: ignore[attr-defined]

    def stitch(self, prepared: object) -> None:
        del prepared


def _over_risk() -> dict[str, object]:
    return {
        "tier": "over",
        "shortfall_kind": "free",
        "over_gib": 1.5,
        "over_ram_gb": None,
        "message": "Over budget.",
    }


def _execute_retake(test_state, store, generation) -> _RecordingRetake:
    executor = _RecordingRetake(
        test_state.pipelines,
        store,
        test_state.text,
        lambda: test_state.video_generation.models_dir,
        "negative",
        _Slot(),
        test_state.prompt_enhancement,
    )
    plans = executor.plan_outputs(generation)
    executor.execute(
        generation,
        (
            OutputAllocation(
                plan=plans[0],
                asset_id="out",
                dest_path=str(generation.id) + ".mp4",
            ),
        ),
        FakeJobSession(),
    )
    return executor


def test_execute_shrinks_stored_over_retake_when_the_free_dip_fits_the_cap(
    test_state, fake_services, create_fake_model_files, tmp_path
) -> None:
    # execute must thread the stored over into prepare. The dip still shrinks
    # to 345; passing the bet on the first load would keep the 369-frame window.
    _install_local(test_state, create_fake_model_files)
    _configure_5090_box(test_state, fake_services)
    store = _store(test_state)
    video = store.ingest_upload(str(_clip_1080p(tmp_path / "clip.mp4")))
    generation = store.insert_generation(
        "retake",
        {
            "params": {
                "prompt": "the character turns",
                "model": "ltx-2.5-fast",
                "startTime": 10.8,
                "duration": 5.3,
                "mode": "replace_video",
            },
            "inputs": {"video": {"assetId": video.id}},
            "_risk": _over_risk(),
        },
    )
    executor = _execute_retake(test_state, store, generation)
    assert executor.encode_frames == 345


def test_execute_stored_over_runs_the_unshrunk_retake_on_a_ram_shortfall(
    test_state, fake_services, create_fake_model_files, tmp_path
) -> None:
    # A hardcoded allow_risk=False inside execute would 422 this. The stored
    # over has to reach prepare, and the window stays the 481-frame total one.
    _install_local(test_state, create_fake_model_files)
    _configure_ram_short_box(test_state, fake_services)
    store = _store(test_state)
    video = store.ingest_upload(str(_clip_1080p(tmp_path / "clip.mp4")))
    refused = store.insert_generation(
        "retake",
        {
            "params": {
                "prompt": "the character turns",
                "model": "ltx-2.5-fast",
                "startTime": 10.8,
                "duration": 5.3,
                "mode": "replace_video",
            },
            "inputs": {"video": {"assetId": video.id}},
        },
    )
    with pytest.raises(CapabilityFailedError) as exc_info:
        _execute_retake(test_state, store, refused)
    assert exc_info.value.code == LOCAL_GENERATION_RAM_SHORTFALL
    generation = store.insert_generation(
        "retake",
        {
            "params": {
                "prompt": "the character turns",
                "model": "ltx-2.5-fast",
                "startTime": 10.8,
                "duration": 5.3,
                "mode": "replace_video",
            },
            "inputs": {"video": {"assetId": video.id}},
            "_risk": _over_risk(),
        },
    )
    executor = _execute_retake(test_state, store, generation)
    assert executor.encode_frames == 481
