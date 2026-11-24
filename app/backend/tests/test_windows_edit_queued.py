"""Queued Home retake/extend must preflight window + budget before insert."""

from __future__ import annotations

from pathlib import Path

from starlette.testclient import TestClient

from app_factory import create_app
from runtime_config.video_job_budget import (
    LOCAL_GENERATION_RAM_SHORTFALL,
    LOCAL_GENERATION_UNSUPPORTED,
    LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
    VIDEO_JOB_TOO_LARGE,
)
from services.retake_pipeline.window import retake_selection_too_long_message
from services.sqlite_store import SqliteStore
from tests.conftest import TEST_ADMIN_TOKEN
from tests.http_error_assertions import assert_http_error

_GIB = 1024**3


def _store(test_state) -> SqliteStore:
    return SqliteStore(test_state.config.app_data_dir)


def _configure_windows_edit(
    test_state, *, vram_gib: float, total_ram_gib: float
) -> None:
    test_state.config.system = "Windows"
    test_state.config.vram_gb = int(vram_gib)
    test_state.config.vram_bytes = int(vram_gib * _GIB)
    test_state.config.total_ram_bytes = int(total_ram_gib * _GIB)
    test_state.config.local_generations_mode = "streaming_models_loading"


def _mp4(
    path: Path, *, width: int = 64, height: int = 64, frames: int = 73, fps: int = 24
) -> Path:
    import imageio.v2 as imageio
    import numpy as np

    writer = imageio.get_writer(
        str(path), fps=fps, codec="libx264", macro_block_size=None
    )
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(frames):
        writer.append_data(frame)
    writer.close()
    return path


def _ingest_video(test_state, path: Path) -> str:
    return _store(test_state).ingest_upload(str(path)).id


def _post_create(test_state, feature: str, body: dict):
    app = create_app(handler=test_state, admin_token=TEST_ADMIN_TOKEN)
    with TestClient(app) as client:
        return client.post(f"/api/generations/{feature}", json=body)


def _retake_body(asset_id: str, *, start: float = 0, duration: float = 2) -> dict:
    return {
        "params": {
            "prompt": "the character turns",
            "model": "ltx-2.5-fast",
            "startTime": start,
            "duration": duration,
            "mode": "replace_video",
        },
        "inputs": {"video": {"assetId": asset_id}},
    }


def _extend_body(asset_id: str, *, duration: float = 4) -> dict:
    return {
        "params": {
            "prompt": "the scene continues",
            "model": "ltx-2.5-fast",
            "duration": duration,
            "mode": "end",
        },
        "inputs": {"video": {"assetId": asset_id}},
    }


def test_queued_retake_over_window_422s_before_insert_with_max_named(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _configure_windows_edit(test_state, vram_gib=12, total_ram_gib=64)
    asset_id = _ingest_video(
        test_state,
        _mp4(tmp_path / "clip.mp4", width=1920, height=1080, frames=481, fps=24),
    )

    response = _post_create(
        test_state, "retake", _retake_body(asset_id, start=0, duration=10)
    )

    assert_http_error(
        response,
        status_code=422,
        code=VIDEO_JOB_TOO_LARGE,
        message=(
            "Retake selection of 10s is too long for 1920x1080 on this machine "
            "(at most 3s)."
        ),
    )
    assert _store(test_state).list_generations("retake") == []


def test_queued_retake_ram_shortfall_422s_before_insert(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _configure_windows_edit(test_state, vram_gib=31, total_ram_gib=16)
    asset_id = _ingest_video(test_state, _mp4(tmp_path / "clip.mp4"))

    response = _post_create(
        test_state, "retake", _retake_body(asset_id, start=0, duration=2)
    )

    assert_http_error(
        response,
        status_code=422,
        code=LOCAL_GENERATION_RAM_SHORTFALL,
        message=(
            "This job needs 64 GB of system RAM but this machine has 16 GB. "
            "Close heavy programs; on desktops, more RAM is the real fix."
        ),
    )
    assert _store(test_state).list_generations("retake") == []


def test_queued_retake_fitting_selection_still_enqueues(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _configure_windows_edit(test_state, vram_gib=31, total_ram_gib=64)
    asset_id = _ingest_video(test_state, _mp4(tmp_path / "clip.mp4"))

    response = _post_create(
        test_state, "retake", _retake_body(asset_id, start=0, duration=2)
    )

    assert response.status_code == 200, response.text
    assert _store(test_state).list_generations("retake") != []


def test_queued_retake_admits_on_total_while_free_depressed(
    test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    # A running job depresses free; the queue exists to wait for it, so create gates on total.
    create_fake_model_files()
    _configure_windows_edit(test_state, vram_gib=31, total_ram_gib=64)
    fake_services.gpu_info.free_vram_bytes = 4 * _GIB
    asset_id = _ingest_video(test_state, _mp4(tmp_path / "clip.mp4"))

    response = _post_create(
        test_state, "retake", _retake_body(asset_id, start=0, duration=2)
    )

    assert response.status_code == 200, response.text
    assert _store(test_state).list_generations("retake") != []


def test_queued_extend_over_cap_422s_before_insert_with_max_named(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _configure_windows_edit(test_state, vram_gib=12, total_ram_gib=64)
    asset_id = _ingest_video(
        test_state, _mp4(tmp_path / "clip.mp4", width=1920, height=1080)
    )

    response = _post_create(test_state, "extend", _extend_body(asset_id, duration=10))

    assert_http_error(
        response,
        status_code=422,
        code=VIDEO_JOB_TOO_LARGE,
        message=(
            "Extend of 10s is too long for 1920x1080 on this machine (at most 0s)."
        ),
    )
    assert _store(test_state).list_generations("extend") == []


def test_queued_extend_ram_shortfall_422s_before_insert(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _configure_windows_edit(test_state, vram_gib=31, total_ram_gib=16)
    asset_id = _ingest_video(test_state, _mp4(tmp_path / "clip.mp4"))

    response = _post_create(test_state, "extend", _extend_body(asset_id, duration=4))

    assert_http_error(
        response,
        status_code=422,
        code=LOCAL_GENERATION_RAM_SHORTFALL,
        message=(
            "This job needs 64 GB of system RAM but this machine has 16 GB. "
            "Close heavy programs; on desktops, more RAM is the real fix."
        ),
    )
    assert _store(test_state).list_generations("extend") == []


def test_queued_extend_fitting_job_still_enqueues(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _configure_windows_edit(test_state, vram_gib=31, total_ram_gib=64)
    asset_id = _ingest_video(test_state, _mp4(tmp_path / "clip.mp4"))

    response = _post_create(test_state, "extend", _extend_body(asset_id, duration=4))

    assert response.status_code == 200, response.text
    assert _store(test_state).list_generations("extend") != []


def test_queued_edit_unsupported_mode_422s_before_insert(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _configure_windows_edit(test_state, vram_gib=31, total_ram_gib=64)
    test_state.config.local_generations_mode = "unsupported"
    asset_id = _ingest_video(test_state, _mp4(tmp_path / "clip.mp4"))

    response = _post_create(
        test_state, "retake", _retake_body(asset_id, start=0, duration=2)
    )

    assert_http_error(
        response,
        status_code=422,
        code=LOCAL_GENERATION_UNSUPPORTED,
        message=LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
    )
    assert _store(test_state).list_generations("retake") == []


def test_retake_too_long_message_floors_max_below_cap() -> None:
    assert retake_selection_too_long_message(
        duration_s=20,
        target_width=1920,
        target_height=1056,
        frame_cap=480,
        fps=24,
    ) == (
        "Retake selection of 20s is too long for 1920x1056 on this machine "
        "(at most 19s)."
    )
