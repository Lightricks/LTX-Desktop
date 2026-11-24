"""POST /api/generations/edit/preflight returns advisory per-cell verdicts."""

from __future__ import annotations

from pathlib import Path

from starlette.testclient import TestClient

from app_factory import create_app
from runtime_config.video_job_budget import VIDEO_JOB_TOO_LARGE
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


def _png(path: Path, size: tuple[int, int] = (16, 16)) -> Path:
    from PIL import Image

    Image.new("RGB", size).save(path)
    return path


def _ingest(test_state, path: Path) -> str:
    return _store(test_state).ingest_upload(str(path)).id


def _client(test_state) -> TestClient:
    return TestClient(create_app(handler=test_state, admin_token=TEST_ADMIN_TOKEN))


def _preflight_body(
    asset_id: str,
    *,
    feature: str = "extend",
    candidates: list[dict] | None = None,
    durations: list[float] | None = None,
    start_time: float = 0,
    mode: str = "end",
) -> dict:
    return {
        "feature": feature,
        "assetId": asset_id,
        "startTime": start_time,
        "mode": mode,
        "candidates": (
            candidates
            if candidates is not None
            else [{"key": "original", "width": None, "height": None}]
        ),
        "durations": durations if durations is not None else [4],
    }


def _clip_1080p(test_state, tmp_path: Path) -> str:
    return _ingest(
        test_state,
        _mp4(tmp_path / "clip.mp4", width=1920, height=1080, frames=481, fps=24),
    )


def test_preflight_cell_mirrors_create_422(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _configure_windows_edit(test_state, vram_gib=12, total_ram_gib=64)
    asset_id = _clip_1080p(test_state, tmp_path)
    with _client(test_state) as client:
        preflight = client.post(
            "/api/generations/edit/preflight",
            json=_preflight_body(asset_id, durations=[12]),
        )
        assert preflight.status_code == 200
        (cell,) = [c for c in preflight.json()["cells"] if c["duration"] == 12]
        assert cell["ok"] is False

        create = client.post(
            "/api/generations/extend",
            json={
                "params": {
                    "prompt": "the scene continues",
                    "model": "ltx-2.5-fast",
                    "duration": 12,
                    "mode": "end",
                },
                "inputs": {"video": {"assetId": asset_id}},
            },
        )
        assert create.status_code == 422
        payload = create.json()
        assert payload["code"] == VIDEO_JOB_TOO_LARGE
        assert cell["message"] == payload["message"]
        assert cell["code"] == payload["code"]


def test_preflight_max_shrinks_at_higher_tier(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _configure_windows_edit(test_state, vram_gib=12, total_ram_gib=64)
    asset_id = _clip_1080p(test_state, tmp_path)
    with _client(test_state) as client:
        response = client.post(
            "/api/generations/edit/preflight",
            json=_preflight_body(
                asset_id,
                feature="retake",
                candidates=[
                    {"key": "original", "width": None, "height": None},
                    {"key": "720", "width": 1280, "height": 720},
                ],
                durations=[2],
            ),
        )
        assert response.status_code == 200
        rows = {row["key"]: row for row in response.json()["resolutions"]}
        assert (rows["original"]["width"], rows["original"]["height"]) == (
            1920,
            1080,
        )
        assert (rows["720"]["width"], rows["720"]["height"]) == (1280, 720)
        assert (
            0
            < rows["original"]["maxDurationSeconds"]
            < rows["720"]["maxDurationSeconds"]
        )


def test_preflight_min_pass_with_preset_fail_keeps_resolution(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _configure_windows_edit(test_state, vram_gib=12, total_ram_gib=64)
    asset_id = _clip_1080p(test_state, tmp_path)
    with _client(test_state) as client:
        response = client.post(
            "/api/generations/edit/preflight",
            json=_preflight_body(
                asset_id,
                candidates=[{"key": "720", "width": 1280, "height": 720}],
                durations=[2, 12],
            ),
        )
        assert response.status_code == 200
        cells = {
            (cell["key"], cell["duration"]): cell
            for cell in response.json()["cells"]
        }
        assert cells[("720", 2)]["ok"] is True
        failed = cells[("720", 12)]
        assert failed["ok"] is False
        assert failed["code"] == VIDEO_JOB_TOO_LARGE
        assert failed["message"] == (
            "Extend of 12s is too long for 1280x720 on this machine (at most 3s)."
        )


def test_preflight_named_max_passes_its_own_cell(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _configure_windows_edit(test_state, vram_gib=12, total_ram_gib=64)
    asset_id = _clip_1080p(test_state, tmp_path)
    with _client(test_state) as client:
        first = client.post(
            "/api/generations/edit/preflight",
            json=_preflight_body(asset_id, feature="retake", durations=[2]),
        )
        assert first.status_code == 200
        (row,) = first.json()["resolutions"]
        named_max = row["maxDurationSeconds"]
        assert named_max > 0
        assert named_max * 2 == int(named_max * 2)

        second = client.post(
            "/api/generations/edit/preflight",
            json=_preflight_body(
                asset_id, feature="retake", durations=[2, named_max]
            ),
        )
        assert second.status_code == 200
        cells = {
            (cell["key"], cell["duration"]): cell
            for cell in second.json()["cells"]
        }
        assert cells[("original", 2)]["ok"] is True
        assert cells[("original", named_max)]["ok"] is True


def test_preflight_out_of_range_start_time_fails_all_cells(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _configure_windows_edit(test_state, vram_gib=12, total_ram_gib=64)
    asset_id = _clip_1080p(test_state, tmp_path)
    with _client(test_state) as client:
        response = client.post(
            "/api/generations/edit/preflight",
            json=_preflight_body(
                asset_id, feature="retake", durations=[2], start_time=999
            ),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["cells"] != []
        for cell in body["cells"]:
            assert cell["ok"] is False
            assert cell["code"] == "INVALID_GENERATION_SPEC"
            assert (
                cell["message"] == "Selection is outside the usable video range"
            )
        for row in body["resolutions"]:
            assert row["maxDurationSeconds"] == 0


def test_preflight_unknown_asset_422s_like_create(
    test_state, create_fake_model_files
) -> None:
    create_fake_model_files()
    _configure_windows_edit(test_state, vram_gib=12, total_ram_gib=64)
    with _client(test_state) as client:
        response = client.post(
            "/api/generations/edit/preflight",
            json=_preflight_body("missing-asset-id"),
        )
        data = assert_http_error(
            response,
            status_code=422,
            code="INPUT_ASSET_NOT_FOUND",
            message="video is no longer available",
        )
        assert "missing-asset" not in data["message"]


def test_preflight_non_video_asset_rejected(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _configure_windows_edit(test_state, vram_gib=12, total_ram_gib=64)
    asset_id = _ingest(test_state, _png(tmp_path / "still.png"))
    with _client(test_state) as client:
        response = client.post(
            "/api/generations/edit/preflight",
            json=_preflight_body(asset_id),
        )
        assert_http_error(
            response,
            status_code=422,
            code="INVALID_GENERATION_SPEC",
            message="video must reference a video asset",
        )


def test_preflight_oversized_video_rejected_like_create(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    from services.media_probe import MAX_VIDEO_BYTES

    create_fake_model_files()
    _configure_windows_edit(test_state, vram_gib=12, total_ram_gib=64)
    asset_id = _clip_1080p(test_state, tmp_path)
    with Path(_store(test_state).get_asset(asset_id).path).open("r+b") as handle:
        handle.truncate(MAX_VIDEO_BYTES + 1)
    with _client(test_state) as client:
        response = client.post(
            "/api/generations/edit/preflight",
            json=_preflight_body(asset_id),
        )
        assert_http_error(response, status_code=400, code="FILE_TOO_LARGE")


def test_preflight_half_size_candidate_422s(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _configure_windows_edit(test_state, vram_gib=12, total_ram_gib=64)
    asset_id = _clip_1080p(test_state, tmp_path)
    with _client(test_state) as client:
        response = client.post(
            "/api/generations/edit/preflight",
            json=_preflight_body(
                asset_id, candidates=[{"key": "720", "width": 1280}]
            ),
        )
        assert response.status_code == 422


def test_preflight_non_finite_duration_422s(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    import json

    create_fake_model_files()
    _configure_windows_edit(test_state, vram_gib=12, total_ram_gib=64)
    asset_id = _clip_1080p(test_state, tmp_path)
    with _client(test_state) as client:
        for bad in (float("nan"), float("inf")):
            response = client.post(
                "/api/generations/edit/preflight",
                content=json.dumps(_preflight_body(asset_id, durations=[bad])),
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code == 422


def test_remote_preflight_matches_desktop(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    from remote.app import create_remote_app

    create_fake_model_files()
    _configure_windows_edit(test_state, vram_gib=12, total_ram_gib=64)
    asset_id = _clip_1080p(test_state, tmp_path)
    body = _preflight_body(
        asset_id,
        feature="retake",
        candidates=[
            {"key": "original", "width": None, "height": None},
            {"key": "720", "width": 1280, "height": 720},
        ],
        durations=[2, 8],
    )
    with _client(test_state) as client:
        desktop = client.post("/api/generations/edit/preflight", json=body)
        assert desktop.status_code == 200
    remote = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(remote) as remote_client:
        response = remote_client.post(
            "/api/generations/edit/preflight",
            json=body,
            headers={"Authorization": "Bearer pair-token"},
        )
        assert response.status_code == 200
        assert response.json() == desktop.json()
