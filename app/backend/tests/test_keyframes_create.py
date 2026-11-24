"""POST /api/generations/keyframes: the flag gate, what the row stores, and the refusals."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image
from pydantic import JsonValue
from starlette.testclient import TestClient

from remote.app import create_remote_app
from services.sqlite_store import SqliteStore
from tests.http_error_assertions import assert_http_error

_AUTH = {"Authorization": "Bearer pair-token"}


def _rows(test_state) -> list[object]:
    return list(SqliteStore(test_state.config.app_data_dir).list_generations("keyframes"))


def _asset(client: TestClient, tmp_path: Path, name: str) -> tuple[str, str]:
    path = tmp_path / name
    Image.new("RGB", (16, 16), color=(1, 2, 3)).save(path)
    created = client.post("/api/assets", json={"path": str(path)})
    assert created.status_code == 200, created.text
    return created.json()["id"], str(path)


def _body(
    keyframes: list[dict[str, JsonValue]], *, duration: int | None = 5
) -> dict[str, JsonValue]:
    return {
        "params": {
            "prompt": "a fox runs",
            "model": "ltx-2.5-fast",
            "aspectRatio": "16:9",
            "resolution": "540p",
            "duration": duration,
            "fps": 24,
        },
        "inputs": {"keyframes": keyframes},
    }


def _enable(client: TestClient) -> None:
    flags = client.request("PATCH", "/api/feature-flags", json={"keyframes": True})
    assert flags.status_code == 200, flags.text


def test_create_refuses_while_the_flag_is_off_and_queues_nothing(
    client, test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    asset_id, _ = _asset(client, tmp_path, "a.png")

    refused = client.post(
        "/api/generations/keyframes",
        json=_body([{"assetId": asset_id, "frameIndex": 0, "strength": 1}]),
    )

    assert_http_error(refused, status_code=422, code="KEYFRAMES_NOT_ENABLED")
    assert _rows(test_state) == []


def test_create_stores_asset_ids_and_never_a_file_path(
    client, test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _enable(client)
    first_id, first_path = _asset(client, tmp_path, "a.png")
    last_id, last_path = _asset(client, tmp_path, "b.png")
    keyframes: list[dict[str, JsonValue]] = [
        {"assetId": first_id, "frameIndex": 0, "strength": 1},
        {"assetId": last_id, "frameIndex": 120, "strength": 0.5},
    ]

    created = client.post("/api/generations/keyframes", json=_body(keyframes))

    assert created.status_code == 200, created.text
    spec = created.json()["spec"]
    assert spec["inputs"] == {"keyframes": keyframes}
    stored = json.dumps(spec)
    assert first_path not in stored
    assert last_path not in stored
    assert len(_rows(test_state)) == 1


@pytest.mark.parametrize(
    ("case", "frames", "duration"),
    [
        ("two stills on one frame", [10, 10], 5),
        # 5 s at 24 fps is 121 frames, so the last frame index is 120.
        ("a frame past the clip end", [0, 121], 5),
        ("no clip length", [0, 10], None),
        ("more stills than the limit", list(range(0, 11 * 8, 8)), 8),
    ],
)
def test_create_refuses_bad_timelines_and_queues_nothing(
    client,
    test_state,
    create_fake_model_files,
    tmp_path: Path,
    case: str,
    frames: list[int],
    duration: int | None,
) -> None:
    create_fake_model_files()
    _enable(client)
    asset_id, _ = _asset(client, tmp_path, "a.png")
    keyframes: list[dict[str, JsonValue]] = [
        {"assetId": asset_id, "frameIndex": frame, "strength": 1} for frame in frames
    ]

    refused = client.post("/api/generations/keyframes", json=_body(keyframes, duration=duration))

    assert refused.status_code == 422, case
    assert _rows(test_state) == []


def test_remote_create_is_blocked_by_the_flag_and_then_hides_paths(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    path = tmp_path / "a.png"
    Image.new("RGB", (16, 16), color=(1, 2, 3)).save(path)
    with TestClient(app) as remote:
        uploaded = remote.post(
            "/api/assets/upload",
            headers=_AUTH,
            files={"file": ("a.png", path.read_bytes(), "image/png")},
        )
        asset_id = uploaded.json()["id"]
        body = _body([{"assetId": asset_id, "frameIndex": 0, "strength": 1}])

        refused = remote.post("/api/generations/keyframes", headers=_AUTH, json=body)
        remote.request("PATCH", "/api/feature-flags", headers=_AUTH, json={"keyframes": True})
        created = remote.post("/api/generations/keyframes", headers=_AUTH, json=body)

    assert_http_error(refused, status_code=422, code="KEYFRAMES_NOT_ENABLED")
    assert created.status_code == 200, created.text
    assert created.json()["spec"]["inputs"] == {
        "keyframes": [{"assetId": asset_id, "frameIndex": 0, "strength": 1}]
    }
    assert "path" not in json.dumps(created.json()["spec"])
