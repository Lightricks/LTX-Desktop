"""Remote generation allowlist: T2V/I2V/A2V; desktop stays unfiltered."""

from __future__ import annotations

import json
import sqlite3
import wave
from pathlib import Path

import pytest
from PIL import Image
from pydantic import JsonValue
from starlette.testclient import TestClient

from app_factory import create_app
from remote.app import create_remote_app
from services.records import OutputSpec
from services.sqlite_store import SqliteStore
from tests.http_error_assertions import assert_http_error

AUTH = {"Authorization": "Bearer pair-token"}
_APPROVED_PARAM_KEYS = frozenset(
    {
        "prompt",
        "model",
        "resolution",
        "duration",
        "fps",
        "aspectRatio",
        "scale",
        "numFrames",
        "promptProvenance",
        "mode",
        "startTime",
        "seed",
    }
)
_SECRET_PATHS = (
    "/Users/me/secret.png",
    "/tmp/nested.mp4",
    "/models/secret.safetensors",
    "/Users/me/start.png",
    "/Users/me/input.png",
    "/etc/passwd",
)


def _store(test_state) -> SqliteStore:
    return SqliteStore(test_state.config.app_data_dir)


def _spec(prompt: str) -> dict[str, JsonValue]:
    return {"params": {"prompt": prompt, "model": "ltx-2.5-fast"}, "inputs": {}}


def test_remote_lists_allowed_features_and_rejects_others(test_state) -> None:
    store = _store(test_state)
    allowed = store.insert_generation("text-to-video", _spec("fox"))
    recipe = store.insert_generation("cozy-felt", _spec("felt"))
    store.insert_generation("desktop-only-feature", _spec("hidden"))

    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        t2v = client.get(
            "/api/generations",
            params={"feature": "text-to-video"},
            headers=AUTH,
        )
        i2v = client.get(
            "/api/generations",
            params={"feature": "image-to-video"},
            headers=AUTH,
        )
        recipe_list = client.get(
            "/api/generations",
            params={"feature": "cozy-felt"},
            headers=AUTH,
        )
        denied = client.get(
            "/api/generations",
            params={"feature": "desktop-only-feature"},
            headers=AUTH,
        )

    assert t2v.status_code == 200
    body = t2v.json()
    assert len(body) == 1
    assert body[0]["id"] == allowed.id
    assert body[0]["feature"] == "text-to-video"
    assert i2v.status_code == 200
    assert i2v.json() == []
    # LoRA recipes are Home features too, so they are allowed on remote.
    assert recipe_list.status_code == 200
    assert [generation["id"] for generation in recipe_list.json()] == [recipe.id]
    assert_http_error(denied, status_code=404, code="GENERATION_NOT_FOUND")


def test_remote_cannot_fetch_or_operate_unsupported_feature_ids(test_state) -> None:
    store = _store(test_state)
    hidden = store.insert_generation("desktop-only-feature", _spec("hidden"))
    allowed = store.insert_generation("text-to-video", _spec("fox"))

    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        hidden_get = client.get(f"/api/generations/{hidden.id}", headers=AUTH)
        hidden_retry = client.post(
            f"/api/generations/{hidden.id}/retry", headers=AUTH
        )
        hidden_cancel = client.post(
            f"/api/generations/{hidden.id}/cancel", headers=AUTH
        )
        hidden_delete = client.delete(
            f"/api/generations/{hidden.id}", headers=AUTH
        )
        allowed_get = client.get(f"/api/generations/{allowed.id}", headers=AUTH)
        allowed_cancel = client.post(
            f"/api/generations/{allowed.id}/cancel", headers=AUTH
        )
        missing = client.get("/api/generations/not-a-real-id", headers=AUTH)

    for response in (
        hidden_get,
        hidden_retry,
        hidden_cancel,
        hidden_delete,
        missing,
    ):
        assert_http_error(response, status_code=404, code="GENERATION_NOT_FOUND")
    assert allowed_get.status_code == 200
    assert allowed_get.json()["id"] == allowed.id
    assert allowed_cancel.status_code == 200
    assert allowed_cancel.json()["status"] == "cancelled"
    loaded = store.get_generation(hidden.id)
    assert loaded is not None
    assert loaded.status == "queued"


def test_remote_retry_failed_allowed_generation(test_state) -> None:
    store = _store(test_state)
    generation = store.insert_generation("text-to-video", _spec("retry"))
    claimed = store.claim_next_queued()
    assert claimed is not None
    store.mark_failed(
        generation.id,
        "OUTPUT_UNREADABLE",
        attempt_count=claimed.attempt_count,
    )

    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        retried = client.post(
            f"/api/generations/{generation.id}/retry", headers=AUTH
        )

    assert retried.status_code == 200, retried.text
    body = retried.json()
    assert body["id"] == generation.id
    assert body["status"] == "queued"
    assert body["feature"] == "text-to-video"
    _assert_no_path_keys(body)
    loaded = store.get_generation(generation.id)
    assert loaded is not None
    assert loaded.status == "queued"


def test_desktop_still_lists_and_operates_unsupported_feature(test_state) -> None:
    store = _store(test_state)
    assert test_state.generation.try_reserve_generation_start() is True
    try:
        generation = store.insert_generation("desktop-only-feature", _spec("desktop"))
        app = create_app(handler=test_state)
        with TestClient(app) as client:
            listed = client.get(
                "/api/generations", params={"feature": "desktop-only-feature"}
            )
            fetched = client.get(f"/api/generations/{generation.id}")
            cancelled = client.post(f"/api/generations/{generation.id}/cancel")
        assert listed.status_code == 200
        assert listed.json()[0]["id"] == generation.id
        assert fetched.status_code == 200
        assert fetched.json()["feature"] == "desktop-only-feature"
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"
    finally:
        test_state.generation.release_generation_start_reservation()


def test_recent_features_match_on_desktop_and_remote(test_state) -> None:
    store = _store(test_state)
    older = store.insert_generation("text-to-video", _spec("older"))
    newer = store.insert_generation("extend", _spec("newer"))
    hidden = store.insert_generation("desktop-only-feature", _spec("hidden"))
    conn = sqlite3.connect(test_state.config.app_data_dir / "store.sqlite3")
    conn.executemany(
        "UPDATE generations SET created_at = ? WHERE id = ?",
        [(1, older.id), (2, newer.id), (3, hidden.id)],
    )
    conn.commit()
    conn.close()

    desktop = create_app(handler=test_state)
    remote = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(desktop) as desktop_client, TestClient(remote) as remote_client:
        desktop_recent = desktop_client.get("/api/generations/recent-features")
        remote_recent = remote_client.get(
            "/api/generations/recent-features", headers=AUTH
        )
        remote_open = remote_client.get("/api/generations/recent-features")

    assert desktop_recent.status_code == 200
    assert remote_recent.status_code == 200
    assert remote_open.status_code == 401
    assert desktop_recent.json() == ["extend", "text-to-video"]
    assert remote_recent.json() == desktop_recent.json()


def test_remote_creates_lora_recipe_and_strips_ref(
    test_state, monkeypatch, create_fake_model_files
) -> None:
    from runtime_config.model_download_specs import resolve_lora_path

    # Pin a device cozy-felt supports; the gate uses the real accelerator, "cpu" on CI.
    from handlers import queued_generation_handler

    create_fake_model_files()
    monkeypatch.setattr(queued_generation_handler, "accelerator_backend", lambda: "mps")

    weights = resolve_lora_path(
        test_state.config.default_models_dir, "cozy-felt-style", "CozyFelt.safetensors"
    )
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"\x00" * 64)

    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        created = client.post(
            "/api/generations/recipes/cozy-felt",
            headers=AUTH,
            json={
                "params": {
                    "prompt": "a felt fox",
                    "model": "ltx-2.5-fast",
                    "catalogId": "cozy-felt-style",
                    "scale": 1.0,
                    "variantId": "default",
                    "resolution": "720p",
                    "duration": 8,
                    "fps": 24,
                }
            },
        )

    assert created.status_code == 200, created.text
    body = created.json()
    assert body["feature"] == "cozy-felt"
    assert body["status"] == "queued"
    # The resolved LoRA ref never reaches the phone (path-free RemoteGeneration DTO),
    # but the strength is projected up so the slider can be restored.
    assert body["spec"]["params"]["prompt"] == "a felt fox"
    assert body["spec"]["params"]["scale"] == 1.0
    assert "loras" not in body["spec"]["params"]
    _assert_no_secret_paths(body)
    _assert_no_path_keys(body)
    # But it is persisted server-side for execution.
    stored = _store(test_state).get_generation(body["id"])
    assert stored is not None
    stored_params = stored.spec["params"]
    assert isinstance(stored_params, dict)
    assert stored_params["loras"][0]["catalogId"] == "cozy-felt-style"


def test_remote_creates_dolly_in_and_projects_start_frame(
    test_state, monkeypatch, create_fake_model_files, tmp_path: Path
) -> None:
    from runtime_config.model_download_specs import resolve_lora_path

    from handlers import queued_generation_handler

    create_fake_model_files()
    monkeypatch.setattr(queued_generation_handler, "accelerator_backend", lambda: "mps")

    weights = resolve_lora_path(
        test_state.config.default_models_dir,
        "dolly-in",
        "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
    )
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"\x00" * 64)

    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        start = client.post(
            "/api/assets/upload",
            headers=AUTH,
            files={
                "file": (
                    "start.png",
                    _png(tmp_path / "dolly-start.png").read_bytes(),
                    "image/png",
                )
            },
        )
        assert start.status_code == 200, start.text
        start_id = start.json()["id"]
        created = client.post(
            "/api/generations/recipes/dolly-in",
            headers=AUTH,
            json={
                "params": {
                    "prompt": "push in on her face",
                    "model": "ltx-2.5-fast",
                    "catalogId": "dolly-in",
                    "scale": 1.0,
                    "variantId": "default",
                    "resolution": "720p",
                    "duration": 8,
                    "fps": 24,
                    "aspectRatio": "auto",
                },
                "inputs": {"startFrame": {"assetId": start_id}},
            },
        )

    assert created.status_code == 200, created.text
    body = created.json()
    assert body["feature"] == "dolly-in"
    assert body["spec"]["params"]["prompt"] == "push in on her face"
    assert body["spec"]["params"]["scale"] == 1.0
    assert body["spec"]["inputs"]["startFrame"]["assetId"] == start_id
    assert "loras" not in body["spec"]["params"]
    _assert_no_secret_paths(body)
    _assert_no_path_keys(body)


def test_remote_creates_transition_and_projects_end_frame(
    test_state, monkeypatch, create_fake_model_files, tmp_path: Path
) -> None:
    from runtime_config.model_download_specs import resolve_lora_path

    from handlers import queued_generation_handler

    create_fake_model_files()
    monkeypatch.setattr(queued_generation_handler, "accelerator_backend", lambda: "mps")

    weights = resolve_lora_path(
        test_state.config.default_models_dir,
        "transition",
        "ltx2.3-transition.safetensors",
    )
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"\x00" * 64)

    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        start = client.post(
            "/api/assets/upload",
            headers=AUTH,
            files={
                "file": (
                    "start.png",
                    _png(tmp_path / "transition-start.png").read_bytes(),
                    "image/png",
                )
            },
        )
        end = client.post(
            "/api/assets/upload",
            headers=AUTH,
            files={
                "file": (
                    "end.png",
                    _png(tmp_path / "transition-end.png").read_bytes(),
                    "image/png",
                )
            },
        )
        assert start.status_code == 200, start.text
        assert end.status_code == 200, end.text
        start_id = start.json()["id"]
        end_id = end.json()["id"]
        created = client.post(
            "/api/generations/recipes/transition",
            headers=AUTH,
            json={
                "params": {
                    "prompt": "a wave becomes a mountain",
                    "model": "ltx-2.5-fast",
                    "catalogId": "transition",
                    "scale": 1.0,
                    "variantId": "default",
                    "resolution": "720p",
                    "duration": 8,
                    "fps": 24,
                    "aspectRatio": "auto",
                },
                "inputs": {
                    "startFrame": {"assetId": start_id},
                    "endFrame": {"assetId": end_id},
                },
            },
        )

    assert created.status_code == 200, created.text
    body = created.json()
    assert body["feature"] == "transition"
    assert body["spec"]["inputs"]["startFrame"]["assetId"] == start_id
    assert body["spec"]["inputs"]["endFrame"]["assetId"] == end_id
    assert "loras" not in body["spec"]["params"]
    _assert_no_secret_paths(body)
    _assert_no_path_keys(body)


def test_remote_generation_openapi_is_queued_only(test_state) -> None:
    schema = create_remote_app(handler=test_state, remote_token="pair-token").openapi()
    http_methods = {"get", "post", "put", "delete", "head", "patch"}
    generation_paths = {
        path: sorted(method for method in methods if method in http_methods)
        for path, methods in schema["paths"].items()
        if path.startswith("/api/generation")
    }
    assert generation_paths == {
        "/api/generation-queue": ["get"],
        "/api/generation-queue/done/clear": ["post"],
        "/api/generation-queue/done/{generation_id}/dismiss": ["post"],
        "/api/generation-queue/done/{generation_id}/seen": ["post"],
        "/api/generation-queue/failed/clear": ["post"],
        "/api/generation-queue/reorder": ["post"],
        "/api/generation-seed": ["get", "post"],
        "/api/generations": ["get"],
        "/api/generations/audio-to-video": ["post"],
        "/api/generations/ic-lora-recipes/{recipe_id}": ["post"],
        "/api/generations/extend": ["post"],
        "/api/generations/image-to-video": ["post"],
        "/api/generations/recipes/{recipe_id}": ["post"],
        "/api/generations/recent-features": ["get"],
        "/api/generations/retake": ["post"],
        "/api/generations/text-to-video": ["post"],
        "/api/generations/{generation_id}": ["delete", "get"],
        "/api/generations/{generation_id}/cancel": ["post"],
        "/api/generations/{generation_id}/retry": ["post"],
    }
    schemas = schema["components"]["schemas"]
    assert "audio" in schemas["RemoteVideoInputs"]["properties"]
    assert "video" in schemas["RemoteVideoInputs"]["properties"]
    remote_asset = schemas["RemoteAsset"]["properties"]
    remote_generation = schemas["RemoteGeneration"]["properties"]
    assert "path" not in remote_asset
    assert "thumbnail_path" not in remote_asset
    assert "path" not in remote_generation
    assert schemas["RemoteGeneration"]["properties"]["outputs"]["items"][
        "$ref"
    ].endswith("/RemoteAsset")
    assert schemas["RemoteQueueEntry"]["properties"]["generation"]["$ref"].endswith(
        "/RemoteGeneration"
    )
    assert schemas["RemoteQueueSnapshot"]["properties"]["queued"]["items"]["$ref"].endswith(
        "/RemoteQueueEntry"
    )
    assert schemas["RemoteQueueSnapshot"]["properties"]["done"]["items"]["$ref"].endswith(
        "/RemoteQueueEntry"
    )
    assert "unseen_ids" in schemas["RemoteQueueSnapshot"]["properties"]
    assert schema["paths"]["/api/generation-queue/reorder"]["post"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"]["$ref"].endswith("/RemoteQueueSnapshot")
    if "AssetCore" in schemas:
        assert "path" not in schemas["AssetCore"]["properties"]
        assert "thumbnail_path" not in schemas["AssetCore"]["properties"]
    assert "Asset" not in schemas or "path" in schemas["Asset"]["properties"]


def test_remote_queue_snapshot_excludes_desktop_file_paths(
    test_state, tmp_path: Path
) -> None:
    store = _store(test_state)
    input_asset = store.ingest_upload(str(_png(tmp_path / "start.png")))
    generation = store.insert_generation(
        "image-to-video",
        {
            "params": {"prompt": "a drummer"},
            "inputs": {"startFrame": {"assetId": input_asset.id}},
        },
    )
    hidden = store.insert_generation("desktop-only-feature", _spec("hidden"))
    app = create_remote_app(handler=test_state, remote_token="pair-token")

    with TestClient(app) as client:
        snapshot = client.get("/api/generation-queue", headers=AUTH)

    assert snapshot.status_code == 200, snapshot.text
    body = snapshot.json()
    assert body["active"] is None
    assert body["done"] == []
    assert body["failed"] == []
    assert body["unseen_ids"] == []
    assert [entry["generation"]["id"] for entry in body["queued"]] == [generation.id]
    assert hidden.id not in str(body)
    assert body["queued"][0]["input_assets"][0]["id"] == input_asset.id
    _assert_remote_asset_payload(body["queued"][0]["input_assets"][0])
    _assert_no_path_keys(body)
    _assert_no_secret_paths(body)


def test_remote_reorder_returns_path_free_queue_snapshot(
    test_state, tmp_path: Path
) -> None:
    store = _store(test_state)
    input_asset = store.ingest_upload(str(_png(tmp_path / "start.png")))
    first = store.insert_generation("text-to-video", _spec("first"))
    second = store.insert_generation(
        "image-to-video",
        {
            "params": {"prompt": "second"},
            "inputs": {"startFrame": {"assetId": input_asset.id}},
        },
    )
    hidden = store.insert_generation("desktop-only-feature", _spec("hidden"))
    app = create_remote_app(handler=test_state, remote_token="pair-token")

    with TestClient(app) as client:
        reordered = client.post(
            "/api/generation-queue/reorder",
            json={"generation_id": second.id, "before_generation_id": first.id},
            headers=AUTH,
        )

    assert reordered.status_code == 200, reordered.text
    body = reordered.json()
    assert "id" not in body
    assert body["active"] is None
    assert [entry["generation"]["id"] for entry in body["queued"]] == [
        second.id,
        first.id,
    ]
    assert hidden.id not in str(body)
    assert body["queued"][0]["generation"]["status"] == "queued"
    assert body["queued"][0]["input_assets"][0]["id"] == input_asset.id
    assert body["queued"][0]["progress"] is None
    assert body["queued"][1]["input_assets"] == []
    _assert_no_path_keys(body)
    _assert_no_secret_paths(body)
    _assert_remote_asset_payload(body["queued"][0]["input_assets"][0])


def _png(path: Path, size: tuple[int, int] = (32, 24)) -> Path:
    Image.new("RGB", size, color=(10, 20, 30)).save(path)
    return path


def _wav(path: Path, *, duration_seconds: float = 8.0) -> Path:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(b"\x00\x00" * int(duration_seconds * 8000))
    return path


def _mp4(path: Path, *, duration_seconds: float = 4.0, fps: int = 8) -> Path:
    import numpy as np
    import imageio.v2 as imageio

    frames = max(1, int(duration_seconds * fps))
    writer = imageio.get_writer(
        str(path), fps=fps, codec="libx264", macro_block_size=None
    )
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    for _ in range(frames):
        writer.append_data(frame)
    writer.close()
    return path


def _assert_no_path_keys(payload: object) -> None:
    if isinstance(payload, dict):
        assert "path" not in payload
        assert "thumbnail_path" not in payload
        for value in payload.values():
            _assert_no_path_keys(value)
        return
    if isinstance(payload, list):
        for item in payload:
            _assert_no_path_keys(item)


def _assert_no_secret_paths(payload: object) -> None:
    blob = json.dumps(payload)
    for secret in _SECRET_PATHS:
        assert secret not in blob


def _assert_remote_asset_payload(payload: object) -> dict[str, object]:
    assert isinstance(payload, dict)
    assert "path" not in payload
    assert "thumbnail_path" not in payload
    assert isinstance(payload.get("id"), str) and payload["id"]
    assert payload.get("media_kind") in {"image", "video", "audio"}
    assert "metadata" in payload
    assert isinstance(payload.get("bytes_url"), str)
    assert str(payload["bytes_url"]).startswith(f"/api/assets/{payload['id']}/bytes?")
    return payload


def test_remote_create_t2v_keeps_form_fields_and_drops_lora_refs(test_state) -> None:
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        created = client.post(
            "/api/generations/text-to-video",
            headers=AUTH,
            json={
                "params": {
                    "prompt": "a fox on a beach",
                    "model": "ltx-2.5-fast",
                    "aspectRatio": "16:9",
                    "resolution": "720p",
                    "duration": 8,
                    "fps": 24,
                    "cameraMotion": "dolly_in",
                    "negativePrompt": "blurry",
                    "loras": [
                        {
                            "ref": "/models/secret.safetensors",
                            "scale": 0.5,
                            "catalogId": "lora-1",
                        }
                    ],
                }
            },
        )
        listed = client.get(
            "/api/generations",
            params={"feature": "text-to-video"},
            headers=AUTH,
        )

    assert created.status_code == 200, created.text
    body = created.json()
    assert body["feature"] == "text-to-video"
    assert body["status"] == "queued"
    params = body["spec"]["params"]
    assert params == {
        "prompt": "a fox on a beach",
        "model": "ltx-2.5-fast",
        "aspectRatio": "16:9",
        "resolution": "720p",
        "duration": 8,
        "fps": 24,
        "scale": 0.5,
        "promptProvenance": "typed",
    }
    assert set(params) <= _APPROVED_PARAM_KEYS
    assert "inputs" not in body["spec"]
    assert "loras" not in body["spec"]["params"]
    _assert_no_secret_paths(body)
    _assert_no_path_keys(body)
    assert listed.status_code == 200
    assert listed.json()[0]["spec"]["params"]["prompt"] == "a fox on a beach"

    stored = _store(test_state).get_generation(body["id"])
    assert stored is not None
    stored_params = stored.spec["params"]
    assert isinstance(stored_params, dict)
    assert stored_params["loras"][0]["ref"] == "/models/secret.safetensors"


@pytest.mark.parametrize("pipeline_id", ["fast", "pro", "pro-2.5", "fast-2.5"])
def test_remote_create_t2v_rejects_pipeline_model_ids(
    test_state, pipeline_id: str
) -> None:
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        created = client.post(
            "/api/generations/text-to-video",
            headers=AUTH,
            json={"params": {"prompt": "a fox", "model": pipeline_id}},
        )
    assert created.status_code == 422
    assert _store(test_state).list_generations("text-to-video") == []


def test_remote_create_t2v_queues_2_3_offering_without_changing_active_settings(
    test_state, create_fake_model_files
) -> None:
    create_fake_model_files()
    create_fake_model_files(model_id="ltx-2.3-22b-distilled-1.1")
    test_state.state.app_settings.active_ltx_model_id = "ltx-2.5-22b-distilled"
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        created = client.post(
            "/api/generations/text-to-video",
            headers=AUTH,
            json={
                "params": {
                    "prompt": "a fox on a beach",
                    "model": "ltx-2.3-fast",
                    "aspectRatio": "16:9",
                    "resolution": "720p",
                    "duration": 8,
                    "fps": 24,
                }
            },
        )

    assert created.status_code == 200, created.text
    body = created.json()
    assert body["status"] == "queued"
    assert body["spec"]["params"]["model"] == "ltx-2.3-fast"
    assert test_state.state.app_settings.active_ltx_model_id == "ltx-2.5-22b-distilled"


def test_remote_create_i2v_keeps_asset_ids_for_form_restore(
    test_state, tmp_path: Path
) -> None:
    png_bytes = _png(tmp_path / "start.png").read_bytes()
    end_bytes = _png(tmp_path / "end.png").read_bytes()
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        start = client.post(
            "/api/assets/upload",
            headers=AUTH,
            files={"file": ("start.png", png_bytes, "image/png")},
        ).json()
        end = client.post(
            "/api/assets/upload",
            headers=AUTH,
            files={"file": ("end.png", end_bytes, "image/png")},
        ).json()
        created = client.post(
            "/api/generations/image-to-video",
            headers=AUTH,
            json={
                "params": {
                    "prompt": "a drummer",
                    "model": "ltx-2.5-fast",
                    "aspectRatio": "auto",
                    "resolution": "1080p",
                    "duration": 8,
                    "fps": 50,
                },
                "inputs": {
                    "startFrame": {"assetId": start["id"]},
                    "endFrame": {"assetId": end["id"]},
                },
            },
        )
        fetched_start = client.get(f"/api/assets/{start['id']}", headers=AUTH)

    assert created.status_code == 200, created.text
    body = created.json()
    assert body["feature"] == "image-to-video"
    assert body["spec"]["params"] == {
        "prompt": "a drummer",
        "model": "ltx-2.5-fast",
        "aspectRatio": "auto",
        "resolution": "1080p",
        "duration": 8,
        "fps": 50,
        "promptProvenance": "typed",
    }
    assert body["spec"]["inputs"] == {
        "startFrame": {"assetId": start["id"]},
        "endFrame": {"assetId": end["id"]},
    }
    _assert_remote_asset_payload(start)
    _assert_remote_asset_payload(fetched_start.json())
    _assert_no_path_keys(body)


def test_remote_create_a2v_keeps_audio_asset_id_and_omits_paths(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    store = _store(test_state)
    audio = store.ingest_upload(str(_wav(tmp_path / "line.wav")))
    start = store.ingest_upload(str(_png(tmp_path / "start.png")))
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        created = client.post(
            "/api/generations/audio-to-video",
            headers=AUTH,
            json={
                "params": {
                    "prompt": "the character speaks",
                    "model": "ltx-2.5-fast",
                    "aspectRatio": "16:9",
                    "resolution": "540p",
                    "fps": 24,
                },
                "inputs": {
                    "audio": {"assetId": audio.id},
                    "startFrame": {"assetId": start.id},
                },
            },
        )
        assert created.status_code == 200, created.text
        body = created.json()
        generation_id = body["id"]
        listed = client.get(
            "/api/generations",
            params={"feature": "audio-to-video"},
            headers=AUTH,
        )
        fetched = client.get(f"/api/generations/{generation_id}", headers=AUTH)

    assert body["feature"] == "audio-to-video"
    assert body["status"] == "queued"
    assert body["spec"]["inputs"] == {
        "audio": {"assetId": audio.id},
        "startFrame": {"assetId": start.id},
    }
    params = body["spec"]["params"]
    assert set(params) <= _APPROVED_PARAM_KEYS
    assert params["numFrames"] == 185
    assert params["fps"] == 24
    assert "duration" not in params
    assert audio.path not in json.dumps(body)
    assert start.path not in json.dumps(body)
    _assert_no_path_keys(body)
    _assert_no_secret_paths(body)
    assert listed.status_code == 200
    assert listed.json()[0]["id"] == generation_id
    assert listed.json()[0]["spec"]["inputs"]["audio"]["assetId"] == audio.id
    _assert_no_path_keys(listed.json())
    assert fetched.status_code == 200
    assert fetched.json()["spec"]["inputs"]["audio"]["assetId"] == audio.id
    _assert_no_path_keys(fetched.json())

    claimed = store.claim_next_queued()
    assert claimed is not None
    store.mark_failed(
        generation_id, "OUTPUT_UNREADABLE", attempt_count=claimed.attempt_count
    )
    with TestClient(app) as client:
        retried = client.post(
            f"/api/generations/{generation_id}/retry", headers=AUTH
        )
        deleted = client.delete(
            f"/api/generations/{generation_id}", headers=AUTH
        )
        missing = client.get(f"/api/generations/{generation_id}", headers=AUTH)

    assert retried.status_code == 200, retried.text
    assert retried.json()["id"] == generation_id
    assert retried.json()["status"] == "queued"
    assert retried.json()["feature"] == "audio-to-video"
    _assert_no_path_keys(retried.json())
    assert deleted.status_code == 200
    assert deleted.json() == {"status": "ok"}
    assert_http_error(missing, status_code=404, code="GENERATION_NOT_FOUND")
    assert store.get_generation(generation_id) is None

    audio_only = store.insert_generation(
        "audio-to-video",
        {
            "params": {"prompt": "speak"},
            "inputs": {"audio": {"assetId": audio.id, "path": audio.path}},
        },
    )
    with TestClient(app) as client:
        projected = client.get(f"/api/generations/{audio_only.id}", headers=AUTH)
    assert projected.status_code == 200, projected.text
    assert projected.json()["spec"]["inputs"] == {"audio": {"assetId": audio.id}}
    assert audio.path not in json.dumps(projected.json())
    _assert_no_path_keys(projected.json())


def test_remote_create_retake_rejects_selection_over_cap(
    test_state, tmp_path: Path
) -> None:
    video = _store(test_state).ingest_upload(str(_mp4(tmp_path / "clip.mp4")))
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        created = client.post(
            "/api/generations/retake",
            headers=AUTH,
            json={
                "params": {
                    "model": "ltx-2.5-fast",
                    "startTime": 0,
                    "duration": 10.5,
                },
                "inputs": {"video": {"assetId": video.id}},
            },
        )
    assert created.status_code == 422
    assert _store(test_state).list_generations("retake") == []


def test_remote_create_retake_keeps_video_asset_id_and_omits_paths(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    store = _store(test_state)
    video = store.ingest_upload(str(_mp4(tmp_path / "clip.mp4")))
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        created = client.post(
            "/api/generations/retake",
            headers=AUTH,
            json={
                "params": {
                    "prompt": "the character turns",
                    "model": "ltx-2.5-fast",
                    "startTime": 0,
                    "duration": 2,
                    "mode": "replace_video",
                },
                "inputs": {"video": {"assetId": video.id}},
            },
        )
        assert created.status_code == 200, created.text
        body = created.json()
        generation_id = body["id"]
        listed = client.get(
            "/api/generations",
            params={"feature": "retake"},
            headers=AUTH,
        )
        fetched = client.get(f"/api/generations/{generation_id}", headers=AUTH)

    assert body["feature"] == "retake"
    assert body["status"] == "queued"
    assert body["spec"]["inputs"] == {"video": {"assetId": video.id}}
    params = body["spec"]["params"]
    assert set(params) <= _APPROVED_PARAM_KEYS
    assert params["model"] == "ltx-2.5-fast"
    assert params["startTime"] == 0
    assert params["duration"] == 2
    assert params["mode"] == "replace_video"
    assert video.path not in json.dumps(body)
    _assert_no_path_keys(body)
    _assert_no_secret_paths(body)
    assert listed.status_code == 200
    assert listed.json()[0]["id"] == generation_id
    assert listed.json()[0]["spec"]["inputs"]["video"]["assetId"] == video.id
    _assert_no_path_keys(listed.json())
    assert fetched.status_code == 200
    assert fetched.json()["spec"]["inputs"]["video"]["assetId"] == video.id
    _assert_no_path_keys(fetched.json())

    leaked = store.insert_generation(
        "retake",
        {
            "params": {"prompt": "turn", "model": "ltx-2.5-fast"},
            "inputs": {"video": {"assetId": video.id, "path": video.path}},
        },
    )
    with TestClient(app) as client:
        projected = client.get(f"/api/generations/{leaked.id}", headers=AUTH)
    assert projected.status_code == 200, projected.text
    assert projected.json()["spec"]["inputs"] == {"video": {"assetId": video.id}}
    assert video.path not in json.dumps(projected.json())
    _assert_no_path_keys(projected.json())


def test_remote_create_extend_keeps_video_asset_id_and_omits_paths(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    store = _store(test_state)
    video = store.ingest_upload(str(_mp4(tmp_path / "clip.mp4")))
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        created = client.post(
            "/api/generations/extend",
            headers=AUTH,
            json={
                "params": {
                    "prompt": "the scene continues",
                    "model": "ltx-2.5-fast",
                    "duration": 4,
                    "mode": "start",
                },
                "inputs": {"video": {"assetId": video.id}},
            },
        )
        assert created.status_code == 200, created.text
        body = created.json()
        generation_id = body["id"]
        listed = client.get(
            "/api/generations",
            params={"feature": "extend"},
            headers=AUTH,
        )
        fetched = client.get(f"/api/generations/{generation_id}", headers=AUTH)

    assert body["feature"] == "extend"
    assert body["status"] == "queued"
    assert body["spec"]["inputs"] == {"video": {"assetId": video.id}}
    params = body["spec"]["params"]
    assert set(params) <= _APPROVED_PARAM_KEYS
    assert params["model"] == "ltx-2.5-fast"
    assert params["duration"] == 4
    assert params["mode"] == "start"
    assert video.path not in json.dumps(body)
    _assert_no_path_keys(body)
    _assert_no_secret_paths(body)
    assert listed.status_code == 200
    assert listed.json()[0]["id"] == generation_id
    assert listed.json()[0]["spec"]["inputs"]["video"]["assetId"] == video.id
    _assert_no_path_keys(listed.json())
    assert fetched.status_code == 200
    assert fetched.json()["spec"]["inputs"]["video"]["assetId"] == video.id
    _assert_no_path_keys(fetched.json())

    leaked = store.insert_generation(
        "extend",
        {
            "params": {"prompt": "continue", "model": "ltx-2.5-fast"},
            "inputs": {"video": {"assetId": video.id, "path": video.path}},
        },
    )
    with TestClient(app) as client:
        projected = client.get(f"/api/generations/{leaked.id}", headers=AUTH)
    assert projected.status_code == 200, projected.text
    assert projected.json()["spec"]["inputs"] == {"video": {"assetId": video.id}}
    assert video.path not in json.dumps(projected.json())
    _assert_no_path_keys(projected.json())


def test_remote_projects_an_integer_seed_and_drops_other_values(test_state) -> None:
    store = _store(test_state)
    seeded = store.insert_generation(
        "text-to-video",
        {"params": {"prompt": "fox", "model": "ltx-2.5-fast", "seed": 1234}, "inputs": {}},
    )
    junk = store.insert_generation(
        "text-to-video",
        {"params": {"prompt": "fox", "seed": "/Users/me/secret"}, "inputs": {}},
    )
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        seeded_body = client.get(f"/api/generations/{seeded.id}", headers=AUTH).json()
        junk_body = client.get(f"/api/generations/{junk.id}", headers=AUTH).json()

    assert seeded_body["spec"]["params"]["seed"] == 1234
    assert "seed" not in junk_body["spec"]["params"]
    _assert_no_secret_paths(junk_body)


def test_remote_get_strips_nested_path_like_spec_values(
    test_state, tmp_path: Path
) -> None:
    store = _store(test_state)
    start = store.ingest_upload(str(_png(tmp_path / "start.png")))
    end = store.ingest_upload(str(_png(tmp_path / "end.png")))
    generation = store.insert_generation(
        "image-to-video",
        {
            "params": {
                "prompt": "a drummer",
                "model": "fast-2.5",
                "aspectRatio": "auto",
                "resolution": "720p",
                "duration": 8,
                "fps": 24,
                "imagePath": "/Users/me/secret.png",
                "nested": {"path": "/tmp/nested.mp4"},
                "loras": [
                    {
                        "ref": "/models/secret.safetensors",
                        "scale": 0.8,
                        "catalogId": "cat-1",
                    }
                ],
                "cameraMotion": "dolly_in",
            },
            "inputs": {
                "startFrame": {
                    "assetId": start.id,
                    "path": "/Users/me/start.png",
                },
                "endFrame": {"assetId": end.id},
                "imagePath": "/Users/me/input.png",
                "extra": {"path": "/etc/passwd"},
            },
        },
    )

    remote = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(remote) as client:
        fetched = client.get(f"/api/generations/{generation.id}", headers=AUTH)
        listed = client.get(
            "/api/generations",
            params={"feature": "image-to-video"},
            headers=AUTH,
        )
        cancelled = client.post(
            f"/api/generations/{generation.id}/cancel", headers=AUTH
        )

    assert fetched.status_code == 200
    body = fetched.json()
    assert body["spec"]["params"] == {
        "prompt": "a drummer",
        "model": "fast-2.5",
        "aspectRatio": "auto",
        "resolution": "720p",
        "duration": 8,
        "fps": 24,
        "scale": 0.8,
    }
    assert body["spec"]["inputs"] == {
        "startFrame": {"assetId": start.id},
        "endFrame": {"assetId": end.id},
    }
    _assert_no_secret_paths(body)
    _assert_no_path_keys(body)
    assert listed.status_code == 200
    _assert_no_secret_paths(listed.json())
    assert cancelled.status_code == 200
    _assert_no_secret_paths(cancelled.json())
    assert cancelled.json()["status"] == "cancelled"

    assert test_state.generation.try_reserve_generation_start() is True
    try:
        desktop = create_app(handler=test_state)
        with TestClient(desktop) as client:
            desktop_body = client.get(f"/api/generations/{generation.id}").json()
        desktop_params = desktop_body["spec"]["params"]
        assert desktop_params["imagePath"] == "/Users/me/secret.png"
        assert desktop_params["loras"][0]["ref"] == "/models/secret.safetensors"
        assert desktop_body["spec"]["inputs"]["extra"]["path"] == "/etc/passwd"
    finally:
        test_state.generation.release_generation_start_reservation()


def test_remote_generation_outputs_omit_paths_and_bytes_still_work(
    test_state, tmp_path: Path
) -> None:
    store = _store(test_state)
    created = store.insert_generation("text-to-video", _spec("output"))
    claimed = store.claim_next_queued()
    assert claimed is not None
    asset_id, dest = store.allocate_output_path("image", "image/png")
    png_path = Path(dest)
    Image.new("RGB", (8, 8), color=(4, 5, 6)).save(png_path)
    png_bytes = png_path.read_bytes()
    store.mark_succeeded(
        created.id,
        [
            OutputSpec(
                asset_id=asset_id,
                dest_path=dest,
                ordinal=0,
                mime_type="image/png",
                name="out.png",
            )
        ],
        attempt_count=claimed.attempt_count,
    )

    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        fetched = client.get(f"/api/generations/{created.id}", headers=AUTH)
        assert fetched.status_code == 200
        body = fetched.json()
        assert body["status"] == "succeeded"
        assert len(body["outputs"]) == 1
        output = _assert_remote_asset_payload(body["outputs"][0])
        output_id = output["id"]
        assert isinstance(output_id, str)
        bytes_response = client.get(str(output["bytes_url"]))
        asset_json = client.get(f"/api/assets/{output_id}", headers=AUTH)

    _assert_no_path_keys(body)
    assert bytes_response.status_code == 200
    assert bytes_response.content == png_bytes
    _assert_remote_asset_payload(asset_json.json())


def test_desktop_asset_and_generation_keep_filesystem_paths(
    test_state, tmp_path: Path
) -> None:
    source = _png(tmp_path / "desktop.png")
    assert test_state.generation.try_reserve_generation_start() is True
    try:
        app = create_app(handler=test_state)
        with TestClient(app) as client:
            ingested = client.post("/api/assets", json={"path": str(source)})
            assert ingested.status_code == 200
            asset = ingested.json()
            assert Path(asset["path"]).is_file()
            assert "thumbnail_path" in asset
            created = client.post(
                "/api/generations/text-to-video",
                json={"params": {"prompt": "desktop fox", "model": "ltx-2.5-fast"}},
            )
            assert created.status_code == 200
            generation = created.json()
            assert generation["spec"]["params"]["prompt"] == "desktop fox"
            assert "loras" in generation["spec"]["params"]
            fetched_asset = client.get(f"/api/assets/{asset['id']}")
        assert fetched_asset.status_code == 200
        assert Path(fetched_asset.json()["path"]).is_file()
    finally:
        test_state.generation.release_generation_start_reservation()
