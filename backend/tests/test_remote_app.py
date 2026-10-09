"""Remote FastAPI: public health/static, fail-closed Bearer on /api, LAN hosts only."""

from __future__ import annotations

import asyncio
import subprocess
import tempfile
import wave
from pathlib import Path

import imageio_ffmpeg
import pytest
from PIL import Image
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.testclient import TestClient

from _routes.assets import (
    _MULTIPART_OVERHEAD_BYTES,
    _UPLOAD_TEMP_PREFIX,
    _upload_body_exceeds_limit,
)
from remote.app import SpaStaticFiles, create_remote_app
from remote.redact import redact_asset_paths
from services.media_probe import ALLOWED_IMAGE_SUFFIXES, MAX_IMAGE_BYTES, MAX_VIDEO_BYTES
from services.records import AssetRecord, OutputSpec, UnavailableError
from services.sqlite_store import SqliteStore
from tests.http_error_assertions import assert_http_error


def _png(path: Path, size: tuple[int, int] = (32, 24)) -> Path:
    Image.new("RGB", size, color=(10, 20, 30)).save(path)
    return path


def _wav(path: Path, *, duration_seconds: float = 0.1) -> Path:
    frame_count = max(1, int(duration_seconds * 8000))
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(b"\x00\x00" * frame_count)
    return path


def _encoded_audio(path: Path) -> Path:
    wav_path = path.with_name(f"{path.stem}-src.wav")
    _wav(wav_path)
    suffix = path.suffix.lower()
    codec = {".mp3": "libmp3lame", ".m4a": "aac"}[suffix]
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-i",
            str(wav_path),
            "-c:a",
            codec,
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _mp4(path: Path) -> Path:
    import imageio.v2 as imageio
    import numpy as np

    writer = imageio.get_writer(
        str(path), fps=8, codec="libx264", macro_block_size=None
    )
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    for _ in range(8):
        writer.append_data(frame)
    writer.close()
    return path


AUTH = {"Authorization": "Bearer pair-token"}
WRONG_AUTH = {"Authorization": "Bearer wrong"}
_IMAGE_SAVE_FORMAT = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".webp": "WEBP",
}


def _bytes_url(client: TestClient, asset_id: str) -> str:
    fetched = client.get(f"/api/assets/{asset_id}", headers=AUTH)
    assert fetched.status_code == 200
    url = fetched.json()["bytes_url"]
    assert isinstance(url, str)
    assert url.startswith(f"/api/assets/{asset_id}/bytes?")
    return url


def _store(test_state) -> SqliteStore:
    return SqliteStore(test_state.config.app_data_dir)


def _succeed_png(test_state, *, name: str = "out.png") -> AssetRecord:
    store = _store(test_state)
    gen = store.insert_generation("text-to-video", {"params": {}, "inputs": {}})
    claimed = store.claim_next_queued()
    assert claimed is not None
    asset_id, dest = store.allocate_output_path("image", "image/png")
    Image.new("RGB", (8, 8)).save(dest)
    done = store.mark_succeeded(
        gen.id,
        [
            OutputSpec(
                asset_id=asset_id,
                dest_path=dest,
                ordinal=0,
                mime_type="image/png",
                name=name,
            )
        ],
        attempt_count=claimed.attempt_count,
    )
    return done.outputs[0]


def _persistent_asset_files(test_state) -> list[Path]:
    assets_dir = test_state.config.app_data_dir / "assets"
    if not assets_dir.exists():
        return []
    return sorted(path for path in assets_dir.rglob("*") if path.is_file())


def _upload_temp_files() -> list[Path]:
    return sorted(Path(tempfile.gettempdir()).glob(f"{_UPLOAD_TEMP_PREFIX}*"))


def _assert_no_leaked_upload_files(test_state) -> None:
    assert _persistent_asset_files(test_state) == []
    assert _upload_temp_files() == []


def test_health_is_public(test_state) -> None:
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_serves_bundled_index_when_present(test_state, tmp_path) -> None:
    (tmp_path / "index.html").write_text("<html>remote-ok</html>", encoding="utf-8")
    app = create_remote_app(
        handler=test_state,
        remote_token="pair-token",
        client_dir=tmp_path,
    )
    with TestClient(app) as client:
        response = client.get("/")
        authed_api = client.get(
            "/api/generations",
            params={"feature": "text-to-video"},
            headers=AUTH,
        )
        authed_session = client.get(
            "/api/session",
            headers=AUTH,
        )
    assert response.status_code == 200
    assert "remote-ok" in response.text
    assert authed_api.status_code == 200
    assert authed_api.json() == []
    assert "remote-ok" not in authed_api.text
    assert authed_session.status_code == 200
    assert authed_session.json() == {"status": "ok"}
    app = create_remote_app(handler=test_state, remote_token="pair-token", client_dir=None)
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "Remote client is not built" in response.text


def test_spa_fallback_does_not_mask_api_404s(test_state, tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<html>remote-ok</html>", encoding="utf-8")
    app = create_remote_app(
        handler=test_state,
        remote_token="pair-token",
        client_dir=tmp_path,
    )
    with TestClient(app) as client:
        spa = client.get("/text-to-video")
        unknown_api = client.get("/api/not-a-route", headers=AUTH)
    assert spa.status_code == 200
    assert "remote-ok" in spa.text
    assert spa.headers["content-type"].startswith("text/html")
    assert "remote-ok" not in unknown_api.text
    assert "text/html" not in unknown_api.headers.get("content-type", "")
    assert_http_error(
        unknown_api, status_code=404, code="HTTP_404", message="Not Found"
    )
    files = SpaStaticFiles(directory=str(tmp_path), html=True)
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/not-a-route",
        "headers": [],
    }
    for mounted_path in ("api/not-a-route", "api\\not-a-route"):
        with pytest.raises(StarletteHTTPException) as raised:
            asyncio.run(files.get_response(mounted_path, scope))
        assert raised.value.status_code == 404


def test_api_requires_bearer_and_exposes_queued_generation_routes(test_state) -> None:
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        unauth = client.get("/api/generations", params={"feature": "text-to-video"})
        wrong = client.get(
            "/api/generations",
            params={"feature": "text-to-video"},
            headers=WRONG_AUTH,
        )
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
        specs = client.get("/api/generate/models-specs", headers=AUTH)
        create_t2v = client.post("/api/generations/text-to-video", headers=AUTH)
        create_i2v = client.post("/api/generations/image-to-video", headers=AUTH)
        legacy_generate = client.post("/api/generate", headers=AUTH, json={})
    assert unauth.status_code == 401
    assert wrong.status_code == 401
    assert t2v.status_code == 200
    assert t2v.json() == []
    assert i2v.status_code == 200
    assert i2v.json() == []
    assert specs.status_code == 200
    assert "local_models" in specs.json()
    assert "downloaded_local_models" in specs.json()
    assert create_t2v.status_code == 422
    assert create_i2v.status_code == 422
    assert legacy_generate.status_code == 404


def test_settings_and_path_ingest_stay_off_remote(test_state, tmp_path: Path) -> None:
    source = _png(tmp_path / "secret.png")
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        settings = client.post(
            "/api/settings",
            headers=AUTH,
            json={"remoteExposure": "lan"},
        )
        active_model = client.post(
            "/api/models/active-ltx-model",
            headers=AUTH,
            json={"modelId": "ltx-2.5-22b-distilled"},
        )
        ingest = client.post(
            "/api/assets",
            headers=AUTH,
            json={"path": str(source)},
        )
    assert settings.status_code == 404
    assert active_model.status_code == 404
    assert ingest.status_code == 405


def test_remote_can_save_dashboard_selection(test_state) -> None:
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    selected = {
        "range": "all",
        "models": ["ltx-2.5-fast"],
        "resolutions": ["540p"],
        "aspectRatios": ["16:9"],
        "fps": ["24"],
    }
    with TestClient(app) as client:
        assert client.get("/api/settings", headers=AUTH).status_code == 404
        saved = client.post("/api/stats/activity-dashboard-selections", headers=AUTH, json=selected)
        loaded = client.get("/api/stats/activity-dashboard-selections", headers=AUTH)
    assert saved.status_code == 200
    assert saved.json() == selected
    assert loaded.status_code == 200
    assert loaded.json() == selected


def test_remote_stats_surface_is_the_two_phone_routes(test_state) -> None:
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    stats_paths = {path for path in app.openapi()["paths"] if path.startswith("/api/stats/")}
    assert stats_paths == {"/api/stats/dashboard", "/api/stats/activity-dashboard-selections"}


def test_remote_cannot_grow_settings_with_an_unbounded_selection(test_state) -> None:
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        too_many = client.post(
            "/api/stats/activity-dashboard-selections",
            headers=AUTH,
            json={"range": "7d", "models": [f"m{i}" for i in range(33)]},
        )
        too_long = client.post(
            "/api/stats/activity-dashboard-selections",
            headers=AUTH,
            json={"range": "7d", "models": ["x" * 65]},
        )
    assert too_many.status_code == 422
    assert too_long.status_code == 422


def test_remote_openapi_omits_desktop_status_route(test_state) -> None:
    schema = create_remote_app(handler=test_state, remote_token="pair-token").openapi()
    assert "/api/remote/status" not in schema["paths"]


def test_remote_asset_openapi_keeps_phone_routes_with_unique_operation_ids(
    test_state,
) -> None:
    schema = create_remote_app(handler=test_state, remote_token="pair-token").openapi()
    http_methods = {"get", "post", "put", "delete", "head", "patch"}
    asset_paths = {
        path: sorted(method for method in methods if method in http_methods)
        for path, methods in schema["paths"].items()
        if path.startswith("/api/assets")
    }
    assert asset_paths == {
        "/api/assets": ["get"],
        "/api/assets/upload": ["post"],
        "/api/assets/{asset_id}": ["get"],
        "/api/assets/{asset_id}/bytes": ["get", "head"],
        "/api/assets/{asset_id}/extract-audio": ["post"],
        "/api/assets/{asset_id}/thumbnail/bytes": ["get", "head"],
        "/api/assets/{asset_id}/trim-audio": ["post"],
        "/api/assets/{asset_id}/trim-video": ["post"],
    }
    assert "delete" not in {
        method
        for methods in asset_paths.values()
        for method in methods
    }
    operation_ids = [
        operation["operationId"]
        for path_item in schema["paths"].values()
        for method, operation in path_item.items()
        if method in http_methods and "operationId" in operation
    ]
    assert len(operation_ids) == len(set(operation_ids))
    assert "route_list_phone_assets" in operation_ids
    assert "route_delete_asset" not in operation_ids


def test_list_phone_assets_omits_paths_and_rejects_delete(
    test_state, tmp_path: Path
) -> None:
    png_bytes = _png(tmp_path / "phone.png").read_bytes()
    output = _succeed_png(test_state)
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        uploaded = client.post(
            "/api/assets/upload",
            headers=AUTH,
            files={"file": ("phone.png", png_bytes, "image/png")},
        )
        assert uploaded.status_code == 200
        listed = client.get("/api/assets", headers=AUTH)
        assert listed.status_code == 200
        body = listed.json()
        ids = [item["id"] for item in body["items"]]
        assert uploaded.json()["id"] not in ids
        item = body["items"][0]
        assert item["id"] == output.id
        assert item["origin"] == "generated"
        assert "path" not in item
        assert "thumbnail_path" not in item
        assert "in_use" in item
        assert item["has_thumbnail"] is True
        assert item["bytes_url"].startswith(f"/api/assets/{output.id}/bytes?")
        assert item["thumbnail_url"].startswith(
            f"/api/assets/{output.id}/thumbnail/bytes?"
        )
        deleted = client.delete(f"/api/assets/{output.id}", headers=AUTH)
        assert deleted.status_code == 405


def test_list_and_get_phone_assets_do_not_tombstone_missing_files(
    test_state,
) -> None:
    store = _store(test_state)
    gone = _succeed_png(test_state, name="gone.png")
    payload = Path(gone.path).read_bytes()
    Path(gone.path).unlink()
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        listed = client.get("/api/assets", headers=AUTH)
        assert listed.status_code == 200
        assert gone.id not in [item["id"] for item in listed.json()["items"]]
        fetched = client.get(f"/api/assets/{gone.id}", headers=AUTH)
        assert fetched.status_code == 404
    Path(gone.path).write_bytes(payload)
    assert store.get_asset(gone.id) is not None


def test_upload_and_bytes_round_trip_redacts_local_paths(test_state, tmp_path: Path) -> None:
    png_bytes = _png(tmp_path / "phone.png").read_bytes()
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        uploaded = client.post(
            "/api/assets/upload",
            headers=AUTH,
            files={"file": ("phone.png", png_bytes, "image/png")},
        )
        assert uploaded.status_code == 200
        asset = uploaded.json()
        asset_id = asset["id"]
        assert asset["media_kind"] == "image"
        assert "path" not in asset
        assert "thumbnail_path" not in asset
        assert asset["id"]
        assert asset["origin"] == "uploaded"
        assert "metadata" in asset

        fetched = client.get(f"/api/assets/{asset_id}", headers=AUTH)
        assert fetched.status_code == 200
        fetched_asset = fetched.json()
        assert "path" not in fetched_asset
        assert "thumbnail_path" not in fetched_asset
        assert fetched_asset["id"] == asset_id
        assert fetched_asset["bytes_url"].startswith(f"/api/assets/{asset_id}/bytes?")
        assert fetched_asset["thumbnail_url"] is not None
        assert fetched_asset["thumbnail_url"].startswith(
            f"/api/assets/{asset_id}/thumbnail/bytes?"
        )

        unauth_bytes = client.get(f"/api/assets/{asset_id}/bytes")
        query_bytes = client.get(f"/api/assets/{asset_id}/bytes?t=pair-token")
        signed_bytes = client.get(asset["bytes_url"])
        header_bytes = client.get(
            f"/api/assets/{asset_id}/bytes",
            headers=AUTH,
        )
        signed_thumb = (
            client.get(asset["thumbnail_url"])
            if asset["thumbnail_url"] is not None
            else None
        )
        unauth_query = client.get(f"/api/assets/{asset_id}/bytes?t=wrong")

    assert unauth_bytes.status_code == 401
    assert unauth_query.status_code == 401
    assert query_bytes.status_code == 401
    assert signed_bytes.status_code == 200
    assert signed_bytes.content == png_bytes
    assert signed_bytes.headers["content-type"].startswith("image/png")
    assert "inline" in signed_bytes.headers.get("content-disposition", "")
    assert "attachment" not in signed_bytes.headers.get("content-disposition", "")
    assert header_bytes.status_code == 401
    assert signed_thumb is not None
    assert signed_thumb.status_code == 200
    assert signed_thumb.content
    assert signed_thumb.headers["content-type"].startswith("image/jpeg")
    assert "inline" in signed_thumb.headers.get("content-disposition", "")


def test_remote_upload_accepts_allowed_image_suffixes(
    test_state, tmp_path: Path
) -> None:
    assert set(_IMAGE_SAVE_FORMAT) == set(ALLOWED_IMAGE_SUFFIXES)
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        for suffix, fmt in _IMAGE_SAVE_FORMAT.items():
            source = tmp_path / f"ok{suffix}"
            Image.new("RGB", (32, 24), color=(10, 20, 30)).save(source, format=fmt)
            uploaded = client.post(
                "/api/assets/upload",
                headers=AUTH,
                files={"file": (source.name, source.read_bytes())},
            )
            assert uploaded.status_code == 200, uploaded.text
            asset = uploaded.json()
            assert asset["media_kind"] == "image"
            assert "path" not in asset
            assert "thumbnail_path" not in asset
            fetched = client.get(asset["bytes_url"])
            assert fetched.status_code == 200
            assert fetched.content == source.read_bytes()


def test_upload_size_guards_honor_file_bytes_not_multipart_framing() -> None:
    wrapping = 512
    assert wrapping < _MULTIPART_OVERHEAD_BYTES
    assert not _upload_body_exceeds_limit(
        file_size=MAX_IMAGE_BYTES,
        content_length=str(MAX_IMAGE_BYTES + wrapping),
        max_bytes=MAX_IMAGE_BYTES,
    )
    assert _upload_body_exceeds_limit(
        file_size=MAX_IMAGE_BYTES + 1,
        content_length=str(MAX_IMAGE_BYTES + wrapping),
        max_bytes=MAX_IMAGE_BYTES,
    )
    assert not _upload_body_exceeds_limit(
        file_size=None,
        content_length=str(MAX_IMAGE_BYTES + wrapping),
        max_bytes=MAX_IMAGE_BYTES,
    )
    assert not _upload_body_exceeds_limit(
        file_size=None,
        content_length=str(MAX_IMAGE_BYTES + _MULTIPART_OVERHEAD_BYTES),
        max_bytes=MAX_IMAGE_BYTES,
    )
    assert _upload_body_exceeds_limit(
        file_size=None,
        content_length=str(MAX_IMAGE_BYTES + _MULTIPART_OVERHEAD_BYTES + 1),
        max_bytes=MAX_IMAGE_BYTES,
    )


def test_remote_upload_accepts_image_when_content_length_includes_framing(
    test_state, tmp_path: Path
) -> None:
    png_bytes = _png(tmp_path / "ok.png").read_bytes()
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        uploaded = client.post(
            "/api/assets/upload",
            headers={**AUTH, "content-length": str(MAX_IMAGE_BYTES + 512)},
            files={"file": ("ok.png", png_bytes, "image/png")},
        )
    assert uploaded.status_code == 200, uploaded.text
    assert uploaded.json()["media_kind"] == "image"


def test_remote_upload_accepts_wav_mp3_m4a(test_state, tmp_path: Path) -> None:
    samples = (
        (_wav(tmp_path / "input.wav"), "audio/wav"),
        (_encoded_audio(tmp_path / "input.mp3"), "audio/mpeg"),
        (_encoded_audio(tmp_path / "input.m4a"), "audio/mp4"),
    )
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        for source, mime in samples:
            uploaded = client.post(
                "/api/assets/upload",
                headers=AUTH,
                files={"file": (source.name, source.read_bytes(), mime)},
            )
            assert uploaded.status_code == 200, uploaded.text
            asset = uploaded.json()
            assert asset["media_kind"] == "audio"
            assert "path" not in asset
            assert "thumbnail_path" not in asset
            fetched = client.get(asset["bytes_url"])
            assert fetched.status_code == 200
            assert fetched.content == source.read_bytes()


def test_remote_upload_accepts_video(test_state, tmp_path: Path) -> None:
    video_bytes = _mp4(tmp_path / "input.mp4").read_bytes()
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        video = client.post(
            "/api/assets/upload",
            headers=AUTH,
            files={"file": ("input.mp4", video_bytes, "video/mp4")},
        )
        assert video.status_code == 200, video.text
        asset = video.json()
        assert asset["media_kind"] == "video"
        assert "path" not in asset
        assert "thumbnail_path" not in asset
        fetched = client.get(asset["bytes_url"])
        assert fetched.status_code == 200
        assert fetched.content == video_bytes


def test_remote_upload_malformed_audio_hides_filesystem_path(test_state) -> None:
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        response = client.post(
            "/api/assets/upload",
            headers=AUTH,
            files={"file": ("bad.wav", b"not audio at all", "audio/wav")},
        )
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "UNREADABLE_MEDIA"
    assert "/" not in response.text
    assert "\\" not in response.text
    _assert_no_leaked_upload_files(test_state)


def test_remote_trim_audio_returns_remote_asset(test_state, tmp_path: Path) -> None:
    store = _store(test_state)
    audio = store.ingest_upload(
        str(_wav(tmp_path / "line.wav", duration_seconds=2.0))
    )
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        trimmed = client.post(
            f"/api/assets/{audio.id}/trim-audio",
            headers=AUTH,
            json={"startSec": 0, "endSec": 1},
        )
    assert trimmed.status_code == 200, trimmed.text
    asset = trimmed.json()
    assert asset["media_kind"] == "audio"
    assert "path" not in asset
    assert "thumbnail_path" not in asset
    assert 900 <= asset["metadata"]["metadata"]["durationMs"] <= 1100


def test_remote_upload_rejects_image_over_max_bytes(
    test_state, tmp_path: Path
) -> None:
    png_bytes = _png(tmp_path / "ok.png").read_bytes()
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        uploaded = client.post(
            "/api/assets/upload",
            headers={
                **AUTH,
                "content-length": str(
                    MAX_IMAGE_BYTES + _MULTIPART_OVERHEAD_BYTES + 1
                ),
            },
            files={"file": ("huge.png", png_bytes, "image/png")},
        )
    assert_http_error(uploaded, status_code=400, code="FILE_TOO_LARGE")
    _assert_no_leaked_upload_files(test_state)


def test_remote_upload_rejects_video_over_max_bytes(
    test_state, tmp_path: Path
) -> None:
    video_bytes = _mp4(tmp_path / "ok.mp4").read_bytes()
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        uploaded = client.post(
            "/api/assets/upload",
            headers={
                **AUTH,
                "content-length": str(
                    MAX_VIDEO_BYTES + _MULTIPART_OVERHEAD_BYTES + 1
                ),
            },
            files={"file": ("huge.mp4", video_bytes, "video/mp4")},
        )
    assert_http_error(uploaded, status_code=400, code="FILE_TOO_LARGE")
    _assert_no_leaked_upload_files(test_state)


def test_bytes_serves_first_last_frame_audio_and_video_inline(
    test_state, tmp_path: Path
) -> None:
    store = _store(test_state)
    start_path = _png(tmp_path / "start.png")
    end_path = _png(tmp_path / "end.png")
    audio_path = _wav(tmp_path / "input.wav")
    video_path = _mp4(tmp_path / "input.mp4")
    start = store.ingest_upload(str(start_path))
    end = store.ingest_upload(str(end_path))
    audio = store.ingest_upload(str(audio_path))
    video = store.ingest_upload(str(video_path))
    samples = (
        (start, start_path.read_bytes(), "image/"),
        (end, end_path.read_bytes(), "image/"),
        (audio, audio_path.read_bytes(), "audio/"),
        (video, video_path.read_bytes(), "video/"),
    )
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        for record, payload, mime_prefix in samples:
            fetched = client.get(f"/api/assets/{record.id}", headers=AUTH)
            assert fetched.status_code == 200
            asset = fetched.json()
            assert asset["media_kind"] == record.media_kind
            assert "path" not in asset
            assert "thumbnail_path" not in asset
            response = client.get(asset["bytes_url"])
            assert response.status_code == 200
            assert response.content == payload
            assert response.headers["content-type"].startswith(mime_prefix)
            disposition = response.headers.get("content-disposition", "")
            assert "inline" in disposition
            assert "attachment" not in disposition

        head = client.head(_bytes_url(client, video.id))
        ranged = client.get(
            _bytes_url(client, video.id),
            headers={"Range": "bytes=0-15"},
        )

    video_bytes = video_path.read_bytes()
    assert head.status_code == 200
    assert head.content == b""
    assert head.headers["content-type"].startswith("video/")
    assert ranged.status_code == 206
    assert ranged.content == video_bytes[:16]


def test_session_requires_bearer(test_state) -> None:
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        unauth = client.get("/api/session")
        wrong = client.get(
            "/api/session",
            headers=WRONG_AUTH,
        )
        authed = client.get(
            "/api/session",
            headers=AUTH,
        )
    assert unauth.status_code == 401
    assert wrong.status_code == 401
    assert authed.status_code == 200
    assert authed.json() == {"status": "ok"}


def test_session_store_outage_does_not_clear_as_unauthorized(test_state) -> None:
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    pairing = app.state.remote_pairing

    def boom(_token: str):
        raise UnavailableError()

    pairing._lookup_active_session = boom  # type: ignore[method-assign]
    with TestClient(app) as client:
        response = client.get("/api/session", headers=AUTH)
    assert_http_error(response, status_code=503, code="STORE_UNAVAILABLE")


def test_exchange_store_outage_is_unavailable(test_state) -> None:
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    pairing = app.state.remote_pairing
    grant = pairing.current_grant()

    def boom(_token_hash: str):
        raise UnavailableError()

    pairing._store.get_paired_device_by_token_hash = boom  # type: ignore[method-assign]
    with TestClient(app) as client:
        response = client.post(
            "/api/pairing/exchange",
            headers=AUTH,
            json={"code": grant},
        )
    assert_http_error(response, status_code=503, code="STORE_UNAVAILABLE")
    assert pairing.current_grant() == grant


def test_media_store_outage_is_unavailable(test_state, tmp_path: Path) -> None:
    png_path = _png(tmp_path / "phone.png")
    asset = _store(test_state).ingest_upload(str(png_path))
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    pairing = app.state.remote_pairing
    with TestClient(app) as client:
        url = _bytes_url(client, asset.id)

        def boom(_device_id: str):
            raise UnavailableError()

        pairing._store.get_paired_device = boom  # type: ignore[method-assign]
        response = client.get(url)
    assert_http_error(response, status_code=503, code="STORE_UNAVAILABLE")


def test_query_token_does_not_authenticate(test_state, tmp_path: Path) -> None:
    png_path = _png(tmp_path / "phone.png")
    png_bytes = png_path.read_bytes()
    asset = _store(test_state).ingest_upload(str(png_path))
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        session = client.get("/api/session", params={"t": "pair-token"})
        generations = client.get(
            "/api/generations",
            params={"feature": "text-to-video", "t": "pair-token"},
        )
        upload = client.post(
            "/api/assets/upload",
            params={"t": "pair-token"},
            files={"file": ("phone.png", png_bytes, "image/png")},
        )
        query_get = client.get(
            f"/api/assets/{asset.id}/bytes", params={"t": "pair-token"}
        )
        query_head = client.head(
            f"/api/assets/{asset.id}/bytes", params={"t": "pair-token"}
        )

    assert session.status_code == 401
    assert generations.status_code == 401
    assert upload.status_code == 401
    assert query_get.status_code == 401
    assert query_head.status_code == 401


def test_pairing_exchange_issues_a_session(test_state) -> None:
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    grant = app.state.remote_pairing.current_grant()
    with TestClient(app) as client:
        exchanged = client.post("/api/pairing/exchange", json={"code": grant})
        assert exchanged.status_code == 200
        token = exchanged.json()["token"]
        replay = client.post("/api/pairing/exchange", json={"code": grant})
        session = client.get(
            "/api/session", headers={"Authorization": f"Bearer {token}"}
        )
    assert replay.status_code == 401
    assert session.status_code == 200


_CHROME_MAC_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def test_bearer_session_backfills_first_seen_ip_and_name(test_state) -> None:
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    pairing = app.state.remote_pairing
    assert pairing.list_public_devices()[0].ip is None
    with TestClient(app, client=("10.0.0.4", 50000)) as client:
        session = client.get(
            "/api/session",
            headers={
                "Authorization": "Bearer pair-token",
                "User-Agent": _CHROME_MAC_UA,
            },
        )
    assert session.status_code == 200
    listed = pairing.list_public_devices()[0]
    assert listed.ip == "10.0.0.4"
    assert listed.name == "Chrome on Mac"


def test_pairing_exchange_records_peer_ip_and_friendly_name(test_state) -> None:
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    grant = app.state.remote_pairing.current_grant()
    with TestClient(app, client=("192.168.1.4", 50000)) as client:
        exchanged = client.post(
            "/api/pairing/exchange",
            json={"code": grant},
            headers={"User-Agent": _CHROME_MAC_UA},
        )
    assert exchanged.status_code == 200
    listed = next(
        item
        for item in app.state.remote_pairing.list_public_devices()
        if item.id == exchanged.json()["device_id"]
    )
    assert listed.ip == "192.168.1.4"
    assert listed.name == "Chrome on Mac"


def test_pairing_exchange_reuses_device_when_bearer_present(test_state) -> None:
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    pairing = app.state.remote_pairing
    with TestClient(app) as client:
        first = client.post(
            "/api/pairing/exchange", json={"code": pairing.current_grant()}
        )
        assert first.status_code == 200
        token = first.json()["token"]
        device_id = first.json()["device_id"]
        second = client.post(
            "/api/pairing/exchange",
            json={"code": pairing.current_grant()},
            headers={"Authorization": f"Bearer {token}"},
        )
        reused = client.get("/api/session", headers={"Authorization": f"Bearer {token}"})
    assert second.status_code == 200
    assert second.json()["device_id"] == device_id
    assert second.json()["token"] == token
    assert reused.status_code == 200


def test_pairing_exchange_ignores_invalid_bearer(test_state) -> None:
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    pairing = app.state.remote_pairing
    with TestClient(app) as client:
        first = client.post(
            "/api/pairing/exchange", json={"code": pairing.current_grant()}
        )
        bogus = client.post(
            "/api/pairing/exchange",
            json={"code": pairing.current_grant()},
            headers={"Authorization": "Bearer not-a-session"},
        )
    assert first.status_code == 200
    assert bogus.status_code == 200
    assert bogus.json()["device_id"] != first.json()["device_id"]
    assert bogus.json()["token"] != first.json()["token"]


def test_trycloudflare_host_is_rejected(test_state) -> None:
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        response = client.get("/health", headers={"host": "abc.trycloudflare.com"})
    assert response.status_code == 400


def test_trusted_host_allows_configured_lan_and_rejects_unrelated(
    test_state, monkeypatch
) -> None:
    monkeypatch.setattr("remote.app.lan_ip", lambda: "192.168.50.12")
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        allowed = [
            client.get("/health", headers={"host": host})
            for host in ("testserver", "localhost", "127.0.0.1", "192.168.50.12")
        ]
        rejected = client.get("/health", headers={"host": "attacker.example"})
    assert [response.status_code for response in allowed] == [200, 200, 200, 200]
    assert [response.json() for response in allowed] == [{"status": "ok"}] * 4
    assert rejected.status_code == 400


def test_redact_asset_paths_only_strips_asset_shaped_objects() -> None:
    payload = {
        "id": "asset-1",
        "media_kind": "image",
        "path": "/Users/me/secret.png",
        "thumbnail_path": "/Users/me/secret-thumb.jpg",
        "outputs": [
            {
                "id": "out-1",
                "media_kind": "video",
                "path": "/tmp/out.mp4",
                "thumbnail_path": None,
            }
        ],
        "spec": {"params": {"prompt": "keep me"}},
    }
    redacted = redact_asset_paths(payload)
    assert isinstance(redacted, dict)
    assert redacted["path"] == ""
    assert redacted["thumbnail_path"] is None
    outputs = redacted["outputs"]
    assert isinstance(outputs, list)
    nested = outputs[0]
    assert isinstance(nested, dict)
    assert nested["path"] == ""
    spec = redacted["spec"]
    assert isinstance(spec, dict)
    assert spec == {"params": {"prompt": "keep me"}}
