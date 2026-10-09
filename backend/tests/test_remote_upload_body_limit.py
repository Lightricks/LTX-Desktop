"""Remote upload body limit: bounded ASGI bytes before multipart parsing."""

from __future__ import annotations

import asyncio
import json
import tempfile
import wave
from pathlib import Path
from typing import Any

import _routes.assets as assets_module
from _routes.assets import _MULTIPART_OVERHEAD_BYTES, _UPLOAD_TEMP_PREFIX
from remote.app import create_remote_app
from services.media_probe import MAX_AUDIO_BYTES, MAX_IMAGE_BYTES, MAX_VIDEO_BYTES
from starlette.testclient import TestClient
from tests.http_error_assertions import assert_http_error

AUTH = {"Authorization": "Bearer pair-token"}


def _wav_bytes(path: Path, *, duration_seconds: float = 1.0) -> bytes:
    frame_count = max(1, int(duration_seconds * 8000))
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(b"\x00\x00" * frame_count)
    return path.read_bytes()


def _png_bytes(path: Path) -> bytes:
    from PIL import Image

    Image.new("RGB", (32, 24), color=(10, 20, 30)).save(path)
    return path.read_bytes()


def _persistent_asset_files(test_state: Any) -> list[Path]:
    assets_dir = test_state.config.app_data_dir / "assets"
    if not assets_dir.exists():
        return []
    return sorted(p for p in assets_dir.rglob("*") if p.is_file())


def _upload_temp_files() -> list[Path]:
    return sorted(Path(tempfile.gettempdir()).glob(f"{_UPLOAD_TEMP_PREFIX}*"))


def _assert_no_leaked_upload_files(test_state: Any) -> None:
    assert _persistent_asset_files(test_state) == []
    assert _upload_temp_files() == []


def _multipart_body(payload: bytes, filename: str, mime: str, boundary: str) -> bytes:
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    return head + payload + tail


def _call_remote_upload_asgi(
    app: Any,
    *,
    body: bytes,
    boundary: str,
    content_length: bytes | None,
    chunk_size: int | None = None,
    with_auth: bool = True,
) -> tuple[int, bytes]:
    headers: list[tuple[bytes, bytes]] = [
        (b"host", b"testserver"),
        (
            b"content-type",
            f"multipart/form-data; boundary={boundary}".encode(),
        ),
    ]
    if with_auth:
        headers.append((b"authorization", b"Bearer pair-token"))
    if content_length is not None:
        headers.append((b"content-length", content_length))
    if chunk_size is None:
        chunks = [body]
    else:
        chunks = [body[i : i + chunk_size] for i in range(0, len(body), chunk_size)] or [b""]

    async def _run() -> tuple[int, bytes]:
        pending = list(chunks)

        async def receive() -> dict[str, Any]:
            if pending:
                chunk = pending.pop(0)
                return {
                    "type": "http.request",
                    "body": chunk,
                    "more_body": bool(pending),
                }
            return {"type": "http.request", "body": b"", "more_body": False}

        messages: list[dict[str, Any]] = []

        async def send(message: dict[str, Any]) -> None:
            messages.append(message)

        scope: dict[str, Any] = {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/assets/upload",
            "raw_path": b"/api/assets/upload",
            "query_string": b"",
            "root_path": "",
            "headers": headers,
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }
        await app(scope, receive, send)
        start = next(m for m in messages if m["type"] == "http.response.start")
        payload = b"".join(
            m.get("body", b"") for m in messages if m["type"] == "http.response.body"
        )
        status = start["status"]
        assert isinstance(status, int)
        return status, payload

    return asyncio.run(_run())


def test_upload_body_limit_derived_from_existing_constants() -> None:
    assert (
        assets_module.MAX_REMOTE_UPLOAD_BODY_BYTES
        == max(MAX_IMAGE_BYTES, MAX_AUDIO_BYTES, MAX_VIDEO_BYTES)
        + _MULTIPART_OVERHEAD_BYTES
    )


def test_declared_oversize_rejected_before_multipart_parsing(
    test_state: Any, tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        assets_module, "MAX_REMOTE_UPLOAD_BODY_BYTES", 4096, raising=False
    )
    png = _png_bytes(tmp_path / "ok.png")
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        response = client.post(
            "/api/assets/upload",
            headers={**AUTH, "content-length": "5000"},
            files={"file": ("ok.png", png, "image/png")},
        )
    assert_http_error(response, status_code=400, code="FILE_TOO_LARGE")
    _assert_no_leaked_upload_files(test_state)


def test_actual_bytes_rejected_despite_lying_content_length(
    test_state: Any, tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        assets_module, "MAX_REMOTE_UPLOAD_BODY_BYTES", 4096, raising=False
    )
    audio = _wav_bytes(tmp_path / "input.wav", duration_seconds=1.0)
    assert len(audio) > 4096
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        response = client.post(
            "/api/assets/upload",
            headers={**AUTH, "content-length": "10"},
            files={"file": ("input.wav", audio, "audio/wav")},
        )
    assert_http_error(response, status_code=400, code="FILE_TOO_LARGE")
    _assert_no_leaked_upload_files(test_state)


def test_actual_bytes_without_content_length_rejected(
    test_state: Any, tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        assets_module, "MAX_REMOTE_UPLOAD_BODY_BYTES", 2048, raising=False
    )
    audio = _wav_bytes(tmp_path / "input.wav", duration_seconds=1.0)
    boundary = "----testboundarywithoutlength"
    body = _multipart_body(audio, "input.wav", "audio/wav", boundary)
    assert len(body) > 2048
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    status, payload = _call_remote_upload_asgi(
        app, body=body, boundary=boundary, content_length=None, chunk_size=1024
    )
    assert status == 400
    assert json.loads(payload.decode()) == {
        "code": "FILE_TOO_LARGE",
        "message": "FILE_TOO_LARGE",
    }
    _assert_no_leaked_upload_files(test_state)


def test_single_oversize_event_not_passed_through(
    test_state: Any, tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        assets_module, "MAX_REMOTE_UPLOAD_BODY_BYTES", 2048, raising=False
    )
    audio = _wav_bytes(tmp_path / "input.wav", duration_seconds=1.0)
    boundary = "----testboundarysingleevent"
    body = _multipart_body(audio, "input.wav", "audio/wav", boundary)
    assert len(body) > 2048
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    status, payload = _call_remote_upload_asgi(
        app,
        body=body,
        boundary=boundary,
        content_length=b"10",
        chunk_size=None,
    )
    assert status == 400
    assert json.loads(payload.decode()) == {
        "code": "FILE_TOO_LARGE",
        "message": "FILE_TOO_LARGE",
    }
    _assert_no_leaked_upload_files(test_state)


def test_upload_at_limit_succeeds_and_one_byte_over_rejected(
    test_state: Any, tmp_path: Path, monkeypatch: Any
) -> None:
    png = _png_bytes(tmp_path / "ok.png")
    boundary = "----testboundaryexactlimit"
    body = _multipart_body(png, "ok.png", "image/png", boundary)
    app = create_remote_app(handler=test_state, remote_token="pair-token")

    monkeypatch.setattr(
        assets_module, "MAX_REMOTE_UPLOAD_BODY_BYTES", len(body), raising=False
    )
    status_ok, payload_ok = _call_remote_upload_asgi(
        app, body=body, boundary=boundary, content_length=str(len(body)).encode()
    )
    assert status_ok == 200, payload_ok.decode()[:500]
    asset = json.loads(payload_ok.decode())
    assert asset["media_kind"] == "image"
    assert "path" not in asset

    monkeypatch.setattr(
        assets_module, "MAX_REMOTE_UPLOAD_BODY_BYTES", len(body) - 1, raising=False
    )
    status_over, payload_over = _call_remote_upload_asgi(
        app,
        body=body,
        boundary=boundary,
        content_length=str(len(body)).encode(),
        chunk_size=512,
    )
    assert status_over == 400
    assert json.loads(payload_over.decode()) == {
        "code": "FILE_TOO_LARGE",
        "message": "FILE_TOO_LARGE",
    }


def test_body_limit_does_not_apply_to_unrelated_remote_routes(
    test_state: Any, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        assets_module, "MAX_REMOTE_UPLOAD_BODY_BYTES", 16, raising=False
    )
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        response = client.post("/api/generations/text-to-video", headers=AUTH)
    assert response.status_code == 422


def test_oversize_upload_still_requires_auth(
    test_state: Any, tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        assets_module, "MAX_REMOTE_UPLOAD_BODY_BYTES", 16, raising=False
    )
    png = _png_bytes(tmp_path / "ok.png")
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        response = client.post(
            "/api/assets/upload",
            files={"file": ("ok.png", png, "image/png")},
        )
    assert response.status_code == 401
