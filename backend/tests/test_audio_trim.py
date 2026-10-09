from __future__ import annotations

import subprocess
import sys
import wave
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
import pytest
from PIL import Image
from starlette.testclient import TestClient

from pydantic import ValidationError

from api_types import TrimMediaRequest
from remote.app import create_remote_app
from services.audio_trim import (
    _write_audio_dest,
    extract_audio_file,
    trim_audio_file,
)
from services.ffmpeg import FFMPEG_TIMEOUT_SECONDS, run_ffmpeg
from services.media_probe import MAX_AUDIO_BYTES
from services.records import MediaError
from services.sqlite_store import SqliteStore
from tests.http_error_assertions import assert_http_error

AUTH = {"Authorization": "Bearer pair-token"}


def _png(path: Path) -> Path:
    Image.new("RGB", (16, 16), color=(1, 2, 3)).save(path)
    return path


def _wav(path: Path, *, duration_seconds: float = 2.0, sample_rate: int = 8000) -> Path:
    frame_count = max(1, int(duration_seconds * sample_rate))
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frame_count)
    return path


def _mp4(path: Path) -> Path:
    writer = imageio.get_writer(
        str(path), fps=8, codec="libx264", macro_block_size=None
    )
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    for _ in range(8):
        writer.append_data(frame)
    writer.close()
    return path


def _mp4_with_audio(path: Path) -> Path:
    video = _mp4(path.with_name(f"{path.stem}-silent.mp4"))
    audio = _wav(path.with_name(f"{path.stem}.wav"), duration_seconds=1.0)
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-i",
            str(video),
            "-i",
            str(audio),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _store(test_state) -> SqliteStore:
    return SqliteStore(test_state.config.app_data_dir)


def _ingest_audio(client: TestClient, tmp_path: Path) -> dict[str, object]:
    source = _wav(tmp_path / "input.wav")
    ingested = client.post("/api/assets", json={"path": str(source)})
    assert ingested.status_code == 200
    payload = ingested.json()
    assert isinstance(payload, dict)
    return payload


def test_trim_audio_creates_new_clip_and_leaves_original(
    client, tmp_path: Path
) -> None:
    original = _ingest_audio(client, tmp_path)
    original_id = original["id"]
    original_path = original["path"]
    original_duration = original["metadata"]["metadata"]["durationMs"]
    assert original_duration == 2000

    trimmed = client.post(
        f"/api/assets/{original_id}/trim-audio",
        json={"startSec": 0, "endSec": 1},
    )
    assert trimmed.status_code == 200
    data = trimmed.json()
    assert data["id"] != original_id
    assert data["media_kind"] == "audio"
    assert data["origin"] == "uploaded"
    duration_ms = data["metadata"]["metadata"]["durationMs"]
    assert 900 <= duration_ms <= 1100

    fetched = client.get(f"/api/assets/{original_id}")
    assert fetched.status_code == 200
    unchanged = fetched.json()
    assert unchanged["id"] == original_id
    assert unchanged["path"] == original_path
    assert unchanged["metadata"]["metadata"]["durationMs"] == original_duration
    assert Path(str(original_path)).is_file()


def test_trim_audio_rejects_end_not_after_start(client, tmp_path: Path) -> None:
    original = _ingest_audio(client, tmp_path)
    response = client.post(
        f"/api/assets/{original['id']}/trim-audio",
        json={"startSec": 1, "endSec": 1},
    )
    assert response.status_code == 422


def test_trim_audio_rejects_image_asset(client, tmp_path: Path) -> None:
    ingested = client.post(
        "/api/assets", json={"path": str(_png(tmp_path / "still.png"))}
    )
    assert ingested.status_code == 200
    response = client.post(
        f"/api/assets/{ingested.json()['id']}/trim-audio",
        json={"startSec": 0, "endSec": 1},
    )
    assert response.status_code == 422


def test_trim_audio_rejects_missing_asset(client) -> None:
    response = client.post(
        "/api/assets/missing-asset/trim-audio",
        json={"startSec": 0, "endSec": 1},
    )
    assert_http_error(response, status_code=404, code="ASSET_NOT_FOUND")


def test_trim_audio_missing_source_file_is_not_found(
    client, tmp_path: Path
) -> None:
    original = _ingest_audio(client, tmp_path)
    Path(str(original["path"])).unlink()
    response = client.post(
        f"/api/assets/{original['id']}/trim-audio",
        json={"startSec": 0, "endSec": 1},
    )
    assert_http_error(response, status_code=404, code="ASSET_NOT_FOUND")


def test_missing_ffmpeg_binary_is_unreadable_media(tmp_path: Path) -> None:
    missing = tmp_path / "ffmpeg-not-installed"
    with pytest.raises(MediaError) as caught:
        run_ffmpeg(["-version"], ffmpeg_exe=str(missing))
    assert caught.value.code == "UNREADABLE_MEDIA"


def test_hung_ffmpeg_is_unreadable_media_instead_of_blocking() -> None:
    """A wedged encoder must not pin the worker thread forever."""
    # A shebang script is not a Win32 PE, so CreateProcess raises WinError 193
    # before the timeout can fire. Python -c sleeps on every platform.
    stalling = [sys.executable, "-c", "import time; time.sleep(30)"]

    with pytest.raises(MediaError) as caught:
        run_ffmpeg(["-version"], ffmpeg_exe=stalling, timeout=0.2)

    assert caught.value.code == "UNREADABLE_MEDIA"
    assert "timed out" in str(caught.value)


def test_ffmpeg_timeout_default_is_generous_enough_for_long_audio() -> None:
    # 5s is the GPU-probe budget; trimming a long import legitimately exceeds it.
    assert FFMPEG_TIMEOUT_SECONDS >= 60.0


def test_extract_audio_from_video_creates_audio_asset(
    client, tmp_path: Path
) -> None:
    video = client.post(
        "/api/assets", json={"path": str(_mp4_with_audio(tmp_path / "clip.mp4"))}
    )
    assert video.status_code == 200
    video_id = video.json()["id"]
    video_path = video.json()["path"]

    extracted = client.post(f"/api/assets/{video_id}/extract-audio")
    assert extracted.status_code == 200
    data = extracted.json()
    assert data["id"] != video_id
    assert data["media_kind"] == "audio"
    assert data["metadata"]["mediaType"] == "audio"
    assert data["metadata"]["metadata"]["durationMs"] >= 1

    fetched = client.get(f"/api/assets/{video_id}")
    assert fetched.status_code == 200
    assert fetched.json()["path"] == video_path
    assert fetched.json()["media_kind"] == "video"


def test_extract_audio_rejects_video_without_audio_track(
    client, tmp_path: Path
) -> None:
    video = client.post(
        "/api/assets", json={"path": str(_mp4(tmp_path / "silent.mp4"))}
    )
    assert video.status_code == 200
    response = client.post(f"/api/assets/{video.json()['id']}/extract-audio")
    assert_http_error(
        response,
        status_code=400,
        code="NO_AUDIO_STREAM",
        message="video has no audio track",
    )
    assert "/" not in response.text
    assert "\\" not in response.text


def test_extract_audio_rejects_image_asset(client, tmp_path: Path) -> None:
    ingested = client.post(
        "/api/assets", json={"path": str(_png(tmp_path / "still.png"))}
    )
    assert ingested.status_code == 200
    response = client.post(f"/api/assets/{ingested.json()['id']}/extract-audio")
    assert response.status_code == 422


def test_extract_audio_rejects_missing_asset(client) -> None:
    response = client.post("/api/assets/missing-asset/extract-audio")
    assert_http_error(response, status_code=404, code="ASSET_NOT_FOUND")


def test_phone_trim_and_extract_return_remote_assets_without_paths(
    test_state, tmp_path: Path
) -> None:
    store = _store(test_state)
    audio = store.ingest_upload(str(_wav(tmp_path / "line.wav")))
    video = store.ingest_upload(str(_mp4_with_audio(tmp_path / "clip.mp4")))
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        trimmed = client.post(
            f"/api/assets/{audio.id}/trim-audio",
            headers=AUTH,
            json={"startSec": 0, "endSec": 1},
        )
        extracted = client.post(
            f"/api/assets/{video.id}/extract-audio",
            headers=AUTH,
        )
        missing = client.post(
            "/api/assets/missing-asset/trim-audio",
            headers=AUTH,
            json={"startSec": 0, "endSec": 1},
        )

    assert trimmed.status_code == 200
    trimmed_asset = trimmed.json()
    assert trimmed_asset["media_kind"] == "audio"
    assert "path" not in trimmed_asset
    assert "thumbnail_path" not in trimmed_asset
    assert 900 <= trimmed_asset["metadata"]["metadata"]["durationMs"] <= 1100

    assert extracted.status_code == 200
    extracted_asset = extracted.json()
    assert extracted_asset["media_kind"] == "audio"
    assert "path" not in extracted_asset
    assert "thumbnail_path" not in extracted_asset

    assert_http_error(missing, status_code=404, code="ASSET_NOT_FOUND")


@pytest.mark.parametrize("operation", ["trim-audio", "extract-audio"])
def test_phone_trim_extract_missing_source_hides_filesystem_path(
    test_state, tmp_path: Path, operation: str
) -> None:
    store = _store(test_state)
    if operation == "trim-audio":
        asset = store.ingest_upload(str(_wav(tmp_path / "line.wav")))
        request_json: dict[str, object] | None = {"startSec": 0, "endSec": 1}
    else:
        asset = store.ingest_upload(str(_mp4_with_audio(tmp_path / "clip.mp4")))
        request_json = None
    Path(asset.path).unlink()

    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        if request_json is None:
            response = client.post(
                f"/api/assets/{asset.id}/{operation}", headers=AUTH
            )
        else:
            response = client.post(
                f"/api/assets/{asset.id}/{operation}",
                headers=AUTH,
                json=request_json,
            )

    assert_http_error(response, status_code=404, code="ASSET_NOT_FOUND")
    assert "/" not in response.text
    assert "\\" not in response.text


def test_trim_audio_rejects_unknown_fields(client, tmp_path: Path) -> None:
    original = _ingest_audio(client, tmp_path)
    response = client.post(
        f"/api/assets/{original['id']}/trim-audio",
        json={"startSec": 0, "endSec": 1, "unexpected": 1},
    )
    assert response.status_code == 422


def test_trim_audio_request_rejects_non_finite_and_extra() -> None:
    for payload in (
        {"startSec": 0, "endSec": float("inf")},
        {"startSec": 0, "endSec": float("-inf")},
        {"startSec": float("inf"), "endSec": float("inf")},
        {"startSec": 0, "endSec": float("nan")},
        {"startSec": float("nan"), "endSec": 1},
        {"startSec": 0, "endSec": 1, "unexpected": 1},
    ):
        with pytest.raises(ValidationError):
            TrimMediaRequest.model_validate(payload)


@pytest.mark.parametrize(
    "raw_json",
    [
        '{"startSec": 0, "endSec": Infinity}',
        '{"startSec": 0, "endSec": NaN}',
        '{"startSec": NaN, "endSec": 1}',
    ],
)
def test_trim_audio_rejects_non_finite_at_boundary(
    client, tmp_path: Path, raw_json: str
) -> None:
    # httpx json= rejects inf/nan client-side; send raw JSON so the FastAPI
    # boundary itself must reject the non-finite payload with a clean 422.
    original = _ingest_audio(client, tmp_path)
    response = client.post(
        f"/api/assets/{original['id']}/trim-audio",
        content=raw_json,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


@pytest.mark.parametrize("operation", ["trim-audio", "extract-audio"])
def test_trim_extract_directory_source_is_not_found(
    client, tmp_path: Path, operation: str
) -> None:
    if operation == "trim-audio":
        original = _ingest_audio(client, tmp_path)
        asset_id = str(original["id"])
        asset_path = Path(str(original["path"]))
        request_json: dict[str, object] | None = {"startSec": 0, "endSec": 1}
    else:
        video = client.post(
            "/api/assets", json={"path": str(_mp4_with_audio(tmp_path / "clip.mp4"))}
        )
        assert video.status_code == 200
        asset_id = str(video.json()["id"])
        asset_path = Path(str(video.json()["path"]))
        request_json = None
    asset_path.unlink()
    asset_path.mkdir()
    response = (
        client.post(f"/api/assets/{asset_id}/{operation}")
        if request_json is None
        else client.post(f"/api/assets/{asset_id}/{operation}", json=request_json)
    )
    assert_http_error(response, status_code=404, code="ASSET_NOT_FOUND")
    assert "/" not in response.text
    assert "\\" not in response.text


def test_trim_ffmpeg_args_restrict_protocols_and_cap_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[list[str]] = []

    def _capture(
        args: list[str],
        *,
        ffmpeg_exe: str | list[str] | None = None,
        timeout: float = FFMPEG_TIMEOUT_SECONDS,
    ) -> None:
        del ffmpeg_exe, timeout
        captured.append(list(args))

    monkeypatch.setattr("services.audio_trim.run_ffmpeg", _capture)
    trim_audio_file(tmp_path / "in.wav", tmp_path / "out.m4a", 0, 1)
    extract_audio_file(tmp_path / "in.mp4", tmp_path / "out2.m4a")

    assert len(captured) == 2
    for args in captured:
        assert "-protocol_whitelist" in args
        whitelist = args[args.index("-protocol_whitelist") + 1]
        assert whitelist == "file,pipe"
        assert "-fs" in args
        assert args[args.index("-fs") + 1] == str(MAX_AUDIO_BYTES)
        # Input allowlist must precede the input it constrains; the size cap
        # is an output option and must precede the output path.
        assert args.index("-protocol_whitelist") < args.index("-i")
        assert args.index("-fs") < len(args) - 1


def test_trim_seek_args_are_output_side_for_accuracy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[list[str]] = []

    def _capture(
        args: list[str],
        *,
        ffmpeg_exe: str | list[str] | None = None,
        timeout: float = FFMPEG_TIMEOUT_SECONDS,
    ) -> None:
        del ffmpeg_exe, timeout
        captured.append(list(args))

    monkeypatch.setattr("services.audio_trim.run_ffmpeg", _capture)
    trim_audio_file(tmp_path / "in.wav", tmp_path / "out.m4a", 0.5, 1.5)

    assert len(captured) == 1
    args = captured[0]
    input_index = args.index("-i")
    # Input allowlist stays immediately before the input it constrains.
    assert args[input_index - 2 : input_index] == [
        "-protocol_whitelist",
        "file,pipe",
    ]
    # Output-side seek decodes up to the In point for precise AAC/m4a cuts.
    assert args.index("-ss") > input_index + 1
    assert args.index("-t") > input_index + 1
    assert args[args.index("-ss") + 1] == "0.500000"
    assert args[args.index("-t") + 1] == "1.000000"


def test_write_audio_dest_rejects_oversized_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "services.audio_trim.MAX_AUDIO_BYTES", 1024, raising=False
    )

    def _encode(dest: Path) -> None:
        dest.write_bytes(b"x" * 2048)

    with pytest.raises(MediaError) as caught:
        _write_audio_dest(tmp_path, "input.wav", "trim", _encode)
    assert caught.value.code == "FILE_TOO_LARGE"


def test_write_audio_dest_fallback_wav_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "services.audio_trim.MAX_AUDIO_BYTES", 1024, raising=False
    )
    attempted: list[str] = []

    def _encode(dest: Path) -> None:
        attempted.append(dest.suffix)
        if dest.suffix == ".m4a":
            raise subprocess.CalledProcessError(1, ["ffmpeg"])
        dest.write_bytes(b"x" * 2048)

    with pytest.raises(MediaError) as caught:
        _write_audio_dest(tmp_path, "input.wav", "trim", _encode)
    assert caught.value.code == "FILE_TOO_LARGE"
    assert attempted == [".m4a", ".wav"]


def test_write_audio_dest_fallback_succeeds_when_wav_small(tmp_path: Path) -> None:
    def _encode(dest: Path) -> None:
        if dest.suffix == ".m4a":
            raise subprocess.CalledProcessError(1, ["ffmpeg"])
        dest.write_bytes(b"x" * 16)

    result = _write_audio_dest(tmp_path, "input.wav", "trim", _encode)
    assert result.suffix == ".wav"
    assert result.is_file()


def test_trim_output_cap_surfaces_controlled_error(
    client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1s of silenced AAC is ~895 bytes; cap below that so the post-encode
    # size check (not ingest validation) must reject the clip.
    monkeypatch.setattr("services.audio_trim.MAX_AUDIO_BYTES", 512, raising=False)
    original = _ingest_audio(client, tmp_path)
    response = client.post(
        f"/api/assets/{original['id']}/trim-audio",
        json={"startSec": 0, "endSec": 1},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "FILE_TOO_LARGE"
    assert "/" not in response.text
    assert "\\" not in response.text
