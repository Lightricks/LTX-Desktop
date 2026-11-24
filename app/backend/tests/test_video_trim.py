from __future__ import annotations

import subprocess
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
import pytest
from starlette.testclient import TestClient

from remote.app import create_remote_app
from services import video_trim
from services.encoded_media import derived_file_name
from services.ffmpeg import FFMPEG_TIMEOUT_SECONDS
from services.media_probe import MAX_VIDEO_BYTES
from services.video_trim import trim_video_file
from tests.http_error_assertions import assert_http_error
from tests.test_audio_trim import AUTH, _png, _store, _wav


def _mp4(path: Path, *, seconds: int = 3, fps: int = 8) -> Path:
    writer = imageio.get_writer(
        str(path), fps=fps, codec="libx264", macro_block_size=None
    )
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    for _ in range(seconds * fps):
        writer.append_data(frame)
    writer.close()
    return path


def _mp4_with_audio(path: Path) -> Path:
    video = _mp4(path.with_name(f"{path.stem}-silent.mp4"))
    audio = _wav(path.with_name(f"{path.stem}.wav"), duration_seconds=3.0)
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


def _ingest(client: TestClient, source: Path) -> dict[str, object]:
    ingested = client.post("/api/assets", json={"path": str(source)})
    assert ingested.status_code == 200
    payload = ingested.json()
    assert isinstance(payload, dict)
    return payload


@pytest.mark.parametrize("with_audio", [True, False])
def test_trim_video_creates_new_clip_and_leaves_original(
    client, tmp_path: Path, with_audio: bool
) -> None:
    source = (
        _mp4_with_audio(tmp_path / "clip.mp4")
        if with_audio
        else _mp4(tmp_path / "clip.mp4")
    )
    original = _ingest(client, source)
    original_meta = original["metadata"]["metadata"]
    assert 2900 <= original_meta["durationMs"] <= 3100

    trimmed = client.post(
        f"/api/assets/{original['id']}/trim-video",
        json={"startSec": 1, "endSec": 2},
    )
    assert trimmed.status_code == 200
    data = trimmed.json()
    assert data["id"] != original["id"]
    assert data["media_kind"] == "video"
    meta = data["metadata"]["metadata"]
    assert 900 <= meta["durationMs"] <= 1100
    assert meta["audioStreamCount"] == (1 if with_audio else 0)

    fetched = client.get(f"/api/assets/{original['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["path"] == original["path"]
    assert Path(str(original["path"])).is_file()


def test_trim_video_rejects_end_not_after_start(client, tmp_path: Path) -> None:
    original = _ingest(client, _mp4(tmp_path / "clip.mp4"))
    response = client.post(
        f"/api/assets/{original['id']}/trim-video",
        json={"startSec": 1, "endSec": 1},
    )
    assert response.status_code == 422


@pytest.mark.parametrize("make_source", [_png, _wav])
def test_trim_video_rejects_non_video_asset(
    client, tmp_path: Path, make_source
) -> None:
    suffix = ".png" if make_source is _png else ".wav"
    original = _ingest(client, make_source(tmp_path / f"input{suffix}"))
    response = client.post(
        f"/api/assets/{original['id']}/trim-video",
        json={"startSec": 0, "endSec": 1},
    )
    assert response.status_code == 422


def test_trim_video_rejects_missing_asset(client) -> None:
    response = client.post(
        "/api/assets/missing-asset/trim-video",
        json={"startSec": 0, "endSec": 1},
    )
    assert_http_error(response, status_code=404, code="ASSET_NOT_FOUND")


def test_trim_video_output_cap_surfaces_controlled_error(
    client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("services.video_trim.MAX_VIDEO_BYTES", 64, raising=False)
    original = _ingest(client, _mp4(tmp_path / "clip.mp4"))
    response = client.post(
        f"/api/assets/{original['id']}/trim-video",
        json={"startSec": 0, "endSec": 1},
    )
    assert_http_error(
        response,
        status_code=400,
        code="FILE_TOO_LARGE",
        message="encoded video exceeds size limit",
    )


def _odd_size_mp4(path: Path) -> Path:
    # yuv444p lets the source itself have odd dimensions.
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=15x17:d=3:r=8",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv444p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def test_trim_video_evens_out_odd_dimensions(client, tmp_path: Path) -> None:
    original = _ingest(client, _odd_size_mp4(tmp_path / "odd.mp4"))
    response = client.post(
        f"/api/assets/{original['id']}/trim-video",
        json={"startSec": 0, "endSec": 1},
    )
    assert response.status_code == 200, response.text
    meta = response.json()["metadata"]["metadata"]
    assert (meta["width"], meta["height"]) == (14, 16)


def test_trim_video_accepts_a_tiny_start_and_names_the_clip(
    client, tmp_path: Path
) -> None:
    original = _ingest(client, _mp4(tmp_path / "talk.mp4"))
    response = client.post(
        f"/api/assets/{original['id']}/trim-video",
        json={"startSec": 0.00001, "endSec": 1},
    )
    assert response.status_code == 200, response.text
    assert response.json()["name"] == "talk-trim.mp4"


def test_trim_video_rejects_a_start_past_the_end(client, tmp_path: Path) -> None:
    original = _ingest(client, _mp4(tmp_path / "clip.mp4"))
    response = client.post(
        f"/api/assets/{original['id']}/trim-video",
        json={"startSec": 5, "endSec": 6},
    )
    assert_http_error(
        response,
        status_code=422,
        code="INVALID_TRIM_RANGE",
        message="startSec is past the end of the video",
    )


def test_trim_video_reports_busy_while_another_trim_encodes(
    client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("services.video_trim.VIDEO_TRIM_QUEUE_TIMEOUT_SECONDS", 0.01)
    original = _ingest(client, _mp4(tmp_path / "clip.mp4"))
    with video_trim._ENCODE_SLOT:
        busy = client.post(
            f"/api/assets/{original['id']}/trim-video",
            json={"startSec": 0, "endSec": 1},
        )
    assert_http_error(
        busy,
        status_code=503,
        code="TRIM_BUSY",
        message="another video trim is in progress",
    )


def test_trim_video_releases_the_slot_after_a_failed_encode(
    client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fail(*_args: object, **_kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, "ffmpeg")

    original = _ingest(client, _mp4(tmp_path / "clip.mp4"))
    monkeypatch.setattr("services.video_trim.run_ffmpeg", _fail)
    failed = client.post(
        f"/api/assets/{original['id']}/trim-video",
        json={"startSec": 0, "endSec": 1},
    )
    assert_http_error(
        failed,
        status_code=400,
        code="UNREADABLE_MEDIA",
        message="ffmpeg failed to trim video",
    )
    assert not video_trim._ENCODE_SLOT.locked()


@pytest.mark.parametrize(
    ("source_name", "expected"),
    [
        ("talk.mov", "talk-trim.mp4"),
        ("a/b:c?.mp4", "b_c_-trim.mp4"),
        ("...", "clip-trim.mp4"),
    ],
)
def test_derived_file_name_is_safe(source_name: str, expected: str) -> None:
    assert derived_file_name(source_name, "trim", ".mp4") == expected


def test_trim_video_ffmpeg_args_seek_accurately_and_cap_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[list[str]] = []
    timeouts: list[float] = []

    def _capture(
        args: list[str],
        *,
        ffmpeg_exe: str | list[str] | None = None,
        timeout: float = FFMPEG_TIMEOUT_SECONDS,
    ) -> None:
        del ffmpeg_exe
        captured.append(list(args))
        timeouts.append(timeout)

    monkeypatch.setattr("services.video_trim.run_ffmpeg", _capture)
    trim_video_file(tmp_path / "in.mp4", tmp_path / "out.mp4", 0.5, 1.5)

    args = captured[0]
    input_index = args.index("-i")
    assert args[input_index - 2 : input_index] == ["-protocol_whitelist", "file,pipe"]
    assert args.index("-ss") < input_index
    assert args[args.index("-ss") + 1] == "0.500000"
    assert args[args.index("-t") + 1] == "1.000000"
    assert "0:a:0?" in args
    assert args[args.index("-vf") + 1] == "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    assert args[args.index("-c:v") + 1] == "libx264"
    assert args[args.index("-fs") + 1] == str(MAX_VIDEO_BYTES)
    assert timeouts[0] > FFMPEG_TIMEOUT_SECONDS


def test_phone_trim_video_returns_remote_asset_without_paths(
    test_state, tmp_path: Path
) -> None:
    store = _store(test_state)
    video = store.ingest_upload(str(_mp4_with_audio(tmp_path / "clip.mp4")))
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        trimmed = client.post(
            f"/api/assets/{video.id}/trim-video",
            headers=AUTH,
            json={"startSec": 0, "endSec": 1},
        )
        missing = client.post(
            "/api/assets/missing-asset/trim-video",
            headers=AUTH,
            json={"startSec": 0, "endSec": 1},
        )

    assert trimmed.status_code == 200
    asset = trimmed.json()
    assert asset["media_kind"] == "video"
    assert "path" not in asset
    assert "thumbnail_path" not in asset
    assert 900 <= asset["metadata"]["metadata"]["durationMs"] <= 1100
    assert_http_error(missing, status_code=404, code="ASSET_NOT_FOUND")
