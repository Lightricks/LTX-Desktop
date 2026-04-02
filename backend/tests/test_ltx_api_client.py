"""Unit tests for LTX API client service."""

from __future__ import annotations

import pytest

from services.ltx_api_client.ltx_api_client_impl import LTXAPIClientImpl
from services.ltx_api_client.ltx_api_client import LTXAPIClientError
from tests.fakes.services import FakeHTTPClient, FakeResponse


def test_generate_text_to_video_returns_binary_content() -> None:
    http = FakeHTTPClient()
    http.queue(
        "post",
        FakeResponse(
            status_code=200,
            headers={"Content-Type": "video/mp4"},
            content=b"video-bytes",
        ),
    )

    client = LTXAPIClientImpl(http=http, ltx_api_base_url="https://api.ltx.video")
    out = client.generate_text_to_video(
        api_key="test-key",
        prompt="A mountain",
        model="ltx-2-3-pro",
        resolution="1920x1080",
        duration=5.0,
        fps=24.0,
        generate_audio=False,
        camera_motion="dolly_in",
    )

    assert out == b"video-bytes"
    assert len(http.calls) == 1
    call = http.calls[0]
    assert call.url == "https://api.ltx.video/v1/text-to-video"
    assert call.headers == {
        "Authorization": "Bearer test-key",
        "Content-Type": "application/json",
    }
    assert call.json_payload is not None
    assert call.json_payload["prompt"] == "A mountain"
    assert call.json_payload["resolution"] == "1920x1080"
    assert call.json_payload["camera_motion"] == "dolly_in"


def test_generate_image_to_video_with_image_uri_downloads_video() -> None:
    http = FakeHTTPClient()
    http.queue(
        "post",
        FakeResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            json_payload={"video_url": "https://cdn.example.com/output.mp4"},
        ),
    )
    http.queue(
        "get",
        FakeResponse(status_code=200, content=b"downloaded-video"),
    )

    client = LTXAPIClientImpl(http=http, ltx_api_base_url="https://api.ltx.video")
    out = client.generate_image_to_video(
        api_key="test-key",
        prompt="Animate this frame",
        image_uri="storage://image/123",
        model="ltx-2-3-pro",
        resolution="1920x1080",
        duration=4.0,
        fps=24.0,
        generate_audio=True,
        camera_motion="jib_up",
    )

    assert out == b"downloaded-video"
    assert len(http.calls) == 2
    assert http.calls[0].url == "https://api.ltx.video/v1/image-to-video"
    assert http.calls[0].json_payload is not None
    assert http.calls[0].json_payload["image_uri"] == "storage://image/123"
    assert http.calls[0].json_payload["camera_motion"] == "jib_up"
    assert http.calls[1].url == "https://cdn.example.com/output.mp4"


def test_generate_text_to_video_omits_camera_motion_when_none() -> None:
    http = FakeHTTPClient()
    http.queue(
        "post",
        FakeResponse(
            status_code=200,
            headers={"Content-Type": "video/mp4"},
            content=b"video-bytes",
        ),
    )

    client = LTXAPIClientImpl(http=http, ltx_api_base_url="https://api.ltx.video")
    out = client.generate_text_to_video(
        api_key="test-key",
        prompt="A mountain",
        model="ltx-2-3-pro",
        resolution="1920x1080",
        duration=5.0,
        fps=24.0,
        generate_audio=False,
        camera_motion="none",
    )

    assert out == b"video-bytes"
    call = http.calls[0]
    assert call.json_payload is not None
    assert "camera_motion" not in call.json_payload


def test_generate_text_to_video_raises_on_non_200() -> None:
    http = FakeHTTPClient()
    http.queue(
        "post",
        FakeResponse(
            status_code=401,
            text="unauthorized",
            headers={"Content-Type": "application/json"},
            json_payload={"error": "unauthorized"},
        ),
    )

    client = LTXAPIClientImpl(http=http, ltx_api_base_url="https://api.ltx.video")
    with pytest.raises(RuntimeError, match="401"):
        client.generate_text_to_video(
            api_key="bad-key",
            prompt="A mountain",
            model="ltx-2-3-pro",
            resolution="1920x1080",
            duration=5.0,
            fps=24.0,
            generate_audio=False,
        )


def test_upload_file_returns_storage_uri(tmp_path) -> None:
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"fake-audio")

    http = FakeHTTPClient()
    http.queue(
        "post",
        FakeResponse(
            status_code=200,
            json_payload={
                "upload_url": "https://upload.example.com/audio",
                "storage_uri": "storage://audio/123",
                "required_headers": {"x-ms-blob-type": "BlockBlob"},
            },
        ),
    )
    http.queue("put", FakeResponse(status_code=200))

    client = LTXAPIClientImpl(http=http, ltx_api_base_url="https://api.ltx.video")
    out = client.upload_file(
        api_key="test-key",
        file_path=str(audio_path),
    )

    assert out == "storage://audio/123"
    assert len(http.calls) == 2
    assert http.calls[0].url == "https://api.ltx.video/v1/upload"
    assert http.calls[1].method == "put"


def test_generate_audio_to_video_with_audio_uri_downloads_video() -> None:
    http = FakeHTTPClient()
    http.queue(
        "post",
        FakeResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            json_payload={"video_url": "https://cdn.example.com/a2v.mp4"},
        ),
    )
    http.queue("get", FakeResponse(status_code=200, content=b"downloaded-a2v-video"))

    client = LTXAPIClientImpl(http=http, ltx_api_base_url="https://api.ltx.video")
    out = client.generate_audio_to_video(
        api_key="test-key",
        prompt="Sync to this song",
        audio_uri="storage://audio/123",
        image_uri=None,
        model="ltx-2-3-fast",
        resolution="1920x1080",
    )

    assert out == b"downloaded-a2v-video"
    assert len(http.calls) == 2
    assert http.calls[0].url == "https://api.ltx.video/v1/audio-to-video"
    assert http.calls[0].json_payload is not None
    assert http.calls[0].json_payload["audio_uri"] == "storage://audio/123"
    assert "image_uri" not in http.calls[0].json_payload
    assert http.calls[1].url == "https://cdn.example.com/a2v.mp4"


def test_generate_audio_to_video_with_image_uri_posts_both_inputs() -> None:
    http = FakeHTTPClient()
    http.queue(
        "post",
        FakeResponse(
            status_code=200,
            headers={"Content-Type": "video/mp4"},
            content=b"direct-a2v-video",
        ),
    )

    client = LTXAPIClientImpl(http=http, ltx_api_base_url="https://api.ltx.video")
    out = client.generate_audio_to_video(
        api_key="test-key",
        prompt="Animate from image and audio",
        audio_uri="storage://audio/123",
        image_uri="storage://image/456",
        model="ltx-2-3-pro",
        resolution="3840x2160",
    )

    assert out == b"direct-a2v-video"
    assert len(http.calls) == 1
    assert http.calls[0].url == "https://api.ltx.video/v1/audio-to-video"
    assert http.calls[0].json_payload is not None
    assert http.calls[0].json_payload["audio_uri"] == "storage://audio/123"
    assert http.calls[0].json_payload["image_uri"] == "storage://image/456"
    assert http.calls[0].json_payload["model"] == "ltx-2-3-pro"
    assert http.calls[0].json_payload["resolution"] == "3840x2160"


def test_generate_audio_to_video_raises_on_non_200() -> None:
    http = FakeHTTPClient()
    http.queue("post", FakeResponse(status_code=422, text="unprocessable"))

    client = LTXAPIClientImpl(http=http, ltx_api_base_url="https://api.ltx.video")
    with pytest.raises(RuntimeError, match="422"):
        client.generate_audio_to_video(
            api_key="bad-key",
            prompt="Bad request",
            audio_uri="storage://audio/123",
            image_uri=None,
            model="ltx-2-3-fast",
            resolution="1920x1080",
        )


def _write_dummy_video(tmp_path) -> str:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"fake-video")
    return str(input_path)


def test_retake_returns_direct_video_bytes(tmp_path) -> None:
    http = FakeHTTPClient()
    input_path = _write_dummy_video(tmp_path)
    http.queue(
        "post",
        FakeResponse(
            status_code=200,
            json_payload={
                "upload_url": "https://upload.example.com/retake",
                "storage_uri": "storage://retake/123",
                "required_headers": {},
            },
        ),
        FakeResponse(
            status_code=200,
            headers={"Content-Type": "video/mp4"},
            content=b"retake-bytes",
        ),
    )
    http.queue("put", FakeResponse(status_code=200))

    client = LTXAPIClientImpl(http=http, ltx_api_base_url="https://api.ltx.video")
    result = client.retake(
        api_key="test-key",
        video_path=input_path,
        start_time=1.0,
        duration=3.0,
        prompt="make it dramatic",
        mode="replace_audio_and_video",
    )

    assert result.video_bytes == b"retake-bytes"
    assert result.result_payload is None
    assert len(http.calls) == 3
    assert http.calls[0].url == "https://api.ltx.video/v1/upload"
    assert http.calls[1].method == "put"
    assert http.calls[2].url == "https://api.ltx.video/v1/retake"
    assert http.calls[2].json_payload is not None
    assert http.calls[2].json_payload["video_uri"] == "storage://retake/123"


def test_retake_json_video_url_downloads_bytes(tmp_path) -> None:
    http = FakeHTTPClient()
    input_path = _write_dummy_video(tmp_path)
    http.queue(
        "post",
        FakeResponse(
            status_code=200,
            json_payload={
                "upload_url": "https://upload.example.com/retake",
                "storage_uri": "storage://retake/456",
                "required_headers": {},
            },
        ),
        FakeResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            json_payload={"video_url": "https://cdn.example.com/retake.mp4"},
        ),
    )
    http.queue("put", FakeResponse(status_code=200))
    http.queue("get", FakeResponse(status_code=200, content=b"downloaded-retake"))

    client = LTXAPIClientImpl(http=http, ltx_api_base_url="https://api.ltx.video")
    result = client.retake(
        api_key="test-key",
        video_path=input_path,
        start_time=2.0,
        duration=4.0,
        prompt="test",
        mode="replace_video",
    )

    assert result.video_bytes == b"downloaded-retake"
    assert result.result_payload is None
    assert http.calls[-1].url == "https://cdn.example.com/retake.mp4"


def test_retake_json_without_video_url_returns_payload(tmp_path) -> None:
    http = FakeHTTPClient()
    input_path = _write_dummy_video(tmp_path)
    http.queue(
        "post",
        FakeResponse(
            status_code=200,
            json_payload={
                "upload_url": "https://upload.example.com/retake",
                "storage_uri": "storage://retake/789",
                "required_headers": {},
            },
        ),
        FakeResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            json_payload={"status": "processing"},
        ),
    )
    http.queue("put", FakeResponse(status_code=200))

    client = LTXAPIClientImpl(http=http, ltx_api_base_url="https://api.ltx.video")
    result = client.retake(
        api_key="test-key",
        video_path=input_path,
        start_time=0.0,
        duration=2.5,
        prompt="test",
        mode="replace_audio_and_video",
    )

    assert result.video_bytes is None
    assert result.result_payload is not None
    assert result.result_payload["status"] == "processing"


def test_retake_422_maps_to_safety_filter_error(tmp_path) -> None:
    http = FakeHTTPClient()
    input_path = _write_dummy_video(tmp_path)
    http.queue(
        "post",
        FakeResponse(
            status_code=200,
            json_payload={
                "upload_url": "https://upload.example.com/retake",
                "storage_uri": "storage://retake/000",
                "required_headers": {},
            },
        ),
        FakeResponse(status_code=422, text="Content filtered"),
    )
    http.queue("put", FakeResponse(status_code=200))

    client = LTXAPIClientImpl(http=http, ltx_api_base_url="https://api.ltx.video")
    with pytest.raises(LTXAPIClientError, match="Content rejected by safety filters") as exc:
        client.retake(
            api_key="test-key",
            video_path=input_path,
            start_time=1.0,
            duration=3.0,
            prompt="test",
            mode="replace_audio_and_video",
        )
    assert exc.value.status_code == 422


def test_retake_upload_init_failure_maps_message() -> None:
    http = FakeHTTPClient()
    http.queue("post", FakeResponse(status_code=401, text="Unauthorized"))

    client = LTXAPIClientImpl(http=http, ltx_api_base_url="https://api.ltx.video")
    with pytest.raises(LTXAPIClientError, match="Failed to get upload URL: Unauthorized") as exc:
        client.retake(
            api_key="test-key",
            video_path="/tmp/input.mp4",
            start_time=1.0,
            duration=3.0,
            prompt="test",
            mode="replace_audio_and_video",
        )
    assert exc.value.status_code == 401
