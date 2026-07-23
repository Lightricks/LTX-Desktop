"""Unit tests for LTX API client service."""

from __future__ import annotations

import pytest

from services.ltx_api_client.ltx_api_client_impl import LTXAPIClientImpl
from services.ltx_api_client.ltx_api_client import LTXAPIClientError
from services.http_client.http_client import HttpTransportError
from tests.fakes.services import FakeHTTPClient, FakeResponse


def _async_client(http: FakeHTTPClient) -> LTXAPIClientImpl:
    # Tiny poll interval so the submit→poll→download loop runs instantly in tests.
    return LTXAPIClientImpl(
        http=http,
        ltx_api_base_url="https://api.ltx.video",
        poll_interval_s=0.0,
        async_max_wait_s=5.0,
    )


def _queue_upload(http: FakeHTTPClient) -> None:
    http.queue(
        "post",
        FakeResponse(
            status_code=200,
            json_payload={
                "upload_url": "https://upload.example.com/extend",
                "storage_uri": "storage://extend/123",
                "required_headers": {},
            },
        ),
    )
    http.queue("put", FakeResponse(status_code=200))


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
    with pytest.raises(LTXAPIClientError, match="401") as exc:
        client.generate_text_to_video(
            api_key="bad-key",
            prompt="A mountain",
            model="ltx-2-3-pro",
            resolution="1920x1080",
            duration=5.0,
            fps=24.0,
            generate_audio=False,
        )
    assert exc.value.status_code == 401
    assert exc.value.stage == "generation"


def test_generate_text_to_video_extracts_insufficient_funds_error() -> None:
    http = FakeHTTPClient()
    http.queue(
        "post",
        FakeResponse(
            status_code=402,
            text='{"type":"error","error":{"type":"insufficient_funds_error","message":"Insufficient funds. Required: 36 cents"}}',
            headers={"Content-Type": "application/json", "x-request-id": "req-123"},
            json_payload={
                "type": "error",
                "error": {
                    "type": "insufficient_funds_error",
                    "message": "Insufficient funds. Required: 36 cents",
                },
            },
        ),
    )

    client = LTXAPIClientImpl(http=http, ltx_api_base_url="https://api.ltx.video")
    with pytest.raises(LTXAPIClientError, match="insufficient_funds_error") as exc:
        client.generate_text_to_video(
            api_key="bad-key",
            prompt="A mountain",
            model="ltx-2-3-pro",
            resolution="1920x1080",
            duration=5.0,
            fps=24.0,
            generate_audio=False,
        )
    assert exc.value.status_code == 402
    assert exc.value.stage == "generation"
    assert exc.value.provider_error_type == "insufficient_funds_error"
    assert exc.value.provider_message == "Insufficient funds. Required: 36 cents"
    assert exc.value.request_id == "req-123"


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
    with pytest.raises(LTXAPIClientError, match="422") as exc:
        client.generate_audio_to_video(
            api_key="bad-key",
            prompt="Bad request",
            audio_uri="storage://audio/123",
            image_uri=None,
            model="ltx-2-3-fast",
            resolution="1920x1080",
        )
    assert exc.value.status_code == 422
    assert exc.value.stage == "generation"


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


def test_extend_async_submits_polls_and_downloads(tmp_path) -> None:
    http = FakeHTTPClient()
    input_path = _write_dummy_video(tmp_path)
    _queue_upload(http)
    # submit -> 202 with job id
    http.queue("post", FakeResponse(status_code=202, json_payload={"id": "job-1"}))
    # poll: processing, then completed with a result url
    http.queue("get", FakeResponse(status_code=200, json_payload={"status": "processing"}))
    http.queue(
        "get",
        FakeResponse(
            status_code=200,
            json_payload={"status": "completed", "result": {"video_url": "https://cdn.example.com/extended.mp4"}},
        ),
    )
    # download
    http.queue("get", FakeResponse(status_code=200, content=b"extended-bytes"))

    client = _async_client(http)
    result = client.extend(
        api_key="test-key",
        video_path=input_path,
        duration=12.0,
        prompt="continue the motion",
        mode="end",
    )

    assert result.video_bytes == b"extended-bytes"
    assert result.result_payload is None
    submit_call = next(c for c in http.calls if c.url == "https://api.ltx.video/v2/extend")
    assert submit_call.json_payload is not None
    assert submit_call.json_payload["video_uri"] == "storage://extend/123"
    assert submit_call.json_payload["duration"] == 12.0
    assert submit_call.json_payload["mode"] == "end"
    poll_calls = [c for c in http.calls if c.url == "https://api.ltx.video/v2/extend/job-1"]
    assert len(poll_calls) == 2
    assert http.calls[-1].url == "https://cdn.example.com/extended.mp4"


def test_extend_async_retries_transient_poll_blip(tmp_path) -> None:
    # A single transport blip on a poll GET must not discard the minutes-long job — the
    # poll is retried and the job still completes.
    http = FakeHTTPClient()
    input_path = _write_dummy_video(tmp_path)
    _queue_upload(http)
    http.queue("post", FakeResponse(status_code=202, json_payload={"id": "job-3"}))
    http.queue("get", HttpTransportError("connection reset"))  # transient blip, retried
    http.queue(
        "get",
        FakeResponse(
            status_code=200,
            json_payload={"status": "completed", "result": {"video_url": "https://cdn.example.com/extended.mp4"}},
        ),
    )
    http.queue("get", FakeResponse(status_code=200, content=b"extended-bytes"))

    client = _async_client(http)
    result = client.extend(api_key="k", video_path=input_path, duration=4.0, prompt="", mode="end")
    assert result.video_bytes == b"extended-bytes"


def test_extend_async_unknown_terminal_status_surfaces(tmp_path) -> None:
    # An unrecognized terminal status (not completed / not in-progress) is surfaced as a
    # failure instead of being polled until the timeout hides it behind a 504.
    http = FakeHTTPClient()
    input_path = _write_dummy_video(tmp_path)
    _queue_upload(http)
    http.queue("post", FakeResponse(status_code=202, json_payload={"id": "job-4"}))
    http.queue("get", FakeResponse(status_code=200, json_payload={"status": "rejected"}))

    client = _async_client(http)
    with pytest.raises(LTXAPIClientError, match="rejected") as exc:
        client.extend(api_key="k", video_path=input_path, duration=4.0, prompt="", mode="end")
    assert exc.value.status_code == 500


def test_extend_async_job_failed_raises(tmp_path) -> None:
    http = FakeHTTPClient()
    input_path = _write_dummy_video(tmp_path)
    _queue_upload(http)
    http.queue("post", FakeResponse(status_code=202, json_payload={"id": "job-2"}))
    http.queue(
        "get",
        FakeResponse(
            status_code=200,
            json_payload={"status": "failed", "error": {"message": "model exploded"}},
        ),
    )

    client = _async_client(http)
    with pytest.raises(LTXAPIClientError, match="model exploded"):
        client.extend(api_key="k", video_path=input_path, duration=4.0, prompt="", mode="end")


def test_extend_async_422_maps_to_safety_filter(tmp_path) -> None:
    http = FakeHTTPClient()
    input_path = _write_dummy_video(tmp_path)
    _queue_upload(http)
    http.queue("post", FakeResponse(status_code=422, text="filtered"))

    client = _async_client(http)
    with pytest.raises(LTXAPIClientError, match="Content rejected by safety filters") as exc:
        client.extend(api_key="k", video_path=input_path, duration=4.0, prompt="", mode="end")
    assert exc.value.status_code == 422


def test_extend_async_connection_reset_maps_to_504(tmp_path) -> None:
    http = FakeHTTPClient()
    input_path = _write_dummy_video(tmp_path)
    _queue_upload(http)
    # The submit POST fails at the transport layer (connection reset surfaced as timeout).
    http.queue("post", HttpTransportError("Connection reset by peer"))

    client = _async_client(http)
    with pytest.raises(LTXAPIClientError, match="please retry") as exc:
        client.extend(api_key="k", video_path=input_path, duration=12.0, prompt="", mode="end")
    assert exc.value.status_code == 504


def test_extend_async_completed_without_url_raises(tmp_path) -> None:
    http = FakeHTTPClient()
    input_path = _write_dummy_video(tmp_path)
    _queue_upload(http)
    http.queue("post", FakeResponse(status_code=202, json_payload={"id": "job-3"}))
    http.queue("get", FakeResponse(status_code=200, json_payload={"status": "completed", "result": {}}))

    client = _async_client(http)
    with pytest.raises(LTXAPIClientError, match="without a video_url"):
        client.extend(api_key="k", video_path=input_path, duration=4.0, prompt="", mode="end")


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
