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


def _queue_upload(http: FakeHTTPClient, storage_uri: str = "storage://extend/123") -> None:
    http.queue(
        "post",
        FakeResponse(
            status_code=200,
            json_payload={
                "upload_url": "https://upload.example.com/extend",
                "storage_uri": storage_uri,
                "required_headers": {},
            },
        ),
    )
    http.queue("put", FakeResponse(status_code=200))


def _queue_completed_job(
    http: FakeHTTPClient, *, job_id: str = "job-1", video_url: str, video_bytes: bytes
) -> None:
    http.queue("post", FakeResponse(status_code=202, json_payload={"id": job_id}))
    http.queue(
        "get",
        FakeResponse(
            status_code=200,
            json_payload={"status": "completed", "result": {"video_url": video_url}},
        ),
    )
    http.queue("get", FakeResponse(status_code=200, content=video_bytes))


def test_generate_text_to_video_returns_downloaded_bytes() -> None:
    http = FakeHTTPClient()
    _queue_completed_job(http, video_url="https://cdn.example.com/t2v.mp4", video_bytes=b"video-bytes")

    client = _async_client(http)
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
    assert len(http.calls) == 3
    submit, poll, download = http.calls
    assert submit.url == "https://api.ltx.video/v2/text-to-video"
    assert submit.headers == {
        "Authorization": "Bearer test-key",
        "Content-Type": "application/json",
    }
    assert submit.json_payload is not None
    assert submit.json_payload["prompt"] == "A mountain"
    assert submit.json_payload["model"] == "ltx-2-3-pro"
    assert submit.json_payload["resolution"] == "1920x1080"
    assert submit.json_payload["duration"] == 5.0
    assert submit.json_payload["fps"] == 24.0
    assert submit.json_payload["generate_audio"] is False
    assert submit.json_payload["camera_motion"] == "dolly_in"
    assert submit.json_payload["enhance_prompt"] is True
    assert poll.url == "https://api.ltx.video/v2/text-to-video/job-1"
    assert download.url == "https://cdn.example.com/t2v.mp4"
    # Signed result URL: no Authorization header on the download.
    assert download.headers is None


def test_generate_text_to_video_omits_camera_motion_when_none() -> None:
    http = FakeHTTPClient()
    _queue_completed_job(http, video_url="https://cdn.example.com/t2v.mp4", video_bytes=b"video-bytes")

    client = _async_client(http)
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


def test_generate_text_to_video_forwards_enhance_prompt_false() -> None:
    http = FakeHTTPClient()
    _queue_completed_job(http, video_url="https://cdn.example.com/t2v.mp4", video_bytes=b"video-bytes")

    client = _async_client(http)
    client.generate_text_to_video(
        api_key="test-key",
        prompt="A mountain",
        model="ltx-2-3-pro",
        resolution="1920x1080",
        duration=5.0,
        fps=24.0,
        generate_audio=False,
        enhance_prompt=False,
    )

    call = http.calls[0]
    assert call.json_payload is not None
    assert call.json_payload["enhance_prompt"] is False


def test_generate_text_to_video_submit_401_passes_through() -> None:
    http = FakeHTTPClient()
    http.queue(
        "post",
        FakeResponse(
            status_code=401,
            text="unauthorized",
            headers={"Content-Type": "application/json", "x-request-id": "req-401"},
            json_payload={"error": "unauthorized"},
        ),
    )

    client = _async_client(http)
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
    assert exc.value.request_id == "req-401"


def test_generate_text_to_video_submit_402_attaches_insufficient_funds() -> None:
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

    client = _async_client(http)
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
    # Async submit errors carry no stage by design (handlers key on status + provider type).
    assert exc.value.provider_error_type == "insufficient_funds_error"
    assert exc.value.provider_message == "Insufficient funds. Required: 36 cents"
    assert exc.value.request_id == "req-123"


def test_generate_text_to_video_transport_failure_maps_to_504() -> None:
    http = FakeHTTPClient()
    http.queue("post", HttpTransportError("Connection reset by peer"))

    client = _async_client(http)
    with pytest.raises(LTXAPIClientError, match="please retry") as exc:
        client.generate_text_to_video(
            api_key="k",
            prompt="A mountain",
            model="ltx-2-3-pro",
            resolution="1920x1080",
            duration=5.0,
            fps=24.0,
            generate_audio=False,
        )
    assert exc.value.status_code == 504


def test_generate_image_to_video_with_image_uri_downloads_video() -> None:
    http = FakeHTTPClient()
    _queue_completed_job(http, video_url="https://cdn.example.com/output.mp4", video_bytes=b"downloaded-video")

    client = _async_client(http)
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
    assert len(http.calls) == 3
    assert http.calls[0].url == "https://api.ltx.video/v2/image-to-video"
    assert http.calls[0].json_payload is not None
    assert http.calls[0].json_payload["image_uri"] == "storage://image/123"
    assert http.calls[0].json_payload["camera_motion"] == "jib_up"
    assert "last_frame_uri" not in http.calls[0].json_payload
    assert http.calls[1].url == "https://api.ltx.video/v2/image-to-video/job-1"
    assert http.calls[2].url == "https://cdn.example.com/output.mp4"


def test_generate_image_to_video_with_last_frame_uri() -> None:
    http = FakeHTTPClient()
    _queue_completed_job(http, video_url="https://cdn.example.com/output.mp4", video_bytes=b"downloaded-video")

    client = _async_client(http)
    out = client.generate_image_to_video(
        api_key="test-key",
        prompt="Animate from first to last",
        image_uri="storage://image/123",
        last_frame_uri="storage://image/456",
        model="ltx-2-3-pro",
        resolution="1920x1080",
        duration=4.0,
        fps=24.0,
        generate_audio=True,
        camera_motion="jib_up",
    )

    assert out == b"downloaded-video"
    assert http.calls[0].json_payload is not None
    assert http.calls[0].json_payload["image_uri"] == "storage://image/123"
    assert http.calls[0].json_payload["last_frame_uri"] == "storage://image/456"


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
    _queue_completed_job(http, video_url="https://cdn.example.com/a2v.mp4", video_bytes=b"downloaded-a2v-video")

    client = _async_client(http)
    out = client.generate_audio_to_video(
        api_key="test-key",
        prompt="Sync to this song",
        audio_uri="storage://audio/123",
        image_uri=None,
        model="ltx-2-3-fast",
        resolution="1920x1080",
    )

    assert out == b"downloaded-a2v-video"
    assert len(http.calls) == 3
    assert http.calls[0].url == "https://api.ltx.video/v2/audio-to-video"
    assert http.calls[0].json_payload is not None
    assert http.calls[0].json_payload["audio_uri"] == "storage://audio/123"
    assert "image_uri" not in http.calls[0].json_payload
    assert http.calls[2].url == "https://cdn.example.com/a2v.mp4"


def test_generate_audio_to_video_with_image_uri_posts_both_inputs() -> None:
    http = FakeHTTPClient()
    _queue_completed_job(http, video_url="https://cdn.example.com/a2v.mp4", video_bytes=b"downloaded-a2v-video")

    client = _async_client(http)
    out = client.generate_audio_to_video(
        api_key="test-key",
        prompt="Animate from image and audio",
        audio_uri="storage://audio/123",
        image_uri="storage://image/456",
        model="ltx-2-3-pro",
        resolution="3840x2160",
    )

    assert out == b"downloaded-a2v-video"
    assert http.calls[0].url == "https://api.ltx.video/v2/audio-to-video"
    assert http.calls[0].json_payload is not None
    assert http.calls[0].json_payload["audio_uri"] == "storage://audio/123"
    assert http.calls[0].json_payload["image_uri"] == "storage://image/456"
    assert http.calls[0].json_payload["model"] == "ltx-2-3-pro"
    assert http.calls[0].json_payload["resolution"] == "3840x2160"
    assert "last_frame_uri" not in http.calls[0].json_payload


def test_generate_audio_to_video_with_last_frame_uri() -> None:
    http = FakeHTTPClient()
    _queue_completed_job(http, video_url="https://cdn.example.com/a2v.mp4", video_bytes=b"downloaded-a2v-video")

    client = _async_client(http)
    out = client.generate_audio_to_video(
        api_key="test-key",
        prompt="Animate from image and audio",
        audio_uri="storage://audio/123",
        image_uri="storage://image/456",
        last_frame_uri="storage://image/789",
        model="ltx-2-3-pro",
        resolution="3840x2160",
    )

    assert out == b"downloaded-a2v-video"
    assert http.calls[0].json_payload is not None
    assert http.calls[0].json_payload["last_frame_uri"] == "storage://image/789"


def test_generate_audio_to_video_submit_422_maps_to_safety_filter() -> None:
    http = FakeHTTPClient()
    http.queue("post", FakeResponse(status_code=422, text="unprocessable"))

    client = _async_client(http)
    with pytest.raises(LTXAPIClientError, match="Content rejected by safety filters") as exc:
        client.generate_audio_to_video(
            api_key="bad-key",
            prompt="Bad request",
            audio_uri="storage://audio/123",
            image_uri=None,
            model="ltx-2-3-fast",
            resolution="1920x1080",
        )
    assert exc.value.status_code == 422


def _write_dummy_video(tmp_path) -> str:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"fake-video")
    return str(input_path)


def test_retake_async_submits_polls_and_downloads(tmp_path) -> None:
    http = FakeHTTPClient()
    input_path = _write_dummy_video(tmp_path)
    _queue_upload(http, storage_uri="storage://retake/123")
    http.queue("post", FakeResponse(status_code=202, json_payload={"id": "job-rt"}))
    http.queue("get", FakeResponse(status_code=200, json_payload={"status": "processing"}))
    http.queue(
        "get",
        FakeResponse(
            status_code=200,
            json_payload={"status": "completed", "result": {"video_url": "https://cdn.example.com/retake.mp4"}},
        ),
    )
    http.queue("get", FakeResponse(status_code=200, content=b"retake-bytes"))

    client = _async_client(http)
    result = client.retake(
        api_key="test-key",
        video_path=input_path,
        start_time=1.0,
        duration=3.0,
        prompt="make it dramatic",
        mode="replace_audio_and_video",
        model="ltx-2-3-pro",
    )

    assert result.video_bytes == b"retake-bytes"
    assert result.result_payload is None
    submit_call = next(c for c in http.calls if c.url == "https://api.ltx.video/v2/retake")
    assert submit_call.json_payload is not None
    assert submit_call.json_payload["video_uri"] == "storage://retake/123"
    assert submit_call.json_payload["start_time"] == 1.0
    assert submit_call.json_payload["duration"] == 3.0
    assert submit_call.json_payload["mode"] == "replace_audio_and_video"
    assert submit_call.json_payload["model"] == "ltx-2-3-pro"
    assert submit_call.json_payload["prompt"] == "make it dramatic"
    poll_calls = [c for c in http.calls if c.url == "https://api.ltx.video/v2/retake/job-rt"]
    assert len(poll_calls) == 2
    assert http.calls[-1].url == "https://cdn.example.com/retake.mp4"


def test_retake_async_422_maps_to_safety_filter(tmp_path) -> None:
    http = FakeHTTPClient()
    input_path = _write_dummy_video(tmp_path)
    _queue_upload(http)
    http.queue("post", FakeResponse(status_code=422, text="filtered"))

    client = _async_client(http)
    with pytest.raises(LTXAPIClientError, match="Content rejected by safety filters") as exc:
        client.retake(
            api_key="test-key",
            video_path=input_path,
            start_time=1.0,
            duration=3.0,
            prompt="test",
            mode="replace_audio_and_video",
            model="ltx-2-3-pro",
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
        model="ltx-2-3-pro",
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
    result = client.extend(api_key="k", video_path=input_path, duration=4.0, prompt="", mode="end", model="ltx-2-3-pro")
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
        client.extend(api_key="k", video_path=input_path, duration=4.0, prompt="", mode="end", model="ltx-2-3-pro")
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
        client.extend(api_key="k", video_path=input_path, duration=4.0, prompt="", mode="end", model="ltx-2-3-pro")


def test_extend_async_422_maps_to_safety_filter(tmp_path) -> None:
    http = FakeHTTPClient()
    input_path = _write_dummy_video(tmp_path)
    _queue_upload(http)
    http.queue(
        "post",
        FakeResponse(
            status_code=422,
            text='{"type":"error","error":{"type":"content_filtered_error","message":"blocked"}}',
            headers={"x-request-id": "req-422"},
            json_payload={
                "type": "error",
                "error": {"type": "content_filtered_error", "message": "blocked"},
            },
        ),
    )

    client = _async_client(http)
    with pytest.raises(LTXAPIClientError, match="Content rejected by safety filters") as exc:
        client.extend(api_key="k", video_path=input_path, duration=4.0, prompt="", mode="end", model="ltx-2-3-pro")
    assert exc.value.status_code == 422
    assert exc.value.request_id == "req-422"
    assert exc.value.provider_error_type == "content_filtered_error"
    assert exc.value.provider_message == "blocked"


def test_async_submit_missing_job_id_attaches_request_id(tmp_path) -> None:
    http = FakeHTTPClient()
    input_path = _write_dummy_video(tmp_path)
    _queue_upload(http)
    http.queue(
        "post",
        FakeResponse(status_code=202, json_payload={}, headers={"x-request-id": "req-noid"}),
    )

    client = _async_client(http)
    with pytest.raises(LTXAPIClientError, match="no job id") as exc:
        client.extend(api_key="k", video_path=input_path, duration=4.0, prompt="", mode="end", model="ltx-2-3-pro")
    assert exc.value.request_id == "req-noid"


def test_async_submit_malformed_body_attaches_request_id(tmp_path) -> None:
    http = FakeHTTPClient()
    input_path = _write_dummy_video(tmp_path)
    _queue_upload(http)
    http.queue(
        "post",
        FakeResponse(status_code=202, json_payload=[], headers={"x-request-id": "req-malformed"}),
    )

    client = _async_client(http)
    with pytest.raises(LTXAPIClientError, match="Unexpected") as exc:
        client.extend(api_key="k", video_path=input_path, duration=4.0, prompt="", mode="end", model="ltx-2-3-pro")
    assert exc.value.request_id == "req-malformed"


def test_extend_async_connection_reset_maps_to_504(tmp_path) -> None:
    http = FakeHTTPClient()
    input_path = _write_dummy_video(tmp_path)
    _queue_upload(http)
    # The submit POST fails at the transport layer (connection reset surfaced as timeout).
    http.queue("post", HttpTransportError("Connection reset by peer"))

    client = _async_client(http)
    with pytest.raises(LTXAPIClientError, match="please retry") as exc:
        client.extend(api_key="k", video_path=input_path, duration=12.0, prompt="", mode="end", model="ltx-2-3-pro")
    assert exc.value.status_code == 504


def test_extend_async_completed_without_url_raises(tmp_path) -> None:
    http = FakeHTTPClient()
    input_path = _write_dummy_video(tmp_path)
    _queue_upload(http)
    http.queue("post", FakeResponse(status_code=202, json_payload={"id": "job-3"}))
    http.queue("get", FakeResponse(status_code=200, json_payload={"status": "completed", "result": {}}))

    client = _async_client(http)
    with pytest.raises(LTXAPIClientError, match="without a video_url"):
        client.extend(api_key="k", video_path=input_path, duration=4.0, prompt="", mode="end", model="ltx-2-3-pro")


def test_async_submit_402_attaches_provider_error_fields(tmp_path) -> None:
    http = FakeHTTPClient()
    input_path = _write_dummy_video(tmp_path)
    _queue_upload(http)
    http.queue(
        "post",
        FakeResponse(
            status_code=402,
            text='{"type":"error","error":{"type":"insufficient_funds_error","message":"Insufficient funds. Required: 36 cents"}}',
            headers={"Content-Type": "application/json", "x-request-id": "req-submit-1"},
            json_payload={
                "type": "error",
                "error": {
                    "type": "insufficient_funds_error",
                    "message": "Insufficient funds. Required: 36 cents",
                },
            },
        ),
    )

    client = _async_client(http)
    with pytest.raises(LTXAPIClientError, match="Insufficient funds") as exc:
        client.extend(api_key="k", video_path=input_path, duration=4.0, prompt="", mode="end", model="ltx-2-3-pro")
    assert exc.value.status_code == 402
    assert exc.value.provider_error_type == "insufficient_funds_error"
    assert exc.value.provider_message == "Insufficient funds. Required: 36 cents"
    assert exc.value.request_id == "req-submit-1"


def test_async_submit_401_passes_through_with_request_id(tmp_path) -> None:
    http = FakeHTTPClient()
    input_path = _write_dummy_video(tmp_path)
    _queue_upload(http)
    http.queue(
        "post",
        FakeResponse(
            status_code=401,
            text="unauthorized",
            headers={"x-request-id": "req-submit-2"},
            json_payload={"error": "unauthorized"},
        ),
    )

    client = _async_client(http)
    with pytest.raises(LTXAPIClientError, match="unauthorized") as exc:
        client.extend(api_key="bad-key", video_path=input_path, duration=4.0, prompt="", mode="end", model="ltx-2-3-pro")
    assert exc.value.status_code == 401
    assert exc.value.request_id == "req-submit-2"


def test_retake_upload_transport_failure_maps_to_504(tmp_path) -> None:
    http = FakeHTTPClient()
    input_path = _write_dummy_video(tmp_path)
    http.queue("post", HttpTransportError("connection reset"))

    client = _async_client(http)
    with pytest.raises(LTXAPIClientError, match="please retry") as exc:
        client.retake(
            api_key="k",
            video_path=input_path,
            start_time=1.0,
            duration=3.0,
            prompt="",
            mode="replace_audio_and_video",
            model="ltx-2-3-pro",
        )
    assert exc.value.status_code == 504


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
            model="ltx-2-3-pro",
        )
    assert exc.value.status_code == 401


def test_async_failed_job_insufficient_funds_maps_to_402(tmp_path) -> None:
    http = FakeHTTPClient()
    input_path = _write_dummy_video(tmp_path)
    _queue_upload(http)
    http.queue("post", FakeResponse(status_code=202, json_payload={"id": "job-funds"}))
    http.queue(
        "get",
        FakeResponse(
            status_code=200,
            json_payload={
                "status": "failed",
                "error": {
                    "type": "insufficient_funds_error",
                    "message": "Insufficient funds. Required: 36 cents",
                },
            },
        ),
    )

    client = _async_client(http)
    with pytest.raises(LTXAPIClientError, match="Insufficient funds") as exc:
        client.extend(api_key="k", video_path=input_path, duration=4.0, prompt="", mode="end", model="ltx-2-3-pro")
    assert exc.value.status_code == 402
    assert exc.value.provider_error_type == "insufficient_funds_error"
    assert exc.value.provider_message == "Insufficient funds. Required: 36 cents"


def test_async_failed_job_content_filtered_maps_to_422(tmp_path) -> None:
    http = FakeHTTPClient()
    input_path = _write_dummy_video(tmp_path)
    _queue_upload(http)
    http.queue("post", FakeResponse(status_code=202, json_payload={"id": "job-filter"}))
    http.queue(
        "get",
        FakeResponse(
            status_code=200,
            json_payload={
                "status": "failed",
                "error": {"type": "content_filtered_error", "message": "blocked by filter"},
            },
        ),
    )

    client = _async_client(http)
    with pytest.raises(LTXAPIClientError, match="Content rejected by safety filters") as exc:
        client.extend(api_key="k", video_path=input_path, duration=4.0, prompt="", mode="end", model="ltx-2-3-pro")
    assert exc.value.status_code == 422
    assert exc.value.provider_error_type == "content_filtered_error"


def test_async_failed_job_unknown_type_surfaces_with_type(tmp_path) -> None:
    http = FakeHTTPClient()
    input_path = _write_dummy_video(tmp_path)
    _queue_upload(http)
    http.queue("post", FakeResponse(status_code=202, json_payload={"id": "job-over"}))
    http.queue(
        "get",
        FakeResponse(
            status_code=200,
            json_payload={
                "status": "failed",
                "error": {"type": "overloaded_error", "message": "backend overloaded"},
            },
        ),
    )

    client = _async_client(http)
    with pytest.raises(LTXAPIClientError, match="backend overloaded") as exc:
        client.extend(api_key="k", video_path=input_path, duration=4.0, prompt="", mode="end", model="ltx-2-3-pro")
    assert exc.value.status_code == 500
    assert exc.value.provider_error_type == "overloaded_error"
    assert exc.value.provider_message == "backend overloaded"


def test_async_download_empty_body_raises(tmp_path) -> None:
    http = FakeHTTPClient()
    input_path = _write_dummy_video(tmp_path)
    _queue_upload(http)
    http.queue("post", FakeResponse(status_code=202, json_payload={"id": "job-empty"}))
    http.queue(
        "get",
        FakeResponse(
            status_code=200,
            json_payload={"status": "completed", "result": {"video_url": "https://cdn.example.com/empty.mp4"}},
        ),
    )
    http.queue("get", FakeResponse(status_code=200, content=b""))

    client = _async_client(http)
    with pytest.raises(LTXAPIClientError, match="empty") as exc:
        client.extend(api_key="k", video_path=input_path, duration=4.0, prompt="", mode="end", model="ltx-2-3-pro")
    assert exc.value.status_code == 500


def test_async_poll_402_attaches_provider_error_fields(tmp_path) -> None:
    http = FakeHTTPClient()
    input_path = _write_dummy_video(tmp_path)
    _queue_upload(http)
    http.queue("post", FakeResponse(status_code=202, json_payload={"id": "job-poll402"}))
    http.queue(
        "get",
        FakeResponse(
            status_code=402,
            text='{"type":"error","error":{"type":"insufficient_funds_error","message":"Insufficient funds"}}',
            headers={"x-request-id": "req-poll-402"},
            json_payload={"type": "error", "error": {"type": "insufficient_funds_error", "message": "Insufficient funds"}},
        ),
    )

    client = _async_client(http)
    with pytest.raises(LTXAPIClientError, match="status check failed") as exc:
        client.extend(api_key="k", video_path=input_path, duration=4.0, prompt="", mode="end", model="ltx-2-3-pro")
    assert exc.value.status_code == 402
    assert exc.value.provider_error_type == "insufficient_funds_error"
    assert exc.value.provider_message == "Insufficient funds"
    assert exc.value.request_id == "req-poll-402"


def test_async_poll_422_maps_to_safety_filter(tmp_path) -> None:
    http = FakeHTTPClient()
    input_path = _write_dummy_video(tmp_path)
    _queue_upload(http)
    http.queue("post", FakeResponse(status_code=202, json_payload={"id": "job-poll422"}))
    http.queue(
        "get",
        FakeResponse(
            status_code=422,
            text='{"type":"error","error":{"type":"content_filtered_error","message":"blocked"}}',
            headers={"x-request-id": "req-poll-422"},
            json_payload={"type": "error", "error": {"type": "content_filtered_error", "message": "blocked"}},
        ),
    )

    client = _async_client(http)
    with pytest.raises(LTXAPIClientError, match="Content rejected by safety filters") as exc:
        client.extend(api_key="k", video_path=input_path, duration=4.0, prompt="", mode="end", model="ltx-2-3-pro")
    assert exc.value.status_code == 422
    assert exc.value.provider_error_type == "content_filtered_error"
    assert exc.value.request_id == "req-poll-422"


def test_async_failed_job_payment_declined_maps_to_402(tmp_path) -> None:
    http = FakeHTTPClient()
    input_path = _write_dummy_video(tmp_path)
    _queue_upload(http)
    http.queue("post", FakeResponse(status_code=202, json_payload={"id": "job-declined"}))
    http.queue(
        "get",
        FakeResponse(
            status_code=200,
            json_payload={
                "status": "failed",
                "error": {"type": "payment_declined_error", "message": "Card declined"},
            },
        ),
    )

    client = _async_client(http)
    with pytest.raises(LTXAPIClientError, match="Card declined") as exc:
        client.extend(api_key="k", video_path=input_path, duration=4.0, prompt="", mode="end", model="ltx-2-3-pro")
    assert exc.value.status_code == 402
    assert exc.value.provider_error_type == "payment_declined_error"
    assert exc.value.provider_message == "Card declined"


def test_async_poll_error_attaches_request_id(tmp_path) -> None:
    http = FakeHTTPClient()
    input_path = _write_dummy_video(tmp_path)
    _queue_upload(http)
    http.queue("post", FakeResponse(status_code=202, json_payload={"id": "job-pollerr"}))
    http.queue("get", FakeResponse(status_code=500, text="upstream exploded", headers={"x-request-id": "req-poll-1"}))

    client = _async_client(http)
    with pytest.raises(LTXAPIClientError, match="status check failed") as exc:
        client.extend(api_key="k", video_path=input_path, duration=4.0, prompt="", mode="end", model="ltx-2-3-pro")
    assert exc.value.status_code == 500
    assert exc.value.request_id == "req-poll-1"


def test_async_default_max_wait_is_1200s() -> None:
    http = FakeHTTPClient()
    client = LTXAPIClientImpl(http=http, ltx_api_base_url="https://api.ltx.video")
    assert client._async_max_wait_s == 1200.0
