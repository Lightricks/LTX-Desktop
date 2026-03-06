"""Tests for the ReplicateVideoClientImpl video API client."""

from __future__ import annotations

import pytest

from services.video_api_client.replicate_video_client_impl import ReplicateVideoClientImpl
from tests.fakes.services import FakeHTTPClient, FakeResponse


API_KEY = "test-replicate-key"
SEEDANCE_MODEL = "seedance-1.5-pro"
BASE_URL = "https://api.replicate.com/v1"


def _make_client(http: FakeHTTPClient) -> ReplicateVideoClientImpl:
    return ReplicateVideoClientImpl(http=http, api_base_url=BASE_URL)


def _default_kwargs() -> dict[str, object]:
    return {
        "api_key": API_KEY,
        "model": SEEDANCE_MODEL,
        "prompt": "A cat walking on a beach",
        "duration": 5,
        "resolution": "720p",
        "aspect_ratio": "16:9",
        "generate_audio": False,
    }


def test_seedance_text_to_video_sync_success() -> None:
    http = FakeHTTPClient()

    # POST returns succeeded immediately
    http.queue(
        "post",
        FakeResponse(
            status_code=201,
            json_payload={
                "id": "pred123",
                "status": "succeeded",
                "output": "https://replicate.delivery/video.mp4",
            },
        ),
    )

    # GET downloads video bytes
    video_bytes = b"fake-mp4-video-content"
    http.queue(
        "get",
        FakeResponse(status_code=200, content=video_bytes),
    )

    client = _make_client(http)
    result = client.generate_text_to_video(**_default_kwargs())

    assert result == video_bytes
    assert len(http.calls) == 2
    assert http.calls[0].method == "post"
    assert "bytedance/seedance-1.5-pro" in http.calls[0].url
    assert http.calls[1].method == "get"


def test_seedance_text_to_video_polling_success() -> None:
    http = FakeHTTPClient()

    # POST returns processing
    http.queue(
        "post",
        FakeResponse(
            status_code=201,
            json_payload={
                "id": "pred456",
                "status": "processing",
                "urls": {"get": f"{BASE_URL}/predictions/pred456"},
            },
        ),
    )

    # First poll: still processing
    http.queue(
        "get",
        FakeResponse(
            status_code=200,
            json_payload={
                "id": "pred456",
                "status": "processing",
            },
        ),
    )

    # Second poll: succeeded
    http.queue(
        "get",
        FakeResponse(
            status_code=200,
            json_payload={
                "id": "pred456",
                "status": "succeeded",
                "output": "https://replicate.delivery/video2.mp4",
            },
        ),
    )

    # Download
    video_bytes = b"polled-video-content"
    http.queue(
        "get",
        FakeResponse(status_code=200, content=video_bytes),
    )

    client = _make_client(http)
    result = client.generate_text_to_video(**_default_kwargs())

    assert result == video_bytes
    # POST + 2 polls + 1 download = 4 calls
    assert len(http.calls) == 4


def test_unknown_model_raises() -> None:
    http = FakeHTTPClient()
    client = _make_client(http)

    kwargs = _default_kwargs()
    kwargs["model"] = "nonexistent-model"

    with pytest.raises(RuntimeError, match="Unknown video model"):
        client.generate_text_to_video(**kwargs)


def test_prediction_failure_raises() -> None:
    http = FakeHTTPClient()

    http.queue(
        "post",
        FakeResponse(
            status_code=201,
            json_payload={
                "id": "pred789",
                "status": "failed",
                "error": "GPU out of memory",
            },
        ),
    )

    client = _make_client(http)

    with pytest.raises(RuntimeError, match="Replicate prediction failed"):
        client.generate_text_to_video(**_default_kwargs())
