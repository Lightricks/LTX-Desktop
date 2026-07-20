"""Opaque media upload and generated artifact integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from _routes._errors import HTTPError
from tests.fakes.services import FakeResponse


def _upload_image(client, make_test_image) -> dict[str, object]:
    response = client.post(
        "/api/media",
        data={"media_type": "image"},
        files={"file": ("frame.png", make_test_image(), "image/png")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_upload_resolve_and_delete_image(client, test_state, make_test_image) -> None:
    data = _upload_image(client, make_test_image)
    media_id = str(data["media_id"])
    assert media_id.startswith("med_")
    assert data["media_type"] == "image"
    assert data["content_type"] == "image/png"
    assert int(data["size_bytes"]) > 0

    record = test_state.media_store.resolve_upload(media_id)
    assert record is not None
    assert record.path.is_file()
    assert record.path.name == "content.png"

    deleted = client.delete(f"/api/media/{media_id}")
    assert deleted.status_code == 200
    assert test_state.media_store.resolve_upload(media_id) is None


def test_upload_rejects_invalid_image(client) -> None:
    response = client.post(
        "/api/media",
        data={"media_type": "image"},
        files={"file": ("frame.png", b"not-an-image", "image/png")},
    )
    assert response.status_code == 400


def test_uploaded_media_type_must_match_requested_input(
    client,
    test_state,
    make_test_image,
) -> None:
    uploaded = _upload_image(client, make_test_image)

    with pytest.raises(HTTPError) as exc_info:
        test_state.media.resolve_input(
            media_id=str(uploaded["media_id"]),
            legacy_path=None,
            expected_type="video",
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "MEDIA_TYPE_MISMATCH"


def test_artifact_download_range_and_delete(client, test_state) -> None:
    output = test_state.config.outputs_dir / "generated.mp4"
    output.write_bytes(b"0123456789")
    artifact = test_state.media.register_artifact(output, media_type="video")

    response = client.get(
        f"/api/artifacts/{artifact.artifact_id}",
        headers={"Range": "bytes=2-5"},
    )
    assert response.status_code == 206
    assert response.content == b"2345"
    assert response.headers["content-range"] == "bytes 2-5/10"
    assert response.headers["x-content-type-options"] == "nosniff"

    deleted = client.delete(f"/api/artifacts/{artifact.artifact_id}")
    assert deleted.status_code == 200
    assert not output.exists()
    assert client.get(f"/api/artifacts/{artifact.artifact_id}").status_code == 404


def test_artifact_registration_rejects_file_outside_outputs(test_state, tmp_path: Path) -> None:
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"video")
    try:
        test_state.media.register_artifact(outside, media_type="video")
    except Exception as exc:
        assert getattr(exc, "code", None) == "ARTIFACT_REGISTRATION_FAILED"
    else:
        raise AssertionError("outside artifact was accepted")


def test_standalone_rejects_legacy_path_and_accepts_media_id(
    client,
    test_state,
    make_test_image,
) -> None:
    uploaded = _upload_image(client, make_test_image)
    test_state.config.deployment_mode = "standalone"
    test_state.config.allow_legacy_path_inputs = False

    local_image = test_state.config.outputs_dir / "local.png"
    local_image.write_bytes(b"irrelevant")
    rejected = client.post(
        "/api/generate",
        json={"prompt": "test", "imagePath": str(local_image)},
    )
    assert rejected.status_code == 422
    assert rejected.json()["code"] == "REMOTE_PATH_INPUT_DISABLED"

    # ID resolution itself remains available in standalone mode. Generation
    # may fail later because this test intentionally does not install models.
    resolved = test_state.media.resolve_input(
        media_id=str(uploaded["media_id"]),
        legacy_path=None,
        expected_type="image",
        required=True,
    )
    assert resolved is not None
    assert resolved.is_file()


def test_generate_with_media_id_returns_downloadable_artifact(
    client,
    test_state,
    make_test_image,
    create_fake_model_files,
) -> None:
    create_fake_model_files()
    test_state.state.app_settings.use_local_text_encoder = True
    uploaded = _upload_image(client, make_test_image)

    generated = client.post(
        "/api/generate",
        json={"prompt": "remote image to video", "imageMediaId": uploaded["media_id"]},
    )
    assert generated.status_code == 200, generated.text
    artifact = generated.json()["artifact"]
    assert artifact["artifact_id"].startswith("art_")
    downloaded = client.get(f"/api/artifacts/{artifact['artifact_id']}")
    assert downloaded.status_code == 200
    assert downloaded.content == b"fake-video"


def test_ic_lora_extract_accepts_video_media_id(client) -> None:
    uploaded = client.post(
        "/api/media",
        data={"media_type": "video"},
        files={"file": ("input.mp4", b"fake-video", "video/mp4")},
    )
    assert uploaded.status_code == 200, uploaded.text
    extracted = client.post(
        "/api/ic-lora/extract-conditioning",
        json={"video_media_id": uploaded.json()["media_id"], "conditioning_type": "canny"},
    )
    assert extracted.status_code == 200, extracted.text
    assert extracted.json()["conditioning"].startswith("data:image/jpeg;base64,")


def test_gap_prompt_accepts_frame_media_id(client, test_state, fake_services, make_test_image) -> None:
    uploaded = _upload_image(client, make_test_image)
    test_state.state.app_settings.gemini_api_key = "gemini-test"
    fake_services.http.queue(
        "post",
        FakeResponse(
            json_payload={
                "candidates": [{"content": {"parts": [{"text": "A seamless transition"}]}}]
            }
        ),
    )
    response = client.post(
        "/api/suggest-gap-prompt",
        json={"beforeFrameMediaId": uploaded["media_id"], "gapDuration": 3},
    )
    assert response.status_code == 200, response.text
    assert response.json()["suggested_prompt"] == "A seamless transition"
