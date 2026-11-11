"""Remote Explore Enhance: status + /api/enhance-prompt without filesystem paths."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from starlette.testclient import TestClient

from remote.app import create_remote_app
from tests.fakes import FakeResponse
from tests.http_error_assertions import assert_http_error
from tests.lora_catalog_helpers import add_ic_lora
from tests.video_fixtures import write_mp4

AUTH = {"Authorization": "Bearer pair-token"}
_LOCAL_ENHANCER_MODEL_ID = "ltx-2.3-22b-distilled-1.1"


def _gemini_ok(text: str = "enhanced via gemini") -> FakeResponse:
    return FakeResponse(
        status_code=200,
        json_payload={"candidates": [{"content": {"parts": [{"text": text}]}}]},
    )


def _png_bytes(tmp_path: Path) -> bytes:
    path = tmp_path / "frame.png"
    Image.new("RGB", (32, 24), color=(10, 20, 30)).save(path)
    return path.read_bytes()


def test_prompt_enhancer_requires_auth(test_state) -> None:
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        assert client.get("/api/prompt-enhancer").status_code == 401
        assert client.post("/api/enhance-prompt", json={"prompt": "a cat"}).status_code == 401


def test_remote_status_prefers_gemini_when_key_is_set(test_state) -> None:
    test_state.state.app_settings.gemini_api_key = "gemini-key"
    test_state.state.app_settings.explore_auto_enhance_prompts = True
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        r = client.get("/api/prompt-enhancer", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["hasGeminiApiKey"] is True
    assert body["defaultProvider"] == "api"
    assert body["showManualEnhance"] is False
    assert body["exploreAutoEnhancePrompts"] is True
    assert "gemini_api_key" not in body


def test_remote_status_shows_manual_enhance_when_auto_is_off(test_state) -> None:
    test_state.state.app_settings.explore_auto_enhance_prompts = False
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        r = client.get("/api/prompt-enhancer", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["showManualEnhance"] is True
    assert body["exploreAutoEnhancePrompts"] is False


def test_remote_status_falls_back_to_local_without_gemini_key(
    test_state, create_fake_model_files
) -> None:
    create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)
    test_state.state.app_settings.gemini_api_key = ""
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        r = client.get("/api/prompt-enhancer", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["hasGeminiApiKey"] is False
    assert body["localEnhancementSupported"] is True
    assert body["defaultProvider"] == "local"
    assert body["canToggleProvider"] is False


def test_remote_enhance_uses_gemini_when_key_is_set(test_state) -> None:
    test_state.state.app_settings.gemini_api_key = "gemini-key"
    test_state.http.queue("post", _gemini_ok("a cat, enhanced"))
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        r = client.post(
            "/api/enhance-prompt",
            headers=AUTH,
            json={"prompt": "a cat", "provider": "api"},
        )
    assert r.status_code == 200
    assert r.json()["enhancedPrompt"] == "a cat, enhanced"


def test_remote_enhance_uses_local_gemma(
    test_state, fake_services, create_fake_model_files
) -> None:
    create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)
    fake_services.prompt_enhancer_pipeline.enhanced_prompt = "a long local caption"
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        r = client.post(
            "/api/enhance-prompt",
            headers=AUTH,
            json={"prompt": "a cat", "provider": "local"},
        )
    assert r.status_code == 200
    assert r.json()["enhancedPrompt"] == "a long local caption"


def test_remote_enhance_rejects_filesystem_paths(test_state, tmp_path: Path) -> None:
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        r = client.post(
            "/api/enhance-prompt",
            headers=AUTH,
            json={"prompt": "a cat", "imagePath": str(tmp_path / "secret.png")},
        )
    assert r.status_code == 422


def test_remote_enhance_rejects_blank_asset_ids(test_state) -> None:
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        whitespace = client.post(
            "/api/enhance-prompt",
            headers=AUTH,
            json={"prompt": "a cat", "imageAssetId": "   "},
        )
        empty = client.post(
            "/api/enhance-prompt",
            headers=AUTH,
            json={"prompt": "a cat", "imageAssetId": ""},
        )
        last_only = client.post(
            "/api/enhance-prompt",
            headers=AUTH,
            json={"prompt": "a cat", "lastImageAssetId": "frame-2"},
        )
    assert whitespace.status_code == 422
    assert empty.status_code == 422
    assert last_only.status_code == 422


def test_remote_enhance_resolves_image_asset_id(
    test_state, tmp_path: Path
) -> None:
    test_state.state.app_settings.gemini_api_key = "gemini-key"
    test_state.http.queue("post", _gemini_ok("a still, enhanced"))
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        uploaded = client.post(
            "/api/assets/upload",
            headers=AUTH,
            files={"file": ("frame.png", _png_bytes(tmp_path), "image/png")},
        )
        assert uploaded.status_code == 200
        asset_id = uploaded.json()["id"]
        assert "path" not in uploaded.json() or uploaded.json().get("path") in ("", None)
        r = client.post(
            "/api/enhance-prompt",
            headers=AUTH,
            json={
                "prompt": "walk in",
                "provider": "api",
                "imageAssetId": asset_id,
            },
        )
    assert r.status_code == 200
    assert r.json()["enhancedPrompt"] == "a still, enhanced"


def test_remote_enhance_reads_keyframe_asset_ids_with_a_blank_prompt(
    test_state, tmp_path: Path
) -> None:
    test_state.state.app_settings.gemini_api_key = "gemini-key"
    test_state.http.queue("post", _gemini_ok("a hall, enhanced"))
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        uploaded = client.post(
            "/api/assets/upload",
            headers=AUTH,
            files={"file": ("frame.png", _png_bytes(tmp_path), "image/png")},
        )
        assert uploaded.status_code == 200
        r = client.post(
            "/api/enhance-prompt",
            headers=AUTH,
            json={
                "prompt": "",
                "provider": "api",
                "duration": 5,
                "fps": 24,
                "keyframeAssets": [{"assetId": uploaded.json()["id"], "frameIndex": 0}],
            },
        )
    assert r.status_code == 200, r.text
    assert r.json()["enhancedPrompt"] == "a hall, enhanced"


def test_remote_enhance_accepts_the_empty_custom_lora_list_every_home_form_sends(
    test_state,
) -> None:
    # The Home Enhance request always carries customLoraIds. A body model that forbids it
    # fails Enhance on the phone for every feature.
    test_state.state.app_settings.gemini_api_key = "gemini-key"
    test_state.http.queue("post", _gemini_ok("enhanced"))
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        r = client.post(
            "/api/enhance-prompt",
            headers=AUTH,
            json={"prompt": "a cat", "provider": "api", "customLoraIds": []},
        )
    assert r.status_code == 200, r.text


def test_full_settings_stay_off_remote(test_state) -> None:
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        settings_get = client.get("/api/settings", headers=AUTH)
        settings_post = client.post(
            "/api/settings",
            headers=AUTH,
            json={"remoteExposure": "lan"},
        )
    assert settings_get.status_code == 404
    assert settings_post.status_code == 404


def test_missing_gemini_key_is_rejected_on_api_provider(test_state) -> None:
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        r = client.post(
            "/api/enhance-prompt",
            headers=AUTH,
            json={"prompt": "a cat", "provider": "api"},
        )
    assert_http_error(r, status_code=400, code="GEMINI_API_KEY_MISSING")


def test_omitted_provider_uses_status_default(
    test_state, fake_services, create_fake_model_files
) -> None:
    create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)
    fake_services.prompt_enhancer_pipeline.enhanced_prompt = "a long local caption"
    test_state.state.app_settings.gemini_api_key = ""
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        r = client.post(
            "/api/enhance-prompt",
            headers=AUTH,
            json={"prompt": "a cat"},
        )
    assert r.status_code == 200
    assert r.json()["enhancedPrompt"] == "a long local caption"


def test_remote_enhance_does_not_leak_path_when_image_is_invalid(
    test_state, tmp_path: Path
) -> None:
    test_state.state.app_settings.gemini_api_key = "gemini-key"
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        uploaded = client.post(
            "/api/assets/upload",
            headers=AUTH,
            files={"file": ("frame.png", _png_bytes(tmp_path), "image/png")},
        )
        assert uploaded.status_code == 200
        asset_id = uploaded.json()["id"]
        asset_path = Path(test_state.assets.get_asset(asset_id).path)
        asset_path.write_bytes(b"not-an-image")
        r = client.post(
            "/api/enhance-prompt",
            headers=AUTH,
            json={
                "prompt": "walk in",
                "provider": "api",
                "imageAssetId": asset_id,
            },
        )
    assert_http_error(r, status_code=400, code="ASSET_UNAVAILABLE")
    body = r.json()
    assert str(asset_path) not in body["message"]
    assert "/" not in body["message"]
    assert "\\" not in body["message"]


def test_remote_enhance_resolves_a_video_asset_id_for_an_ic_lora_with_no_prompt(
    test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)
    add_ic_lora(fake_services, id="day-to-night", name="Day to Night")
    fake_services.prompt_enhancer_pipeline.enhanced_prompt = "a night scene from the stills"
    video = write_mp4(tmp_path / "clip.mp4", seconds=6, fps=12)
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        uploaded = client.post(
            "/api/assets/upload",
            headers=AUTH,
            files={"file": ("clip.mp4", video.read_bytes(), "video/mp4")},
        )
        assert uploaded.status_code == 200
        r = client.post(
            "/api/enhance-prompt",
            headers=AUTH,
            json={
                "prompt": "",
                "provider": "local",
                "icLoraId": "day-to-night",
                "videoAssetId": uploaded.json()["id"],
            },
        )
    assert r.status_code == 200
    assert r.json()["enhancedPrompt"] == "a night scene from the stills"
    call = fake_services.prompt_enhancer_pipeline.enhance_video_frames_calls[0]
    assert call["role"] == "driving"
    assert call["frames_existed"]


def test_remote_enhance_passes_a_retake_window_on_so_the_phone_matches_the_desktop(
    test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)
    video = write_mp4(tmp_path / "clip.mp4", seconds=6, fps=12)
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        uploaded = client.post(
            "/api/assets/upload",
            headers=AUTH,
            files={"file": ("clip.mp4", video.read_bytes(), "video/mp4")},
        )
        r = client.post(
            "/api/enhance-prompt",
            headers=AUTH,
            json={
                "prompt": "the woman turns away",
                "provider": "local",
                "videoAssetId": uploaded.json()["id"],
                "replacedWindow": {"start": 2, "end": 4},
            },
        )

    assert r.status_code == 200
    times = fake_services.prompt_enhancer_pipeline.enhance_video_frames_calls[0]["frame_times"]
    assert not any(2 < time < 4 for time in times)


@pytest.mark.parametrize(
    "body",
    [
        {"icLoraId": "day-to-night", "loraCatalogIds": ["a"]},
        {"icLoraId": "day-to-night", "videoAssetId": "v", "replacedWindow": {"start": 1, "end": 2}},
        {"replacedWindow": {"start": 1, "end": 2}},
    ],
    ids=["ic-lora-with-lora", "window-with-ic-lora", "window-without-video"],
)
def test_remote_enhance_rejects_a_bad_selection_mix_with_422_not_500(
    test_state, fake_services, body: dict[str, object]
) -> None:
    add_ic_lora(fake_services, id="day-to-night", name="Day to Night")
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        r = client.post(
            "/api/enhance-prompt",
            headers=AUTH,
            json={"prompt": "a cat", "provider": "local", **body},
        )

    assert r.status_code == 422
    assert "validation error" not in r.text


def test_remote_enhance_accepts_media_type_image_with_the_image_system_prompt(test_state) -> None:
    test_state.state.app_settings.gemini_api_key = "gemini-key"
    test_state.http.queue("post", _gemini_ok("a shiny red sedan"))
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        r = client.post(
            "/api/enhance-prompt",
            headers=AUTH,
            json={"prompt": "a red car", "provider": "api", "mediaType": "image"},
        )

    assert r.status_code == 200
    assert r.json()["enhancedPrompt"] == "a shiny red sedan"
    instruction = test_state.http.calls[-1].json_payload["systemInstruction"]["parts"][0]["text"]
    assert "Z-Image-Turbo" in instruction


def test_remote_image_enhance_refuses_an_input_image_so_edit_stays_unreachable(test_state) -> None:
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        r = client.post(
            "/api/enhance-prompt",
            headers=AUTH,
            json={"prompt": "a red car", "mediaType": "image", "imageAssetId": "asset-1"},
        )

    assert r.status_code == 422
