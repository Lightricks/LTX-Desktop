"""Remote Explore Enhance: status + /api/enhance-prompt without filesystem paths."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from starlette.testclient import TestClient

from remote.app import create_remote_app
from tests.fakes import FakeResponse
from tests.http_error_assertions import assert_http_error

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
