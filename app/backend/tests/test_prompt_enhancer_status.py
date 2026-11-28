"""GET /api/prompt-enhancer and Explore asset-id Enhance on the Desktop app."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


_LOCAL_ENHANCER_MODEL_ID = "ltx-2.3-22b-distilled-1.1"


class TestPromptEnhancerStatus:
    def test_hides_manual_enhance_when_explore_auto_is_on(
        self, client, test_state, create_fake_model_files
    ):
        create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)
        test_state.state.app_settings.explore_auto_enhance_prompts = True
        test_state.state.app_settings.gemini_api_key = ""

        r = client.get("/api/prompt-enhancer")
        assert r.status_code == 200
        body = r.json()
        assert body["showManualEnhance"] is False
        assert body["defaultProvider"] == "local"
        assert body["localEnhancementSupported"] is True
        assert body["hasGeminiApiKey"] is False

    def test_whitespace_gemini_key_is_not_a_key(self, client, test_state):
        test_state.state.app_settings.gemini_api_key = "   "
        test_state.state.app_settings.explore_auto_enhance_prompts = False

        r = client.get("/api/prompt-enhancer")
        assert r.status_code == 200
        assert r.json()["hasGeminiApiKey"] is False

    def test_prefers_persisted_api_when_both_providers_exist(
        self, client, test_state, create_fake_model_files
    ):
        create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)
        test_state.state.app_settings.gemini_api_key = "key"
        test_state.state.app_settings.prompt_enhancer_provider_preference = "api"
        test_state.state.app_settings.explore_auto_enhance_prompts = False

        r = client.get("/api/prompt-enhancer")
        assert r.status_code == 200
        body = r.json()
        assert body["showManualEnhance"] is True
        assert body["defaultProvider"] == "api"
        assert body["canToggleProvider"] is True


class TestEnhancePromptAssetIds:
    def test_image_asset_id_routes_to_enhance_i2v(
        self, client, fake_services, create_fake_model_files, tmp_path
    ):
        create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)
        image_path = tmp_path / "cat.png"
        Image.new("RGB", (32, 24)).save(image_path)
        ingested = client.post("/api/assets", json={"path": str(image_path)})
        assert ingested.status_code == 200
        asset_id = ingested.json()["id"]

        r = client.post(
            "/api/enhance-prompt",
            json={"prompt": "a cat", "imageAssetId": asset_id},
        )
        assert r.status_code == 200
        call = fake_services.prompt_enhancer_pipeline.enhance_i2v_calls[0]
        assert call["image_path"]
        assert Path(call["image_path"]).is_file()
