"""Tests for GET /api/settings and POST /api/settings."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from state.app_settings import AppSettings, UpdateSettingsRequest, resolved_use_conv_vae
from services.gemini_text_client import (
    DEFAULT_GEMINI_MODEL,
    apply_gemini_thinking_config,
    clear_gemini_models_cache,
    gemini_thinking_config_for_model,
    is_text_to_text_gemini_model,
    resolve_gemini_model,
)
from services.interfaces import HttpTransportError
from state import build_initial_state
from app_handler import ServiceBundle
from tests.conftest import TEST_ADMIN_TOKEN
from tests.fakes import FakeResponse
from tests.fakes.services import FakeServices
from tests.http_error_assertions import assert_http_error


class TestGetSettings:
    def test_default_settings(self, client, default_app_settings, test_state):
        r = client.get("/api/settings")
        assert r.status_code == 200
        data = r.json()
        assert data["useTorchCompile"] is False
        assert data["hasLtxApiKey"] is False
        assert data["userPrefersLtxApiVideoGenerations"] is False
        assert data["hasFalApiKey"] is False
        assert data["useLocalTextEncoder"] is False
        assert data["promptCacheSize"] == 100
        assert data["promptEnhancerEnabledT2V"] is True
        assert data["promptEnhancerEnabledI2V"] is False
        assert data["hasGeminiApiKey"] is False
        assert data["geminiModel"] == ""
        assert data["seedLocked"] is False
        assert data["lockedSeed"] == 42
        assert data["useConvVae"] is resolved_use_conv_vae(AppSettings())
        # When no custom path is set, the response surfaces the runtime default
        # so the first-run UI can show the install location.
        assert data["modelsDir"] == str(test_state.config.default_models_dir)
        assert "fastModel" not in data
        assert "proModel" not in data
        assert "ltxApiKey" not in data
        assert "falApiKey" not in data
        assert "geminiApiKey" not in data

    def test_reflects_changed_settings(self, client, test_state):
        test_state.state.app_settings.use_torch_compile = True
        r = client.get("/api/settings")
        assert r.json()["useTorchCompile"] is True

    def test_has_api_key_true_when_set(self, client, test_state):
        test_state.state.app_settings.ltx_api_key = "test-key-123"
        r = client.get("/api/settings")
        data = r.json()
        assert data["hasLtxApiKey"] is True
        assert "ltxApiKey" not in data


class TestPostSettings:
    def test_update_single_field(self, client, test_state):
        r = client.post("/api/settings", json={"useTorchCompile": True})
        assert r.status_code == 200
        assert test_state.state.app_settings.use_torch_compile is True

    def test_update_multiple_fields(self, client, test_state):
        r = client.post("/api/settings", json={"useTorchCompile": True, "promptCacheSize": 42})
        assert r.status_code == 200
        assert test_state.state.app_settings.use_torch_compile is True
        assert test_state.state.app_settings.prompt_cache_size == 42

    def test_prompt_cache_size_clamped_max(self, client, test_state):
        r = client.post("/api/settings", json={"promptCacheSize": 5000})
        assert r.status_code == 200
        assert test_state.state.app_settings.prompt_cache_size <= 1000

    def test_prompt_cache_size_clamped_min(self, client, test_state):
        r = client.post("/api/settings", json={"promptCacheSize": -10})
        assert r.status_code == 200
        assert test_state.state.app_settings.prompt_cache_size >= 0

    def test_locked_seed_clamped_range(self, client, test_state):
        r = client.post("/api/settings", json={"lockedSeed": 9_999_999_999})
        assert r.status_code == 200
        assert test_state.state.app_settings.locked_seed == 2_147_483_647

    def test_prompt_cache_shrinks_cache(self, client, test_state):
        te = test_state.state.text_encoder
        assert te is not None
        for i in range(5):
            te.prompt_cache[(f"key_{i}", False)] = f"value_{i}"  # type: ignore[assignment]

        r = client.post("/api/settings", json={"promptCacheSize": 2})
        assert r.status_code == 200
        assert len(te.prompt_cache) <= 2

    def test_update_api_keys(self, client, test_state):
        r = client.post(
            "/api/settings",
            json={
                "ltxApiKey": "ltx-key-abc",
                "geminiApiKey": "gemini-key-xyz",
                "falApiKey": "fal-key-123",
            },
        )
        assert r.status_code == 200
        assert test_state.state.app_settings.ltx_api_key == "ltx-key-abc"
        assert test_state.state.app_settings.gemini_api_key == "gemini-key-xyz"
        assert test_state.state.app_settings.fal_api_key == "fal-key-123"

    def test_update_user_prefers_api_video_generations(self, client, test_state):
        r = client.post("/api/settings", json={"userPrefersLtxApiVideoGenerations": True})
        assert r.status_code == 200
        assert test_state.state.app_settings.user_prefers_ltx_api_video_generations is True

    def test_empty_string_does_not_erase_key(self, client, test_state):
        test_state.state.app_settings.ltx_api_key = "real-key"
        test_state.state.app_settings.fal_api_key = "fal-key"
        r = client.post("/api/settings", json={"ltxApiKey": "", "falApiKey": ""})
        assert r.status_code == 200
        assert test_state.state.app_settings.ltx_api_key == "real-key"
        assert test_state.state.app_settings.fal_api_key == "fal-key"

    def test_empty_gemini_model_persists_as_use_default(self, client, test_state):
        # Unlike API keys, an empty model string is stored so generate-time resolution can
        # fall back to DEFAULT_GEMINI_MODEL without keeping a stale explicit id.
        r = client.post("/api/settings", json={"geminiModel": "gemini-2.0-flash"})
        assert r.status_code == 200
        assert test_state.state.app_settings.gemini_model == "gemini-2.0-flash"
        assert client.get("/api/settings").json()["geminiModel"] == "gemini-2.0-flash"

        r = client.post("/api/settings", json={"geminiModel": ""})
        assert r.status_code == 200
        assert test_state.state.app_settings.gemini_model == ""
        assert client.get("/api/settings").json()["geminiModel"] == ""

    def test_non_text_gemini_model_persists_as_use_default(self, client, test_state):
        for model_id in ("nano-banana-pro", "gemma-4-31b-it"):
            r = client.post("/api/settings", json={"geminiModel": model_id})
            assert r.status_code == 200
            assert test_state.state.app_settings.gemini_model == ""
            assert client.get("/api/settings").json()["geminiModel"] == ""

    def test_omitted_gemini_model_does_not_erase(self, client, test_state):
        test_state.state.app_settings.gemini_model = "gemini-2.0-flash"
        r = client.post("/api/settings", json={"useTorchCompile": True})
        assert r.status_code == 200
        assert test_state.state.app_settings.gemini_model == "gemini-2.0-flash"

    def test_omitted_key_does_not_erase_key(self, client, test_state):
        test_state.state.app_settings.ltx_api_key = "real-key"
        r = client.post("/api/settings", json={"useTorchCompile": True})
        assert r.status_code == 200
        assert test_state.state.app_settings.ltx_api_key == "real-key"

    def test_unknown_field_rejected(self, client):
        r = client.post("/api/settings", json={"unknownSetting": True})
        assert r.status_code == 422

    def test_use_conv_vae_round_trip(self, client, test_state):
        r = client.post("/api/settings", json={"useConvVae": True})
        assert r.status_code == 200
        assert test_state.state.app_settings.use_conv_vae is True
        assert client.get("/api/settings").json()["useConvVae"] is True

        r = client.post("/api/settings", json={"useConvVae": False})
        assert r.status_code == 200
        assert test_state.state.app_settings.use_conv_vae is False
        assert client.get("/api/settings").json()["useConvVae"] is False

    def test_use_conv_vae_change_unloads_gpu_pipeline(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        create_fake_model_files()
        test_state.state.app_settings.use_conv_vae = False
        test_state.pipelines.load_gpu_pipeline("fast")
        assert test_state.state.gpu_slot is not None
        cleanup_before = fake_services.gpu_cleaner.cleanup_calls

        r = client.post("/api/settings", json={"useConvVae": True})
        assert r.status_code == 200
        assert test_state.state.gpu_slot is None
        assert fake_services.gpu_cleaner.cleanup_calls > cleanup_before

    def test_use_conv_vae_unchanged_does_not_unload_gpu_pipeline(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        create_fake_model_files()
        test_state.state.app_settings.use_conv_vae = True
        test_state.pipelines.load_gpu_pipeline("fast")
        cleanup_before = fake_services.gpu_cleaner.cleanup_calls

        r = client.post("/api/settings", json={"useConvVae": True})
        assert r.status_code == 200
        assert test_state.state.gpu_slot is not None
        assert fake_services.gpu_cleaner.cleanup_calls == cleanup_before


class TestModelsDirAdminGuard:
    def test_models_dir_requires_admin_token(self, client, test_state):
        r = client.post("/api/settings", json={"modelsDir": "/tmp/new-models"})
        assert r.status_code == 403

    def test_models_dir_with_wrong_admin_token(self, client, test_state):
        r = client.post(
            "/api/settings",
            json={"modelsDir": "/tmp/new-models"},
            headers={"X-Admin-Token": "wrong-token"},
        )
        assert r.status_code == 403

    def test_models_dir_with_valid_admin_token(self, client, test_state):
        r = client.post(
            "/api/settings",
            json={"modelsDir": "/tmp/new-models"},
            headers={"X-Admin-Token": TEST_ADMIN_TOKEN},
        )
        assert r.status_code == 200
        assert test_state.state.app_settings.models_dir == "/tmp/new-models"

    def test_non_admin_fields_without_admin_token(self, client, test_state):
        r = client.post("/api/settings", json={"useTorchCompile": True})
        assert r.status_code == 200
        assert test_state.state.app_settings.use_torch_compile is True

    def test_effective_models_dir_uses_custom(self, client, test_state):
        test_state.state.app_settings.models_dir = "/custom/models"
        assert test_state.models.models_dir == Path("/custom/models")

    def test_effective_models_dir_fallback(self, client, test_state):
        assert test_state.state.app_settings.models_dir == ""
        assert test_state.models.models_dir == test_state.config.default_models_dir

    def test_models_dir_persists_and_loads(self, client, test_state, default_app_settings):
        r = client.post(
            "/api/settings",
            json={"modelsDir": "/tmp/persisted-models"},
            headers={"X-Admin-Token": TEST_ADMIN_TOKEN},
        )
        assert r.status_code == 200

        fake_services = FakeServices()
        bundle = ServiceBundle(
            http=fake_services.http,
            gpu_cleaner=fake_services.gpu_cleaner,
            model_downloader=fake_services.model_downloader,
            lora_catalog_provider=fake_services.lora_catalog_provider,
            gpu_info=fake_services.gpu_info,
            video_processor=fake_services.video_processor,
            text_encoder=fake_services.text_encoder,
            task_runner=fake_services.task_runner,
            ltx_api_client=fake_services.ltx_api_client,
            zit_api_client=fake_services.zit_api_client,
            fast_video_pipeline_class=type(fake_services.fast_video_pipeline),
            image_generation_pipeline_class=type(fake_services.image_generation_pipeline),
            ic_lora_pipeline_class=type(fake_services.ic_lora_pipeline),
            depth_processor_pipeline_class=type(fake_services.depth_processor_pipeline),
            pose_processor_pipeline_class=type(fake_services.pose_processor_pipeline),
            a2v_pipeline_class=type(fake_services.a2v_pipeline),
            retake_pipeline_class=type(fake_services.retake_pipeline),
            prompt_enhancer_pipeline_class=type(fake_services.prompt_enhancer_pipeline),
        )
        loaded = build_initial_state(test_state.config, default_app_settings.model_copy(deep=True), service_bundle=bundle)
        assert loaded.state.app_settings.models_dir == "/tmp/persisted-models"
        assert loaded.models.models_dir == Path("/tmp/persisted-models")


class TestSettingsPersistence:
    def _new_state(self, test_state, default_app_settings):
        fake_services = FakeServices()
        bundle = ServiceBundle(
            http=fake_services.http,
            gpu_cleaner=fake_services.gpu_cleaner,
            model_downloader=fake_services.model_downloader,
            lora_catalog_provider=fake_services.lora_catalog_provider,
            gpu_info=fake_services.gpu_info,
            video_processor=fake_services.video_processor,
            text_encoder=fake_services.text_encoder,
            task_runner=fake_services.task_runner,
            ltx_api_client=fake_services.ltx_api_client,
            zit_api_client=fake_services.zit_api_client,
            fast_video_pipeline_class=type(fake_services.fast_video_pipeline),
            image_generation_pipeline_class=type(fake_services.image_generation_pipeline),
            ic_lora_pipeline_class=type(fake_services.ic_lora_pipeline),
            depth_processor_pipeline_class=type(fake_services.depth_processor_pipeline),
            pose_processor_pipeline_class=type(fake_services.pose_processor_pipeline),
            a2v_pipeline_class=type(fake_services.a2v_pipeline),
            retake_pipeline_class=type(fake_services.retake_pipeline),
            prompt_enhancer_pipeline_class=type(fake_services.prompt_enhancer_pipeline),
        )
        return build_initial_state(test_state.config, default_app_settings.model_copy(deep=True), service_bundle=bundle)

    def test_load_settings_clamps_from_disk_and_ignores_removed_fields(self, test_state, default_app_settings):
        test_state.config.settings_file.write_text(
            json.dumps(
                {
                    "prompt_cache_size": 5000,
                    "locked_seed": -55,
                    "fast_model": {"use_upscaler": False},
                    "pro_model": {"steps": 999},
                }
            ),
            encoding="utf-8",
        )

        loaded = self._new_state(test_state, default_app_settings)
        assert loaded.state.app_settings.prompt_cache_size == 1000
        assert loaded.state.app_settings.locked_seed == 0
        assert "fast_model" not in loaded.state.app_settings.model_dump(by_alias=False)
        assert "pro_model" not in loaded.state.app_settings.model_dump(by_alias=False)

    def test_legacy_prompt_enhancer_key_migrates(self, test_state, default_app_settings):
        test_state.config.settings_file.write_text(
            json.dumps({"prompt_enhancer_enabled": False}),
            encoding="utf-8",
        )

        loaded = self._new_state(test_state, default_app_settings)
        assert loaded.state.app_settings.prompt_enhancer_enabled_t2v is False
        assert loaded.state.app_settings.prompt_enhancer_enabled_i2v is False

    def test_user_prefers_api_video_generations_persists(self, client, test_state, default_app_settings):
        r = client.post("/api/settings", json={"userPrefersLtxApiVideoGenerations": True})
        assert r.status_code == 200
        assert test_state.state.app_settings.user_prefers_ltx_api_video_generations is True

        loaded = self._new_state(test_state, default_app_settings)
        assert loaded.state.app_settings.user_prefers_ltx_api_video_generations is True

    def test_gemini_model_persists_and_loads(self, client, test_state, default_app_settings):
        r = client.post("/api/settings", json={"geminiModel": "gemini-2.0-flash"})
        assert r.status_code == 200
        assert test_state.state.app_settings.gemini_model == "gemini-2.0-flash"

        loaded = self._new_state(test_state, default_app_settings)
        assert loaded.state.app_settings.gemini_model == "gemini-2.0-flash"


class TestSettingsSchemaDrift:
    def test_update_request_tracks_app_settings_fields(self):
        assert set(AppSettings.model_fields) == set(UpdateSettingsRequest.model_fields)


class TestResolvedUseConvVae:
    def test_none_defaults_on_for_darwin(self, monkeypatch):
        monkeypatch.setattr("state.app_settings.sys.platform", "darwin")
        assert resolved_use_conv_vae(AppSettings()) is True

    def test_none_defaults_off_for_linux(self, monkeypatch):
        monkeypatch.setattr("state.app_settings.sys.platform", "linux")
        assert resolved_use_conv_vae(AppSettings()) is False

    def test_none_defaults_off_for_windows(self, monkeypatch):
        monkeypatch.setattr("state.app_settings.sys.platform", "win32")
        assert resolved_use_conv_vae(AppSettings()) is False

    def test_explicit_true_overrides_linux_default(self, monkeypatch):
        monkeypatch.setattr("state.app_settings.sys.platform", "linux")
        assert resolved_use_conv_vae(AppSettings(use_conv_vae=True)) is True

    def test_explicit_false_overrides_darwin_default(self, monkeypatch):
        monkeypatch.setattr("state.app_settings.sys.platform", "darwin")
        assert resolved_use_conv_vae(AppSettings(use_conv_vae=False)) is False


def _gemini_listed_model(
    name: str,
    *,
    display_name: str | None = None,
    methods: list[str] | None = None,
    description: str = "",
    input_modalities: list[str] | None = None,
    output_modalities: list[str] | None = None,
) -> dict[str, object]:
    model_id = name.removeprefix("models/")
    payload: dict[str, object] = {
        "name": name,
        "displayName": display_name if display_name is not None else model_id,
        "description": description,
        "supportedGenerationMethods": methods if methods is not None else ["generateContent"],
    }
    if input_modalities is not None:
        payload["supportedInputModalities"] = input_modalities
    if output_modalities is not None:
        payload["supportedOutputModalities"] = output_modalities
    return payload


@pytest.mark.parametrize(
    ("model_id", "description", "display_name", "expected"),
    [
        ("gemini-2.5-flash-lite", "Fast text model", "Gemini 2.5 Flash-Lite", True),
        ("gemini-2.5-flash", "Multimodal model that understands images and video", "Gemini 2.5 Flash", True),
        ("gemini-2.5-pro", "", "Gemini 2.5 Pro", True),
        ("gemini-2.5-pro-preview-06-05", "", "Gemini 2.5 Pro Preview", True),
        ("gemini-3-flash-preview", "", "Gemini 3 Flash Preview", True),
        ("gemini-3.1-pro-preview", "", "Gemini 3.1 Pro Preview", True),
        ("gemini-pro", "", "Gemini Pro", True),
        ("gemma-3-27b-it", "Open text model", "Gemma 4 31B IT", False),
        ("gemma-4-31b-it", "", "Gemma 4 31B IT", False),
        ("nano-banana-pro", "", "Nano Banana Pro", False),
        ("gemini-3-pro-image", "", "Nano Banana Pro", False),
        ("lyria-3-clip-preview", "", "Lyria 3 Clip Preview", False),
        ("lyria-3-pro-preview", "", "Lyria 3 Pro Preview", False),
        ("antigravity-preview-05-2026", "", "Antigravity Agent Preview", False),
        ("deep-research-preview-04-2026", "", "Deep Research Preview (Apr-21-2026)", False),
        ("gemini-omni-flash-preview", "", "Gemini Omni Flash Preview", False),
        ("text-embedding-004", "", "", False),
        ("gemini-embedding-001", "", "", False),
        ("gemini-2.5-flash-image", "", "Gemini 2.5 Flash Image", False),
        ("gemini-2.0-flash-preview-image-generation", "", "", False),
        ("gemini-2.5-flash-preview-tts", "", "", False),
        ("gemini-2.5-flash-native-audio-dialog", "", "", False),
        ("veo-2.0-generate-001", "", "", False),
        ("imagen-3.0-generate-002", "", "", False),
        ("gemini-2.0-flash-exp", "Experimental release of Gemini 2.0 Flash", "", True),
        ("gemini-2.5-flash", "Image generation model", "", False),
    ],
)
def test_is_text_to_text_gemini_model(
    model_id: str, description: str, display_name: str, expected: bool
) -> None:
    assert (
        is_text_to_text_gemini_model(model_id, description, display_name=display_name) is expected
    )


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        ("", DEFAULT_GEMINI_MODEL),
        ("   ", DEFAULT_GEMINI_MODEL),
        ("gemini-2.0-flash", "gemini-2.0-flash"),
        ("models/gemini-2.0-flash", "gemini-2.0-flash"),
        ("gemini-custom-id", "gemini-custom-id"),
        ("nano-banana-pro", DEFAULT_GEMINI_MODEL),
        ("gemma-4-31b-it", DEFAULT_GEMINI_MODEL),
        ("gemini-2.5-pro", "gemini-2.5-pro"),
        ("gemini-3-flash-preview", "gemini-3-flash-preview"),
        ("gemini-3.1-pro-preview", "gemini-3.1-pro-preview"),
        ("gemini-3-pro-image", DEFAULT_GEMINI_MODEL),
        ("gemini-omni-flash-preview", DEFAULT_GEMINI_MODEL),
    ],
)
def test_resolve_gemini_model(stored: str, expected: str) -> None:
    assert resolve_gemini_model(stored) == expected


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gemini-2.5-flash-lite", {"thinkingBudget": 0}),
        ("gemini-2.5-flash", {"thinkingBudget": 0}),
        ("gemini-2.5-pro", {"thinkingBudget": 128}),
        ("gemini-2.5-pro-preview-06-05", {"thinkingBudget": 128}),
        ("gemini-2.0-flash", None),
        ("gemini-3-flash-preview", {"thinkingLevel": "LOW"}),
        ("gemini-3.1-pro-preview", {"thinkingLevel": "LOW"}),
    ],
)
def test_gemini_thinking_config_for_model(model: str, expected: dict[str, object] | None) -> None:
    assert gemini_thinking_config_for_model(model) == expected


def test_apply_gemini_thinking_config_raises_output_cap_when_thinking_stays_on() -> None:
    flash = apply_gemini_thinking_config("gemini-2.5-flash-lite", {"maxOutputTokens": 512})
    assert flash["maxOutputTokens"] == 512
    assert flash["thinkingConfig"] == {"thinkingBudget": 0}

    pro = apply_gemini_thinking_config("gemini-3.1-pro-preview", {"maxOutputTokens": 512})
    assert pro["maxOutputTokens"] == 2048
    assert pro["thinkingConfig"] == {"thinkingLevel": "LOW"}


def test_is_text_to_text_gemini_model_uses_output_modalities_when_present() -> None:
    assert (
        is_text_to_text_gemini_model(
            "gemini-2.5-flash",
            input_modalities=["TEXT", "IMAGE"],
            output_modalities=["TEXT"],
        )
        is True
    )
    assert (
        is_text_to_text_gemini_model(
            "gemini-3-pro-image",
            display_name="Nano Banana Pro",
            input_modalities=["TEXT"],
            output_modalities=["IMAGE"],
        )
        is False
    )
    assert (
        is_text_to_text_gemini_model(
            "gemini-2.5-flash",
            input_modalities=["TEXT"],
            output_modalities=["TEXT"],
        )
        is True
    )
    assert (
        is_text_to_text_gemini_model(
            "gemini-2.5-pro",
            input_modalities=["TEXT", "IMAGE"],
            output_modalities=["TEXT"],
        )
        is True
    )
    assert (
        is_text_to_text_gemini_model(
            "gemini-3-flash-preview",
            input_modalities=["TEXT"],
            output_modalities=["TEXT"],
        )
        is True
    )
    assert (
        is_text_to_text_gemini_model(
            "gemma-4-31b-it",
            input_modalities=["TEXT"],
            output_modalities=["TEXT"],
        )
        is False
    )


class TestListGeminiModels:
    def test_missing_key_400(self, client):
        r = client.get("/api/settings/gemini-models")
        assert_http_error(r, status_code=400, code="GEMINI_API_KEY_MISSING")

    def test_filters_generate_content_and_sorts_flash_lite_first(self, client, test_state):
        test_state.state.app_settings.gemini_api_key = "key"
        test_state.http.queue(
            "get",
            FakeResponse(
                status_code=200,
                json_payload={
                    "models": [
                        _gemini_listed_model("models/gemini-pro", display_name="Gemini Pro"),
                        _gemini_listed_model(
                            "models/text-embedding-004",
                            display_name="Embedding 004",
                            methods=["embedContent"],
                        ),
                        _gemini_listed_model(
                            "models/gemini-2.5-flash",
                            display_name="Gemini 2.5 Flash",
                            description="Multimodal model that understands images and video",
                        ),
                        _gemini_listed_model(
                            "models/gemini-2.0-flash-lite",
                            display_name="Gemini 2.0 Flash-Lite",
                        ),
                        _gemini_listed_model("models/gemini-2.5-flash-lite"),
                        _gemini_listed_model("models/gemini-1.5-pro", display_name="Gemini 1.5 Pro"),
                        _gemini_listed_model(
                            "models/gemini-2.5-flash-preview-tts",
                            display_name="Gemini 2.5 Flash Preview TTS",
                        ),
                        _gemini_listed_model(
                            "models/gemini-2.5-flash-image",
                            display_name="Gemini 2.5 Flash Image",
                            description="Image generation model",
                        ),
                        _gemini_listed_model(
                            "models/imagen-3.0-generate-002",
                            display_name="Imagen 3",
                        ),
                        _gemini_listed_model(
                            "models/gemini-2.5-flash-native-audio-dialog",
                            display_name="Native Audio Dialog",
                        ),
                        _gemini_listed_model(
                            "models/nano-banana-pro",
                            display_name="Nano Banana Pro",
                        ),
                        _gemini_listed_model(
                            "models/gemma-4-31b-it",
                            display_name="Gemma 4 31B IT",
                        ),
                        _gemini_listed_model(
                            "models/gemini-3-pro-image",
                            display_name="Nano Banana Pro",
                        ),
                        _gemini_listed_model(
                            "models/lyria-3-clip-preview",
                            display_name="Lyria 3 Clip Preview",
                        ),
                        _gemini_listed_model(
                            "models/antigravity-preview-05-2026",
                            display_name="Antigravity Agent Preview",
                        ),
                        _gemini_listed_model(
                            "models/deep-research-preview-04-2026",
                            display_name="Deep Research Preview (Apr-21-2026)",
                        ),
                        _gemini_listed_model(
                            "models/gemini-omni-flash-preview",
                            display_name="Gemini Omni Flash Preview",
                        ),
                        _gemini_listed_model(
                            "models/gemini-2.5-pro",
                            display_name="Gemini 2.5 Pro",
                        ),
                        _gemini_listed_model(
                            "models/gemini-3-flash-preview",
                            display_name="Gemini 3 Flash Preview",
                        ),
                    ]
                },
            ),
        )

        r = client.get("/api/settings/gemini-models")
        assert r.status_code == 200
        data = r.json()
        ids = [model["id"] for model in data["models"]]
        assert ids == [
            "gemini-2.0-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-3-flash-preview",
            "gemini-1.5-pro",
            "gemini-2.5-pro",
            "gemini-pro",
        ]
        assert "text-embedding-004" not in ids
        assert "gemini-2.5-flash-preview-tts" not in ids
        assert "gemini-2.5-flash-image" not in ids
        assert "imagen-3.0-generate-002" not in ids
        assert "gemini-2.5-flash-native-audio-dialog" not in ids
        assert "nano-banana-pro" not in ids
        assert "gemma-4-31b-it" not in ids
        assert "gemini-3-pro-image" not in ids
        assert "lyria-3-clip-preview" not in ids
        assert "antigravity-preview-05-2026" not in ids
        assert "deep-research-preview-04-2026" not in ids
        assert "gemini-omni-flash-preview" not in ids
        flash = next(model for model in data["models"] if model["id"] == "gemini-2.5-flash")
        assert flash["description"] == "Multimodal model that understands images and video"
        assert data["resolvedModel"] == DEFAULT_GEMINI_MODEL

        call = test_state.http.calls[-1]
        assert call.method == "get"
        assert call.url.startswith("https://generativelanguage.googleapis.com/v1beta/models?")
        assert "pageSize=1000" in call.url
        assert "key=" not in call.url
        assert call.headers is not None
        assert call.headers["x-goog-api-key"] == "key"

    def test_resolved_model_is_stored_setting_when_present(self, client, test_state):
        test_state.state.app_settings.gemini_api_key = "key"
        test_state.state.app_settings.gemini_model = "gemini-2.5-flash"
        test_state.http.queue(
            "get",
            FakeResponse(
                status_code=200,
                json_payload={
                    "models": [
                        _gemini_listed_model("models/gemini-2.5-flash", display_name="Gemini 2.5 Flash"),
                        _gemini_listed_model("models/gemini-2.5-flash-lite"),
                    ]
                },
            ),
        )

        r = client.get("/api/settings/gemini-models")
        assert r.status_code == 200
        data = r.json()
        assert data["resolvedModel"] == "gemini-2.5-flash"
        assert [model["id"] for model in data["models"]] == [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ]

    def test_stored_model_missing_from_list_is_still_included(self, client, test_state):
        test_state.state.app_settings.gemini_api_key = "key"
        test_state.state.app_settings.gemini_model = "gemini-custom-id"
        test_state.http.queue(
            "get",
            FakeResponse(
                status_code=200,
                json_payload={"models": [_gemini_listed_model("models/gemini-2.5-flash-lite")]},
            ),
        )

        r = client.get("/api/settings/gemini-models")
        assert r.status_code == 200
        data = r.json()
        assert data["resolvedModel"] == "gemini-custom-id"
        ids = [model["id"] for model in data["models"]]
        assert "gemini-custom-id" in ids
        assert "gemini-2.5-flash-lite" in ids

    def test_stored_non_text_model_is_not_re_injected(self, client, test_state):
        test_state.state.app_settings.gemini_api_key = "key"
        for stored in ("nano-banana-pro", "gemma-4-31b-it"):
            clear_gemini_models_cache()
            test_state.state.app_settings.gemini_model = stored
            test_state.http.queue(
                "get",
                FakeResponse(
                    status_code=200,
                    json_payload={"models": [_gemini_listed_model("models/gemini-2.5-flash-lite")]},
                ),
            )

            r = client.get("/api/settings/gemini-models")
            assert r.status_code == 200
            data = r.json()
            assert data["resolvedModel"] == DEFAULT_GEMINI_MODEL
            ids = [model["id"] for model in data["models"]]
            assert stored not in ids
            assert DEFAULT_GEMINI_MODEL in ids

    def test_default_model_included_when_missing_from_upstream_list(self, client, test_state):
        test_state.state.app_settings.gemini_api_key = "key"
        test_state.http.queue(
            "get",
            FakeResponse(
                status_code=200,
                json_payload={"models": [_gemini_listed_model("models/gemini-2.5-flash")]},
            ),
        )

        r = client.get("/api/settings/gemini-models")
        assert r.status_code == 200
        data = r.json()
        assert data["resolvedModel"] == DEFAULT_GEMINI_MODEL
        assert DEFAULT_GEMINI_MODEL in [model["id"] for model in data["models"]]

    def test_paginates_until_next_page_token_is_empty(self, client, test_state):
        test_state.state.app_settings.gemini_api_key = "key"
        test_state.http.queue(
            "get",
            FakeResponse(
                status_code=200,
                json_payload={
                    "models": [_gemini_listed_model("models/gemini-2.5-flash")],
                    "nextPageToken": "page-2",
                },
            ),
            FakeResponse(
                status_code=200,
                json_payload={"models": [_gemini_listed_model("models/gemini-2.5-flash-lite")]},
            ),
        )

        r = client.get("/api/settings/gemini-models")
        assert r.status_code == 200
        ids = [model["id"] for model in r.json()["models"]]
        assert ids == ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
        assert len(test_state.http.calls) == 2
        assert "pageToken=page-2" in test_state.http.calls[1].url

    def test_upstream_error_maps_status(self, client, test_state):
        test_state.state.app_settings.gemini_api_key = "key"
        test_state.http.queue("get", FakeResponse(status_code=403, text="forbidden"))

        r = client.get("/api/settings/gemini-models")
        assert r.status_code == 403

    def test_timeout_504(self, client, test_state):
        test_state.state.app_settings.gemini_api_key = "key"
        test_state.http.queue("get", HttpTransportError("timeout"))

        r = client.get("/api/settings/gemini-models")
        assert r.status_code == 504

    def test_successful_list_is_cached_per_api_key(self, client, test_state):
        test_state.state.app_settings.gemini_api_key = "key"
        test_state.http.queue(
            "get",
            FakeResponse(
                status_code=200,
                json_payload={"models": [_gemini_listed_model("models/gemini-2.5-flash-lite")]},
            ),
        )

        first = client.get("/api/settings/gemini-models")
        second = client.get("/api/settings/gemini-models")
        assert first.status_code == 200
        assert second.status_code == 200
        assert first.json()["models"] == second.json()["models"]
        assert len(test_state.http.calls) == 1

    def test_cache_misses_when_api_key_changes(self, client, test_state):
        test_state.state.app_settings.gemini_api_key = "key-a"
        test_state.http.queue(
            "get",
            FakeResponse(
                status_code=200,
                json_payload={"models": [_gemini_listed_model("models/gemini-2.5-flash")]},
            ),
            FakeResponse(
                status_code=200,
                json_payload={"models": [_gemini_listed_model("models/gemini-2.5-flash-lite")]},
            ),
        )

        first = client.get("/api/settings/gemini-models")
        test_state.state.app_settings.gemini_api_key = "key-b"
        second = client.get("/api/settings/gemini-models")
        assert first.status_code == 200
        assert second.status_code == 200
        assert [model["id"] for model in first.json()["models"]] == [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ]
        assert [model["id"] for model in second.json()["models"]] == ["gemini-2.5-flash-lite"]
        assert len(test_state.http.calls) == 2

    def test_included_missing_id_is_not_written_into_the_cache(self, client, test_state):
        test_state.state.app_settings.gemini_api_key = "key"
        test_state.state.app_settings.gemini_model = "gemini-custom-id"
        test_state.http.queue(
            "get",
            FakeResponse(
                status_code=200,
                json_payload={"models": [_gemini_listed_model("models/gemini-2.5-flash-lite")]},
            ),
        )

        first = client.get("/api/settings/gemini-models")
        test_state.state.app_settings.gemini_model = ""
        second = client.get("/api/settings/gemini-models")
        assert "gemini-custom-id" in [model["id"] for model in first.json()["models"]]
        assert "gemini-custom-id" not in [model["id"] for model in second.json()["models"]]
        assert len(test_state.http.calls) == 1

    def test_errors_are_not_cached(self, client, test_state):
        test_state.state.app_settings.gemini_api_key = "key"
        test_state.http.queue(
            "get",
            FakeResponse(status_code=403, text="forbidden"),
            FakeResponse(
                status_code=200,
                json_payload={"models": [_gemini_listed_model("models/gemini-2.5-flash-lite")]},
            ),
        )

        assert client.get("/api/settings/gemini-models").status_code == 403
        ok = client.get("/api/settings/gemini-models")
        assert ok.status_code == 200
        assert [model["id"] for model in ok.json()["models"]] == ["gemini-2.5-flash-lite"]
        assert len(test_state.http.calls) == 2
