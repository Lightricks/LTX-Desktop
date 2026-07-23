"""Integration tests for POST /api/enhance-prompt (Starlette TestClient + Fake pipeline)."""

from __future__ import annotations

from api_types import (
    IcLoraCatalogItem,
    InputSpec,
    InstructionSection,
    LoraCatalogItem,
    PromptTemplatePlaceholder,
    PromptTemplateSpec,
)
from tests.fakes import FakeResponse
from tests.http_error_assertions import assert_http_error


def _gemini_ok(text: str = "enhanced via gemini") -> FakeResponse:
    return FakeResponse(
        status_code=200,
        json_payload={"candidates": [{"content": {"parts": [{"text": text}]}}]},
    )


def _gemini_error(status: int = 429, body: str = "rate limited") -> FakeResponse:
    return FakeResponse(status_code=status, text=body)


def _dl(filename: str = "x.safetensors"):
    from api_types import DownloadSpec, DownloadVariant

    return DownloadSpec(
        repo_id="org/x",
        variants=[DownloadVariant(id="default", label="Default", filename=filename, size_bytes=10)],
    )


def _add_lora(fake_services, **overrides: object) -> LoraCatalogItem:
    defaults: dict[str, object] = dict(
        id="test-lora", name="Test Lora", description="d", download=_dl(), requires_hf_login=False,
    )
    defaults.update(overrides)
    lora = LoraCatalogItem(**defaults)
    fake_services.lora_catalog_provider._catalog.loras.append(lora)
    return lora


def _add_ic_lora(fake_services, **overrides: object) -> IcLoraCatalogItem:
    defaults: dict[str, object] = dict(
        id="test-ic-lora", name="Test IC-LoRA", description="d", download=_dl(), requires_hf_login=False,
        input=InputSpec(kind="video"),
    )
    defaults.update(overrides)
    ic_lora = IcLoraCatalogItem(**defaults)
    fake_services.lora_catalog_provider._catalog.ic_loras.append(ic_lora)
    return ic_lora


class TestNoSelection:
    def test_generic_fallback_uses_no_system_prompt(self, client, fake_services, create_fake_model_files):
        create_fake_model_files()
        r = client.post("/api/enhance-prompt", json={"prompt": "a cat"})
        assert r.status_code == 200
        assert r.json()["enhancedPrompt"] == fake_services.prompt_enhancer_pipeline.enhanced_prompt
        call = fake_services.prompt_enhancer_pipeline.enhance_t2v_calls[0]
        assert call["prompt"] == "a cat"
        assert call["system_prompt"] is None

    def test_image_path_routes_to_enhance_i2v(
        self, client, fake_services, create_fake_model_files, make_test_image, tmp_path
    ):
        create_fake_model_files()
        image_path = tmp_path / "cat.png"
        image_path.write_bytes(make_test_image().getvalue())

        r = client.post("/api/enhance-prompt", json={"prompt": "a cat", "imagePath": str(image_path)})
        assert r.status_code == 200
        assert len(fake_services.prompt_enhancer_pipeline.enhance_i2v_calls) == 1
        assert fake_services.prompt_enhancer_pipeline.enhance_i2v_calls[0]["image_path"] == str(image_path)
        assert len(fake_services.prompt_enhancer_pipeline.enhance_t2v_calls) == 0

    def test_invalid_image_path_rejected(self, client, create_fake_model_files):
        create_fake_model_files()
        r = client.post("/api/enhance-prompt", json={"prompt": "a cat", "imagePath": "/tmp/does-not-exist.png"})
        assert r.status_code == 400

    def test_seed_is_independent_of_dev_mode_lock(self, client, test_state, fake_services, create_fake_model_files):
        # Regression: enhance used StateHandlerBase._resolve_seed(), which returns a fixed
        # constant (1000) whenever dev mode is on — every call, including a "redo", would then
        # produce the exact same output. Two consecutive calls must not collapse to that.
        create_fake_model_files()
        test_state.config.dev_mode = True
        client.post("/api/enhance-prompt", json={"prompt": "a cat"})
        client.post("/api/enhance-prompt", json={"prompt": "a cat"})
        calls = fake_services.prompt_enhancer_pipeline.enhance_t2v_calls
        assert len(calls) == 2
        assert calls[0]["seed"] != 1000
        assert calls[0]["seed"] != calls[1]["seed"]


class TestLoraSelection:
    def test_single_lora_system_prompt_and_trigger_enforced(self, client, fake_services, create_fake_model_files):
        create_fake_model_files()
        _add_lora(
            fake_services, id="cozy-felt", name="Cozy Felt", trigger="F3ltCut0u7", trigger_placement="anywhere",
            instructions=[InstructionSection(kind="summary", title="What it does", body="Felt look.")],
        )
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = "a felt fox in a garden"

        r = client.post("/api/enhance-prompt", json={"prompt": "a fox", "loraCatalogIds": ["cozy-felt"]})
        assert r.status_code == 200
        assert r.json()["enhancedPrompt"] == "a felt fox in a garden F3ltCut0u7"

        call = fake_services.prompt_enhancer_pipeline.enhance_t2v_calls[0]
        assert "Cozy Felt" in call["system_prompt"]
        assert "Felt look." in call["system_prompt"]

    def test_multi_lora_prompt_includes_both(self, client, fake_services, create_fake_model_files):
        create_fake_model_files()
        _add_lora(fake_services, id="a", name="Alpha")
        _add_lora(fake_services, id="b", name="Beta")

        r = client.post("/api/enhance-prompt", json={"prompt": "x", "loraCatalogIds": ["a", "b"]})
        assert r.status_code == 200
        call = fake_services.prompt_enhancer_pipeline.enhance_t2v_calls[0]
        assert "Alpha" in call["system_prompt"] and "Beta" in call["system_prompt"]

    def test_unknown_lora_id_rejected(self, client, create_fake_model_files):
        create_fake_model_files()
        r = client.post("/api/enhance-prompt", json={"prompt": "x", "loraCatalogIds": ["does-not-exist"]})
        assert_http_error(r, status_code=404, code="LORA_CATALOG_ID_NOT_FOUND")


class TestIcLoraSelection:
    def test_ic_lora_without_template_free_rewrite_and_trigger_enforced(
        self, client, fake_services, create_fake_model_files
    ):
        create_fake_model_files()
        _add_ic_lora(
            fake_services, id="day-to-night", name="Day to Night",
            instructions=[InstructionSection(kind="summary", title="What it does", body="Relights to night.")],
        )
        r = client.post("/api/enhance-prompt", json={"prompt": "a street", "icLoraId": "day-to-night"})
        assert r.status_code == 200
        call = fake_services.prompt_enhancer_pipeline.enhance_t2v_calls[0]
        assert "Day to Night" in call["system_prompt"]

    def test_unknown_ic_lora_id_rejected(self, client, create_fake_model_files):
        create_fake_model_files()
        r = client.post("/api/enhance-prompt", json={"prompt": "x", "icLoraId": "does-not-exist"})
        assert_http_error(r, status_code=404, code="LORA_CATALOG_ID_NOT_FOUND")

    def test_free_text_template_fill_stitches_deterministically(self, client, fake_services, create_fake_model_files):
        create_fake_model_files()
        _add_ic_lora(
            fake_services, id="colorization", name="Colorization",
            prompt_template=PromptTemplateSpec(
                template="Reference shows {reference}. COLORIZE {result}.",
                placeholders={"reference": PromptTemplatePlaceholder(), "result": PromptTemplatePlaceholder()},
            ),
        )
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = (
            '{"reference": "a grey rabbit", "result": "a brown rabbit"}'
        )

        r = client.post("/api/enhance-prompt", json={"prompt": "colorize the rabbit", "icLoraId": "colorization"})
        assert r.status_code == 200
        assert r.json()["enhancedPrompt"] == "Reference shows a grey rabbit. COLORIZE a brown rabbit."

    def test_enum_template_fill_stitches_deterministically(self, client, fake_services, create_fake_model_files):
        create_fake_model_files()
        _add_ic_lora(
            fake_services, id="crossview-prompt", name="CrossView",
            prompt_template=PromptTemplateSpec(
                template="crossview. new camera angle: {azimuth}, {elevation}, {distance}.",
                placeholders={
                    "azimuth": PromptTemplatePlaceholder(choices=["to the left", "to the right"]),
                    "elevation": PromptTemplatePlaceholder(choices=["lower", "higher"]),
                    "distance": PromptTemplatePlaceholder(choices=["closer", "further"]),
                },
            ),
        )
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = (
            '{"azimuth": "to the right", "elevation": "lower", "distance": "closer"}'
        )

        r = client.post("/api/enhance-prompt", json={"prompt": "move right and closer", "icLoraId": "crossview-prompt"})
        assert r.status_code == 200
        assert r.json()["enhancedPrompt"] == "crossview. new camera angle: to the right, lower, closer."

    def test_template_fill_rejects_invalid_enum_choice(self, client, fake_services, create_fake_model_files):
        create_fake_model_files()
        _add_ic_lora(
            fake_services, id="crossview-prompt", name="CrossView",
            prompt_template=PromptTemplateSpec(
                template="crossview. new camera angle: {azimuth}.",
                placeholders={"azimuth": PromptTemplatePlaceholder(choices=["to the left", "to the right"])},
            ),
        )
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = '{"azimuth": "sideways"}'

        r = client.post("/api/enhance-prompt", json={"prompt": "x", "icLoraId": "crossview-prompt"})
        assert r.status_code == 500

    def test_template_fill_rejects_non_json_response(self, client, fake_services, create_fake_model_files):
        create_fake_model_files()
        _add_ic_lora(
            fake_services, id="upscale", name="Upscale",
            prompt_template=PromptTemplateSpec(template="upscale", placeholders={}),
        )
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = "not json"

        r = client.post("/api/enhance-prompt", json={"prompt": "x", "icLoraId": "upscale"})
        assert r.status_code == 500


class TestConditioningType:
    # canny/depth: the built-in "bring your own IC-LoRA" conditioning modes, no catalog entry.
    def test_depth_uses_dedicated_system_prompt(self, client, fake_services, create_fake_model_files):
        create_fake_model_files()
        r = client.post(
            "/api/enhance-prompt",
            json={"prompt": "a fox riding a skateboard", "conditioningType": "depth"},
        )
        assert r.status_code == 200
        call = fake_services.prompt_enhancer_pipeline.enhance_t2v_calls[0]
        assert call["system_prompt"] is not None
        assert "depth" in call["system_prompt"].lower()
        assert "faithfully" in call["system_prompt"].lower()

    def test_canny_uses_dedicated_system_prompt(self, client, fake_services, create_fake_model_files):
        create_fake_model_files()
        r = client.post(
            "/api/enhance-prompt",
            json={"prompt": "a fox riding a skateboard", "conditioningType": "canny"},
        )
        assert r.status_code == 200
        call = fake_services.prompt_enhancer_pipeline.enhance_t2v_calls[0]
        assert call["system_prompt"] is not None
        assert "edge" in call["system_prompt"].lower()


class TestRequestValidation:
    def test_lora_ids_and_ic_lora_id_mutually_exclusive(self, client, create_fake_model_files):
        create_fake_model_files()
        r = client.post(
            "/api/enhance-prompt",
            json={"prompt": "x", "loraCatalogIds": ["a"], "icLoraId": "b"},
        )
        assert r.status_code == 422

    def test_conditioning_type_and_ic_lora_id_mutually_exclusive(self, client, create_fake_model_files):
        create_fake_model_files()
        r = client.post(
            "/api/enhance-prompt",
            json={"prompt": "x", "conditioningType": "depth", "icLoraId": "b"},
        )
        assert r.status_code == 422


class TestGating:
    def test_local_enhance_works_even_when_generation_prefers_api_text_encoding(
        self, client, test_state, create_fake_model_files
    ):
        # Regression: PromptEnhancementHandler used to call resolve_gemma_root(), which returns
        # None whenever should_use_local_encoding()'s API-key tiebreaker picks API — a GENERATION
        # policy question, unrelated to whether local Enhance can run. A user with both an LTX
        # API key and the checkpoint downloaded (the tiebreaker defaults to API) could previously
        # never use "Local" Enhance, even though the frontend's own checkpoint-presence check
        # offered it.
        create_fake_model_files()
        test_state.state.app_settings.ltx_api_key = "ltx-key"
        assert test_state.text.should_use_local_encoding() is False  # tiebreaker picks API

        r = client.post("/api/enhance-prompt", json={"prompt": "x", "provider": "local"})
        assert r.status_code == 200

    def test_missing_local_gemma_rejected(self, client):
        # create_fake_model_files() not called -> resolve_gemma_root_if_downloaded() is None.
        r = client.post("/api/enhance-prompt", json={"prompt": "x"})
        assert_http_error(r, status_code=409, code="LOCAL_TEXT_ENCODER_NOT_AVAILABLE")

    def test_rejected_while_generation_running(self, client, test_state, create_fake_model_files):
        create_fake_model_files()
        test_state.pipelines.load_gpu_pipeline("fast")
        test_state.generation.start_generation("gen-1")

        r = client.post("/api/enhance-prompt", json={"prompt": "x"})
        assert r.status_code == 409

    def test_enhance_failure_returns_500(self, client, fake_services, create_fake_model_files):
        create_fake_model_files()
        fake_services.prompt_enhancer_pipeline.raise_on_enhance = RuntimeError("boom")
        r = client.post("/api/enhance-prompt", json={"prompt": "x"})
        assert r.status_code == 500

    def test_enhance_marks_itself_busy_then_frees_the_slot(self, client, test_state, create_fake_model_files):
        # Regression: enhance() must participate in the same generation-mutex bookkeeping every
        # other handler uses, so an orphaned enhance can't race a Generate click — and must
        # release it again on success so a later call isn't blocked forever.
        create_fake_model_files()
        assert test_state.generation.is_generation_running() is False
        r = client.post("/api/enhance-prompt", json={"prompt": "x"})
        assert r.status_code == 200
        assert test_state.generation.is_generation_running() is False

    def test_enhance_frees_the_slot_after_failure(self, client, fake_services, test_state, create_fake_model_files):
        create_fake_model_files()
        fake_services.prompt_enhancer_pipeline.raise_on_enhance = RuntimeError("boom")
        client.post("/api/enhance-prompt", json={"prompt": "x"})
        assert test_state.generation.is_generation_running() is False

        fake_services.prompt_enhancer_pipeline.raise_on_enhance = None
        r = client.post("/api/enhance-prompt", json={"prompt": "x"})
        assert r.status_code == 200


class TestApiProvider:
    # provider="api" never touches the local Gemma pipeline — no create_fake_model_files() call
    # anywhere in this class, proving the local-checkpoint requirement is fully bypassed.
    def test_free_rewrite_calls_gemini_without_local_gemma(self, client, test_state):
        test_state.state.app_settings.gemini_api_key = "key"
        test_state.http.queue("post", _gemini_ok("a cat, enhanced"))

        r = client.post("/api/enhance-prompt", json={"prompt": "a cat", "provider": "api"})
        assert r.status_code == 200
        assert r.json()["enhancedPrompt"] == "a cat, enhanced"

    def test_no_selection_still_gets_a_system_instruction(self, client, test_state):
        # Regression: Gemini (unlike local Gemma) has no implicit default system prompt of its
        # own — omitting systemInstruction entirely produced a chatty markdown essay instead of
        # a rewritten prompt on real hardware.
        test_state.state.app_settings.gemini_api_key = "key"
        test_state.http.queue("post", _gemini_ok())

        r = client.post("/api/enhance-prompt", json={"prompt": "a cat", "provider": "api"})
        assert r.status_code == 200
        payload = test_state.http.calls[-1].json_payload
        system_instruction = payload["systemInstruction"]["parts"][0]["text"]
        assert "ONLY the rewritten prompt" in system_instruction

    def test_image_path_sends_inline_image_data(self, client, test_state, make_test_image, tmp_path):
        test_state.state.app_settings.gemini_api_key = "key"
        test_state.http.queue("post", _gemini_ok())
        image_path = tmp_path / "cat.png"
        image_path.write_bytes(make_test_image().getvalue())

        r = client.post(
            "/api/enhance-prompt",
            json={"prompt": "a cat", "provider": "api", "imagePath": str(image_path)},
        )
        assert r.status_code == 200
        parts = test_state.http.calls[-1].json_payload["contents"][0]["parts"]
        inline_parts = [p for p in parts if "inlineData" in p]
        assert len(inline_parts) == 1
        # Regression: mimeType was hardcoded to image/jpeg regardless of the real file format —
        # make_test_image() writes a real PNG, so this fails loudly on the old hardcoded value.
        assert inline_parts[0]["inlineData"]["mimeType"] == "image/png"

    def test_gif_rejected_before_reaching_gemini(self, client, test_state, tmp_path):
        # Our own validate_image_file() allows GIF (for the local provider's benefit), but
        # Gemini's inlineData doesn't support it — this should fail fast with a clear message
        # rather than bouncing off Gemini as an opaque upstream error.
        from PIL import Image

        test_state.state.app_settings.gemini_api_key = "key"
        image_path = tmp_path / "cat.gif"
        Image.new("RGB", (8, 8), "red").save(image_path, format="GIF")

        r = client.post(
            "/api/enhance-prompt",
            json={"prompt": "a cat", "provider": "api", "imagePath": str(image_path)},
        )
        assert_http_error(
            r, status_code=400, code="GEMINI_UNSUPPORTED_IMAGE_FORMAT",
            message="Image format GIF isn't supported by the Gemini API provider "
                    "(use Local, or convert the image to PNG/JPEG/WEBP)",
        )
        assert len(test_state.http.calls) == 0

    def test_catalog_lora_system_prompt_reused_for_api_provider(self, client, fake_services, test_state):
        test_state.state.app_settings.gemini_api_key = "key"
        _add_lora(
            fake_services, id="cozy-felt", name="Cozy Felt", trigger="F3ltCut0u7", trigger_placement="anywhere",
        )
        test_state.http.queue("post", _gemini_ok("a felt fox"))

        r = client.post(
            "/api/enhance-prompt",
            json={"prompt": "a fox", "provider": "api", "loraCatalogIds": ["cozy-felt"]},
        )
        assert r.status_code == 200
        assert r.json()["enhancedPrompt"] == "a felt fox F3ltCut0u7"
        system_instruction = test_state.http.calls[-1].json_payload["systemInstruction"]["parts"][0]["text"]
        assert "Cozy Felt" in system_instruction

    def test_missing_gemini_key_rejected(self, client):
        r = client.post("/api/enhance-prompt", json={"prompt": "x", "provider": "api"})
        assert_http_error(r, status_code=400, code="GEMINI_API_KEY_MISSING")

    def test_upstream_error_propagates(self, client, test_state):
        test_state.state.app_settings.gemini_api_key = "key"
        test_state.http.queue("post", _gemini_error(429, "rate limited"))

        r = client.post("/api/enhance-prompt", json={"prompt": "x", "provider": "api"})
        assert r.status_code == 429

    def test_empty_response_rejected_instead_of_wiping_the_prompt(self, client, test_state):
        # Regression: whitespace-only text passes the strict schema (it's a valid, non-blocked
        # candidate) — but handing it back as enhancedPrompt would silently erase the user's
        # prompt on what looks like a "success" response.
        test_state.state.app_settings.gemini_api_key = "key"
        test_state.http.queue("post", _gemini_ok("   "))

        r = client.post("/api/enhance-prompt", json={"prompt": "x", "provider": "api"})
        assert_http_error(r, status_code=500, code="GEMINI_EMPTY_RESPONSE",
                           message="Gemini returned an empty response")

    def test_blocked_prompt_returns_422(self, client, test_state):
        # A prompt rejected before any candidate is generated: HTTP 200, empty candidates,
        # promptFeedback.blockReason set. Previously fell through to a strict-schema
        # ValidationError and surfaced as an opaque 500.
        test_state.state.app_settings.gemini_api_key = "key"
        test_state.http.queue("post", FakeResponse(
            status_code=200,
            json_payload={"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}},
        ))

        r = client.post("/api/enhance-prompt", json={"prompt": "x", "provider": "api"})
        assert_http_error(r, status_code=422, code="GEMINI_CONTENT_BLOCKED",
                           message="Prompt rejected by Gemini safety filters (SAFETY)")

    def test_blocked_completion_returns_422(self, client, test_state):
        # A candidate generated then withheld: finishReason set, no `content` key at all.
        test_state.state.app_settings.gemini_api_key = "key"
        test_state.http.queue("post", FakeResponse(
            status_code=200,
            json_payload={"candidates": [{"finishReason": "SAFETY", "safetyRatings": []}]},
        ))

        r = client.post("/api/enhance-prompt", json={"prompt": "x", "provider": "api"})
        assert_http_error(r, status_code=422, code="GEMINI_CONTENT_BLOCKED",
                           message="Response withheld by Gemini safety filters (SAFETY)")

    def test_local_provider_still_requires_gemma_even_with_gemini_key_set(self, client, test_state):
        # provider defaults to "local" — a configured Gemini key must not bypass the local gate.
        test_state.state.app_settings.gemini_api_key = "key"
        r = client.post("/api/enhance-prompt", json={"prompt": "x"})
        assert_http_error(r, status_code=409, code="LOCAL_TEXT_ENCODER_NOT_AVAILABLE")


class TestImageMediaType:
    def test_generation_uses_image_generation_system_prompt(
        self, client, fake_services, create_fake_model_files
    ):
        create_fake_model_files()
        r = client.post("/api/enhance-prompt", json={"prompt": "a red car", "mediaType": "image"})
        assert r.status_code == 200
        call = fake_services.prompt_enhancer_pipeline.enhance_t2v_calls[0]
        assert "Z-Image-Turbo" in call["system_prompt"]
        assert "text-to-image" in call["system_prompt"]

    def test_editing_uses_image_edit_system_prompt_and_routes_to_i2v(
        self, client, fake_services, create_fake_model_files, make_test_image, tmp_path
    ):
        create_fake_model_files()
        image_path = tmp_path / "src.png"
        image_path.write_bytes(make_test_image().getvalue())

        r = client.post(
            "/api/enhance-prompt",
            json={"prompt": "make the sky purple", "mediaType": "image", "imagePath": str(image_path)},
        )
        assert r.status_code == 200
        assert len(fake_services.prompt_enhancer_pipeline.enhance_i2v_calls) == 1
        call = fake_services.prompt_enhancer_pipeline.enhance_i2v_calls[0]
        assert "image-to-image editing" in call["system_prompt"]
        assert "DESIRED RESULT" in call["system_prompt"]

    def test_image_media_type_rejects_lora_selection(self, client, create_fake_model_files):
        create_fake_model_files()
        r = client.post(
            "/api/enhance-prompt",
            json={"prompt": "x", "mediaType": "image", "loraCatalogIds": ["a"]},
        )
        assert r.status_code == 422

    def test_image_generation_works_via_api_provider_too(self, client, test_state):
        test_state.state.app_settings.gemini_api_key = "key"
        test_state.http.queue("post", _gemini_ok("a shiny red sedan"))

        r = client.post(
            "/api/enhance-prompt",
            json={"prompt": "a red car", "mediaType": "image", "provider": "api"},
        )
        assert r.status_code == 200
        assert r.json()["enhancedPrompt"] == "a shiny red sedan"
        system_instruction = test_state.http.calls[-1].json_payload["systemInstruction"]["parts"][0]["text"]
        assert "Z-Image-Turbo" in system_instruction
