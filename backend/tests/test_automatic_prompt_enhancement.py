"""Automatic, provenance-aware prompt enhancement for local video and IC-LoRA generation.

Local encoding rewrites on Gen Space (/api/generate) regardless of Explore auto-enhance.
Home/Explore (generate_local_reserved) additionally honors explore_auto_enhance_prompts.
API encoding still honors the T2V/I2V settings toggles for the server-side enhance_prompt
flag, except when Explore auto-enhance is off on the Home path. Home auto-enhance rewrites
locally first when the enhancer checkpoint is installed, even if text embeddings come from
the API; a style-recipe wrap then keeps the API flag off so the scaffold is not rewritten.
Recipe wraps are skipped for typed Explore jobs when auto-enhance is off so the pipeline
sees the prompt as typed.
"""

from __future__ import annotations

import logging
import threading
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from api_types import (
    GenerateVideoRequest,
    IcLoraCatalogItem,
    InputSpec,
    InstructionSection,
    PreprocessingStep,
    PromptTemplatePlaceholder,
    PromptTemplateSpec,
)
from runtime_config.model_download_specs import get_latest_ltx_model_id, get_ltx_model_spec
from services.features.lora_recipes import get_lora_recipe, make_recipe_prompt_wrap
from services.generation_interrupt import GenerationCancelledError
from tests.fakes import FakeCapture
from tests.lora_catalog_helpers import add_ic_lora, add_lora, download_spec

# 2.3's gemma3 text encoder doubles as the local prompt enhancer, so a plain bundle is enough to
# make automatic local enhancement runnable (2.5 needs the separate opt-in enhancer download).
_LOCAL_ENHANCER_MODEL_ID = "ltx-2.3-22b-distilled-1.1"

_T2V_JSON = {
    "prompt": "test",
    "resolution": "540p",
    "model": "fast",
    "duration": 5,
    "fps": 24,
}


@dataclass
class _FakeEncodingResult:
    video_context: object = "fake_tensor"
    audio_context: object = None


def _install_local_enhancer(test_state, create_fake_model_files) -> None:
    create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)
    test_state.state.app_settings.active_ltx_model_id = _LOCAL_ENHANCER_MODEL_ID
    test_state.state.app_settings.use_local_text_encoder = True
    # Local enhancement is automatic: persisted API opt-out must not suppress it.
    test_state.state.app_settings.prompt_enhancer_enabled = False


def _install_api_encoding(test_state, fake_services, create_fake_model_files) -> None:
    create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)
    test_state.state.app_settings.active_ltx_model_id = _LOCAL_ENHANCER_MODEL_ID
    test_state.state.app_settings.ltx_api_key = "test-key"
    test_state.state.app_settings.use_local_text_encoder = False
    fake_services.text_encoder.encode_responses.append(_FakeEncodingResult())


def _write_test_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(b"\x00\x00" * 800)


def _add_image_ic_lora(fake_services, **overrides: object) -> IcLoraCatalogItem:
    """An IC-LoRA driven by a still, which is what every entry built here generates from."""
    return add_ic_lora(
        fake_services,
        input=InputSpec(kind="image"),
        preprocessing=[PreprocessingStep(utility="image_to_frames", params={})],
        instructions=[InstructionSection(kind="summary", title="What it does", body="use it")],
        **overrides,
    )


def _download_ic_lora(client, ic_lora_id: str) -> None:
    assert client.post("/api/ic-loras/download", json={"ic_lora_id": ic_lora_id}).status_code == 200


def _write_png(path: Path, size: tuple[int, int] = (64, 64)) -> Path:
    Image.new("RGB", size, color=(1, 2, 3)).save(path, format="PNG")
    return path


def _register_input_video(test_state, name: str = "input.mp4") -> Path:
    video_path = test_state.config.outputs_dir / name
    video_path.write_bytes(b"\x00" * 100)
    test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))
    return video_path


def _register_outpaint_source(fake_services, tmp_path: Path) -> Path:
    """Real numpy frames so outpaint_canvas can composite onto the larger canvas."""
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00")
    fake_services.video_processor.videos[str(src)] = FakeCapture(
        frames=[np.full((96, 96, 3), 0x40, dtype=np.uint8) for _ in range(9)],
        width=96,
        height=96,
        fps=24,
    )
    return src


class TestLocalVideoAlwaysEnhances:
    def test_typed_t2v_prompt_is_enhanced_even_with_the_legacy_toggle_off(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        _install_local_enhancer(test_state, create_fake_model_files)
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = "a long descriptive caption"

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200

        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls[0]["prompt"] == "test"
        assert fake_services.fast_video_pipeline.generate_calls[0]["prompt"] == "a long descriptive caption"

    def test_typed_i2v_prompt_is_enhanced_even_with_the_legacy_toggle_off(
        self, client, test_state, fake_services, create_fake_model_files, make_test_image, tmp_path
    ):
        _install_local_enhancer(test_state, create_fake_model_files)
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = "a long descriptive caption"
        image_path = tmp_path / "input.png"
        image_path.write_bytes(make_test_image().getvalue())

        r = client.post("/api/generate", json={**_T2V_JSON, "imagePath": str(image_path)})
        assert r.status_code == 200

        assert len(fake_services.prompt_enhancer_pipeline.enhance_i2v_calls) == 1
        assert fake_services.fast_video_pipeline.generate_calls[0]["prompt"] == "a long descriptive caption"

    def test_typed_a2v_prompt_is_enhanced(
        self, client, test_state, fake_services, create_fake_model_files, tmp_path
    ):
        _install_local_enhancer(test_state, create_fake_model_files)
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = "a long descriptive caption"
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post("/api/generate", json={**_T2V_JSON, "audioPath": str(audio_file)})
        assert r.status_code == 200

        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls[0]["prompt"] == "test"
        assert fake_services.a2v_pipeline.generate_calls[0]["prompt"] == "a long descriptive caption"

    def test_typed_keyframe_prompt_is_enhanced(
        self, client, test_state, fake_services, create_fake_model_files, make_test_image, tmp_path
    ):
        _install_local_enhancer(test_state, create_fake_model_files)
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = "a long descriptive caption"
        opening = tmp_path / "opening.png"
        closing = tmp_path / "closing.png"
        for path in (opening, closing):
            path.write_bytes(make_test_image().getvalue())

        r = client.post(
            "/api/generate",
            json={
                **_T2V_JSON,
                "keyframes": [
                    {"imagePath": str(opening), "frameIndex": 0},
                    {"imagePath": str(closing), "frameIndex": 80},
                ],
            },
        )
        assert r.status_code == 200

        assert len(fake_services.prompt_enhancer_pipeline.enhance_i2v_calls) == 1
        assert fake_services.fast_video_pipeline.generate_calls[0]["prompt"] == "a long descriptive caption"

    def test_missing_local_enhancer_generates_with_the_prompt_as_typed(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        # 2.5 without its opt-in enhancer download and no 2.3 install to fall back to.
        create_fake_model_files(include_prompt_enhancer=False)
        test_state.state.app_settings.use_local_text_encoder = True

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200

        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls == []
        assert fake_services.fast_video_pipeline.generate_calls[0]["prompt"] == "test"

    def test_failing_local_enhancer_generates_with_the_prompt_as_typed(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        _install_local_enhancer(test_state, create_fake_model_files)
        fake_services.prompt_enhancer_pipeline.raise_on_enhance = RuntimeError("boom")

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200
        assert fake_services.fast_video_pipeline.generate_calls[0]["prompt"] == "test"

    def test_cancellation_from_local_enhancer_returns_cancelled_response(
        self, client, test_state, fake_services, create_fake_model_files, caplog
    ):
        _install_local_enhancer(test_state, create_fake_model_files)
        fake_services.prompt_enhancer_pipeline.raise_on_enhance = GenerationCancelledError()
        caplog.set_level(logging.WARNING, logger="handlers.prompt_enhancement_handler")

        response = client.post("/api/generate", json=_T2V_JSON)

        assert response.status_code == 200
        assert response.json() == {"status": "cancelled"}
        assert fake_services.fast_video_pipeline.generate_calls == []
        assert not any(
            record.getMessage().startswith("Automatic local enhancement failed")
            for record in caplog.records
        )

    def test_success_info_log_reports_lengths_without_prompt_content(
        self, client, test_state, fake_services, create_fake_model_files, caplog
    ):
        _install_local_enhancer(test_state, create_fake_model_files)
        prompt = "private typed prompt"
        enhanced = "private rewritten prompt"
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = enhanced
        caplog.set_level(logging.INFO, logger="handlers.prompt_enhancement_handler")

        r = client.post("/api/generate", json={**_T2V_JSON, "prompt": prompt})

        assert r.status_code == 200
        success_logs = [
            record.getMessage()
            for record in caplog.records
            if record.getMessage().startswith("Enhanced prompt locally for generation")
        ]
        assert success_logs == [
            f"Enhanced prompt locally for generation ({len(prompt)} -> {len(enhanced)} chars)"
        ]
        assert prompt not in success_logs[0]
        assert enhanced not in success_logs[0]


class TestIcLoraEnhancementCancellation:
    def test_catalog_enhancer_cancellation_returns_cancelled_response(
        self, client, test_state, fake_services, create_fake_model_files, tmp_path
    ):
        _install_local_enhancer(test_state, create_fake_model_files)
        fake_services.prompt_enhancer_pipeline.raise_on_enhance = GenerationCancelledError()
        _download_ic_lora(client, "ingredients-v1")
        image_path = _write_png(tmp_path / "in.png")

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "ic_lora_id": "ingredients-v1",
                "input_path": str(image_path),
                "control_values": {"duration": 5},
                "prompt": "a slow orbit",
                "conditioning_type": "custom",
            },
        )

        assert response.status_code == 200
        assert response.json() == {"status": "cancelled"}
        assert fake_services.ic_lora_pipeline.generate_calls == []

    def test_builtin_enhancer_cancellation_returns_cancelled_response(
        self,
        client,
        test_state,
        fake_services,
        create_fake_model_files,
        create_fake_ic_lora_files,
    ):
        _install_local_enhancer(test_state, create_fake_model_files)
        create_fake_ic_lora_files()
        fake_services.prompt_enhancer_pipeline.raise_on_enhance = GenerationCancelledError()
        video_path = _register_input_video(test_state)

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [],
            },
        )

        assert response.status_code == 200
        assert response.json() == {"status": "cancelled"}
        assert fake_services.ic_lora_pipeline.generate_calls == []


class TestRecommendedEnhancerProvisioning:
    """The recommended quality download is exactly what the automatic rewrite looks for.

    2.5's text encoder can only encode, so on 2.5 the automatic rewrite depends on the separately
    recommended enhancer checkpoint. Two contracts meet here — what the recommendation offers and
    what the generation path resolves — and nothing else asserts they name the same checkpoint.
    """

    def test_the_checkpoint_first_run_recommends_is_what_makes_generation_enhance_locally(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        spec = get_ltx_model_spec(get_latest_ltx_model_id())
        assert spec.prompt_enhancer_cp is not None

        first_run = client.get("/api/models/ltx-recommendation").json()
        assert first_run["status"] == "download"
        assert first_run["recommended_quality_cp_ids"] == [spec.prompt_enhancer_cp]

        # Accept the recommendation: the required bundle plus exactly that checkpoint.
        create_fake_model_files(include_prompt_enhancer=True)
        test_state.state.app_settings.use_local_text_encoder = True
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = "a long descriptive caption"

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200

        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls[0]["prompt"] == "test"
        assert fake_services.fast_video_pipeline.generate_calls[0]["prompt"] == "a long descriptive caption"

    def test_skipping_the_recommended_download_still_generates_and_stays_offerable(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        """Skip or download failure is non-blocking: it costs the rewrite, not the generation."""
        spec = get_ltx_model_spec(get_latest_ltx_model_id())
        create_fake_model_files(include_prompt_enhancer=False)
        test_state.state.app_settings.use_local_text_encoder = True

        # The required bundle is complete, so the skipped enhancer must not re-open the gate,
        # while Settings' own recommendation still offers it as the retry path.
        assert client.get("/api/models/ltx-recommendation").json()["status"] == "ok"
        settings_offer = client.get("/api/models/text-encoder-recommendation").json()
        assert settings_offer["local_enhancer_cp"] == spec.prompt_enhancer_cp
        assert settings_offer["local_enhancement_supported"] is False

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200

        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls == []
        assert fake_services.fast_video_pipeline.generate_calls[0]["prompt"] == "test"


class TestPromptProvenance:
    def test_enhanced_provenance_is_generated_exactly_as_submitted(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        _install_local_enhancer(test_state, create_fake_model_files)

        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "prompt": "already rewritten", "promptProvenance": "enhanced"},
        )
        assert r.status_code == 200

        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls == []
        assert fake_services.prompt_enhancer_pipeline.enhance_i2v_calls == []
        assert fake_services.fast_video_pipeline.generate_calls[0]["prompt"] == "already rewritten"

    def test_typed_provenance_is_the_default_for_clients_that_omit_it(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        _install_local_enhancer(test_state, create_fake_model_files)
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = "a long descriptive caption"

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200
        assert fake_services.fast_video_pipeline.generate_calls[0]["prompt"] == "a long descriptive caption"

    def test_enhanced_provenance_leaves_the_a2v_prompt_alone(
        self, client, test_state, fake_services, create_fake_model_files, tmp_path
    ):
        _install_local_enhancer(test_state, create_fake_model_files)
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                **_T2V_JSON,
                "prompt": "already rewritten",
                "promptProvenance": "enhanced",
                "audioPath": str(audio_file),
            },
        )
        assert r.status_code == 200
        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls == []
        assert fake_services.a2v_pipeline.generate_calls[0]["prompt"] == "already rewritten"


def _cozy_felt_wrap():
    recipe = get_lora_recipe("cozy-felt")
    assert recipe is not None
    return make_recipe_prompt_wrap(recipe)


def _generate_explore(test_state, json_body: dict | None = None, *, prompt_wrap=None) -> None:
    shutdown = threading.Event()
    with test_state.generation.wait_for_generation_slot(shutdown):
        test_state.video_generation.generate_local_reserved(
            GenerateVideoRequest.model_validate(json_body or _T2V_JSON),
            generation_id="explore-1",
            local_model_id=_LOCAL_ENHANCER_MODEL_ID,
            prompt_wrap=prompt_wrap,
        )


class TestExploreAutoEnhance:
    def test_auto_off_leaves_home_prompt_as_typed(
        self, test_state, fake_services, create_fake_model_files
    ):
        _install_local_enhancer(test_state, create_fake_model_files)
        test_state.state.app_settings.explore_auto_enhance_prompts = False
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = "a long descriptive caption"

        _generate_explore(test_state)

        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls == []
        assert fake_services.fast_video_pipeline.generate_calls[0]["prompt"] == "test"

    def test_resolve_skip_recipe_wrap_only_for_auto_off_typed_home(
        self, test_state
    ):
        test_state.state.app_settings.explore_auto_enhance_prompts = False
        typed = test_state.prompt_enhancement.resolve_for_generation(
            "mouse in a house",
            provenance="typed",
            generation_seed=1,
            explicit_generation_seed=True,
            explore_generation=True,
        )
        enhanced = test_state.prompt_enhancement.resolve_for_generation(
            "a rewritten scene",
            provenance="enhanced",
            generation_seed=1,
            explicit_generation_seed=True,
            explore_generation=True,
        )
        gen_space = test_state.prompt_enhancement.resolve_for_generation(
            "mouse in a house",
            provenance="typed",
            generation_seed=1,
            explicit_generation_seed=True,
            explore_generation=False,
        )
        assert typed.skip_recipe_wrap is True
        assert enhanced.skip_recipe_wrap is False
        assert gen_space.skip_recipe_wrap is False

    def test_auto_off_does_not_skip_gen_space_rewrite(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        _install_local_enhancer(test_state, create_fake_model_files)
        test_state.state.app_settings.explore_auto_enhance_prompts = False
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = "a long descriptive caption"

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200

        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls[0]["prompt"] == "test"
        assert fake_services.fast_video_pipeline.generate_calls[0]["prompt"] == "a long descriptive caption"

    def test_auto_on_still_rewrites_home(
        self, test_state, fake_services, create_fake_model_files
    ):
        _install_local_enhancer(test_state, create_fake_model_files)
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = "a long descriptive caption"

        _generate_explore(test_state)

        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls[0]["prompt"] == "test"
        assert fake_services.fast_video_pipeline.generate_calls[0]["prompt"] == "a long descriptive caption"

    def test_auto_off_drops_home_api_enhance_flag(
        self, test_state, fake_services, create_fake_model_files
    ):
        _install_api_encoding(test_state, fake_services, create_fake_model_files)
        test_state.state.app_settings.explore_auto_enhance_prompts = False

        _generate_explore(test_state)

        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is False
        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls == []

    def test_auto_off_does_not_drop_gen_space_api_enhance_flag(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        _install_api_encoding(test_state, fake_services, create_fake_model_files)
        test_state.state.app_settings.explore_auto_enhance_prompts = False

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200

        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is True

    def test_auto_off_leaves_typed_recipe_prompt_unwrapped(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        # Cozy Felt's scaffold is a prompt mutation. Auto-off + a typed scene must
        # reach the pipeline as typed — the LLM rewrite *and* the felt wrap stay off.
        _install_local_enhancer(test_state, create_fake_model_files)
        r = client.post("/api/settings", json={"exploreAutoEnhancePrompts": False})
        assert r.status_code == 200
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = (
            "a rewritten felt palace with slow camera moves"
        )
        wrap = _cozy_felt_wrap()
        scene = "mouse in a house"

        _generate_explore(
            test_state,
            {**_T2V_JSON, "prompt": scene},
            prompt_wrap=wrap,
        )

        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls == []
        assert fake_services.fast_video_pipeline.generate_calls[0]["prompt"] == scene

    def test_auto_off_still_wraps_a_manually_enhanced_recipe_prompt(
        self, test_state, fake_services, create_fake_model_files
    ):
        _install_local_enhancer(test_state, create_fake_model_files)
        test_state.state.app_settings.explore_auto_enhance_prompts = False
        wrap = _cozy_felt_wrap()
        rewritten = "a rewritten felt palace with slow camera moves"

        _generate_explore(
            test_state,
            {**_T2V_JSON, "prompt": rewritten, "promptProvenance": "enhanced"},
            prompt_wrap=wrap,
        )

        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls == []
        assert fake_services.fast_video_pipeline.generate_calls[0]["prompt"] == wrap(rewritten)

    def test_auto_on_rewrites_then_wraps_recipe_scene(
        self, test_state, fake_services, create_fake_model_files
    ):
        _install_local_enhancer(test_state, create_fake_model_files)
        rewritten = "a rewritten felt palace with slow camera moves"
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = rewritten
        wrap = _cozy_felt_wrap()
        scene = "mouse in a house"

        _generate_explore(
            test_state,
            {**_T2V_JSON, "prompt": scene},
            prompt_wrap=wrap,
        )

        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls[0]["prompt"] == scene
        assert fake_services.fast_video_pipeline.generate_calls[0]["prompt"] == wrap(rewritten)

    def test_api_text_encoding_still_rewrites_a_recipe_locally_before_the_wrap(
        self, test_state, fake_services, create_fake_model_files
    ):
        # An API key selects API text encoding by default. The recipe wrap then
        # turns server-side enhance off so the scaffold survives. Home auto-enhance
        # still has to rewrite the scene locally first, or Generate starts immediately
        # on the typed prompt.
        _install_api_encoding(test_state, fake_services, create_fake_model_files)
        rewritten = "a rewritten felt palace with slow camera moves"
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = rewritten
        wrap = _cozy_felt_wrap()
        scene = "mouse in a house"

        _generate_explore(
            test_state,
            {**_T2V_JSON, "prompt": scene},
            prompt_wrap=wrap,
        )

        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls[0]["prompt"] == scene
        assert fake_services.fast_video_pipeline.generate_calls[0]["prompt"] == wrap(rewritten)
        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is False


class TestApiTextEncodingEnhanceFlag:
    def test_typed_prompt_keeps_the_server_side_enhance_flag_on(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        _install_api_encoding(test_state, fake_services, create_fake_model_files)

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200

        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is True
        # API text encoding rewrites server-side; the local enhancer must stay out of it.
        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls == []

    def test_enhanced_provenance_turns_the_server_side_enhance_flag_off(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        _install_api_encoding(test_state, fake_services, create_fake_model_files)

        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "prompt": "already rewritten", "promptProvenance": "enhanced"},
        )
        assert r.status_code == 200

        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is False


    def test_toggle_off_turns_the_server_side_enhance_flag_off_for_t2v(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        _install_api_encoding(test_state, fake_services, create_fake_model_files)
        test_state.state.app_settings.prompt_enhancer_enabled = False

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200

        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is False
        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls == []

    def test_toggle_on_by_default_leaves_the_server_side_flag_on_for_i2v(
        self, client, test_state, fake_services, create_fake_model_files, make_test_image, tmp_path
    ):
        _install_api_encoding(test_state, fake_services, create_fake_model_files)
        image_path = tmp_path / "input.png"
        image_path.write_bytes(make_test_image().getvalue())

        r = client.post("/api/generate", json={**_T2V_JSON, "imagePath": str(image_path)})
        assert r.status_code == 200

        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is True

    def test_toggle_off_turns_the_server_side_enhance_flag_off_for_i2v(
        self, client, test_state, fake_services, create_fake_model_files, make_test_image, tmp_path
    ):
        _install_api_encoding(test_state, fake_services, create_fake_model_files)
        test_state.state.app_settings.prompt_enhancer_enabled = False
        image_path = tmp_path / "input.png"
        image_path.write_bytes(make_test_image().getvalue())

        r = client.post("/api/generate", json={**_T2V_JSON, "imagePath": str(image_path)})
        assert r.status_code == 200

        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is False

    def test_toggle_off_turns_the_server_side_enhance_flag_off_for_a2v(
        self, client, test_state, fake_services, create_fake_model_files, tmp_path
    ):
        _install_api_encoding(test_state, fake_services, create_fake_model_files)
        test_state.state.app_settings.prompt_enhancer_enabled = False
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post("/api/generate", json={**_T2V_JSON, "audioPath": str(audio_file)})
        assert r.status_code == 200

        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is False

    def test_a2v_with_start_frame_follows_the_same_single_toggle(
        self, client, test_state, fake_services, create_fake_model_files, make_test_image, tmp_path
    ):
        # Mixed conditioning used to pick between two gates; one switch now covers them all.
        _install_api_encoding(test_state, fake_services, create_fake_model_files)
        test_state.state.app_settings.prompt_enhancer_enabled = False
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)
        image_path = tmp_path / "input.png"
        image_path.write_bytes(make_test_image().getvalue())

        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "audioPath": str(audio_file), "imagePath": str(image_path)},
        )
        assert r.status_code == 200

        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is False

    def test_enhanced_provenance_wins_over_a_toggle_on(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        _install_api_encoding(test_state, fake_services, create_fake_model_files)
        test_state.state.app_settings.prompt_enhancer_enabled = True

        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "prompt": "already rewritten", "promptProvenance": "enhanced"},
        )
        assert r.status_code == 200

        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is False


class TestLockedSeedAutomaticEnhancement:
    """A locked seed has to cover the whole local generation, rewrite included.

    Automatic enhancement is part of what produces the video, so drawing it independently would
    leave a "locked" local generation irreproducible: the same seed would still yield a different
    prompt, and so a different clip. Manual Enhance is a separate, exploratory action and keeps
    its independent draw.
    """

    def test_locked_seed_is_reused_for_the_automatic_local_rewrite(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        _install_local_enhancer(test_state, create_fake_model_files)
        test_state.state.app_settings.seed_locked = True
        test_state.state.app_settings.locked_seed = 4242

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200

        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls[0]["seed"] == 4242
        assert fake_services.fast_video_pipeline.generate_calls[0]["seed"] == 4242

    def test_locked_i2v_rewrite_uses_the_generation_seed(
        self, client, test_state, fake_services, create_fake_model_files, make_test_image, tmp_path
    ):
        _install_local_enhancer(test_state, create_fake_model_files)
        test_state.state.app_settings.seed_locked = True
        test_state.state.app_settings.locked_seed = 4242
        image_path = tmp_path / "input.png"
        image_path.write_bytes(make_test_image().getvalue())

        r = client.post("/api/generate", json={**_T2V_JSON, "imagePath": str(image_path)})
        assert r.status_code == 200

        assert fake_services.prompt_enhancer_pipeline.enhance_i2v_calls[0]["seed"] == 4242
        assert fake_services.fast_video_pipeline.generate_calls[0]["seed"] == 4242

    def test_an_explicit_request_seed_still_wins_and_drives_the_rewrite(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        _install_local_enhancer(test_state, create_fake_model_files)
        test_state.state.app_settings.seed_locked = True
        test_state.state.app_settings.locked_seed = 4242

        r = client.post("/api/generate", json={**_T2V_JSON, "seed": 77})
        assert r.status_code == 200

        assert fake_services.fast_video_pipeline.generate_calls[0]["seed"] == 77
        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls[0]["seed"] == 77

    def test_an_explicit_request_seed_pins_the_rewrite_while_unlocked(
        self, client, test_state, fake_services, create_fake_model_files, monkeypatch
    ):
        _install_local_enhancer(test_state, create_fake_model_files)
        test_state.state.app_settings.seed_locked = False
        monkeypatch.setattr("handlers.base.random.randint", lambda _start, _end: 909)

        r = client.post("/api/generate", json={**_T2V_JSON, "seed": 77})
        assert r.status_code == 200

        assert fake_services.fast_video_pipeline.generate_calls[0]["seed"] == 77
        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls[0]["seed"] == 77

    def test_an_unlocked_rewrite_keeps_its_own_random_seed(
        self, client, test_state, fake_services, create_fake_model_files, monkeypatch
    ):
        _install_local_enhancer(test_state, create_fake_model_files)
        # Both handlers draw from the one stdlib module, so a single patch covers the generation
        # seed (drawn first) and the enhancement seed (drawn second).
        draws = iter((101, 909))
        monkeypatch.setattr("handlers.base.random.randint", lambda _start, _end: next(draws))

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200

        assert fake_services.fast_video_pipeline.generate_calls[0]["seed"] == 101
        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls[0]["seed"] == 909

    def test_manual_enhance_stays_random_under_a_seed_lock(
        self, client, test_state, fake_services, create_fake_model_files, monkeypatch
    ):
        _install_local_enhancer(test_state, create_fake_model_files)
        test_state.state.app_settings.seed_locked = True
        test_state.state.app_settings.locked_seed = 4242
        monkeypatch.setattr("handlers.base.random.randint", lambda _start, _end: 909)

        r = client.post("/api/enhance-prompt", json={"prompt": "a cat"})
        assert r.status_code == 200

        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls[0]["seed"] == 909

    def test_locked_seed_reaches_the_catalog_ic_lora_rewrite(
        self, client, test_state, fake_services, create_fake_model_files, tmp_path
    ):
        _install_local_enhancer(test_state, create_fake_model_files)
        test_state.state.app_settings.seed_locked = True
        test_state.state.app_settings.locked_seed = 4242
        _download_ic_lora(client, "ingredients-v1")
        img = _write_png(tmp_path / "in.png")

        r = client.post(
            "/api/ic-lora/generate",
            json={
                "ic_lora_id": "ingredients-v1",
                "input_path": str(img),
                "control_values": {"duration": 5},
                "prompt": "a slow orbit",
                "conditioning_type": "custom",
            },
        )
        assert r.status_code == 200

        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls[0]["seed"] == 4242
        assert fake_services.ic_lora_pipeline.generate_calls[-1]["seed"] == 4242

    def test_locked_seed_reaches_the_builtin_ic_lora_rewrite(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files
    ):
        create_fake_model_files(include_prompt_enhancer=True)
        create_fake_ic_lora_files()
        test_state.state.app_settings.use_local_text_encoder = True
        test_state.state.app_settings.seed_locked = True
        test_state.state.app_settings.locked_seed = 4242
        video_path = _register_input_video(test_state)

        r = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [],
            },
        )
        assert r.status_code == 200

        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls[0]["seed"] == 4242
        assert fake_services.ic_lora_pipeline.generate_calls[-1]["seed"] == 4242


class TestIcLoraRunLogEnhancementField:
    """The run log's `enhance=` field names where the rewrite happened, not a request flag."""

    def _generate_canny(self, client, video_path) -> None:
        r = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [],
            },
        )
        assert r.status_code == 200

    def _run_line(self, caplog) -> str:
        lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[ic-lora] run:")]
        assert len(lines) == 1
        return lines[0]

    def test_a_local_rewrite_is_logged_as_local(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files, caplog
    ):
        create_fake_model_files(include_prompt_enhancer=True)
        create_fake_ic_lora_files()
        test_state.state.app_settings.use_local_text_encoder = True
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = "a long descriptive caption"
        video_path = _register_input_video(test_state)

        with caplog.at_level(logging.INFO):
            self._generate_canny(client, video_path)

        assert "enhance=local" in self._run_line(caplog)

    def test_a_missing_enhancer_is_logged_as_none(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files, caplog
    ):
        create_fake_model_files(include_prompt_enhancer=False)
        create_fake_ic_lora_files()
        test_state.state.app_settings.use_local_text_encoder = True
        video_path = _register_input_video(test_state)

        with caplog.at_level(logging.INFO):
            self._generate_canny(client, video_path)

        assert "enhance=none" in self._run_line(caplog)

    def test_server_side_enhancement_is_logged_as_api(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files, caplog
    ):
        _install_api_encoding(test_state, fake_services, create_fake_model_files)
        create_fake_ic_lora_files()
        video_path = _register_input_video(test_state)

        with caplog.at_level(logging.INFO):
            self._generate_canny(client, video_path)

        assert "enhance=api" in self._run_line(caplog)


class TestCatalogIcLoraAutomaticEnhancement:
    def _setup(self, test_state, create_fake_model_files) -> None:
        _install_local_enhancer(test_state, create_fake_model_files)

    def _generate_trigger_ic_lora(self, client, fake_services, tmp_path) -> str:
        _add_image_ic_lora(
            fake_services,
            id="shave-v1",
            name="Instant Shave",
            trigger="REMOVEBEARD",
            trigger_placement="anywhere",
        )
        _download_ic_lora(client, "shave-v1")
        img = _write_png(tmp_path / "in.png")

        r = client.post(
            "/api/ic-lora/generate",
            json={
                "ic_lora_id": "shave-v1",
                "input_path": str(img),
                "prompt": "shave him",
                "conditioning_type": "custom",
            },
        )
        assert r.status_code == 200
        return fake_services.ic_lora_pipeline.generate_calls[-1]["prompt"]

    def test_prompt_is_enhanced_with_the_catalog_aware_system_prompt(
        self, client, test_state, fake_services, create_fake_model_files, tmp_path
    ):
        self._setup(test_state, create_fake_model_files)
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = "a long descriptive caption"
        _download_ic_lora(client, "ingredients-v1")
        img = _write_png(tmp_path / "in.png")

        r = client.post(
            "/api/ic-lora/generate",
            json={
                "ic_lora_id": "ingredients-v1",
                "input_path": str(img),
                "control_values": {"duration": 5},
                "prompt": "a slow orbit",
                "conditioning_type": "custom",
            },
        )
        assert r.status_code == 200

        call = fake_services.prompt_enhancer_pipeline.enhance_t2v_calls[0]
        assert call["prompt"] == "a slow orbit"
        assert call["system_prompt"] is not None
        assert "Ingredients" in call["system_prompt"]
        assert fake_services.ic_lora_pipeline.generate_calls[-1]["prompt"] == "a long descriptive caption"

    def test_enhanced_provenance_is_submitted_as_is(
        self, client, test_state, fake_services, create_fake_model_files, tmp_path
    ):
        self._setup(test_state, create_fake_model_files)
        _download_ic_lora(client, "ingredients-v1")
        img = _write_png(tmp_path / "in.png")

        r = client.post(
            "/api/ic-lora/generate",
            json={
                "ic_lora_id": "ingredients-v1",
                "input_path": str(img),
                "control_values": {"duration": 5},
                "prompt": "already rewritten",
                "prompt_provenance": "enhanced",
                "conditioning_type": "custom",
            },
        )
        assert r.status_code == 200

        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls == []
        assert fake_services.ic_lora_pipeline.generate_calls[-1]["prompt"] == "already rewritten"

    def test_empty_prompt_is_never_enhanced(
        self, client, test_state, fake_services, create_fake_model_files, tmp_path
    ):
        self._setup(test_state, create_fake_model_files)
        _download_ic_lora(client, "outpaint-v1")
        src = _register_outpaint_source(fake_services, tmp_path)

        r = client.post(
            "/api/ic-lora/generate",
            json={
                "ic_lora_id": "outpaint-v1",
                "input_path": str(src),
                "prompt": "",
                "conditioning_type": "custom",
                "outpaint_pads": {"left": 48, "right": 0, "top": 0, "bottom": 0},
            },
        )
        assert r.status_code == 200

        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls == []
        assert fake_services.ic_lora_pipeline.generate_calls[-1]["prompt"] == ""

    def test_template_entry_is_filled_automatically(
        self, client, test_state, fake_services, create_fake_model_files, tmp_path
    ):
        self._setup(test_state, create_fake_model_files)
        _add_image_ic_lora(
            fake_services,
            id="colorization",
            name="Colorization",
            prompt_template=PromptTemplateSpec(
                template="Reference shows {reference}. COLORIZE {result}.",
                placeholders={"reference": PromptTemplatePlaceholder(), "result": PromptTemplatePlaceholder()},
            ),
        )
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = (
            '{"reference": "a grey rabbit", "result": "a brown rabbit"}'
        )
        _download_ic_lora(client, "colorization")
        img = _write_png(tmp_path / "in.png")

        r = client.post(
            "/api/ic-lora/generate",
            json={
                "ic_lora_id": "colorization",
                "input_path": str(img),
                "prompt": "colorize the rabbit",
                "conditioning_type": "custom",
            },
        )
        assert r.status_code == 200
        assert (
            fake_services.ic_lora_pipeline.generate_calls[-1]["prompt"]
            == "Reference shows a grey rabbit. COLORIZE a brown rabbit."
        )

    def test_unfillable_template_falls_back_to_the_typed_prompt(
        self, client, test_state, fake_services, create_fake_model_files, tmp_path
    ):
        self._setup(test_state, create_fake_model_files)
        _add_image_ic_lora(
            fake_services,
            id="upscale",
            name="Upscale",
            prompt_template=PromptTemplateSpec(template="upscale", placeholders={}),
        )
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = "not json"
        _download_ic_lora(client, "upscale")
        img = _write_png(tmp_path / "in.png")

        r = client.post(
            "/api/ic-lora/generate",
            json={
                "ic_lora_id": "upscale",
                "input_path": str(img),
                "prompt": "sharpen it",
                "conditioning_type": "custom",
            },
        )
        assert r.status_code == 200
        assert fake_services.ic_lora_pipeline.generate_calls[-1]["prompt"] == "sharpen it"

    def test_trigger_is_enforced_on_the_automatic_rewrite(
        self, client, test_state, fake_services, create_fake_model_files, tmp_path
    ):
        self._setup(test_state, create_fake_model_files)
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = "a clean shaven man"
        assert "REMOVEBEARD" in self._generate_trigger_ic_lora(client, fake_services, tmp_path)

    def test_missing_local_enhancer_fallback_preserves_the_catalog_trigger(
        self, client, test_state, fake_services, create_fake_model_files, tmp_path
    ):
        create_fake_model_files(include_prompt_enhancer=False)
        test_state.state.app_settings.use_local_text_encoder = True

        assert self._generate_trigger_ic_lora(client, fake_services, tmp_path) == "shave him REMOVEBEARD"

    def test_failing_local_enhancer_fallback_preserves_the_catalog_trigger(
        self, client, test_state, fake_services, create_fake_model_files, tmp_path
    ):
        self._setup(test_state, create_fake_model_files)
        fake_services.prompt_enhancer_pipeline.raise_on_enhance = RuntimeError("boom")

        assert self._generate_trigger_ic_lora(client, fake_services, tmp_path) == "shave him REMOVEBEARD"

    def test_blank_local_enhancer_fallback_preserves_the_catalog_trigger(
        self, client, test_state, fake_services, create_fake_model_files, tmp_path
    ):
        self._setup(test_state, create_fake_model_files)
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = "   "

        assert self._generate_trigger_ic_lora(client, fake_services, tmp_path) == "shave him REMOVEBEARD"

    def test_missing_local_enhancer_falls_back_to_the_typed_prompt(
        self, client, test_state, fake_services, create_fake_model_files, tmp_path
    ):
        create_fake_model_files(include_prompt_enhancer=False)
        test_state.state.app_settings.use_local_text_encoder = True
        _download_ic_lora(client, "ingredients-v1")
        img = _write_png(tmp_path / "in.png")

        r = client.post(
            "/api/ic-lora/generate",
            json={
                "ic_lora_id": "ingredients-v1",
                "input_path": str(img),
                "control_values": {"duration": 5},
                "prompt": "a slow orbit",
                "conditioning_type": "custom",
            },
        )
        assert r.status_code == 200
        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls == []
        assert fake_services.ic_lora_pipeline.generate_calls[-1]["prompt"] == "a slow orbit"

    def test_api_text_encoding_keeps_the_server_side_flag_for_a_typed_prompt(
        self, client, test_state, fake_services, create_fake_model_files, tmp_path
    ):
        _install_api_encoding(test_state, fake_services, create_fake_model_files)
        _download_ic_lora(client, "ingredients-v1")
        img = _write_png(tmp_path / "in.png")

        r = client.post(
            "/api/ic-lora/generate",
            json={
                "ic_lora_id": "ingredients-v1",
                "input_path": str(img),
                "control_values": {"duration": 5},
                "prompt": "a slow orbit",
                "conditioning_type": "custom",
            },
        )
        assert r.status_code == 200
        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is True

    def test_api_text_encoding_drops_the_server_side_flag_for_an_enhanced_prompt(
        self, client, test_state, fake_services, create_fake_model_files, tmp_path
    ):
        _install_api_encoding(test_state, fake_services, create_fake_model_files)
        _download_ic_lora(client, "ingredients-v1")
        img = _write_png(tmp_path / "in.png")

        r = client.post(
            "/api/ic-lora/generate",
            json={
                "ic_lora_id": "ingredients-v1",
                "input_path": str(img),
                "control_values": {"duration": 5},
                "prompt": "already rewritten",
                "prompt_provenance": "enhanced",
                "conditioning_type": "custom",
            },
        )
        assert r.status_code == 200
        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is False

    def test_api_text_encoding_honors_the_enhance_toggle_off(
        self, client, test_state, fake_services, create_fake_model_files, tmp_path
    ):
        _install_api_encoding(test_state, fake_services, create_fake_model_files)
        test_state.state.app_settings.prompt_enhancer_enabled = False
        _download_ic_lora(client, "ingredients-v1")
        img = _write_png(tmp_path / "in.png")

        r = client.post(
            "/api/ic-lora/generate",
            json={
                "ic_lora_id": "ingredients-v1",
                "input_path": str(img),
                "control_values": {"duration": 5},
                "prompt": "a slow orbit",
                "conditioning_type": "custom",
            },
        )
        assert r.status_code == 200
        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is False


class TestBuiltinIcLoraAutomaticEnhancement:
    def test_conditioning_prompt_is_enhanced_with_the_conditioning_system_prompt(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files
    ):
        create_fake_model_files(include_prompt_enhancer=True)
        create_fake_ic_lora_files()
        test_state.state.app_settings.use_local_text_encoder = True
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = "a long descriptive caption"
        video_path = _register_input_video(test_state)

        r = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [],
            },
        )
        assert r.status_code == 200

        call = fake_services.prompt_enhancer_pipeline.enhance_t2v_calls[0]
        assert call["prompt"] == "test prompt"
        assert call["system_prompt"] is not None
        assert "edge" in call["system_prompt"].lower()
        assert fake_services.ic_lora_pipeline.generate_calls[-1]["prompt"] == "a long descriptive caption"

    def test_enhanced_provenance_is_submitted_as_is(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files
    ):
        create_fake_model_files(include_prompt_enhancer=True)
        create_fake_ic_lora_files()
        test_state.state.app_settings.use_local_text_encoder = True
        video_path = _register_input_video(test_state)

        r = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "already rewritten",
                "prompt_provenance": "enhanced",
                "images": [],
            },
        )
        assert r.status_code == 200

        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls == []
        assert fake_services.ic_lora_pipeline.generate_calls[-1]["prompt"] == "already rewritten"


class TestCatalogPlainLoraAutomaticEnhancement:
    """A catalog LoRA on the normal video picker gets manual Enhance's catalog treatment.

    The generic rewrite has no idea the adapter needs a trigger phrase, so left to itself it
    paraphrases the one the user typed away and the LoRA silently does nothing. The automatic
    path therefore has to reach the same catalog system prompt and the same deterministic
    trigger enforcement the Enhance button already applies — resolved from the catalog id the
    request carries, never from the weights' filename.
    """

    _CATALOG_ID = "cozy-felt"
    _TRIGGER = "F3ltCut0u7"

    def _add_catalog_lora(self, fake_services, create_fake_lora, *, filename: str = "cozy-felt.safetensors") -> str:
        add_lora(
            fake_services,
            id=self._CATALOG_ID,
            name="Cozy Felt",
            trigger=self._TRIGGER,
            trigger_placement="anywhere",
            download=download_spec(filename),
            instructions=[InstructionSection(kind="summary", title="What it does", body="Felt look.")],
        )
        return create_fake_lora(filename)

    def _generate(self, client, ref: str, *, catalog_id: str | None = _CATALOG_ID, **extra: object):
        entry: dict[str, object] = {"ref": ref, "scale": 0.8}
        if catalog_id is not None:
            entry["catalogId"] = catalog_id
        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "prompt": "a fox", "loras": [entry], **extra},
        )
        assert r.status_code == 200, r.text
        return r

    def test_t2v_prompt_is_enhanced_with_the_catalog_aware_system_prompt(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_lora
    ):
        _install_local_enhancer(test_state, create_fake_model_files)
        ref = self._add_catalog_lora(fake_services, create_fake_lora)
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = "a felt fox in a garden"

        self._generate(client, ref)

        call = fake_services.prompt_enhancer_pipeline.enhance_t2v_calls[0]
        assert call["prompt"] == "a fox"
        assert call["system_prompt"] is not None
        assert "Cozy Felt" in call["system_prompt"]
        assert "Felt look." in call["system_prompt"]
        assert (
            fake_services.fast_video_pipeline.generate_calls[0]["prompt"]
            == f"a felt fox in a garden {self._TRIGGER}"
        )

    def test_i2v_prompt_is_enhanced_with_the_catalog_aware_system_prompt(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_lora,
        make_test_image, tmp_path,
    ):
        _install_local_enhancer(test_state, create_fake_model_files)
        ref = self._add_catalog_lora(fake_services, create_fake_lora)
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = "a felt fox in a garden"
        image_path = tmp_path / "input.png"
        image_path.write_bytes(make_test_image().getvalue())

        self._generate(client, ref, imagePath=str(image_path))

        call = fake_services.prompt_enhancer_pipeline.enhance_i2v_calls[0]
        assert call["system_prompt"] is not None
        assert "Cozy Felt" in call["system_prompt"]
        assert (
            fake_services.fast_video_pipeline.generate_calls[0]["prompt"]
            == f"a felt fox in a garden {self._TRIGGER}"
        )

    def test_a2v_prompt_is_enhanced_with_the_catalog_aware_system_prompt(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_lora, tmp_path
    ):
        _install_local_enhancer(test_state, create_fake_model_files)
        ref = self._add_catalog_lora(fake_services, create_fake_lora)
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = "a felt fox in a garden"
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        self._generate(client, ref, audioPath=str(audio_file))

        call = fake_services.prompt_enhancer_pipeline.enhance_t2v_calls[0]
        assert call["system_prompt"] is not None
        assert "Cozy Felt" in call["system_prompt"]
        assert (
            fake_services.a2v_pipeline.generate_calls[0]["prompt"]
            == f"a felt fox in a garden {self._TRIGGER}"
        )

    def test_missing_local_enhancer_fallback_preserves_the_catalog_trigger(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_lora
    ):
        create_fake_model_files(include_prompt_enhancer=False)
        test_state.state.app_settings.use_local_text_encoder = True
        ref = self._add_catalog_lora(fake_services, create_fake_lora)

        self._generate(client, ref)

        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls == []
        assert (
            fake_services.fast_video_pipeline.generate_calls[0]["prompt"]
            == f"a fox {self._TRIGGER}"
        )

    def test_failing_local_enhancer_fallback_preserves_the_catalog_trigger(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_lora
    ):
        _install_local_enhancer(test_state, create_fake_model_files)
        ref = self._add_catalog_lora(fake_services, create_fake_lora)
        fake_services.prompt_enhancer_pipeline.raise_on_enhance = RuntimeError("boom")

        self._generate(client, ref)

        assert (
            fake_services.fast_video_pipeline.generate_calls[0]["prompt"]
            == f"a fox {self._TRIGGER}"
        )

    def test_blank_local_rewrite_fallback_preserves_the_catalog_trigger(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_lora
    ):
        _install_local_enhancer(test_state, create_fake_model_files)
        ref = self._add_catalog_lora(fake_services, create_fake_lora)
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = "   "

        self._generate(client, ref)

        assert (
            fake_services.fast_video_pipeline.generate_calls[0]["prompt"]
            == f"a fox {self._TRIGGER}"
        )

    def test_custom_lora_stays_weights_only_even_when_a_catalog_entry_shares_its_filename(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_lora
    ):
        """No catalog id, no catalog treatment — matching on filename would attach a
        stranger's trigger phrase to an unrelated file the user trained themselves.
        """
        _install_local_enhancer(test_state, create_fake_model_files)
        ref = self._add_catalog_lora(fake_services, create_fake_lora, filename="mine.safetensors")
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = "a fox in a garden"

        self._generate(client, ref, catalog_id=None)

        call = fake_services.prompt_enhancer_pipeline.enhance_t2v_calls[0]
        assert call["system_prompt"] is None
        assert fake_services.fast_video_pipeline.generate_calls[0]["prompt"] == "a fox in a garden"

    def test_a_catalog_id_the_catalog_no_longer_knows_still_generates(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_lora
    ):
        """Enhancement is a quality step: a stale id costs the catalog treatment, not the run."""
        _install_local_enhancer(test_state, create_fake_model_files)
        ref = create_fake_lora("orphan.safetensors")
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = "a fox in a garden"

        self._generate(client, ref, catalog_id="retired-lora")

        assert fake_services.fast_video_pipeline.generate_calls[0]["prompt"] == "a fox in a garden"

    def test_enhanced_provenance_is_generated_exactly_as_submitted(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_lora
    ):
        _install_local_enhancer(test_state, create_fake_model_files)
        ref = self._add_catalog_lora(fake_services, create_fake_lora)

        self._generate(client, ref, promptProvenance="enhanced")

        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls == []
        assert fake_services.fast_video_pipeline.generate_calls[0]["prompt"] == "a fox"
