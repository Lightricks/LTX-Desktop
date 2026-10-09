"""Integration-style tests for generation and image endpoints."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from _routes._errors import HTTPError
from api_types import (
    LOCAL_MULTI_KEYFRAME_MAX_COUNT,
    GenerateImageRequest,
    GenerateVideoCompleteResponse,
    GenerateVideoRequest,
)
from frame_math import AutoDurationSpec, compute_num_frames
from runtime_config.model_download_specs import delete_cp_path, get_ltx_model_spec, resolve_model_path
from services import generation_interrupt
from services.generation_interrupt import GenerationCancelledError
from services.ltx_api_client.ltx_api_client import LTXAPIClientError
from state.app_state_types import GpuSlot, VideoPipelineState
from tests.http_error_assertions import assert_http_error
from tests.fakes.services import FakeFastVideoPipeline
from runtime_config.video_job_budget import (
    LOCAL_GENERATION_UNSUPPORTED,
    LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
    VIDEO_JOB_TOO_LARGE,
    VIDEO_JOB_TOO_LARGE_MESSAGE,
)


@dataclass
class _FakeEncodingResult:
    """Minimal stand-in for TextEncodingResult in tests."""

    video_context: object = "fake_tensor"
    audio_context: object = None

_API_ENCODING_MODEL_ID = "ltx-2.3-22b-distilled-1.1"
_LOCAL_2_3 = "ltx-2.3-22b-distilled-1.1"

_T2V_JSON = {
    "prompt": "test",
    "resolution": "540p",
    "model": "fast",
    "duration": 5,
    "fps": 24,
}


def _install_local_2_3(test_state, create_fake_model_files, **kwargs) -> None:
    create_fake_model_files(model_id=_LOCAL_2_3, **kwargs)
    test_state.state.app_settings.active_ltx_model_id = _LOCAL_2_3


def _write_test_wav(path: Path, *, duration_seconds: float = 0.1, sample_rate: int = 8000) -> None:
    import wave

    frame_count = max(1, int(duration_seconds * sample_rate))
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frame_count)


def _enable_local_text_encoding(test_state) -> None:
    test_state.state.app_settings.use_local_text_encoder = True


def _fake_running_generation_state(test_state) -> None:
    pipeline = FakeFastVideoPipeline()
    test_state.state.gpu_slot = GpuSlot(
        active_pipeline=VideoPipelineState(
            pipeline=pipeline,
            is_compiled=False,
            ltx_model_id="ltx-2.5-22b-distilled",
            loading_mode="full_models_loading",
        ),
    )
    test_state.generation.start_generation("running")


def _cancel_in_flight(client, entered_inference: threading.Event, run: Callable[[], object]) -> object:
    """Run `run` on a thread, cancel via HTTP once inference has started, return its result.

    TestClient is not safe for overlapping POSTs; the in-flight generate is a direct
    handler call so cancel can use the HTTP client.
    """
    result: dict[str, object] = {}

    def target() -> None:
        try:
            result["value"] = run()
        except Exception as exc:
            result["error"] = exc

    thread = threading.Thread(target=target)
    thread.start()
    assert entered_inference.wait(timeout=5.0), "pipeline never entered inference"
    cancel = client.post("/api/generate/cancel")
    assert cancel.status_code == 200
    thread.join(timeout=8.0)
    assert not thread.is_alive(), "in-flight generate did not finish after cancel"
    if "error" in result:
        raise result["error"]  # type: ignore[misc]
    return result["value"]


class TestGenerate:
    def test_t2v_requires_downloaded_ltx_model(self, client):
        r = client.post("/api/generate", json=_T2V_JSON)
        assert_http_error(r, status_code=409, code="NO_DOWNLOADED_LTX_MODEL")

    def test_t2v_on_2_5_uses_api_model_selector_without_local_text_encoder(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        create_fake_model_files()
        test_state.state.app_settings.ltx_api_key = "api-key"
        test_state.state.app_settings.use_local_text_encoder = False
        spec = get_ltx_model_spec("ltx-2.5-22b-distilled")
        resolve_model_path(test_state.config.default_models_dir, spec.text_encoder_cp).unlink()
        fake_services.text_encoder.encode_responses.append(_FakeEncodingResult())

        r = client.post("/api/generate", json=_T2V_JSON)

        assert r.status_code == 200
        assert fake_services.text_encoder.encode_calls[0]["api_model"] == spec.api_prompt_embedding_model
        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is True

    def test_t2v_happy_path(self, client, test_state, fake_services, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A beautiful sunset",
                "resolution": "1080p",
                "model": "fast",
                "duration": 5,
                "fps": 24,
                "cameraMotion": "none",
            },
        )

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "complete"
        assert data["video_path"] is not None
        assert Path(data["video_path"]).exists()

        pipeline = fake_services.fast_video_pipeline
        assert len(pipeline.generate_calls) == 1

    def test_generate_local_reserved_writes_to_caller_output_path(
        self, test_state, fake_services, create_fake_model_files, tmp_path
    ):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        reserved_path = tmp_path / "assets" / "queued.mp4"
        reserved_path.parent.mkdir(parents=True)

        shutdown = threading.Event()
        with test_state.generation.wait_for_generation_slot(shutdown):
            result = test_state.video_generation.generate_local_reserved(
                GenerateVideoRequest.model_validate(_T2V_JSON),
                generation_id="reserved-1",
                output_path=reserved_path,
                local_model_id="ltx-2.5-22b-distilled",
            )

        assert isinstance(result, GenerateVideoCompleteResponse)
        assert result.status == "complete"
        assert result.video_path == str(reserved_path)
        assert reserved_path.exists()
        assert not Path(result.video_path).is_relative_to(test_state.config.outputs_dir)
        assert fake_services.fast_video_pipeline.generate_calls[0]["output_path"] == str(reserved_path)

    def test_recipe_wrap_is_last_mutation_and_survives_api_encoding(
        self, test_state, create_fake_model_files, tmp_path, monkeypatch
    ):
        # Verifies the recipe scaffold wrap is the *last* text mutation and reaches
        # encoding intact on the API text-encoding path: resolve_for_generation
        # returns the raw prompt with enhance_via_api=True, and the wrap must both
        # prepend the scaffold and force enhance_via_api=False so /prompt-embedding
        # can't paraphrase the style lock away. Captures the args that reach
        # generate_video (the encoding boundary) and short-circuits there.
        from services.features.lora_recipes import (
            get_lora_recipe,
            make_recipe_prompt_wrap,
        )

        create_fake_model_files()
        monkeypatch.setattr(
            test_state.text,
            "should_use_local_encoding",
            lambda model_id=None: False,
        )

        captured: dict[str, object] = {}

        class _StopAfterCapture(Exception):
            pass

        def _capture_generate_video(**kwargs: object) -> str:
            captured["prompt"] = kwargs["prompt"]
            captured["enhance_via_api"] = kwargs["enhance_via_api"]
            raise _StopAfterCapture

        monkeypatch.setattr(
            test_state.video_generation, "generate_video", _capture_generate_video
        )

        recipe = get_lora_recipe("cozy-felt")
        assert recipe is not None
        req = GenerateVideoRequest.model_validate(_T2V_JSON)
        out = tmp_path / "assets" / "recipe.mp4"
        out.parent.mkdir(parents=True)
        shutdown = threading.Event()

        # Baseline: without a recipe wrap, the API path keeps enhance_via_api=True.
        with test_state.generation.wait_for_generation_slot(shutdown):
            with pytest.raises(HTTPError):
                test_state.video_generation.generate_local_reserved(
                    req, generation_id="no-wrap", output_path=out,
                    local_model_id="ltx-2.5-22b-distilled",
                )
        assert captured["enhance_via_api"] is True

        captured.clear()
        with test_state.generation.wait_for_generation_slot(shutdown):
            with pytest.raises(HTTPError):
                test_state.video_generation.generate_local_reserved(
                    req,
                    generation_id="with-wrap",
                    output_path=out,
                    prompt_wrap=make_recipe_prompt_wrap(recipe),
                    local_model_id="ltx-2.5-22b-distilled",
                )
        prompt = captured["prompt"]
        assert isinstance(prompt, str)
        assert prompt.startswith("F3ltCut0u7 handcrafted felt")
        assert captured["enhance_via_api"] is False

    def test_generate_local_reserved_override_loads_non_active_weights(
        self, test_state, fake_services, create_fake_model_files, tmp_path
    ):
        create_fake_model_files()
        create_fake_model_files(model_id="ltx-2.3-22b-distilled-1.1")
        _enable_local_text_encoding(test_state)
        test_state.state.app_settings.active_ltx_model_id = "ltx-2.5-22b-distilled"
        reserved_path = tmp_path / "assets" / "queued.mp4"
        reserved_path.parent.mkdir(parents=True)

        shutdown = threading.Event()
        with test_state.generation.wait_for_generation_slot(shutdown):
            result = test_state.video_generation.generate_local_reserved(
                GenerateVideoRequest.model_validate(_T2V_JSON),
                generation_id="reserved-2.3",
                output_path=reserved_path,
                local_model_id="ltx-2.3-22b-distilled-1.1",
            )

        assert isinstance(result, GenerateVideoCompleteResponse)
        assert result.status == "complete"
        slot = test_state.state.gpu_slot
        assert slot is not None
        assert slot.active_pipeline.ltx_model_id == "ltx-2.3-22b-distilled-1.1"
        assert test_state.state.app_settings.active_ltx_model_id == "ltx-2.5-22b-distilled"

    def test_generate_local_reserved_2_5_auto_duration_ignores_active_2_3_head(
        self, test_state, fake_services, create_fake_model_files, tmp_path
    ):
        create_fake_model_files()
        create_fake_model_files(model_id="ltx-2.3-22b-distilled-1.1")
        _enable_local_text_encoding(test_state)
        test_state.state.app_settings.active_ltx_model_id = "ltx-2.3-22b-distilled-1.1"
        reserved_path = tmp_path / "assets" / "queued-auto-2.5.mp4"
        reserved_path.parent.mkdir(parents=True)

        shutdown = threading.Event()
        with test_state.generation.wait_for_generation_slot(shutdown):
            result = test_state.video_generation.generate_local_reserved(
                GenerateVideoRequest.model_validate({**_T2V_JSON, "duration": None}),
                generation_id="reserved-2.5-auto",
                output_path=reserved_path,
                local_model_id="ltx-2.5-22b-distilled",
            )

        assert isinstance(result, GenerateVideoCompleteResponse)
        call = fake_services.fast_video_pipeline.generate_calls[0]
        assert call["num_frames"] == AutoDurationSpec(min_seconds=2, max_seconds=20)
        assert test_state.state.app_settings.active_ltx_model_id == "ltx-2.3-22b-distilled-1.1"

    def test_generate_local_reserved_2_5_api_embeddings_ignore_active_2_3(
        self, test_state, fake_services, create_fake_model_files, tmp_path
    ):
        create_fake_model_files()
        create_fake_model_files(model_id="ltx-2.3-22b-distilled-1.1")
        spec = get_ltx_model_spec("ltx-2.5-22b-distilled")
        delete_cp_path(test_state.config.default_models_dir, spec.text_encoder_cp)
        test_state.state.app_settings.active_ltx_model_id = "ltx-2.3-22b-distilled-1.1"
        test_state.state.app_settings.ltx_api_key = "api-key"
        test_state.state.app_settings.use_local_text_encoder = False
        fake_services.text_encoder.encode_responses.append(_FakeEncodingResult())
        reserved_path = tmp_path / "assets" / "queued-api-2.5.mp4"
        reserved_path.parent.mkdir(parents=True)

        shutdown = threading.Event()
        with test_state.generation.wait_for_generation_slot(shutdown):
            result = test_state.video_generation.generate_local_reserved(
                GenerateVideoRequest.model_validate(_T2V_JSON),
                generation_id="reserved-2.5-api",
                output_path=reserved_path,
                local_model_id="ltx-2.5-22b-distilled",
            )

        assert isinstance(result, GenerateVideoCompleteResponse)
        assert fake_services.text_encoder.encode_calls[0]["api_model"] == spec.api_prompt_embedding_model
        assert "ltx-2.5-22b-distilled" in fake_services.text_encoder.encode_calls[0]["checkpoint_path"]
        assert test_state.state.app_settings.active_ltx_model_id == "ltx-2.3-22b-distilled-1.1"

    def test_ordinary_generate_output_path_stays_under_outputs_dir(
        self, test_state, fake_services, create_fake_model_files
    ):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        result = test_state.video_generation.generate(
            GenerateVideoRequest.model_validate(_T2V_JSON)
        )

        assert isinstance(result, GenerateVideoCompleteResponse)
        assert result.status == "complete"
        video_path = Path(result.video_path)
        assert video_path.exists()
        assert video_path.is_relative_to(test_state.config.outputs_dir)
        assert fake_services.fast_video_pipeline.generate_calls[0]["output_path"] == str(video_path)

    def test_local_fast_generates_six_seconds_at_48_fps(
        self, test_state, fake_services, create_fake_model_files
    ):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        result = test_state.video_generation.generate(
            GenerateVideoRequest.model_validate({**_T2V_JSON, "duration": 6, "fps": 48})
        )

        assert isinstance(result, GenerateVideoCompleteResponse)
        call = fake_services.fast_video_pipeline.generate_calls[0]
        assert call["frame_rate"] == 48
        assert call["num_frames"] == compute_num_frames(6, 48)

    def test_t2v_auto_duration_on_2_5_forwards_envelope_range(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A lighthouse keeper climbs the stairs",
                "resolution": "540p",
                "model": "fast",
                "duration": None,
                "fps": 24,
            },
        )

        assert r.status_code == 200
        call = fake_services.fast_video_pipeline.generate_calls[0]
        assert call["num_frames"] == AutoDurationSpec(min_seconds=2, max_seconds=20)

    def test_t2v_auto_duration_16gb_720p_caps_at_advertised_max(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        test_state.config.vram_gb = 16
        test_state.config.local_generations_mode = "streaming_models_loading"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A lighthouse keeper climbs the stairs",
                "resolution": "720p",
                "model": "fast",
                "duration": None,
                "fps": 24,
            },
        )

        assert r.status_code == 200
        call = fake_services.fast_video_pipeline.generate_calls[0]
        assert call["num_frames"] == AutoDurationSpec(min_seconds=2, max_seconds=10)

    def test_t2v_auto_duration_rejected_on_2_3(
        self, client, test_state, create_fake_model_files
    ):
        _install_local_2_3(test_state, create_fake_model_files)
        _enable_local_text_encoding(test_state)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A lighthouse keeper climbs the stairs",
                "resolution": "540p",
                "model": "fast",
                "duration": None,
                "fps": 24,
            },
        )

        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message="Automatic duration is not supported for local pipeline 'fast'",
        )

    def test_t2v_auto_duration_rejected_without_duration_head(
        self, client, test_state, create_fake_model_files
    ):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        delete_cp_path(test_state.config.default_models_dir, "ltx-2.5-duration-head")

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A lighthouse keeper climbs the stairs",
                "resolution": "540p",
                "model": "fast",
                "duration": None,
                "fps": 24,
            },
        )

        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message="Automatic duration is not supported for local pipeline 'fast'",
        )

    def test_t2v_loras_forwarded_to_pipeline(self, client, test_state, fake_services, create_fake_model_files, create_fake_lora):
        _install_local_2_3(test_state, create_fake_model_files)
        _enable_local_text_encoding(test_state)
        lora_ref = create_fake_lora("style.safetensors")

        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "loras": [{"ref": lora_ref, "scale": 0.8}]},
        )

        assert r.status_code == 200
        pipeline = fake_services.fast_video_pipeline
        assert pipeline.create_loras[-1] == [(lora_ref, 0.8)]

    def test_same_loras_reuse_loaded_pipeline(self, client, test_state, fake_services, create_fake_model_files, create_fake_lora):
        _install_local_2_3(test_state, create_fake_model_files)
        _enable_local_text_encoding(test_state)
        lora_ref = create_fake_lora("a.safetensors")
        # An already-enhanced prompt skips the enhancer, which would otherwise evict the
        # resident video pipeline to claim its VRAM and defeat the reuse this asserts.
        body = {
            **_T2V_JSON,
            "promptProvenance": "enhanced",
            "loras": [{"ref": lora_ref, "scale": 1.0}],
        }

        assert client.post("/api/generate", json=body).status_code == 200
        assert client.post("/api/generate", json=body).status_code == 200

        # Same loras → pipeline built once and reused across both requests.
        assert fake_services.fast_video_pipeline.create_loras == [[(lora_ref, 1.0)]]

    def test_changed_loras_reload_pipeline(self, client, test_state, fake_services, create_fake_model_files, create_fake_lora):
        _install_local_2_3(test_state, create_fake_model_files)
        _enable_local_text_encoding(test_state)
        lora_ref = create_fake_lora("b.safetensors")

        assert client.post("/api/generate", json={**_T2V_JSON}).status_code == 200
        assert client.post(
            "/api/generate",
            json={**_T2V_JSON, "loras": [{"ref": lora_ref, "scale": 0.5}]},
        ).status_code == 200

        # Different loras → pipeline rebuilt (cache miss on the loras key).
        assert fake_services.fast_video_pipeline.create_loras == [
            [],
            [(lora_ref, 0.5)],
        ]

    def test_video_vae_path_change_reloads_pipeline(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        test_state.state.app_settings.use_conv_vae = False

        assert client.post("/api/generate", json=_T2V_JSON).status_code == 200
        assert client.post("/api/generate", json=_T2V_JSON).status_code == 200
        assert fake_services.fast_video_pipeline.create_loras == [[]]

        test_state.state.app_settings.use_conv_vae = True
        assert client.post("/api/generate", json=_T2V_JSON).status_code == 200
        assert fake_services.fast_video_pipeline.create_loras == [[], []]

    def test_t2v_loras_unknown_ref_rejected(self, client, test_state, create_fake_model_files):
        _install_local_2_3(test_state, create_fake_model_files)
        _enable_local_text_encoding(test_state)

        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "loras": [{"ref": "/etc/passwd", "scale": 0.8}]},
        )

        assert r.status_code == 400

    def test_t2v_loras_forwarded_on_2_5(self, client, test_state, fake_services, create_fake_model_files, create_fake_lora):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        lora_ref = create_fake_lora("style.safetensors")

        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "loras": [{"ref": lora_ref, "scale": 0.8}]},
        )

        assert r.status_code == 200
        assert fake_services.fast_video_pipeline.create_loras[-1] == [(lora_ref, 0.8)]

    def test_already_running(self, client, test_state):
        _fake_running_generation_state(test_state)

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 409

    def test_i2v_nonexistent_image(self, client, test_state, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "imagePath": "/no/such/file.png"},
        )
        assert r.status_code == 400

    def test_i2v_rejects_invalid_image_content_400(self, client, test_state, create_fake_model_files, tmp_path):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        bad_image = tmp_path / "bad.png"
        bad_image.write_bytes(b"not-a-real-png")

        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "imagePath": str(bad_image)},
        )
        data = assert_http_error(
            r,
            status_code=400,
            code="HTTP_400",
            message=f"Invalid image file: {bad_image}",
        )
        assert "Invalid image file" in data["message"]

    def test_last_frame_without_first_rejected(self, client):
        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "lastImagePath": "/tmp/last.png"},
        )
        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message="Last frame requires a first-frame image",
        )

    def test_last_frame_with_whitespace_first_rejected(self, client):
        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "imagePath": "  ", "lastImagePath": "/tmp/last.png"},
        )
        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message="Last frame requires a first-frame image",
        )

    def test_last_frame_with_auto_duration_rejected(self, client):
        r = client.post(
            "/api/generate",
            json={
                **_T2V_JSON,
                "duration": None,
                "imagePath": "/tmp/first.png",
                "lastImagePath": "/tmp/last.png",
            },
        )
        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message="Last frame cannot be combined with automatic duration",
        )

    def test_i2v_last_frame_sends_two_conditionings(
        self, client, test_state, fake_services, create_fake_model_files, make_test_image, tmp_path
    ):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        first = tmp_path / "first.png"
        last = tmp_path / "last.png"
        first.write_bytes(make_test_image().getvalue())
        last.write_bytes(make_test_image().getvalue())

        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "imagePath": str(first), "lastImagePath": str(last)},
        )

        assert r.status_code == 200
        call = fake_services.fast_video_pipeline.generate_calls[0]
        images = call["images"]
        assert call["guide_all_images"] is False
        assert len(images) == 2
        assert images[0].frame_idx == 0
        assert images[0].strength == 1.0
        assert images[1].frame_idx == compute_num_frames(5, 24) - 1
        assert images[1].strength == 1.0
        assert images[0].path != images[1].path

    def test_keyframes_cannot_mix_with_first_frame(self, client):
        r = client.post(
            "/api/generate",
            json={
                **_T2V_JSON,
                "imagePath": "/tmp/first.png",
                "keyframes": [{"imagePath": "/tmp/opening.png", "frameIndex": 0}],
            },
        )
        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message="Keyframes cannot be combined with a first or last frame",
        )

    def test_keyframes_cannot_mix_with_audio(self, client):
        r = client.post(
            "/api/generate",
            json={
                **_T2V_JSON,
                "audioPath": "/tmp/audio.wav",
                "keyframes": [{"imagePath": "/tmp/opening.png", "frameIndex": 0}],
            },
        )
        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message="Keyframes cannot be combined with audio-to-video",
        )

    def test_keyframes_cannot_use_auto_duration(self, client):
        r = client.post(
            "/api/generate",
            json={
                **_T2V_JSON,
                "duration": None,
                "keyframes": [{"imagePath": "/tmp/opening.png", "frameIndex": 0}],
            },
        )
        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message="Keyframes cannot be combined with automatic duration",
        )

    def test_keyframes_rejected_on_api_backend(self, client, test_state):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "test",
                "resolution": "1080p",
                "model": "fast",
                "duration": 5,
                "fps": 24,
                "keyframes": [{"imagePath": "/tmp/opening.png", "frameIndex": 0}],
            },
        )
        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message="Multi-keyframe generation is only available for local generation",
        )

    def test_keyframes_cap_is_enforced(self, client):
        r = client.post(
            "/api/generate",
            json={
                **_T2V_JSON,
                "keyframes": [
                    {"imagePath": f"/tmp/kf-{index}.png", "frameIndex": index}
                    for index in range(LOCAL_MULTI_KEYFRAME_MAX_COUNT + 1)
                ],
            },
        )
        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message=f"You can place up to {LOCAL_MULTI_KEYFRAME_MAX_COUNT} keyframes",
        )

    def test_keyframes_cap_allows_exact_limit(self, client):
        r = client.post(
            "/api/generate",
            json={
                **_T2V_JSON,
                "keyframes": [
                    {"imagePath": f"/tmp/kf-{index}.png", "frameIndex": index}
                    for index in range(LOCAL_MULTI_KEYFRAME_MAX_COUNT)
                ],
            },
        )

        assert_http_error(r, status_code=409, code="NO_DOWNLOADED_LTX_MODEL")

    def test_keyframes_send_conditionings_at_requested_frames(
        self, client, test_state, fake_services, create_fake_model_files, make_test_image, tmp_path
    ):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        opening = tmp_path / "opening.png"
        closing = tmp_path / "closing.png"
        opening.write_bytes(make_test_image().getvalue())
        closing.write_bytes(make_test_image().getvalue())

        r = client.post(
            "/api/generate",
            json={
                **_T2V_JSON,
                "keyframes": [
                    {"imagePath": str(opening), "frameIndex": 0, "strength": 1.0},
                    {"imagePath": str(closing), "frameIndex": 80, "strength": 1.0},
                ],
            },
        )

        assert r.status_code == 200
        call = fake_services.fast_video_pipeline.generate_calls[0]
        images = call["images"]
        assert call["guide_all_images"] is True
        assert [(image.frame_idx, image.strength) for image in images] == [(0, 1.0), (80, 1.0)]
        assert images[0].path != images[1].path

    def test_resolution_mapping_540p_on_2_5(self, client, test_state, fake_services, create_fake_model_files):
        # 2.5 540p is legal 16:9 on the /64 two-stage grid (1024×576).
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200

        pipeline = fake_services.fast_video_pipeline
        call = pipeline.generate_calls[0]
        assert call["width"] == 1024
        assert call["height"] == 576

    def test_resolution_mapping_540p_on_2_3(self, client, test_state, fake_services, create_fake_model_files):
        # Same 540p as 2.5: 1024×576 is already on the /64 two-stage grid.
        create_fake_model_files(model_id="ltx-2.3-22b-distilled-1.1")
        test_state.state.app_settings.active_ltx_model_id = "ltx-2.3-22b-distilled-1.1"
        _enable_local_text_encoding(test_state)

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200

        pipeline = fake_services.fast_video_pipeline
        call = pipeline.generate_calls[0]
        assert call["width"] == 1024
        assert call["height"] == 576

    def test_local_resolutions_are_all_on_the_two_stage_grid(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        # Both 2.3 and 2.5 Fast sizes must already be /64. Two-stage halves each
        # dimension onto a /32 latent grid; sizes not divisible by 64 are rejected.
        for model_id in ("ltx-2.5-22b-distilled", "ltx-2.3-22b-distilled-1.1"):
            create_fake_model_files(model_id=model_id)
            test_state.state.app_settings.active_ltx_model_id = model_id
            _enable_local_text_encoding(test_state)

            for resolution in ("270p", "360p", "540p", "720p", "1080p"):
                for aspect_ratio in ("21:9", "16:9", "3:2", "4:3", "1:1", "4:5", "9:16"):
                    fake_services.fast_video_pipeline.generate_calls.clear()
                    r = client.post(
                        "/api/generate",
                        json={**_T2V_JSON, "resolution": resolution, "aspectRatio": aspect_ratio, "duration": 5},
                    )
                    assert r.status_code == 200
                    call = fake_services.fast_video_pipeline.generate_calls[0]
                    assert call["width"] % 64 == 0, f"{model_id} {resolution} {aspect_ratio}: width {call['width']}"
                    assert call["height"] % 64 == 0, f"{model_id} {resolution} {aspect_ratio}: height {call['height']}"
                    assert call["skip_stage_2"] is (resolution in ("270p", "360p"))

    def test_resolution_mapping_720p(self, client, test_state, fake_services, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post("/api/generate", json={**_T2V_JSON, "resolution": "720p"})
        assert r.status_code == 200

        pipeline = fake_services.fast_video_pipeline
        call = pipeline.generate_calls[0]
        assert call["width"] == 1280
        assert call["height"] == 704

    def test_locked_seed(self, client, test_state, fake_services, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        test_state.config.dev_mode = True
        test_state.state.app_settings.seed_locked = True
        test_state.state.app_settings.locked_seed = 123

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200

        pipeline = fake_services.fast_video_pipeline
        assert pipeline.generate_calls[0]["seed"] == 123

    def test_unlocked_seed_is_fresh_for_each_dev_request(
        self, client, test_state, fake_services, create_fake_model_files, monkeypatch
    ):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        test_state.config.dev_mode = True
        resolved_seeds = iter((101, 202))
        monkeypatch.setattr("handlers.base.random.randint", lambda _start, _end: next(resolved_seeds))

        first = client.post("/api/generate", json=_T2V_JSON)
        second = client.post("/api/generate", json=_T2V_JSON)

        assert first.status_code == 200
        assert second.status_code == 200
        seeds = [call["seed"] for call in fake_services.fast_video_pipeline.generate_calls]
        assert len(seeds) == 2
        assert seeds == [101, 202]

    def test_error_sets_generation_error(self, client, test_state, fake_services, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        fake_services.fast_video_pipeline.raise_on_generate = RuntimeError("GPU OOM")

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 500

        progress = test_state.generation.get_generation_progress()
        assert progress.status == "error"

    def test_cancelled_response(self, client, test_state, fake_services, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        fake_services.fast_video_pipeline.raise_on_generate = GenerationCancelledError()

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200
        assert r.json()["status"] == "cancelled"


class TestA2VGenerate:
    def test_a2v_generation_happy_path(self, client, test_state, fake_services, create_fake_model_files, tmp_path):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "resolution": "540p",
                "model": "fast",
                "duration": 5,
                "fps": 24,
                "audioPath": str(audio_file),
            },
        )

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "complete"
        assert data["video_path"] is not None
        assert Path(data["video_path"]).exists()

        pipeline = fake_services.a2v_pipeline
        assert len(pipeline.generate_calls) == 1
        call = pipeline.generate_calls[0]
        assert call["audio_path"] == str(audio_file)
        assert call["audio_start_time"] == 0.0
        # Direct /api/generate keeps legacy behavior: only queued A2V
        # (persisted numFrames + probed audio metadata) passes a finite cap.
        assert call["audio_max_duration"] is None

    def test_a2v_local_inference_emits_heartbeat_done(
        self, client, test_state, create_fake_model_files, tmp_path, caplog
    ):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)
        caplog.set_level(logging.INFO, logger="server_utils.heartbeat")

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "resolution": "540p",
                "model": "fast",
                "duration": 5,
                "fps": 24,
                "audioPath": str(audio_file),
            },
        )

        assert r.status_code == 200
        done_records = [
            record
            for record in caplog.records
            if record.name == "server_utils.heartbeat"
            and "a2v inference done" in record.getMessage()
        ]
        assert len(done_records) == 1

    def test_a2v_loras_forwarded_to_pipeline(self, client, test_state, fake_services, create_fake_model_files, create_fake_lora, tmp_path):
        _install_local_2_3(test_state, create_fake_model_files)
        _enable_local_text_encoding(test_state)
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)
        lora_ref = create_fake_lora("groove.safetensors")

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "resolution": "540p",
                "model": "fast",
                "duration": 5,
                "fps": 24,
                "audioPath": str(audio_file),
                "loras": [{"ref": lora_ref, "scale": 0.7}],
            },
        )

        assert r.status_code == 200
        assert fake_services.a2v_pipeline.create_loras[-1] == [(lora_ref, 0.7)]

    def test_a2v_loras_forwarded_on_2_5(self, client, test_state, fake_services, create_fake_model_files, create_fake_lora, tmp_path):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)
        lora_ref = create_fake_lora("groove.safetensors")

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "resolution": "540p",
                "model": "fast",
                "duration": 5,
                "fps": 24,
                "audioPath": str(audio_file),
                "loras": [{"ref": lora_ref, "scale": 0.7}],
            },
        )

        assert r.status_code == 200
        assert fake_services.a2v_pipeline.create_loras[-1] == [(lora_ref, 0.7)]

    def test_a2v_last_frame_sends_two_conditionings(
        self, client, test_state, fake_services, create_fake_model_files, make_test_image, tmp_path
    ):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)
        first = tmp_path / "first.png"
        last = tmp_path / "last.png"
        first.write_bytes(make_test_image().getvalue())
        last.write_bytes(make_test_image().getvalue())

        r = client.post(
            "/api/generate",
            json={
                **_T2V_JSON,
                "audioPath": str(audio_file),
                "imagePath": str(first),
                "lastImagePath": str(last),
            },
        )

        assert r.status_code == 200
        images = fake_services.a2v_pipeline.generate_calls[0]["images"]
        assert len(images) == 2
        assert images[0].frame_idx == 0
        assert images[1].frame_idx == compute_num_frames(5, 24) - 1

    def test_a2v_rejects_missing_audio_file(self, client, test_state, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "model": "fast",
                "duration": 5,
                "fps": 24,
                "audioPath": "/no/such/audio.wav",
            },
        )
        assert r.status_code == 400

    def test_a2v_rejects_invalid_audio_content_400(self, client, test_state, create_fake_model_files, tmp_path):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        audio_file = tmp_path / "bad.wav"
        audio_file.write_bytes(b"not-a-real-wav")

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "model": "fast",
                "duration": 5,
                "fps": 24,
                "audioPath": str(audio_file),
            },
        )
        data = assert_http_error(
            r,
            status_code=400,
            code="HTTP_400",
            message=f"Invalid audio file: {audio_file}",
        )
        assert "Invalid audio file" in data["message"]

    def test_a2v_forced_api_routes_to_ltx_api(self, client, test_state, fake_services, tmp_path):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "resolution": "1080p",
                "model": "pro",
                "duration": 6,
                "fps": 50,
                "audioPath": str(audio_file),
            },
        )
        assert r.status_code == 200
        assert r.json()["status"] == "complete"
        assert len(fake_services.ltx_api_client.upload_file_calls) == 1
        assert fake_services.ltx_api_client.upload_file_calls[0]["file_path"] == str(audio_file)
        assert len(fake_services.ltx_api_client.audio_to_video_calls) == 1
        call = fake_services.ltx_api_client.audio_to_video_calls[0]
        assert call["audio_uri"] == "storage://uploaded/test_audio.wav"
        assert call["image_uri"] is None
        assert call["model"] == "ltx-2-3-pro"
        assert call["resolution"] == "1920x1080"

    def test_a2v_forced_api_routes_to_ltx_api_for_ltx_2_5_pro(self, client, test_state, fake_services, tmp_path):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "resolution": "1080p",
                "model": "pro-2.5",
                "duration": 6,
                "fps": 50,
                "audioPath": str(audio_file),
            },
        )
        assert r.status_code == 200
        assert r.json()["status"] == "complete"
        assert len(fake_services.ltx_api_client.audio_to_video_calls) == 1
        call = fake_services.ltx_api_client.audio_to_video_calls[0]
        assert call["model"] == "ltx-2-5-pro"

    def test_a2v_forced_api_routes_to_ltx_api_for_ltx_2_5_fast(self, client, test_state, fake_services, tmp_path):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "resolution": "1080p",
                "model": "fast-2.5",
                "duration": 6,
                "fps": 50,
                "audioPath": str(audio_file),
            },
        )
        assert r.status_code == 200
        assert r.json()["status"] == "complete"
        assert len(fake_services.ltx_api_client.audio_to_video_calls) == 1
        call = fake_services.ltx_api_client.audio_to_video_calls[0]
        assert call["model"] == "ltx-2-5-fast"
        assert call["resolution"] == "1920x1080"

    def test_a2v_prefers_api_routes_to_ltx_api(self, client, test_state, fake_services, tmp_path):
        test_state.config.local_generations_mode = "full_models_loading"
        test_state.state.app_settings.user_prefers_ltx_api_video_generations = True
        test_state.state.app_settings.ltx_api_key = "api-key"
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "resolution": "1080p",
                "model": "pro",
                "duration": 6,
                "fps": 50,
                "audioPath": str(audio_file),
            },
        )

        assert r.status_code == 200
        assert r.json()["status"] == "complete"
        assert len(fake_services.ltx_api_client.upload_file_calls) == 1
        assert fake_services.ltx_api_client.upload_file_calls[0]["file_path"] == str(audio_file)
        assert len(fake_services.ltx_api_client.audio_to_video_calls) == 1
        assert len(fake_services.a2v_pipeline.generate_calls) == 0

    def test_a2v_prefers_api_without_key_falls_back_to_local(self, client, test_state, fake_services, create_fake_model_files, tmp_path):
        test_state.config.local_generations_mode = "full_models_loading"
        test_state.state.app_settings.user_prefers_ltx_api_video_generations = True
        test_state.state.app_settings.ltx_api_key = ""
        _enable_local_text_encoding(test_state)
        create_fake_model_files()
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "resolution": "540p",
                "model": "fast",
                "duration": 5,
                "fps": 24,
                "audioPath": str(audio_file),
            },
        )

        assert r.status_code == 200
        assert r.json()["status"] == "complete"
        assert len(fake_services.ltx_api_client.audio_to_video_calls) == 0
        assert len(fake_services.a2v_pipeline.generate_calls) == 1

    def test_a2v_forced_api_routes_to_ltx_api_with_audio_and_image(
        self, client, test_state, fake_services, make_test_image, tmp_path
    ):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)
        image_path = tmp_path / "input.png"
        image_path.write_bytes(make_test_image().getvalue())

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video with a still frame",
                "resolution": "1080p",
                "model": "pro",
                "duration": 6,
                "fps": 50,
                "audioPath": str(audio_file),
                "imagePath": str(image_path),
            },
        )

        assert r.status_code == 200
        assert r.json()["status"] == "complete"
        assert len(fake_services.ltx_api_client.upload_file_calls) == 2
        assert fake_services.ltx_api_client.upload_file_calls[0]["file_path"] == str(audio_file)
        assert fake_services.ltx_api_client.upload_file_calls[1]["file_path"] == str(image_path)
        assert len(fake_services.ltx_api_client.audio_to_video_calls) == 1
        call = fake_services.ltx_api_client.audio_to_video_calls[0]
        assert call["audio_uri"] == "storage://uploaded/test_audio.wav"
        assert call["image_uri"] == "storage://uploaded/input.png"
        assert call["model"] == "ltx-2-3-pro"
        assert call["resolution"] == "1920x1080"

    def test_a2v_forced_api_sends_last_frame_uri(
        self, client, test_state, fake_services, make_test_image, tmp_path
    ):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)
        first = tmp_path / "first.png"
        last = tmp_path / "last.png"
        first.write_bytes(make_test_image().getvalue())
        last.write_bytes(make_test_image().getvalue())

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video with start and end frames",
                "resolution": "1080p",
                "model": "pro",
                "duration": 6,
                "fps": 50,
                "audioPath": str(audio_file),
                "imagePath": str(first),
                "lastImagePath": str(last),
            },
        )

        assert r.status_code == 200
        assert [c["file_path"] for c in fake_services.ltx_api_client.upload_file_calls] == [
            str(audio_file),
            str(first),
            str(last),
        ]
        call = fake_services.ltx_api_client.audio_to_video_calls[0]
        assert call["audio_uri"] == "storage://uploaded/test_audio.wav"
        assert call["image_uri"] == "storage://uploaded/first.png"
        assert call["last_frame_uri"] == "storage://uploaded/last.png"

    def test_a2v_uses_resolution_map_on_2_5(self, client, test_state, fake_services, create_fake_model_files, tmp_path):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        for resolution, expected_w, expected_h in [
            ("270p", 576, 320),
            ("360p", 704, 384),
            ("540p", 1024, 576),
            ("720p", 1280, 704),
            ("1080p", 1920, 1088),
        ]:
            fake_services.a2v_pipeline.generate_calls.clear()
            r = client.post(
                "/api/generate",
                json={
                    "prompt": "A music video",
                    "resolution": resolution,
                    "model": "fast",
                    "duration": 5,
                    "fps": 24,
                    "audioPath": str(audio_file),
                },
            )

            assert r.status_code == 200
            call = fake_services.a2v_pipeline.generate_calls[0]
            assert call["width"] == expected_w, f"{resolution}: expected width {expected_w}, got {call['width']}"
            assert call["height"] == expected_h, f"{resolution}: expected height {expected_h}, got {call['height']}"

    def test_a2v_540p_on_2_3_matches_2_5_pixels(self, client, test_state, fake_services, create_fake_model_files, tmp_path):
        # A2V does not snap; 960×544 used to fail assert_resolution (not ÷64).
        create_fake_model_files(model_id="ltx-2.3-22b-distilled-1.1")
        test_state.state.app_settings.active_ltx_model_id = "ltx-2.3-22b-distilled-1.1"
        _enable_local_text_encoding(test_state)
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "resolution": "540p",
                "model": "fast",
                "duration": 5,
                "fps": 24,
                "audioPath": str(audio_file),
            },
        )

        assert r.status_code == 200
        call = fake_services.a2v_pipeline.generate_calls[0]
        assert call["width"] == 1024
        assert call["height"] == 576

    def test_a2v_forced_api_rejects_missing_audio_file(self, client, test_state):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "resolution": "1080p",
                "model": "pro",
                "duration": 6,
                "fps": 50,
                "audioPath": "/no/such/audio.wav",
            },
        )

        data = assert_http_error(
            r,
            status_code=400,
            code="HTTP_400",
            message="Audio file not found: /no/such/audio.wav",
        )
        assert "Audio file not found" in data["message"]

    def test_a2v_forced_api_missing_key_returns_integrity_error(self, client, test_state, tmp_path):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = ""
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "resolution": "1080p",
                "model": "pro",
                "duration": 6,
                "fps": 50,
                "audioPath": str(audio_file),
            },
        )

        assert_http_error(
            r,
            status_code=422,
            code=LOCAL_GENERATION_UNSUPPORTED,
            message=LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
        )

    def test_a2v_forced_api_cancelled_response(self, client, test_state, fake_services, tmp_path):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        fake_services.ltx_api_client.raise_on_audio_to_video = GenerationCancelledError()
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "resolution": "1080p",
                "model": "pro",
                "duration": 6,
                "fps": 50,
                "audioPath": str(audio_file),
            },
        )

        assert r.status_code == 200
        assert r.json()["status"] == "cancelled"


class TestForcedApiGenerate:
    def test_prefers_api_video_routes_to_ltx_api(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "full_models_loading"
        test_state.state.app_settings.user_prefers_ltx_api_video_generations = True
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A mountain lake",
                "resolution": "1080p",
                "model": "fast",
                "duration": 6,
                "fps": 50,
                "audio": True,
                "cameraMotion": "dolly_in",
            },
        )

        assert r.status_code == 200
        assert r.json()["status"] == "complete"
        assert len(fake_services.ltx_api_client.text_to_video_calls) == 1
        assert len(fake_services.fast_video_pipeline.generate_calls) == 0

    def test_prefers_api_video_without_key_falls_back_to_local(self, client, test_state, fake_services, create_fake_model_files):
        test_state.config.local_generations_mode = "full_models_loading"
        test_state.state.app_settings.user_prefers_ltx_api_video_generations = True
        test_state.state.app_settings.ltx_api_key = ""
        _enable_local_text_encoding(test_state)
        create_fake_model_files()

        r = client.post("/api/generate", json=_T2V_JSON)

        assert r.status_code == 200
        assert r.json()["status"] == "complete"
        assert len(fake_services.ltx_api_client.text_to_video_calls) == 0
        assert len(fake_services.fast_video_pipeline.generate_calls) == 1

    def test_t2v_routes_to_ltx_api(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A mountain lake",
                "resolution": "1080p",
                "model": "fast",
                "duration": 6,
                "fps": 50,
                "audio": True,
                "cameraMotion": "dolly_in",
            },
        )

        assert r.status_code == 200
        assert r.json()["status"] == "complete"
        assert len(fake_services.ltx_api_client.text_to_video_calls) == 1
        call = fake_services.ltx_api_client.text_to_video_calls[0]
        assert call["model"] == "ltx-2-3-fast"
        assert call["resolution"] == "1920x1080"
        assert call["duration"] == 6.0
        assert call["fps"] == 50.0
        assert call["generate_audio"] is True
        assert call["camera_motion"] == "dolly_in"
        assert call["enhance_prompt"] is True

    def test_t2v_sends_enhance_prompt_false_when_the_toggle_is_off(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        test_state.state.app_settings.prompt_enhancer_enabled = False

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A mountain lake",
                "resolution": "1080p",
                "model": "pro-2.5",
                "duration": 6,
                "fps": 24,
            },
        )

        assert r.status_code == 200
        call = fake_services.ltx_api_client.text_to_video_calls[0]
        assert call["enhance_prompt"] is False

    def test_t2v_sends_enhance_prompt_false_for_enhanced_provenance(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "already rewritten",
                "promptProvenance": "enhanced",
                "resolution": "1080p",
                "model": "pro-2.5",
                "duration": 6,
                "fps": 24,
            },
        )

        assert r.status_code == 200
        call = fake_services.ltx_api_client.text_to_video_calls[0]
        assert call["enhance_prompt"] is False

    def test_i2v_sends_enhance_prompt_true_by_default(
        self, client, test_state, fake_services, make_test_image, tmp_path
    ):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        image_path = tmp_path / "input.png"
        image_path.write_bytes(make_test_image().getvalue())

        r = client.post(
            "/api/generate",
            json={
                "prompt": "Animate this frame",
                "resolution": "1080p",
                "model": "pro-2.5",
                "duration": 8,
                "fps": 25,
                "imagePath": str(image_path),
            },
        )

        assert r.status_code == 200
        call = fake_services.ltx_api_client.image_to_video_calls[0]
        assert call["enhance_prompt"] is True

    def test_i2v_sends_enhance_prompt_false_when_the_toggle_is_off(
        self, client, test_state, fake_services, make_test_image, tmp_path
    ):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        test_state.state.app_settings.prompt_enhancer_enabled = False
        image_path = tmp_path / "input.png"
        image_path.write_bytes(make_test_image().getvalue())

        r = client.post(
            "/api/generate",
            json={
                "prompt": "Animate this frame",
                "resolution": "1080p",
                "model": "pro-2.5",
                "duration": 8,
                "fps": 25,
                "imagePath": str(image_path),
            },
        )

        assert r.status_code == 200
        call = fake_services.ltx_api_client.image_to_video_calls[0]
        assert call["enhance_prompt"] is False

    def test_a2v_with_start_frame_sends_enhance_prompt_from_the_single_toggle(
        self, client, test_state, fake_services, make_test_image, tmp_path
    ):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        test_state.state.app_settings.prompt_enhancer_enabled = False
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)
        image_path = tmp_path / "input.png"
        image_path.write_bytes(make_test_image().getvalue())

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video with a still frame",
                "resolution": "1080p",
                "model": "pro",
                "duration": 6,
                "fps": 50,
                "audioPath": str(audio_file),
                "imagePath": str(image_path),
            },
        )

        assert r.status_code == 200
        call = fake_services.ltx_api_client.audio_to_video_calls[0]
        assert call["enhance_prompt"] is False

    def test_t2v_routes_to_ltx_api_for_ltx_2_5_fast(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A mountain lake",
                "resolution": "1080p",
                "model": "fast-2.5",
                "duration": 6,
                "fps": 50,
                "audio": True,
                "cameraMotion": "dolly_in",
            },
        )

        assert r.status_code == 200
        assert r.json()["status"] == "complete"
        assert len(fake_services.ltx_api_client.text_to_video_calls) == 1
        call = fake_services.ltx_api_client.text_to_video_calls[0]
        assert call["model"] == "ltx-2-5-fast"
        assert call["resolution"] == "1920x1080"
        assert call["duration"] == 6.0
        assert call["fps"] == 50.0
        assert call["generate_audio"] is True
        assert call["camera_motion"] == "dolly_in"

    def test_t2v_auto_duration_sends_null_for_ltx_2_5_fast(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A lighthouse keeper climbs the stairs",
                "resolution": "1080p",
                "model": "fast-2.5",
                "duration": None,
                "fps": 24,
                "audio": True,
            },
        )

        assert r.status_code == 200
        call = fake_services.ltx_api_client.text_to_video_calls[0]
        assert call["model"] == "ltx-2-5-fast"
        assert call["duration"] is None

    def test_api_auto_duration_does_not_require_local_duration_head(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        create_fake_model_files()
        delete_cp_path(test_state.config.default_models_dir, "ltx-2.5-duration-head")
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A lighthouse keeper climbs the stairs",
                "resolution": "1080p",
                "model": "fast-2.5",
                "duration": None,
                "fps": 24,
            },
        )

        assert r.status_code == 200
        call = fake_services.ltx_api_client.text_to_video_calls[0]
        assert call["model"] == "ltx-2-5-fast"
        assert call["duration"] is None

    def test_t2v_auto_duration_rejected_for_ltx_2_3_fast(self, client, test_state):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A lighthouse keeper climbs the stairs",
                "resolution": "1080p",
                "model": "fast",
                "duration": None,
                "fps": 24,
            },
        )

        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message="Automatic duration is not supported for api pipeline 'fast'",
        )

    def test_a2v_rejects_auto_duration(self, client, test_state, tmp_path):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "resolution": "1080p",
                "model": "fast-2.5",
                "duration": None,
                "fps": 24,
                "audioPath": str(audio_file),
            },
        )

        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message="Automatic duration cannot be combined with audio-to-video",
        )

    def test_i2v_routes_to_ltx_api_for_ltx_2_5_pro(self, client, test_state, fake_services, make_test_image, tmp_path):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        image_path = tmp_path / "input.png"
        image_path.write_bytes(make_test_image().getvalue())

        r = client.post(
            "/api/generate",
            json={
                "prompt": "Animate this frame",
                "resolution": "1080p",
                "model": "pro-2.5",
                "duration": 8,
                "fps": 25,
                "audio": False,
                "cameraMotion": "jib_up",
                "imagePath": str(image_path),
            },
        )

        assert r.status_code == 200
        assert r.json()["status"] == "complete"
        assert len(fake_services.ltx_api_client.upload_file_calls) == 1
        assert fake_services.ltx_api_client.upload_file_calls[0]["file_path"] == str(image_path)
        assert len(fake_services.ltx_api_client.image_to_video_calls) == 1
        call = fake_services.ltx_api_client.image_to_video_calls[0]
        assert call["image_uri"] == "storage://uploaded/input.png"
        assert call["model"] == "ltx-2-5-pro"
        assert call["resolution"] == "1920x1080"
        assert call["duration"] == 8.0
        assert call["enhance_prompt"] is True
        assert call["fps"] == 25.0
        assert call["camera_motion"] == "jib_up"

    def test_i2v_routes_to_ltx_api(self, client, test_state, fake_services, make_test_image, tmp_path):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        image_path = tmp_path / "input.png"
        image_path.write_bytes(make_test_image().getvalue())

        r = client.post(
            "/api/generate",
            json={
                "prompt": "Animate this frame",
                "resolution": "2160p",
                "model": "pro",
                "duration": 8,
                "fps": 25,
                "audio": False,
                "cameraMotion": "jib_up",
                "imagePath": str(image_path),
            },
        )

        assert r.status_code == 200
        assert r.json()["status"] == "complete"
        assert len(fake_services.ltx_api_client.upload_file_calls) == 1
        assert fake_services.ltx_api_client.upload_file_calls[0]["file_path"] == str(image_path)
        assert len(fake_services.ltx_api_client.image_to_video_calls) == 1
        call = fake_services.ltx_api_client.image_to_video_calls[0]
        assert call["image_uri"] == "storage://uploaded/input.png"
        assert call["model"] == "ltx-2-3-pro"
        assert call["resolution"] == "3840x2160"
        assert call["duration"] == 8.0
        assert call["fps"] == 25.0
        assert call["camera_motion"] == "jib_up"

    def test_camera_motion_none_maps_to_none_for_t2v(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A mountain lake",
                "resolution": "1080p",
                "model": "fast",
                "duration": 6,
                "fps": 50,
                "audio": True,
                "cameraMotion": "none",
            },
        )

        assert r.status_code == 200
        assert len(fake_services.ltx_api_client.text_to_video_calls) == 1
        call = fake_services.ltx_api_client.text_to_video_calls[0]
        assert call["camera_motion"] == "none"

    def test_camera_motion_none_maps_to_none_for_i2v(self, client, test_state, fake_services, make_test_image, tmp_path):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        image_path = tmp_path / "input-none.png"
        image_path.write_bytes(make_test_image().getvalue())

        r = client.post(
            "/api/generate",
            json={
                "prompt": "Animate this frame",
                "resolution": "2160p",
                "model": "pro",
                "duration": 8,
                "fps": 25,
                "audio": False,
                "cameraMotion": "none",
                "imagePath": str(image_path),
            },
        )

        assert r.status_code == 200
        assert len(fake_services.ltx_api_client.upload_file_calls) == 1
        assert fake_services.ltx_api_client.upload_file_calls[0]["file_path"] == str(image_path)
        assert len(fake_services.ltx_api_client.image_to_video_calls) == 1
        call = fake_services.ltx_api_client.image_to_video_calls[0]
        assert call["image_uri"] == "storage://uploaded/input-none.png"
        assert call["camera_motion"] == "none"

    def test_i2v_fast_routes_to_fast_model(self, client, test_state, fake_services, make_test_image, tmp_path):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        image_path = tmp_path / "input-fast.png"
        image_path.write_bytes(make_test_image().getvalue())

        r = client.post(
            "/api/generate",
            json={
                "prompt": "Animate this frame quickly",
                "resolution": "1080p",
                "model": "fast",
                "duration": 6,
                "fps": 25,
                "audio": False,
                "imagePath": str(image_path),
            },
        )

        assert r.status_code == 200
        assert r.json()["status"] == "complete"
        assert len(fake_services.ltx_api_client.upload_file_calls) == 1
        assert fake_services.ltx_api_client.upload_file_calls[0]["file_path"] == str(image_path)
        assert len(fake_services.ltx_api_client.image_to_video_calls) == 1
        call = fake_services.ltx_api_client.image_to_video_calls[0]
        assert call["image_uri"] == "storage://uploaded/input-fast.png"
        assert call["model"] == "ltx-2-3-fast"
        assert call["resolution"] == "1920x1080"
        assert call["duration"] == 6.0
        assert call["fps"] == 25.0

    def test_i2v_sends_last_frame_uri(self, client, test_state, fake_services, make_test_image, tmp_path):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        first = tmp_path / "first.png"
        last = tmp_path / "last.png"
        first.write_bytes(make_test_image().getvalue())
        last.write_bytes(make_test_image().getvalue())

        r = client.post(
            "/api/generate",
            json={
                "prompt": "Animate from first to last",
                "resolution": "1080p",
                "model": "fast",
                "duration": 6,
                "fps": 25,
                "audio": False,
                "imagePath": str(first),
                "lastImagePath": str(last),
            },
        )

        assert r.status_code == 200
        assert [c["file_path"] for c in fake_services.ltx_api_client.upload_file_calls] == [
            str(first),
            str(last),
        ]
        call = fake_services.ltx_api_client.image_to_video_calls[0]
        assert call["image_uri"] == "storage://uploaded/first.png"
        assert call["last_frame_uri"] == "storage://uploaded/last.png"
        assert call["model"] == "ltx-2-3-fast"

    def test_invalid_forced_model_rejected(self, client, test_state):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A city skyline",
                "resolution": "1080p",
                "model": "ultra",
                "duration": 6,
                "fps": 25,
                "audio": False,
            },
        )

        assert r.status_code == 422

    def test_missing_api_key_returns_integrity_error(self, client, test_state):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = ""

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A city skyline",
                "resolution": "1080p",
                "model": "pro",
                "duration": 6,
                "fps": 25,
                "audio": False,
            },
        )

        assert_http_error(
            r,
            status_code=422,
            code=LOCAL_GENERATION_UNSUPPORTED,
            message=LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
        )

    def test_invalid_forced_resolution_rejected(self, client, test_state):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A city skyline",
                "resolution": "540p",
                "model": "pro",
                "duration": 6,
                "fps": 25,
                "audio": False,
            },
        )

        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message="Unsupported api text-to-video resolution '540p' for pipeline 'pro'",
        )

    def test_pro_2_5_rejects_4k_resolution(self, client, test_state):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A city skyline",
                "resolution": "2160p",
                "model": "pro-2.5",
                "duration": 6,
                "fps": 25,
                "audio": False,
            },
        )

        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message="Unsupported api text-to-video resolution '2160p' for pipeline 'pro-2.5'",
        )

    def test_pro_2_5_rejects_48_fps(self, client, test_state):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A city skyline",
                "resolution": "1080p",
                "model": "pro-2.5",
                "duration": 6,
                "fps": 48,
                "audio": False,
            },
        )

        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message="Unsupported api text-to-video fps '48' for pipeline 'pro-2.5' at resolution '1080p'",
        )

    def test_invalid_forced_duration_rejected(self, client, test_state):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A city skyline",
                "resolution": "1080p",
                "model": "pro",
                "duration": 12,
                "fps": 25,
                "audio": False,
            },
        )

        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message="Unsupported api text-to-video duration '12' for pipeline 'pro' at resolution '1080p' and fps '25'",
        )

    def test_forced_api_a2v_rejects_fast_tier_pipeline(self, client, test_state, tmp_path):
        # ltxv-api audio-to-video does not accept ltx-2-3-fast. Reject pipeline "fast"
        # here rather than a downstream 400. (ltx-2-5-fast does accept A2V.)
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "resolution": "1080p",
                "model": "fast",
                "duration": 6,
                "fps": 24,
                "audioPath": str(audio_file),
            },
        )

        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message="Unsupported api audio-to-video resolution '1080p' for pipeline 'fast'",
        )

    def test_invalid_forced_fps_rejected(self, client, test_state):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A city skyline",
                "resolution": "1080p",
                "model": "pro",
                "duration": 6,
                "fps": 30,
                "audio": False,
            },
        )

        assert r.status_code == 422

    def test_forced_api_surfaces_insufficient_funds_as_custom_402(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        fake_services.ltx_api_client.raise_on_text_to_video = LTXAPIClientError(
            402,
            'LTX API generation failed (402): {"type":"error","error":{"type":"insufficient_funds_error","message":"Insufficient funds. Required: 36 cents"}}',
            provider_error_type="insufficient_funds_error",
            provider_message="Insufficient funds. Required: 36 cents",
            request_id="req-123",
        )

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A city skyline",
                "resolution": "1080p",
                "model": "pro",
                "duration": 6,
                "fps": 25,
                "audio": False,
            },
        )

        assert_http_error(
            r,
            status_code=402,
            code="LTX_INSUFFICIENT_FUNDS",
            message="Your LTX API credits are insufficient for this generation. Buy more credits and try again.",
        )

        progress = test_state.generation.get_generation_progress()
        assert progress.status == "error"
        assert progress.phase == "error"

    def test_forced_api_maps_rejected_key_to_one_code(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        fake_services.ltx_api_client.raise_on_text_to_video = LTXAPIClientError(
            401,
            "LTX API generation failed (401): unauthorized",
        )

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A city skyline",
                "resolution": "1080p",
                "model": "pro",
                "duration": 6,
                "fps": 25,
                "audio": False,
            },
        )

        assert_http_error(
            r,
            status_code=401,
            code="LTX_INVALID_API_KEY",
            message="This LTX API key isn’t valid.",
        )

    def test_invalid_camera_motion_rejected_with_422(self, client, test_state):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A city skyline",
                "resolution": "1080p",
                "model": "pro",
                "duration": 6,
                "fps": 25,
                "audio": False,
                "cameraMotion": "orbit",
            },
        )

        assert r.status_code == 422

    def test_forced_api_cancelled_response(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        fake_services.ltx_api_client.raise_on_text_to_video = GenerationCancelledError()

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A mountain lake",
                "resolution": "1080p",
                "model": "pro",
                "duration": 6,
                "fps": 25,
                "audio": False,
            },
        )

        assert r.status_code == 200
        assert r.json()["status"] == "cancelled"

    def test_portrait_resolution_1080p(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A portrait video",
                "resolution": "1080p",
                "model": "fast",
                "duration": 6,
                "fps": 25,
                "aspectRatio": "9:16",
            },
        )

        assert r.status_code == 200
        call = fake_services.ltx_api_client.text_to_video_calls[0]
        assert call["resolution"] == "1080x1920"

    def test_portrait_resolution_720p(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A portrait video",
                "resolution": "720p",
                "model": "fast",
                "duration": 6,
                "fps": 25,
                "aspectRatio": "9:16",
            },
        )

        assert r.status_code == 200
        call = fake_services.ltx_api_client.text_to_video_calls[0]
        assert call["resolution"] == "720x1280"

    def test_landscape_resolution_720p(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A landscape video",
                "resolution": "720p",
                "model": "pro-2.5",
                "duration": 6,
                "fps": 25,
            },
        )

        assert r.status_code == 200
        call = fake_services.ltx_api_client.text_to_video_calls[0]
        assert call["resolution"] == "1280x720"
        assert call["model"] == "ltx-2-5-pro"

    def test_portrait_resolution_1440p(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A portrait video",
                "resolution": "1440p",
                "model": "fast",
                "duration": 6,
                "fps": 25,
                "aspectRatio": "9:16",
            },
        )

        assert r.status_code == 200
        call = fake_services.ltx_api_client.text_to_video_calls[0]
        assert call["resolution"] == "1440x2560"

    def test_portrait_resolution_4k(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A portrait video",
                "resolution": "2160p",
                "model": "pro",
                "duration": 6,
                "fps": 25,
                "aspectRatio": "9:16",
            },
        )

        assert r.status_code == 200
        call = fake_services.ltx_api_client.text_to_video_calls[0]
        assert call["resolution"] == "2160x3840"

    def test_default_landscape_when_aspect_ratio_omitted(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A landscape video",
                "resolution": "1080p",
                "model": "fast",
                "duration": 6,
                "fps": 25,
            },
        )

        assert r.status_code == 200
        call = fake_services.ltx_api_client.text_to_video_calls[0]
        assert call["resolution"] == "1920x1080"

    def test_invalid_aspect_ratio_rejected(self, client, test_state):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A video",
                "resolution": "1080p",
                "model": "fast",
                "duration": 6,
                "fps": 25,
                "aspectRatio": "4:3",
            },
        )

        assert r.status_code == 422

    def test_extended_durations_for_fast_1080p_24fps(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A long video",
                "resolution": "1080p",
                "model": "fast",
                "duration": 20,
                "fps": 24,
            },
        )

        assert r.status_code == 200
        call = fake_services.ltx_api_client.text_to_video_calls[0]
        assert call["duration"] == 20.0

    def test_extended_duration_rejected_for_pro_1080p_24fps(self, client, test_state):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A long video",
                "resolution": "1080p",
                "model": "pro",
                "duration": 20,
                "fps": 24,
            },
        )

        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message="Unsupported api text-to-video duration '20' for pipeline 'pro' at resolution '1080p' and fps '24'",
        )

    def test_extended_duration_rejected_for_fast_1440p_24fps(self, client, test_state):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A long video",
                "resolution": "1440p",
                "model": "fast",
                "duration": 20,
                "fps": 24,
            },
        )

        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message="Unsupported api text-to-video duration '20' for pipeline 'fast' at resolution '1440p' and fps '24'",
        )

    def test_fps_24_accepted(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A video",
                "resolution": "1080p",
                "model": "fast",
                "duration": 6,
                "fps": 24,
            },
        )

        assert r.status_code == 200
        call = fake_services.ltx_api_client.text_to_video_calls[0]
        assert call["fps"] == 24.0

    def test_fps_48_accepted(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A video",
                "resolution": "1080p",
                "model": "fast",
                "duration": 6,
                "fps": 48,
            },
        )

        assert r.status_code == 200
        call = fake_services.ltx_api_client.text_to_video_calls[0]
        assert call["fps"] == 48.0

    def test_a2v_portrait_resolution(self, client, test_state, fake_services, tmp_path):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A portrait music video",
                "resolution": "1080p",
                "model": "pro",
                "duration": 6,
                "fps": 25,
                "audioPath": str(audio_file),
                "aspectRatio": "9:16",
            },
        )

        assert r.status_code == 200
        call = fake_services.ltx_api_client.audio_to_video_calls[0]
        assert call["resolution"] == "1080x1920"

    def test_a2v_720p_routes_to_ltx_api(self, client, test_state, fake_services, tmp_path):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "resolution": "720p",
                "model": "pro",
                "duration": 6,
                "fps": 25,
                "audioPath": str(audio_file),
            },
        )

        assert r.status_code == 200
        call = fake_services.ltx_api_client.audio_to_video_calls[0]
        assert call["resolution"] == "1280x720"

    def test_a2v_1440p_capped_at_10s(self, client, test_state, tmp_path):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "resolution": "1440p",
                "model": "pro",
                "duration": 20,
                "fps": 24,
                "audioPath": str(audio_file),
            },
        )

        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message="Unsupported api audio-to-video duration '20' for pipeline 'pro' at resolution '1440p' and fps '24'",
        )

    def test_a2v_1440p_10s_routes_to_ltx_api(self, client, test_state, fake_services, tmp_path):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "resolution": "1440p",
                "model": "pro",
                "duration": 10,
                "fps": 24,
                "audioPath": str(audio_file),
            },
        )

        assert r.status_code == 200
        call = fake_services.ltx_api_client.audio_to_video_calls[0]
        assert call["resolution"] == "2560x1440"

    def test_a2v_pro_2_5_rejects_20s(self, client, test_state, tmp_path):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "resolution": "1080p",
                "model": "pro-2.5",
                "duration": 20,
                "fps": 24,
                "audioPath": str(audio_file),
            },
        )

        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message="Unsupported api audio-to-video duration '20' for pipeline 'pro-2.5' at resolution '1080p' and fps '24'",
        )

    def test_a2v_forced_api_rejects_fast(self, client, test_state, fake_services, tmp_path):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "test_key"
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A big video",
                "resolution": "2160p",
                "model": "fast",
                "duration": 6,
                "fps": 25,
                "audioPath": str(audio_file),
                "aspectRatio": "9:16",
            },
        )

        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message="Unsupported api audio-to-video resolution '2160p' for pipeline 'fast'",
        )

    def test_a2v_forced_api_passes_through_model_and_aspect(self, client, test_state, fake_services, tmp_path):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "test_key"
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A portrait music video",
                "resolution": "1080p",
                "model": "pro",
                "duration": 6,
                "fps": 25,
                "audioPath": str(audio_file),
                "aspectRatio": "9:16",
            },
        )

        assert r.status_code == 200
        assert r.json()["status"] == "complete"
        call = fake_services.ltx_api_client.audio_to_video_calls[0]
        assert call["resolution"] == "1080x1920"
        assert call["model"] == "ltx-2-3-pro"


class TestVideoJobLoadMode:
    def test_540p_5s_stays_full(self, client, test_state, fake_services, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        assert test_state.config.local_generations_mode == "full_models_loading"

        r = client.post("/api/generate", json=_T2V_JSON)

        assert r.status_code == 200
        assert fake_services.fast_video_pipeline.last_streaming_prefetch_count is None
        assert test_state.config.local_generations_mode == "full_models_loading"

    def test_720p_20s_streams_without_flipping_process_mode(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "resolution": "720p", "duration": 20},
        )

        assert r.status_code == 200
        assert fake_services.fast_video_pipeline.last_streaming_prefetch_count == 2
        assert test_state.config.local_generations_mode == "full_models_loading"

    def test_1080p_10s_streams(self, client, test_state, fake_services, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "resolution": "1080p", "duration": 10},
        )

        assert r.status_code == 200
        assert fake_services.fast_video_pipeline.last_streaming_prefetch_count == 2
        assert test_state.config.local_generations_mode == "full_models_loading"

    def test_stream_only_process_never_full(self, client, test_state, fake_services, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        test_state.config.local_generations_mode = "streaming_models_loading"
        test_state.config.vram_gb = 24

        r = client.post("/api/generate", json=_T2V_JSON)

        assert r.status_code == 200
        assert fake_services.fast_video_pipeline.last_streaming_prefetch_count == 2

    def test_darwin_never_full(self, client, test_state, fake_services, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        test_state.config.darwin_unified_memory = True
        test_state.config.available_ram_gb = 128

        r = client.post("/api/generate", json=_T2V_JSON)

        assert r.status_code == 200
        assert fake_services.fast_video_pipeline.last_streaming_prefetch_count == 2
        assert test_state.config.local_generations_mode == "full_models_loading"

    def test_darwin_m4_48gb_does_not_422_720p_20s(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        test_state.config.darwin_unified_memory = True
        # Free RAM on a 48 GB Mac, not total. Pre-fix 0.4× CUDA curve 422'd this job.
        test_state.config.available_ram_gb = 20

        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "resolution": "720p", "duration": 20},
        )

        assert r.status_code == 200
        assert fake_services.fast_video_pipeline.last_streaming_prefetch_count == 2

    def test_darwin_m4_48gb_does_not_422_1080p_10s(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        test_state.config.darwin_unified_memory = True
        test_state.config.available_ram_gb = 20

        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "resolution": "1080p", "duration": 10},
        )

        assert r.status_code == 200
        assert fake_services.fast_video_pipeline.last_streaming_prefetch_count == 2

    def test_cache_rebuilds_when_job_switches_to_stream(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        assert client.post("/api/generate", json=_T2V_JSON).status_code == 200
        assert fake_services.fast_video_pipeline.last_streaming_prefetch_count is None
        first_creates = fake_services.fast_video_pipeline.create_count

        assert client.post(
            "/api/generate",
            json={**_T2V_JSON, "resolution": "720p", "duration": 20},
        ).status_code == 200
        assert fake_services.fast_video_pipeline.last_streaming_prefetch_count == 2
        assert fake_services.fast_video_pipeline.create_count == first_creates + 1

    def test_a2v_720p_20s_streams(self, client, test_state, fake_services, create_fake_model_files, tmp_path):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                **_T2V_JSON,
                "resolution": "720p",
                "duration": 20,
                "audioPath": str(audio_file),
            },
        )

        assert r.status_code == 200
        assert fake_services.a2v_pipeline.last_streaming_prefetch_count == 2
        assert test_state.config.local_generations_mode == "full_models_loading"

    def test_job_too_large_returns_422(self, client, test_state, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        test_state.config.vram_gb = 8

        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "resolution": "720p", "duration": 20},
        )
        assert_http_error(
            r,
            status_code=422,
            code=VIDEO_JOB_TOO_LARGE,
            message=VIDEO_JOB_TOO_LARGE_MESSAGE,
        )

    def test_360p_20s_on_8gb_budgets_the_half_canvas(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        # 360p is the 720p canvas with stage 2 skipped. 720p/20s 422s on 8 GB;
        # budgeting the parent canvas would 422 this job too.
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        test_state.config.vram_gb = 8

        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "resolution": "360p", "duration": 20},
        )

        assert r.status_code == 200
        call = fake_services.fast_video_pipeline.generate_calls[0]
        assert call["width"] == 1280
        assert call["height"] == 704
        assert call["skip_stage_2"] is True
        assert fake_services.fast_video_pipeline.last_streaming_prefetch_count == 2

    def test_16gb_540p_20s_21_9_returns_422_while_16_9_runs(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        # 1024×576 estimates 14.17 GiB and fits. 1344×576 estimates 17.63 GiB.
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        test_state.config.vram_gb = 16
        test_state.config.local_generations_mode = "streaming_models_loading"

        wide = client.post(
            "/api/generate",
            json={**_T2V_JSON, "resolution": "540p", "duration": 20, "aspectRatio": "21:9"},
        )
        assert_http_error(
            wide,
            status_code=422,
            code=VIDEO_JOB_TOO_LARGE,
            message=VIDEO_JOB_TOO_LARGE_MESSAGE,
        )
        assert fake_services.fast_video_pipeline.generate_calls == []

        landscape = client.post(
            "/api/generate",
            json={**_T2V_JSON, "resolution": "540p", "duration": 20, "aspectRatio": "16:9"},
        )
        assert landscape.status_code == 200

    def test_16gb_720p_20s_returns_422(self, client, test_state, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        test_state.config.vram_gb = 16
        test_state.config.local_generations_mode = "streaming_models_loading"

        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "resolution": "720p", "duration": 20},
        )
        assert_http_error(
            r,
            status_code=422,
            code=VIDEO_JOB_TOO_LARGE,
            message=VIDEO_JOB_TOO_LARGE_MESSAGE,
        )

    def test_job_too_large_does_not_run_local_enhance(
        self, client, test_state, create_fake_model_files
    ):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        test_state.config.vram_gb = 16
        test_state.config.local_generations_mode = "streaming_models_loading"
        calls = {"n": 0}
        original = test_state.prompt_enhancement.enhance_for_generation

        def spy(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        test_state.prompt_enhancement.enhance_for_generation = spy  # type: ignore[method-assign]
        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "resolution": "720p", "duration": 20},
        )
        assert_http_error(
            r,
            status_code=422,
            code=VIDEO_JOB_TOO_LARGE,
            message=VIDEO_JOB_TOO_LARGE_MESSAGE,
        )
        assert calls["n"] == 0

    def test_a2v_job_too_large_does_not_run_local_enhance(
        self, client, test_state, create_fake_model_files, tmp_path
    ):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        test_state.config.vram_gb = 16
        test_state.config.local_generations_mode = "streaming_models_loading"
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)
        calls = {"n": 0}
        original = test_state.prompt_enhancement.enhance_for_generation

        def spy(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        test_state.prompt_enhancement.enhance_for_generation = spy  # type: ignore[method-assign]
        r = client.post(
            "/api/generate",
            json={
                **_T2V_JSON,
                "resolution": "720p",
                "duration": 20,
                "audioPath": str(audio_file),
            },
        )
        assert_http_error(
            r,
            status_code=422,
            code=VIDEO_JOB_TOO_LARGE,
            message=VIDEO_JOB_TOO_LARGE_MESSAGE,
        )
        assert calls["n"] == 0

    def test_generate_local_reserved_unsupported_is_not_job_too_large(
        self, test_state, create_fake_model_files, tmp_path
    ):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        test_state.config.local_generations_mode = "unsupported"
        reserved_path = tmp_path / "assets" / "queued.mp4"
        reserved_path.parent.mkdir(parents=True)
        shutdown = threading.Event()
        with test_state.generation.wait_for_generation_slot(shutdown):
            with pytest.raises(HTTPError) as exc:
                test_state.video_generation.generate_local_reserved(
                    GenerateVideoRequest.model_validate(_T2V_JSON),
                    generation_id="reserved-unsupported",
                    output_path=reserved_path,
                    local_model_id="ltx-2.5-22b-distilled",
                )
        assert exc.value.code == LOCAL_GENERATION_UNSUPPORTED
        assert exc.value.detail == LOCAL_GENERATION_UNSUPPORTED_MESSAGE
        assert "api" not in exc.value.detail.lower()
        assert "resolution" not in exc.value.detail.lower()

    def test_generate_local_reserved_auto_duration_unsupported_is_not_a_crash(
        self, test_state, create_fake_model_files, tmp_path
    ):
        """Auto reads the filtered local specs, which unsupported mode empties."""
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        test_state.config.local_generations_mode = "unsupported"
        reserved_path = tmp_path / "assets" / "queued-auto.mp4"
        reserved_path.parent.mkdir(parents=True)
        shutdown = threading.Event()
        with test_state.generation.wait_for_generation_slot(shutdown):
            with pytest.raises(HTTPError) as exc:
                test_state.video_generation.generate_local_reserved(
                    GenerateVideoRequest.model_validate({**_T2V_JSON, "duration": None}),
                    generation_id="reserved-unsupported-auto",
                    output_path=reserved_path,
                    local_model_id="ltx-2.5-22b-distilled",
                )
        assert exc.value.status_code == 422
        assert exc.value.code == LOCAL_GENERATION_UNSUPPORTED

    def test_generate_local_reserved_unsupported_is_not_rescued_by_an_ltx_key(
        self, test_state, create_fake_model_files, tmp_path
    ):
        """This entry point never routes to the API, so a key cannot make it viable."""
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "ltx-key"
        reserved_path = tmp_path / "assets" / "queued-keyed.mp4"
        reserved_path.parent.mkdir(parents=True)
        shutdown = threading.Event()
        with test_state.generation.wait_for_generation_slot(shutdown):
            with pytest.raises(HTTPError) as exc:
                test_state.video_generation.generate_local_reserved(
                    GenerateVideoRequest.model_validate(_T2V_JSON),
                    generation_id="reserved-unsupported-keyed",
                    output_path=reserved_path,
                    local_model_id="ltx-2.5-22b-distilled",
                )
        assert exc.value.code == LOCAL_GENERATION_UNSUPPORTED


class TestGenerateCancel:
    def test_cancel_active(self, client, test_state):
        _fake_running_generation_state(test_state)

        r = client.post("/api/generate/cancel")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "cancelling"
        assert generation_interrupt.is_requested()

        idle = client.post("/api/generate/cancel")
        assert idle.status_code == 200
        assert idle.json()["status"] == "no_active_generation"

    def test_cancel_no_active(self, client):
        r = client.post("/api/generate/cancel")
        assert r.status_code == 200
        assert r.json()["status"] == "no_active_generation"
        assert not generation_interrupt.is_requested()

    def test_in_flight_cancel_stops_before_remaining_steps(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        pipeline = fake_services.fast_video_pipeline
        pipeline.inference_steps = 12
        pipeline.step_delay_s = 0.05

        response = _cancel_in_flight(
            client,
            pipeline.entered_inference,
            lambda: test_state.video_generation.generate(GenerateVideoRequest.model_validate(_T2V_JSON)),
        )

        assert response.status == "cancelled"
        assert pipeline.steps_completed < pipeline.inference_steps
        assert pipeline.generate_calls == []
        assert test_state.state.gpu_slot is not None

    def test_generate_after_user_stop_is_not_immediately_cancelled(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        pipeline = fake_services.fast_video_pipeline
        pipeline.inference_steps = 8
        pipeline.step_delay_s = 0.02

        cancelled = _cancel_in_flight(
            client,
            pipeline.entered_inference,
            lambda: test_state.video_generation.generate(
                GenerateVideoRequest.model_validate(_T2V_JSON)
            ),
        )
        assert cancelled.status == "cancelled"

        pipeline.inference_steps = 1
        pipeline.step_delay_s = 0
        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200
        assert r.json()["status"] == "complete"

    def test_second_generate_409s_until_cancelled_job_unwinds(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        pipeline = fake_services.fast_video_pipeline
        pipeline.inference_steps = 20
        pipeline.step_delay_s = 0.05

        result: dict[str, object] = {}

        def run() -> None:
            result["value"] = test_state.video_generation.generate(
                GenerateVideoRequest.model_validate(_T2V_JSON)
            )

        thread = threading.Thread(target=run)
        thread.start()
        assert pipeline.entered_inference.wait(timeout=5.0)
        cancel = client.post("/api/generate/cancel")
        assert cancel.status_code == 200
        assert cancel.json()["status"] == "cancelling"

        with pytest.raises(HTTPError) as exc_info:
            test_state.video_generation.generate(GenerateVideoRequest.model_validate(_T2V_JSON))
        assert exc_info.value.status_code == 409

        thread.join(timeout=8.0)
        assert not thread.is_alive()
        assert result["value"].status == "cancelled"  # type: ignore[union-attr]
        idle = client.post("/api/generate/cancel")
        assert idle.status_code == 200
        assert idle.json()["status"] == "no_active_generation"
        assert test_state.generation.try_reserve_generation_start() is True
        test_state.generation.release_generation_start_reservation()


class TestGenerateModelSpecs:
    def test_models_specs_endpoint_returns_ordered_backend_specs(self, client):
        r = client.get("/api/generate/models-specs")

        assert r.status_code == 200
        data = r.json()
        assert [item["pipeline"] for item in data["local_models"]] == ["fast"]
        assert data["local_models"][0]["spec"]["display_name"] == "LTX 2.5 Fast"
        assert data["downloaded_local_models"] == []
        local_t2v = data["local_models"][0]["spec"]["supported_resolutions_durations"]
        assert list(local_t2v.keys()) == ["270p", "360p", "540p", "720p", "1080p"]
        local_a2v = data["local_models"][0]["spec"]["a2v_supported_resolutions_durations"]
        assert list(local_a2v.keys()) == ["270p", "360p", "540p", "720p", "1080p"]
        for table in (local_t2v, local_a2v):
            for cell in table.values():
                assert list(cell["fps_to_durations"].keys()) == ["24", "25", "48", "50"]
        for fps in ("24", "25", "48", "50"):
            assert local_t2v["270p"]["fps_to_durations"][fps] == [2, 3, 4, 5, 6, 8, 10, 20]
            assert local_t2v["540p"]["fps_to_durations"][fps] == [2, 3, 4, 5, 6, 8, 10, 20]
            assert local_a2v["540p"]["fps_to_durations"][fps] == [5, 6, 8, 10, 20]
        assert local_t2v["720p"]["fps_to_durations"]["24"] == [2, 3, 4, 5, 6, 8, 10, 20]
        assert local_t2v["1080p"]["fps_to_durations"]["24"] == [2, 3, 4, 5, 10]
        assert local_a2v["1080p"]["fps_to_durations"]["24"] == [5, 10]
        assert local_t2v["540p"]["aspect_ratios"] == ["21:9", "16:9", "3:2", "4:3", "1:1", "4:5", "9:16"]
        assert local_a2v["540p"]["aspect_ratios"] == ["21:9", "16:9", "3:2", "4:3", "1:1", "4:5", "9:16"]
        assert [item["pipeline"] for item in data["api_models"]] == ["fast", "pro", "fast-2.5", "pro-2.5"]
        api_models_by_pipeline = {item["pipeline"]: item for item in data["api_models"]}
        assert api_models_by_pipeline["fast"]["spec"]["supported_resolutions_durations"]["1080p"]["aspect_ratios"] == [
            "16:9",
            "9:16",
        ]
        assert api_models_by_pipeline["fast-2.5"]["spec"]["supported_resolutions_durations"]["1080p"]["aspect_ratios"] == [
            "16:9",
            "3:2",
            "1:1",
            "9:16",
        ]
        assert api_models_by_pipeline["pro-2.5"]["spec"]["supported_resolutions_durations"]["1080p"]["aspect_ratios"] == [
            "16:9",
            "3:2",
            "1:1",
            "9:16",
        ]
        assert api_models_by_pipeline["fast-2.5"]["spec"]["a2v_supported_resolutions_durations"]["1080p"]["aspect_ratios"] == [
            "16:9",
            "9:16",
        ]
        assert data["api_models"][0]["spec"]["supported_resolutions_durations"]["1080p"]["fps_to_durations"]["24"] == [
            2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20,
        ]

        # A2V: none on fast; 720p–4K on pro/fast-2.5; 720p+1080p on pro-2.5 (no 48 fps).
        assert api_models_by_pipeline["fast"]["spec"]["a2v_supported_resolutions_durations"] is None
        assert list(api_models_by_pipeline["pro"]["spec"]["a2v_supported_resolutions_durations"].keys()) == [
            "720p", "1080p", "1440p", "2160p",
        ]
        assert api_models_by_pipeline["fast-2.5"]["spec"]["display_name"] == "LTX-2.5 Fast (API)"
        assert list(api_models_by_pipeline["fast-2.5"]["spec"]["a2v_supported_resolutions_durations"].keys()) == [
            "720p", "1080p", "1440p", "2160p",
        ]
        assert list(api_models_by_pipeline["pro-2.5"]["spec"]["a2v_supported_resolutions_durations"].keys()) == [
            "720p", "1080p",
        ]
        assert list(api_models_by_pipeline["pro-2.5"]["spec"]["supported_resolutions_durations"].keys()) == [
            "720p", "1080p",
        ]
        assert list(api_models_by_pipeline["fast"]["spec"]["supported_resolutions_durations"].keys()) == [
            "720p", "1080p", "1440p", "2160p",
        ]
        assert "48" not in api_models_by_pipeline["pro-2.5"]["spec"]["supported_resolutions_durations"]["1080p"]["fps_to_durations"]
        assert api_models_by_pipeline["pro"]["spec"]["a2v_supported_resolutions_durations"]["1440p"]["fps_to_durations"]["24"] == [
            2, 3, 4, 5, 6, 8, 10,
        ]
        assert api_models_by_pipeline["pro"]["spec"]["a2v_supported_resolutions_durations"]["720p"]["fps_to_durations"]["24"] == [
            2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20,
        ]

        local_caps = data["local_models"][0]["spec"]["capabilities"]
        assert local_caps["a2v"] is True
        assert local_caps["ic_lora"] is True
        assert local_caps["user_loras"] is True
        assert local_caps["retake"] is True
        assert local_caps["extend"] is True
        assert local_caps["multi_keyframe"] is True
        # No DurationHead on disk in this fixture — Auto stays off until that file is present.
        assert local_caps["auto_duration"] is False
        assert api_models_by_pipeline["fast"]["spec"]["capabilities"]["a2v"] is False
        assert api_models_by_pipeline["fast"]["spec"]["capabilities"]["multi_keyframe"] is False
        assert api_models_by_pipeline["fast"]["spec"]["capabilities"]["auto_duration"] is False
        assert api_models_by_pipeline["fast-2.5"]["spec"]["capabilities"]["a2v"] is True
        assert api_models_by_pipeline["fast-2.5"]["spec"]["capabilities"]["multi_keyframe"] is False
        assert api_models_by_pipeline["fast-2.5"]["spec"]["capabilities"]["auto_duration"] is True
        assert api_models_by_pipeline["pro"]["spec"]["capabilities"]["retake"] is True
        assert api_models_by_pipeline["pro"]["spec"]["capabilities"]["multi_keyframe"] is False
        assert api_models_by_pipeline["pro-2.5"]["spec"]["capabilities"]["retake"] is False
        assert api_models_by_pipeline["pro-2.5"]["spec"]["capabilities"]["multi_keyframe"] is False
        assert api_models_by_pipeline["pro-2.5"]["spec"]["capabilities"]["auto_duration"] is True

    def test_local_fast_16gb_hides_cells_generate_would_reject(self, client, test_state):
        test_state.config.vram_gb = 16
        test_state.config.local_generations_mode = "streaming_models_loading"
        local_t2v = client.get("/api/generate/models-specs").json()["local_models"][0]["spec"][
            "supported_resolutions_durations"
        ]
        assert local_t2v["540p"]["fps_to_durations"]["24"] == [2, 3, 4, 5, 6, 8, 10, 20]
        assert local_t2v["720p"]["fps_to_durations"]["24"] == [2, 3, 4, 5, 6, 8, 10]
        assert local_t2v["1080p"]["fps_to_durations"]["24"] == [2, 3, 4, 5]
        api_1080 = client.get("/api/generate/models-specs").json()["api_models"][0]["spec"][
            "supported_resolutions_durations"
        ]["1080p"]["fps_to_durations"]["24"]
        assert 20 in api_1080

    def test_local_fast_24gb_keeps_1080p_10s(self, client, test_state):
        test_state.config.vram_gb = 24
        test_state.config.local_generations_mode = "streaming_models_loading"
        local_t2v = client.get("/api/generate/models-specs").json()["local_models"][0]["spec"][
            "supported_resolutions_durations"
        ]
        assert local_t2v["1080p"]["fps_to_durations"]["24"] == [2, 3, 4, 5, 10]
        assert local_t2v["720p"]["fps_to_durations"]["24"] == [2, 3, 4, 5, 6, 8, 10, 20]

    def test_local_fast_darwin_20gib_keeps_ceiling(self, client, test_state):
        test_state.config.darwin_unified_memory = True
        test_state.config.available_ram_gb = 20
        test_state.config.local_generations_mode = "streaming_models_loading"
        local_t2v = client.get("/api/generate/models-specs").json()["local_models"][0]["spec"][
            "supported_resolutions_durations"
        ]
        assert local_t2v["720p"]["fps_to_durations"]["24"] == [2, 3, 4, 5, 6, 8, 10, 20]
        assert local_t2v["1080p"]["fps_to_durations"]["24"] == [2, 3, 4, 5, 10]

    def test_local_fast_unsupported_is_empty(self, client, test_state):
        test_state.config.local_generations_mode = "unsupported"
        spec = client.get("/api/generate/models-specs").json()["local_models"][0]["spec"]
        assert spec["supported_resolutions_durations"] == {}
        assert spec["a2v_supported_resolutions_durations"] == {}

    def test_local_fast_never_advertises_1080p_20s(self, client):
        local_t2v = client.get("/api/generate/models-specs").json()["local_models"][0]["spec"][
            "supported_resolutions_durations"
        ]
        assert 20 not in local_t2v["1080p"]["fps_to_durations"]["24"]

    def test_local_auto_duration_requires_duration_head_on_disk(self, client, create_fake_model_files):
        create_fake_model_files()
        r = client.get("/api/generate/models-specs")
        assert r.status_code == 200
        assert r.json()["local_models"][0]["spec"]["capabilities"]["auto_duration"] is True

    def test_local_auto_duration_hidden_when_duration_head_missing(
        self, client, test_state, create_fake_model_files
    ):
        create_fake_model_files()
        delete_cp_path(test_state.config.default_models_dir, "ltx-2.5-duration-head")
        r = client.get("/api/generate/models-specs")
        assert r.status_code == 200
        data = r.json()
        assert data["local_models"][0]["spec"]["capabilities"]["auto_duration"] is False
        api_by_pipeline = {item["pipeline"]: item for item in data["api_models"]}
        assert api_by_pipeline["fast-2.5"]["spec"]["capabilities"]["auto_duration"] is True
        assert api_by_pipeline["pro-2.5"]["spec"]["capabilities"]["auto_duration"] is True


class TestGenerationProgress:
    def test_idle(self, client):
        r = client.get("/api/generation/progress")
        assert r.status_code == 200
        assert r.json()["status"] == "idle"
        assert r.json()["cancellable"] is False

    def test_running(self, client, test_state):
        _fake_running_generation_state(test_state)
        test_state.generation.update_progress("inference", 50, 4, 8)

        r = client.get("/api/generation/progress")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "running"
        assert data["phase"] == "inference"
        assert data["progress"] == 50
        assert data["currentStep"] == 4
        assert data["totalSteps"] == 8
        assert data["cancellable"] is True

    def test_running_from_api_generation_state(self, client, test_state):
        test_state.generation.start_api_generation("api-running")
        test_state.generation.update_progress("inference", 35)

        r = client.get("/api/generation/progress")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "running"
        assert data["phase"] == "inference"
        assert data["progress"] == 35
        assert data["currentStep"] is None
        assert data["totalSteps"] is None
        assert data["cancellable"] is False

    def test_starting_reservation_is_cancellable(self, client, test_state):
        assert test_state.generation.try_reserve_generation_start() is True
        r = client.get("/api/generation/progress")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "running"
        assert data["phase"] == "starting"
        assert data["cancellable"] is True
        test_state.generation.release_generation_start_reservation()


class TestGenerateImage:
    def test_happy_path(self, client, create_fake_model_files):
        create_fake_model_files(include_zit=True)
        r = client.post(
            "/api/generate-image",
            json={"prompt": "A cat", "width": 1024, "height": 1024, "numSteps": 4},
        )

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "complete"
        assert len(data["image_paths"]) == 1
        assert Path(data["image_paths"][0]).exists()

    def test_dimension_clamping(self, client, fake_services, create_fake_model_files):
        create_fake_model_files(include_zit=True)
        r = client.post(
            "/api/generate-image",
            json={"prompt": "test", "width": 1023, "height": 1023},
        )
        assert r.status_code == 200

        call = fake_services.image_generation_pipeline.generate_calls[0]
        assert call["width"] == 1008
        assert call["height"] == 1008

    def test_num_images_clamped(self, client, fake_services, create_fake_model_files):
        create_fake_model_files(include_zit=True)
        r = client.post(
            "/api/generate-image",
            json={"prompt": "test", "numImages": 20},
        )
        assert r.status_code == 200

        assert len(fake_services.image_generation_pipeline.generate_calls) == 12

    def test_error(self, client, fake_services, create_fake_model_files):
        create_fake_model_files(include_zit=True)
        fake_services.image_generation_pipeline.raise_on_generate = RuntimeError("GPU OOM")

        r = client.post("/api/generate-image", json={"prompt": "test"})
        assert r.status_code == 500

    def test_cancelled(self, client, fake_services, create_fake_model_files):
        create_fake_model_files(include_zit=True)
        fake_services.image_generation_pipeline.raise_on_generate = GenerationCancelledError()

        r = client.post("/api/generate-image", json={"prompt": "test"})
        assert r.status_code == 200
        assert r.json()["status"] == "cancelled"

    def test_in_flight_cancel_stops_before_remaining_steps(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        create_fake_model_files(include_zit=True)
        pipeline = fake_services.image_generation_pipeline
        pipeline.inference_steps = 12
        pipeline.step_delay_s = 0.05

        response = _cancel_in_flight(
            client,
            pipeline.entered_inference,
            lambda: test_state.image_generation.generate(
                GenerateImageRequest.model_validate({"prompt": "test", "numSteps": 4})
            ),
        )

        assert response.status == "cancelled"
        assert pipeline.steps_completed < pipeline.inference_steps
        assert test_state.state.gpu_slot is not None

    def test_partial_outputs_cleaned_up_on_mid_batch_error(self, client, fake_services, create_fake_model_files, tmp_path):
        create_fake_model_files(include_zit=True)
        fake_services.image_generation_pipeline.fail_generate_after = 1
        fake_services.image_generation_pipeline.raise_on_generate = RuntimeError("GPU OOM")

        r = client.post("/api/generate-image", json={"prompt": "test", "numImages": 3})
        assert r.status_code == 500
        assert list((tmp_path / "outputs").glob("zit_image_*.png")) == []


class TestForcedApiGenerateImage:
    def test_generate_image_routes_to_zit_api(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.fal_api_key = "fal-key"

        r = client.post(
            "/api/generate-image",
            json={"prompt": "A cat", "width": 1024, "height": 1024, "numSteps": 4, "numImages": 2},
        )

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "complete"
        assert len(data["image_paths"]) == 2
        assert len(fake_services.zit_api_client.text_to_image_calls) == 2
        assert len(fake_services.image_generation_pipeline.generate_calls) == 0

    def test_generate_image_missing_fal_key(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.fal_api_key = ""

        r = client.post("/api/generate-image", json={"prompt": "A cat"})

        assert_http_error(
            r,
            status_code=422,
            code=LOCAL_GENERATION_UNSUPPORTED,
            message=LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
        )

    def test_generate_image_cancelled(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.fal_api_key = "fal-key"
        fake_services.zit_api_client.raise_on_text_to_image = GenerationCancelledError()

        r = client.post("/api/generate-image", json={"prompt": "A cat"})

        assert r.status_code == 200
        assert r.json()["status"] == "cancelled"

    def test_partial_outputs_cleaned_up_on_mid_batch_error(self, client, test_state, fake_services, tmp_path):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.fal_api_key = "fal-key"
        fake_services.zit_api_client.fail_text_to_image_after = 1
        fake_services.zit_api_client.raise_on_text_to_image = RuntimeError("boom")

        r = client.post("/api/generate-image", json={"prompt": "A cat", "numImages": 3})

        assert r.status_code == 500
        assert list((tmp_path / "outputs").glob("zit_api_image_*.png")) == []


class TestEmptyPromptRejected:
    def test_empty_prompt_rejected(self, client):
        r = client.post("/api/generate", json={"prompt": ""})
        assert r.status_code == 422

    def test_whitespace_prompt_rejected(self, client):
        r = client.post("/api/generate", json={"prompt": "   "})
        assert r.status_code == 422

    def test_missing_prompt_rejected(self, client):
        r = client.post("/api/generate", json={})
        assert r.status_code == 422

    def test_empty_image_prompt_rejected(self, client):
        r = client.post("/api/generate-image", json={"prompt": ""})
        assert r.status_code == 422

    def test_whitespace_image_prompt_rejected(self, client):
        r = client.post("/api/generate-image", json={"prompt": "   "})
        assert r.status_code == 422

    def test_missing_image_prompt_rejected(self, client):
        r = client.post("/api/generate-image", json={})
        assert r.status_code == 422


class TestEnhancePromptFlag:
    """Verify enhance_prompt is passed correctly to the text encoder API.

    Typed T2V prompts carry the flag by default (T2V toggle on). Provenance and the
    T2V/I2V settings live in test_automatic_prompt_enhancement.py.
    """

    def _setup_api_encoding(self, test_state, fake_services, create_fake_model_files):
        create_fake_model_files(model_id=_API_ENCODING_MODEL_ID)
        test_state.state.app_settings.active_ltx_model_id = _API_ENCODING_MODEL_ID
        test_state.state.app_settings.ltx_api_key = "test-key"
        test_state.state.app_settings.use_local_text_encoder = False
        fake_services.text_encoder.encode_responses.append(_FakeEncodingResult())

    def test_t2v_enhances_server_side(self, client, test_state, fake_services, create_fake_model_files):
        self._setup_api_encoding(test_state, fake_services, create_fake_model_files)

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200

        assert len(fake_services.text_encoder.encode_calls) == 1
        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is True

    def test_empty_prompt_api_encoding_uses_placeholder(
        self, test_state, fake_services, create_fake_model_files
    ):
        # The LTX API rejects an empty prompt, but empty prompts are valid for some IC-LoRAs
        # (e.g. outpainting). In API mode (no gemma fallback) the empty prompt must be encoded
        # via a neutral placeholder instead of raising "API text encoding failed".
        self._setup_api_encoding(test_state, fake_services, create_fake_model_files)

        # Must not raise.
        test_state.text.prepare_text_encoding("", enhance_prompt=True)

        assert len(fake_services.text_encoder.encode_calls) == 1
        assert fake_services.text_encoder.encode_calls[0]["prompt"] == " "  # placeholder, not ""
        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is False  # nothing to enhance

    def test_i2v_enhances_server_side(self, client, test_state, fake_services, create_fake_model_files, make_test_image, tmp_path):
        self._setup_api_encoding(test_state, fake_services, create_fake_model_files)
        image_path = tmp_path / "input.png"
        image_path.write_bytes(make_test_image().getvalue())

        r = client.post("/api/generate", json={**_T2V_JSON, "imagePath": str(image_path)})
        assert r.status_code == 200

        assert len(fake_services.text_encoder.encode_calls) == 1
        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is True

    def test_a2v_without_image_enhances_server_side(self, client, test_state, fake_services, create_fake_model_files, tmp_path):
        self._setup_api_encoding(test_state, fake_services, create_fake_model_files)
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post("/api/generate", json={**_T2V_JSON, "model": "fast", "audioPath": str(audio_file)})
        assert r.status_code == 200

        assert len(fake_services.text_encoder.encode_calls) == 1
        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is True

    def test_a2v_with_image_enhances_server_side(self, client, test_state, fake_services, create_fake_model_files, make_test_image, tmp_path):
        self._setup_api_encoding(test_state, fake_services, create_fake_model_files)
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)
        image_path = tmp_path / "input.png"
        image_path.write_bytes(make_test_image().getvalue())

        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "model": "fast", "audioPath": str(audio_file), "imagePath": str(image_path)},
        )
        assert r.status_code == 200

        assert len(fake_services.text_encoder.encode_calls) == 1
        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is True

    def test_local_encoding_skips_api(self, client, test_state, fake_services, create_fake_model_files):
        create_fake_model_files()
        test_state.state.app_settings.ltx_api_key = "test-key"
        test_state.state.app_settings.use_local_text_encoder = True

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200

        assert len(fake_services.text_encoder.encode_calls) == 0


class TestLocalEncodingEnhancement:
    """The rewrite that API encoding gets server-side has to happen here for local encoding.

    Without it nothing rewrites the prompt whenever the local encoder is selected, and the model
    sees the prompt exactly as typed.
    """

    def _setup_local(self, test_state, create_fake_model_files, *, with_enhancer: bool):
        create_fake_model_files(include_prompt_enhancer=with_enhancer)
        test_state.state.app_settings.use_local_text_encoder = True

    def test_t2v_prompt_is_enhanced_before_it_reaches_the_pipeline(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        self._setup_local(test_state, create_fake_model_files, with_enhancer=True)
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = "a long descriptive caption"

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200

        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls[0]["prompt"] == "test"
        assert fake_services.fast_video_pipeline.generate_calls[0]["prompt"] == "a long descriptive caption"

    def test_enhancer_runs_before_the_generation_is_marked_running(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        # The enhancer evicts whatever pipeline is resident to claim its VRAM, and eviction is
        # refused once a generation is running — so getting a pipeline built at all is the
        # assertion that the ordering held.
        self._setup_local(test_state, create_fake_model_files, with_enhancer=True)

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200
        assert len(fake_services.prompt_enhancer_pipeline.created_with) == 1

    def test_missing_enhancer_generates_with_the_prompt_as_typed(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        self._setup_local(test_state, create_fake_model_files, with_enhancer=False)

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200

        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls == []
        assert fake_services.fast_video_pipeline.generate_calls[0]["prompt"] == "test"

    def test_2_5_uses_gemma3_fallback_to_enhance_before_generate(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        create_fake_model_files(include_prompt_enhancer=False)
        create_fake_model_files(model_id="ltx-2.3-22b-distilled-1.1")
        test_state.state.app_settings.active_ltx_model_id = "ltx-2.5-22b-distilled"
        test_state.state.app_settings.use_local_text_encoder = True
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = "a long descriptive caption"

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200
        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls[0]["prompt"] == "test"
        assert fake_services.fast_video_pipeline.generate_calls[0]["prompt"] == "a long descriptive caption"

    def test_enhancer_failure_does_not_fail_the_generation(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        self._setup_local(test_state, create_fake_model_files, with_enhancer=True)
        fake_services.prompt_enhancer_pipeline.raise_on_enhance = RuntimeError("boom")

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200
        assert fake_services.fast_video_pipeline.generate_calls[0]["prompt"] == "test"

    def test_camera_motion_is_appended_after_the_rewrite(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        self._setup_local(test_state, create_fake_model_files, with_enhancer=True)
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = "a long descriptive caption"
        suffix = test_state.config.camera_motion_prompts["dolly_in"]

        r = client.post("/api/generate", json={**_T2V_JSON, "cameraMotion": "dolly_in"})
        assert r.status_code == 200

        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls[0]["prompt"] == "test"
        assert (
            fake_services.fast_video_pipeline.generate_calls[0]["prompt"]
            == "a long descriptive caption" + suffix
        )

    def test_i2v_routes_the_conditioning_image_to_the_enhancer(
        self, client, test_state, fake_services, create_fake_model_files, make_test_image, tmp_path
    ):
        self._setup_local(test_state, create_fake_model_files, with_enhancer=True)
        image_path = tmp_path / "input.png"
        image_path.write_bytes(make_test_image().getvalue())

        r = client.post("/api/generate", json={**_T2V_JSON, "imagePath": str(image_path)})
        assert r.status_code == 200

        assert len(fake_services.prompt_enhancer_pipeline.enhance_i2v_calls) == 1
        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls == []
        assert fake_services.prompt_enhancer_pipeline.enhance_i2v_calls[0]["last_image_path"] is None

    def test_i2v_last_frame_is_passed_to_the_enhancer(
        self, client, test_state, fake_services, create_fake_model_files, make_test_image, tmp_path
    ):
        self._setup_local(test_state, create_fake_model_files, with_enhancer=True)
        first = tmp_path / "first.png"
        last = tmp_path / "last.png"
        first.write_bytes(make_test_image().getvalue())
        last.write_bytes(make_test_image().getvalue())

        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "imagePath": str(first), "lastImagePath": str(last)},
        )
        assert r.status_code == 200
        call = fake_services.prompt_enhancer_pipeline.enhance_i2v_calls[0]
        assert call["image_path"] == str(first)
        assert call["last_image_path"] == str(last)

    def test_keyframes_are_all_passed_to_the_enhancer(
        self, client, test_state, fake_services, create_fake_model_files, make_test_image, tmp_path
    ):
        self._setup_local(test_state, create_fake_model_files, with_enhancer=True)
        opening = tmp_path / "opening.png"
        middle = tmp_path / "middle.png"
        closing = tmp_path / "closing.png"
        for path in (opening, middle, closing):
            path.write_bytes(make_test_image().getvalue())

        r = client.post(
            "/api/generate",
            json={
                **_T2V_JSON,
                "keyframes": [
                    {"imagePath": str(closing), "frameIndex": 80},
                    {"imagePath": str(opening), "frameIndex": 0},
                    {"imagePath": str(middle), "frameIndex": 40},
                ],
            },
        )
        assert r.status_code == 200
        call = fake_services.prompt_enhancer_pipeline.enhance_i2v_calls[0]
        assert call["keyframes"] == [
            (str(opening), 0, 1.0),
            (str(middle), 40, 1.0),
            (str(closing), 80, 1.0),
        ]
        assert call["duration"] == 5
        assert call["fps"] == 24
        assert call["last_image_path"] is None
        assert call["system_prompt"] is not None
        assert "visual ground truth" in call["system_prompt"]
        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls == []

    def test_api_encoding_still_enhances_server_side(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        create_fake_model_files(model_id=_API_ENCODING_MODEL_ID, include_prompt_enhancer=True)
        test_state.state.app_settings.active_ltx_model_id = _API_ENCODING_MODEL_ID
        test_state.state.app_settings.ltx_api_key = "test-key"
        test_state.state.app_settings.use_local_text_encoder = False
        fake_services.text_encoder.encode_responses.append(_FakeEncodingResult())

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200

        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls == []
        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is True
