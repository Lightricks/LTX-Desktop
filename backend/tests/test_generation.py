"""Integration-style tests for generation and image endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from frame_math import AutoDurationSpec
from runtime_config.model_download_specs import delete_cp_path, get_ltx_model_spec, resolve_model_path
from services.ltx_api_client.ltx_api_client import LTXAPIClientError
from state.app_state_types import GpuSlot, VideoPipelineState
from tests.http_error_assertions import assert_http_error
from tests.fakes.services import FakeFastVideoPipeline


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
        ),
    )
    test_state.generation.start_generation("running")


class TestGenerate:
    def test_t2v_requires_downloaded_ltx_model(self, client):
        r = client.post("/api/generate", json=_T2V_JSON)
        assert_http_error(r, status_code=409, code="NO_DOWNLOADED_LTX_MODEL")

    def test_t2v_on_2_5_uses_api_model_id_without_local_text_encoder(
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
        assert fake_services.text_encoder.encode_calls[0]["api_model_id"] == spec.api_text_encoder_model_id
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
        assert call["num_frames"] == AutoDurationSpec(min_seconds=5, max_seconds=20)

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
        test_state.state.app_settings.prompt_enhancer_enabled_t2v = False
        lora_ref = create_fake_lora("a.safetensors")
        body = {**_T2V_JSON, "loras": [{"ref": lora_ref, "scale": 1.0}]}

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
        test_state.state.app_settings.prompt_enhancer_enabled_t2v = False
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
        # 2.3 540p is 960×544; snap_up_to_multiple(..., 64) maps height to 576
        # on the two-stage grid.
        create_fake_model_files(model_id="ltx-2.3-22b-distilled-1.1")
        test_state.state.app_settings.active_ltx_model_id = "ltx-2.3-22b-distilled-1.1"
        _enable_local_text_encoding(test_state)

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200

        pipeline = fake_services.fast_video_pipeline
        call = pipeline.generate_calls[0]
        assert call["width"] == 960
        assert call["height"] == 576

    def test_local_resolutions_are_all_on_the_two_stage_grid(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        # 2.5 Fast sizes are already /64. Two-stage halves each dimension onto a /32 latent grid,
        # so anything not divisible by 64 gets silently snapped.
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        for resolution in ("540p", "720p", "1080p"):
            for aspect_ratio in ("16:9", "9:16"):
                fake_services.fast_video_pipeline.generate_calls.clear()
                r = client.post(
                    "/api/generate",
                    json={**_T2V_JSON, "resolution": resolution, "aspectRatio": aspect_ratio, "duration": 5},
                )
                assert r.status_code == 200
                call = fake_services.fast_video_pipeline.generate_calls[0]
                assert call["width"] % 64 == 0, f"{resolution} {aspect_ratio}: width {call['width']}"
                assert call["height"] % 64 == 0, f"{resolution} {aspect_ratio}: height {call['height']}"

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
        test_state.state.app_settings.seed_locked = True
        test_state.state.app_settings.locked_seed = 123

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200

        pipeline = fake_services.fast_video_pipeline
        assert pipeline.generate_calls[0]["seed"] == 123

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
        fake_services.fast_video_pipeline.raise_on_generate = RuntimeError("cancelled")

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
        assert call["audio_max_duration"] is None

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

    def test_a2v_uses_resolution_map_on_2_5(self, client, test_state, fake_services, create_fake_model_files, tmp_path):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        for resolution, expected_w, expected_h in [
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

    def test_a2v_540p_on_2_3_uses_historical_pixels(self, client, test_state, fake_services, create_fake_model_files, tmp_path):
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
        assert call["width"] == 960
        assert call["height"] == 544

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

        assert_http_error(r, status_code=400, code="PRO_API_KEY_REQUIRED")

    def test_a2v_forced_api_cancelled_response(self, client, test_state, fake_services, tmp_path):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"
        fake_services.ltx_api_client.raise_on_audio_to_video = RuntimeError("cancelled")
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
                "resolution": "2160p",
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
        assert call["resolution"] == "3840x2160"
        assert call["duration"] == 8.0
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

        assert_http_error(r, status_code=400, code="PRO_API_KEY_REQUIRED")

    def test_invalid_forced_resolution_rejected(self, client, test_state):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.ltx_api_key = "api-key"

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A city skyline",
                "resolution": "720p",
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
            message="Unsupported api text-to-video resolution '720p' for pipeline 'pro'",
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
                "duration": 5,
                "fps": 25,
                "audio": False,
            },
        )

        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message="Unsupported api text-to-video duration '5' for pipeline 'pro' at resolution '1080p' and fps '25'",
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
            stage="generation",
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
        fake_services.ltx_api_client.raise_on_text_to_video = RuntimeError("cancelled")

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

    def test_a2v_forced_api_rejects_non_1080p(self, client, test_state, fake_services, tmp_path):
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


class TestGenerateCancel:
    def test_cancel_active(self, client, test_state):
        _fake_running_generation_state(test_state)

        r = client.post("/api/generate/cancel")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "cancelling"

    def test_cancel_no_active(self, client):
        r = client.post("/api/generate/cancel")
        assert r.status_code == 200
        assert r.json()["status"] == "no_active_generation"


class TestGenerateModelSpecs:
    def test_models_specs_endpoint_returns_ordered_backend_specs(self, client):
        r = client.get("/api/generate/models-specs")

        assert r.status_code == 200
        data = r.json()
        assert [item["pipeline"] for item in data["local_models"]] == ["fast"]
        assert data["local_models"][0]["spec"]["display_name"] == "LTX 2.5 Fast"
        assert list(data["local_models"][0]["spec"]["supported_resolutions_durations"]["540p"]["fps_to_durations"].keys()) == ["24"]
        assert [item["pipeline"] for item in data["api_models"]] == ["fast", "pro", "fast-2.5", "pro-2.5"]
        assert data["api_models"][0]["spec"]["supported_resolutions_durations"]["1080p"]["fps_to_durations"]["24"] == [
            6, 8, 10, 12, 14, 16, 18, 20,
        ]

        api_models_by_pipeline = {item["pipeline"]: item for item in data["api_models"]}
        # A2V envelope: none on fast; 1080p on pro, fast-2.5, and pro-2.5.
        assert api_models_by_pipeline["fast"]["spec"]["a2v_supported_resolutions_durations"] is None
        assert list(api_models_by_pipeline["pro"]["spec"]["a2v_supported_resolutions_durations"].keys()) == ["1080p"]
        assert api_models_by_pipeline["fast-2.5"]["spec"]["display_name"] == "LTX-2.5 Fast (API)"
        assert list(api_models_by_pipeline["fast-2.5"]["spec"]["a2v_supported_resolutions_durations"].keys()) == ["1080p"]
        assert list(api_models_by_pipeline["pro-2.5"]["spec"]["a2v_supported_resolutions_durations"].keys()) == ["1080p"]

        local_caps = data["local_models"][0]["spec"]["capabilities"]
        assert local_caps["a2v"] is True
        assert local_caps["ic_lora"] is True
        assert local_caps["user_loras"] is True
        assert local_caps["retake"] is False
        assert local_caps["extend"] is False
        # No DurationHead on disk in this fixture — Auto stays off until that file is present.
        assert local_caps["auto_duration"] is False
        assert api_models_by_pipeline["fast"]["spec"]["capabilities"]["a2v"] is False
        assert api_models_by_pipeline["fast"]["spec"]["capabilities"]["auto_duration"] is False
        assert api_models_by_pipeline["fast-2.5"]["spec"]["capabilities"]["a2v"] is True
        assert api_models_by_pipeline["fast-2.5"]["spec"]["capabilities"]["auto_duration"] is True
        assert api_models_by_pipeline["pro"]["spec"]["capabilities"]["retake"] is True
        assert api_models_by_pipeline["pro-2.5"]["spec"]["capabilities"]["retake"] is False
        assert api_models_by_pipeline["pro-2.5"]["spec"]["capabilities"]["auto_duration"] is True

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
        fake_services.image_generation_pipeline.raise_on_generate = RuntimeError("cancelled")

        r = client.post("/api/generate-image", json={"prompt": "test"})
        assert r.status_code == 200
        assert r.json()["status"] == "cancelled"

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

        assert_http_error(r, status_code=500, code="FAL_API_KEY_NOT_CONFIGURED")

    def test_generate_image_cancelled(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.fal_api_key = "fal-key"
        fake_services.zit_api_client.raise_on_text_to_image = RuntimeError("cancelled")

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
    """Verify enhance_prompt is passed correctly to the text encoder API."""

    def _setup_api_encoding(self, test_state, fake_services, create_fake_model_files):
        create_fake_model_files(model_id=_API_ENCODING_MODEL_ID)
        test_state.state.app_settings.active_ltx_model_id = _API_ENCODING_MODEL_ID
        test_state.state.app_settings.ltx_api_key = "test-key"
        test_state.state.app_settings.use_local_text_encoder = False
        fake_services.text_encoder.encode_responses.append(_FakeEncodingResult())

    def test_t2v_enhance_enabled(self, client, test_state, fake_services, create_fake_model_files):
        self._setup_api_encoding(test_state, fake_services, create_fake_model_files)
        test_state.state.app_settings.prompt_enhancer_enabled_t2v = True

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200

        assert len(fake_services.text_encoder.encode_calls) == 1
        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is True

    def test_t2v_enhance_disabled(self, client, test_state, fake_services, create_fake_model_files):
        self._setup_api_encoding(test_state, fake_services, create_fake_model_files)
        test_state.state.app_settings.prompt_enhancer_enabled_t2v = False

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200

        assert len(fake_services.text_encoder.encode_calls) == 1
        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is False

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

    def test_i2v_enhance_enabled(self, client, test_state, fake_services, create_fake_model_files, make_test_image, tmp_path):
        self._setup_api_encoding(test_state, fake_services, create_fake_model_files)
        test_state.state.app_settings.prompt_enhancer_enabled_i2v = True
        image_path = tmp_path / "input.png"
        image_path.write_bytes(make_test_image().getvalue())

        r = client.post("/api/generate", json={**_T2V_JSON, "imagePath": str(image_path)})
        assert r.status_code == 200

        assert len(fake_services.text_encoder.encode_calls) == 1
        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is True

    def test_i2v_enhance_disabled(self, client, test_state, fake_services, create_fake_model_files, make_test_image, tmp_path):
        self._setup_api_encoding(test_state, fake_services, create_fake_model_files)
        test_state.state.app_settings.prompt_enhancer_enabled_i2v = False
        image_path = tmp_path / "input.png"
        image_path.write_bytes(make_test_image().getvalue())

        r = client.post("/api/generate", json={**_T2V_JSON, "imagePath": str(image_path)})
        assert r.status_code == 200

        assert len(fake_services.text_encoder.encode_calls) == 1
        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is False

    def test_a2v_without_image_uses_t2v_setting(self, client, test_state, fake_services, create_fake_model_files, tmp_path):
        self._setup_api_encoding(test_state, fake_services, create_fake_model_files)
        test_state.state.app_settings.prompt_enhancer_enabled_t2v = True
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post("/api/generate", json={**_T2V_JSON, "model": "fast", "audioPath": str(audio_file)})
        assert r.status_code == 200

        assert len(fake_services.text_encoder.encode_calls) == 1
        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is True

    def test_a2v_with_image_uses_i2v_setting(self, client, test_state, fake_services, create_fake_model_files, make_test_image, tmp_path):
        self._setup_api_encoding(test_state, fake_services, create_fake_model_files)
        test_state.state.app_settings.prompt_enhancer_enabled_i2v = True
        test_state.state.app_settings.prompt_enhancer_enabled_t2v = False
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
        test_state.state.app_settings.prompt_enhancer_enabled_t2v = True

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200

        assert len(fake_services.text_encoder.encode_calls) == 0


class TestLocalEncodingEnhancement:
    """The rewrite that API encoding gets server-side has to happen here for local encoding.

    Without it the enhancer setting silently does nothing whenever the local encoder is
    selected, and the model sees the prompt exactly as typed.
    """

    def _setup_local(self, test_state, create_fake_model_files, *, with_enhancer: bool):
        create_fake_model_files(include_prompt_enhancer=with_enhancer)
        test_state.state.app_settings.use_local_text_encoder = True
        test_state.state.app_settings.prompt_enhancer_enabled_t2v = True
        test_state.state.app_settings.prompt_enhancer_enabled_i2v = True

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

    def test_disabled_setting_leaves_the_prompt_alone(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        self._setup_local(test_state, create_fake_model_files, with_enhancer=True)
        test_state.state.app_settings.prompt_enhancer_enabled_t2v = False

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200

        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls == []
        assert fake_services.fast_video_pipeline.generate_calls[0]["prompt"] == "test"

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
        test_state.state.app_settings.prompt_enhancer_enabled_t2v = True
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

    def test_api_encoding_still_enhances_server_side(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        create_fake_model_files(model_id=_API_ENCODING_MODEL_ID, include_prompt_enhancer=True)
        test_state.state.app_settings.active_ltx_model_id = _API_ENCODING_MODEL_ID
        test_state.state.app_settings.ltx_api_key = "test-key"
        test_state.state.app_settings.use_local_text_encoder = False
        test_state.state.app_settings.prompt_enhancer_enabled_t2v = True
        fake_services.text_encoder.encode_responses.append(_FakeEncodingResult())

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200

        assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls == []
        assert fake_services.text_encoder.encode_calls[0]["enhance_prompt"] is True
