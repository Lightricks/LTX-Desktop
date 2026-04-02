"""Integration-style tests for generation and image endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from state.app_state_types import GpuSlot, VideoPipelineState, VideoPipelineWarmth
from tests.fakes.services import FakeFastVideoPipeline


@dataclass
class _FakeEncodingResult:
    """Minimal stand-in for TextEncodingResult in tests."""

    video_context: object = "fake_tensor"
    audio_context: object = None

_T2V_JSON = {
    "prompt": "test",
    "resolution": "540p",
    "model": "fast",
    "duration": 2,
    "fps": 24,
}


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
            warmth=VideoPipelineWarmth.COLD,
            is_compiled=False,
        ),
    )
    test_state.generation.start_generation("running")


class TestGenerate:
    def test_t2v_happy_path(self, client, test_state, fake_services, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A beautiful sunset",
                "resolution": "1080p",
                "model": "fast",
                "duration": 2,
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
        assert r.status_code == 400
        assert "Invalid image file" in r.json()["error"]

    def test_resolution_mapping_540p(self, client, test_state, fake_services, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200

        pipeline = fake_services.fast_video_pipeline
        call = pipeline.generate_calls[0]
        assert call["width"] == 960
        assert call["height"] == 512

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
                "model": "pro",
                "duration": 2,
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

    def test_a2v_rejects_missing_audio_file(self, client, test_state, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "model": "pro",
                "duration": 2,
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
                "model": "pro",
                "duration": 2,
                "fps": 24,
                "audioPath": str(audio_file),
            },
        )
        assert r.status_code == 400
        assert "Invalid audio file" in r.json()["error"]

    def test_a2v_forced_api_routes_to_ltx_api(self, client, test_state, fake_services, tmp_path):
        test_state.config.force_api_generations = True
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

    def test_a2v_prefers_api_routes_to_ltx_api(self, client, test_state, fake_services, tmp_path):
        test_state.config.force_api_generations = False
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
        test_state.config.force_api_generations = False
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
                "model": "pro",
                "duration": 2,
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
        test_state.config.force_api_generations = True
        test_state.state.app_settings.ltx_api_key = "api-key"
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)
        image_path = tmp_path / "input.png"
        image_path.write_bytes(make_test_image().getvalue())

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video with a still frame",
                "resolution": "2160p",
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

    def test_a2v_uses_resolution_map(self, client, test_state, fake_services, create_fake_model_files, tmp_path):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        for resolution, expected_w, expected_h in [
            ("540p", 960, 576),
            ("720p", 1280, 704),
            ("1080p", 1920, 1088),
        ]:
            fake_services.a2v_pipeline.generate_calls.clear()
            r = client.post(
                "/api/generate",
                json={
                    "prompt": "A music video",
                    "resolution": resolution,
                    "model": "pro",
                    "duration": 2,
                    "fps": 24,
                    "audioPath": str(audio_file),
                },
            )

            assert r.status_code == 200
            call = fake_services.a2v_pipeline.generate_calls[0]
            assert call["width"] == expected_w, f"{resolution}: expected width {expected_w}, got {call['width']}"
            assert call["height"] == expected_h, f"{resolution}: expected height {expected_h}, got {call['height']}"

    def test_a2v_forced_api_rejects_missing_audio_file(self, client, test_state):
        test_state.config.force_api_generations = True
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

        assert r.status_code == 400
        assert "Audio file not found" in r.json()["error"]

    def test_a2v_forced_api_missing_key_returns_integrity_error(self, client, test_state, tmp_path):
        test_state.config.force_api_generations = True
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

        assert r.status_code == 400
        assert r.json()["error"] == "PRO_API_KEY_REQUIRED"

    def test_a2v_forced_api_cancelled_response(self, client, test_state, fake_services, tmp_path):
        test_state.config.force_api_generations = True
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
        test_state.config.force_api_generations = False
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
        test_state.config.force_api_generations = False
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
        test_state.config.force_api_generations = True
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

    def test_i2v_routes_to_ltx_api(self, client, test_state, fake_services, make_test_image, tmp_path):
        test_state.config.force_api_generations = True
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
        test_state.config.force_api_generations = True
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
        test_state.config.force_api_generations = True
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
        test_state.config.force_api_generations = True
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
        test_state.config.force_api_generations = True
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
        test_state.config.force_api_generations = True
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

        assert r.status_code == 400
        assert r.json()["error"] == "PRO_API_KEY_REQUIRED"

    def test_invalid_forced_resolution_rejected(self, client, test_state):
        test_state.config.force_api_generations = True
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

        assert r.status_code == 400
        assert r.json()["error"] == "INVALID_FORCED_API_RESOLUTION"

    def test_invalid_forced_duration_rejected(self, client, test_state):
        test_state.config.force_api_generations = True
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

        assert r.status_code == 400
        assert r.json()["error"] == "INVALID_FORCED_API_DURATION"

    def test_invalid_forced_fps_rejected(self, client, test_state):
        test_state.config.force_api_generations = True
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

        assert r.status_code == 400
        assert r.json()["error"] == "INVALID_FORCED_API_FPS"

    def test_invalid_camera_motion_rejected_with_422(self, client, test_state):
        test_state.config.force_api_generations = True
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
        test_state.config.force_api_generations = True
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
        test_state.config.force_api_generations = True
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
        test_state.config.force_api_generations = True
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
        test_state.config.force_api_generations = True
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
        test_state.config.force_api_generations = True
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
        test_state.config.force_api_generations = True
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
        test_state.config.force_api_generations = True
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
        test_state.config.force_api_generations = True
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

        assert r.status_code == 400
        assert r.json()["error"] == "INVALID_FORCED_API_DURATION"

    def test_extended_duration_rejected_for_fast_1440p_24fps(self, client, test_state):
        test_state.config.force_api_generations = True
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

        assert r.status_code == 400
        assert r.json()["error"] == "INVALID_FORCED_API_DURATION"

    def test_fps_24_accepted(self, client, test_state, fake_services):
        test_state.config.force_api_generations = True
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
        test_state.config.force_api_generations = True
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
        test_state.config.force_api_generations = True
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
        assert call["resolution"] == "1920x1080"

    def test_a2v_forced_api_overrides_resolution_and_aspect_ratio(self, client, test_state, fake_services, tmp_path):
        test_state.config.force_api_generations = True
        test_state.state.app_settings.ltx_api_key = "test_key"
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A big video",
                "resolution": "2160p",
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
        assert call["resolution"] == "1920x1080"
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


class TestForcedApiGenerateImage:
    def test_generate_image_routes_to_zit_api(self, client, test_state, fake_services):
        test_state.config.force_api_generations = True
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
        test_state.config.force_api_generations = True
        test_state.state.app_settings.fal_api_key = ""

        r = client.post("/api/generate-image", json={"prompt": "A cat"})

        assert r.status_code == 500
        assert r.json()["error"] == "FAL_API_KEY_NOT_CONFIGURED"

    def test_generate_image_cancelled(self, client, test_state, fake_services):
        test_state.config.force_api_generations = True
        test_state.state.app_settings.fal_api_key = "fal-key"
        fake_services.zit_api_client.raise_on_text_to_image = RuntimeError("cancelled")

        r = client.post("/api/generate-image", json={"prompt": "A cat"})

        assert r.status_code == 200
        assert r.json()["status"] == "cancelled"


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
        create_fake_model_files()
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

        r = client.post("/api/generate", json={**_T2V_JSON, "model": "pro", "audioPath": str(audio_file)})
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
            json={**_T2V_JSON, "model": "pro", "audioPath": str(audio_file), "imagePath": str(image_path)},
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
