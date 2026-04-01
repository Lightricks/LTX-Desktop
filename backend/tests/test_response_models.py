"""Response key casing contract tests."""

from __future__ import annotations

from state.app_state_types import DownloadingSession, FileDownloadRunning, GpuSlot, VideoPipelineState, VideoPipelineWarmth
from tests.fakes.services import FakeFastVideoPipeline


class TestGenerationProgressCamelCaseKeys:
    def test_camelcase_keys(self, client, test_state):
        pipeline = FakeFastVideoPipeline()
        test_state.state.gpu_slot = GpuSlot(
            active_pipeline=VideoPipelineState(
                pipeline=pipeline,
                warmth=VideoPipelineWarmth.COLD,
                is_compiled=False,
            ),
            generation=None,
        )
        test_state.generation.start_generation("gen-1")
        test_state.generation.update_progress("inference", 50, 5, 20)

        r = client.get("/api/generation/progress")
        assert r.status_code == 200
        data = r.json()
        assert "currentStep" in data
        assert "totalSteps" in data
        assert "current_step" not in data
        assert "total_steps" not in data
        assert data["currentStep"] == 5
        assert data["totalSteps"] == 20


class TestDownloadProgressSnakeCaseKeys:
    def test_snake_case_keys(self, client, test_state):
        test_state.state.downloading_session = DownloadingSession(
            id="test-session",
            current_running_file=FileDownloadRunning(
                file_type="checkpoint",
                target_path="checkpoint",
                downloaded_bytes=5_000_000_000,
                speed_bytes_per_sec=50_000_000.0,
            ),
            files_to_download={"checkpoint"},
            completed_files=set(),
            completed_bytes=0,
        )

        r = client.get("/api/models/download/progress", params={"sessionId": "test-session"})
        assert r.status_code == 200
        data = r.json()

        expected_keys = {
            "status",
            "current_downloading_file",
            "current_file_progress",
            "total_progress",
            "total_downloaded_bytes",
            "expected_total_bytes",
            "completed_files",
            "all_files",
            "error",
            "speed_bytes_per_sec",
        }
        assert set(data.keys()) == expected_keys


class TestSettingsCamelCaseKeys:
    def test_camelcase_keys(self, client):
        r = client.get("/api/settings")
        assert r.status_code == 200
        data = r.json()

        assert "useTorchCompile" in data
        assert "use_torch_compile" not in data
        assert "fastModel" in data
        assert "fast_model" not in data
        assert "seedLocked" in data
        assert "seed_locked" not in data
        assert "hasFalApiKey" in data


class TestGenerateSnakeCaseKeys:
    def test_snake_case_keys(self, client, test_state, create_fake_model_files):
        create_fake_model_files()
        test_state.state.app_settings.use_local_text_encoder = True

        r = client.post("/api/generate", json={"prompt": "test"})
        assert r.status_code == 200
        data = r.json()
        assert "video_path" in data
        assert "videoPath" not in data


class TestGenerateImageSnakeCaseKeys:
    def test_snake_case_keys(self, client, create_fake_model_files):
        create_fake_model_files(include_zit=True)
        r = client.post("/api/generate-image", json={"prompt": "test"})
        assert r.status_code == 200
        data = r.json()
        assert "image_paths" in data
        assert "imagePaths" not in data
