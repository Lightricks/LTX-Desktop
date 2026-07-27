"""Tests for /health and /api/gpu-info endpoints."""

import json

from services.fast_video_pipeline.mlx_profile import (
    begin_mlx_profile,
    finish_mlx_profile,
    reset_mlx_profile_for_tests,
)
from state.app_state_types import GpuSlot, VideoPipelineState
from tests.fakes.services import FakeFastVideoPipeline


def _set_video_pipeline(state):
    state.state.gpu_slot = GpuSlot(
        active_pipeline=VideoPipelineState(
            pipeline=FakeFastVideoPipeline(),
            is_compiled=False,
        ),
    )


class TestHealth:
    def test_no_models_loaded(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["models_loaded"] is False
        assert data["active_model"] is None

    def test_fast_model_loaded(self, client, test_state):
        _set_video_pipeline(test_state)
        r = client.get("/health")
        data = r.json()
        assert data["models_loaded"] is True
        assert data["active_model"] == "fast"
        assert data["models_loaded"] is True

    def test_models_downloaded(self, client, create_fake_model_files):
        create_fake_model_files()
        r = client.get("/health")
        data = r.json()
        assert len(data["models_status"]) == 1
        assert data["models_status"][0]["downloaded"] is True

    def test_cors_header(self, client):
        r = client.get("/health", headers={"Origin": "http://localhost:5173"})
        assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


class TestGpuInfo:
    def test_no_gpu(self, client, test_state):
        test_state.gpu_info.cuda_available = False
        test_state.gpu_info.mps_available = False
        test_state.gpu_info.gpu_name = None
        test_state.gpu_info.vram_gb = None
        test_state.gpu_info.gpu_info = {"name": "Unknown", "vram": 0, "vramUsed": 0}

        r = client.get("/api/gpu-info")
        assert r.status_code == 200
        data = r.json()
        assert data["cuda_available"] is False
        assert data["mps_available"] is False
        assert data["gpu_available"] is False
        assert data["gpu_name"] is None
        assert data["vram_gb"] is None

    def test_with_cuda(self, client, test_state):
        test_state.gpu_info.cuda_available = True
        test_state.gpu_info.mps_available = False
        test_state.gpu_info.gpu_name = "RTX 5090"
        test_state.gpu_info.vram_gb = 32
        test_state.gpu_info.gpu_info = {"name": "Test GPU", "vram": 8192, "vramUsed": 1024}

        r = client.get("/api/gpu-info")
        assert r.status_code == 200
        data = r.json()
        assert data["cuda_available"] is True
        assert data["mps_available"] is False
        assert data["gpu_available"] is True
        assert data["gpu_name"] == "RTX 5090"
        assert data["vram_gb"] == 32

    def test_with_mps(self, client, test_state):
        test_state.gpu_info.cuda_available = False
        test_state.gpu_info.mps_available = True
        test_state.gpu_info.gpu_name = "Apple Silicon (MPS)"
        test_state.gpu_info.vram_gb = 36
        test_state.gpu_info.gpu_info = {"name": "Apple Silicon (MPS)", "vram": 36864, "vramUsed": 0}

        r = client.get("/api/gpu-info")
        assert r.status_code == 200
        data = r.json()
        assert data["cuda_available"] is False
        assert data["mps_available"] is True
        assert data["gpu_available"] is True
        assert data["gpu_name"] == "Apple Silicon (MPS)"
        assert data["vram_gb"] == 36


class TestMpsMemory:
    def test_returns_typed_snapshot(self, client):
        """Contract-shape check that holds on any platform: 200 + a bool `available`,
        and the mib fields are ints when available / null when not (off MPS)."""
        r = client.get("/api/gpu-info/mps")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data["available"], bool)
        for key in ("allocated_mib", "driver_mib", "recommended_max_mib"):
            assert data[key] is None or isinstance(data[key], int)


class TestRuntimeTelemetry:
    def test_returns_process_system_and_lease_snapshot(self, client):
        r = client.get("/api/runtime-telemetry")
        assert r.status_code == 200
        data = r.json()
        assert data["process_rss_mib"] > 0
        assert data["system_total_mib"] >= data["system_available_mib"] > 0
        assert data["local_metal_lease_status"] in {"idle", "waiting", "held"}

    def test_reports_terminal_mlx_profile_after_child_exit(self, client, tmp_path):
        reset_mlx_profile_for_tests()
        profile = tmp_path / "job.jsonl"
        begin_mlx_profile(profile)
        profile.write_text(
            json.dumps(
                {
                    "event": "run_end",
                    "timestamp_unix_seconds": 1002.0,
                    "mlx_active_gb": 0.25,
                    "mlx_cache_gb": 0.125,
                    "mlx_peak_gb": 6.0,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        finish_mlx_profile(profile, "success")
        try:
            response = client.get("/api/runtime-telemetry")
            assert response.status_code == 200
            data = response.json()
            assert data["mlx_profile_status"] == "success"
            assert data["mlx_active_mib"] == 256
            assert data["mlx_cache_mib"] == 128
            assert data["mlx_peak_mib"] == 6144
            assert data["mlx_profile_path"] == str(profile)
        finally:
            reset_mlx_profile_for_tests()
