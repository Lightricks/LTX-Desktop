"""Tests for /health and /api/gpu-info endpoints."""

from state.app_state_types import GpuSlot, VideoPipelineState, VideoPipelineWarmth
from tests.fakes.services import FakeFastVideoPipeline


def _set_video_pipeline(state):
    state.state.gpu_slot = GpuSlot(
        active_pipeline=VideoPipelineState(
            pipeline=FakeFastVideoPipeline(),
            warmth=VideoPipelineWarmth.COLD,
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
