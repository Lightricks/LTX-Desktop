"""Internal raw-byte GPU/RAM probes for Windows Fast compatibility plumbing."""

from __future__ import annotations

from types import SimpleNamespace

import psutil
import torch

import services.gpu_info.gpu_info_impl as gpu_info_impl
from runtime_config.local_runtime import resolve_local_runtime
from runtime_config.runtime_config import RuntimeConfig
from runtime_config.video_job_budget import build_local_job_filter, budget_route
from services.gpu_info.gpu_info_impl import GpuInfoImpl
from services.storage_info.storage_info import DriveType
from tests.fakes.fake_gpu_info import FakeGpuInfo
from tests.fakes.fake_storage_info import FakeStorageInfo

# Typical "16 GB" NVIDIA device total that integer-GiB floor reports as 15.
_SIXTEEN_GB_CARD_VRAM_BYTES = 16376 * 1024 * 1024
# A "12 GB" card reporting just under nominal: 11.994 GiB, floored to 11.
_TWELVE_GB_CARD_VRAM_BYTES = 12 * 1024**3 - 6 * 1024**2
# 11 GB and 10 GB classes, same below-nominal shape: 10.994 and 9.994 GiB.
_ELEVEN_GB_CARD_VRAM_BYTES = 11 * 1024**3 - 6 * 1024**2
_TEN_GB_CARD_VRAM_BYTES = 10 * 1024**3 - 6 * 1024**2


def test_fake_cuda_vram_bytes_are_not_gib_floor_truncated() -> None:
    gpu = FakeGpuInfo()
    gpu.cuda_available = True
    gpu.vram_gb = _SIXTEEN_GB_CARD_VRAM_BYTES // (1024**3)
    gpu.vram_bytes = _SIXTEEN_GB_CARD_VRAM_BYTES

    assert gpu.get_vram_total_gb() == 15
    assert gpu.get_cuda_total_vram_bytes() == _SIXTEEN_GB_CARD_VRAM_BYTES
    assert gpu.get_cuda_total_vram_bytes() != gpu.get_vram_total_gb() * (1024**3)


def test_fake_total_ram_bytes_are_exact() -> None:
    gpu = FakeGpuInfo()
    gpu.total_ram_bytes = 64 * 1024**3 + 123456789

    assert gpu.get_total_ram_bytes() == 64 * 1024**3 + 123456789
    assert gpu.get_available_ram_gb() is None


def test_gpu_info_impl_total_ram_bytes_are_discoverable_on_windows(monkeypatch) -> None:
    monkeypatch.setattr(gpu_info_impl.sys, "platform", "win32")
    expected = int(psutil.virtual_memory().total)
    actual = GpuInfoImpl().get_total_ram_bytes()

    assert actual == expected
    assert actual is not None
    assert actual > 0
    assert actual != actual // (1024**3)


def test_gpu_info_impl_total_ram_bytes_none_on_query_failure(monkeypatch) -> None:
    def _boom() -> object:
        raise RuntimeError("virtual_memory unavailable")

    monkeypatch.setattr(gpu_info_impl.psutil, "virtual_memory", _boom)
    assert GpuInfoImpl().get_total_ram_bytes() is None


def test_gpu_info_impl_cuda_vram_bytes_are_not_gib_floor_truncated(monkeypatch) -> None:
    monkeypatch.setattr(gpu_info_impl.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        gpu_info_impl.torch.cuda,
        "get_device_properties",
        lambda _idx: SimpleNamespace(total_memory=_SIXTEEN_GB_CARD_VRAM_BYTES),
    )

    gpu = GpuInfoImpl()
    assert gpu.get_cuda_total_vram_bytes() == _SIXTEEN_GB_CARD_VRAM_BYTES
    assert gpu.get_vram_total_gb() == 15
    assert gpu.get_cuda_total_vram_bytes() != gpu.get_vram_total_gb() * (1024**3)


def test_gpu_info_impl_cuda_vram_bytes_none_when_cuda_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(gpu_info_impl.torch.cuda, "is_available", lambda: False)
    assert GpuInfoImpl().get_cuda_total_vram_bytes() is None


def test_gpu_info_impl_cuda_vram_bytes_none_on_query_failure(monkeypatch) -> None:
    def _boom(_idx: int) -> object:
        raise RuntimeError("cuda properties unavailable")

    monkeypatch.setattr(gpu_info_impl.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gpu_info_impl.torch.cuda, "get_device_properties", _boom)
    assert GpuInfoImpl().get_cuda_total_vram_bytes() is None
    assert GpuInfoImpl().get_vram_total_gb() is None


def test_runtime_config_stores_platform_and_raw_capacity(tmp_path) -> None:
    config = RuntimeConfig(
        device=torch.device("cpu"),
        app_data_dir=tmp_path / "app_data",
        default_models_dir=tmp_path / "models",
        outputs_dir=tmp_path / "outputs",
        settings_file=tmp_path / "settings.json",
        ltx_api_base_url="https://api.ltx.video",
        local_generations_mode="streaming_models_loading",
        use_sage_attention=False,
        camera_motion_prompts={},
        default_negative_prompt="",
        dev_mode=False,
        backend_port=8000,
        vram_gb=15,
        system="Windows",
        vram_bytes=_SIXTEEN_GB_CARD_VRAM_BYTES,
        total_ram_bytes=32 * 1024**3 + 1,
    )

    assert config.system == "Windows"
    assert config.vram_gb == 15
    assert config.vram_bytes == _SIXTEEN_GB_CARD_VRAM_BYTES
    assert config.total_ram_bytes == 32 * 1024**3 + 1
    assert config.vram_bytes != config.vram_gb * (1024**3)


def test_gpu_info_endpoint_exposes_telemetry_hardware_fields(client, test_state) -> None:
    """Raw capacities the renderer cannot observe itself (telemetry reads these)."""
    test_state.gpu_info.cuda_available = True
    test_state.gpu_info.gpu_name = "RTX 5060 Ti"
    test_state.gpu_info.vram_gb = 15
    test_state.gpu_info.vram_bytes = _SIXTEEN_GB_CARD_VRAM_BYTES
    test_state.gpu_info.total_ram_bytes = 48 * 1024**3
    test_state.gpu_info.gpu_info = {"name": "RTX 5060 Ti", "vram": 16376, "vramUsed": 0}
    test_state.config.system = "Windows"
    test_state.config.checkpoint_drive_type = "nvme"
    test_state.gpu_info.free_vram_bytes = int(14.5 * 1024**3)
    test_state.gpu_info.mps_pool_bytes = 115448725504

    data = client.get("/api/gpu-info").json()
    assert data["vram_gb"] == 15
    assert data["gpu_name"] == "RTX 5060 Ti"
    assert data["total_vram_bytes"] == _SIXTEEN_GB_CARD_VRAM_BYTES
    assert data["free_vram_bytes"] == int(14.5 * 1024**3)
    assert data["total_ram_bytes"] == 48 * 1024**3
    assert data["system"] == "Windows"
    assert data["checkpoint_drive_type"] == "nvme"
    assert data["mps_pool_bytes"] == 115448725504


def test_resolve_local_runtime_populates_runtime_config_from_gpu_info(tmp_path) -> None:
    gpu = FakeGpuInfo()
    gpu.cuda_available = True
    gpu.vram_gb = 15
    gpu.vram_bytes = _SIXTEEN_GB_CARD_VRAM_BYTES
    gpu.available_ram_gb = 20
    gpu.total_ram_bytes = 32 * 1024**3 + 1

    runtime = resolve_local_runtime(
        gpu, system="Windows", fp8_capable=True, storage_info=FakeStorageInfo("ssd")
    )
    config = RuntimeConfig(
        device=torch.device("cpu"),
        app_data_dir=tmp_path / "app_data",
        default_models_dir=tmp_path / "models",
        outputs_dir=tmp_path / "outputs",
        settings_file=tmp_path / "settings.json",
        ltx_api_base_url="https://api.ltx.video",
        use_sage_attention=False,
        camera_motion_prompts={},
        default_negative_prompt="",
        dev_mode=False,
        backend_port=8000,
        **runtime.runtime_config_kwargs(),
    )

    assert config.system == "Windows"
    assert config.vram_bytes == _SIXTEEN_GB_CARD_VRAM_BYTES
    assert config.total_ram_bytes == 32 * 1024**3 + 1
    assert config.vram_gb == 15
    assert config.available_ram_gb is None
    assert config.darwin_unified_memory is False
    assert config.local_generations_mode == "streaming_models_loading"
    assert config.vram_bytes != config.vram_gb * (1024**3)
    assert config.checkpoint_drive_type == "ssd"


def _resolve_mode(
    vram_bytes: int | None,
    vram_gb: int | None,
    system: str,
    drive: DriveType | None = None,
) -> str:
    """Production path: FakeGpuInfo probes into resolve_local_runtime's mode."""
    gpu = FakeGpuInfo()
    gpu.cuda_available = True
    gpu.vram_bytes = vram_bytes
    gpu.vram_gb = vram_gb
    gpu.total_ram_bytes = 48 * 1024**3
    return resolve_local_runtime(
        gpu, system=system, fp8_capable=True, storage_info=FakeStorageInfo(drive)
    ).mode


def test_resolve_local_runtime_carries_checkpoint_drive_type() -> None:
    gpu = FakeGpuInfo()
    gpu.cuda_available = True
    gpu.vram_gb = 16
    gpu.vram_bytes = 16 * 1024**3
    gpu.total_ram_bytes = 48 * 1024**3

    runtime = resolve_local_runtime(
        gpu, system="Windows", fp8_capable=True, storage_info=FakeStorageInfo("nvme")
    )
    assert runtime.checkpoint_drive_type == "nvme"

    unknown = resolve_local_runtime(
        gpu, system="Windows", fp8_capable=True, storage_info=FakeStorageInfo(None)
    )
    assert unknown.checkpoint_drive_type is None


def test_local_runtime_carries_fp8_capable_to_config_and_route(tmp_path) -> None:
    """The startup probe must reach the budget route, not die in resolve."""
    gpu = FakeGpuInfo()
    gpu.cuda_available = True
    gpu.vram_gb = 24
    gpu.vram_bytes = 24 * 1024**3
    gpu.total_ram_bytes = 48 * 1024**3

    runtime = resolve_local_runtime(
        gpu, system="Windows", fp8_capable=False, storage_info=FakeStorageInfo("nvme")
    )
    assert runtime.fp8_capable is False
    config = RuntimeConfig(
        device=torch.device("cpu"),
        app_data_dir=tmp_path / "app_data",
        default_models_dir=tmp_path / "models",
        outputs_dir=tmp_path / "outputs",
        settings_file=tmp_path / "settings.json",
        ltx_api_base_url="https://api.ltx.video",
        use_sage_attention=False,
        camera_motion_prompts={},
        default_negative_prompt="",
        dev_mode=False,
        backend_port=8000,
        **runtime.runtime_config_kwargs(),
    )
    assert config.fp8_capable is False
    filt = build_local_job_filter(
        config,
        model_id="ltx-2.5-22b-distilled",
        free_vram_bytes=None,
        mps_pool_bytes=None,
    )
    assert filt.profile.fp8_capable is False
    assert budget_route("fast", filt.profile) == "legacy"


def test_resolve_local_runtime_carries_gpu_device_name() -> None:
    gpu = FakeGpuInfo()
    gpu.cuda_available = True
    gpu.vram_gb = 32
    gpu.vram_bytes = 32 * 1024**3
    gpu.total_ram_bytes = 64 * 1024**3
    gpu.gpu_name = "NVIDIA GeForce RTX 5090"

    runtime = resolve_local_runtime(
        gpu, system="Windows", fp8_capable=True, storage_info=FakeStorageInfo("nvme")
    )
    assert runtime.gpu_device_name == "NVIDIA GeForce RTX 5090"

    gpu.gpu_name = None
    unknown = resolve_local_runtime(
        gpu, system="Windows", fp8_capable=True, storage_info=FakeStorageInfo("nvme")
    )
    assert unknown.gpu_device_name is None


def test_resolve_local_runtime_supports_12gb_card_reporting_11_point_99() -> None:
    """The floor drop serves the cards it targeted: derive, don't hand-feed.

    The integer figure is computed from the bytes exactly as production does
    (``total_memory // 1024**3``) — a literal ``vram_gb=12`` here is what let
    this ship, since no production card reports an exact integer.
    """
    vram_gb = _TWELVE_GB_CARD_VRAM_BYTES // (1024**3)
    assert vram_gb == 11
    assert _resolve_mode(_TWELVE_GB_CARD_VRAM_BYTES, vram_gb, "Windows") == "streaming_models_loading"
    assert _resolve_mode(_TWELVE_GB_CARD_VRAM_BYTES, vram_gb, "Linux") == "unsupported"


def test_resolve_local_runtime_supports_11gb_card_reporting_10_point_99() -> None:
    """GTX 1080 Ti / RTX 2080 Ti class: the floor must not repeat the 12 GB bug."""
    vram_gb = _ELEVEN_GB_CARD_VRAM_BYTES // (1024**3)
    assert vram_gb == 10
    assert _resolve_mode(_ELEVEN_GB_CARD_VRAM_BYTES, vram_gb, "Windows") == "streaming_models_loading"
    assert _resolve_mode(_ELEVEN_GB_CARD_VRAM_BYTES, vram_gb, "Linux") == "unsupported"


def test_resolve_local_runtime_supports_10gb_card_reporting_9_point_99() -> None:
    vram_gb = _TEN_GB_CARD_VRAM_BYTES // (1024**3)
    assert vram_gb == 9
    assert _resolve_mode(_TEN_GB_CARD_VRAM_BYTES, vram_gb, "Windows") == "streaming_models_loading"
    assert _resolve_mode(_TEN_GB_CARD_VRAM_BYTES, vram_gb, "Linux") == "unsupported"


def test_resolve_local_runtime_rejects_below_early_out_floor() -> None:
    vram_bytes = int(8.8 * 1024**3)
    vram_gb = vram_bytes // (1024**3)
    assert _resolve_mode(vram_bytes, vram_gb, "Windows") == "unsupported"
    assert _resolve_mode(vram_bytes, vram_gb, "Linux") == "unsupported"


def test_resolve_local_runtime_falls_back_to_integer_when_bytes_missing() -> None:
    assert _resolve_mode(None, 9, "Windows") == "streaming_models_loading"
    assert _resolve_mode(None, 8, "Windows") == "unsupported"


def test_resolve_local_runtime_rejects_when_vram_fully_unknown() -> None:
    assert _resolve_mode(None, None, "Windows") == "unsupported"


def _mac_runtime_mode(macos_major: int | None) -> str:
    gpu = FakeGpuInfo()
    gpu.mps_available = True
    gpu.vram_gb = 64
    gpu.total_ram_bytes = 64 * 1024**3
    return resolve_local_runtime(
        gpu, system="Darwin", fp8_capable=False, storage_info=FakeStorageInfo(None), macos_major=macos_major
    ).mode


def test_resolve_local_runtime_gates_old_macos() -> None:
    assert _mac_runtime_mode(14) == "unsupported"
    assert _mac_runtime_mode(26) == "streaming_models_loading"
    assert _mac_runtime_mode(None) == "streaming_models_loading"
