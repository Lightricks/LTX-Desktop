"""Tests for ROCm vs CUDA accelerator detection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from runtime_config.accelerator import (
    accelerator_backend,
    accelerator_supports_fp8,
    verify_rocm_execution,
)
from services.gpu_info.gpu_info_impl import GpuInfoImpl
from services.services_utils import device_supports_fp8


def test_accelerator_backend_rocm_when_hip_is_set(monkeypatch) -> None:
    import runtime_config.accelerator as accel

    monkeypatch.setattr(accel.torch.version, "hip", "6.3.0", raising=False)
    monkeypatch.setattr(accel.torch.cuda, "is_available", lambda: True)
    assert accelerator_backend() == "rocm"


def test_hip_build_without_a_visible_device_cannot_advertise_rocm(monkeypatch) -> None:
    monkeypatch.setattr(torch.version, "hip", "10.0", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert accelerator_backend() == "cpu"
    with pytest.raises(RuntimeError, match="cannot see an AMD GPU"):
        verify_rocm_execution()


def test_amd_startup_rejects_a_visible_device_in_a_non_hip_build(monkeypatch) -> None:
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="without HIP"):
        verify_rocm_execution()


def test_amd_startup_reports_failed_kernels_on_an_enumerated_gpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.version, "hip", "10.0", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Radeon 8060S")

    def fail_allocation(*args: object, **kwargs: object) -> None:
        raise RuntimeError("invalid device function")

    monkeypatch.setattr(torch, "ones", fail_allocation)
    with pytest.raises(RuntimeError, match=r"Radeon 8060S.*invalid device function"):
        verify_rocm_execution()


def test_amd_telemetry_reads_used_gpu_memory_from_hip(monkeypatch) -> None:
    gib = 1024**3
    monkeypatch.setattr(torch.version, "hip", "10.0", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Radeon 8060S")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (60 * gib, 64 * gib))
    assert GpuInfoImpl().get_gpu_info() == {
        "name": "Radeon 8060S", "vram": 64 * 1024, "vramUsed": 4 * 1024,
    }


def test_accelerator_backend_cuda_when_hip_is_none(monkeypatch) -> None:
    import runtime_config.accelerator as accel

    monkeypatch.setattr(accel.torch.version, "hip", None, raising=False)
    monkeypatch.setattr(accel.torch.cuda, "is_available", lambda: True)
    assert accelerator_backend() == "cuda"


def test_accelerator_backend_mps_when_no_cuda(monkeypatch) -> None:
    import runtime_config.accelerator as accel

    monkeypatch.setattr(accel.torch.version, "hip", None, raising=False)
    monkeypatch.setattr(accel.torch.cuda, "is_available", lambda: False)
    mps = getattr(accel.torch.backends, "mps", None)
    if mps is None:
        monkeypatch.setattr(
            accel.torch.backends, "mps", SimpleNamespace(is_available=lambda: True), raising=False
        )
    else:
        monkeypatch.setattr(mps, "is_available", lambda: True)
    assert accelerator_backend() == "mps"


def test_accelerator_backend_cpu_fallback(monkeypatch) -> None:
    import runtime_config.accelerator as accel

    monkeypatch.setattr(accel.torch.version, "hip", None, raising=False)
    monkeypatch.setattr(accel.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(accel.torch.backends.mps, "is_available", lambda: False)
    assert accelerator_backend() == "cpu"


def test_device_supports_fp8_false_on_mps() -> None:
    assert device_supports_fp8("mps") is False
    assert device_supports_fp8(SimpleNamespace(type="mps")) is False


def test_device_supports_fp8_false_on_cpu() -> None:
    assert device_supports_fp8("cpu") is False


def test_device_supports_fp8_true_on_cuda_not_rocm(monkeypatch) -> None:
    monkeypatch.setattr(
        "runtime_config.accelerator.accelerator_supports_fp8", lambda device_index=0: True
    )
    assert device_supports_fp8("cuda") is True
    assert device_supports_fp8(SimpleNamespace(type="cuda")) is True


def test_device_supports_fp8_false_when_accelerator_lacks_fp8(monkeypatch) -> None:
    monkeypatch.setattr(
        "runtime_config.accelerator.accelerator_supports_fp8", lambda device_index=0: False
    )
    assert device_supports_fp8("cuda") is False
    assert device_supports_fp8(SimpleNamespace(type="cuda")) is False


def test_accelerator_supports_fp8_true_on_ada_and_newer(monkeypatch) -> None:
    import runtime_config.accelerator as accel

    monkeypatch.setattr(accel.torch.version, "hip", None, raising=False)
    monkeypatch.setattr(accel.torch.cuda, "is_available", lambda: True)
    for capability in [(8, 9), (9, 0), (10, 0)]:
        monkeypatch.setattr(
            accel.torch.cuda, "get_device_capability", lambda device=0, cap=capability: cap
        )
        assert accelerator_supports_fp8() is True


def test_accelerator_supports_fp8_false_on_pre_ada(monkeypatch) -> None:
    import runtime_config.accelerator as accel

    monkeypatch.setattr(accel.torch.version, "hip", None, raising=False)
    monkeypatch.setattr(accel.torch.cuda, "is_available", lambda: True)
    for capability in [(7, 5), (8, 0), (8, 6)]:
        monkeypatch.setattr(
            accel.torch.cuda, "get_device_capability", lambda device=0, cap=capability: cap
        )
        assert accelerator_supports_fp8() is False


def test_accelerator_supports_fp8_false_when_backend_is_not_cuda(monkeypatch) -> None:
    for backend in ("rocm", "mps", "cpu"):
        monkeypatch.setattr(
            "runtime_config.accelerator.accelerator_backend", lambda b=backend: b
        )
        assert accelerator_supports_fp8() is False


def test_accelerator_supports_fp8_false_on_probe_failure(monkeypatch) -> None:
    import runtime_config.accelerator as accel

    monkeypatch.setattr(accel.torch.version, "hip", None, raising=False)
    monkeypatch.setattr(accel.torch.cuda, "is_available", lambda: True)

    def _boom(device: object = 0) -> tuple[int, int]:
        raise RuntimeError("no NVML")

    monkeypatch.setattr(accel.torch.cuda, "get_device_capability", _boom)
    assert accelerator_supports_fp8() is False


def test_accelerator_supports_fp8_probes_device_zero(monkeypatch) -> None:
    import runtime_config.accelerator as accel

    monkeypatch.setattr(accel.torch.version, "hip", None, raising=False)
    monkeypatch.setattr(accel.torch.cuda, "is_available", lambda: True)
    seen: list[object] = []

    def _capture(device: object = None) -> tuple[int, int]:
        seen.append(device)
        return (8, 9)

    monkeypatch.setattr(accel.torch.cuda, "get_device_capability", _capture)
    assert accelerator_supports_fp8() is True
    assert seen == [0]


def test_accelerator_supports_fp8_probes_given_index(monkeypatch) -> None:
    import runtime_config.accelerator as accel

    monkeypatch.setattr(accel.torch.version, "hip", None, raising=False)
    monkeypatch.setattr(accel.torch.cuda, "is_available", lambda: True)
    seen: list[object] = []

    def _capture(device: object = None) -> tuple[int, int]:
        seen.append(device)
        return (8, 9)

    monkeypatch.setattr(accel.torch.cuda, "get_device_capability", _capture)
    assert accelerator_supports_fp8(1) is True
    assert seen == [1]


def test_device_supports_fp8_probes_the_passed_device_index(monkeypatch) -> None:
    import torch

    seen: list[object] = []

    def _capture(device_index: object = 0) -> bool:
        seen.append(device_index)
        return True

    monkeypatch.setattr("runtime_config.accelerator.accelerator_supports_fp8", _capture)
    assert device_supports_fp8("cuda:1") is True
    assert device_supports_fp8(torch.device("cuda:2")) is True
    assert device_supports_fp8("cuda") is True
    assert device_supports_fp8(torch.device("cuda")) is True
    assert seen == [1, 2, 0, 0]


def test_gpu_info_impl_free_vram_none_when_cuda_unavailable(monkeypatch) -> None:
    import services.gpu_info.gpu_info_impl as gpu_info_impl

    monkeypatch.setattr(gpu_info_impl.torch.cuda, "is_available", lambda: False)
    assert GpuInfoImpl().get_free_cuda_vram_bytes() is None


def test_gpu_info_impl_free_vram_returns_free_bytes(monkeypatch) -> None:
    import services.gpu_info.gpu_info_impl as gpu_info_impl

    monkeypatch.setattr(gpu_info_impl.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        gpu_info_impl.torch.cuda, "mem_get_info", lambda device=0: (15 * 1024**3, 16 * 1024**3)
    )
    assert GpuInfoImpl().get_free_cuda_vram_bytes() == 15 * 1024**3


def test_gpu_info_impl_free_vram_none_on_probe_failure(monkeypatch) -> None:
    import services.gpu_info.gpu_info_impl as gpu_info_impl

    monkeypatch.setattr(gpu_info_impl.torch.cuda, "is_available", lambda: True)

    def _boom(device: object = 0) -> tuple[int, int]:
        raise RuntimeError("no NVML")

    monkeypatch.setattr(gpu_info_impl.torch.cuda, "mem_get_info", _boom)
    assert GpuInfoImpl().get_free_cuda_vram_bytes() is None


def test_gpu_info_impl_live_allocated_none_when_cuda_unavailable(monkeypatch) -> None:
    import services.gpu_info.gpu_info_impl as gpu_info_impl

    monkeypatch.setattr(gpu_info_impl.torch.cuda, "is_available", lambda: False)
    assert GpuInfoImpl().get_cuda_live_allocated_bytes() is None


def test_gpu_info_impl_live_allocated_returns_allocated_bytes(monkeypatch) -> None:
    import services.gpu_info.gpu_info_impl as gpu_info_impl

    monkeypatch.setattr(gpu_info_impl.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        gpu_info_impl.torch.cuda, "memory_allocated", lambda device=0: 4 * 1024**3
    )
    assert GpuInfoImpl().get_cuda_live_allocated_bytes() == 4 * 1024**3


def test_gpu_info_impl_live_allocated_none_on_probe_failure(monkeypatch) -> None:
    import services.gpu_info.gpu_info_impl as gpu_info_impl

    monkeypatch.setattr(gpu_info_impl.torch.cuda, "is_available", lambda: True)

    def _boom_allocated(device: object = 0) -> int:
        raise RuntimeError("no NVML")

    monkeypatch.setattr(gpu_info_impl.torch.cuda, "memory_allocated", _boom_allocated)
    assert GpuInfoImpl().get_cuda_live_allocated_bytes() is None


def test_gpu_info_impl_mps_pool_none_when_mps_unavailable(monkeypatch) -> None:
    import services.gpu_info.gpu_info_impl as gpu_info_impl

    monkeypatch.setattr(
        gpu_info_impl.torch.backends,
        "mps",
        SimpleNamespace(is_available=lambda: False),
        raising=False,
    )
    assert GpuInfoImpl().get_mps_pool_bytes() is None


def test_gpu_info_impl_mps_pool_returns_recommended_max(monkeypatch) -> None:
    import services.gpu_info.gpu_info_impl as gpu_info_impl

    monkeypatch.setattr(
        gpu_info_impl.torch.backends,
        "mps",
        SimpleNamespace(is_available=lambda: True),
        raising=False,
    )
    monkeypatch.setattr(
        gpu_info_impl.torch,
        "mps",
        SimpleNamespace(recommended_max_memory=lambda: 115448725504),
        raising=False,
    )
    assert GpuInfoImpl().get_mps_pool_bytes() == 115448725504


def test_gpu_info_impl_mps_pool_none_on_probe_failure(monkeypatch) -> None:
    import services.gpu_info.gpu_info_impl as gpu_info_impl

    def _boom() -> int:
        raise RuntimeError("no MPS")

    monkeypatch.setattr(
        gpu_info_impl.torch.backends,
        "mps",
        SimpleNamespace(is_available=lambda: True),
        raising=False,
    )
    monkeypatch.setattr(
        gpu_info_impl.torch, "mps", SimpleNamespace(recommended_max_memory=_boom), raising=False
    )
    assert GpuInfoImpl().get_mps_pool_bytes() is None
