"""Unit tests for device utility functions in services.services_utils."""

from __future__ import annotations

import pytest

import torch

from services.services_utils import device_supports_fp8, is_rocm_device


def test_is_rocm_device_returns_false_without_hip(monkeypatch: pytest.MonkeyPatch) -> None:
    """is_rocm_device() returns False when torch.version.hip is None (standard CUDA/CPU build)."""
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    assert is_rocm_device() is False


def test_is_rocm_device_returns_true_with_hip_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """is_rocm_device() returns True when torch.version.hip contains a HIP version string."""
    monkeypatch.setattr(torch.version, "hip", "7.2.26024-f6f897bd3d", raising=False)
    assert is_rocm_device() is True


def test_device_supports_fp8_false_for_cpu() -> None:
    assert device_supports_fp8("cpu") is False


def test_device_supports_fp8_false_for_mps() -> None:
    assert device_supports_fp8("mps") is False


def test_device_supports_fp8_false_for_rocm(monkeypatch: pytest.MonkeyPatch) -> None:
    """FP8 is disabled for AMD ROCm — not hardware-accelerated on RDNA 3.x."""
    monkeypatch.setattr(torch.version, "hip", "7.2.26024-f6f897bd3d", raising=False)
    assert device_supports_fp8("cuda") is False


def test_device_supports_fp8_false_for_pre_ada_nvidia(monkeypatch: pytest.MonkeyPatch) -> None:
    """FP8 is disabled for NVIDIA GPUs older than Ada Lovelace (compute capability < 8.9)."""
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))
    assert device_supports_fp8("cuda") is False


def test_device_supports_fp8_true_for_ada_nvidia(monkeypatch: pytest.MonkeyPatch) -> None:
    """FP8 is enabled for NVIDIA Ada Lovelace (sm_89)."""
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 9))
    assert device_supports_fp8("cuda") is True


def test_device_supports_fp8_true_for_hopper_nvidia(monkeypatch: pytest.MonkeyPatch) -> None:
    """FP8 is enabled for NVIDIA Hopper (sm_90) and newer."""
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    assert device_supports_fp8("cuda") is True
