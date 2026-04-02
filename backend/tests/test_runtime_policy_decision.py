"""Tests for runtime policy decision helper."""

from __future__ import annotations

from runtime_config.runtime_policy import decide_force_api_generations


def test_darwin_always_forces_api() -> None:
    assert decide_force_api_generations(system="Darwin", cuda_available=True, vram_gb=24) is True
    assert decide_force_api_generations(system="Darwin", cuda_available=False, vram_gb=None) is True


def test_windows_without_cuda_forces_api() -> None:
    assert decide_force_api_generations(system="Windows", cuda_available=False, vram_gb=24) is True


def test_windows_with_low_vram_forces_api() -> None:
    assert decide_force_api_generations(system="Windows", cuda_available=True, vram_gb=14) is True


def test_windows_with_unknown_vram_forces_api() -> None:
    assert decide_force_api_generations(system="Windows", cuda_available=True, vram_gb=None) is True


def test_windows_with_required_vram_allows_local_mode() -> None:
    assert decide_force_api_generations(system="Windows", cuda_available=True, vram_gb=15) is False


def test_linux_without_cuda_forces_api() -> None:
    assert decide_force_api_generations(system="Linux", cuda_available=False, vram_gb=24) is True


def test_linux_with_low_vram_forces_api() -> None:
    assert decide_force_api_generations(system="Linux", cuda_available=True, vram_gb=14) is True


def test_linux_with_unknown_vram_forces_api() -> None:
    assert decide_force_api_generations(system="Linux", cuda_available=True, vram_gb=None) is True


def test_linux_with_required_vram_allows_local_mode() -> None:
    assert decide_force_api_generations(system="Linux", cuda_available=True, vram_gb=15) is False


def test_other_systems_fail_closed() -> None:
    assert decide_force_api_generations(system="FreeBSD", cuda_available=True, vram_gb=48) is True
