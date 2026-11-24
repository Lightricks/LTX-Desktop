"""Pinned host buffers must degrade to pageable memory, not crash or surface as a CUDA VRAM OOM."""

from __future__ import annotations

import pytest
import torch
from ltx_core.loader import buffers, registry, state_loader, streaming_builder

import services.patches.pinned_pool_fix as patch

_PINNED_REQUEST_BYTES = 4 << 20


@pytest.fixture(autouse=True)
def _reset_windows_log_flag() -> None:
    patch._windows_pageable_logged = False
    yield
    patch._windows_pageable_logged = False


def test_every_importer_of_host_bytes_is_rebound() -> None:
    """A module that kept the upstream reference would still crash on Windows."""
    assert buffers.host_bytes is patch._patched_host_bytes
    for module in (registry, state_loader, streaming_builder):
        assert module.host_bytes is patch._patched_host_bytes


def test_unpinned_request_is_upstream_behavior() -> None:
    buf = patch._patched_host_bytes(64)
    assert buf.shape == (64,)
    assert buf.dtype == torch.uint8
    assert not buf.is_pinned()


def test_windows_pin_request_uses_pageable_memory_without_touching_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(patch.sys, "platform", "win32")

    def upstream_must_not_run(_nbytes: int, *, pin: bool = False) -> torch.Tensor:
        raise AssertionError("upstream pins through mmap.MAP_ANONYMOUS, which Windows lacks")

    monkeypatch.setattr(patch, "_orig_host_bytes", upstream_must_not_run)

    buf = patch._patched_host_bytes(_PINNED_REQUEST_BYTES, pin=True)

    assert buf.numel() == _PINNED_REQUEST_BYTES
    assert buf.dtype == torch.uint8
    assert not buf.is_pinned()


def test_linux_registration_failure_clears_sticky_error_and_falls_back_to_pageable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(patch.sys, "platform", "linux")
    cleared: list[bool] = []
    monkeypatch.setattr(patch, "_clear_cuda_sticky_error", lambda: cleared.append(True))

    def failing_register(nbytes: int, *, pin: bool = False) -> torch.Tensor:
        raise RuntimeError(f"could not page-lock {nbytes} host bytes")

    monkeypatch.setattr(patch, "_orig_host_bytes", failing_register)

    buf = patch._patched_host_bytes(_PINNED_REQUEST_BYTES, pin=True)

    assert buf.numel() == _PINNED_REQUEST_BYTES
    assert not buf.is_pinned()
    assert cleared == [True]


def test_linux_pinned_pool_cuda_oom_falls_back_to_pageable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(patch.sys, "platform", "linux")
    monkeypatch.setattr(patch, "_clear_cuda_sticky_error", lambda: None)

    def oom_pin(_nbytes: int, *, pin: bool = False) -> torch.Tensor:
        raise RuntimeError("CUDA error: out of memory")

    monkeypatch.setattr(patch, "_orig_host_bytes", oom_pin)

    buf = patch._patched_host_bytes(128, pin=True)

    assert buf.numel() == 128
    assert not buf.is_pinned()


def test_linux_unrelated_runtime_error_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(patch.sys, "platform", "linux")

    def invalid_argument(_nbytes: int, *, pin: bool = False) -> torch.Tensor:
        raise RuntimeError("CUDA error: invalid argument")

    monkeypatch.setattr(patch, "_orig_host_bytes", invalid_argument)

    with pytest.raises(RuntimeError, match="invalid argument"):
        patch._patched_host_bytes(8, pin=True)
