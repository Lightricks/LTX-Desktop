"""Streaming pinned-host alloc must not surface as a CUDA VRAM OOM."""

from __future__ import annotations

import pytest
import torch
from ltx_core.block_streaming import utils as bs_utils

import services.patches.pinned_pool_fix as patch


@pytest.fixture(autouse=True)
def _reset_windows_log_flag() -> None:
    patch._windows_pageable_logged = False
    yield
    patch._windows_pageable_logged = False


def test_patch_rebinds_alloc_buffer() -> None:
    assert bs_utils.alloc_buffer is patch._patched_alloc_buffer


def test_require_attr_fails_loudly_when_symbol_missing() -> None:
    with pytest.raises(RuntimeError, match="definitely_missing_symbol not found"):
        patch._require_attr("definitely_missing_symbol")


def test_unpinned_allocation_unchanged() -> None:
    buf = patch._patched_alloc_buffer(64, torch.device("cpu"), pin_memory=False)
    assert buf.shape == (64,)
    assert buf.dtype == torch.uint8
    assert not buf.is_pinned()


def test_windows_pin_request_uses_pageable_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(patch.sys, "platform", "win32")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def register_must_not_run(_nbytes: int) -> torch.Tensor | None:
        raise AssertionError("cudaHostRegister path must be skipped on Windows")

    monkeypatch.setattr(patch, "_alloc_pinned_exact_cleared", register_must_not_run)

    buf = patch._patched_alloc_buffer(1024, torch.device("cpu"), pin_memory=True)
    assert buf.numel() == 1024
    assert buf.dtype == torch.uint8
    assert not buf.is_pinned()


def test_windows_allocate_layout_views_stays_pageable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(patch.sys, "platform", "win32")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    views = bs_utils.allocate_layout_views({"k": (torch.Size([4, 4]), torch.bfloat16)}, pin_memory=True)
    assert not views["k"].is_pinned()
    assert views["k"].shape == torch.Size([4, 4])


def test_linux_clears_sticky_error_after_register_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(patch.sys, "platform", "linux")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(patch, "_orig_alloc_pinned_exact", lambda _nbytes: None)
    cleared: list[bool] = []
    monkeypatch.setattr(patch, "_clear_cuda_sticky_error", lambda: cleared.append(True))

    real_empty = torch.empty

    def empty_unpinned(*args: object, pin_memory: bool = False, **kwargs: object) -> torch.Tensor:
        return real_empty(*args, pin_memory=False, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(torch, "empty", empty_unpinned)

    buf = patch._patched_alloc_buffer(32, torch.device("cpu"), pin_memory=True)
    assert buf.numel() == 32
    assert cleared == [True]


def test_linux_pin_oom_falls_back_to_pageable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(patch.sys, "platform", "linux")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(patch, "_alloc_pinned_exact_cleared", lambda _nbytes: None)
    cleared: list[bool] = []
    monkeypatch.setattr(patch, "_clear_cuda_sticky_error", lambda: cleared.append(True))

    real_empty = torch.empty

    def empty_maybe_pin(*args: object, pin_memory: bool = False, **kwargs: object) -> torch.Tensor:
        if pin_memory:
            raise RuntimeError("CUDA error: out of memory")
        return real_empty(*args, pin_memory=False, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(torch, "empty", empty_maybe_pin)

    buf = patch._patched_alloc_buffer(128, torch.device("cpu"), pin_memory=True)
    assert buf.numel() == 128
    assert not buf.is_pinned()
    assert cleared == [True]


def test_linux_pin_non_oom_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(patch.sys, "platform", "linux")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(patch, "_alloc_pinned_exact_cleared", lambda _nbytes: None)

    real_empty = torch.empty

    def empty_maybe_pin(*args: object, pin_memory: bool = False, **kwargs: object) -> torch.Tensor:
        if pin_memory:
            raise RuntimeError("CUDA error: invalid argument")
        return real_empty(*args, pin_memory=False, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(torch, "empty", empty_maybe_pin)

    with pytest.raises(RuntimeError, match="invalid argument"):
        patch._patched_alloc_buffer(8, torch.device("cpu"), pin_memory=True)
