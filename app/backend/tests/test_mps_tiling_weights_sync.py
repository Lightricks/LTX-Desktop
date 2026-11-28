"""Tile blend weights must not take upstream's non-blocking upload off CUDA (see ``mps_tiling_weights_sync``)."""

from __future__ import annotations

import pytest
import torch
from ltx_core import modality_tiling

from services.patches import mps_tiling_weights_sync as patch


def _upstream_must_not_run(*_args: object) -> torch.Tensor:
    raise AssertionError("a non-CUDA upload went through upstream's non-blocking copy")


def test_patch_is_installed_on_modality_tiling() -> None:
    assert modality_tiling._device_weights is patch._device_weights_ordered  # pyright: ignore[reportPrivateUsage]


def test_non_cuda_upload_skips_upstream_and_keeps_values_and_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(patch, "_orig_device_weights", _upstream_must_not_run)
    cache: dict[tuple[torch.dtype, torch.device], torch.Tensor] = {}
    like = torch.zeros(1, dtype=torch.bfloat16)

    result = patch._device_weights_ordered(cache, lambda: torch.tensor([0.0, 0.5, 1.0]), like)  # pyright: ignore[reportPrivateUsage]

    assert result.dtype == torch.bfloat16
    assert result.tolist() == [0.0, 0.5, 1.0]
    assert cache[(torch.bfloat16, like.device)] is result


def test_weights_are_built_once_per_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(patch, "_orig_device_weights", _upstream_must_not_run)
    built: list[int] = []

    def build() -> torch.Tensor:
        built.append(1)
        return torch.ones(3)

    cache: dict[tuple[torch.dtype, torch.device], torch.Tensor] = {}
    like = torch.zeros(1, dtype=torch.bfloat16)
    first = patch._device_weights_ordered(cache, build, like)  # pyright: ignore[reportPrivateUsage]
    second = patch._device_weights_ordered(cache, build, like)  # pyright: ignore[reportPrivateUsage]

    assert second is first
    assert len(built) == 1


def test_cuda_keeps_the_upstream_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    # No CUDA device here, so the backend rule is forced on for a CPU tensor.
    sentinel = torch.full((1,), 7.0)
    monkeypatch.setattr(patch, "allow_async_transfer", lambda _device: True)  # pyright: ignore[reportUnknownArgumentType,reportUnknownLambdaType]
    monkeypatch.setattr(patch, "_orig_device_weights", lambda _cache, _weights, _like: sentinel)

    result = patch._device_weights_ordered({}, lambda: torch.ones(1), torch.zeros(1))  # pyright: ignore[reportPrivateUsage]

    assert result is sentinel
