"""Tile blend weights must reach MPS with a blocking copy (see ``mps_tiling_weights_sync``)."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from ltx_core import modality_tiling

from services.patches import mps_tiling_weights_sync as patch


class _HostWeights:
    """Stands in for the host tensor and records how it is moved."""

    def __init__(self, moves: list[dict[str, Any]]) -> None:
        self._moves = moves

    def to(self, **kwargs: Any) -> _HostWeights:
        self._moves.append(kwargs)
        return self


class _Like:
    def __init__(self, device: str) -> None:
        self.dtype = torch.bfloat16
        self.device = torch.device(device)


def _like(device: str) -> torch.Tensor:
    return cast(torch.Tensor, _Like(device))


def test_patch_replaces_modality_tiling_device_weights() -> None:
    assert modality_tiling._device_weights is patch._device_weights_ordered  # pyright: ignore[reportPrivateUsage]


def test_mps_upload_is_never_non_blocking() -> None:
    moves: list[dict[str, Any]] = []
    host = _HostWeights(moves)
    patch._device_weights_ordered({}, lambda: cast(torch.Tensor, host), _like("mps"))  # pyright: ignore[reportPrivateUsage]
    assert {"device": torch.device("mps")} in moves
    assert not any(move.get("non_blocking") for move in moves)


def test_mps_weights_are_built_and_uploaded_once_per_plan() -> None:
    moves: list[dict[str, Any]] = []
    built: list[int] = []

    def build() -> torch.Tensor:
        built.append(1)
        return cast(torch.Tensor, _HostWeights(moves))

    cache: dict[tuple[torch.dtype, torch.device], torch.Tensor] = {}
    first = patch._device_weights_ordered(cache, build, _like("mps"))  # pyright: ignore[reportPrivateUsage]
    second = patch._device_weights_ordered(cache, build, _like("mps"))  # pyright: ignore[reportPrivateUsage]
    assert second is first
    assert len(built) == 1


def test_cuda_keeps_the_upstream_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    sentinel = torch.zeros(1)

    def upstream(_cache: object, _weights: object, like: torch.Tensor) -> torch.Tensor:
        calls.append(like.device.type)
        return sentinel

    monkeypatch.setattr(patch, "_orig_device_weights", upstream)
    result = patch._device_weights_ordered({}, lambda: torch.ones(1), _like("cuda"))  # pyright: ignore[reportPrivateUsage]
    assert result is sentinel
    assert calls == ["cuda"]
