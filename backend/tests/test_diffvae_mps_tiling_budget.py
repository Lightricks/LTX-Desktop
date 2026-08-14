"""DiffVAE MPS tiling budget: upstream treats non-CUDA free_bytes as 0."""

from __future__ import annotations

import torch
from ltx_core.tiling import DimensionSizeConfig, TileSizeConfig, split_by_size

from services.ltx_pipeline_common import (
    diffvae_activation_budget_bytes,
    host_available_bytes,
    resolve_diffvae_free_bytes,
)
from services.patches.diffvae_mps_tiling_budget import _ensure_mps_width_split


def _size_config(*, width_size: int, width_overlap: int = 160) -> TileSizeConfig:
    return TileSizeConfig(
        frames=DimensionSizeConfig(tile_size=128, overlap=40),
        height=DimensionSizeConfig(tile_size=576, overlap=160),
        width=DimensionSizeConfig(tile_size=width_size, overlap=width_overlap),
    )


def _width_tiles(config: TileSizeConfig, width: int) -> int:
    return len(split_by_size(config.width.tile_size, config.width.overlap)(width).intervals)


def test_host_available_bytes_is_positive() -> None:
    assert host_available_bytes() > 0


def test_cpu_budget_is_positive() -> None:
    assert diffvae_activation_budget_bytes(torch.device("cpu")) > 0


def test_unset_non_cuda_free_bytes_uses_available_ram() -> None:
    budget = resolve_diffvae_free_bytes(torch.device("cpu"), None)
    assert budget is not None and budget > 0


def test_explicit_free_bytes_are_kept() -> None:
    assert resolve_diffvae_free_bytes(torch.device("cpu"), 3 * 1024**3) == 3 * 1024**3


def test_zero_non_cuda_free_bytes_are_replaced() -> None:
    # Upstream's MPS path passes 0, which makes usable_bytes=0 and raises before decode.
    budget = resolve_diffvae_free_bytes(torch.device("mps"), 0)
    assert budget is not None and budget > 0


def test_cuda_unset_budget_stays_none() -> None:
    # Let tiling_config_for_vae call cuda_activation_budget_bytes itself.
    assert resolve_diffvae_free_bytes(torch.device("cuda"), None) is None


def test_patch_rebinds_tiling_config_for_vae() -> None:
    import services.patches.diffvae_mps_tiling_budget as patch
    from ltx_pipelines.utils import helpers

    assert helpers.tiling_config_for_vae is patch._patched_tiling_config_for_vae


def test_mps_full_width_540p_becomes_two_width_tiles() -> None:
    original = _size_config(width_size=1024)
    clamped = _ensure_mps_width_split(original, width=1024, device=torch.device("mps"))
    assert clamped.width.tile_size == 608
    assert clamped.width.overlap == 160
    assert _width_tiles(clamped, 1024) == 2
    assert clamped.height == original.height
    assert clamped.frames == original.frames


def test_mps_already_split_width_is_unchanged() -> None:
    original = _size_config(width_size=608)
    clamped = _ensure_mps_width_split(original, width=1024, device=torch.device("mps"))
    assert clamped is original


def test_cuda_full_width_is_unchanged() -> None:
    original = _size_config(width_size=1024)
    clamped = _ensure_mps_width_split(original, width=1024, device=torch.device("cuda"))
    assert clamped is original


def test_mps_too_narrow_to_split_is_unchanged() -> None:
    original = _size_config(width_size=320)
    clamped = _ensure_mps_width_split(original, width=320, device=torch.device("mps"))
    assert clamped is original


def test_patched_tiling_config_applies_mps_width_split(monkeypatch) -> None:
    import services.patches.diffvae_mps_tiling_budget as patch

    monkeypatch.setattr(patch, "_orig_tiling_config_for_vae", lambda *args, **kwargs: _size_config(width_size=1024))
    config = patch._patched_tiling_config_for_vae(
        "unused.safetensors",
        height=576,
        width=1024,
        num_frames=121,
        device=torch.device("mps"),
        free_bytes=17 * 1024**3,
    )
    assert config.width.tile_size == 608
    assert _width_tiles(config, 1024) == 2
