"""Monkey-patch: DiffVAE AUTO_TILING on non-CUDA devices.

``ltx_pipelines.utils.helpers.tiling_config_for_vae`` (and
``DiffusionVideoDecoder.recommended_tiling_config``) only query the CUDA
allocator. On MPS that yields ``free_bytes=0`` → ``usable_bytes=0`` →
``Cannot fit a DiffVAE decode tile under the memory budget`` before decode
starts. The min tile for a 1024×576×121 job is ~2.4 GiB; the Mac has the RAM,
upstream just never counts it.

Filling that budget lets AUTO_TILING pick a single full-width stage-5 tile
(1024 on 540p). Eager ``K=11`` neighborhood attention on MPS then silently
zeros the tail frames. After the budget is filled, this patch forces a 2-way
width split when MPS would otherwise decode in one width tile.

``DistilledPipeline`` resolves ``AUTO_TILING`` internally and never goes through
Desktop's ``resolve_tiling_config``, so callers cannot pass a budget.

Remove once ltx-pipelines/ltx-core query MPS/unified free memory themselves
and refuse full-width eager NA tiles on MPS.

Usage:
    import services.patches.diffvae_mps_tiling_budget  # noqa: F401
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import replace
from typing import Any

import torch
from ltx_core.model.video_vae.diffusion_video_decoder import DiffusionVideoDecoder
from ltx_core.tiling import DimensionSizeConfig, split_by_size
from ltx_pipelines.utils import helpers

from services.ltx_pipeline_common import resolve_diffvae_free_bytes

logger = logging.getLogger(__name__)

_logged_budget = False
_WIDTH_ALIGN = 32


def _width_tile_count(tile_size: int, overlap: int, width: int) -> int:
    if tile_size <= 0 or tile_size >= width or overlap >= tile_size:
        return 1
    return len(split_by_size(tile_size, overlap)(width).intervals)


def _ensure_mps_width_split(config: Any, *, width: object, device: torch.device) -> Any:
    """Force n_w >= 2 on MPS when AUTO_TILING would cover the full frame in one tile."""
    if device.type != "mps" or not isinstance(width, int) or width < 1:
        return config
    width_axis = getattr(config, "width", None)
    if width_axis is None:
        return config
    current = int(getattr(width_axis, "tile_size", 0) or 0)
    overlap = int(getattr(width_axis, "overlap", 0) or 0)
    if overlap <= 0:
        height_axis = getattr(config, "height", None)
        overlap = int(getattr(height_axis, "overlap", 0) or 0) if height_axis is not None else 0
    if overlap <= 0:
        return config
    n_w = _width_tile_count(current if current > 0 else width, overlap, width)
    if n_w >= 2:
        return config
    min_size = max(overlap + _WIDTH_ALIGN, 2 * overlap)
    if width <= min_size:
        return config
    raw = (width + overlap) // 2
    tile = ((raw + _WIDTH_ALIGN - 1) // _WIDTH_ALIGN) * _WIDTH_ALIGN
    tile = min(max(tile, min_size), width - _WIDTH_ALIGN)
    if tile <= overlap or tile >= width:
        return config
    new_n = _width_tile_count(tile, overlap, width)
    if new_n < 2:
        return config
    logger.info("DiffVAE MPS width clamp: %s → %s (n_w %s → %s)", current or width, tile, n_w, new_n)
    return replace(config, width=DimensionSizeConfig(tile_size=tile, overlap=overlap))


assert "free_bytes" in inspect.signature(helpers.tiling_config_for_vae).parameters, (
    "tiling_config_for_vae missing free_bytes — patch needs updating."
)
assert hasattr(DiffusionVideoDecoder, "recommended_tiling_config"), (
    "DiffusionVideoDecoder.recommended_tiling_config not found — patch needs updating."
)
assert "free_bytes" in inspect.signature(DiffusionVideoDecoder.recommended_tiling_config).parameters, (
    "recommended_tiling_config missing free_bytes — patch needs updating."
)

_orig_tiling_config_for_vae = helpers.tiling_config_for_vae
_orig_recommended_tiling_config = DiffusionVideoDecoder.recommended_tiling_config


def _maybe_log_injected_budget(device: torch.device, free_bytes: int) -> None:
    global _logged_budget
    if _logged_budget:
        return
    _logged_budget = True
    logger.info(
        "DiffVAE tiling uses available system RAM on %s: %.1f GiB "
        "(upstream non-CUDA budget is 0).",
        device.type,
        free_bytes / (1024**3),
    )


def _patched_tiling_config_for_vae(*args: Any, **kwargs: Any) -> Any:
    device = kwargs.get("device")
    if not isinstance(device, torch.device):
        device = helpers.get_device()
        kwargs["device"] = device
    raw = kwargs.get("free_bytes")
    incoming = raw if isinstance(raw, int) else None
    filled = resolve_diffvae_free_bytes(device, incoming)
    if (incoming is None or incoming == 0) and filled is not None and filled > 0 and device.type != "cuda":
        _maybe_log_injected_budget(device, filled)
    kwargs["free_bytes"] = filled
    return _ensure_mps_width_split(
        _orig_tiling_config_for_vae(*args, **kwargs),
        width=kwargs.get("width"),
        device=device,
    )


def _patched_recommended_tiling_config(self: DiffusionVideoDecoder, *args: Any, **kwargs: Any) -> Any:
    device = next(self.parameters()).device
    raw = kwargs.get("free_bytes")
    incoming = raw if isinstance(raw, int) else None
    filled = resolve_diffvae_free_bytes(device, incoming)
    if (incoming is None or incoming == 0) and filled is not None and filled > 0 and device.type != "cuda":
        _maybe_log_injected_budget(device, filled)
    kwargs["free_bytes"] = filled
    return _ensure_mps_width_split(
        _orig_recommended_tiling_config(self, *args, **kwargs),
        width=kwargs.get("width"),
        device=device,
    )


helpers.tiling_config_for_vae = _patched_tiling_config_for_vae
DiffusionVideoDecoder.recommended_tiling_config = _patched_recommended_tiling_config  # type: ignore[method-assign]
