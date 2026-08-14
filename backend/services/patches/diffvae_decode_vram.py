"""Free denoise weights before DiffVAE decode so 32 GB CUDA can finish.

Full-resident 2.5 keeps the ~23 GiB fp8 transformer on GPU through decode
(``diffusion_stage_cache`` holds it; even without the cache, CUDA reserved
memory stays at device capacity). DiffVAE then builds on ``free=0`` and
neighborhood attention never returns — 32 GB Windows hung past 6 minutes.

``DistilledPipeline`` also resolves ``AUTO_TILING`` *before* denoise, while the
GPU is almost empty (~28 GiB free). Decode then runs after evict with ~6 GiB
driver-free — or, after a clean ``empty_cache``, ``mem_get_info`` reports ~30 GiB
free. Planning 1024×576 against that 30 GiB budget picks a near-full-frame tile;
the first NA slab then drives ``reserved`` to capacity and hangs. Cap the tile
budget so 540p always splits.

``DistilledPipeline`` calls ``VideoDecoder`` only after both denoise stages,
so evicting here cannot interrupt a mid-denoise checkout.

Remove once ltx-pipelines offloads the transformer before DiffVAE decode and
resolves AUTO_TILING against that post-offload budget.

Usage:
    import services.patches.diffvae_decode_vram  # noqa: F401
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from ltx_core.devices import cuda_activation_budget_bytes
from ltx_core.model.video_vae.model_configurator import is_diffusion_video_vae
from ltx_core.types import VideoLatentShape
from ltx_pipelines.utils.blocks import VideoDecoder
from ltx_pipelines.utils.helpers import cleanup_memory, tiling_config_for_vae

from services.patches import diffusion_stage_cache

logger = logging.getLogger(__name__)

# After evict, Windows ``mem_get_info`` can still report ~30 GiB free. AUTO_TILING
# then covers 540p in one tile. 8 GiB is above the ~2.4 GiB min 540p/5s tile.
_CUDA_DIFFVAE_TILE_BUDGET_CAP_BYTES = 8 * 1024**3

_orig_video_decoder_call = VideoDecoder.__call__


def _cuda_tile_budget_bytes(measured: int) -> int:
    return min(max(int(measured), 0), _CUDA_DIFFVAE_TILE_BUDGET_CAP_BYTES)


def _release_denoise_weights() -> None:
    diffusion_stage_cache.evict()
    cleanup_memory()
    logger.info("Freed resident transformer before DiffVAE decode")


def _pixel_shape_from_latent(latent: torch.Tensor) -> tuple[int, int, int]:
    """Pixel (height, width, frames) from a transformer latent ``(B, C, T, H, W)``.

    Uses ``VIDEO_SCALE_FACTORS`` (8×32×32), not DiffVAE ``pixel_scale``. The latter
    is stage-4-feature → pixel (8 spatial) and would report 540p as 256×144.
    """
    pixels = VideoLatentShape.from_torch_shape(latent.shape).upscale()
    return int(pixels.height), int(pixels.width), int(pixels.frames)


def _replace_tiling_arg(
    args: tuple[Any, ...], kwargs: dict[str, Any], tiling_config: Any
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if "tiling_config" in kwargs:
        return args, {**kwargs, "tiling_config": tiling_config}
    if len(args) >= 2:
        return (args[0], tiling_config, *args[2:]), kwargs
    return args, {**kwargs, "tiling_config": tiling_config}


def _with_post_evict_cuda_tiling(
    decoder: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Re-resolve DiffVAE AUTO tiles against the CUDA budget after transformer evict."""
    device = getattr(decoder, "_device", None)
    if not isinstance(device, torch.device) or device.type != "cuda":
        return args, kwargs
    latent = kwargs.get("latent", args[0] if args else None)
    if not isinstance(latent, torch.Tensor) or latent.ndim != 5:
        return args, kwargs
    checkpoint = getattr(decoder, "checkpoint_path", None)
    if not isinstance(checkpoint, str):
        checkpoint = getattr(decoder, "_checkpoint_path", None)
    if not isinstance(checkpoint, str) or not is_diffusion_video_vae(checkpoint):
        return args, kwargs

    height, width, num_frames = _pixel_shape_from_latent(latent)
    measured = int(cuda_activation_budget_bytes(device))
    budget = _cuda_tile_budget_bytes(measured)
    recommend_kwargs: dict[str, Any] = {
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "device": device,
        "free_bytes": budget,
    }
    optimization = getattr(decoder, "diffvae_optimization", None)
    if optimization is not None:
        recommend_kwargs["diffvae_optimization"] = optimization
    tiling = tiling_config_for_vae(checkpoint, **recommend_kwargs)
    logger.info(
        "DiffVAE CUDA tiling after evict: measured=%.1f GiB budget=%.1f GiB "
        "%sx%sx%s t/h/w tiles=%s/%s/%s",
        measured / 1024**3,
        budget / 1024**3,
        width,
        height,
        num_frames,
        getattr(getattr(tiling, "frames", None), "tile_size", "?"),
        getattr(getattr(tiling, "height", None), "tile_size", "?"),
        getattr(getattr(tiling, "width", None), "tile_size", "?"),
    )
    return _replace_tiling_arg(args, kwargs, tiling)


def _patched_video_decoder_call(self: VideoDecoder, *args: Any, **kwargs: Any) -> Any:
    _release_denoise_weights()
    args, kwargs = _with_post_evict_cuda_tiling(self, args, kwargs)
    return _orig_video_decoder_call(self, *args, **kwargs)


VideoDecoder.__call__ = _patched_video_decoder_call  # type: ignore[method-assign]
