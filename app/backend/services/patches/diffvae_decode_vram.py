"""Free denoise weights before DiffVAE decode so 32 GB CUDA can finish.

Full-resident 2.5 keeps the ~23 GiB fp8 transformer on GPU through decode
(``diffusion_stage_cache`` holds it; even without the cache, CUDA reserved
memory stays at device capacity). DiffVAE then builds on ``free=0`` and
neighborhood attention never returns — 32 GB Windows hung past 6 minutes.

``DistilledPipeline`` also resolves ``AUTO_TILING`` *before* denoise, while the
GPU is almost empty (~28 GiB free). Decode then runs after evict with ~6 GiB
driver-free — or, after a clean ``empty_cache``, ``mem_get_info`` reports ~30 GiB
free. Planning 1024×576 against that 30 GiB budget picks a near-full-frame tile;
the first NA slab then drives ``reserved`` to capacity and hangs.

Cap **tile size** (80f × 608px, the 540p layout that used to come from an 8 GiB
``free_bytes`` cap), not the whole job budget. Capping ``free_bytes`` at 8 GiB
left 1080p/10s unable to place a min tile after ~4 GiB of resident stage-4
features.

The hang is neighborhood-attention **tile geometry**, not CUDA reserved memory.
A tile larger than 80f × 608px is the layout that never returned on the 5090;
the same max applies on MPS. CUDA still re-resolves AUTO tiles against
``mem_get_info`` after transformer evict, then clamps. MPS has no equivalent
budget call, so it clamps the pipeline's AUTO / untiled config to the same
max. Do not gate the MPS clamp on unified free RAM — that unit-mix is what
wrongly 422'd Darwin jobs against the 5090 stream curve.

``DistilledPipeline`` calls ``VideoDecoder`` only after both denoise stages,
so evicting here cannot interrupt a mid-denoise checkout.

Remove once ltx-pipelines offloads the transformer before DiffVAE decode and
resolves AUTO_TILING against that post-offload budget with a max-tile constraint.

Usage:
    import services.patches.diffvae_decode_vram  # noqa: F401
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import torch
from ltx_core.devices import cuda_activation_budget_bytes
from ltx_core.model.video_vae import DimensionSizeConfig, TileSizeConfig
from ltx_core.model.video_vae.model_configurator import is_diffusion_video_vae
from ltx_core.types import VIDEO_SCALE_FACTORS, VideoLatentShape
from ltx_pipelines.utils.blocks import VideoDecoder
from ltx_pipelines.utils.helpers import cleanup_memory, tiling_config_for_vae

from services.ltx_pipeline_common import is_decode_tile_shortage
from services.patches import diffusion_stage_cache

logger = logging.getLogger(__name__)

# 8 GiB-budget 540p layout: temporal 80 (ConvVAE auto too), spatial width 608.
_MAX_DIFFVAE_TILE_FRAMES = 80
_MAX_DIFFVAE_TILE_SPATIAL = 608
_DEFAULT_DIFFVAE_FRAME_OVERLAP = 40
_DEFAULT_DIFFVAE_SPATIAL_OVERLAP = 160

# Share of the card the decode plan leaves unspent. The planner sizes the largest tile that
# fits the budget it is given, so a plan against the whole post-evict budget drives CUDA
# ``reserved`` to device capacity (``free`` 0.00). Allocator blocks are tile-shaped and do not
# fit each other, so at capacity every tile allocation flushes the cache and retries; 4K Refine
# windows then crawl for hours instead of minutes. Planning against less shrinks the temporal
# tile (its accumulator is the largest term that scales with it) and leaves room for those
# blocks. A card where this leaves no fitting tile plans against the whole budget as before.
_DECODE_HEADROOM_FRACTION = 0.15

# Experiment override, in GiB: plan against exactly min(measured, this) and skip the headroom
# fraction. At 4K x 97f the planner needs >=22.5 GiB for a min tile, and 15% headroom on a 32 GB
# card (~22.3 GiB) may not fit one. Unset keeps the headroom plan.
_DECODE_BUDGET_CAP_ENV = "LTX_DIFFVAE_DECODE_BUDGET_GIB"

# Debug: when set to a directory, every DiffVAE decode first saves its latent and the resolved
# tiling there, so a separate script can replay just the decode (and preview the latent) without
# re-running a 12-27 min denoise. Unset = off.
_DUMP_LATENT_DIR_ENV = "LTX_DIFFVAE_DUMP_LATENT_DIR"

_orig_video_decoder_call = VideoDecoder.__call__


def _snap_down(value: int, grid: int) -> int:
    return max(grid, (value // grid) * grid)


def _clamp_axis(
    tile_size: int,
    overlap: int,
    extent: int,
    max_tile: int,
    grid: int,
    default_overlap: int,
) -> DimensionSizeConfig:
    was_untiled = tile_size <= 0
    # Compare against the grid-snapped extent, not the raw one: a tile one cell short of an
    # 8k+1 extent would only split off a sub-grid remainder, so it counts as untiled. Explicit
    # near-full tiles are dropped on purpose; do not "fix" this to ``>= extent``.
    if extent <= max_tile and (was_untiled or tile_size >= _snap_down(extent, grid)):
        # Nothing forces a split. Return a truly untiled axis: upstream treats any
        # tile_size > 0 as tiled and rejects it unless overlap meets its floor, and
        # snapping an 8k+1 extent down to the grid would otherwise tile it by accident.
        return DimensionSizeConfig()
    size = extent if was_untiled else tile_size
    size = min(size, max_tile, extent)
    size = _snap_down(size, grid)
    if size > extent:
        size = _snap_down(extent, grid)
    if was_untiled:
        overlap = 0 if size >= extent else default_overlap
    if overlap >= size:
        overlap = _snap_down(size - grid, grid) if size > grid else 0
        if overlap >= size:
            overlap = 0
    return DimensionSizeConfig(tile_size=size, overlap=overlap)


def _clamp_diffvae_tiles(tiling: Any, height: int, width: int, num_frames: int) -> TileSizeConfig:
    """Force NA-safe max tiles. ``tile_size==0`` means the full axis extent."""
    frames = getattr(tiling, "frames", None)
    heights = getattr(tiling, "height", None)
    widths = getattr(tiling, "width", None)
    return TileSizeConfig(
        frames=_clamp_axis(
            int(getattr(frames, "tile_size", 0) or 0),
            int(getattr(frames, "overlap", 0) or 0),
            num_frames,
            _MAX_DIFFVAE_TILE_FRAMES,
            VIDEO_SCALE_FACTORS.time,
            _DEFAULT_DIFFVAE_FRAME_OVERLAP,
        ),
        height=_clamp_axis(
            int(getattr(heights, "tile_size", 0) or 0),
            int(getattr(heights, "overlap", 0) or 0),
            height,
            _MAX_DIFFVAE_TILE_SPATIAL,
            VIDEO_SCALE_FACTORS.height,
            _DEFAULT_DIFFVAE_SPATIAL_OVERLAP,
        ),
        width=_clamp_axis(
            int(getattr(widths, "tile_size", 0) or 0),
            int(getattr(widths, "overlap", 0) or 0),
            width,
            _MAX_DIFFVAE_TILE_SPATIAL,
            VIDEO_SCALE_FACTORS.width,
            _DEFAULT_DIFFVAE_SPATIAL_OVERLAP,
        ),
    )


def _decoder_label(decoder: Any) -> str:
    """Log name of the VAE this decoder loads: ``DiffVAE``, ``ConvVAE``, or ``video VAE`` when unknown."""
    checkpoint = getattr(decoder, "checkpoint_path", None)
    if not isinstance(checkpoint, str):
        checkpoint = getattr(decoder, "_checkpoint_path", None)
    if not isinstance(checkpoint, str):
        return "video VAE"
    try:
        return "DiffVAE" if is_diffusion_video_vae(checkpoint) else "ConvVAE"
    except Exception:  # noqa: BLE001  the label is for a log line only; it must never fail a generation
        return "video VAE"


def _release_denoise_weights(decoder: Any) -> None:
    diffusion_stage_cache.evict()
    cleanup_memory()
    logger.info("Freed resident transformer before %s decode", _decoder_label(decoder))


def _decode_budget_cap_bytes() -> int | None:
    raw = os.environ.get(_DECODE_BUDGET_CAP_ENV, "").strip()
    if not raw:
        return None
    try:
        gib = float(raw)
    except ValueError:
        logger.warning("Ignoring %s=%r: not a number", _DECODE_BUDGET_CAP_ENV, raw)
        return None
    return int(gib * 1024**3) if gib > 0 else None


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


def _device_total_bytes(device: torch.device) -> int:
    return int(torch.cuda.get_device_properties(device).total_memory)


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
    measured = max(int(cuda_activation_budget_bytes(device)), 0)
    cap = _decode_budget_cap_bytes()
    if cap is None:
        total = _device_total_bytes(device)
        headroom_budget = max(0, measured - int(_DECODE_HEADROOM_FRACTION * total))
    else:
        headroom_budget = min(measured, cap)
    optimization = getattr(decoder, "diffvae_optimization", None)

    def plan(budget: int) -> TileSizeConfig:
        recommend_kwargs: dict[str, Any] = {
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "device": device,
            "free_bytes": budget,
        }
        if optimization is not None:
            recommend_kwargs["diffvae_optimization"] = optimization
        return _clamp_diffvae_tiles(
            tiling_config_for_vae(checkpoint, **recommend_kwargs),
            height,
            width,
            num_frames,
        )

    budget = headroom_budget
    try:
        tiling = plan(budget)
    except ValueError as exc:
        if not is_decode_tile_shortage(exc):
            raise
        budget = measured
        tiling = plan(budget)
    logger.info(
        "DiffVAE CUDA tiling after evict: measured=%.1f GiB budget=%.1f GiB "
        "%sx%sx%s t/h/w tiles=%s/%s/%s",
        measured / 1024**3,
        budget / 1024**3,
        width,
        height,
        num_frames,
        tiling.frames.tile_size,
        tiling.height.tile_size,
        tiling.width.tile_size,
    )
    return _replace_tiling_arg(args, kwargs, tiling)


def _with_mps_clamped_tiles(
    decoder: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Apply the NA-safe max tile on MPS. No CUDA mem_get_info re-resolve.

    Same 80f × 608 clamp as CUDA: NA hang is tile layout, not device VRAM.
    """
    device = getattr(decoder, "_device", None)
    if not isinstance(device, torch.device) or device.type != "mps":
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
    existing = kwargs["tiling_config"] if "tiling_config" in kwargs else (args[1] if len(args) >= 2 else None)
    if existing is None or not hasattr(existing, "frames"):
        existing = TileSizeConfig(
            frames=DimensionSizeConfig(tile_size=0, overlap=0),
            height=DimensionSizeConfig(tile_size=0, overlap=0),
            width=DimensionSizeConfig(tile_size=0, overlap=0),
        )
    tiling = _clamp_diffvae_tiles(existing, height, width, num_frames)
    logger.info(
        "DiffVAE MPS tiling clamp %sx%sx%s t/h/w tiles=%s/%s/%s",
        width,
        height,
        num_frames,
        tiling.frames.tile_size,
        tiling.height.tile_size,
        tiling.width.tile_size,
    )
    return _replace_tiling_arg(args, kwargs, tiling)


def _axis_dict(axis: Any) -> dict[str, int]:
    return {
        "tile_size": int(getattr(axis, "tile_size", 0) or 0),
        "overlap": int(getattr(axis, "overlap", 0) or 0),
    }


def _dump_latent_for_replay(decoder: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    """Save the decode input before it runs, so a hung decode still leaves a replayable file.

    Never raises: a debug dump must not fail the generation.
    """
    raw_dir = os.environ.get(_DUMP_LATENT_DIR_ENV, "").strip()
    if not raw_dir:
        return
    try:
        latent = kwargs.get("latent", args[0] if args else None)
        if not isinstance(latent, torch.Tensor):
            return
        tiling = kwargs["tiling_config"] if "tiling_config" in kwargs else (args[1] if len(args) >= 2 else None)
        out_dir = Path(raw_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        shape = "x".join(str(d) for d in latent.shape)
        path = out_dir / f"diffvae_latent_{time.strftime('%Y%m%d_%H%M%S')}_{shape}.pt"
        torch.save(
            {
                "latent": latent.detach().cpu(),
                "checkpoint_path": getattr(decoder, "checkpoint_path", None),
                "dtype": str(getattr(decoder, "_dtype", None)),
                "diffvae_optimization": str(getattr(decoder, "diffvae_optimization", None)),
                "tiling": None
                if tiling is None
                else {name: _axis_dict(getattr(tiling, name, None)) for name in ("frames", "height", "width")},
            },
            path,
        )
        logger.info("Dumped DiffVAE decode latent %s to %s", tuple(latent.shape), path)
    except Exception:
        logger.warning("Could not dump the DiffVAE decode latent", exc_info=True)


def _patched_video_decoder_call(self: VideoDecoder, *args: Any, **kwargs: Any) -> Any:
    _release_denoise_weights(self)
    args, kwargs = _with_post_evict_cuda_tiling(self, args, kwargs)
    args, kwargs = _with_mps_clamped_tiles(self, args, kwargs)
    _dump_latent_for_replay(self, args, kwargs)
    return _orig_video_decoder_call(self, *args, **kwargs)


VideoDecoder.__call__ = _patched_video_decoder_call  # type: ignore[method-assign]
