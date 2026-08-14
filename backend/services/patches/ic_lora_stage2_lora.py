"""Monkey-patch (EXPERIMENTAL): keep the IC-LoRA active in stage 2.

Ports upstream PR #494's ``use_lora_in_stage_2`` (not yet synced to the public
Lightricks/LTX-2 repo) to the pinned rev. Stock ``ICLoraPipeline`` builds the stage-2 transformer with
``loras=()`` and conditions stage 2 on images only — so refinement pulls the output
back toward the base-model distribution and the IC-LoRA transformation (background
swap / depth / pose / canny / outpaint) blurs out. With the patch, when a pipeline
has ``use_lora_in_stage_2 = True``:

  1. stage 2 is rebuilt with the same LoRAs as stage 1 (cheap: ``DiffusionStage``
     builds the transformer per call and frees on exit, so this adds only the LoRA
     deltas during the stage-2 window — no second resident transformer), and
  2. stage 2's conditionings include the VAE-encoded reference video (via the
     existing ``_create_conditionings``) instead of image-only.

NOT a VRAM fix: stage-2 peak is set by the full output resolution (same as
skip_stage_2 at that res) plus a full-res reference re-encode. That reference encode
(``_create_conditionings`` -> ``VideoEncoder.forward``) is tiled here — scoped to
IC-LoRA pipeline runs only via the thread-local flag below — so it doesn't blow VRAM
(Linux OOM) or crawl on Windows; see
ltx-engine-planning/2026-06-29-ic-lora-resolution-factor-2-hang.md. The real low-VRAM
lever is weight streaming — see
ltx-engine-planning/2026-06-29-ic-lora-vram-auto-clamp-design.md.

Scoped conditioning-encode tiling: the reference video is VAE-encoded un-tiled by
default; at high resolution_factor (skip_stage_2) or full Stage-2 res
(use_lora_in_stage_2) that working set exceeds VRAM. We wrap ``VideoEncoder.forward``
to route oversized encodes through the encoder's own ``tiled_encode``, but ONLY while
a thread-local flag is set — and we set it only around this pipeline's reference-encode
steps. The main t2v/i2v pipelines never call ``ICLoraPipeline.__call__``, so their
encodes are untouched (no global ride-along).

EXPERIMENTAL: pinned to ltx-2 rev a2c3f24 (the ``__call__`` body is copied from the
installed ``ltx_pipelines/ic_lora.py`` with two edits, marked PATCH below). Re-verify
when bumping that rev. Drop once upstream ships PR #494 (and tiles the conditioning
encode itself).

Stage-2 reference downscale (EXPERIMENTAL, env IC_LORA_STAGE2_REF_DOWNSCALE, default 1=off):
use_lora_in_stage_2 re-conditions Stage 2 on the full-res reference video, which ~doubles the
attention sequence and drives the VRAM spike on long/large clips. Setting the env to e.g. 2
temporarily raises the LoRA's reference_downscale_factor for the Stage-2 conditioning build, so
the reference is encoded at lower res and contributes ~factor^2 fewer tokens (RoPE positions are
rescaled by downscale_factor to stay aligned with the full-res output — the same mechanism the
model already uses, and the resolution Stage 1 conditions at). Off the LoRA's *trained* ratio,
so the transformation may soften — A/B against the default before relying on it.

Usage:
    import services.patches.ic_lora_stage2_lora  # noqa: F401
    pipeline.use_lora_in_stage_2 = True  # opt in per pipeline before __call__
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager

import torch

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.model.video_vae import AUTO_TILING, AutoTiling, TileSizeConfig, TilingConfig
from ltx_core.model.video_vae.video_vae import VideoEncoder
from ltx_core.types import Audio, VideoPixelShape
from ltx_pipelines.ic_lora import ICLoraPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import assert_resolution, combined_image_conditionings, ensure_tiling_config, tiling_scale_factors_for_vae
from ltx_pipelines.utils.media_io import HDRColorSpace
from ltx_pipelines.utils.types import ModalitySpec
from services.patches import diffusion_stage_cache

logger = logging.getLogger(__name__)


# --- Scoped conditioning-encode tiling (see module docstring) ---
_orig_encoder_forward = VideoEncoder.forward
_encode_state = threading.local()


def _should_tile(shape: torch.Size, cfg: TileSizeConfig) -> bool:
    """True when ``tiled_encode`` would split this input into more than one tile.

    Below the tile size in every dimension a single tile == the un-tiled forward, so we
    skip tiling to keep small encodes (e.g. resolution_factor <= 1.0) exact. ``shape`` is
    (B, C, F, H, W).
    """
    f, h, w = shape[-3], shape[-2], shape[-1]
    if cfg.height.is_tiled() and h > cfg.height.tile_size:
        return True
    if cfg.width.is_tiled() and w > cfg.width.tile_size:
        return True
    if cfg.frames.is_tiled() and f > cfg.frames.tile_size:
        return True
    return False


def _scoped_tiling_forward(self: VideoEncoder, sample: torch.Tensor) -> torch.Tensor:
    # Off unless an IC-LoRA reference encode opted in; per-tile re-entry (tiled_encode ->
    # self.forward) and small inputs pass straight through to the original forward.
    if not getattr(_encode_state, "tile", False) or getattr(_encode_state, "in_tiled", False):
        return _orig_encoder_forward(self, sample)
    cfg = TileSizeConfig.default()
    if not _should_tile(sample.shape, cfg):
        return _orig_encoder_forward(self, sample)
    logger.info(
        "[ic-lora] tiling conditioning VAE encode %s (tile %dpx) to avoid VRAM blow-up",
        tuple(sample.shape), cfg.height.tile_size,
    )
    _encode_state.in_tiled = True
    try:
        return self.tiled_encode(sample, cfg)
    finally:
        _encode_state.in_tiled = False


VideoEncoder.forward = _scoped_tiling_forward  # type: ignore[assignment]


@contextmanager
def _tile_ic_lora_encodes() -> Iterator[None]:
    """Enable tiling of oversized VAE encodes for the wrapped block (this thread only)."""
    prev = getattr(_encode_state, "tile", False)
    _encode_state.tile = True
    try:
        yield
    finally:
        _encode_state.tile = prev


# --- Stage-2 reference downscale (see module docstring) ---
_STAGE2_REF_DOWNSCALE = max(1, int(os.environ.get("IC_LORA_STAGE2_REF_DOWNSCALE", "1")))


@contextmanager
def _stage2_reference_downscale(pipeline: object, extra: int, height: int, width: int) -> Iterator[None]:
    """Temporarily raise the pipeline's reference_downscale_factor by ``extra`` so the Stage-2
    reference video is encoded at lower res (~extra^2 fewer conditioning tokens). Restores it
    after. No-op when extra <= 1 or the output dims aren't divisible by the effective factor
    (which _create_conditionings would otherwise reject)."""
    base = int(getattr(pipeline, "reference_downscale_factor", 1))
    eff = base * extra
    if extra <= 1 or height % eff != 0 or width % eff != 0:
        if extra > 1:
            logger.warning(
                "[ic-lora] stage-2 ref downscale x%d skipped: output %dx%d not divisible by %d",
                extra, width, height, eff,
            )
        yield
        return
    setattr(pipeline, "reference_downscale_factor", eff)
    logger.info("[ic-lora] stage-2 reference downscale x%d (downscale_factor %d -> %d) to cut VRAM", extra, base, eff)
    try:
        yield
    finally:
        setattr(pipeline, "reference_downscale_factor", base)


def _patched_call(  # noqa: PLR0913
    self: ICLoraPipeline,
    prompt: str,
    seed: int,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    images: list[ImageConditioningInput],
    video_conditioning: list[tuple[str, float]],
    enhance_prompt: bool = False,
    enhance_static_cache: bool = False,
    vae_dtype: torch.dtype | None = None,
    tiling_config: TilingConfig | AutoTiling | None = AUTO_TILING,
    conditioning_attention_strength: float = 1.0,
    skip_stage_2: bool = False,
    conditioning_attention_mask: torch.Tensor | None = None,
    stage_1_sigmas: torch.Tensor = DISTILLED_SIGMAS,
    stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
    color_space: HDRColorSpace | None = None,
) -> tuple[Iterator[torch.Tensor], Audio, TilingConfig | None]:
    """Copy of ICLoraPipeline.__call__ (pinned upstream rev) + IC-LoRA-in-stage-2 (PATCH)."""
    use_lora_in_stage_2 = getattr(self, "use_lora_in_stage_2", False)

    # PATCH (was): use_lora_in_stage_2 used to force stage 2 onto pinned-RAM streaming per call
    # via streaming_prefetch_count, since the full-res reference conditioning roughly doubles the
    # attention sequence and can overflow VRAM held resident. ltx_pipelines now fixes weight
    # streaming (OffloadMode) at DiffusionStage construction time — LTXIcLoraPipeline.__init__
    # bakes it into self.stage_2 via offload_mode_for_prefetch_count, so it can no longer be
    # bumped per call here. If VRAM pressure resurfaces on use_lora_in_stage_2, the fix is a
    # construction-time offload_mode override for this pipeline, not a per-call one.

    images = self.image_conditioner.resolve_crf(images)
    assert_resolution(height=height, width=width, is_two_stage=True)
    if not (0.0 <= conditioning_attention_strength <= 1.0):
        raise ValueError(
            f"conditioning_attention_strength must be in [0.0, 1.0], got {conditioning_attention_strength}"
        )

    generator = torch.Generator(device=self.device).manual_seed(seed)
    noiser = GaussianNoiser(generator=generator)
    if vae_dtype is None:
        vae_dtype = self.dtype

    (ctx_p,) = self.prompt_encoder(
        [prompt],
        enhance_first_prompt=enhance_prompt,
        enhance_static_cache=enhance_static_cache,
        enhance_prompt_image=images[0][0] if len(images) > 0 else None,
        enhance_prompt_seed=seed,
    )
    video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding

    scale_factors = tiling_scale_factors_for_vae(self.video_decoder.checkpoint_path)
    tiling_config = ensure_tiling_config(
        tiling_config,
        scale_factors=scale_factors,
        vae_checkpoint_path=self.video_decoder.checkpoint_path,
        video_shape=VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=frame_rate),
        diffvae_optimization=self.video_decoder.diffvae_optimization,
        device=self.device,
    )

    # Stage 1: Initial low resolution video generation.
    stage_1_output_shape = VideoPixelShape(
        batch=1,
        frames=num_frames,
        width=width // 2,
        height=height // 2,
        fps=frame_rate,
    )

    # PATCH: tile the reference-video VAE encode (oversized at high resolution_factor) —
    # scoped to this block so the main pipeline's encodes are unaffected.
    with _tile_ic_lora_encodes():
        stage_1_conditionings = self.image_conditioner(
            lambda enc: self._create_conditionings(
                images=images,
                video_conditioning=video_conditioning,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                video_encoder=enc,
                num_frames=num_frames,
                conditioning_attention_strength=conditioning_attention_strength,
                conditioning_attention_mask=conditioning_attention_mask,
                color_space=color_space,
            )
        )

    stage_1_sigmas = stage_1_sigmas.to(dtype=torch.float32, device=self.device)

    video_state, audio_state = self.stage_1(
        denoiser=SimpleDenoiser(video_context, audio_context),
        sigmas=stage_1_sigmas,
        noiser=noiser,
        width=stage_1_output_shape.width,
        height=stage_1_output_shape.height,
        frames=num_frames,
        fps=frame_rate,
        video=ModalitySpec(
            context=video_context,
            conditionings=stage_1_conditionings,
        ),
        audio=ModalitySpec(
            context=audio_context,
        ),
    )

    # EXPERIMENTAL: stage_1 and stage_2 never share a diffusion_stage_cache key here --
    # stage_2 either goes streaming (use_lora_in_stage_2, see below) or uses different/
    # empty LoRAs, so a stage_1 cache hit on stage_2 never happens for this pipeline.
    # Evict eagerly (costs nothing) so the still-resident stage_1 transformer doesn't
    # compete for VRAM with the stage_2 conditioning encode below, which runs before
    # stage_2's own DiffusionStage call would otherwise trigger an eviction. Live repro:
    # reserved VRAM climbed to ~41GB on a 31.82GB card and hung during that encode.
    diffusion_stage_cache.evict()

    if skip_stage_2:
        # Skip Stage 2: Decode directly from Stage 1 output at half resolution
        logging.info("[IC-LoRA] Skipping Stage 2 (--skip-stage-2 enabled)")
        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator, dtype=vae_dtype)
        decoded_audio = self.audio_decoder(audio_state.latent)
        return decoded_video, decoded_audio, tiling_config

    # Stage 2: Upsample and refine the video at higher resolution.
    upscaled_video_latent = self.upsampler(video_state.latent[:1])

    stage_2_sigmas = stage_2_sigmas.to(dtype=torch.float32, device=self.device)
    stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)

    # PATCH: keep stage 2's LoRAs in sync with the flag on EVERY call. DiffusionStage
    # builds-on-call, so the builder's LoRAs decide what the next stage_2(...) constructs.
    # Set both ways (LoRAs when on, () when off) so a cached pipeline reused with the flag
    # flipped off doesn't carry stale LoRAs into a canonical refine.
    self.stage_2._transformer_builder = self.stage_2._transformer_builder.with_loras(
        self.stage_1._transformer_builder.loras if use_lora_in_stage_2 else ()
    )

    if use_lora_in_stage_2:
        # Condition stage 2 on the reference video (not image-only) so the IC-LoRA
        # transformation survives refinement instead of being repainted from the prompt.
        logger.info("[ic-lora] stage 2 running WITH IC-LoRA (use_lora_in_stage_2)")
        # PATCH: the full Stage-2-res reference encode/conditioning is the big VRAM spike. Tile
        # the encode (scoped), and optionally downscale the reference (IC_LORA_STAGE2_REF_DOWNSCALE)
        # so it carries fewer tokens — like Stage 1's half-res reference. See module docstring.
        with _tile_ic_lora_encodes(), _stage2_reference_downscale(
            self, _STAGE2_REF_DOWNSCALE, stage_2_output_shape.height, stage_2_output_shape.width
        ):
            stage_2_conditionings = self.image_conditioner(
                lambda enc: self._create_conditionings(
                    images=images,
                    video_conditioning=video_conditioning,
                    height=stage_2_output_shape.height,
                    width=stage_2_output_shape.width,
                    video_encoder=enc,
                    num_frames=num_frames,
                    conditioning_attention_strength=conditioning_attention_strength,
                    conditioning_attention_mask=conditioning_attention_mask,
                    color_space=color_space,
                )
            )
    else:
        stage_2_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=stage_2_output_shape.height,
                width=stage_2_output_shape.width,
                video_encoder=enc,
                dtype=self.dtype,
                device=self.device,
                color_space=color_space,
            )
        )

    video_state, audio_state = self.stage_2(
        denoiser=SimpleDenoiser(video_context, audio_context),
        sigmas=stage_2_sigmas,
        noiser=noiser,
        width=width,
        height=height,
        frames=num_frames,
        fps=frame_rate,
        video=ModalitySpec(
            context=video_context,
            conditionings=stage_2_conditionings,
            noise_scale=stage_2_sigmas[0].item(),
            initial_latent=upscaled_video_latent,
        ),
        audio=ModalitySpec(
            context=audio_context,
            noise_scale=stage_2_sigmas[0].item(),
            initial_latent=audio_state.latent,
        ),
    )

    decoded_video = self.video_decoder(video_state.latent, tiling_config, generator, dtype=vae_dtype)
    decoded_audio = self.audio_decoder(audio_state.latent)
    return decoded_video, decoded_audio, tiling_config


ICLoraPipeline.__call__ = _patched_call  # type: ignore[method-assign]


if __name__ == "__main__":
    cfg = TileSizeConfig.default()
    sp = cfg.height.tile_size
    fr = cfg.frames.tile_size
    assert not _should_tile(torch.Size([1, 3, fr, sp, sp]), cfg), "exactly one tile -> no tiling"
    assert _should_tile(torch.Size([1, 3, fr, sp, sp + 1]), cfg), "wider than a tile -> tile"
    assert _should_tile(torch.Size([1, 3, fr + 1, sp, sp]), cfg), "more frames than a tile -> tile"
    print("ic_lora_stage2_lora: tiling gate self-check OK")
