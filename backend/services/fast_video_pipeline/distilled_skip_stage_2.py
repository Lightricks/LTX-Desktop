"""Stage-1-only decode for ``DistilledPipeline``.

The vendor method has no ``skip_stage_2`` and always upsamples and refines.
``skip_stage_2=False`` forwards to that method. ``skip_stage_2=True`` denoises
the half-resolution canvas and decodes it, using the latent-entry stage.

Delete this file when a tag adds ``skip_stage_2`` to ``DistilledPipeline``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from ltx_core.model.video_vae import AUTO_TILING, AutoTiling, TilingConfig
from ltx_core.types import VideoPixelShape
import ltx_pipelines.distilled as distilled
from ltx_pipelines.distilled import ANCESTRAL_NOISE_SEED_OFFSET, _DistilledRunContext  # pyright: ignore[reportPrivateUsage]
from ltx_pipelines.chunks.conditionings import assert_image_frames_in_clip
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import (
    create_initial_av_latents,
    ensure_tiling_config,
    generated_keyframe_conditionings,
    image_conditionings,
)
from ltx_pipelines.utils.media_io import EXRColorSpace
from ltx_pipelines.utils.types import (
    DEFAULT_AUTO_DURATION,
    AutoDuration,
    ImageConditioningInput,
    ModalitySpec,
    PipelineOutput,
    VideoAudio,
)

from services.fast_video_pipeline.nag_config import NagConfig
from services.fast_video_pipeline.nag_distilled import NagDistilledMixin

logger = logging.getLogger(__name__)


class DistilledPipelineWithSkipStage2(NagDistilledMixin[_DistilledRunContext], distilled.DistilledPipeline):
    def __call__(
        self, *args: Any, skip_stage_2: bool = False, nag: NagConfig | None = None, **kwargs: Any
    ) -> PipelineOutput:
        with self._nag_request(nag):
            if skip_stage_2:
                return self._call_stage_1_only(*args, **kwargs)
            return super().__call__(*args, **kwargs)

    def _call_stage_1_only(  # noqa: PLR0913
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        num_frames: int | AutoDuration = DEFAULT_AUTO_DURATION,
        vae_dtype: torch.dtype | None = None,
        tiling_config: TilingConfig | AutoTiling | None = AUTO_TILING,
        enhance_prompt: bool = False,
        enhance_static_cache: bool = False,
        stage_1_sigmas: torch.Tensor = DISTILLED_SIGMAS,
        color_space: EXRColorSpace | None = None,
        generated_keyframes: int | Sequence[int] = 0,
    ) -> PipelineOutput:
        """Denoise stage 1 at canvas/2 and decode.

        Stage 2, keyframe decode, and chunking arguments are deliberately not accepted, so a
        caller that passes one fails with ``TypeError`` instead of being ignored.
        """
        # tiling_config=None leaves the shared prologue untiled: it sizes decode tiles for the
        # post-upsample canvas, and this path decodes the half-resolution canvas instead.
        ctx = self._prepare_run(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            frame_rate=frame_rate,
            images=images,
            color_space=color_space,
            num_frames=num_frames,
            vae_dtype=vae_dtype,
            tiling_config=None,
            enhance_prompt=enhance_prompt,
            enhance_static_cache=enhance_static_cache,
            generated_keyframes=generated_keyframes,
            decode_with_keyframes=False,
        )
        assert_image_frames_in_clip(ctx.images, ctx.num_frames)

        num_frames = ctx.num_frames
        stage_1_w, stage_1_h = ctx.stage_1_width, ctx.stage_1_height
        tiling_config = ensure_tiling_config(
            tiling_config,
            vae_checkpoint_path=self.video_decoder.checkpoint_path,
            video_shape=VideoPixelShape(
                batch=1, frames=num_frames, height=stage_1_h, width=stage_1_w, fps=frame_rate
            ),
            diffvae_optimization=self.video_decoder.diffvae_optimization,
            device=self.device,
            vae_dtype=ctx.vae_dtype,
        )

        stage_1_sigmas = stage_1_sigmas.to(dtype=torch.float32, device=self.device)
        stage_1_conditionings = self.image_conditioner(
            lambda enc: image_conditionings(
                images=ctx.images,
                height=stage_1_h,
                width=stage_1_w,
                video_encoder=enc,
                dtype=ctx.dtype,
                device=self.device,
            )
        )
        stage_1_conditionings.extend(generated_keyframe_conditionings(generated_keyframes, num_frames))

        video_latent, audio_latent = create_initial_av_latents(
            pixel_shape=VideoPixelShape(
                batch=1, frames=num_frames, height=stage_1_h, width=stage_1_w, fps=frame_rate
            ),
            device=self.device,
            dtype=ctx.dtype,
            # Transformer latent geometry, not the decoder's tiling scale (2x8x8 on the 2.5
            # DiffVAE), which only sizes decode tiles.
            video_scale_factors=self.stage.video_scale_factors,
        )
        with self._nag_denoise_scope():
            video_state, audio_state = self.stage(
                denoiser=SimpleDenoiser(),
                sigmas=stage_1_sigmas,
                noiser=ctx.noiser,
                modalities=VideoAudio(
                    video=ModalitySpec(
                        latent=video_latent,
                        conditioning_fps=frame_rate,
                        context=ctx.video_context,
                        conditionings=stage_1_conditionings,
                    ),
                    audio=ModalitySpec(
                        latent=audio_latent,
                        conditioning_fps=frame_rate,
                        context=ctx.audio_context,
                    ),
                ),
                **self._sampler_kwargs(seed, ANCESTRAL_NOISE_SEED_OFFSET),
            )

        logger.info("[fast] Skipping Stage 2 (skip_stage_2=True); decode %dx%d", stage_1_w, stage_1_h)
        assert video_state is not None
        assert audio_state is not None
        latent = video_state.latent
        decoded_audio = self.audio_decoder(audio_state.latent)
        return PipelineOutput(self._decode_lazily(latent, tiling_config, ctx), decoded_audio, num_frames, tiling_config)

    def _decode_lazily(self, latent: torch.Tensor, tiling_config: TilingConfig | None, ctx: _DistilledRunContext) -> Iterator[torch.Tensor]:
        """Build the decoder when the first chunk is requested.

        ``VideoDecoder.__call__`` builds eagerly but frees the model only once its iterator has
        started, so a stream closed before it is read would leave the decoder resident.
        """
        yield from self.video_decoder(latent, tiling_config, ctx.generator, dtype=ctx.vae_dtype)
