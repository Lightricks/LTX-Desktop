"""Stage-1-only decode for ``DistilledPipeline``.

The vendor method has no ``skip_stage_2`` and always upsamples and refines.
``skip_stage_2=False`` forwards to that method. ``skip_stage_2=True`` denoises
the half-resolution canvas and decodes it, using the v1.4 latent-entry stage.

Delete this file when a tag adds ``skip_stage_2`` to ``DistilledPipeline``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import torch
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.model.video_vae import AUTO_TILING, AutoTiling, TilingConfig
from ltx_core.types import VideoPixelShape
import ltx_pipelines.distilled as distilled
import ltx_pipelines.utils.helpers as ltx_helpers
from ltx_pipelines.distilled import ANCESTRAL_NOISE_SEED_OFFSET
from ltx_pipelines.utils.blocks import require_num_frames_source, resolve_num_frames
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import (
    assert_stage_supports_generated_keyframes,
    create_initial_av_latents,
    ensure_tiling_config,
    generated_keyframe_conditionings,
    has_generated_keyframes,
    snap_frames_to_grid,
    tiling_scale_factors_for_vae,
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

logger = logging.getLogger(__name__)


class DistilledPipelineWithSkipStage2(distilled.DistilledPipeline):
    def __call__(self, *args: Any, skip_stage_2: bool = False, **kwargs: Any) -> PipelineOutput:
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
        require_num_frames_source(num_frames, self.duration_predictor)
        images = self.image_conditioner.resolve_crf(images)
        distilled.assert_resolution(height=height, width=width, is_two_stage=True)
        if has_generated_keyframes(generated_keyframes):
            assert_stage_supports_generated_keyframes(self.stage)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = torch.bfloat16
        if vae_dtype is None:
            vae_dtype = dtype

        (ctx_p,) = self.prompt_encoder(
            [prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_static_cache=enhance_static_cache,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
        )
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding

        num_frames = snap_frames_to_grid(
            resolve_num_frames(
                num_frames,
                self.duration_predictor,
                video_encoding=video_context,
                audio_encoding=audio_context,
                frame_rate=frame_rate,
            )
        )

        # Decode shape is stage-1 pixels (canvas/2). Distilled sizes tiling for the
        # post-upsample canvas because it always runs stage 2.
        stage_1_w, stage_1_h = width // 2, height // 2
        scale_factors = tiling_scale_factors_for_vae(self.video_decoder.checkpoint_path)
        tiling_config = ensure_tiling_config(
            tiling_config,
            scale_factors=scale_factors,
            vae_checkpoint_path=self.video_decoder.checkpoint_path,
            video_shape=VideoPixelShape(
                batch=1, frames=num_frames, height=stage_1_h, width=stage_1_w, fps=frame_rate
            ),
            diffvae_optimization=self.video_decoder.diffvae_optimization,
            device=self.device,
        )

        stage_1_sigmas = stage_1_sigmas.to(dtype=torch.float32, device=self.device)
        # Look up on helpers so a keyframe patch can swap the helper.
        stage_1_conditionings = self.image_conditioner(
            lambda enc: ltx_helpers.combined_image_conditionings(
                images=images,
                height=stage_1_h,
                width=stage_1_w,
                video_encoder=enc,
                dtype=dtype,
                device=self.device,
                color_space=color_space,
            )
        )
        stage_1_conditionings.extend(generated_keyframe_conditionings(generated_keyframes, num_frames))

        video_latent, audio_latent = create_initial_av_latents(
            width=stage_1_w,
            height=stage_1_h,
            frames=num_frames,
            fps=frame_rate,
            device=self.device,
            dtype=dtype,
            # Transformer latent geometry. `scale_factors` above is the decoder's tiling scale
            # (2x8x8 on the 2.5 DiffVAE) and is only right for sizing decode tiles.
            video_scale_factors=self.stage.video_scale_factors,
        )
        video_state, audio_state = self.stage(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_1_sigmas,
            noiser=noiser,
            modalities=VideoAudio(
                video=ModalitySpec(
                    latent=video_latent,
                    conditioning_fps=frame_rate,
                    context=video_context,
                    conditionings=stage_1_conditionings,
                ),
                audio=ModalitySpec(
                    latent=audio_latent,
                    conditioning_fps=frame_rate,
                    context=audio_context,
                ),
            ),
            **self._sampler_kwargs(seed, ANCESTRAL_NOISE_SEED_OFFSET),
        )

        logger.info("[fast] Skipping Stage 2 (skip_stage_2=True); decode %dx%d", stage_1_w, stage_1_h)
        assert video_state is not None
        assert audio_state is not None
        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator, dtype=vae_dtype)
        decoded_audio = self.audio_decoder(audio_state.latent)
        return PipelineOutput(decoded_video, decoded_audio, num_frames, tiling_config)
