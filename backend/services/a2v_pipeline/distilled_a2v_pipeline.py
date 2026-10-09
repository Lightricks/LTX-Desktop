"""Distilled A2V (Audio-to-Video) pipeline.

Combines the distilled denoising approach (fixed sigmas, SimpleDenoiser,
block-based model lifecycle) with A2V-specific behaviour (audio encoding,
video-only denoising with frozen audio, returning original audio).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol, cast

import torch

from services.audio_channels import stereo_waveform
from services.ltx_pipeline_common import offload_mode_for_prefetch_count
from services.services_utils import AudioOrNone, TilingConfigType

if TYPE_CHECKING:
    from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_pipelines.utils.types import ImageConditioningInput as LtxImageInput


class _ImageCrfResolver(Protocol):
    def resolve_crf(self, images: Sequence[LtxImageInput]) -> list[LtxImageInput]: ...


def resolve_image_conditionings(
    images: Sequence[tuple[str, int, float]],
    image_conditioner: _ImageCrfResolver,
) -> list[LtxImageInput]:
    """Build LTX image inputs and fill checkpoint CRF via ImageConditioner.resolve_crf."""
    from ltx_pipelines.utils.types import ImageConditioningInput as LtxImageInput

    ltx_images = [LtxImageInput(path, frame_idx, strength) for path, frame_idx, strength in images]
    return image_conditioner.resolve_crf(ltx_images)


def encode_stage_image_conditionings(
    images: list[LtxImageInput],
    *,
    height: int,
    width: int,
    video_encoder: Any,
    dtype: torch.dtype,
    device: torch.device,
) -> list[Any]:
    """Pixel-index image conds for distilled A2V stages.

    Same helper as DistilledPipeline and a2vid_two_stage: frame 0 replaces latent 0;
    later frames are keyframes at that pixel index.

    Do not use image_conditionings_by_replacing_latent — it treats frame_idx as a
    latent index. Last pixel frame (num_frames-1, e.g. 240 at 10s/24fps) is past the
    VAE-compressed sequence (~31 latents) and apply_to empty-slices.
    """
    from ltx_pipelines.utils.helpers import combined_image_conditionings

    return combined_image_conditionings(
        images=images,
        height=height,
        width=width,
        video_encoder=video_encoder,
        dtype=dtype,
        device=device,
    )


class DistilledA2VPipeline:
    """Two-stage distilled audio-to-video pipeline.

    Stage 1 generates video at half resolution with frozen audio conditioning,
    then Stage 2 upsamples by 2x and refines with additional distilled steps.
    Uses block-based model lifecycle (no LoRA swap between stages).
    """

    def __init__(
        self,
        model_paths: ModelPaths,
        spatial_upsampler_path: str,
        loras: Sequence[LoraPathStrengthAndSDOps] | None = None,
        device: torch.device | None = None,
        quantization: Any | None = None,
        streaming_prefetch_count: int | None = None,
    ) -> None:
        from ltx_pipelines.utils.blocks import (
            AudioConditioner,
            DiffusionStage,
            ImageConditioner,
            PromptEncoder,
            VideoDecoder,
            VideoUpsampler,
        )
        from ltx_pipelines.utils.helpers import get_device

        if device is None:
            device = get_device()

        self.device = device
        self.dtype = torch.bfloat16
        offload_mode = offload_mode_for_prefetch_count(streaming_prefetch_count, device)

        self.prompt_encoder = PromptEncoder(
            model_paths, self.dtype, device, offload_mode=offload_mode,
        )
        self.image_conditioner = ImageConditioner(
            model_paths.video_vae(), self.dtype, device,
        )
        self.audio_conditioner = AudioConditioner(
            model_paths.audio_vae(), self.dtype, device,
        )
        self.stage = DiffusionStage.from_checkpoint(  # type: ignore[reportUnknownMemberType]
            model_paths.transformer(),
            self.dtype,
            device,
            loras=tuple(loras) if loras else (),
            quantization=quantization,
            offload_mode=offload_mode,
        )
        self.upsampler = VideoUpsampler(
            model_paths.video_vae(), spatial_upsampler_path, self.dtype, device,
        )
        self.video_decoder = VideoDecoder(
            model_paths.video_vae(), self.dtype, device,
        )

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[tuple[str, int, float]],
        audio_path: str,
        audio_start_time: float = 0.0,
        audio_max_duration: float | None = None,
        tiling_config: TilingConfigType | None = None,
    ) -> tuple[Iterator[torch.Tensor], AudioOrNone]:
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
        from ltx_core.types import Audio, AudioLatentShape
        from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
        from ltx_pipelines.utils.denoisers import SimpleDenoiser
        from ltx_pipelines.utils.helpers import assert_resolution, create_initial_video_latent
        from ltx_pipelines.utils.media_io import decode_audio_from_file
        from ltx_pipelines.utils.types import ModalitySpec, VideoAudio

        assert_resolution(height=height, width=width, is_two_stage=True)

        ltx_images = resolve_image_conditionings(images, self.image_conditioner)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = torch.bfloat16

        # Text encode (positive only).
        (ctx_p,) = self.prompt_encoder([prompt])
        video_context = ctx_p.video_encoding
        audio_context = ctx_p.audio_encoding
        assert audio_context is not None, "A2V pipeline requires audio context from text encoder"

        # Audio encode.
        decoded_audio = decode_audio_from_file(audio_path, self.device, audio_start_time, audio_max_duration)
        assert decoded_audio is not None, "Audio file contains no audio stream"
        decoded_audio = replace(
            decoded_audio,
            waveform=stereo_waveform(decoded_audio.waveform),
        )
        encoded_audio_latent = self.audio_conditioner(
            lambda enc: vae_encode_audio(decoded_audio, cast(Any, enc), None)
        )
        audio_shape = AudioLatentShape.from_duration(batch=1, duration=num_frames / frame_rate, channels=8, mel_bins=16)
        target_frames = audio_shape.frames
        if encoded_audio_latent.shape[2] < target_frames:
            pad_size = target_frames - encoded_audio_latent.shape[2]
            encoded_audio_latent = torch.nn.functional.pad(encoded_audio_latent, (0, 0, 0, pad_size))
        else:
            encoded_audio_latent = encoded_audio_latent[:, :, :target_frames]

        # The same frozen, unnoised source audio drives both stages.
        audio_spec = ModalitySpec(
            latent=encoded_audio_latent,
            conditioning_fps=frame_rate,
            context=audio_context,
            frozen=True,
            noise_scale=0.0,
        )

        # Stage 1: Half-resolution video generation with frozen audio.
        stage_1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)
        stage_1_w, stage_1_h = width // 2, height // 2
        stage_1_conditionings = self.image_conditioner(
            lambda enc: encode_stage_image_conditionings(
                ltx_images,
                height=stage_1_h,
                width=stage_1_w,
                video_encoder=enc,
                dtype=dtype,
                device=self.device,
            )
        )
        stage_1_video_latent = create_initial_video_latent(
            width=stage_1_w,
            height=stage_1_h,
            frames=num_frames,
            fps=frame_rate,
            device=self.device,
            dtype=dtype,
            # Transformer latent geometry, not the decoder's tiling scale (2x8x8 on the 2.5 DiffVAE).
            scale_factors=self.stage.video_scale_factors,
        )

        video_state, _ = self.stage(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_1_sigmas,
            noiser=noiser,
            modalities=VideoAudio(
                video=ModalitySpec(
                    latent=stage_1_video_latent,
                    conditioning_fps=frame_rate,
                    context=video_context,
                    conditionings=stage_1_conditionings,
                ),
                audio=audio_spec,
            ),
        )

        # Upsample video 2x.
        assert video_state is not None
        upscaled_video_latent = self.upsampler(video_state.latent[:1])

        # Stage 2: Full-resolution refinement with frozen audio.
        stage_2_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)
        stage_2_conditionings = self.image_conditioner(
            lambda enc: encode_stage_image_conditionings(
                ltx_images,
                height=height,
                width=width,
                video_encoder=enc,
                dtype=dtype,
                device=self.device,
            )
        )

        video_state, _ = self.stage(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_2_sigmas,
            noiser=noiser,
            modalities=VideoAudio(
                video=ModalitySpec(
                    latent=upscaled_video_latent,
                    conditioning_fps=frame_rate,
                    context=video_context,
                    conditionings=stage_2_conditionings,
                    noise_scale=stage_2_sigmas[0].item(),
                ),
                audio=audio_spec,
            ),
        )

        # Decode video; return original audio (not VAE-decoded) for fidelity.
        assert video_state is not None
        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)

        # Trim waveform to target video duration so the muxed output doesn't
        # extend beyond the generated video frames.
        max_samples = round(num_frames / frame_rate * decoded_audio.sampling_rate)
        trimmed_waveform = decoded_audio.waveform.squeeze(0)[..., :max_samples]
        original_audio = Audio(waveform=trimmed_waveform, sampling_rate=decoded_audio.sampling_rate)

        return decoded_video, original_audio
