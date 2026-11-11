"""Distilled A2V (Audio-to-Video) pipeline.

Combines the distilled denoising approach (fixed sigmas, SimpleDenoiser,
block-based model lifecycle) with A2V-specific behaviour (audio encoding,
video-only denoising with frozen audio, returning original audio).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

import torch

from services.audio_channels import stereo_waveform
from services.fast_video_pipeline.nag_config import NagConfig
from services.fast_video_pipeline.nag_distilled import (
    prompts_with_negative,
    scoped_chunks,
    with_negative_context,
)
from services.patches.nag_cross_attention import NagState
from services.services_utils import TilingConfigType

if TYPE_CHECKING:
    from ltx_core.conditioning import ConditioningItem
    from ltx_core.loader import LoraPathStrengthAndSDOps
    from ltx_core.types import Audio as LtxAudio
    from ltx_pipelines.chunks import Chunk, ChunkConfig, PlaybackSegment
    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_pipelines.utils.types import ImageConditioningInput as LtxImageInput
    from ltx_pipelines.utils.types import LoadedImageConditioning, OffloadMode, PipelineOutput


@dataclass(frozen=True)
class _A2VInputs:
    """What the run needs after the prompt, the stills and the source audio are encoded."""

    num_frames: int
    images: list[LoadedImageConditioning]
    generator: torch.Generator
    noiser: Any
    video_context: torch.Tensor
    audio_context: torch.Tensor
    decoded_audio: LtxAudio
    encoded_audio_latent: torch.Tensor
    nag_state: NagState | None = None


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
        *,
        offload_mode: OffloadMode,
    ) -> None:
        from ltx_pipelines.utils.blocks import (
            AudioConditioner,
            DiffusionStage,
            ImageConditioner,
            PromptEncoder,
            VideoDecoder,
            VideoUpsampler,
        )
        from ltx_pipelines.distilled import should_use_ancestral_sampler
        from ltx_pipelines.utils.helpers import get_device

        if device is None:
            device = get_device()

        self.device = device
        self.dtype = torch.bfloat16

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
        self.use_ancestral_sampler = should_use_ancestral_sampler(model_paths.transformer())

    def _sampler_kwargs(self, seed: int, noise_seed_offset: int) -> dict[str, Any]:
        """Ancestral overrides on an LTX 2.5+ checkpoint, as ``DistilledPipeline`` does. Empty keeps Euler."""
        from ltx_pipelines.distilled import ancestral_sampler_kwargs

        if not self.use_ancestral_sampler:
            return {}
        return ancestral_sampler_kwargs(seed, self.dtype, noise_seed_offset)

    def _prepare(
        self,
        *,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[LtxImageInput],
        audio_path: str,
        audio_start_time: float,
        audio_max_duration: float | None,
        nag: NagConfig | None = None,
    ) -> _A2VInputs:
        """Validate, encode the prompt, the stills and the source audio. Shared by both run paths."""
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
        from ltx_core.types import AudioLatentShape
        from ltx_pipelines.chunks.conditionings import assert_image_frames_in_clip
        from ltx_pipelines.utils.helpers import assert_resolution, snap_frames_to_grid
        from ltx_pipelines.utils.media_io import decode_audio_from_file
        assert_resolution(height=height, width=width, is_two_stage=True)

        # Every later step sizes the clip from the snapped count, so a still is checked against it too.
        num_frames = snap_frames_to_grid(num_frames)
        loaded_images = self.image_conditioner.load_conditionings(images)
        assert_image_frames_in_clip(loaded_images, num_frames)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)

        # Text encode (positive, plus negative when NAG is active).
        contexts = self.prompt_encoder(prompts_with_negative(prompt, nag))
        ctx_p = contexts[0]
        if nag is None:
            video_context = ctx_p.video_encoding
            nag_state = None
        else:
            video_context, nag_state = with_negative_context(
                ctx_p.video_encoding, contexts[1].video_encoding, nag
            )
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

        return _A2VInputs(
            num_frames=num_frames,
            images=loaded_images,
            generator=generator,
            noiser=noiser,
            video_context=video_context,
            audio_context=audio_context,
            decoded_audio=decoded_audio,
            encoded_audio_latent=encoded_audio_latent,
            nag_state=nag_state,
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
        images: list[LtxImageInput],
        audio_path: str,
        audio_start_time: float = 0.0,
        audio_max_duration: float | None = None,
        tiling_config: TilingConfigType | None = None,
        nag: NagConfig | None = None,
    ) -> PipelineOutput:
        """One window over the whole clip: the same two stages as ``stream_chunks``, without carry."""
        from ltx_pipelines.chunks import ChunkConfig, pipeline_output_from_chunks
        from ltx_pipelines.utils.helpers import snap_frames_to_grid

        snapped = snap_frames_to_grid(num_frames)
        chunks = self.stream_chunks(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            audio_path=audio_path,
            chunk_config=ChunkConfig(chunk_pixel_frames=snapped, next_video_carry_frames=0),
            audio_start_time=audio_start_time,
            audio_max_duration=audio_max_duration,
            tiling_config=tiling_config,
            nag=nag,
        )
        return pipeline_output_from_chunks(chunks, num_frames=snapped, tiling_config=tiling_config)

    @torch.inference_mode()
    def stream_chunks(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[LtxImageInput],
        audio_path: str,
        chunk_config: ChunkConfig,
        audio_start_time: float = 0.0,
        audio_max_duration: float | None = None,
        tiling_config: TilingConfigType | None = None,
        nag: NagConfig | None = None,
    ) -> Iterator[PlaybackSegment]:
        """The same two stages in overlapping temporal windows, as ``a2vid_two_stage`` does it.

        Each window takes its slice of the frozen source audio. The decoded windows carry
        the matching slice of the original waveform, so the stitched track stays contiguous.
        """
        from ltx_core.types import Audio, VideoPixelShape
        from ltx_pipelines.chunks import (
            decode_chunks,
            denoise_chunks,
            generate_uniform_chunks,
            replace_chunks_audio,
            spatially_upsample_chunks,
        )
        from ltx_pipelines.chunks.conditionings import image_conditionings_for_chunk, images_for_chunk
        from ltx_pipelines.distilled import ANCESTRAL_NOISE_SEED_OFFSET, ANCESTRAL_STAGE_2_NOISE_SEED_OFFSET
        from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
        from ltx_pipelines.utils.types import VideoAudio

        run = self._prepare(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            audio_path=audio_path,
            audio_start_time=audio_start_time,
            audio_max_duration=audio_max_duration,
            nag=nag,
        )
        num_frames = run.num_frames

        def make_video_conditionings(chunk: Chunk) -> list[ConditioningItem]:
            if chunk.video is None or not images_for_chunk(chunk, run.images):
                return []
            return self.image_conditioner(
                lambda enc: image_conditionings_for_chunk(
                    chunk, images=run.images, video_encoder=enc
                )
            )

        chunks = generate_uniform_chunks(
            target=VideoPixelShape(
                batch=1, frames=num_frames, height=height // 2, width=width // 2, fps=frame_rate
            ),
            context=VideoAudio(video=run.video_context, audio=run.audio_context),
            device=self.device,
            dtype=torch.bfloat16,
            config=chunk_config,
            video_scale_factors=self.stage.video_scale_factors,
            audio_latent=run.encoded_audio_latent,
            make_video_conditionings=make_video_conditionings,
        )
        stage_1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)
        stage_2_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)
        base = denoise_chunks(
            chunks,
            self.stage,
            sigmas=stage_1_sigmas,
            noiser=run.noiser,
            fps=frame_rate,
            freeze_audio=True,
            **self._sampler_kwargs(seed, ANCESTRAL_NOISE_SEED_OFFSET),
        )
        refined = denoise_chunks(
            spatially_upsample_chunks(base, self.upsampler),
            self.stage,
            sigmas=stage_2_sigmas,
            noiser=run.noiser,
            fps=frame_rate,
            freeze_audio=True,
            **self._sampler_kwargs(seed, ANCESTRAL_STAGE_2_NOISE_SEED_OFFSET),
        )
        decoded = decode_chunks(
            refined,
            self.video_decoder,
            audio_decoder=None,
            fps=frame_rate,
            tiling_config=tiling_config,
            generator=run.generator,
            dtype=torch.bfloat16,
        )
        source = Audio(
            waveform=run.decoded_audio.waveform.squeeze(0), sampling_rate=run.decoded_audio.sampling_rate
        )
        out = replace_chunks_audio(decoded, source, frame_rate)
        return scoped_chunks(out, run.nag_state) if run.nag_state is not None else out
