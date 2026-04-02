"""LTX retake pipeline wrapper.

Forked orchestration of the retake pipeline flow from ``ltx_pipelines.retake``
with the following adjustments:

* ``@torch.no_grad()`` instead of ``@torch.inference_mode()`` — the
  transformer checkpoint uses custom autograd functions incompatible with
  inference-mode tensors.
* Tiled video encoding via ``video_latent_from_file(..., tiling_config)``
  — the original encodes all frames in a single pass which OOMs on most GPUs.
* Tiled video decoding via ``VideoDecoder(..., tiling_config)`` — the
  original omits the tiling argument.
"""

from __future__ import annotations

from collections.abc import Iterator
import torch

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio
from ltx_pipelines.utils.media_io import encode_video, get_videostream_metadata

from services.retake_pipeline.retake_pipeline import RetakePipeline



class LTXRetakePipeline:
    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        device: torch.device,
        *,
        loras: list[LoraPathStrengthAndSDOps] | None = None,
        quantization: QuantizationPolicy | None = None,
    ) -> RetakePipeline:
        return LTXRetakePipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            device=device,
            loras=loras or [],
            quantization=quantization,
        )

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        device: torch.device,
        *,
        loras: list[LoraPathStrengthAndSDOps],
        quantization: QuantizationPolicy | None,
    ) -> None:
        from ltx_pipelines.utils.blocks import (
            AudioConditioner,
            AudioDecoder,
            DiffusionStage,
            ImageConditioner,
            PromptEncoder,
            VideoDecoder,
        )

        self.device = device
        self.dtype = torch.bfloat16

        self.prompt_encoder = PromptEncoder(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root or "",
            dtype=self.dtype,
            device=device,
        )
        self.image_conditioner = ImageConditioner(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=device,
        )
        self.audio_conditioner = AudioConditioner(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=device,
        )
        self.stage = DiffusionStage(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=device,
            loras=tuple(loras),
            quantization=quantization,
        )
        self.video_decoder = VideoDecoder(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=device,
        )
        self.audio_decoder = AudioDecoder(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=device,
        )

    @torch.no_grad()
    def _run(  # noqa: PLR0913, PLR0915
        self,
        video_path: str,
        prompt: str,
        start_time: float,
        end_time: float,
        seed: int,
        *,
        negative_prompt: str = "",
        num_inference_steps: int = 40,
        video_guider_params: MultiModalGuiderParams | None = None,
        audio_guider_params: MultiModalGuiderParams | None = None,
        regenerate_video: bool = True,
        regenerate_audio: bool = True,
        enhance_prompt: bool = False,
        distilled: bool = False,
        streaming_prefetch_count: int | None = None,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        from ltx_core.components.guiders import MultiModalGuider
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.components.schedulers import LTX2Scheduler
        from ltx_core.conditioning.types.noise_mask_cond import TemporalRegionMask
        from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES as _distilled_sigmas
        from ltx_pipelines.utils.denoisers import GuidedDenoiser, SimpleDenoiser
        from ltx_pipelines.utils.helpers import audio_latent_from_file, video_latent_from_file
        from ltx_pipelines.utils.types import ModalitySpec

        if start_time >= end_time:
            raise ValueError(f"start_time ({start_time}) must be less than end_time ({end_time})")

        effective_seed = int(torch.randint(0, 2**31, (1,)).item()) if seed < 0 else seed
        generator = torch.Generator(device=self.device).manual_seed(effective_seed)
        noiser = GaussianNoiser(generator=generator)
        from ltx_core.model.video_vae import SpatialTilingConfig, TemporalTilingConfig

        dtype = self.dtype
        tiling = TilingConfig.default()
        # Smaller tiles for source video encoding to reduce peak VRAM allocation
        # during the VAE encoder forward pass.
        encoding_tiling = TilingConfig(
            spatial_config=SpatialTilingConfig(tile_size_in_pixels=256, tile_overlap_in_pixels=64),
            temporal_config=TemporalTilingConfig(tile_size_in_frames=24, tile_overlap_in_frames=16),
        )

        # --- Encode source video (tiled) ---
        output_shape = get_videostream_metadata(video_path)

        initial_video_latent = self.image_conditioner(
            lambda enc: video_latent_from_file(
                video_encoder=enc,
                file_path=video_path,
                output_shape=output_shape,
                dtype=dtype,
                device=self.device,
                tiling_config=encoding_tiling,
            )
        )


        # --- Encode source audio ---

        initial_audio_latent = self.audio_conditioner(
            lambda enc: audio_latent_from_file(
                audio_encoder=enc,
                file_path=video_path,
                output_shape=output_shape,
                dtype=dtype,
                device=self.device,
            )
        )


        # --- Text encoding ---

        prompts_to_encode = [prompt] if distilled else [prompt, negative_prompt]
        contexts = self.prompt_encoder(
            prompts_to_encode,
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_seed=effective_seed,
            streaming_prefetch_count=streaming_prefetch_count,
        )


        v_context_p, a_context_p = contexts[0].video_encoding, contexts[0].audio_encoding

        # --- Build modality specs ---
        video_modality_spec = ModalitySpec(
            context=v_context_p,
            conditionings=[TemporalRegionMask(start_time=start_time, end_time=end_time, fps=output_shape.fps)]
            if regenerate_video
            else [],
            initial_latent=initial_video_latent,
            frozen=not regenerate_video,
        )
        audio_modality_spec: ModalitySpec | None = None
        if a_context_p is not None:
            audio_modality_spec = ModalitySpec(
                context=a_context_p,
                conditionings=[TemporalRegionMask(start_time=start_time, end_time=end_time, fps=output_shape.fps)]
                if (initial_audio_latent is not None and regenerate_audio)
                else [],
                initial_latent=initial_audio_latent,
                frozen=initial_audio_latent is not None and not regenerate_audio,
            )

        # --- Build denoiser ---
        if distilled:
            sigmas = torch.tensor(_distilled_sigmas).to(dtype=torch.float32, device=self.device)
            denoiser = SimpleDenoiser(v_context=v_context_p, a_context=a_context_p)
        else:
            sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=self.device)  # type: ignore[no-untyped-call]
            assert video_guider_params is not None, "video_guider_params required for non-distilled"
            assert audio_guider_params is not None, "audio_guider_params required for non-distilled"
            v_context_n, a_context_n = contexts[1].video_encoding, contexts[1].audio_encoding
            denoiser = GuidedDenoiser(
                v_context=v_context_p,
                a_context=a_context_p,
                video_guider=MultiModalGuider(params=video_guider_params, negative_context=v_context_n),
                audio_guider=MultiModalGuider(params=audio_guider_params, negative_context=a_context_n),
            )

        # --- Run diffusion stage ---

        video_state, audio_state = self.stage(
            denoiser=denoiser,
            sigmas=sigmas,
            noiser=noiser,
            width=output_shape.width,
            height=output_shape.height,
            frames=output_shape.frames,
            fps=output_shape.fps,
            video=video_modality_spec,
            audio=audio_modality_spec,
            streaming_prefetch_count=streaming_prefetch_count,
        )


        # --- Decode audio first (eager, small) ---
        assert audio_state is not None
        decoded_audio = self.audio_decoder(audio_state.latent)

        # --- Decode video (lazy generator, tiled) ---
        assert video_state is not None
        decoded_video = self.video_decoder(video_state.latent, tiling, generator)

        return decoded_video, decoded_audio

    @torch.no_grad()
    def generate(
        self,
        *,
        video_path: str,
        prompt: str,
        start_time: float,
        end_time: float,
        seed: int,
        output_path: str,
        negative_prompt: str = "",
        num_inference_steps: int = 40,
        video_guider_params: MultiModalGuiderParams | None = None,
        audio_guider_params: MultiModalGuiderParams | None = None,
        regenerate_video: bool = True,
        regenerate_audio: bool = True,
        enhance_prompt: bool = False,
        distilled: bool = True,
    ) -> None:
        meta = get_videostream_metadata(video_path)
        fps, num_frames = meta.fps, meta.frames
        video_iter, audio = self._run(
            video_path=video_path,
            prompt=prompt,
            start_time=start_time,
            end_time=end_time,
            seed=seed,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
            regenerate_video=regenerate_video,
            regenerate_audio=regenerate_audio,
            enhance_prompt=enhance_prompt,
            distilled=distilled,
            streaming_prefetch_count=2,
        )
        audio_out: Audio | None = audio
        tiling_config = TilingConfig.default()
        video_chunks = get_video_chunks_number(num_frames, tiling_config)
        encode_video(
            video=video_iter,
            fps=int(fps),
            audio=audio_out,
            output_path=output_path,
            video_chunks_number=video_chunks,
        )
