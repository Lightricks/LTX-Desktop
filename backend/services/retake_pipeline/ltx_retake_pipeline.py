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
from ltx_core.model.video_vae import DimensionSizeConfig, TileSizeConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio
from ltx_pipelines.utils.media_io import encode_video, get_videostream_metadata

from api_types import ExtendMode
from services.ltx_pipeline_common import build_model_paths, offload_mode_for_prefetch_count, resolve_tiling_config
from services.services_utils import TilingConfigType
from services.retake_pipeline.retake_pipeline import RetakePipeline


# Seam feather for extend: widen the regenerated region this far INTO the kept source so the
# frozen->generated boundary blends instead of cutting hard. Mirrors the cloud gateway's
# MASK_DELTA_SECONDS (ltxv-api retake-edit-window.computePadding). 0 disables the feather.
_EXTEND_MASK_DELTA_SECONDS = 0.5



class LTXRetakePipeline:
    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        device: torch.device,
        streaming_prefetch_count: int | None,
        *,
        loras: list[LoraPathStrengthAndSDOps] | None = None,
        quantization: QuantizationPolicy | None = None,
        video_vae_path: str | None = None,
        audio_vae_path: str | None = None,
        duration_head_path: str | None = None,
    ) -> RetakePipeline:
        return LTXRetakePipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            device=device,
            streaming_prefetch_count=streaming_prefetch_count,
            loras=loras or [],
            quantization=quantization,
            video_vae_path=video_vae_path,
            audio_vae_path=audio_vae_path,
            duration_head_path=duration_head_path,
        )

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        device: torch.device,
        streaming_prefetch_count: int | None,
        *,
        loras: list[LoraPathStrengthAndSDOps],
        quantization: QuantizationPolicy | None,
        video_vae_path: str | None = None,
        audio_vae_path: str | None = None,
        duration_head_path: str | None = None,
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
        offload_mode = offload_mode_for_prefetch_count(streaming_prefetch_count, device)
        model_paths = build_model_paths(
            checkpoint_path,
            gemma_root,
            video_vae_path=video_vae_path,
            audio_vae_path=audio_vae_path,
            duration_head_path=duration_head_path,
        )
        video_vae = model_paths.video_vae()
        audio_vae = model_paths.audio_vae()
        transformer = model_paths.transformer()

        self.prompt_encoder = PromptEncoder(
            model_paths,
            self.dtype,
            device,
            offload_mode=offload_mode,
        )
        self.image_conditioner = ImageConditioner(
            video_vae,
            self.dtype,
            device,
        )
        self.audio_conditioner = AudioConditioner(
            audio_vae,
            self.dtype,
            device,
        )
        self.stage = DiffusionStage.from_checkpoint(  # type: ignore[reportUnknownMemberType]
            transformer,
            self.dtype,
            device,
            loras=tuple(loras),
            quantization=quantization,
            offload_mode=offload_mode,
        )
        self.video_decoder = VideoDecoder(
            video_vae,
            self.dtype,
            device,
        )
        self.audio_decoder = AudioDecoder(
            audio_vae,
            self.dtype,
            device,
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
        extend_frames: int = 0,
        extend_at: ExtendMode = "end",
        target_width: int | None = None,
        target_height: int | None = None,
        target_frames: int | None = None,
    ) -> tuple[Iterator[torch.Tensor], Audio, TilingConfigType]:
        from ltx_core.components.guiders import MultiModalGuider
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.components.schedulers import LTX2Scheduler
        from ltx_core.conditioning.types.noise_mask_cond import TemporalRegionMask
        from ltx_core.types import AudioLatentShape, VideoLatentShape
        from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES as _distilled_sigmas
        from ltx_pipelines.utils.denoisers import GuidedDenoiser, SimpleDenoiser
        from ltx_pipelines.utils.helpers import audio_latent_from_file, video_latent_from_file
        from ltx_pipelines.utils.types import ModalitySpec

        is_extend = extend_frames > 0
        if not is_extend and start_time >= end_time:
            raise ValueError(f"start_time ({start_time}) must be less than end_time ({end_time})")

        effective_seed = int(torch.randint(0, 2**31, (1,)).item()) if seed < 0 else seed
        generator = torch.Generator(device=self.device).manual_seed(effective_seed)
        noiser = GaussianNoiser(generator=generator)

        dtype = self.dtype
        # Smaller tiles for source video encoding to reduce peak VRAM allocation
        # during the VAE encoder forward pass.
        encoding_tiling = TileSizeConfig(
            frames=DimensionSizeConfig(tile_size=24, overlap=16),
            height=DimensionSizeConfig(tile_size=256, overlap=64),
            width=DimensionSizeConfig(tile_size=256, overlap=64),
        )

        # --- Encode source video (tiled) ---
        output_shape = get_videostream_metadata(video_path)
        # Optional downscale: encode/generate at the requested (already 32-corrected) size.
        # video_latent_from_file resizes the source frames to output_shape during encoding.
        if target_width is not None and target_height is not None:
            output_shape = output_shape._replace(width=target_width, height=target_height)
        # Optional frame-count trim: corrected to a valid 8k+1 source length.
        if target_frames is not None:
            output_shape = output_shape._replace(frames=target_frames)

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

        # --- Resolve target shape + the temporal region to regenerate ---
        # Retake regenerates an interior window [start_time, end_time]; extend grows the
        # latent (zeros prepended/appended in latent-frame space, VAE temporal factor 8)
        # and regenerates only the new region. The source is frozen by the mask either way.
        if is_extend:
            target_shape = output_shape._replace(frames=output_shape.frames + extend_frames)
            pad_video_frames = (
                VideoLatentShape.from_pixel_shape(target_shape).frames
                - VideoLatentShape.from_pixel_shape(output_shape).frames
            )
            if initial_video_latent is not None:
                initial_video_latent = self._pad_latent_frames(initial_video_latent, pad_video_frames, extend_at)
            if initial_audio_latent is not None:
                pad_audio_frames = (
                    AudioLatentShape.from_video_pixel_shape(target_shape).frames
                    - AudioLatentShape.from_video_pixel_shape(output_shape).frames
                )
                initial_audio_latent = self._pad_latent_frames(initial_audio_latent, pad_audio_frames, extend_at)
            # Feather the seam by MASK_DELTA frames INTO the kept source on the side adjacent
            # to the new region (matches the cloud gateway), so the model regenerates a short
            # tail/lead of real source and blends the join instead of cutting hard.
            mask_delta_frames = round(_EXTEND_MASK_DELTA_SECONDS * output_shape.fps)
            if extend_at == "start":
                # New content leads; extend the mask forward into the source start.
                region_start = 0.0
                region_end = min(target_shape.frames, extend_frames + mask_delta_frames) / output_shape.fps
            else:
                # New content trails; pull the mask back into the source tail.
                region_start = max(0, output_shape.frames - mask_delta_frames) / output_shape.fps
                region_end = target_shape.frames / output_shape.fps
        else:
            target_shape = output_shape
            region_start, region_end = start_time, end_time


        # --- Text encoding ---

        prompts_to_encode = [prompt] if distilled else [prompt, negative_prompt]
        contexts = self.prompt_encoder(
            prompts_to_encode,
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_seed=effective_seed,
        )


        v_context_p, a_context_p = contexts[0].video_encoding, contexts[0].audio_encoding

        # --- Build modality specs ---
        video_modality_spec = ModalitySpec(
            context=v_context_p,
            conditionings=[TemporalRegionMask(start_time=region_start, end_time=region_end, fps=output_shape.fps)]
            if regenerate_video
            else [],
            initial_latent=initial_video_latent,
            frozen=not regenerate_video,
        )
        audio_modality_spec: ModalitySpec | None = None
        if a_context_p is not None:
            audio_modality_spec = ModalitySpec(
                context=a_context_p,
                conditionings=[TemporalRegionMask(start_time=region_start, end_time=region_end, fps=output_shape.fps)]
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
            width=target_shape.width,
            height=target_shape.height,
            frames=target_shape.frames,
            fps=target_shape.fps,
            video=video_modality_spec,
            audio=audio_modality_spec,
        )


        # --- Decode audio first (eager, small) ---
        assert audio_state is not None
        decoded_audio = self.audio_decoder(audio_state.latent)

        # --- Decode video (lazy generator, tiled) ---
        assert video_state is not None
        tiling = resolve_tiling_config(
            self.video_decoder.checkpoint_path,
            height=target_shape.height,
            width=target_shape.width,
            num_frames=target_shape.frames,
            device=self.device,
        )
        decoded_video = self.video_decoder(video_state.latent, tiling, generator)

        return decoded_video, decoded_audio, tiling

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
        target_width: int | None = None,
        target_height: int | None = None,
        target_frames: int | None = None,
    ) -> None:
        meta = get_videostream_metadata(video_path)
        fps = meta.fps
        num_frames = target_frames if target_frames is not None else meta.frames
        video_iter, audio, tiling_config = self._run(
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
            target_width=target_width,
            target_height=target_height,
            target_frames=target_frames,
        )
        audio_out: Audio | None = audio
        video_chunks = get_video_chunks_number(num_frames, tiling_config)
        encode_video(
            video=video_iter,
            fps=int(fps),
            audio=audio_out,
            output_path=output_path,
            video_chunks_number=video_chunks,
        )

    @torch.no_grad()
    def extend(
        self,
        *,
        video_path: str,
        prompt: str,
        extend_frames: int,
        mode: ExtendMode,
        seed: int,
        output_path: str,
        negative_prompt: str = "",
        regenerate_audio: bool = True,
        enhance_prompt: bool = False,
        distilled: bool = True,
        target_width: int | None = None,
        target_height: int | None = None,
        target_frames: int | None = None,
    ) -> None:
        meta = get_videostream_metadata(video_path)
        fps = meta.fps
        source_frames = target_frames if target_frames is not None else meta.frames
        total_frames = source_frames + extend_frames
        video_iter, audio, tiling_config = self._run(
            video_path=video_path,
            prompt=prompt,
            start_time=0.0,
            end_time=0.0,
            seed=seed,
            negative_prompt=negative_prompt,
            video_guider_params=None,
            audio_guider_params=None,
            regenerate_video=True,
            regenerate_audio=regenerate_audio,
            enhance_prompt=enhance_prompt,
            distilled=distilled,
            extend_frames=extend_frames,
            extend_at=mode,
            target_width=target_width,
            target_height=target_height,
            target_frames=target_frames,
        )
        video_chunks = get_video_chunks_number(total_frames, tiling_config)
        encode_video(
            video=video_iter,
            fps=int(fps),
            audio=audio,
            output_path=output_path,
            video_chunks_number=video_chunks,
        )

    @staticmethod
    def _pad_latent_frames(latent: torch.Tensor, pad_frames: int, at: ExtendMode) -> torch.Tensor:
        """Zero-pad a latent on its temporal axis (dim 2): front for ``start``, back for ``end``."""
        if pad_frames <= 0:
            return latent
        pad_shape = list(latent.shape)
        pad_shape[2] = pad_frames
        pad = torch.zeros(pad_shape, device=latent.device, dtype=latent.dtype)
        return torch.cat([pad, latent] if at == "start" else [latent, pad], dim=2)
