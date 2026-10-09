"""LTX retake pipeline wrapper.

Forked orchestration of the retake pipeline flow from ``ltx_pipelines.retake``
with the following adjustments:

* ``@torch.no_grad()`` instead of ``@torch.inference_mode()`` — the
  transformer checkpoint uses custom autograd functions incompatible with
  inference-mode tensors.
* Tiled video encoding from CPU pixels via ``encode_source_video_latent``
  (assembling the frame tensor on CUDA reserved ~67 GiB on Windows WDDM and
  hung the second encode after DiffVAE decode).
* Tiled video decoding via ``VideoDecoder(..., tiling_config)`` — the
  original omits the tiling argument.
* Distilled 2.5 retake and extend use ancestral stage 1 plus a same-resolution
  Euler refine (T2V's pair without the 2× upsample). Euler-only on that
  8-step schedule never lets the prompt take.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from collections.abc import Iterator
from typing import Any
import torch

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import get_video_chunks_number
from ltx_core.loader.registry import DummyRegistry
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, SpatioTemporalScaleFactors
from ltx_pipelines.distilled import should_use_ancestral_sampler
from ltx_pipelines.utils.media_io import encode_video, get_videostream_metadata
from ltx_pipelines.utils.types import ModalitySpec, VideoAudio

from api_types import ExtendMode
from services.ltx_pipeline_common import build_model_paths, offload_mode_for_prefetch_count, resolve_tiling_config
from services.services_utils import TilingConfigType
from services.retake_pipeline.frame_rate import restamp_to_source_rate
from services.retake_pipeline.retake_pipeline import RetakePipeline
from services.retake_pipeline.retake_sampler import resolve_sampler_plan
from services.retake_pipeline.source_encode import (
    encode_source_video_latent,
    extend_freeze_pad_frames,
    source_encode_tiling,
)
from services.retake_pipeline.window import MASK_DELTA_SECONDS

logger = logging.getLogger(__name__)


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
        self.use_ancestral_sampler = should_use_ancestral_sampler(transformer)

        self.prompt_encoder = PromptEncoder(
            model_paths,
            self.dtype,
            device,
            offload_mode=offload_mode,
        )
        # cache_models shells keep DiffVAE NA workspaces across dispose; the next
        # encode then hangs on Windows WDDM with reserved>>device. Do not cache.
        vae_registry = DummyRegistry()
        self.image_conditioner = ImageConditioner(
            video_vae,
            self.dtype,
            device,
            registry=vae_registry,
        )
        self.audio_conditioner = AudioConditioner(
            audio_vae,
            self.dtype,
            device,
            registry=vae_registry,
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
            registry=vae_registry,
        )
        self.audio_decoder = AudioDecoder(
            audio_vae,
            self.dtype,
            device,
            registry=vae_registry,
        )

    def _denoise_masked_latents(
        self,
        *,
        distilled: bool,
        seed: int,
        denoiser: Any,
        sigmas: torch.Tensor,
        noiser: Any,
        video: ModalitySpec,
        audio: ModalitySpec | None,
    ) -> Any:
        """Ancestral stage 1 + Euler refine for distilled 2.5; Euler-only otherwise."""
        plan = resolve_sampler_plan(
            distilled=distilled,
            use_ancestral=self.use_ancestral_sampler,
            seed=seed,
            dtype=self.dtype,
        )
        video_state, audio_state = self.stage(
            denoiser=denoiser,
            sigmas=sigmas,
            noiser=noiser,
            modalities=VideoAudio(
                video=_pin_frozen_noise(video),
                audio=None if audio is None else _pin_frozen_noise(audio),
            ),
            **plan.stage1,
        )
        video_state, audio_state = _restore_frozen(video_state, video), _restore_frozen(audio_state, audio)
        if not plan.two_stage:
            return video_state, audio_state
        assert video_state is not None
        from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMA_VALUES

        stage_2_sigmas = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(
            dtype=torch.float32, device=self.device
        )
        noise_scale = float(stage_2_sigmas[0].item())
        video_state, audio_state = self.stage(
            denoiser=denoiser,
            sigmas=stage_2_sigmas,
            noiser=noiser,
            modalities=VideoAudio(
                video=_with_stage2_latent(video, video_state.latent, noise_scale),
                audio=(
                    None
                    if audio is None or audio_state is None
                    else _with_stage2_latent(audio, audio_state.latent, noise_scale)
                ),
            ),
            **plan.refine,
        )
        return _restore_frozen(video_state, video), _restore_frozen(audio_state, audio)

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
        encode_start_time: float = 0.0,
        encode_max_duration: float | None = None,
        content_width: int | None = None,
        content_height: int | None = None,
    ) -> tuple[Iterator[torch.Tensor], Audio, TilingConfigType]:
        from ltx_core.components.guiders import MultiModalGuider
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.components.schedulers import LTX2Scheduler
        from ltx_core.conditioning.types.noise_mask_cond import TemporalRegionMask
        from ltx_core.types import AudioLatentShape
        from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES as _distilled_sigmas
        from ltx_pipelines.utils.denoisers import GuidedDenoiser, SimpleDenoiser
        from ltx_pipelines.utils.helpers import audio_latent_from_file
        from ltx_pipelines.utils.types import ModalitySpec

        is_extend = extend_frames > 0
        if not is_extend and start_time >= end_time:
            raise ValueError(f"start_time ({start_time}) must be less than end_time ({end_time})")

        effective_seed = int(torch.randint(0, 2**31, (1,)).item()) if seed < 0 else seed
        generator = torch.Generator(device=self.device).manual_seed(effective_seed)
        noiser = GaussianNoiser(generator=generator)

        dtype = self.dtype

        # --- Encode source video (tiled) ---
        output_shape = get_videostream_metadata(video_path)
        # Home retake/extend pass a ÷32 canvas plus the picture size. The encode
        # letterboxes the picture onto that canvas. Legacy callers omit the
        # picture size and keep the center-crop resize.
        if target_width is not None and target_height is not None:
            output_shape = output_shape._replace(width=target_width, height=target_height)
        # Optional frame-count trim: corrected to a valid 8k+1 source length.
        if target_frames is not None:
            output_shape = output_shape._replace(frames=target_frames)

        encoding_tiling = source_encode_tiling(
            self.device, height=output_shape.height, width=output_shape.width
        )
        logger.info(
            "Encoding source video %dx%d %d frames (tiled, CPU pixels) start=%.2fs duration=%s",
            output_shape.width,
            output_shape.height,
            output_shape.frames,
            encode_start_time,
            encode_max_duration,
        )
        video_encode_shape = output_shape
        freeze_pad_frames = 0
        if is_extend:
            video_encode_shape = output_shape._replace(
                frames=output_shape.frames + extend_frames
            )
            freeze_pad_frames = extend_freeze_pad_frames(self.device, extend_frames)
            if freeze_pad_frames == 0:
                logger.info(
                    "MPS extend encodes %d source frames and repeats the edge latent for +%d",
                    output_shape.frames,
                    extend_frames,
                )
        initial_video_latent = self.image_conditioner(
            lambda enc: encode_source_video_latent(
                enc,
                video_path,
                video_encode_shape,
                dtype=dtype,
                tiling_config=encoding_tiling,
                start_time=encode_start_time,
                max_duration=encode_max_duration,
                freeze_pad_frames=freeze_pad_frames,
                freeze_at=extend_at,
                content_width=content_width,
                content_height=content_height,
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
                start_time=encode_start_time,
                max_duration=encode_max_duration,
            )
        )

        # --- Resolve target shape + the temporal region to regenerate ---
        # Retake regenerates an interior window [start_time, end_time]. Extend
        # on CUDA freeze-pads pixels (cloud ``pad_video``) so seam tiles encode
        # against a held frame. MPS repeats the edge latent instead. Either
        # way only the new region plus MASK_DELTA into the source is denoised.
        # Audio still grows in latent space (silence), matching cloud's silent pad.
        if is_extend:
            target_shape = output_shape._replace(frames=output_shape.frames + extend_frames)
            if initial_audio_latent is not None:
                pad_audio_frames = (
                    AudioLatentShape.from_video_pixel_shape(target_shape).frames
                    - AudioLatentShape.from_video_pixel_shape(output_shape).frames
                )
                initial_audio_latent = self._pad_latent_frames(
                    initial_audio_latent, pad_audio_frames, extend_at
                )
            # Feather the seam by MASK_DELTA frames INTO the kept source on the side adjacent
            # to the new region (matches the cloud gateway), so the model regenerates a short
            # tail/lead of real source and blends the join instead of cutting hard.
            mask_delta_frames = round(MASK_DELTA_SECONDS * output_shape.fps)
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
            latent=initial_video_latent,
            conditioning_fps=output_shape.fps,
            context=v_context_p,
            conditionings=[TemporalRegionMask(start_time=region_start, end_time=region_end, fps=output_shape.fps)]
            if regenerate_video
            else [],
            frozen=not regenerate_video,
        )
        audio_modality_spec = (
            None
            if a_context_p is None
            else _audio_modality_spec(
                self.stage.video_scale_factors,
                source_latent=initial_audio_latent,
                video_latent=initial_video_latent,
                context=a_context_p,
                fps=output_shape.fps,
                region_start=region_start,
                region_end=region_end,
                regenerate_audio=regenerate_audio,
            )
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
        video_state, audio_state = self._denoise_masked_latents(
            distilled=distilled,
            seed=effective_seed,
            denoiser=denoiser,
            sigmas=sigmas,
            noiser=noiser,
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
        encode_start_time: float = 0.0,
        encode_max_duration: float | None = None,
        content_width: int | None = None,
        content_height: int | None = None,
    ) -> None:
        meta = get_videostream_metadata(video_path)
        fps = meta.fps
        num_frames = target_frames if target_frames is not None else meta.frames
        from services.denoising_progress import track_denoising

        plan = resolve_sampler_plan(
            distilled=distilled,
            use_ancestral=self.use_ancestral_sampler,
            seed=seed,
            dtype=self.dtype,
        )
        denoise_steps = plan.total_steps if distilled else num_inference_steps
        with track_denoising(denoise_steps):
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
                encode_start_time=encode_start_time,
                encode_max_duration=encode_max_duration,
                content_width=content_width,
                content_height=content_height,
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
        restamp_to_source_rate(output_path, source_fps=fps)

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
        encode_start_time: float = 0.0,
        encode_max_duration: float | None = None,
        content_width: int | None = None,
        content_height: int | None = None,
    ) -> None:
        meta = get_videostream_metadata(video_path)
        fps = meta.fps
        source_frames = target_frames if target_frames is not None else meta.frames
        total_frames = source_frames + extend_frames
        from services.denoising_progress import track_denoising

        plan = resolve_sampler_plan(
            distilled=distilled,
            use_ancestral=self.use_ancestral_sampler,
            seed=seed,
            dtype=self.dtype,
        )
        denoise_steps = plan.total_steps if distilled else 40
        with track_denoising(denoise_steps):
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
                encode_start_time=encode_start_time,
                encode_max_duration=encode_max_duration,
                content_width=content_width,
                content_height=content_height,
            )
        video_chunks = get_video_chunks_number(total_frames, tiling_config)
        encode_video(
            video=video_iter,
            fps=int(fps),
            audio=audio,
            output_path=output_path,
            video_chunks_number=video_chunks,
        )
        restamp_to_source_rate(output_path, source_fps=fps)

    @staticmethod
    def _pad_latent_frames(
        latent: torch.Tensor,
        pad_frames: int,
        at: ExtendMode,
    ) -> torch.Tensor:
        """Pad a latent on its temporal axis (dim 2): front for ``start``, back for ``end``.

        Works for 5-D video ``[B, C, T, H, W]`` and 4-D audio ``[B, C, T, F]``.
        """
        if pad_frames <= 0:
            return latent
        pad_shape = list(latent.shape)
        pad_shape[2] = pad_frames
        zeros = torch.zeros(pad_shape, device=latent.device, dtype=latent.dtype)
        return torch.cat([zeros, latent] if at == "start" else [latent, zeros], dim=2)


def _audio_modality_spec(
    video_scale_factors: SpatioTemporalScaleFactors,
    *,
    source_latent: torch.Tensor | None,
    video_latent: torch.Tensor,
    context: Any,
    fps: float,
    region_start: float,
    region_end: float,
    regenerate_audio: bool,
) -> ModalitySpec:
    """Audio spec for the stage.

    v1.4 needs the latent up front, so a source with no soundtrack gets a zero latent sized to
    the video. That latent is fresh, so it is neither frozen nor masked: only real source
    audio can be kept (``frozen``) or partially regenerated (region mask).
    """
    from ltx_core.conditioning.types.noise_mask_cond import TemporalRegionMask
    from ltx_pipelines.utils.helpers import create_initial_audio_latent

    had_source_audio = source_latent is not None
    latent = (
        source_latent
        if source_latent is not None
        else create_initial_audio_latent(
            video_latent,
            fps=fps,
            # Transformer latent geometry, not the decoder's tiling scale.
            video_scale_factors=video_scale_factors,
        )
    )
    return ModalitySpec(
        latent=latent,
        conditioning_fps=fps,
        context=context,
        conditionings=[TemporalRegionMask(start_time=region_start, end_time=region_end, fps=fps)]
        if (had_source_audio and regenerate_audio)
        else [],
        frozen=had_source_audio and not regenerate_audio,
    )


def _pin_frozen_noise(spec: ModalitySpec) -> ModalitySpec:
    """A frozen modality is clean conditioning, so it must not be noised.

    ``ModalitySpec.noise_scale`` is applied before ``frozen`` zeroes the denoise mask. Left at the
    schedule's scale, the transformer would be fed a (mostly) Gaussian latent that its zero
    timestep labels as clean. Upstream's ``denoise_chunks`` pins frozen specs the same way.
    """
    return replace(spec, noise_scale=0.0) if spec.frozen else spec


def _with_stage2_latent(spec: ModalitySpec, latent: Any, noise_scale: float) -> ModalitySpec:
    return _pin_frozen_noise(replace(spec, latent=latent, noise_scale=noise_scale))


def _restore_frozen(state: Any, spec: ModalitySpec | None) -> Any:
    """Return the input latent for a frozen modality, not the loop's bf16-drifted copy of it."""
    if state is None or spec is None or not spec.frozen:
        return state
    return replace(state, latent=spec.latent)
