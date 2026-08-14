"""Shared helpers and primitives for LTX video pipeline wrappers."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch

from api_types import ImageConditioningInput
from services.services_utils import AudioOrNone, PipelineTilingType, TilingConfigType, device_supports_fp8

if TYPE_CHECKING:
    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_pipelines.utils.types import OffloadMode


def auto_tiling_config() -> PipelineTilingType:
    """Let the pipeline derive decode tiling from the VAE it will decode with.

    A conv VAE (2.3 monolith) and a diffusion VAE (2.5 split) need different tile
    overlaps, so a fixed layout that one accepts the other rejects.
    """
    from ltx_core.model.video_vae import AUTO_TILING

    return AUTO_TILING


def host_available_bytes() -> int:
    """Currently available system RAM in bytes (unified memory on Apple Silicon)."""
    import psutil

    return int(psutil.virtual_memory().available)


def diffvae_activation_budget_bytes(device: torch.device | None = None) -> int:
    """Bytes DiffVAE decode tiling may treat as free activation memory.

    ltx-pipelines only queries the CUDA allocator. On MPS/CPU that path yields 0,
    so AUTO_TILING raises ``Cannot fit a DiffVAE decode tile`` before decode.
    CUDA keeps the upstream allocator budget; everywhere else uses available RAM.
    """
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        from ltx_core.devices import cuda_activation_budget_bytes

        return int(cuda_activation_budget_bytes(device))
    return host_available_bytes()


def resolve_diffvae_free_bytes(device: torch.device | None, free_bytes: int | None) -> int | None:
    """Fill a DiffVAE tiling budget when upstream would treat non-CUDA as 0."""
    if free_bytes is not None and free_bytes > 0:
        return free_bytes
    if device is not None and device.type == "cuda":
        return free_bytes
    return host_available_bytes()


def resolve_tiling_config(
    vae_checkpoint_path: str,
    *,
    height: int,
    width: int,
    num_frames: int,
    device: torch.device | None = None,
) -> TilingConfigType:
    """Same recommendation ``AUTO_TILING`` resolves to, for pipelines that decode themselves."""
    from ltx_pipelines.utils.helpers import get_device, tiling_config_for_vae

    if device is None:
        device = get_device()
    return tiling_config_for_vae(
        vae_checkpoint_path,
        height=height,
        width=width,
        num_frames=num_frames,
        device=device,
        free_bytes=diffvae_activation_budget_bytes(device),
    )


def build_model_paths(
    checkpoint_path: str,
    gemma_root: str | None,
    *,
    video_vae_path: str | None = None,
    audio_vae_path: str | None = None,
    duration_head_path: str | None = None,
) -> ModelPaths:
    """Build ``ModelPaths`` for monolith (2.3) or split (2.5) checkpoint layouts.

    When both VAE paths are provided, uses ``from_split`` (LTX 2.5). Otherwise uses
    ``from_monolith`` where the fat checkpoint also supplies the VAEs and DurationHead.
    Split 2.5 DurationHead is a separate safetensors; omit ``duration_head_path`` and
    AutoDuration fails closed in the pipeline.
    """
    from ltx_pipelines.utils.model_paths import ModelPaths

    if video_vae_path is not None and audio_vae_path is not None:
        return ModelPaths.from_split(
            transformer_path=checkpoint_path,
            text_encoder_path=gemma_root,
            video_vae_path=video_vae_path,
            audio_vae_path=audio_vae_path,
            duration_head_path=duration_head_path,
        )
    return ModelPaths.from_monolith(checkpoint_path, gemma_root, video_vae_path=video_vae_path)


def default_guiders() -> tuple[MultiModalGuiderParams, MultiModalGuiderParams]:
    from ltx_core.components.guiders import MultiModalGuiderParams

    return MultiModalGuiderParams(cfg_scale=3.0), MultiModalGuiderParams(cfg_scale=3.0)


def video_chunks_number(num_frames: int, tiling_config: TilingConfigType | None) -> int:
    from ltx_core.model.video_vae import get_video_chunks_number

    return int(get_video_chunks_number(num_frames, tiling_config))


def offload_mode_for_prefetch_count(streaming_prefetch_count: int | None, device: torch.device) -> OffloadMode:
    """Translate the desktop's streaming_prefetch_count knob to ltx_pipelines' OffloadMode.

    ltx_pipelines moved weight streaming from a per-call prefetch-count int to a
    construction-time OffloadMode enum (NONE/CPU/DISK). Desktop's runtime policy
    (runtime_config/runtime_policy.py) distinguishes fully resident (None) vs streaming
    (an int); which *kind* of streaming depends on the device's memory model:

    - CUDA: system RAM is separate from VRAM, so OffloadMode.CPU pins the blocks in host
      RAM and streams them to the smaller VRAM — the fast streaming path.
    - MPS (Apple Silicon): CPU-pinned weights live in the *same* unified RAM as the GPU,
      so OffloadMode.CPU (which pins every block, ~46 GB for the bf16 transformer) OOMs.
      OffloadMode.DISK mmaps blocks from the checkpoint through a small pinned buffer
      (~5 GB), the only memory-safe streaming path on unified memory. This is the "mmap
      streaming" the upstream MPS-support work validated on an M4 Pro.
    """
    from ltx_pipelines.utils.types import OffloadMode

    if streaming_prefetch_count is None:
        return OffloadMode.NONE
    if device.type == "mps":
        return OffloadMode.DISK
    return OffloadMode.CPU


def encode_video_output(
    video: torch.Tensor | Iterator[torch.Tensor],
    audio: AudioOrNone,
    fps: int,
    output_path: str,
    video_chunks_number_value: int,
) -> None:
    from ltx_pipelines.utils.media_io import encode_video

    encode_video(
        video=video,
        fps=fps,
        audio=audio,
        output_path=output_path,
        video_chunks_number=video_chunks_number_value,
    )


class DistilledNativePipeline:
    """Fast native pipeline implementation moved from ltx2_server.py."""

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        device: torch.device | None = None,
        fp8transformer: bool = False,
    ) -> None:
        from ltx_core.quantization.fp8_cast import build_policy as build_fp8_cast_policy
        from ltx_pipelines.utils.blocks import (
            AudioDecoder,
            DiffusionStage,
            ImageConditioner,
            PromptEncoder,
            VideoDecoder,
        )
        from ltx_pipelines.utils.helpers import get_device

        if device is None:
            device = get_device()

        self.device = device
        self.dtype = torch.bfloat16
        model_paths = build_model_paths(checkpoint_path, gemma_root)

        self.prompt_encoder = PromptEncoder(
            model_paths, self.dtype, device,
        )
        self.image_conditioner = ImageConditioner(
            checkpoint_path, self.dtype, device,
        )
        self.stage = DiffusionStage.from_checkpoint(  # type: ignore[reportUnknownMemberType]
            checkpoint_path,
            self.dtype,
            device,
            quantization=build_fp8_cast_policy(checkpoint_path) if fp8transformer and device_supports_fp8(device) else None,
        )
        self.video_decoder = VideoDecoder(checkpoint_path, self.dtype, device)
        self.audio_decoder = AudioDecoder(checkpoint_path, self.dtype, device)

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfigType | None = None,
    ) -> tuple[torch.Tensor | Iterator[torch.Tensor], AudioOrNone]:
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_pipelines.utils.args import ImageConditioningInput as _LtxImageInput
        from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES
        from ltx_pipelines.utils.denoisers import SimpleDenoiser
        from ltx_pipelines.utils.helpers import image_conditionings_by_replacing_latent
        from ltx_pipelines.utils.types import ModalitySpec

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = torch.bfloat16

        (ctx_p,) = self.prompt_encoder([prompt])
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding

        sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)

        ltx_images = [_LtxImageInput(img.path, img.frame_idx, img.strength) for img in images]
        conditionings = self.image_conditioner(
            lambda enc: image_conditionings_by_replacing_latent(
                images=ltx_images,
                height=height,
                width=width,
                video_encoder=enc,
                dtype=dtype,
                device=self.device,
            )
        )

        video_state, audio_state = self.stage(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(context=video_context, conditionings=conditionings),
            audio=ModalitySpec(context=audio_context) if audio_context is not None else None,
        )

        assert video_state is not None
        decoded_video = self.video_decoder(video_state.latent, tiling_config)
        decoded_audio = self.audio_decoder(audio_state.latent) if audio_state is not None else None
        return decoded_video, decoded_audio
