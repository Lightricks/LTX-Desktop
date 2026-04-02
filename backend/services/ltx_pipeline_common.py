"""Shared helpers and primitives for LTX video pipeline wrappers."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch

from api_types import ImageConditioningInput
from services.services_utils import AudioOrNone, TilingConfigType, device_supports_fp8

if TYPE_CHECKING:
    from ltx_core.components.guiders import MultiModalGuiderParams


def default_tiling_config() -> TilingConfigType:
    from ltx_core.model.video_vae import TilingConfig

    return TilingConfig.default()


def default_guiders() -> tuple[MultiModalGuiderParams, MultiModalGuiderParams]:
    from ltx_core.components.guiders import MultiModalGuiderParams

    return MultiModalGuiderParams(cfg_scale=3.0), MultiModalGuiderParams(cfg_scale=3.0)


def video_chunks_number(num_frames: int, tiling_config: TilingConfigType | None) -> int:
    from ltx_core.model.video_vae import get_video_chunks_number

    return int(get_video_chunks_number(num_frames, tiling_config))


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
        from ltx_core.quantization import QuantizationPolicy
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

        self.prompt_encoder = PromptEncoder(
            checkpoint_path, gemma_root or "", self.dtype, device,
        )
        self.image_conditioner = ImageConditioner(
            checkpoint_path, self.dtype, device,
        )
        self.stage = DiffusionStage(
            checkpoint_path,
            self.dtype,
            device,
            quantization=QuantizationPolicy.fp8_cast() if fp8transformer and device_supports_fp8(device) else None,
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
