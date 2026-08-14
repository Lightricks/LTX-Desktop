"""IC-LoRA pipeline protocol definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from api_types import ImageConditioningInput

if TYPE_CHECKING:
    import torch


class IcLoraPipeline(Protocol):
    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        lora_path: str,
        device: torch.device,
        streaming_prefetch_count: int | None,
        lora_strength: float = 1.0,
        *,
        video_vae_path: str | None = None,
        audio_vae_path: str | None = None,
        duration_head_path: str | None = None,
    ) -> "IcLoraPipeline":
        ...

    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        video_conditioning: list[tuple[str, float]],
        output_path: str,
        skip_stage_2: bool = False,
        use_lora_in_stage_2: bool = False,
        resolution_factor: float = 2.0,
        source_audio_path: str | None = None,
        mute_audio: bool = False,
        conditioning_mask_path: str | None = None,
    ) -> None:
        ...
