"""Retake pipeline protocol definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from api_types import ExtendMode

if TYPE_CHECKING:
    import torch
    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_core.loader import LoraPathStrengthAndSDOps
    from ltx_core.quantization import QuantizationPolicy


class RetakePipeline(Protocol):
    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        device: "torch.device",
        streaming_prefetch_count: int | None,
        *,
        loras: list["LoraPathStrengthAndSDOps"] | None = None,
        quantization: "QuantizationPolicy | None" = None,
    ) -> "RetakePipeline": ...

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
        video_guider_params: "MultiModalGuiderParams | None" = None,
        audio_guider_params: "MultiModalGuiderParams | None" = None,
        regenerate_video: bool = True,
        regenerate_audio: bool = True,
        enhance_prompt: bool = False,
        distilled: bool = True,
        target_width: int | None = None,
        target_height: int | None = None,
        target_frames: int | None = None,
    ) -> None: ...

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
    ) -> None: ...
