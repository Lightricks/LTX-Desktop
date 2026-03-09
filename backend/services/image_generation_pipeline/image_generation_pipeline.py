"""Image generation pipeline protocol definitions."""

from __future__ import annotations

from typing import Protocol

from services.services_utils import ImagePipelineOutputLike, PILImageType


class ImageGenerationPipeline(Protocol):
    @staticmethod
    def create(
        model_path: str,
        device: str | None = None,
    ) -> "ImageGenerationPipeline":
        ...

    def generate(
        self,
        prompt: str,
        height: int,
        width: int,
        guidance_scale: float,
        num_inference_steps: int,
        seed: int,
    ) -> ImagePipelineOutputLike:
        ...

    def img2img(
        self,
        prompt: str,
        image: PILImageType,
        strength: float,
        height: int,
        width: int,
        guidance_scale: float,
        num_inference_steps: int,
        seed: int,
    ) -> ImagePipelineOutputLike:
        ...

    def to(self, device: str) -> None:
        ...

    def load_lora(self, lora_path: str, weight: float = 1.0) -> None:
        ...

    def unload_lora(self) -> None:
        ...
