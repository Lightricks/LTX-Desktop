"""Image API client protocol for cloud image generation."""

from __future__ import annotations

from typing import Protocol


class ImageAPIClient(Protocol):
    def generate_text_to_image(
        self,
        *,
        api_key: str,
        model: str,
        prompt: str,
        width: int,
        height: int,
        seed: int,
        num_inference_steps: int,
    ) -> bytes:
        ...
