"""GGUF quantized LTX video pipeline.

Loads LTX-Video transformer weights from a GGUF file using the diffusers
GGUF quantizer, while loading VAE, text encoder, and upsampler normally.
"""

from __future__ import annotations

import logging
from typing import Final

import torch

from api_types import ImageConditioningInput

logger = logging.getLogger(__name__)


class GGUFFastVideoPipeline:
    """FastVideoPipeline implementation for GGUF quantized models.

    Scaffold only — raises NotImplementedError until tested with real models.
    """

    pipeline_kind: Final = "fast"

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        lora_path: str | None = None,
        lora_weight: float = 1.0,
    ) -> "GGUFFastVideoPipeline":
        return GGUFFastVideoPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            device=device,
            lora_path=lora_path,
            lora_weight=lora_weight,
        )

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        lora_path: str | None = None,
        lora_weight: float = 1.0,
    ) -> None:
        try:
            import gguf  # noqa: F401  # pyright: ignore[reportUnusedImport]
        except ImportError:
            raise RuntimeError(
                "GGUF model support requires the 'gguf' package. "
                "Install it with: pip install gguf>=0.10.0"
            ) from None

        raise NotImplementedError(
            "GGUF pipeline loading is not yet fully implemented. "
            "This requires testing with real GGUF model files."
        )

    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        output_path: str,
    ) -> None:
        raise NotImplementedError

    def warmup(self, output_path: str) -> None:
        raise NotImplementedError

    def compile_transformer(self) -> None:
        logger.info("Skipping torch.compile for GGUF pipeline — not supported with quantized weights")
