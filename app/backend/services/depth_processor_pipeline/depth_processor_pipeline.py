"""Depth processor protocol definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from services.services_utils import FrameArray

if TYPE_CHECKING:
    import torch


class DepthProcessorPipeline(Protocol):
    @staticmethod
    def create(
        model_path: str,
        device: torch.device,
    ) -> "DepthProcessorPipeline":
        ...

    def apply(self, frame: FrameArray) -> FrameArray:
        ...
