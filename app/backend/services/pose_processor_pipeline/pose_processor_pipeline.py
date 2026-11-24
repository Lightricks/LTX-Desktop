"""Pose processor protocol definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from services.services_utils import FrameArray

if TYPE_CHECKING:
    import torch


class PoseProcessorPipeline(Protocol):
    @staticmethod
    def create(
        pose_model_path: str,
        person_detector_model_path: str,
        device: torch.device,
    ) -> "PoseProcessorPipeline":
        ...

    def apply(self, frame: FrameArray) -> FrameArray:
        ...
