"""Prompt enhancer pipeline protocol definitions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from services.prompt_enhancement.i2v_frames import KeyframeStill, SceneFrame, VideoRole


@runtime_checkable
class PromptEnhancerPipeline(Protocol):
    @staticmethod
    def create(gemma_root: str, device: str) -> "PromptEnhancerPipeline":
        ...

    def enhance_t2v(
        self,
        prompt: str,
        system_prompt: str | None,
        seed: int,
        negative_prompt: str = "",
    ) -> str:
        ...

    def enhance_i2v(
        self,
        prompt: str,
        image_path: str,
        system_prompt: str | None,
        seed: int,
        last_image_path: str | None = None,
        keyframes: list[KeyframeStill] | None = None,
        duration: int | None = None,
        fps: int | None = None,
        negative_prompt: str = "",
    ) -> str:
        ...

    def enhance_with_video_frames(
        self,
        prompt: str,
        frames: Sequence[SceneFrame],
        system_prompt: str | None,
        seed: int,
        role: VideoRole,
        negative_prompt: str = "",
    ) -> str:
        ...
