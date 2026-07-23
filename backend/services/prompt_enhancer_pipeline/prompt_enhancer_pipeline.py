"""Prompt enhancer pipeline protocol definitions."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class PromptEnhancerPipeline(Protocol):
    @staticmethod
    def create(gemma_root: str, device: str) -> "PromptEnhancerPipeline":
        ...

    def enhance_t2v(self, prompt: str, system_prompt: str | None, seed: int) -> str:
        ...

    def enhance_i2v(self, prompt: str, image_path: str, system_prompt: str | None, seed: int) -> str:
        ...
