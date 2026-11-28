"""The prompt a queued fal image job submits, and whether fal expands it.

The rewrite runs where capacity is free (see ``PromptEnhancementHandler.rewrite_image_prompt``):

- Auto-enhance off, or the prompt already enhanced: no rewrite, no expansion.
- Local Gemma is downloaded and the GPU is idle: Gemma rewrites, fal does not expand.
- Local Gemma is downloaded and the GPU is busy: fal expands.
- No Gemma, Gemini set up: Gemini rewrites, fal does not expand.
- Nothing available, or the rewrite fails: fal expands.

fal bills its expansion per request, so it is the fallback, never the first choice.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass

from api_types import PromptProvenance, PromptRewriteSource
from handlers.prompt_enhancement_handler import PromptEnhancementHandler


@dataclass(frozen=True)
class FalImagePrompt:
    prompt: str
    expand_on_provider: bool
    rewrite_source: PromptRewriteSource | None


def plan_fal_image_prompt(
    enhancement: PromptEnhancementHandler,
    prompt: str,
    *,
    provenance: PromptProvenance,
    generation_seed: int,
    explicit_generation_seed: bool,
    lease_idle_gpu: Callable[[], AbstractContextManager[bool]],
    raise_if_cancelled: Callable[[], None],
) -> FalImagePrompt:
    """Plan the prompt. ``raise_if_cancelled`` runs before each step that takes long."""
    rewrite = enhancement.rewrite_image_prompt(
        prompt,
        provenance=provenance,
        generation_seed=generation_seed,
        explicit_generation_seed=explicit_generation_seed,
        lease_gpu=lease_idle_gpu,
        raise_if_cancelled=raise_if_cancelled,
    )
    if rewrite.source is not None:
        return FalImagePrompt(rewrite.prompt, expand_on_provider=False, rewrite_source=rewrite.source)
    if not rewrite.wanted:
        return FalImagePrompt(prompt, expand_on_provider=False, rewrite_source=None)
    return FalImagePrompt(prompt, expand_on_provider=True, rewrite_source="fal_expansion")
