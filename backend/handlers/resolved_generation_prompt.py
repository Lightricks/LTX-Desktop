"""The prompt a local generation submits, and how it got enhanced."""

from __future__ import annotations

from typing import Literal, NamedTuple

from api_types import PromptProvenance
from services.analytics import prompt_provenance_label as shared_prompt_provenance_label


class ResolvedGenerationPrompt(NamedTuple):
    """The prompt a local generation should submit, and how it got (or will get) enhanced."""

    prompt: str
    enhance_via_api: bool
    enhanced_locally: bool
    skip_recipe_wrap: bool = False

    @property
    def enhancement(self) -> str:
        """Where the rewrite happens, for run logs: local, server-side, or nowhere."""
        if self.enhanced_locally:
            return "local"
        return "api" if self.enhance_via_api else "none"


PromptProvenanceLabel = Literal["raw", "manually-enhanced", "auto-enhanced"]


def prompt_provenance_label(
    provenance: PromptProvenance,
    resolved: ResolvedGenerationPrompt,
) -> PromptProvenanceLabel:
    """How this run asked for the prompt to be passed. Not the prompt text."""
    return shared_prompt_provenance_label(
        provenance,
        enhanced_locally=resolved.enhanced_locally,
        enhance_via_api=resolved.enhance_via_api,
    )
