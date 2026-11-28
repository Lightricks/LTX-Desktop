"""What the automatic rewrite did to an image prompt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ImagePromptRewrite:
    """``source`` is None when nothing rewrote the prompt. ``wanted`` is False when no rewrite
    was asked for: auto-enhance is off, or the prompt is already enhanced."""

    prompt: str
    source: Literal["local", "gemini"] | None
    wanted: bool
