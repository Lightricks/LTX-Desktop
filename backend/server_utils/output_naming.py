"""Shared output filename generation for all generation handlers."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def _slugify_prompt(prompt: str, max_words: int = 5) -> str:
    """Turn a prompt into a short filesystem-safe slug.

    Takes the first *max_words* words, lowercases, strips non-alphanumeric
    characters, and joins with hyphens.  Returns ``"untitled"`` for empty input.
    """
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", prompt).strip()
    words = cleaned.lower().split()[:max_words]
    slug = "-".join(words)
    return slug or "untitled"


def make_output_filename(
    *,
    model: str,
    prompt: str,
    ext: str = "mp4",
) -> str:
    """Build a descriptive output filename.

    Pattern: ``dd_{model}_{prompt_slug}_{timestamp}.{ext}``

    Examples::

        dd_ltx-fast_elegant-woman-luxury-handbag_20260309_144341.mp4
        dd_seedance_confident-woman-walks-runway_20260309_144342.mp4
        dd_zit_cyberpunk-cityscape-neon-rain_20260309_144343.png
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _slugify_prompt(prompt)
    # Sanitise model name for filesystem
    safe_model = re.sub(r"[^a-zA-Z0-9_-]", "", model) or "unknown"
    return f"dd_{safe_model}_{slug}_{timestamp}.{ext}"


def make_output_path(
    outputs_dir: Path,
    *,
    model: str,
    prompt: str,
    ext: str = "mp4",
) -> Path:
    """Build a full output path under *outputs_dir*."""
    return outputs_dir / make_output_filename(model=model, prompt=prompt, ext=ext)
