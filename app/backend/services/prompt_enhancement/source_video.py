"""The source video of a job whose prompt rewrite sees stills of it."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceVideo:
    """One source video and the rules for how its rewrite uses it."""

    path: str
    # ``(start, end)`` seconds of the video that the job replaces (Retake). The stills come
    # from around it, never from inside it.
    replaced_window: tuple[float, float] | None = None
    # The look image of an IC-LoRA that takes one. The rewrite sees it before the stills.
    reference_image_path: str | None = None
    # Whether a blank prompt is written from the video. Extend and Retake do. An IC-LoRA keeps
    # a blank prompt blank.
    writes_blank_prompt: bool = False
