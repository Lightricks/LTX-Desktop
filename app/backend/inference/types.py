"""Shared inference type aliases used by pipeline wrappers."""

from __future__ import annotations

from typing import Literal, NamedTuple, TypeAlias

LTXLocalModelId = Literal[
    "ltx-2.5-22b-distilled",
    "ltx-2.3-22b-distilled-1.1",
    "ltx-2.3-22b-distilled",
]


class ImageConditioningInput(NamedTuple):
    """Image conditioning triplet used by all video pipelines."""

    path: str
    frame_idx: int
    strength: float


RetakeMode: TypeAlias = Literal[
    "replace_audio_and_video", "replace_video", "replace_audio"
]

ExtendMode: TypeAlias = Literal["start", "end"]

# "long" encodes as much of the clip as fits the window. "short" keeps about 2 s of context
# next to the new content (Extend) or on each side of the selection (Retake), so the job
# denoises a much shorter sequence.
EditContextMode: TypeAlias = Literal["long", "short"]
# The mode a request, a preflight and a window use when none is stated.
DEFAULT_EDIT_CONTEXT_MODE: EditContextMode = "long"

# Look-still frame for Layout To Render. Upstream rejects a negative index, so
# the pipeline keeps the still apart from the clip frames.
REFERENCE_STILL_FRAME_IDX = -1
