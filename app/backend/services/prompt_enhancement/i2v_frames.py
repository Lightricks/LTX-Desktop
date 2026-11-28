"""Resolve the vision stills for an image-conditioned enhance call."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

# path, frame index, lock strength (0–1). Strength is owned by the still; 1.0 is a full lock.
KeyframeStill = tuple[str, int, float]

# How a source video steers the rewrite. "driving": the video is the IC-LoRA input, so the
# rewrite must match what it shows. "scene": the video is context for Extend/Retake, so the
# typed prompt leads and the frames only describe the setting. "panels": the images are the
# panels of an image-list IC-LoRA (Ingredients), not stills of a video.
VideoRole = Literal["driving", "scene", "panels"]


@dataclass(frozen=True)
class SceneFrame:
    """One sampled still of a source video, with its position in the clip."""

    path: str
    time_seconds: float
    # Where the still sits relative to a part of the clip being replaced, or "" for none.
    note: str = ""
    # A look reference image the user supplied, not a frame of the clip. Its time is unused.
    is_reference: bool = False
    # The label the model reads, when the default "Frame i of n" does not fit.
    label: str = ""


def reference_still(path: str, *, is_clip_frame: bool) -> SceneFrame:
    """The look reference image of an IC-LoRA, placed before the clip's stills.

    ``is_clip_frame`` is True when the IC-LoRA conditions the clip's first frame on it
    (Restore) and False when it sits before the clip (Layout to Render).
    """
    note = "the finished first frame of the clip" if is_clip_frame else "not a frame of the clip"
    return SceneFrame(path=path, time_seconds=0.0, note=note, is_reference=True)


def reference_sheet_still(path: str) -> SceneFrame:
    """The stitched reference sheet of an image-list IC-LoRA (Ingredients)."""
    return SceneFrame(path=path, time_seconds=0.0, label="Reference sheet.")


_PANEL_LABEL_RE = re.compile(r"\bpanel\s+\d+(?:\s+of(?:\s+\d+)?)?\s*[:,.-]?\s*", re.IGNORECASE)


def strip_panel_labels(text: str) -> str:
    """Remove the "Panel 1 of 4" labels a rewriter copies from its input into its reply."""
    return _PANEL_LABEL_RE.sub("", text)


def _keyframe_label(index: int, frame_idx: int, fps: int | float | None, strength: float) -> str:
    clock = f" ({frame_idx / fps:.2f}s)" if fps is not None and fps > 0 else ""
    return f"Keyframe {index + 1}: frame {frame_idx}{clock}, strength {strength:g}."


def _scene_label(index: int, total: int, frame: SceneFrame) -> str:
    note = f", {frame.note}" if frame.note else ""
    return f"Frame {index + 1} of {total} ({frame.time_seconds:.2f}s{note})."


def resolve_video_frames(frames: Sequence[SceneFrame]) -> list[tuple[str, str]]:
    """Return ``(path, label)`` pairs for a source video's stills.

    A look reference image comes first. The clip's stills follow in time order.
    """
    references = [frame for frame in frames if frame.is_reference]
    ordered = sorted(
        (frame for frame in frames if not frame.is_reference), key=lambda frame: frame.time_seconds
    )
    return [(frame.path, f"Look reference image ({frame.note}).") for frame in references] + [
        (frame.path, frame.label or _scene_label(index, len(ordered), frame))
        for index, frame in enumerate(ordered)
    ]


def resolve_i2v_frames(
    image_path: str,
    last_image_path: str | None = None,
    keyframes: Sequence[KeyframeStill] | None = None,
    fps: int | float | None = None,
) -> list[tuple[str, str | None]]:
    """Return ``(path, label)`` pairs in vision order.

    A missing label means the still is unlabeled (single-image i2v, or a lone
    multi-keyframe still when fps is unknown). First/last i2v keeps the existing
    ``First frame:`` / ``Last frame:`` copy. Multi-keyframe stills are numbered
    and tagged with their frame index, clock time when fps is known, and lock
    strength so the rewriter can weight identity lock vs interpolation.
    """
    if keyframes:
        ordered = sorted(keyframes, key=lambda item: (item[1], item[0]))
        if len(ordered) == 1 and (fps is None or fps <= 0):
            return [(ordered[0][0], None)]
        return [
            (path, _keyframe_label(index, frame_idx, fps, strength))
            for index, (path, frame_idx, strength) in enumerate(ordered)
        ]
    if last_image_path:
        return [(image_path, "First frame:"), (last_image_path, "Last frame:")]
    return [(image_path, None)]
