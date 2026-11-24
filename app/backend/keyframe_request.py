"""Validation for multi-keyframe inputs on GenerateVideoRequest."""

from __future__ import annotations

from collections.abc import Sequence

from api_types import GenerateVideoRequest
from frame_math import compute_num_frames
from server_utils.media_validation import normalize_optional_path


def validate_keyframe_frames(frame_indices: Sequence[int], *, duration: int, fps: int) -> str | None:
    """The frame checks for any multi-keyframe request: unique, and inside the clip."""
    if len(set(frame_indices)) != len(frame_indices):
        return "Keyframe frame indices must be unique"

    last_frame = compute_num_frames(duration, fps) - 1
    for frame_idx in frame_indices:
        if frame_idx < 0 or frame_idx > last_frame:
            return f"Keyframe frame index {frame_idx} is outside 0..{last_frame}"

    return None


def validate_keyframe_inputs(
    req: GenerateVideoRequest,
    *,
    use_api_specs: bool,
    image_path: str | None,
    last_image_path: str | None,
    audio_path: str | None,
) -> str | None:
    entries = req.keyframes
    if not entries:
        return None

    if use_api_specs:
        return "Multi-keyframe generation is only available for local generation"

    if image_path or last_image_path:
        return "Keyframes cannot be combined with a first or last frame"

    if audio_path:
        return "Keyframes cannot be combined with audio-to-video"

    if req.duration is None:
        return "Keyframes cannot be combined with automatic duration"

    if any(normalize_optional_path(keyframe.imagePath) is None for keyframe in entries):
        return "Each keyframe requires an image path"

    return validate_keyframe_frames(
        [keyframe.frameIndex for keyframe in entries], duration=req.duration, fps=req.fps
    )
