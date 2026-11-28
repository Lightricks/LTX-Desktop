"""Aspect ratio matching shared by the generators and the dashboard."""

from __future__ import annotations

from collections.abc import Sequence

_ASPECT_MATCH_TOLERANCE = 0.01


def find_closest_aspect_ratio_option(
    width: int, height: int, options: Sequence[str]
) -> str:
    if height <= 0:
        raise ValueError("startFrame height must be greater than 0")
    target = width / height
    closest: str | None = None
    closest_diff = float("inf")
    for option in options:
        ratio = parse_aspect_ratio(option)
        if ratio is None:
            continue
        diff = abs(ratio - target)
        if diff < closest_diff:
            closest_diff = diff
            closest = option
    if closest is None:
        raise ValueError("no valid aspect ratio options")
    return closest


def parse_aspect_ratio(option: str) -> float | None:
    parts = option.split(":")
    if len(parts) != 2:
        return None
    try:
        option_width = float(parts[0])
        option_height = float(parts[1])
    except ValueError:
        return None
    if option_width <= 0 or option_height <= 0:
        return None
    return option_width / option_height


def center_crop_box(
    width: int, height: int, aspect: str
) -> tuple[int, int, int, int] | None:
    """Even crop to ``aspect``, or None when the source is already that ratio."""
    target = parse_aspect_ratio(aspect)
    if target is None or height <= 0:
        return None
    current = width / height
    if abs(current - target) / target < _ASPECT_MATCH_TOLERANCE:
        return None
    if current > target:
        crop_w = int(height * target)
        crop_w -= crop_w % 2
        crop_h = height - height % 2
        x = (width - crop_w) // 2
        y = 0
    else:
        crop_h = int(width / target)
        crop_h -= crop_h % 2
        crop_w = width - width % 2
        x = 0
        y = (height - crop_h) // 2
    x -= x % 2
    y -= y % 2
    if crop_w < 2 or crop_h < 2 or (crop_w == width and crop_h == height):
        return None
    return crop_w, crop_h, x, y


def center_cropped_size(width: int, height: int, aspect: str) -> tuple[int, int]:
    """Size of a clip after the ``center_crop_box`` crop to ``aspect``."""
    box = center_crop_box(width, height, aspect)
    return (width, height) if box is None else (box[0], box[1])
