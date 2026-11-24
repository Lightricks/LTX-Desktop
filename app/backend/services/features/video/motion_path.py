"""The Motion Track control clip: one dot for each path on black, each moving along its path.

``positions`` is the Python twin of ``frontend/ltx-io/lib/motionPathSampler.ts``. The editor previews
the dot with that code and the job draws it with this code. ``shared/motion-path-golden.json``
pins both to the same frames, so the preview and the clip cannot drift apart.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from api_motion_path import MotionPath, MotionPathEasing, MotionPathPoint, MotionPathVec2
from frame_math import compute_num_frames
from services.retake_pipeline.rgb_stitch import write_rgb_video

_C1 = 1.70158
_C2 = _C1 * 1.525
_C3 = _C1 + 1
# The dot color of each path, read from the file the editor reads, so the preview and the clip
# have the same colors. The first is the blue of the ltx.io control clip. The dot is 1% of the
# short side, and never under 4 px.
_SHARED = json.loads(
    (Path(__file__).resolve().parents[4] / "shared" / "motion-path.json").read_text(encoding="utf-8")
)
_DOT_RGBS: list[tuple[int, int, int]] = [
    (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))
    for color in _SHARED["dotColors"]
]
_DOT_MIN_RADIUS_PX = 4
_DOT_RADIUS_SHORT_SIDE = 0.01


def _round(value: float) -> int:
    """Round half up, as the editor does. Python's ``round`` goes to the even neighbour."""
    return math.floor(value + 0.5)


def _ease_out_bounce(x: float) -> float:
    n1, d1 = 7.5625, 2.75
    if x < 1 / d1:
        return n1 * x * x
    if x < 2 / d1:
        x -= 1.5 / d1
        return n1 * x * x + 0.75
    if x < 2.5 / d1:
        x -= 2.25 / d1
        return n1 * x * x + 0.9375
    x -= 2.625 / d1
    return n1 * x * x + 0.984375


def _ease_in_out_back(x: float) -> float:
    if x < 0.5:
        return ((2 * x) ** 2 * ((_C2 + 1) * 2 * x - _C2)) / 2
    return ((2 * x - 2) ** 2 * ((_C2 + 1) * (x * 2 - 2) + _C2) + 2) / 2


# Penner easings from easings.net, over progress in [0, 1]. Back and bounce leave [0, 1].
_EASINGS: dict[MotionPathEasing, Callable[[float], float]] = {
    "linear": lambda x: x,
    "easeIn": lambda x: x**3,
    "easeOut": lambda x: 1 - (1 - x) ** 3,
    "easeInOut": lambda x: 4 * x**3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2,
    "easeInBack": lambda x: _C3 * x**3 - _C1 * x**2,
    "easeOutBack": lambda x: 1 + _C3 * (x - 1) ** 3 + _C1 * (x - 1) ** 2,
    "easeInOutBack": _ease_in_out_back,
    "gentle": lambda x: math.sin(x * math.pi / 2),
    "quick": lambda x: 1.0 if x == 1 else 1 - 2 ** (-10 * x),
    "bouncy": _ease_out_bounce,
    "slow": lambda x: -(math.cos(math.pi * x) - 1) / 2,
}


def _control(point: MotionPathPoint, handle: MotionPathVec2 | None) -> tuple[float, float]:
    return (point.x, point.y) if handle is None else (point.x + handle.x, point.y + handle.y)


def _bezier(a: MotionPathPoint, b: MotionPathPoint, t: float) -> tuple[float, float]:
    """The cubic from ``a`` to ``b``. The handles are offsets from their anchors."""
    (c1x, c1y), (c2x, c2y) = _control(a, a.handleOut), _control(b, b.handleIn)
    u = 1 - t
    return (
        u**3 * a.x + 3 * u * u * t * c1x + 3 * u * t * t * c2x + t**3 * b.x,
        u**3 * a.y + 3 * u * u * t * c1y + 3 * u * t * t * c2y + t**3 * b.y,
    )


def _position(points: list[MotionPathPoint], progress: float) -> tuple[float, float]:
    """The dot at ``progress`` along the path, each segment taking the same time.

    Back and bounce easings pass the ends. The dot then follows the end chord, because a
    cubic does not extrapolate for ``t`` outside [0, 1].
    """
    segments = len(points) - 1
    if progress <= 0:
        a, b = points[0], points[1]
        return a.x + (a.x - b.x) * -progress, a.y + (a.y - b.y) * -progress
    if progress >= 1:
        a, b = points[-1], points[-2]
        return a.x + (a.x - b.x) * (progress - 1), a.y + (a.y - b.y) * (progress - 1)
    scaled = progress * segments
    index = min(segments - 1, math.floor(scaled))
    return _bezier(points[index], points[index + 1], scaled - index)


def positions(path: MotionPath, *, duration: float, frames: int) -> list[tuple[float, float]]:
    """The dot of every frame in frame-normalized coordinates. Create keeps the holds under ``duration``."""
    last = frames - 1
    move_start = min(last, _round(path.startHoldSec / duration * last))
    move_end = last - min(last - move_start, _round(path.endHoldSec / duration * last))
    ease = _EASINGS[path.easing]
    result: list[tuple[float, float]] = []
    for index in range(frames):
        if index <= move_start:
            progress = 0.0
        elif index >= move_end:
            progress = 1.0
        else:
            progress = (index - move_start) / (move_end - move_start)
        result.append(_position(path.points, ease(progress)))
    return result


def _stamp_dot(
    frame: NDArray[np.uint8], x: float, y: float, radius: float, rgb: tuple[int, int, int]
) -> None:
    """Paint a dot at pixel ``(x, y)`` over ``frame``. A dot past the frame edge is cut off."""
    height, width = int(frame.shape[0]), int(frame.shape[1])
    x0, x1 = max(0, math.floor(x - radius - 1)), min(width, math.ceil(x + radius + 1))
    y0, y1 = max(0, math.floor(y - radius - 1)), min(height, math.ceil(y + radius + 1))
    if x0 >= x1 or y0 >= y1:
        return
    dx = np.arange(x0, x1) + 0.5 - x
    dy = (np.arange(y0, y1) + 0.5 - y)[:, None]
    # One pixel of falloff at the rim anti-aliases the edge.
    coverage = np.clip(radius + 0.5 - np.hypot(dx, dy), 0.0, 1.0)[..., None]
    region = frame[y0:y1, x0:x1]
    region[:] = (region * (1 - coverage) + coverage * np.array(rgb)).round()


def _track_frame(
    dots: Sequence[tuple[float, float]], *, width: int, height: int, radius: float
) -> NDArray[np.uint8]:
    """One frame of the clip: the dot of each path, in the color of its path.

    Where two dots overlap, the later path is on top.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for number, (x, y) in enumerate(dots):
        _stamp_dot(frame, x * width, y * height, radius, _DOT_RGBS[number % len(_DOT_RGBS)])
    return frame


def render_motion_track(
    paths: Sequence[MotionPath], *, width: int, height: int, duration: int, fps: int, dest: Path
) -> None:
    """Write the control clip: ``compute_num_frames`` frames at ``fps``, one dot per path over black."""
    radius = max(_DOT_MIN_RADIUS_PX, _round(min(width, height) * _DOT_RADIUS_SHORT_SIDE))
    frame_count = compute_num_frames(duration, fps)
    tracks = [positions(path, duration=duration, frames=frame_count) for path in paths]
    frames: Iterator[NDArray[np.uint8]] = (
        _track_frame(dots, width=width, height=height, radius=radius)
        for dots in zip(*tracks, strict=True)
    )
    write_rgb_video(dest, frames, fps=fps, width=width, height=height)
