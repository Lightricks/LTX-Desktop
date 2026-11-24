"""Chunk windows an adaptive LoRA (Refine Details) can run, longest first.

A fixed-window LoRA (Restore) chunks at its window. An adaptive one has no fixed window. The
job budget prices the candidates here from the longest to the shortest, and keeps the safest one
(the longest of equal safety). This is arithmetic done before the job starts. No generation runs
to choose a window.

The resolution caps the longest window:

- Above the frame fence (4K) a window over ``DECODE_MAX_UNTILED_WINDOW`` splits the decode frame
  axis, and a 97-frame window stalled the decode for more than 19 minutes.
- Above 720p (1080p and 1440p) the window stays at ``ADAPTIVE_MAX_WINDOW_FRAMES``, the length that
  the Restore card documents as the drift limit.
- At 720p and below there is no cap, so a clip runs in one pass when the budget admits it, and
  a window longer than 97 frames when it does not.

No imports from the job budget, so the budget can import this module.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal

from frame_math import snap_to_frame_grid
from runtime_config.ic_lora_decode_window import (
    DECODE_MAX_UNTILED_WINDOW,
    is_frame_fenced_canvas,
    shortest_window,
)
from runtime_config.ic_lora_tiling import ADAPTIVE_MAX_WINDOW_FRAMES, IcLoraTiling

# Canvas pixels of 720p (1280x704). Above it a window stays at ADAPTIVE_MAX_WINDOW_FRAMES, and a
# tiled job cannot run unchunked past 97 frames (the decode floor).
HIGH_RES_CANVAS_PIXELS: Final = 1280 * 704

_WINDOW_STEP: Final = 8


def resolution_window_cap(width: int, height: int) -> int | None:
    """The longest window for this canvas, or None when the resolution sets no cap."""
    if is_frame_fenced_canvas(width, height):
        return DECODE_MAX_UNTILED_WINDOW
    if width * height > HIGH_RES_CANVAS_PIXELS:
        return ADAPTIVE_MAX_WINDOW_FRAMES
    return None


def adaptive_window_candidates(
    frames: int, *, cap: int | None, user_max: int | None, carry: int
) -> list[int | None]:
    """The windows to price, longest first. None is one pass over the whole clip.

    The ceiling is the lower of the resolution cap and the user maximum. One pass is a candidate
    when the clip fits inside the ceiling. The other candidates are the 8k + 1 windows from the
    ceiling down to the shortest window that works with the carry. The list is empty when the
    ceiling is under that shortest window and the clip is longer than the ceiling.
    """
    clip = snap_to_frame_grid(frames, floor=1)
    limits = [limit for limit in (cap, user_max) if limit is not None]
    ceiling = min(limits) if limits else None
    candidates: list[int | None] = []
    if ceiling is None or clip <= ceiling:
        candidates.append(None)
    top = clip - _WINDOW_STEP if ceiling is None else min(ceiling, clip - _WINDOW_STEP)
    candidates.extend(range(top, shortest_window(carry) - 1, -_WINDOW_STEP))
    return candidates


# Why no window can run: the maximum of the run is above the cap of the resolution, or the cap is
# under the shortest window that works with the carry.
AdaptiveRefusal = Literal["window_over_cap", "refine_decode_floor"]


@dataclass(frozen=True)
class AdaptiveWindow:
    """The window an adaptive job runs (None is one pass). ``refusal`` is set when none can run."""

    window: int | None
    refusal: AdaptiveRefusal | None = None


def choose_adaptive_window(
    frames: int,
    *,
    tiling: IcLoraTiling,
    cap: int | None,
    carry: int,
    pick: Callable[[list[int | None]], int | None],
) -> AdaptiveWindow:
    """The candidate that ``pick`` takes, or the refusal when the cap leaves no candidate.

    ``pick`` gets the candidates longest first and returns one of them. The job budget keeps the
    safest price and, among equals, the longest window.

    A maximum above the cap is refused and never run shorter, so the choice stays the user's.
    """
    user_max = tiling.max_window_frames
    if user_max is not None and cap is not None and user_max > cap:
        return AdaptiveWindow(user_max, "window_over_cap")
    candidates = adaptive_window_candidates(frames, cap=cap, user_max=user_max, carry=carry)
    if not candidates:
        return AdaptiveWindow(cap, "refine_decode_floor")
    return AdaptiveWindow(pick(candidates))
