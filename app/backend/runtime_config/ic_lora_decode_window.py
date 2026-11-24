"""Decode-safe window of a tiled high-res IC-LoRA job (CUDA).

Every window ends in a DiffVAE decode, and the decode needs a minimum activation budget.
ltx-pipelines refuses the job ("Cannot fit a DiffVAE decode tile") when the budget is
under it, after the denoise has already run. The budget a decode sees also shrinks as
windows pile up: the card holds more from window to window. So a window length that
fits the first window can fail the third, after tens of minutes of denoise.

This module picks the window up front: the longest one whose decode still fits the
budget of the last window of the job, with a margin. It is pure arithmetic, like the
rest of the job budget, so it needs no GPU and no ltx-core.

Measured on the RTX 5090 (31.84 GiB), Refine Details at 3840x2112, 137 frames, 73-frame
windows: the decode budget read 27.8, 22.7 and 20.9 GiB on windows 1 to 3. At 20.9 GiB a
73-frame decode does not fit and a 57-frame one does. The run passed only because its
third window was a 25-frame tail.

``decode_need_gib`` is a fit of the planner's own minimum budget (ltx-core
``recommended_decode_tiling_config``), probed offline on 1080p, 1440p and 2160p canvases
at 25 to 97 frames. It is linear in megapixels and frames up to the 80-frame frame tile,
and the constants sit just above every probe. ``decode_budget_gib`` is one run, so its
reserves are provisional: window 4 and later are unmeasured.

A window over 80 frames splits the frame axis in the decode. At 4K that stalled for 19+
minutes, with the card full and no allocator retries, while 73-frame windows (frame axis
untiled) decoded in about two minutes. So above ``_FRAME_FENCE_MPX`` no window passes 80
frames. 1440p ran 97-frame windows fine and keeps them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from frame_math import snap_to_frame_grid
from runtime_config.fast_chunking import job_window_count

# Probed minimum decode budget: base + per_mpx * Mpx + per_mpx_frame * Mpx * frames.
# 2160p 73f probes at 22.28 GiB (fit 22.54), 2160p 41f at 17.55 (17.77), 1080p 73f at 8.15 (8.25).
_NEED_BASE_GIB: Final = 3.3
_NEED_GIB_PER_MPX: Final = 1.03
_NEED_GIB_PER_MPX_FRAME: Final = 0.0184
# The planner stops growing past its 80-frame frame tile; 97 frames probes like 73.
_NEED_FRAMES_CAP: Final = 73

# Upstream's decode frame tile, and the largest 8k+1 window under it: a window that long or
# shorter decodes with the frame axis untiled.
_FRAME_TILE: Final = 80
DECODE_MAX_UNTILED_WINDOW: Final = 73
# Canvas size above which a window over the frame tile is not allowed. 4K (8.1 Mpx) stalled on a
# 97-frame window; 1440p (3.8 Mpx) did not.
_FRAME_FENCE_MPX: Final = 6.0

_WINDOW_STEP: Final = 8


def is_frame_fenced_canvas(width: int, height: int) -> bool:
    """True when the canvas is past the frame fence, so no window may pass the frame tile."""
    return width * height / 1e6 > _FRAME_FENCE_MPX


def shortest_window(carry_frames: int) -> int:
    """The shortest window that works with this carry (``ic_lora_high_res``).

    A middle window keeps ``window - carry`` frames, and upstream needs them to outnumber the
    carry it cross-fades. A window and a carry are both 8k + 1, so the shortest window is
    ``2 * carry + 7``: 41 for a carry of 17.
    """
    return 2 * carry_frames + _WINDOW_STEP - 1

# Slack kept over the minimum: the budget read 26.5 to 29.7 GiB on window 1 of the same job.
_MARGIN_GIB: Final = 1.0

# VRAM the card holds before a window's decode, by the number of windows in the job,
# scaled linearly by canvas megapixels between the measured 1440p (3.77 Mpx) and 4K
# (8.11 Mpx) anchors and clamped to [0, 1].
# 4K (3840x2112 on 31.84 GiB): total minus decode budget read 4.0 (27.8), 9.1 (22.7),
# 10.9 (20.9) on windows 1 to 3, rounded up to {1: 4.0, 2: 9.2}, >=3: 11.0.
# 1440p (2560x1472 on 16.0 GiB cap, C2/C3, and uncapped N2): post-evict held VRAM read
# [1.0, 2.3, 3.1..3.2] GiB capped and [2.7, 4.0, 4.8] GiB uncapped across windows 1 to 5,
# leaving 11.20 GiB of decode budget on a 16.0 GiB card (admits 41f windows at 11.02 GiB
# and steps 73f/97f windows down to 41f).
_1440P_MPX: Final = 2560 * 1472 / 1e6
_4K_MPX: Final = 3840 * 2112 / 1e6
_RESERVE_1440P_GIB_BY_WINDOWS: Final = {1: 2.0, 2: 4.0}
_RESERVE_1440P_GIB_SATURATED: Final = 4.8
_RESERVE_4K_GIB_BY_WINDOWS: Final = {1: 4.0, 2: 9.2}
_RESERVE_4K_GIB_SATURATED: Final = 11.0


def decode_need_gib(width: int, height: int, frames: int) -> float:
    """Minimum decode budget of a window, in GiB, for the canvas the decoder sees."""
    mpx = width * height / 1e6
    return _NEED_BASE_GIB + mpx * (
        _NEED_GIB_PER_MPX + _NEED_GIB_PER_MPX_FRAME * min(frames, _NEED_FRAMES_CAP)
    )


def decode_budget_gib(
    total_vram_gib: float, windows: int, *, width: int = 3840, height: int = 2112
) -> float:
    """Decode budget of the last window of a job of ``windows`` windows."""
    r1440 = _RESERVE_1440P_GIB_BY_WINDOWS.get(windows, _RESERVE_1440P_GIB_SATURATED)
    r4k = _RESERVE_4K_GIB_BY_WINDOWS.get(windows, _RESERVE_4K_GIB_SATURATED)
    mpx = width * height / 1e6
    t = min(1.0, max(0.0, (mpx - _1440P_MPX) / (_4K_MPX - _1440P_MPX)))
    reserve = r1440 + t * (r4k - r1440)
    return max(0.0, total_vram_gib - reserve)


@dataclass(frozen=True, slots=True)
class DecodeWindowChoice:
    """``window`` is the pixel-frame window to run (None for one unchunked pass).

    ``fits`` is False when no window passes: the job refuses before denoise.
    """

    fits: bool
    window: int | None


def _window_fits(
    *, frames: int, window: int | None, width: int, height: int, total_vram_gib: float, carry_frames: int
) -> bool:
    decoded = frames if window is None else window
    if is_frame_fenced_canvas(width, height) and decoded > _FRAME_TILE:
        return False
    windows = job_window_count(frames, window, carry_frames)
    return decode_need_gib(width, height, decoded) + _MARGIN_GIB <= decode_budget_gib(
        total_vram_gib, windows, width=width, height=height
    )


def choose_decode_window(
    *,
    frames: int,
    window: int | None,
    width: int,
    height: int,
    total_vram_gib: float,
    carry_frames: int,
) -> DecodeWindowChoice:
    """The longest window, at most the LoRA's own, whose decode fits the job's last window.

    ``window`` is the window the chunk rule chose (None: the clip runs in one pass). An
    unchunked clip that does not fit drops to windows. Past the frame fence a window is
    capped at ``DECODE_MAX_UNTILED_WINDOW``. A LoRA window under the shortest working one
    is only tried as given. A cap that takes the window under the shortest working one for
    the carry leaves no candidate, so the job refuses before denoise.
    """
    clip = snap_to_frame_grid(frames, floor=1)
    candidates: list[int | None] = []
    if window is None:
        candidates.append(None)
        top = DECODE_MAX_UNTILED_WINDOW
    else:
        top = window
    if is_frame_fenced_canvas(width, height):
        top = min(top, DECODE_MAX_UNTILED_WINDOW)
    shortest = shortest_window(carry_frames)
    if top < shortest:
        if top == window:
            candidates.append(top)
    else:
        candidates.extend(range(top, shortest - 1, -_WINDOW_STEP))
    for candidate in candidates:
        if candidate is not None and candidate >= clip:
            continue
        if _window_fits(
            frames=clip,
            window=candidate,
            width=width,
            height=height,
            total_vram_gib=total_vram_gib,
            carry_frames=carry_frames,
        ):
            return DecodeWindowChoice(fits=True, window=candidate)
    return DecodeWindowChoice(fits=False, window=window)
