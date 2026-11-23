"""Temporal windows for large Fast video jobs (1440p and 4K, and any canvas one pass cannot fit).

The Fast budget laws were fitted up to a window of about 63k tokens, on the RTX 5090 (Windows)
and the M5 Max (Mac). A 1440p clip of 10 s is 114k tokens and a 4K clip of 10 s is 253k, so a
large canvas runs in overlapping windows (ltx-pipelines ``ChunkConfig``) and the job budget
prices one window, not the clip.
This module is the one rule for the window. The job budget (``decide_video_job``) picks the
largest window that loads and hands its length to the pipeline.

It imports nothing from the budget, so the budget can import it.
"""

from __future__ import annotations

import math
from typing import Final

# Short side in pixels from which a clip longer than one window runs in windows, whatever the
# card can hold (1440p is 1472 px, 1080p is 1088 px). A clip that fits one window is a single
# pass. Matches the 1440p tier floor in ``measured_cells`` so narrower 1440p ratios (1:1
# 1472x1472, 4:5 1472x1856, 4:3 1984x1472) take the chunked budget law instead of the
# single-pass <=1080p curve. A smaller canvas stays on the single-pass rows, and runs in windows
# only when one pass cannot fit the card (``decide_video_job``).
CHUNKED_CANVAS_MIN_SHORT_SIDE: Final = 1408

# The window may not carry more than this many tokens: it is the largest point the budget laws
# were fitted on, and the fit is the RTX 5090's (1080p, 10 s, 24 fps). It is a ceiling for
# cards with room to spare, not a limit on any card: a card the budget cannot give this much is
# held lower by its own budget law, which picks the longest window it loads. Above it there is
# no fit to price a window from, so a card can only be proven there by a clean run of its own.
MAX_WINDOW_TOKENS: Final = 63_240

# ltx-pipelines rejects a carry below three latent frames. The window must also be more than
# twice the carry, and both must sit on the 8k + 1 grid.
CARRY_FRAMES: Final = 17
MIN_WINDOW_FRAMES: Final = 41
_FRAMES_PER_LATENT: Final = 8

# Windows from this count on hold their peak: the allocator keeps what the carry and the
# earlier windows left behind. A shorter run peaks lower.
PEAK_PLATEAU_WINDOWS: Final = 3


def is_chunked_canvas(width: int, height: int) -> bool:
    return min(width, height) >= CHUNKED_CANVAS_MIN_SHORT_SIDE


def _tokens_per_latent_frame(width: int, height: int) -> float:
    return (height / 32.0) * (width / 32.0)


def window_candidates(width: int, height: int) -> list[int]:
    """Window lengths in pixel frames, longest first, one latent frame apart.

    The longest is the largest that stays inside ``MAX_WINDOW_TOKENS``. A machine that cannot
    load it takes the next one down, to ``MIN_WINDOW_FRAMES``.
    """
    latent_frames = int(MAX_WINDOW_TOKENS // _tokens_per_latent_frame(width, height))
    longest = max(MIN_WINDOW_FRAMES, (latent_frames - 1) * _FRAMES_PER_LATENT + 1)
    return list(range(longest, MIN_WINDOW_FRAMES - 1, -_FRAMES_PER_LATENT))


def window_count(frames: int, window_frames: int, carry_frames: int = CARRY_FRAMES) -> int:
    """How many windows a clip of ``frames`` runs in. Each later window adds ``window - carry``."""
    if frames <= window_frames:
        return 1
    return 1 + math.ceil((frames - window_frames) / (window_frames - carry_frames))


def last_window_frames(
    frames: int, window_frames: int, carry_frames: int = CARRY_FRAMES
) -> int:
    """Pixel frames of the last window (``uniform_chunk_layouts``), on the 8k + 1 grid.

    A single-pass clip returns ``frames``. When ``frames - window_frames`` is an exact
    multiple of the stride ``window_frames - carry_frames``, the last window is a full
    ``window_frames``; otherwise it is ``carry_frames + remainder``.
    """
    if frames <= window_frames:
        return frames
    stride = window_frames - carry_frames
    remainder = (frames - window_frames) % stride
    return window_frames if remainder == 0 else carry_frames + remainder


def job_window_count(
    frames: int, chunk_pixel_frames: int | None, carry_frames: int = CARRY_FRAMES
) -> int:
    """Windows of a decided job: one pass when no window was chosen.

    ``carry_frames`` is the carry the pipeline runs with. Fast's is ``CARRY_FRAMES``. An
    IC-LoRA passes its own (``ic_lora_carry_frames``), or its jobs are miscounted.
    """
    if chunk_pixel_frames is None:
        return 1
    return window_count(frames, chunk_pixel_frames, carry_frames)
