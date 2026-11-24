"""The carry of an adaptive IC-LoRA job on Auto (Refine Details).

The shortest window follows the carry, so a shorter carry lets a smaller window run. The job
budget prices the window walk once for each carry that Auto may use and keeps the safest price.
Among equal prices it keeps the longest window, then the longest carry. A clean price (a fitted
or locally verified shape) always beats a bet, so Auto never runs a bet while a clean run exists.

No generation runs to choose it. The carry reaches the pipeline through ``PricedJob.carry_frames``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from runtime_config.budget_admission import PricedJob, safety_rank
from runtime_config.ic_lora_chunking import ic_lora_carry_frames
from runtime_config.ic_lora_tiling import IcLoraTiling, auto_carry_candidates

_ONE_PASS = float("inf")


def _window_length(priced: PricedJob) -> float:
    window = priced.decision.chunk_pixel_frames
    return _ONE_PASS if window is None else window


def price_with_auto_carry(tiling: IcLoraTiling, price: Callable[[IcLoraTiling], PricedJob]) -> PricedJob:
    """The best price over the carries Auto may use, tagged with the carry it runs.

    ``price`` prices the job with one tiling, and walks the windows for it.
    """
    best: tuple[tuple[int, float, int], PricedJob] | None = None
    for carry in auto_carry_candidates(ic_lora_carry_frames(tiling), tiling.max_window_frames):
        priced = replace(price(tiling.model_copy(update={"carry_frames": carry})), carry_frames=carry)
        key = (safety_rank(priced), _window_length(priced), carry)
        if best is None or key > best[0]:
            best = (key, priced)
    assert best is not None  # auto_carry_candidates never returns an empty list
    return best[1]


def picked_carry_tiling(tiling: IcLoraTiling | None, priced: PricedJob) -> IcLoraTiling | None:
    """The tiling the pipeline runs: the carry that Auto picked, else the tiling itself."""
    if tiling is None or priced.carry_frames is None:
        return tiling
    return tiling.model_copy(update={"carry_frames": priced.carry_frames, "carry_pinned": True})
