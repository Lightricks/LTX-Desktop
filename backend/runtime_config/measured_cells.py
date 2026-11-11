"""Which 1440p and 4K cells have a measured run behind them.

The budget laws are fitted at and below 1080p. A 1440p or 4K cell that no run has
confirmed prices off an extrapolation, so it is an unmeasured bet: Cautious mode
refuses it and Brave mode allows it with an unmeasured warning. This module is the
only place that says which cells are measured. Adding a row here is how a cell
opens for everyone.
Imports nothing internal by design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, Never

if TYPE_CHECKING:
    from api_types import LTXLocalModelId

BudgetFeature = Literal["fast", "edit", "iclora"]
BudgetRoute = Literal["mac_measured", "windows_measured", "legacy"]
HighResTier = Literal["1440p", "2160p"]

# Short-side floors that separate the tiers, so the tier follows the resolution name
# at every aspect (a 21:9 1080p canvas is 2560 wide but 1088 high). The canvas short
# edges are 1088 (1080p), 1472 (1440p) and 2176 (2160p). Each floor is the grid step
# below the tier's edge, so only a canvas that is that tier's size or larger counts.
_TIER_MIN_SHORT_SIDE: Final[tuple[tuple[HighResTier, int], ...]] = (
    ("2160p", 2112),
    ("1440p", 1408),
)

# (feature, route, tier) -> the longest job, in windows, that a measured run covers.
# The Windows Fast windowed peak holds flat from the third window and the IC-LoRA tiled peak
# from the fourth, so a row is the longest run measured, not a growth boundary: past it the
# wall-clock and the allocator are untested rather than known to climb. The registry counts
# windows and not their size, so a row cannot say "20 windows at 41 frames but 5 at 129".
MEASURED_HIGH_RES_CELLS: Final[dict[tuple[BudgetFeature, BudgetRoute, HighResTier], int]] = {
    # Mac Fast 4K: capped runs bracket the allocator edge at 2 and 7 windows, and the 7-window
    # one ran to completion, so a longer job is unmeasured. The window count stays at the
    # longest completed run: the Mac peak is flat past the third window, but a 15-window job
    # takes about 6 hours, past any run the heartbeat has covered.
    ("fast", "mac_measured", "2160p"): 7,
    # Mac Fast 1440p: capped runs bracket the allocator edge at 1, 2 and 5 windows of 129 frames, and the
    # 5-window job ran to completion, so a longer job is unmeasured. The Mac peak is flat from the third
    # window; a 9-window job takes about 4 hours, past any run the heartbeat has covered. Shorter windows
    # (smaller pools, and the 9:16 and 21:9 canvases, whose longest window is shorter) carry no floor: the
    # windowed term is priced as a share of the window's tokens, which only the Windows runs show. If that
    # is wrong, the job runs out of memory in the second or third window, and since the frontier records
    # clean runs only, Cautious keeps offering it.
    ("fast", "mac_measured", "1440p"): 5,
    # Windows Fast 1440p: the fence is the 129-frame windows, which a 32 GB card runs and
    # which are measured flat from the third window through nine (20 s at 48 fps, 961 frames).
    # Small cards fall back to 41-frame windows, measured flat to twenty, but a window count
    # cannot tell the two sizes apart, so jobs past nine windows stay unmeasured and refuse in
    # Cautious mode. Raising this to twenty would open 129-frame jobs of ten to twenty windows
    # that no run covers. Non-16:9 aspects (9:16, 1:1, 21:9) sit inside the same token law.
    ("fast", "windows_measured", "1440p"): 9,
    # Windows Fast 4K: held flat from the third window through fifteen (a 50 fps job, T2V and A2V).
    # Non-16:9 aspects (9:16, 1:1) sit inside the same token law (9:16 matches 16:9 per-window to 0.00 GiB).
    ("fast", "windows_measured", "2160p"): 15,
    # Windows Edit 1440p: Retake and Extend never chunk (1 window) and sit inside the 1080p
    # reserved law through 193 frames (90,000 tokens, 29.50 GiB vs 29.93 ask; Retake == Extend).
    ("edit", "windows_measured", "1440p"): 1,
    # Windows Edit 4K: untiled-in-time vae_decode (3.15 + 3.222e-4 * t_decode) is priced in
    # _EDIT_DECODE_STREAM and verified clean through 73,440 single-tile decode tokens (65 frames
    # at 3840x2176, 26.83 GiB reserved, shared_peak 0.08 GB); decode tiles past 73,440 tokens
    # stay unmeasured via MEASURED_MAX_DECODE_TOKENS.
    ("edit", "windows_measured", "2160p"): 1,
    # Windows IC-LoRA 1440p: the peak grows from the second to the third window and holds flat
    # through the fifth (2x2 tiles of 1920x1088, 41-frame windows with carry 17).
    ("iclora", "windows_measured", "1440p"): 5,
    # Windows IC-LoRA 4K: the peak grows through the fourth window and holds to the tenth.
    ("iclora", "windows_measured", "2160p"): 10,
    # Not measured, so absent: Mac IC-LoRA (the per-Mpx canvas term is a Windows number),
    # Mac edits above 1080p (past the fitted token range), Windows 4K edits past the 73,440-token
    # single-tile decode ceiling (see MEASURED_MAX_DECODE_TOKENS below), and any job past the window
    # counts above.
}


# (feature, route) -> the largest single-decode-tile token count that a measured run covers
# cleanly without WDDM paging. Upstream DiffVAE splits the frame axis past 80 frames (largest
# 8k+1 untiled window: 73 frames), so a single temporal decode tile processes at most 73 frames.
# Any decode tile with more tokens than this ceiling is an unmeasured bet.
MEASURED_MAX_DECODE_TOKENS: Final[dict[tuple[BudgetFeature, BudgetRoute], float]] = {
    # Windows Edit: 4K 49f (57,120 decode tokens, 21.57 GiB) and 4K 65f (73,440 decode tokens,
    # 26.83 GiB) ran with zero WDDM paging (shared_peak 0.08 GB), and 1440p 193f (36,000 decode
    # tokens per 73f tile) ran clean at 29.50 GiB; 4K 73f (81,600 decode tokens, 29.45 GiB) paged.
    ("edit", "windows_measured"): 73_440.0,
}


@dataclass(frozen=True, slots=True)
class WindowedCover:
    """What a measured windowed run covers on a canvas below 1440p."""

    max_windows: int
    # Shortest window, in tokens, the row covers. Zero when the route prices every window size on the
    # row's evidence, which both rows do today.
    min_window_tokens: int


# (feature, route) -> the longest job in windows that a measured run covers, and the smallest window
# the row covers, on a canvas below 1440p that runs in windows (``fast_chunking``). A window is held to
# the token size the budget laws were fitted at and the peak is flat from the third window, so the
# ask is the window's whatever the canvas or the clip length: a long 540p clip and a short 1080p
# one are the same window on the same pipeline. The rows follow the 1440p and 4K runs above, which
# are route-wide and so cover every card on the route. No row names a card, because the window
# the picker takes is already the longest the card's own budget loads. A job past the row, or on a
# route with no row, is an unmeasured bet: the budget still runs it in windows, and Cautious mode
# refuses it while Brave mode allows it with a warning. A clean Brave run records the job in the
# card's calibration frontier and opens it in Cautious mode.
MEASURED_WINDOWED_BELOW_1440P_CELLS: Final[dict[tuple[BudgetFeature, BudgetRoute], WindowedCover]] = {
    # Windows Fast: 129-frame windows (62.5k tokens) held flat from the third window through nine,
    # and 41-frame windows (22k tokens) flat to twenty, on the RTX 5090 at 1440p. The 1080p 20 s run
    # (3 windows of 241 frames, 63k tokens) confirms the same windows below 1440p. The count stays
    # at nine because a window count cannot tell the two sizes apart.
    ("fast", "windows_measured"): WindowedCover(max_windows=9, min_window_tokens=0),
    # Mac Fast: the 1440p runs are 129-frame windows flat through five. No floor on the window size, for
    # the reason and with the failure mode given on the Mac Fast 1440p row above.
    ("fast", "mac_measured"): WindowedCover(max_windows=5, min_window_tokens=0),
}


# Catalog IC-LoRA adapters covered by the standard 1440p and 4K tiled rows
# (``MEASURED_HIGH_RES_CELLS[("iclora", ...)]``, measured on Water Simulation and shared by
# architecture-equivalent single-stage tiled adapters) or by ``MEASURED_IC_LORA_COMPOSITION_CELLS``.
# A newly added catalog IC-LoRA not yet listed here defaults to unmeasured at 1440p and 4K until
# its tile/window plan is verified against the shared law or given its own composition row.
HIGH_RES_COVERED_IC_LORA_IDS: Final[frozenset[str]] = frozenset(
    {
        "clean-plate",
        "3d-render-to-real",
        "day-to-night",
        "colorization",
        "decompression",
        "deblur",
        "cross-eyed",
        "water-simulation",
        "ingredients",
        "instant-shave",
        "outpainting",
        "greenscreen-avatar",
        "upscale",
        "crossview-prompt",
        "alpha-gen",
        "layout-to-render",
        "motion-track",
        "video-guide",
        "restore",
        "refine-details",
    }
)


# IC-LoRAs that run their own tile and window plan. A composition in
# ``UNMEASURED_IC_LORA_COMPOSITIONS`` is unmeasured in every high cell unless
# ``MEASURED_IC_LORA_COMPOSITION_CELLS`` has a measured window count for that
# (ic_lora_id, route, tier).
UNMEASURED_IC_LORA_COMPOSITIONS: Final = frozenset({"refine-details"})
MEASURED_IC_LORA_COMPOSITION_CELLS: Final[dict[tuple[str, BudgetRoute, HighResTier], int]] = {
    ("refine-details", "windows_measured", "1440p"): 4,
}


def _unmeasured_model_fallback(_: Never) -> bool:
    return True


def is_unmeasured_model(model_id: LTXLocalModelId | None) -> bool:
    """True for a local model id that has not been measured yet.

    Exhaustive over LTXLocalModelId at typecheck time so adding a new model id
    fails pyright until it is explicitly classified here.
    """
    if model_id is None:
        return False
    match model_id:
        case (
            "ltx-2.5-22b-distilled"
            | "ltx-2.3-22b-distilled"
            | "ltx-2.3-22b-distilled-1.1"
        ):
            return False
        case _ as unhandled:
            return _unmeasured_model_fallback(unhandled)


def is_unmeasured_ic_lora(ic_lora_id: str | None) -> bool:
    """True for an IC-LoRA adapter id not yet covered by the 1440p/4K law."""
    return ic_lora_id is not None and ic_lora_id not in HIGH_RES_COVERED_IC_LORA_IDS


def high_res_tier(width: int, height: int) -> HighResTier | None:
    """The tier of a canvas above 1080p, or None at and below it."""
    short_side = min(width, height)
    for tier, minimum in _TIER_MIN_SHORT_SIDE:
        if short_side >= minimum:
            return tier
    return None


def is_unmeasured_high_res(
    feature: BudgetFeature,
    route: BudgetRoute,
    width: int,
    height: int,
    *,
    windows: int,
    decode_tokens: float | None = None,
    ic_lora_id: str | None = None,
) -> bool:
    """True for a 1440p or 4K job that no measured run covers.

    A job is covered when its (feature, route, tier) has a row, the job runs in no
    more windows than that row's run did, and (when a route has a decode-tile token
    ceiling) its single-tile decode tokens do not exceed that ceiling. An IC-LoRA in
    ``UNMEASURED_IC_LORA_COMPOSITIONS`` or not yet in ``HIGH_RES_COVERED_IC_LORA_IDS``
    is unmeasured unless its own row in ``MEASURED_IC_LORA_COMPOSITION_CELLS`` covers
    ``windows``.
    """
    tier = high_res_tier(width, height)
    if tier is None:
        return False
    if ic_lora_id is not None and (
        ic_lora_id in UNMEASURED_IC_LORA_COMPOSITIONS or is_unmeasured_ic_lora(ic_lora_id)
    ):
        composition_windows = MEASURED_IC_LORA_COMPOSITION_CELLS.get(
            (ic_lora_id, route, tier)
        )
        if composition_windows is None or windows > composition_windows:
            return True
    key = (feature, route, tier)
    measured_windows = MEASURED_HIGH_RES_CELLS.get(key)
    if measured_windows is None or windows > measured_windows:
        return True
    max_decode_tokens = MEASURED_MAX_DECODE_TOKENS.get((feature, route))
    if max_decode_tokens is not None and decode_tokens is not None:
        return decode_tokens > max_decode_tokens
    return False


def is_unmeasured_windowed_below_1440p(
    feature: BudgetFeature,
    route: BudgetRoute,
    width: int,
    height: int,
    *,
    windows: int,
    window_tokens: float,
) -> bool:
    """True for a job below 1440p in windows that no measured run covers.

    One pass is never unmeasured here: it prices on rows fitted at and below 1080p. A job in
    more than one window is covered only when its (feature, route) has a row, the job runs in
    no more windows than that row's run did, and its window is at least the row's floor.
    A route with no row is an unmeasured bet. 1440p and 4K have their own rows
    (``is_unmeasured_high_res``).
    """
    if windows <= 1 or high_res_tier(width, height) is not None:
        return False
    cover = MEASURED_WINDOWED_BELOW_1440P_CELLS.get((feature, route))
    if cover is None:
        return True
    return windows > cover.max_windows or window_tokens < cover.min_window_tokens
