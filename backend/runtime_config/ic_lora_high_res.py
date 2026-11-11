"""IC-LoRA recipes above 1080p: the 1440p and 2160p cells (feature flag ``highResVideoToVideo``,
except for the LoRAs in ``HIGH_RES_WITHOUT_FLAG_IC_LORAS``).

The model runs in tiles, blended inside every denoising step, so the output can be
larger than the local 1080p cell. The clip runs in short windows. Each window
re-renders the tail of the one before it and pins that tail at its start
(``IcLoraTiling.window_frames`` and ``carry_frames``). The carry needs three latent
frames (17 frames) at least. Upstream cross-fades the whole carry, and the frames a middle
window keeps must be more than the carry. So 41 frames is the shortest window that works
with the default carry of 17, and each window adds 24 new frames. A window of 33 fails on a
clip of three windows or more. A run can set a longer carry, and the shortest window then
grows with it (``ic_lora_decode_window``).

Measured on the RTX 5090 at 4K with a 1920x1088 tile (Day to Night and Water Simulation,
233 frames, 9 windows): 50 s per step, about 7.8 minutes per window, 4,226 s for the
clip, peak 15.3 GiB allocated and 22.7 GiB reserved. No run exists at 1440p or on the
Mac. The 4K output has about 1080p detail (one stage, no refinement), so the cells stay
behind the flag. Change the values here, in one place, when a measurement exists.
"""

from __future__ import annotations

from typing import Final, Literal

from api_types import IcLoraSettings
from runtime_config.ic_lora_tiling import HIGH_RES_CARRY_FRAMES, IC_LORA_TILE_OVERLAP, IcLoraTiling
from runtime_config.ltx_capabilities import TWO_PASS_PIXELS_16_9

HighResResolution = Literal["1440p", "2160p"]

# Error code and message of a high cell asked for without the ``highResVideoToVideo`` flag.
HIGH_RES_NOT_ENABLED: Final = "HIGH_RES_NOT_ENABLED"

# (long, short) edge of the stage-1 canvas, on the 64 px grid of the pipeline. One table
# with the Fast cells: the cell is the canvas ``ltx_capabilities`` already gives it.
_CELL_BOX: Final[dict[HighResResolution, tuple[int, int]]] = {
    "1440p": TWO_PASS_PIXELS_16_9["1440p"],
    "2160p": TWO_PASS_PIXELS_16_9["2160p"],
}

_CANVAS_GRID: Final = 64

# Catalog IC-LoRAs that offer the 1440p and 2160p cells without the highResVideoToVideo flag.
# Refine Details is the detail stage of the 4K ladder (its card: "refines and upscales any clip
# to 4K"), so its high cells are the point of the recipe. This list only gates the flag. Which
# cells are measured is measured_cells.py, whatever this says.
HIGH_RES_WITHOUT_FLAG_IC_LORAS: Final = frozenset({"refine-details"})


def high_res_flag_satisfied(catalog_id: str, *, flag_enabled: bool) -> bool:
    """True when the 1440p and 2160p cells are open to this catalog LoRA."""
    return flag_enabled or catalog_id in HIGH_RES_WITHOUT_FLAG_IC_LORAS

# Tile of a LoRA without its own tile plan. A LoRA with one keeps its tile size.
# It is the local spatial cap, the size these LoRAs already run at in one pass. A 4K canvas
# then needs 3x3 tiles. A 1024x576 tile needed 6x6, and every tile streams the whole
# transformer from disk: a 4K Day to Night window took 210 s per step on the RTX 5090.
_DEFAULT_TILE: Final = (1920, 1088)
_DEFAULT_WINDOW_FRAMES: Final = 41


def high_res_tiling(base: IcLoraTiling | None) -> IcLoraTiling:
    """The tile plan of a high cell, for any catalog LoRA.

    The LoRA keeps its own tile size, tile overlap and window when the catalog sets them. An
    adaptive LoRA stays adaptive: the job budget picks its window. A LoRA without a window gets
    the shortest working one. The carry is the high cell default
    (a run can set its own after this), and the spatial cap is raised to the 2160p cell.

    Every tile of every step streams the whole transformer from disk, and that read
    costs the same for a short window as for a long one. So a longer window is cheaper
    per frame, until a pass turns compute-bound.
    """
    long_side, short_side = (
        _DEFAULT_TILE if base is None else (base.long_side, base.short_side)
    )
    adaptive = base is not None and base.adaptive_window
    window_frames = None if adaptive else _DEFAULT_WINDOW_FRAMES
    if base is not None and base.window_frames is not None:
        window_frames = base.window_frames
    return IcLoraTiling(
        long_side=long_side,
        short_side=short_side,
        window_frames=window_frames,
        adaptive_window=adaptive,
        carry_frames=HIGH_RES_CARRY_FRAMES,
        overlap_fraction=IC_LORA_TILE_OVERLAP if base is None else base.overlap_fraction,
        max_output_long_side=_CELL_BOX["2160p"][0],
        max_output_short_side=_CELL_BOX["2160p"][1],
    )


# A cell never upscales, so a clip shorter than the cell (less the 16 px slack of the
# 720p minimum) would run at its own size under a 1440p or 2160p label. The frontend
# copy is MIN_SOURCE_SHORT_EDGE in icLoraResolution.ts and must follow it by hand.
_MIN_SOURCE_SHORT_EDGE: Final[dict[HighResResolution, int]] = {"1440p": 1424, "2160p": 2144}


def high_res_source_large_enough(width: int, height: int, resolution: HighResResolution) -> bool:
    """True when the clip is at least as large as the cell, so the cell is the size it runs at."""
    return min(width, height) >= _MIN_SOURCE_SHORT_EDGE[resolution]


def high_res_settings(settings: IcLoraSettings) -> IcLoraSettings:
    """The settings a high cell runs with: tiled, single stage, whatever the catalog sets."""
    return settings.model_copy(
        update={
            "tiling": high_res_tiling(settings.tiling),
            "skip_stage_2": True,
            "stage_2_ic_lora": False,
        }
    )


def high_res_cell(resolution: str) -> HighResResolution | None:
    """The high-resolution cell for ``resolution``, or None for a local cell."""
    if resolution == "1440p" or resolution == "2160p":
        return resolution
    return None


def high_res_stage1(
    width: int, height: int, resolution: HighResResolution, *, upscale: bool = False
) -> tuple[int, int]:
    """Stage-1 canvas for a clip of this size, on the 64 px grid.

    A clip larger than the cell shrinks into it and keeps its shape. A clip inside the
    cell is not scaled: the canvas is the clip rounded up to the grid, and the border
    is cut off after the run (``cutout_alignment``). ``upscale`` scales a clip inside
    the cell up to it, for a LoRA that refines an upscaled clip (Refine Details: a 1080p
    clip to the 2160p cell is one pass). The cell turns with the clip.
    """
    long_edge, short_edge = _CELL_BOX[resolution]
    box_width, box_height = (
        (long_edge, short_edge) if width >= height else (short_edge, long_edge)
    )
    scale = min(box_width / width, box_height / height)
    if not upscale:
        scale = min(1.0, scale)

    def up_to_grid(edge: float, limit: int) -> int:
        cells = -(-round(edge) // _CANVAS_GRID)
        return min(limit, max(_CANVAS_GRID, cells * _CANVAS_GRID))

    return up_to_grid(width * scale, box_width), up_to_grid(height * scale, box_height)
