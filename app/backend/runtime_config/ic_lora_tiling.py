"""Spatial transformer tiling of a catalog IC-LoRA (Restore, Refine Details).

These LoRAs trained on one tile size. Their cards say to run every output size in
overlapping tiles of that size, merged inside each denoising step, even at full HD. A
single untiled pass is outside the training bucket. ltx-pipelines runs this as
``ICLoraStageConfig.tiling`` (``FixedSizeSpatialTiling``, overlap 0.5 at most).

Tiling is a property of the LoRA, so it lives in the catalog ``default_settings``. The job
budget reads the tile size as the size of the transformer pass. The spatial cap still
reads the output size.

No imports from other runtime modules, so ``api_types`` and the budget can import it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# The limits and the defaults of the tile overlap and the window carry. The advanced controls
# of the Home form read the same file (shared/ic-lora-tiling-controls.json).
_CONTROLS_PATH: Final = Path(__file__).resolve().parents[2] / "shared" / "ic-lora-tiling-controls.json"
_CONTROLS: Final = json.loads(_CONTROLS_PATH.read_text(encoding="utf-8"))

# A run can replace the tile of the catalog entry with one of these. No size keeps the catalog
# tile. The budget prices the tile that runs.
IcLoraTileSize = Literal["large", "extra-large"]
TILE_SIZES: Final[dict[str, tuple[int, int]]] = {
    name: (size["long"], size["short"]) for name, size in _CONTROLS["tileSizes"].items()
}

MIN_TILE_OVERLAP: Final[float] = _CONTROLS["overlap"]["min"]
MAX_TILE_OVERLAP: Final[float] = _CONTROLS["overlap"]["max"]

# The overlap fraction we request when a run sets none. It is the most upstream accepts, and the
# best for quality. A run can lower it down to ``MIN_TILE_OVERLAP`` for speed, and the cost is
# a risk of visible seams. It is a request, not the layout: upstream ``split_by_size_pinned``
# still caps the tile count so that no cell is covered by three tiles. On a 1920x1088 canvas
# Restore's 960x544 tiles lay out 3x2, Refine's 1024x576 tiles 2x2. The count does not bind
# the peak: tiles run sequentially, so one tile plus the canvas decode is the whole ask —
# six tiles peak at tile + 1.28, not at six tiles.
#
# At 0 the tiles still overlap, because the layout needs the tiles to cover the frame.
IC_LORA_TILE_OVERLAP: Final[float] = _CONTROLS["overlap"]["default"]

_TILE_GRID = 32  # pixel multiple of the transformer latent grid
_FRAME_GRID = 8  # pixel frames per latent frame; windows are 8k + 1 frames
# ltx-pipelines refuses a window carry shorter than three latent frames (17 pixel frames).
MIN_CARRY_FRAMES: Final[int] = _CONTROLS["carryFrames"]["min"]
# The shortest window a run can set: more than twice the shortest carry (see ``IcLoraTiling``).
MIN_WINDOW_FRAMES: Final[int] = 2 * MIN_CARRY_FRAMES + _FRAME_GRID - 1
# ltx-pipelines default carry, kept by a tiling that sets a window and no carry.
DEFAULT_CARRY_FRAMES: Final[int] = _CONTROLS["carryFrames"]["default"]
# The carry of a high cell (1440p, 2160p) when a run sets none. The shortest working window
# follows the carry, so a short carry keeps the windows short (``ic_lora_high_res``).
HIGH_RES_CARRY_FRAMES: Final[int] = _CONTROLS["carryFrames"]["highResDefault"]
# The carries a run can pick in the form. Auto searches the ones at or below its carry.
CARRY_PRESETS: Final[tuple[int, ...]] = tuple(_CONTROLS["carryFrames"]["presets"])
# The longest maximum window a run can set on an adaptive LoRA. Restore documents that detail is
# lost and colour drifts past 97 frames.
ADAPTIVE_MAX_WINDOW_FRAMES: Final[int] = _CONTROLS["windowFrames"]["max"]


class IcLoraTiling(BaseModel):
    """Tile size of a catalog IC-LoRA, and an optional fixed temporal window.

    ``long_side`` and ``short_side`` are given for a landscape canvas. A portrait canvas
    swaps them (Refine Details card: "1024x576, or 576x1024 portrait").

    ``window_frames`` is set when the LoRA drifts over a longer clip (Restore: past 97
    frames detail is lost and colour drifts). A clip longer than the window always runs
    in windows of this length. None leaves the choice to the job budget.

    ``adaptive_window`` leaves the window to the job budget for a LoRA that has no measured
    drift length (Refine Details). The budget picks the longest window that fits, up to the cap
    of the resolution (``ic_lora_adaptive_window``). It excludes ``window_frames``, because a
    fixed window chunks every longer clip. ``max_window_frames`` is the run's own maximum
    for it. None keeps the cap of the resolution.

    ``carry_frames`` is the part of a window that the next window re-renders and pins at
    its start. It needs a ``window_frames`` or an ``adaptive_window``. None keeps
    ``DEFAULT_CARRY_FRAMES``. ``carry_pinned`` is set when the run chose the carry. An adaptive
    run that did not is on Auto carry: ``carry_frames`` is then the longest carry that Auto may
    use, and the job budget picks the carry that is safest (``ic_lora_auto_carry``).

    ``overlap_fraction`` is the share of a tile that the next tile covers again. A smaller
    share means fewer tiles per step, and a faster run with a higher risk of seams.

    ``max_output_long_side`` and ``max_output_short_side`` raise the spatial cap of the
    job for this LoRA. The transformer sees one tile at a time, so the output can be
    larger than the local 1080p cell. None keeps that cell.
    """

    model_config = ConfigDict(strict=True, frozen=True)
    long_side: int = Field(ge=64)
    short_side: int = Field(ge=64)
    window_frames: int | None = Field(default=None, ge=9)
    adaptive_window: bool = False
    max_window_frames: int | None = Field(default=None, ge=9)
    carry_frames: int | None = Field(default=None, ge=MIN_CARRY_FRAMES)
    carry_pinned: bool = False
    overlap_fraction: float = Field(default=IC_LORA_TILE_OVERLAP, ge=MIN_TILE_OVERLAP, le=MAX_TILE_OVERLAP)
    max_output_long_side: int | None = Field(default=None, ge=64)
    max_output_short_side: int | None = Field(default=None, ge=64)

    @model_validator(mode="after")
    def _check_grid(self) -> "IcLoraTiling":
        if self.long_side % _TILE_GRID or self.short_side % _TILE_GRID:
            raise ValueError(f"tile sides must be multiples of {_TILE_GRID}")
        if self.short_side > self.long_side:
            raise ValueError("short_side must not exceed long_side")
        if self.window_frames is not None and (self.window_frames - 1) % _FRAME_GRID:
            raise ValueError(f"window_frames must be {_FRAME_GRID}k + 1")
        if self.adaptive_window and self.window_frames is not None:
            raise ValueError("an adaptive window takes no window_frames")
        if self.max_window_frames is not None:
            if not self.adaptive_window:
                raise ValueError("max_window_frames needs adaptive_window")
            if (self.max_window_frames - 1) % _FRAME_GRID:
                raise ValueError(f"max_window_frames must be {_FRAME_GRID}k + 1")
            if self.max_window_frames > ADAPTIVE_MAX_WINDOW_FRAMES:
                raise ValueError(f"max_window_frames cannot exceed {ADAPTIVE_MAX_WINDOW_FRAMES} frames")
        if self.carry_frames is not None:
            if self.window_frames is None and not self.adaptive_window:
                raise ValueError("carry_frames needs window_frames")
            if (self.carry_frames - 1) % _FRAME_GRID:
                raise ValueError(f"carry_frames must be {_FRAME_GRID}k + 1")
            if self.window_frames is not None and self.carry_frames >= self.window_frames:
                raise ValueError("carry_frames must be below window_frames")
        # A middle window keeps window - carry frames. Upstream cross-fades the whole
        # carry, and it needs the cross-fade to be shorter than the kept frames. A
        # window without its own carry runs with the default one.
        carry = DEFAULT_CARRY_FRAMES if self.carry_frames is None else self.carry_frames
        if self.window_frames is not None and self.window_frames <= 2 * carry:
            raise ValueError("window_frames must be more than twice carry_frames")
        if self.max_window_frames is not None and self.max_window_frames <= 2 * carry:
            raise ValueError("The maximum chunk length must be more than twice the carry.")
        if (self.max_output_long_side is None) != (self.max_output_short_side is None):
            raise ValueError("set both max_output sides or neither")
        if (
            self.max_output_long_side is not None
            and self.max_output_short_side is not None
            and self.max_output_short_side > self.max_output_long_side
        ):
            raise ValueError("max_output_short_side must not exceed max_output_long_side")
        return self

    @property
    def auto_carry(self) -> bool:
        """True when the job budget picks the carry: an adaptive window and no carry chosen by the run."""
        return self.adaptive_window and not self.carry_pinned

    def output_cap(self) -> tuple[int, int] | None:
        """(long, short) cap of the output size, or None for the local 1080p cell."""
        if self.max_output_long_side is None or self.max_output_short_side is None:
            return None
        return self.max_output_long_side, self.max_output_short_side

    def tile_size(self, width: int, height: int) -> tuple[int, int]:
        """(width, height) of one tile on a canvas of this size.

        The tile follows the canvas orientation. An axis shorter than its tile stays
        untiled, so the tile never exceeds the canvas.
        """
        if height > width:
            tile_width, tile_height = self.short_side, self.long_side
        else:
            tile_width, tile_height = self.long_side, self.short_side
        return min(width, tile_width), min(height, tile_height)


def transformer_pass_size(width: int, height: int, tiling: IcLoraTiling | None) -> tuple[int, int]:
    """Size of one transformer call on a canvas. The tile when tiled, else the canvas."""
    if tiling is None:
        return width, height
    return tiling.tile_size(width, height)


def auto_carry_candidates(bound: int, max_window_frames: int | None) -> list[int]:
    """The carries Auto can run, longest first: the presets up to ``bound``, and ``bound``.

    A carry stays a candidate only when the maximum chunk length is more than twice the carry.
    """
    carries = sorted({bound, *(c for c in CARRY_PRESETS if MIN_CARRY_FRAMES <= c <= bound)}, reverse=True)
    fitting = [c for c in carries if max_window_frames is None or max_window_frames > 2 * c]
    return fitting or [bound]


def with_tiling_overrides(
    tiling: IcLoraTiling,
    *,
    overlap_fraction: float | None,
    carry_frames: int | None,
    tile_size: IcLoraTileSize | None = None,
    max_window_frames: int | None = None,
) -> IcLoraTiling:
    """The tiling of a run that sets its own overlap, carry, tile size or maximum chunk length.

    None keeps the tiling's value. The maximum chunk length is the longest window the run may
    use. A LoRA with a fixed window always chunks at it, so a shorter maximum becomes the window
    and a longer one is refused: it is outside what the LoRA ran at. An adaptive LoRA keeps the
    maximum as a limit. The cap of the resolution applies later, in the job budget, which knows
    the canvas (a longer decode window stalled the VAE at 4K).

    Raises ``ValueError`` for a pair the pipeline cannot run, for example a carry that
    leaves no frames in a middle window.
    """
    window_field = "max_window_frames" if tiling.adaptive_window else "window_frames"
    if max_window_frames is not None and not tiling.adaptive_window:
        if tiling.window_frames is None:
            raise ValueError("this LoRA runs without a fixed window")
        if max_window_frames > tiling.window_frames:
            raise ValueError(
                f"max_window_frames cannot exceed the {tiling.window_frames} frames of the catalog window"
            )
    updates: dict[str, object] = {
        name: value
        for name, value in (
            ("overlap_fraction", overlap_fraction),
            ("carry_frames", carry_frames),
            (window_field, max_window_frames),
        )
        if value is not None
    }
    if carry_frames is not None:
        updates["carry_pinned"] = True
    elif tiling.auto_carry and max_window_frames is not None:
        # Auto carry under a chosen maximum: start from the longest carry that leaves a valid window.
        bound = DEFAULT_CARRY_FRAMES if tiling.carry_frames is None else tiling.carry_frames
        updates["carry_frames"] = auto_carry_candidates(bound, max_window_frames)[0]
    if tile_size is not None:
        updates["long_side"], updates["short_side"] = TILE_SIZES[tile_size]
    if not updates:
        return tiling
    try:
        return IcLoraTiling.model_validate({**tiling.model_dump(), **updates})
    except ValidationError as exc:
        # The validator raises ValueError. Pydantic stores that exception on the error.
        # Reading it avoids a parse of the "Value error, " prefix, which is Pydantic text.
        original = exc.errors()[0].get("ctx", {}).get("error")
        if not isinstance(original, Exception):
            raise ValueError("invalid tiling") from exc
        raise ValueError(str(original)) from exc
