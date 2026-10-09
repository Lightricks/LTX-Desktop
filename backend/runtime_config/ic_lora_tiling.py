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

from pydantic import BaseModel, ConfigDict, Field, model_validator

# The overlap fraction we request. It is the most upstream accepts, and the least the
# Refine Details card allows ("never lower the overlap below 0.5"). It is a request, not
# the layout: upstream ``split_by_size_pinned`` still caps the tile count so that no cell
# is covered by three tiles. A 1920x1088 canvas gets 2x2 tiles with a small overlap, not
# the card's 3x3 tiles at 50%.
IC_LORA_TILE_OVERLAP = 0.5

_TILE_GRID = 32  # pixel multiple of the transformer latent grid
_FRAME_GRID = 8  # pixel frames per latent frame; windows are 8k + 1 frames


class IcLoraTiling(BaseModel):
    """Tile size of a catalog IC-LoRA, and an optional fixed temporal window.

    ``long_side`` and ``short_side`` are given for a landscape canvas. A portrait canvas
    swaps them (Refine Details card: "1024x576, or 576x1024 portrait").

    ``window_frames`` is set when the LoRA drifts over a longer clip (Restore: past 97
    frames detail is lost and colour drifts). A clip longer than the window always runs
    in windows of this length. None leaves the choice to the job budget.
    """

    model_config = ConfigDict(strict=True, frozen=True)
    long_side: int = Field(ge=64)
    short_side: int = Field(ge=64)
    window_frames: int | None = Field(default=None, ge=9)

    @model_validator(mode="after")
    def _check_grid(self) -> "IcLoraTiling":
        if self.long_side % _TILE_GRID or self.short_side % _TILE_GRID:
            raise ValueError(f"tile sides must be multiples of {_TILE_GRID}")
        if self.short_side > self.long_side:
            raise ValueError("short_side must not exceed long_side")
        if self.window_frames is not None and (self.window_frames - 1) % _FRAME_GRID:
            raise ValueError(f"window_frames must be {_FRAME_GRID}k + 1")
        return self

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
