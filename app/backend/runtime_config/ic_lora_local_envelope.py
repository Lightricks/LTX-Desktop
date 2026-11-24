"""Measured IC-LoRA spatial envelope (LTX-2 1.3.0).

NOT re-measured on 1.5.0 or 1.6.0. 1.4+ moved the reference encode into the chunk flow, so re-run the
bump gate's IC-LoRA scenarios (``iclora_stage2_on``, ``iclora_day_to_night``) on CUDA and
confirm these limits still hold before tightening or loosening them.

On 1.3.0, 720p/1080p source-dim hung in tiled reference VAE encode even after
tia2v-aligned stage-1 encode — an algorithmic failure, not a 5090 VRAM ceiling. The
spatial cap therefore applies on Darwin too. The cap is the local 1080p cell
(1920×1088) on the output frame: stage 1 for a single-stage job, the stage-2
canvas for a two-stage job. Large single-stage jobs run chunked (``ic_lora_chunking``).
A few 1080p runs were done by hand on CUDA and finished. They are not a measured
envelope. The job budget still rejects a job that does not fit the card.
The token/VRAM *curve* does not apply on Darwin (unified memory).
Duration is not a hard 20s cap — CUDA token budget decides stream vs 422; Darwin
has no token ceiling yet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from api_types import TargetResolution
from runtime_config.grid_fit import fit_in_box, grows_into_box
from runtime_config.ic_lora_tiling import IcLoraTiling

IC_LORA_SOURCE_TOO_LARGE = "IC_LORA_SOURCE_TOO_LARGE"
IC_LORA_V1_ENVELOPE_MESSAGE = (
    "This IC-LoRA job is larger than this machine can run locally right now "
    "(max 1080p). Lower the resolution."
)

# Local 1080p grid (ltx_capabilities.TWO_PASS_PIXELS_16_9).
_MAX_SHORT_EDGE = 1088
_MAX_LONG_EDGE = 1920


@dataclass(frozen=True, slots=True)
class EnvelopeError:
    code: str
    message: str


def stage1_size(canvas_width: int, canvas_height: int) -> tuple[int, int]:
    """skip_stage_2 (and two-stage stage-1) run at canvas//2."""
    return canvas_width // 2, canvas_height // 2


def spatial_cap(tiling: IcLoraTiling | None = None) -> tuple[int, int]:
    """(long, short) edge cap of the output. A tiling can raise it (``output_cap``)."""
    raised = None if tiling is None else tiling.output_cap()
    return raised if raised is not None else (_MAX_LONG_EDGE, _MAX_SHORT_EDGE)


def over_spatial_cap(width: int, height: int, tiling: IcLoraTiling | None = None) -> bool:
    """True when a frame is larger than the cell (1920x1088), turned either way.

    Pass the OUTPUT size: stage 1 for a single-stage job, the stage-2 canvas for a
    two-stage job (``heavy_pass_size``). The job budget uses the same rule. A tiling
    can raise the cell for its LoRA.
    """
    max_long, max_short = spatial_cap(tiling)
    too_wide = max(width, height) > max_long
    too_tall = min(width, height) > max_short
    return too_tall or too_wide


def effective_denoise_stage1(
    canvas_width: int,
    canvas_height: int,
    *,
    skip_stage_2: bool,
    resolution_factor: float,
) -> tuple[int, int]:
    """Match ``LTXIcLoraPipeline.generate``: skip_stage_2 scales canvas by factor first."""
    width, height = canvas_width, canvas_height
    if skip_stage_2:
        width = max(128, round(width * resolution_factor / 128) * 128)
        height = max(128, round(height * resolution_factor / 128) * 128)
    return width // 2, height // 2


# A two-stage run that keeps the IC-LoRA on stage 2 (Layout To Render) sizes the
# canvas to the source on a 64 grid, inside the 1920x1088 cell. Upstream needs
# only a multiple of 64 for two stages. The 128 grid stays for every other recipe.
# The VAE spatial stride is 32 px per latent cell. Stage 1 runs at canvas//2, so the
# canvas grid is 2 * 32 = 64 to keep the stage 1 latent on whole cells.
_VAE_SPATIAL_STRIDE = 32
_STAGE_2_IC_LORA_GRID = 2 * _VAE_SPATIAL_STRIDE
_DEFAULT_GRID = 128

# How far a requested resolution may exceed the source, for a recipe that keeps the IC-LoRA
# on stage 2. Stage 1 runs at half the canvas, so a 2x canvas runs stage 1 at the source size
# and stage 2 adds the other 2x. Shared with the Home resolution control.
_STAGE_2_UPSCALE_PATH = Path(__file__).resolve().parents[2] / "shared" / "ic-lora-stage-2-upscale.json"
STAGE_2_MAX_SCALE = float(json.loads(_STAGE_2_UPSCALE_PATH.read_text(encoding="utf-8"))["maxScale"])


def _cap_box(width: int, height: int) -> tuple[int, int]:
    """The 1920x1088 cell, turned the same way as the source."""
    if width >= height:
        return _MAX_LONG_EDGE, _MAX_SHORT_EDGE
    return _MAX_SHORT_EDGE, _MAX_LONG_EDGE


def _two_stage_output_size(
    input_width: int,
    input_height: int,
    resolution: TargetResolution | None,
    *,
    stage_2_ic_lora: bool = False,
) -> tuple[int, int]:
    """Canvas for skip_stage_2=False. Stage 1 runs at canvas//2, and its latent must be
    even for the 2x patchify. The canvas never upscales the source, except that every
    edge has a one-grid-cell minimum (64 or 128). A source below that minimum is
    rendered at the minimum cell.

    With the IC-LoRA on stage 2, the canvas fits the source into the requested
    resolution (or the 1920x1088 cell) on a 64 grid, close to the source aspect
    (``fit_in_box``). A source too small to fill the requested resolution may grow up to
    ``STAGE_2_MAX_SCALE`` times its size and never past the cell, so stage 2 upscales a
    small clip. A source that already fills it keeps its own size, as does a job with no
    requested resolution. The canvas is the stage-2 size, so the heavy pass and the cap
    are those of a source of that size. Every other recipe snaps each edge down to a
    multiple of 128, which mirrors the t2v two-stage landing size.
    """
    if stage_2_ic_lora:
        box_w, box_h = _cap_box(input_width, input_height)
        if resolution is None:
            return fit_in_box(input_width, input_height, box_w, box_h, _STAGE_2_IC_LORA_GRID)
        box_w, box_h = min(box_w, resolution.width), min(box_h, resolution.height)
        grows = grows_into_box(input_width, input_height, box_w, box_h, _STAGE_2_IC_LORA_GRID)
        return fit_in_box(
            input_width,
            input_height,
            box_w,
            box_h,
            _STAGE_2_IC_LORA_GRID,
            max_scale=STAGE_2_MAX_SCALE if grows else 1.0,
        )
    target_w = resolution.width if resolution else input_width
    target_h = resolution.height if resolution else input_height
    width = min(target_w, input_width)
    height = min(target_h, input_height)
    return (
        max(_DEFAULT_GRID, (width // _DEFAULT_GRID) * _DEFAULT_GRID),
        max(_DEFAULT_GRID, (height // _DEFAULT_GRID) * _DEFAULT_GRID),
    )


def fit_ic_lora_stage1(
    width: int, height: int, tiling: IcLoraTiling | None = None
) -> tuple[int, int]:
    """Stage-1 size the skip-stage-2 pipeline will actually emit, inside the cap.

    The pipeline rounds the 2× canvas onto a 128 grid, so a 270p/360p target
    that is not on that grid is not the frame that renders. Sizes over the
    1024×576 cap shrink on the 64px grid and keep the aspect (1344×576 → 896×384).
    """
    canvas_w = max(128, round(width * 2 / 128) * 128)
    canvas_h = max(128, round(height * 2 / 128) * 128)
    stage_w, stage_h = canvas_w // 2, canvas_h // 2
    if not over_spatial_cap(stage_w, stage_h, tiling):
        return stage_w, stage_h
    max_long, max_short = spatial_cap(tiling)
    long_is_width = stage_w >= stage_h
    long_edge = stage_w if long_is_width else stage_h
    short_edge = stage_h if long_is_width else stage_w
    scale = min(max_long / long_edge, max_short / short_edge)
    short = max(64, int(short_edge * scale) // 64 * 64)
    ratio = long_edge / short_edge
    long = max(64, round(short * ratio) // 64 * 64)
    while short > 64 and (long > max_long or short > max_short):
        short -= 64
        long = max(64, round(short * ratio) // 64 * 64)
    if long_is_width:
        return long, short
    return short, long


def ic_lora_output_canvas(
    *,
    skip_stage_2: bool,
    resolution_factor: float,
    input_width: int,
    input_height: int,
    resolution: TargetResolution | None,
    stage_2_ic_lora: bool = False,
    tiling: IcLoraTiling | None = None,
) -> tuple[int, int, float]:
    """Canvas + factor for catalog and built-in (canny/depth) generate paths.

    Two-stage always sizes to source (snapped /128), not the skip-stage-2 768 bucket.
    skip_stage_2 with an explicit resolution renders that size after it is fit
    onto the pipeline's 128 grid and the 1024×576 cap. Canvas is 2× that stage-1
    size and the factor is 1.0. GenSpace's generate() rejects that combination.
    Only the queued catalog entry sends it.
    skip_stage_2 + resolution_factor 0 and no resolution is the "source dimensions"
    sentinel: pass 2×source at factor 1.0. Anything else stays on the 768-wide bucket.
    """
    if not skip_stage_2:
        width, height = _two_stage_output_size(
            input_width, input_height, resolution, stage_2_ic_lora=stage_2_ic_lora
        )
        return width, height, 1.0
    if resolution is not None:
        stage_w, stage_h = fit_ic_lora_stage1(resolution.width, resolution.height, tiling)
        return stage_w * 2, stage_h * 2, 1.0
    if resolution_factor == 0:
        return max(128, 2 * input_width), max(128, 2 * input_height), 1.0
    width = 768
    height = max(round(width * input_height / input_width / 128) * 128, 128)
    return width, height, resolution_factor


def _too_large() -> EnvelopeError:
    return EnvelopeError(code=IC_LORA_SOURCE_TOO_LARGE, message=IC_LORA_V1_ENVELOPE_MESSAGE)


def _spatial_error(width: int, height: int) -> EnvelopeError | None:
    if over_spatial_cap(width, height):
        return _too_large()
    return None


def ic_lora_v1_envelope_error(
    width: int,
    height: int,
    *,
    duration_seconds: float,
    fps: float = 24.0,
) -> EnvelopeError | None:
    """Spatial error for an output size (see ``over_spatial_cap``)."""
    del duration_seconds, fps  # duration is the token budget, not a hard 20s cap
    return _spatial_error(width, height)


def ic_lora_v1_canvas_envelope_error(
    canvas_width: int,
    canvas_height: int,
    *,
    frame_count: int,
    fps: float,
) -> EnvelopeError | None:
    del frame_count, fps
    stage1_w, stage1_h = stage1_size(canvas_width, canvas_height)
    return _spatial_error(stage1_w, stage1_h)
