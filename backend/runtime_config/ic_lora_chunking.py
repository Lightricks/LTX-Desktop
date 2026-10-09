"""IC-LoRA temporal chunking rule, read by the job budget.

A large job runs in overlapping windows (ltx-pipelines ``ChunkConfig``), so memory
follows the window, not the clip. The rule sizes the heaviest pass that carries the
IC-LoRA reference: stage 1 for a single-stage job, the stage-2 canvas for a recipe that
keeps the IC-LoRA on stage 2. A two-stage job without the IC-LoRA on stage 2 never
chunks. The rule fires only above the former 1024x576 size and past the largest
one-window token count with a measured pass. A chunked run is not measured yet.

The job budget (``decide_ic_lora_job``) also chunks a job that may chunk but does not fit
unchunked. It is the one place that decides. The pipeline receives the window it returns.
"""

from __future__ import annotations

from frame_math import snap_to_frame_grid
from runtime_config.ic_lora_stage_mode import IcLoraStageMode, heavy_pass_size
from runtime_config.ic_lora_tiling import IcLoraTiling, transformer_pass_size
from runtime_config.video_job_budget import video_tokens

# ltx-pipelines default layout: 97-frame windows, 25-frame carry.
IC_LORA_CHUNK_PIXEL_FRAMES = 97
IC_LORA_CHUNK_CARRY_FRAMES = 25

_FORMER_CAP_PIXELS = 1024 * 576
IC_LORA_MAX_SINGLE_WINDOW_TOKENS = 32_940


def ic_lora_may_chunk(
    stage1_width: int,
    stage1_height: int,
    frames: int,
    *,
    stage_mode: IcLoraStageMode,
    tiling: IcLoraTiling | None = None,
) -> bool:
    """True when a job may run chunked: a layout that chunks, a heavy pass above the
    former 1024x576 size, and a clip longer than one window.

    A tiled job sizes the pass by its tile, not by the canvas.

    A two-stage job without the IC-LoRA on stage 2 never chunks. The clip length is
    counted in latent frames, so a count off the 8k+1 grid (123) acts like the grid
    count below it (121).
    """
    if stage_mode is IcLoraStageMode.STAGE_1_LORA:
        return False
    if snap_to_frame_grid(frames, floor=1) <= IC_LORA_CHUNK_PIXEL_FRAMES:
        return False
    width, height = transformer_pass_size(*heavy_pass_size(stage_mode, stage1_width, stage1_height), tiling)
    return width * height > _FORMER_CAP_PIXELS


def ic_lora_chunk_pixel_frames(
    stage1_width: int,
    stage1_height: int,
    frames: int,
    *,
    stage_mode: IcLoraStageMode,
    tiling: IcLoraTiling | None = None,
) -> int | None:
    """Window length in pixel frames when a rule says chunk, else None.

    A LoRA with a fixed window (``IcLoraTiling.window_frames``) chunks every clip longer
    than that window, whatever its size. Otherwise the size and token rule decides.
    """
    if tiling is not None and tiling.window_frames is not None:
        if snap_to_frame_grid(frames, floor=1) > tiling.window_frames:
            return tiling.window_frames
    if not ic_lora_may_chunk(stage1_width, stage1_height, frames, stage_mode=stage_mode, tiling=tiling):
        return None
    width, height = transformer_pass_size(*heavy_pass_size(stage_mode, stage1_width, stage1_height), tiling)
    if video_tokens(height, width, snap_to_frame_grid(frames, floor=1)) <= IC_LORA_MAX_SINGLE_WINDOW_TOKENS:
        return None
    return IC_LORA_CHUNK_PIXEL_FRAMES
