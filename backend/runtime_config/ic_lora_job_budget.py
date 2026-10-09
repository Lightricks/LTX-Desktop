"""IC-LoRA job budget: reference-append tokens ×2, plus measured spatial No."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from api_types import REFERENCE_STILL_FRAME_IDX, IcLoraCatalogItem, IcLoraControl, IcLoraSettings
from frame_math import compute_num_frames
from runtime_config.ic_lora_chunking import (
    IC_LORA_CHUNK_PIXEL_FRAMES,
    ic_lora_chunk_pixel_frames,
    ic_lora_may_chunk,
)
from runtime_config.ic_lora_local_envelope import (
    effective_denoise_stage1,
    ic_lora_output_canvas,
    over_spatial_cap,
)
from runtime_config.ic_lora_stage_mode import IcLoraStageMode, heavy_pass_size
from runtime_config.ic_lora_tiling import IcLoraTiling, transformer_pass_size
from runtime_config.runtime_policy import LocalGenerationMode
from runtime_config.video_job_budget import (
    LoadMode,
    VideoJobReject,
    decide_load_mode,
    estimated_giB,
    video_tokens,
)

# Compat-matrix preview of a Home IC-LoRA job. Home runs entries that skip stage 2 and
# entries that keep the IC-LoRA on stage 2 (Layout To Render). This preview is the
# single-stage case, the lightest one. resolution_factor is not a Home control; 1.5 is
# the catalog preview bucket this estimate has always used.
HOME_IC_LORA_PREVIEW_SETTINGS = IcLoraSettings(skip_stage_2=True, resolution_factor=1.5)

# Fast full-vs-stream headroom is 2 GiB. IC-LoRA 5s 540p full ~22.5 GiB is OK;
# 8s/10s Ingredients paged on a 5090 while the linear full estimate (~25–26 GiB)
# still fit 29 usable. 7 GiB so 8s+ streams without changing the stream-over 422.
# 8s/540p vs 32−7=25.0 usable is a ~0.03 GiB miss by design — do not round this
# to 6 or 8 without re-measuring; 6 would full-load and page, 8 would stream 5s.
_IC_LORA_FULL_HEADROOM_GIB = 7.0

# List-time duration filter has no user image. skip_stage_2: 9:16 768×1280 through
# the 768 bucket × factor 1.5 is stage-1 576×960 (Ingredients default). Two-stage
# sizes to the user's source, so preview a local 1080p sheet or the filter is
# optimistic (~8 GiB under a real 1080p two-stage 20s).
_CATALOG_PREVIEW_SKIP_STAGE2_WIDTH = 768
_CATALOG_PREVIEW_SKIP_STAGE2_HEIGHT = 1280
_CATALOG_PREVIEW_TWO_STAGE_WIDTH = 1920
_CATALOG_PREVIEW_TWO_STAGE_HEIGHT = 1088

_DEFAULT_IC_LORA_FPS = 24


@dataclass(frozen=True, slots=True)
class IcLoraJobLoad:
    mode: LoadMode
    seq: float
    full_gib: float
    stream_gib: float
    # Window length when the job runs chunked, else None. The pipeline runs this value.
    chunk_pixel_frames: int | None = None


@dataclass(frozen=True, slots=True)
class IcLoraJobReject:
    reason: Literal["spatial", "stream_over", "unsupported"]
    seq: float
    full_gib: float
    stream_gib: float


IcLoraJobDecision = IcLoraJobLoad | IcLoraJobReject


def ic_lora_sequence_tokens(width: int, height: int, frames: int) -> float:
    """Video tokens plus the appended reference clip (same H×W×frames)."""
    return video_tokens(height, width, frames) * 2.0


def _heaviest_pass_tokens(
    stage_mode: IcLoraStageMode,
    stage1_width: int,
    stage1_height: int,
    frames: int,
    stills: int,
    tiling: IcLoraTiling | None,
) -> float:
    """Tokens of the heaviest pass of the job.

    Stage 2 of a STAGE_1_LORA job has no reference, but it runs at twice the stage-1
    size. That is four times the stage-1 tokens, so it is heavier than stage 1 with its
    reference (two times). The other layouts carry the reference in their largest pass.
    Each reference still adds one latent frame of tokens to that pass. A STAGE_1_LORA job
    takes no still.

    A tiled job (single stage) runs the transformer on one tile at a time. Its pass is the
    tile, with the reference cropped to the same tile.
    """
    width, height = transformer_pass_size(*heavy_pass_size(stage_mode, stage1_width, stage1_height), tiling)
    if stage_mode is IcLoraStageMode.STAGE_1_LORA:
        return max(
            video_tokens(height, width, frames),
            ic_lora_sequence_tokens(stage1_width, stage1_height, frames),
        )
    still_tokens = stills * (height / 32.0) * (width / 32.0)
    return ic_lora_sequence_tokens(width, height, frames) + still_tokens


def decide_ic_lora_job(
    stage1_width: int,
    stage1_height: int,
    frames: int,
    *,
    memory_gb: float | None,
    process_mode: LocalGenerationMode,
    stage_mode: IcLoraStageMode,
    darwin: bool = False,
    stills: int = 0,
    tiling: IcLoraTiling | None = None,
) -> IcLoraJobDecision:
    """422 definite-over jobs; per-job stream when full will not fit.

    ``stage_mode`` sets the heaviest pass (``heavy_pass_size``). That pass sets the priced
    token count, the chunk rule and the spatial cap. The spatial cap checks the output
    size, so a two-stage job is capped at its stage-2 canvas, not at stage 1.

    This is the one place that decides chunking. A job chunks when the size and token
    rule says so, or when it may chunk and does not fit unchunked.
    ``IcLoraJobLoad.chunk_pixel_frames`` carries the decision to the pipeline.

    ``stills`` is the number of reference stills. Each adds one latent frame of tokens to
    each window, so a chunked job pays for it once per window.

    ``tiling`` sizes the priced pass by the tile. The spatial cap still checks the output
    size, so a tiled job is capped like any other.

    Darwin and stream-only processes never return full. Unknown memory streams
    rather than guessing full. Uses extra full-mode headroom so 8s+ 540p streams
    on 32 GB; 5s stays full. Stream-over 422 compares to total VRAM on CUDA only;
    Darwin skips it (unified memory, already-free RAM).
    """
    token_width, token_height = heavy_pass_size(stage_mode, stage1_width, stage1_height)

    def price(window: int | None) -> IcLoraJobDecision:
        # A chunked job holds one window at a time.
        seq = _heaviest_pass_tokens(
            stage_mode, stage1_width, stage1_height, frames if window is None else window, stills, tiling
        )
        full_gib = estimated_giB("full_models_loading", seq)
        stream_gib = estimated_giB("streaming_models_loading", seq)
        if process_mode == "unsupported":
            return IcLoraJobReject(reason="unsupported", seq=seq, full_gib=full_gib, stream_gib=stream_gib)
        if over_spatial_cap(token_width, token_height):
            return IcLoraJobReject(reason="spatial", seq=seq, full_gib=full_gib, stream_gib=stream_gib)
        decision = decide_load_mode(
            seq,
            memory_gb=memory_gb,
            process_mode=process_mode,
            darwin=darwin,
            full_headroom_gib=_IC_LORA_FULL_HEADROOM_GIB,
        )
        if isinstance(decision, VideoJobReject):
            return IcLoraJobReject(
                reason=decision.reason,
                seq=decision.seq,
                full_gib=decision.full_gib,
                stream_gib=decision.stream_gib,
            )
        return IcLoraJobLoad(
            mode=decision.mode,
            seq=decision.seq,
            full_gib=decision.full_gib,
            stream_gib=decision.stream_gib,
            chunk_pixel_frames=window,
        )

    window = ic_lora_chunk_pixel_frames(
        stage1_width, stage1_height, frames, stage_mode=stage_mode, tiling=tiling
    )
    decision = price(window)
    if (
        window is None
        and isinstance(decision, IcLoraJobReject)
        and decision.reason == "stream_over"
        and ic_lora_may_chunk(stage1_width, stage1_height, frames, stage_mode=stage_mode, tiling=tiling)
    ):
        # Without this, a clip just under the token rule is priced whole and a longer
        # clip is priced as one window, so the longer clip would be cheaper.
        return price(IC_LORA_CHUNK_PIXEL_FRAMES)
    return decision


def catalog_preview_stage1(settings: IcLoraSettings) -> tuple[int, int]:
    """Stage-1 pixels for list-time duration filtering (no user image yet)."""
    if settings.skip_stage_2:
        input_width, input_height = (
            _CATALOG_PREVIEW_SKIP_STAGE2_WIDTH,
            _CATALOG_PREVIEW_SKIP_STAGE2_HEIGHT,
        )
    else:
        input_width, input_height = (
            _CATALOG_PREVIEW_TWO_STAGE_WIDTH,
            _CATALOG_PREVIEW_TWO_STAGE_HEIGHT,
        )
    canvas_w, canvas_h, factor = ic_lora_output_canvas(
        skip_stage_2=settings.skip_stage_2,
        resolution_factor=settings.resolution_factor,
        input_width=input_width,
        input_height=input_height,
        resolution=None,
        stage_2_ic_lora=settings.stage_2_ic_lora,
    )
    return effective_denoise_stage1(
        canvas_w, canvas_h, skip_stage_2=settings.skip_stage_2, resolution_factor=factor
    )


def advertised_duration_options(
    options: list[int],
    *,
    stage1_width: int,
    stage1_height: int,
    stage_mode: IcLoraStageMode,
    fps: int,
    memory_gb: float | None,
    process_mode: LocalGenerationMode,
    darwin: bool,
    stills: int = 0,
    tiling: IcLoraTiling | None = None,
) -> list[int]:
    """Intersect catalog duration seconds with ``decide_ic_lora_job``.

    ``stage_mode`` must be the mode the pipeline runs, or the filter prices another job.
    ``stills`` is the number of reference stills the job can carry. ``tiling`` is the
    LoRA's tile size, if any.
    """
    kept: list[int] = []
    for seconds in options:
        frames = compute_num_frames(seconds, max(1, fps))
        decision = decide_ic_lora_job(
            stage1_width,
            stage1_height,
            frames,
            memory_gb=memory_gb,
            process_mode=process_mode,
            stage_mode=stage_mode,
            darwin=darwin,
            stills=stills,
            tiling=tiling,
        )
        if isinstance(decision, IcLoraJobLoad):
            kept.append(seconds)
    return kept


def with_budgeted_duration_controls(
    item: IcLoraCatalogItem,
    *,
    memory_gb: float | None,
    process_mode: LocalGenerationMode,
    darwin: bool,
) -> IcLoraCatalogItem:
    """Copy ``item`` with duration ``options`` intersected against the job budget.

    Generate still 422s if a client posts a dropped duration. If every option is
    over budget, leave the control unchanged so the payload stays valid.
    Unsupported machines get empty duration options instead of the full table.
    """
    if memory_gb is None and process_mode != "unsupported":
        return item

    stage1_w, stage1_h = catalog_preview_stage1(item.default_settings)
    stage_mode = item.default_settings.stage_mode
    fps = catalog_output_fps(item)
    # The optional look still is priced as present, so the filter is not optimistic.
    stills = int(item.allows_reference_image and item.reference_image_frame == REFERENCE_STILL_FRAME_IDX)
    controls: list[IcLoraControl] = []
    changed = False
    for control in item.controls:
        if control.id != "duration" or control.kind != "int" or control.options is None:
            controls.append(control)
            continue
        raw_options = [o for o in control.options if type(o) is int]
        kept = advertised_duration_options(
            raw_options,
            stage1_width=stage1_w,
            stage1_height=stage1_h,
            stage_mode=stage_mode,
            fps=fps,
            memory_gb=memory_gb,
            process_mode=process_mode,
            darwin=darwin,
            stills=stills,
            tiling=item.default_settings.tiling,
        )
        if not kept:
            if process_mode == "unsupported":
                controls.append(control.model_copy(update={"options": []}))
                changed = True
                continue
            controls.append(control)
            continue
        if kept == raw_options:
            controls.append(control)
            continue
        default = control.default
        if not isinstance(default, int) or default not in kept:
            default = min(kept)
        controls.append(control.model_copy(update={"options": kept, "default": default}))
        changed = True
    if not changed:
        return item
    return item.model_copy(update={"controls": controls})


def catalog_output_fps(item: IcLoraCatalogItem) -> int:
    """Control-video fps from ``image_to_frames``, or 24.

    Catalog JSON is ``dict[str, JsonValue]``. A None / object / junk ``fps`` must
    not 500 ``GET /api/ic-loras``.
    """
    for step in item.preprocessing:
        if step.utility == "image_to_frames":
            return _coerce_positive_fps(step.params.get("fps"))
    return _DEFAULT_IC_LORA_FPS


def _coerce_positive_fps(value: object) -> int:
    if type(value) is bool:
        return _DEFAULT_IC_LORA_FPS
    if type(value) is int and value > 0:
        return value
    if type(value) is float and value > 0:
        return int(value)
    if isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError:
            return _DEFAULT_IC_LORA_FPS
        if parsed > 0:
            return parsed
    return _DEFAULT_IC_LORA_FPS
