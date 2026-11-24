"""IC-LoRA endpoints orchestration handler."""

from __future__ import annotations

import base64
import logging
import math
import os
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import assert_never, cast, get_args

from PIL import UnidentifiedImageError

from api_types import (
    IC_LORA_CONTROL_TYPE_ID,
    ConditioningType,
    GenerationRisk,
    IcLoraCatalogItem,
    IcLoraControlType,
    IcLoraExtractRequest,
    IcLoraExtractResponse,
    IcLoraGenerateCancelledResponse,
    IcLoraGenerateCompleteResponse,
    IcLoraGenerateRequest,
    IcLoraGenerateResponse,
    IcLoraAutoSinglePass,
    IcLoraAutoWindowed,
    IcLoraCarryPreflightCell,
    IcLoraMaxWindowPreflightCell,
    IcLoraPreflightCell,
    IcLoraPreflightRequest,
    IcLoraPreflightResponse,
    IcLoraPreflightSource,
    IcLoraPreflightVerdict,
    IcLoraTilePreflightCell,
    IcLoraTilePreflightResponse,
    IcLoraTilingDefaults,
    IcLoraTilePreset,
    IcLoraTileSize,
    IcLoraAudioMode,
    IcLoraCellFit,
    IcLoraRecipeMemoryRequest,
    IcLoraRecipeMemoryResponse,
    IcLoraRecipeResolution,
    IcLoraRecipeStoredParams,
    IcLoraSynthesizedStoredParams,
    IcLoraImageInput,
    IcLoraSettings,
    ImageConditioningInput,
    LTXLocalModelId,
    LTXVideoGenAspectRatio,
    TargetResolution,
)
from _routes._errors import HTTPError
from frame_math import compute_num_frames, snap_to_frame_grid, snap_up_to_frame_grid
from handlers.base import StateHandlerBase
from handlers.generation_handler import GenerationHandler
from handlers.pipelines_handler import PipelinesHandler
from handlers.prompt_enhancement_handler import PromptEnhancementHandler
from services.fast_video_pipeline.nag_config import (
    NagConfig,
    NagDecision,
    decide_nag,
    log_nag_decision,
)
from services.prompt_enhancement.source_video import SourceVideo
from handlers.text_handler import TextHandler
from ic_lora_preprocessing import MediaArtifact, OutpaintParams, PreprocessingContext, run_preprocessing
from runtime_config.budget_admission import (
    Admission,
    Admitted,
    PricedJob,
    Refused,
    RiskAllowance,
    admit,
    admits_within_device_memory,
)
from runtime_config.device_calibration import JobEnvelope
from runtime_config.ic_lora_auto_carry import picked_carry_tiling
from runtime_config.ic_lora_chunking import ic_lora_carry_frames
from runtime_config.ic_lora_high_res import (
    high_res_cell,
    high_res_flag_satisfied,
    high_res_settings,
    high_res_stage1,
)
from runtime_config.ic_lora_job_budget import (
    catalog_output_fps,
    catalog_preview_stills,
    dropped_duration_message_and_code,
    price_ic_lora_job,
)
from runtime_config.ic_lora_local_envelope import (
    effective_denoise_stage1,
    ic_lora_output_canvas,
)
from runtime_config.ic_lora_stage_mode import IcLoraStageMode, ic_lora_stage_mode
from runtime_config.ic_lora_tiling import DEFAULT_CARRY_FRAMES, IcLoraTiling, with_tiling_overrides
from runtime_config.evict_retry import evict_retry_filter
from runtime_config.memory_meter import (
    job_filter_machine_factor,
    job_memory_ask,
    memory_capacity,
)
from runtime_config.video_job_budget import (
    LocalJobFilter,
    budget_route,
    vram_snapshot_suffix,
)
from runtime_config.ltx_capabilities import budget_pixels, local_caps, supports
from runtime_config.model_download_specs import (
    DEPTH_PROCESSOR_CP_ID,
    PERSON_DETECTOR_CP_ID,
    POSE_PROCESSOR_CP_ID,
    catalog_item_visible_for_installed_ltx,
    find_installed_ic_lora_path,
    get_existing_cp_path,
    get_ltx_model_spec,
    ltx_catalog_family_for_model,
    resolve_active_ltx_model_id,
)
from runtime_config.models_scanner import is_ic_lora_file, resolve_lora_ref
from runtime_config.runtime_config import RuntimeConfig
from state.conditioning_cache import ConditioningCacheEntry, ConditioningCacheKey
from services.interfaces import GpuInfo, VideoProcessor
from services.aspect_ratio import center_cropped_size, find_closest_aspect_ratio_option
from services.features.ic_lora_recipes import (
    IC_LORA_SUPPORTED_FPS,
    MAX_IC_LORA_INPUT_FRAMES,
    IcLoraRecipe,
    get_ic_lora_recipe,
)
from services.features.video import SYNTHESIZED_IC_LORA_EXECUTORS
from services.features.video.job_common import denoise_canvas, resolve_job_local_model_id
from services.lora_catalog import LoraCatalogProvider
from services.records import AssetRecord
from services.services_utils import FrameArray
from server_utils.heartbeat import log_heartbeat
from server_utils.oriented_image import open_oriented_image
from services.generation_interrupt import GenerationCancelledError, is_cancel_exception
from state.app_state_types import AppState, ICLoraState

logger = logging.getLogger(__name__)

# A 16:9 clip of each cell's size: what the memory advice prices before a clip is picked. The
# source mostly sets the aspect, and a recipe runs its cell, so the cell's own size stands in.
_STAND_IN_SOURCE: dict[IcLoraRecipeResolution, tuple[int, int]] = {
    "270p": (480, 270),
    "360p": (640, 360),
    "540p": (960, 540),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "2160p": (3840, 2160),
}


@dataclass(frozen=True)
class RecipeRunPlan:
    """What a queued Home recipe runs: the frame count, its risk, and its chunk window.

    ``window`` is the pixel frames of one chunk, or None when the clip runs in one pass.
    ``carry`` is the carry of the chunks, the one Auto picked when the run did not choose it.
    """

    frames: int
    risk: GenerationRisk | None
    window: int | None
    carry: int


_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})
_VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".webm", ".mkv"})


def _validate_input_kind(input_path: str, expected_kind: str) -> None:
    """Reject an input whose file type doesn't match what the IC-LoRA expects.

    Cheap extension check so a mismatched image/video fails with a clean 400 here
    rather than a deep pipeline 500 (the inputs come from the app's own picker/outputs,
    so the extension is reliable).
    """
    suffix = Path(input_path).suffix.lower()
    allowed = _VIDEO_SUFFIXES if expected_kind == "video" else _IMAGE_SUFFIXES
    if suffix not in allowed:
        raise HTTPError(400, f"This IC-LoRA expects {expected_kind} input, but got '{suffix or 'a file with no extension'}'")


def _reference_frame_idx(ic_lora: IcLoraCatalogItem, img: IcLoraImageInput) -> int:
    """The frame a reference image conditions.

    A catalog pin always wins. Every Home recipe is pinned (``is_home_recipe``),
    so it never reads the request frame. Only an unpinned entry, from the Studio
    form, uses the request frame.
    """
    pinned = ic_lora.reference_image_frame
    return pinned if pinned is not None else int(img.frame)


def _with_high_res_plan(settings: IcLoraSettings, *, high_res: bool) -> IcLoraSettings:
    """The settings with the plan of a high cell (1440p, 2160p), before the request's overrides.

    The one place that applies the plan, for the run and for the preflight defaults.
    """
    return high_res_settings(settings) if high_res else settings


def _resolve_settings(
    req: IcLoraGenerateRequest, base: IcLoraSettings, *, high_res: bool = False
) -> IcLoraSettings:
    """Overlay the request's explicit settings onto a base; None means 'keep the base'.

    ``high_res`` runs the LoRA as tiles in windows, for the 1440p and 2160p cells. The
    request's tile overlap and carry come last, so they win over the cell's own plan.
    """
    requested_stage2_lora = (
        base.use_lora_in_stage_2 if req.use_lora_in_stage_2 is None else req.use_lora_in_stage_2
    )
    if requested_stage2_lora:
        logger.warning("[ic-lora] use_lora_in_stage_2 is deprecated and ignored (LTXP-514)")
    # A tiled entry runs one stage only. The catalog sets it, so a request override
    # (GenSpace advanced controls send one) cannot turn stage 2 back on.
    skip_stage_2 = (
        True
        if base.tiling is not None
        else base.skip_stage_2 if req.skip_stage_2 is None else req.skip_stage_2
    )
    settings = IcLoraSettings(
        skip_stage_2=skip_stage_2,
        use_lora_in_stage_2=False,
        # Stage 2 does not run when skipped, so the flag would only change the cache key.
        stage_2_ic_lora=base.stage_2_ic_lora and not skip_stage_2,
        # The tiling is the catalog's. ``_with_requested_tiling`` applies the request's overrides.
        tiling=base.tiling,
        resolution_factor=base.resolution_factor if req.resolution_factor is None else req.resolution_factor,
        audio_mode=base.audio_mode if req.audio_mode is None else req.audio_mode,
        lora_strength=base.lora_strength if req.lora_strength is None else req.lora_strength,
        conditioning_strength=(
            base.conditioning_strength if req.conditioning_strength is None else req.conditioning_strength
        ),
    )
    return _with_requested_tiling(req, _with_high_res_plan(settings, high_res=high_res))


def _tiling_defaults(item: IcLoraCatalogItem, stored: IcLoraRecipeStoredParams) -> IcLoraTilingDefaults:
    """The carry and tile the run keeps when the request omits them.

    The high-cell plan is applied to the catalog settings, as ``_resolve_settings`` does before
    the request's overrides. The window is not here: Auto picks it from the clip, so the job
    plan gives it.
    """
    tiling = _with_high_res_plan(
        item.default_settings, high_res=high_res_cell(stored.resolution) is not None
    ).tiling
    carry = DEFAULT_CARRY_FRAMES if tiling is None or tiling.carry_frames is None else tiling.carry_frames
    return IcLoraTilingDefaults(
        carryFrames=carry, overlapFraction=None if tiling is None else tiling.overlap_fraction
    )


def _with_requested_tiling(req: IcLoraGenerateRequest, settings: IcLoraSettings) -> IcLoraSettings:
    """The settings with the request's tile overlap, carry, tile size and window. 422 when they cannot run."""
    if (
        req.tile_overlap is None
        and req.carry_frames is None
        and req.tile_size is None
        and req.max_window_frames is None
    ):
        return settings
    if settings.tiling is None:
        raise HTTPError(
            422,
            f"{req.ic_lora_id} takes no tile overlap, carry, tile size or window.",
            code="INVALID_GENERATION_SPEC",
        )
    try:
        tiling = with_tiling_overrides(
            settings.tiling,
            overlap_fraction=req.tile_overlap,
            carry_frames=req.carry_frames,
            tile_size=req.tile_size,
            max_window_frames=req.max_window_frames,
        )
    except ValueError as exc:
        raise HTTPError(422, f"Invalid tiling: {exc}.", code="INVALID_GENERATION_SPEC") from exc
    return settings.model_copy(update={"tiling": tiling})


class IcLoraHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        generation_handler: GenerationHandler,
        pipelines_handler: PipelinesHandler,
        text_handler: TextHandler,
        prompt_enhancement_handler: PromptEnhancementHandler,
        video_processor: VideoProcessor,
        lora_catalog: LoraCatalogProvider,
        config: RuntimeConfig,
        gpu_info: GpuInfo,
        high_res_enabled: Callable[[], bool] = lambda: False,
    ) -> None:
        super().__init__(state, lock, config, gpu_info=gpu_info)
        self._high_res_enabled = high_res_enabled
        self._generation = generation_handler
        self._pipelines = pipelines_handler
        self._text = text_handler
        self._prompt_enhancement = prompt_enhancement_handler
        self._video_processor = video_processor
        self._catalog = lora_catalog

    def _price_ic_lora(
        self,
        canvas_width: int,
        canvas_height: int,
        *,
        frame_count: int,
        stage_mode: IcLoraStageMode,
        resolution_factor: float,
        job_filter: LocalJobFilter,
        ic_lora_id: str | None = None,
        stills: int = 0,
        tiling: IcLoraTiling | None = None,
    ) -> PricedJob:
        stage1_w, stage1_h = effective_denoise_stage1(
            canvas_width,
            canvas_height,
            skip_stage_2=stage_mode is IcLoraStageMode.SINGLE_STAGE,
            resolution_factor=resolution_factor,
        )
        return price_ic_lora_job(
            stage1_w,
            stage1_h,
            frame_count,
            stage_mode=stage_mode,
            job_filter=job_filter,
            ic_lora_id=ic_lora_id,
            stills=stills,
            tiling=tiling,
        )

    def _decide_ic_lora_admission(
        self,
        canvas_width: int,
        canvas_height: int,
        *,
        frame_count: int,
        stage_mode: IcLoraStageMode,
        resolution_factor: float,
        job_filter: LocalJobFilter,
        ic_lora_id: str | None = None,
        stills: int = 0,
        tiling: IcLoraTiling | None = None,
        allowance: RiskAllowance | None = None,
    ) -> Admission:
        """Spatial + budget admission without raising. Generate raises on a
        refusal; preflight returns it per duration. ic_lora_id threads to the
        Mac ingredients term (None prices on the line)."""
        return admit(
            self._price_ic_lora(
                canvas_width,
                canvas_height,
                frame_count=frame_count,
                stage_mode=stage_mode,
                resolution_factor=resolution_factor,
                job_filter=job_filter,
                ic_lora_id=ic_lora_id,
                stills=stills,
                tiling=tiling,
            ),
            allowance if allowance is not None else RiskAllowance.for_mode(job_filter.budget_mode),
        )

    def _ic_lora_job(
        self,
        canvas_width: int,
        canvas_height: int,
        *,
        frame_count: int,
        stage_mode: IcLoraStageMode,
        resolution_factor: float,
        ic_lora_id: str | None = None,
        stills: int = 0,
        tiling: IcLoraTiling | None = None,
        total_only: bool = False,
        allow_evict_retry: bool = False,
        allowance: RiskAllowance | None = None,
    ) -> Admitted:
        """422, or admit this job: full vs stream, the chunk window, and any bet.

        Does not change process mode. The pipeline runs the window this returns.
        ``allowance=None`` follows the filter's live mode (every request-scoped
        caller); executors pass an explicit allowance parsed from the stored
        annotation so a mode flip never re-gates a queued job.

        Queued and direct execute (pre-start probe only) pass
        ``allow_evict_retry``: a total-possible CUDA reject whose shortfall
        live weights cover gets one evict-and-resample before admission. The
        post-preprocess re-check must leave it off — a generation is Running
        by then and local refs pin the loaded pipeline, so an evict could
        not free it.
        """
        job_filter = self._local_job_filter(total_only=total_only)

        def price(job_filter: LocalJobFilter) -> PricedJob:
            return self._price_ic_lora(
                canvas_width,
                canvas_height,
                frame_count=frame_count,
                stage_mode=stage_mode,
                resolution_factor=resolution_factor,
                job_filter=job_filter,
                ic_lora_id=ic_lora_id,
                stills=stills,
                tiling=tiling,
            )

        priced = price(job_filter)
        if allow_evict_retry:
            rebuilt = evict_retry_filter(
                priced=priced,
                job_filter=job_filter,
                live_allocated_bytes=self._gpu.get_cuda_live_allocated_bytes(),
                reserved_bytes=self._gpu.get_cuda_reserved_bytes(),
                log_prefix="ic-lora",
                evict=self._pipelines.evict_resident_pipeline,
                rebuild=lambda: self._local_job_filter(total_only=total_only),
            )
            if rebuilt is not None:
                job_filter = rebuilt
                priced = price(job_filter)
        admission = admit(
            priced,
            allowance if allowance is not None else RiskAllowance.for_mode(job_filter.budget_mode),
        )
        snapshot = vram_snapshot_suffix(
            allocated_bytes=self._gpu.get_cuda_live_allocated_bytes(),
            reserved_bytes=self._gpu.get_cuda_reserved_bytes(),
            free_bytes=job_filter.budget.free_vram_bytes,
        )
        if isinstance(admission, Refused):
            logger.info(
                "[ic-lora] job budget seq=%.0f full=%.1fGiB stream=%.1fGiB -> reject:%s%s",
                priced.seq,
                priced.full_gib,
                priced.stream_gib,
                admission.reject.reason,
                snapshot,
            )
            message, code = dropped_duration_message_and_code(admission.reject, job_filter)
            raise HTTPError(422, message, code=code)
        risk = admission.risk
        logger.info(
            "[ic-lora] job budget seq=%.0f full=%.1fGiB stream=%.1fGiB -> %s%s%s",
            priced.seq,
            priced.full_gib,
            priced.stream_gib,
            admission.load.mode,
            (
                f" risk:{risk.tier} over_gib={risk.over_gib} over_ram_gb={risk.over_ram_gb}"
                if risk is not None
                else ""
            ),
            snapshot,
        )
        return admission

    def _probe_ic_lora_source(
        self, path: str, kind: str, requested_frames: int, *, raw_frames: bool = False
    ) -> tuple[int, int, int]:
        """Width, height, and snapped frame count from the user file — before preprocess.

        ``raw_frames`` returns the stored count, for a caller that snaps it itself."""
        if kind == "image":
            try:
                width, height = open_oriented_image(path).size
            except (UnidentifiedImageError, OSError) as exc:
                raise HTTPError(400, "Could not read the input image.") from exc
            return width, height, requested_frames
        cap = self._video_processor.open_video(path)
        try:
            if not cap.isOpened():
                raise HTTPError(400, "Could not read the input video.")
            info = self._video_processor.get_video_info(cap)
        finally:
            self._video_processor.release(cap)
        width, height = int(info["width"]), int(info["height"])
        if width <= 0 or height <= 0:
            raise HTTPError(400, "Could not read the input video.")
        src_frames = int(info["frame_count"]) or requested_frames
        return width, height, src_frames if raw_frames else snap_to_frame_grid(src_frames, floor=9)

    def _drop_preprocess_artifact(self, artifact_path: str | None, source_path: str) -> None:
        if not artifact_path or artifact_path == source_path:
            return
        path = Path(artifact_path).resolve()
        try:
            path.relative_to(self.config.outputs_dir.resolve())
        except ValueError:
            return
        path.unlink(missing_ok=True)

    def _build_conditioning_frame(
        self,
        frame: FrameArray,
        conditioning_type: ConditioningType,
        ic_state: ICLoraState | None = None,
    ) -> FrameArray:
        match conditioning_type:
            case "canny":
                return self._video_processor.apply_canny(frame)
            case "depth":
                if ic_state is None or ic_state.depth_pipeline is None:
                    raise HTTPError(500, "Depth conditioning requires loaded IC-LoRA resources")
                return self._video_processor.apply_depth(frame, ic_state.depth_pipeline)
            case _:
                raise HTTPError(400, f"Unsupported conditioning_type: {conditioning_type}")

    @contextmanager
    def _control_frame_processor(
        self, control_type: IcLoraControlType
    ) -> Iterator[Callable[[FrameArray], FrameArray]]:
        """Frame-to-guide-frame function for a ``video_control`` entry.

        A depth or pose model loads here and leaves the GPU when the block ends, so it is not
        resident during the 22B inference.
        """
        # The guide function reads the model through a list. Clearing the list drops the last reference.
        guide: Callable[[FrameArray], FrameArray]
        release: Callable[[], None]
        match control_type:
            case "canny":
                yield self._video_processor.apply_canny
                return
            case "depth":
                depth = [
                    self._pipelines.create_depth_processor(
                        str(get_existing_cp_path(self.models_dir, DEPTH_PROCESSOR_CP_ID))
                    )
                ]
                guide = lambda frame: self._video_processor.apply_depth(frame, depth[0])
                release = depth.clear
            case "pose":
                pose = [
                    self._pipelines.create_pose_processor(
                        str(get_existing_cp_path(self.models_dir, POSE_PROCESSOR_CP_ID)),
                        str(get_existing_cp_path(self.models_dir, PERSON_DETECTOR_CP_ID)),
                    )
                ]
                guide = lambda frame: self._video_processor.apply_pose(frame, pose[0])
                release = pose.clear
            case _:
                assert_never(control_type)
        try:
            yield guide
        finally:
            release()
            self._pipelines.free_unreferenced_gpu_memory()

    def _require_ic_lora_model_paths(self, conditioning_type: ConditioningType) -> tuple[Path, Path | None]:
        model_id = resolve_active_ltx_model_id(
            self.models_dir, self.state.app_settings.active_ltx_model_id
        )
        if model_id is None:
            raise HTTPError(409, "NO_DOWNLOADED_LTX_MODEL")
        if not supports(local_caps(model_id), "ic_lora"):
            raise HTTPError(
                409,
                "Built-in control IC-LoRA is not available for the active LTX model.",
                code="UNSUPPORTED_IC_LORA",
            )
        ic_loras_spec = get_ltx_model_spec(model_id).ic_loras_spec
        if ic_loras_spec is None:
            raise HTTPError(
                409,
                "Built-in control IC-LoRA is not available for the active LTX model.",
                code="UNSUPPORTED_IC_LORA",
            )
        depth_model_path: Path | None = None
        match conditioning_type:
            case "canny":
                # canny preprocessing uses apply_canny, not the depth processor — don't load it.
                lora_cp_id = ic_loras_spec.canny_cp
            case "depth":
                lora_cp_id = ic_loras_spec.depth_cp
                depth_model_path = get_existing_cp_path(self.models_dir, DEPTH_PROCESSOR_CP_ID)
            case _:
                raise HTTPError(400, f"Unsupported conditioning_type: {conditioning_type}")
        lora_path = get_existing_cp_path(self.models_dir, lora_cp_id)
        return lora_path, depth_model_path

    def extract_conditioning(self, req: IcLoraExtractRequest) -> IcLoraExtractResponse:
        video_file = Path(req.video_path)
        if not video_file.exists():
            raise HTTPError(400, f"Video not found: {req.video_path}")

        cap = self._video_processor.open_video(str(video_file))
        info = self._video_processor.get_video_info(cap)
        target_frame = int(req.frame_time * float(info["fps"]))
        frame = self._video_processor.read_frame(cap, frame_idx=target_frame)
        self._video_processor.release(cap)

        if frame is None:
            raise HTTPError(400, "Could not read frame from video")

        ic_state: ICLoraState | None = None
        if req.conditioning_type == "depth":
            lora_path, depth_model_path = self._require_ic_lora_model_paths(req.conditioning_type)
            # One-frame preview. Streaming matches Darwin and 8s+ generate on 32 GB so
            # extract does not warm a full-resident slot that generate then evicts.
            ic_state = self._pipelines.load_ic_lora(
                str(lora_path),
                str(depth_model_path),
                mode="streaming_models_loading",
            )

        result = self._build_conditioning_frame(frame, req.conditioning_type, ic_state)

        conditioning = self._video_processor.encode_frame_jpeg(result, quality=85)
        original = self._video_processor.encode_frame_jpeg(frame, quality=85)

        return IcLoraExtractResponse(
            conditioning="data:image/jpeg;base64," + base64.b64encode(conditioning).decode("utf-8"),
            original="data:image/jpeg;base64," + base64.b64encode(original).decode("utf-8"),
            conditioning_type=req.conditioning_type,
            frame_time=req.frame_time,
        )

    @staticmethod
    def _resampled_frame_count(frame_count: int, src_fps: float, target_fps: float) -> int:
        return round(frame_count / src_fps * target_fps)

    @staticmethod
    def _frame_count_after_fps_override(frame_count: int, src_fps: float, target_fps: float) -> int:
        return snap_to_frame_grid(
            IcLoraHandler._resampled_frame_count(frame_count, src_fps, target_fps), floor=1
        )

    def _resample_control_fps(
        self, control_video_path: str, frame_count: int, src_fps: float, target_fps: float
    ) -> tuple[str, int, float]:
        """Decimate the control video to target_fps so fewer frames are generated.

        Fewer frames = shorter sequence = less compute/VRAM (fits longer real-time clips),
        at the cost of choppier motion. Keeps the output duration ~constant and the frame
        count valid for the pipeline ((n-1) % 8 == 0).
        """
        new_count = self._frame_count_after_fps_override(frame_count, src_fps, target_fps)
        cap = self._video_processor.open_video(control_video_path)
        info = self._video_processor.get_video_info(cap)
        size = (int(info["width"]), int(info["height"]))
        out_path = str(self.config.outputs_dir / f"_resampled_{target_fps:g}fps_{uuid.uuid4().hex[:8]}.mp4")
        writer = self._video_processor.create_writer(out_path, fourcc="mp4v", fps=target_fps, size=size)
        for i in range(new_count):
            src_idx = min(frame_count - 1, round(i * src_fps / target_fps))
            frame = self._video_processor.read_frame(cap, frame_idx=src_idx)
            if frame is None:
                break
            writer.write(frame)
        self._video_processor.release(cap)
        self._video_processor.release(writer)
        logger.info(
            "[ic-lora] resampled control %g->%gfps: %d -> %d frames", src_fps, target_fps, frame_count, new_count
        )
        return out_path, new_count, target_fps

    @staticmethod
    def _resolve_control_values(req: IcLoraGenerateRequest, ic_lora: IcLoraCatalogItem) -> dict[str, int | str]:
        """Resolve the request's control values against the entry's declared controls.

        Missing keys fall back to each control's default; every value is validated against the
        control's options. Unknown ids in the request are ignored. The handler then maps known
        ids to behaviour (duration → frames, outpaint_* → canvas) — see the call site.
        """
        resolved: dict[str, int | str] = {}
        for c in ic_lora.controls:
            # "position_canvas" controls carry a structured value via a typed field
            # (outpaint_pads), not options — skip them here.
            if c.options is None or c.default is None:
                continue
            value = req.control_values.get(c.id, c.default)
            if value not in c.options:
                raise HTTPError(400, f"{c.id} must be one of {c.options}")
            resolved[c.id] = value
        return resolved

    @staticmethod
    def _outpaint_from_request(req: IcLoraGenerateRequest) -> OutpaintParams | None:
        """Build outpaint params from the request's per-edge pads, or None if absent."""
        p = req.outpaint_pads
        if p is None:
            return None
        return OutpaintParams(left=p.left, right=p.right, top=p.top, bottom=p.bottom)

    def _complete_or_drop_cancelled(self, output_path: Path, *, ran: JobEnvelope | None) -> None:
        # Denoiser interrupt cannot abort VAE decode / ffmpeg; a Stop after the last
        # denoise step still finishes encode, then this check drops the file.
        if self._generation.is_generation_cancelled():
            output_path.unlink(missing_ok=True)
            raise GenerationCancelledError()
        self._generation.update_progress("complete", 100, 1, 1)
        self._generation.complete_generation(str(output_path), ran=ran)

    def generate_local_reserved(
        self,
        req: IcLoraGenerateRequest,
        *,
        generation_id: str,
        output_path: Path,
        local_model_id: LTXLocalModelId,
        seed: int,
        explore_generation: bool = True,
        explicit_generation_seed: bool = False,
        high_res: bool = False,
        allowance: RiskAllowance | None = None,
        panel_paths: Sequence[str] = (),
    ) -> IcLoraGenerateResponse:
        """Local-only Home/Remote entry. The checkpoint id is required.

        The caller already holds the GPU slot and allocated ``output_path``.
        ``panel_paths`` are the ordered images of an image-list recipe. Automatic enhancement
        shows them to the rewrite. The request has no field for them, so a route cannot set them.
        ``high_res`` runs the LoRA as tiles in windows, for the 1440p and 2160p cells.
        GenSpace ``generate()`` still reaches ``_generate_ic_lora`` under
        ``reserved_generation_start()``, where ``local_model_id`` may be unset.
        """
        self._require_local_generation_possible()
        assert req.ic_lora_id is not None
        return self._generate_ic_lora(
            req,
            generation_id=generation_id,
            output_path=output_path,
            local_model_id=local_model_id,
            seed=seed,
            explore_generation=explore_generation,
            explicit_generation_seed=explicit_generation_seed,
            high_res=high_res,
            allowance=allowance,
            panel_paths=panel_paths,
        )

    def _generate_ic_lora(
        self,
        req: IcLoraGenerateRequest,
        *,
        generation_id: str,
        output_path: Path,
        local_model_id: LTXLocalModelId | None,
        seed: int,
        explore_generation: bool,
        explicit_generation_seed: bool = False,
        high_res: bool = False,
        allowance: RiskAllowance | None = None,
        panel_paths: Sequence[str] = (),
    ) -> IcLoraGenerateResponse:
        assert req.ic_lora_id is not None
        ic_lora = self._catalog.get_ic_lora(req.ic_lora_id)
        if ic_lora is None:
            raise HTTPError(404, "UNKNOWN_IC_LORA")
        if not catalog_item_visible_for_installed_ltx(ic_lora.supported_models, self.models_dir):
            raise HTTPError(422, "IC_LORA_UNSUPPORTED_MODEL")
        if (
            local_model_id is not None
            and not ic_lora.supports_family(ltx_catalog_family_for_model(local_model_id))
        ):
            raise HTTPError(422, "IC_LORA_UNSUPPORTED_MODEL")
        # IC-LoRA catalog entries always define a driving input (base field is optional for plain LoRAs).
        assert ic_lora.input is not None, "IC-LoRA ic_lora is missing its input spec"
        has_prompt = bool(req.prompt.strip())
        # A prompt is required unless this catalog entry opts into promptless runs (e.g. outpainting).
        if not has_prompt and not ic_lora.allows_empty_prompt:
            raise HTTPError(400, "Prompt is required for this IC-LoRA")
        if not req.input_path or not Path(req.input_path).exists():
            raise HTTPError(400, f"Input not found: {req.input_path}")
        # The input file must match the kind this IC-LoRA drives on (image vs video) — fail fast
        # with a clean 400 instead of a deep pipeline 500 when e.g. an image is fed to a video entry.
        _validate_input_kind(req.input_path, ic_lora.input.kind)
        # Reference images (when the entry allows them) must exist — fail fast with a clean 400
        # rather than surfacing a deep pipeline error later.
        if ic_lora.reference_image_required and not req.images:
            raise HTTPError(400, "A reference image is required for this IC-LoRA")
        if ic_lora.allows_reference_image:
            for img in req.images:
                if not Path(img.path).exists():
                    raise HTTPError(400, f"Reference image not found: {img.path}")
        if req.variant_id is not None:
            variant = ic_lora.download.resolve_variant(req.variant_id)
            if variant is None:
                raise HTTPError(404, "UNKNOWN_DOWNLOAD_VARIANT")
            candidate_filenames = [variant.filename]
        else:
            candidate_filenames = ic_lora.download.candidate_filenames()
        lora_path = find_installed_ic_lora_path(
            self.models_dir, ic_lora.id, candidate_filenames, ic_lora.requires_checkpoints
        )
        if lora_path is None:
            raise HTTPError(409, "IC_LORA_NOT_DOWNLOADED")

        if req.loop_fps is not None and req.loop_fps not in IC_LORA_SUPPORTED_FPS:
            raise HTTPError(400, f"loop_fps must be one of {IC_LORA_SUPPORTED_FPS}")
        control_values = self._resolve_control_values(req, ic_lora)
        duration = int(control_values.get("duration", 5))
        if req.loop_seconds is not None and ic_lora.is_sheet_recipe():
            duration = req.loop_seconds
        requested_frames = compute_num_frames(duration, catalog_output_fps(ic_lora, req.loop_fps))

        # IC-LoRA defaults, with any explicit UI override (e.g. skip_stage_2) applied on top.
        s = _resolve_settings(req, ic_lora.default_settings, high_res=high_res)
        stage_mode = s.stage_mode
        # Optional reference image (e.g. a photoreal seed for 3D-render): conditions the
        # first frame. Only honoured when the catalog entry opts in; ignored otherwise.
        reference_images: list[ImageConditioningInput] = (
            [
                ImageConditioningInput(
                    path=img.path,
                    frame_idx=_reference_frame_idx(ic_lora, img),
                    strength=float(img.strength),
                )
                for img in req.images
            ]
            if ic_lora.allows_reference_image
            else []
        )
        stills = sum(1 for img in reference_images if img.frame_idx < 0)
        logger.info("[ic-lora] IC-LoRA generation started (ic_lora=%s)", ic_lora.id)
        control: MediaArtifact | None = None
        try:
            probe_w, probe_h, probe_frames = self._probe_ic_lora_source(
                req.input_path, ic_lora.input.kind, requested_frames
            )
            width, height, resolution_factor = ic_lora_output_canvas(
                skip_stage_2=s.skip_stage_2,
                resolution_factor=s.resolution_factor,
                input_width=probe_w,
                input_height=probe_h,
                resolution=req.resolution,
                stage_2_ic_lora=s.stage_2_ic_lora,
                tiling=s.tiling,
            )
            admitted = self._ic_lora_job(
                width,
                height,
                frame_count=probe_frames,
                stage_mode=stage_mode,
                resolution_factor=resolution_factor,
                ic_lora_id=req.ic_lora_id,
                allow_evict_retry=True,
                allowance=allowance,
                stills=stills,
                tiling=s.tiling,
            )
            job = admitted.load
            run_tiling = picked_carry_tiling(s.tiling, admitted.priced)
            nag_decision = self._decide_nag(req, model_id=local_model_id)
            resolved_prompt = self._prompt_enhancement.resolve_for_generation(
                req.prompt,
                provenance=req.prompt_provenance,
                generation_seed=seed,
                explicit_generation_seed=explicit_generation_seed,
                image_path=req.input_path if explore_generation and ic_lora.is_sheet_recipe() else None,
                ic_lora=ic_lora,
                local_model_id=local_model_id,
                explore_generation=explore_generation,
                # Home only. Gen Space keeps its text-only rewrite (it has manual Enhance).
                source_video=(
                    SourceVideo(
                        path=req.input_path,
                        # The look image of an IC-LoRA that takes one.
                        reference_image_path=(
                            req.images[0].path
                            if ic_lora.allows_reference_image and req.images
                            else None
                        ),
                    )
                    if explore_generation and ic_lora.input.kind == "video"
                    else None
                ),
                panel_paths=panel_paths if explore_generation and ic_lora.is_sheet_recipe() else (),
                negative_prompt=nag_decision.enhancer_negative,
            )
            prompt = resolved_prompt.prompt
            enhance_via_api = resolved_prompt.enhance_via_api

            # start_generation requires gpu_slot. Load first so preprocess is not stuck
            # on phase=starting (Stop only takes effect once the job is Running).
            ic_state = self._pipelines.load_ic_lora(
                str(lora_path),
                None,
                s.lora_strength,
                mode=job.mode,
                ltx_model_id=local_model_id,
                stage_2_ic_lora=s.stage_2_ic_lora,
                tiling=run_tiling,
            )
            self._generation.start_generation(generation_id, ran=admitted.envelope)
            self._generation.update_progress("loading_model", 5, 0, 1)

            input_artifact = MediaArtifact(path=req.input_path, kind=ic_lora.input.kind)
            with (
                self._control_frame_processor(
                    cast(IcLoraControlType, control_values[IC_LORA_CONTROL_TYPE_ID])
                )
                if ic_lora.is_video_control_recipe()
                else nullcontext(None)
            ) as control_frame:
                ctx = PreprocessingContext(
                    control_frame=control_frame,
                    num_frames=requested_frames,
                    loop_fps=req.loop_fps,
                    outputs_dir=self.config.outputs_dir,
                    video_processor=self._video_processor,
                    outpaint=self._outpaint_from_request(req),
                    on_frame=self._generation.raise_if_cancelled,
                )
                control = run_preprocessing(ic_lora.preprocessing, input_artifact, ctx)
            control_video_path = control.path

            cap = self._video_processor.open_video(control_video_path)
            info = self._video_processor.get_video_info(cap)
            self._video_processor.release(cap)
            input_width, input_height = int(info["width"]), int(info["height"])

            if control.frame_count:
                # Preprocessed control video (image_to_frames at the chosen duration, or an
                # outpaint canvas). Snap to the pipeline's grid: image_to_frames already coerces,
                # but _outpaint_canvas returns its raw written count, which can be off-grid.
                frame_count, fps = snap_to_frame_grid(control.frame_count, floor=9), control.fps or 24.0
            else:
                # Video IC-LoRAs: transform the whole input clip — match its length and fps,
                # snapped to the pipeline's (n - 1) % 8 == 0 grid (not the duration control).
                src_frames = int(info["frame_count"]) or requested_frames
                frame_count = snap_to_frame_grid(src_frames, floor=9)
                fps = float(info["fps"]) or 24.0

            if (input_width, input_height, frame_count) != (probe_w, probe_h, probe_frames):
                width, height, resolution_factor = ic_lora_output_canvas(
                    skip_stage_2=s.skip_stage_2,
                    resolution_factor=s.resolution_factor,
                    input_width=input_width,
                    input_height=input_height,
                    resolution=req.resolution,
                    stage_2_ic_lora=s.stage_2_ic_lora,
                    tiling=s.tiling,
                )
                admitted = self._ic_lora_job(
                    width,
                    height,
                    frame_count=frame_count,
                    stage_mode=stage_mode,
                    resolution_factor=resolution_factor,
                    ic_lora_id=req.ic_lora_id,
                    allowance=allowance,
                    stills=stills,
                    tiling=s.tiling,
                )
                rerun_tiling = picked_carry_tiling(s.tiling, admitted.priced)
                reload = admitted.load.mode != job.mode or rerun_tiling != run_tiling
                job = admitted.load
                run_tiling = rerun_tiling
                if reload:
                    ic_state = self._pipelines.load_ic_lora(
                        str(lora_path),
                        None,
                        s.lora_strength,
                        mode=job.mode,
                        ltx_model_id=local_model_id,
                        stage_2_ic_lora=s.stage_2_ic_lora,
                        tiling=run_tiling,
                    )

            self._text.prepare_text_encoding(
                prompt,
                enhance_prompt=enhance_via_api,
                model_id=local_model_id,
            )
            nag = self._resolve_nag(req, model_id=local_model_id)
            self._generation.raise_if_cancelled()

            self._generation.update_progress("inference", 15, 0, 1)
            with log_heartbeat("ic-lora inference"):
                ic_state.pipeline.generate(
                    prompt=prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=frame_count,
                    frame_rate=fps,
                    images=reference_images,
                    video_conditioning=[(control_video_path, s.conditioning_strength)],
                    output_path=str(output_path),
                    skip_stage_2=s.skip_stage_2,
                    use_lora_in_stage_2=s.use_lora_in_stage_2,
                    resolution_factor=resolution_factor,
                    chunk_pixel_frames=job.chunk_pixel_frames,
                    source_audio_path=(
                        req.input_path
                        if s.audio_mode == "source" and ic_lora.input.kind == "video"
                        else None
                    ),
                    mute_audio=s.audio_mode == "off",
                    conditioning_mask_path=control.mask_path,
                    nag=nag,
                )
            self._complete_or_drop_cancelled(output_path, ran=admitted.envelope)
            return IcLoraGenerateCompleteResponse(status="complete", video_path=str(output_path))
        except HTTPError:
            self._generation.fail_generation("IC-LoRA generation failed")
            raise
        except ValueError as exc:
            # run_preprocessing raises ValueError for user-fixable input (bad pads, unreadable
            # media, >+100% outpaint) — surface it as a 400 so the user sees what to fix.
            self._generation.fail_generation(str(exc))
            raise HTTPError(400, str(exc)) from exc
        except Exception as exc:
            if is_cancel_exception(exc):
                logger.info("Generation cancelled by user")
                return IcLoraGenerateCancelledResponse(status="cancelled")
            self._generation.fail_generation(str(exc))
            raise HTTPError(500, f"Generation error: {exc}") from exc
        finally:
            # The guide clip, the looped frames and the outpaint canvas are full-size temporary
            # files. Every exit removes them: success, cancel and failure.
            if control is not None:
                self._drop_preprocess_artifact(control.path, req.input_path)
                self._drop_preprocess_artifact(control.mask_path, req.input_path)
            self._text.clear_api_embeddings()

    def _decide_nag(
        self, req: IcLoraGenerateRequest, *, model_id: LTXLocalModelId | None
    ) -> NagDecision:
        with self._lock:
            default_nag_scale = self.state.app_settings.nag_scale
        return decide_nag(
            req.negative_prompt,
            os.environ,
            scale=req.nag_scale,
            alpha=req.nag_alpha,
            tau=req.nag_tau,
            default_scale=default_nag_scale,
            local_encoding=self._text.should_use_local_encoding(model_id),
            compiled=False,
        )

    def _resolve_nag(
        self, req: IcLoraGenerateRequest, *, model_id: LTXLocalModelId | None
    ) -> NagConfig | None:
        return log_nag_decision(
            logger, "ic-lora", self._decide_nag(req, model_id=model_id)
        )

    def preflight(self, req: IcLoraPreflightRequest) -> IcLoraPreflightResponse:
        """Advisory sizing verdicts for a catalog input, before Generate.

        Mirrors the generate probe check (source pixels → output canvas →
        fence + spatial + budget) without preprocessing, weights, or the GPU
        slot. Raises like generate on an unknown entry, a model the install
        can't run, or an unreadable/mismatched input; per-duration failures
        never raise. Skips the download/prompt checks — sizing advice is
        useful before weights exist. One free-VRAM sample for all cells.
        """
        ic_lora = self._catalog.get_ic_lora(req.ic_lora_id)
        if ic_lora is None:
            raise HTTPError(404, "UNKNOWN_IC_LORA")
        if not catalog_item_visible_for_installed_ltx(ic_lora.supported_models, self.models_dir):
            raise HTTPError(422, "IC_LORA_UNSUPPORTED_MODEL")
        assert ic_lora.input is not None, "IC-LoRA ic_lora is missing its input spec"
        if not req.input_path or not Path(req.input_path).exists():
            raise HTTPError(400, f"Input not found: {req.input_path}")
        _validate_input_kind(req.input_path, ic_lora.input.kind)

        base = ic_lora.default_settings
        # Same overlay as generate: a tiled entry always runs one stage.
        skip_stage_2 = (
            True
            if base.tiling is not None
            else base.skip_stage_2 if req.skip_stage_2 is None else req.skip_stage_2
        )
        stage_2_ic_lora = base.stage_2_ic_lora and not skip_stage_2
        stage_mode = ic_lora_stage_mode(skip_stage_2=skip_stage_2, stage_2_ic_lora=stage_2_ic_lora)
        resolution_factor = (
            ic_lora.default_settings.resolution_factor
            if req.resolution_factor is None
            else req.resolution_factor
        )
        kind = ic_lora.input.kind
        fps = catalog_output_fps(ic_lora)
        # Display on total: advisory cells must not grey total-fitting shapes
        # just because free is momentarily depressed; generate stays authoritative.
        job_filter = self._local_job_filter(total_only=True)

        # Probe frames are the requested grid for images (the canvas ignores
        # frames, and each cell grids its own) and the snapped clip for video.
        # The 121 stand-in is discarded for images, ignored for video.
        if kind == "image":
            probe_w, probe_h, _ = self._probe_ic_lora_source(
                req.input_path, kind, 121
            )
            # Frame math runs on whole catalog seconds; ceiling a fractional
            # request verdicts the covering grid (conservative, exact for ints).
            targets = [
                (d, compute_num_frames(math.ceil(d), fps)) for d in req.durations
            ]
            # Images have no source frames — the per-cell grids above are the
            # only frame counts; the source reports dims only.
            source_frames: int | None = None
        else:
            probe_w, probe_h, probe_frames = self._probe_ic_lora_source(
                req.input_path, kind, 121
            )
            # Video entries transform the whole clip: requested durations are
            # ignored, one cell verdicts the clip at budgeted seconds.
            targets = [(probe_frames / fps, probe_frames)]
            source_frames = probe_frames
        width, height, factor = ic_lora_output_canvas(
            skip_stage_2=skip_stage_2,
            resolution_factor=resolution_factor,
            input_width=probe_w,
            input_height=probe_h,
            resolution=req.resolution,
            stage_2_ic_lora=stage_2_ic_lora,
        )
        cells: list[IcLoraPreflightCell] = []
        for duration, frames in targets:
            admission = self._decide_ic_lora_admission(
                width,
                height,
                frame_count=frames,
                stage_mode=stage_mode,
                resolution_factor=factor,
                job_filter=job_filter,
                ic_lora_id=req.ic_lora_id,
                stills=catalog_preview_stills(ic_lora),
                tiling=base.tiling,
            )
            if isinstance(admission, Admitted):
                risk = admission.risk
                message = None if risk is None else risk.message
                code = None
            else:
                risk = admission.bet
                message, code = dropped_duration_message_and_code(admission.reject, job_filter)
            cells.append(
                IcLoraPreflightCell(
                    duration=duration,
                    ok=isinstance(admission, Admitted),
                    message=message,
                    code=code,
                    tier=None if risk is None else risk.tier,
                    over_gib=None if risk is None else risk.over_gib,
                    over_ram_gb=None if risk is None else risk.over_ram_gb,
                    shortfall_kind=None if risk is None else risk.shortfall_kind,
                    readout=None if risk is None else risk.readout,
                )
            )
        fits = sum(1 for cell in cells if cell.ok)
        logger.info(
            "[ic-lora] preflight %s (%dx%d %s): %d/%d durations fit",
            ic_lora.id,
            probe_w,
            probe_h,
            kind,
            fits,
            len(cells),
        )
        return IcLoraPreflightResponse(
            source=IcLoraPreflightSource(
                kind=kind,
                width=probe_w,
                height=probe_h,
                frames=source_frames,
                fps=float(fps),
            ),
            cells=cells,
            budgetMode=job_filter.budget_mode,
        )

    def preflight_queued_recipe_job(
        self,
        *,
        item: IcLoraCatalogItem,
        video: AssetRecord,
        stored: IcLoraRecipeStoredParams,
        source_fps: float | None = None,
        upscale: bool = False,
        cutout: bool = False,
    ) -> GenerationRisk | None:
        """The risk of ``preflight_queued_recipe_plan``. Raises HTTPError like it."""
        return self.preflight_queued_recipe_plan(
            item=item, video=video, stored=stored, source_fps=source_fps, upscale=upscale, cutout=cutout
        ).risk

    def preflight_queued_recipe_plan(
        self,
        *,
        item: IcLoraCatalogItem,
        video: AssetRecord,
        stored: IcLoraRecipeStoredParams,
        source_fps: float | None = None,
        upscale: bool = False,
        cutout: bool = False,
    ) -> RecipeRunPlan:
        """Create-time mirror of IcLoraRecipeExecutor's budget gate.

        Assembles the same generate request execute would (same closest-match
        aspect, same explicit-resolution canvas, estimated resampled source frames)
        and runs it through the same load gate (``recipe_run_frames``), so an
        over-budget create 422s here with the refusal execute would raise
        instead of enqueueing a job that fails in the runner. A job admitted on
        a bet returns the bet's annotation instead. Raises HTTPError;
        the plan carries no risk when the job loads clean.

        Only the probe check is mirrored, not the post-preprocess re-check.
        Both price the resampled frames and the center-cropped canvas. The
        create-time count comes from the stored frame count and resample math,
        and the executor reads the file it cropped, so the two counts can differ
        by a frame. Without ``source_fps`` the probe prices the source rate
        (legacy callers).
        """
        assert item.input is not None, "IC-LoRA recipe entry is missing its input spec"
        if video.metadata.mediaType != "video":
            raise HTTPError(422, "INVALID_VIDEO_ASSET", code="INVALID_GENERATION_SPEC")
        request = IcLoraGenerateRequest(
            ic_lora_id=item.id,
            input_path=video.path,
            prompt=stored.prompt,
            conditioning_type="custom",
        )
        duration = int(self._resolve_control_values(request, item).get("duration", 5))
        requested_frames = compute_num_frames(duration, catalog_output_fps(item))
        probe_w, probe_h, probe_frames = self._probe_ic_lora_source(
            video.path, item.input.kind, requested_frames, raw_frames=True
        )
        # Execute decimates the clip to the stored rate before its probe
        # (center_crop_video output_fps); price the same frames here or
        # resample-down jobs over-refuse. Same 1e-3 guard as the executor:
        # a rate at/above the source is a no-op. Image inputs carry the
        # duration frames, which have no source rate to decimate from.
        if (
            item.input.kind == "video"
            and source_fps is not None
            and stored.fps + 1e-3 < source_fps
        ):
            probe_frames = self._resampled_frame_count(probe_frames, source_fps, stored.fps)
        return self.recipe_run_plan(
            ic_lora_id=item.id,
            source_width=probe_w,
            source_height=probe_h,
            stored=stored,
            clip_frames=probe_frames,
            cutout=cutout,
            upscale=upscale,
        )

    def _recipe_canvas(
        self,
        item: IcLoraCatalogItem,
        *,
        local_model_id: LTXLocalModelId,
        resolution: IcLoraRecipeResolution,
        source_width: int,
        source_height: int,
        audio_mode: IcLoraAudioMode,
        lora_strength: float,
        cutout: bool,
        upscale: bool,
        tile_overlap: float | None = None,
        tile_size: IcLoraTileSize | None = None,
        carry_frames: int | None = None,
        max_window_frames: int | None = None,
    ) -> tuple[int, int, float, IcLoraSettings]:
        """Canvas, resolution factor and settings a Home recipe runs for a source of this size.

        One home for create, execute and the memory advice, so the three price one canvas. The
        tiling arguments are the Advanced choices of the run. None keeps the plan of the cell.
        """
        aspect = cast(
            LTXVideoGenAspectRatio,
            find_closest_aspect_ratio_option(
                source_width, source_height, get_args(LTXVideoGenAspectRatio)
            ),
        )
        high_res = high_res_cell(resolution)
        if high_res is not None:
            explicit_width, explicit_height = high_res_stage1(
                source_width, source_height, high_res, upscale=upscale
            )
        else:
            try:
                explicit_width, explicit_height = budget_pixels(
                    local_caps(local_model_id),
                    resolution,
                    aspect,
                    mode="video",
                )
            except KeyError as exc:
                raise HTTPError(400, "INVALID_LOCAL_RESOLUTION") from exc
        request = IcLoraGenerateRequest(
            ic_lora_id=item.id,
            input_path="",
            prompt="",
            conditioning_type="custom",
            resolution=TargetResolution(width=explicit_width, height=explicit_height),
            audio_mode=audio_mode,
            lora_strength=lora_strength,
            tile_overlap=tile_overlap,
            tile_size=tile_size,
            carry_frames=carry_frames,
            max_window_frames=max_window_frames,
        )
        s = _resolve_settings(request, item.default_settings, high_res=high_res is not None)
        input_width, input_height = (
            center_cropped_size(source_width, source_height, aspect)
            if not cutout
            else (source_width, source_height)
        )
        width, height, resolution_factor = ic_lora_output_canvas(
            skip_stage_2=s.skip_stage_2,
            resolution_factor=s.resolution_factor,
            input_width=input_width,
            input_height=input_height,
            resolution=request.resolution,
            stage_2_ic_lora=s.stage_2_ic_lora,
            tiling=s.tiling,
        )
        return width, height, resolution_factor, s

    def preflight_recipe_tile_sizes(
        self,
        *,
        item: IcLoraCatalogItem,
        video: AssetRecord,
        stored: IcLoraRecipeStoredParams,
        source_fps: float | None,
        upscale: bool,
        cutout: bool,
        tile_sizes: Sequence[IcLoraTilePreset],
        max_windows: Sequence[int],
        carries: Sequence[int] = (),
    ) -> IcLoraTilePreflightResponse:
        """Advisory verdict per tile size and per window, from the create-time gate of each one.

        A refusal is returned in its cell, never raised, so the form can disable the option
        and show why. A job the budget mode admits on a bet returns the bet in its cell.
        """
        budget_mode = self._local_job_filter(total_only=True).budget_mode

        def verdict(candidate: IcLoraRecipeStoredParams) -> IcLoraPreflightVerdict:
            try:
                risk = self.preflight_queued_recipe_job(
                    item=item,
                    video=video,
                    stored=candidate,
                    source_fps=source_fps,
                    upscale=upscale,
                    cutout=cutout,
                )
            except HTTPError as exc:
                return IcLoraPreflightVerdict(ok=False, message=exc.detail, code=exc.code)
            if risk is None:
                return IcLoraPreflightVerdict(ok=True)
            return IcLoraPreflightVerdict(
                ok=True,
                message=risk.message,
                tier=risk.tier,
                over_gib=risk.over_gib,
                over_ram_gb=risk.over_ram_gb,
                shortfall_kind=risk.shortfall_kind,
                readout=risk.readout,
            )

        def auto_plan(**unset: None) -> RecipeRunPlan | None:
            try:
                return self.preflight_queued_recipe_plan(
                    item=item,
                    video=video,
                    stored=stored.model_copy(update=unset),
                    source_fps=source_fps,
                    upscale=upscale,
                    cutout=cutout,
                )
            except HTTPError:
                return None  # Auto is refused: the cells carry the refusal, and the label names none.

        defaults = _tiling_defaults(item, stored)
        # Each Auto reads the form as it is: Auto chunk length keeps the carry the form holds, and
        # Auto carry keeps the chunk length. A control that is on Auto leaves the other pins alone.
        auto_window = auto_plan(maxWindowFrames=None)
        auto_carry = auto_plan(carryFrames=None)
        if auto_window is not None:
            defaults = defaults.model_copy(
                update={
                    "auto": IcLoraAutoSinglePass()
                    if auto_window.window is None
                    else IcLoraAutoWindowed(frames=auto_window.window)
                }
            )
        if auto_carry is not None:
            defaults = defaults.model_copy(update={"carryFrames": auto_carry.carry})

        return IcLoraTilePreflightResponse(
            cells=[
                IcLoraTilePreflightCell(
                    tileSize=tile_size,
                    **verdict(
                        stored.model_copy(update={"tilingSize": None if tile_size == "standard" else tile_size})
                    ).model_dump(),
                )
                for tile_size in tile_sizes
            ],
            maxWindowCells=[
                IcLoraMaxWindowPreflightCell(
                    maxWindowFrames=window,
                    **verdict(stored.model_copy(update={"maxWindowFrames": window})).model_dump(),
                )
                for window in max_windows
            ],
            carryCells=[
                IcLoraCarryPreflightCell(
                    carryFrames=carry, **verdict(stored.model_copy(update={"carryFrames": carry})).model_dump()
                )
                for carry in carries
            ],
            defaults=defaults,
            budgetMode=budget_mode,
        )

    def recipe_run_frames(
        self,
        *,
        ic_lora_id: str,
        source_width: int,
        source_height: int,
        stored: IcLoraRecipeStoredParams,
        clip_frames: int,
        cutout: bool = False,
        upscale: bool = False,
        allowance: RiskAllowance | None = None,
    ) -> tuple[int, GenerationRisk | None]:
        """The frame count and the risk of ``recipe_run_plan``."""
        plan = self.recipe_run_plan(
            ic_lora_id=ic_lora_id,
            source_width=source_width,
            source_height=source_height,
            stored=stored,
            clip_frames=clip_frames,
            cutout=cutout,
            upscale=upscale,
            allowance=allowance,
        )
        return plan.frames, plan.risk

    def recipe_run_plan(
        self,
        *,
        ic_lora_id: str,
        source_width: int,
        source_height: int,
        stored: IcLoraRecipeStoredParams,
        clip_frames: int,
        cutout: bool = False,
        upscale: bool = False,
        allowance: RiskAllowance | None = None,
    ) -> RecipeRunPlan:
        """Frame count a queued Home recipe runs for a clip of ``clip_frames`` frames, its risk, and its window.

        The count snaps UP, so no frame is dropped: the recipe repeats the last frame to
        reach it and cuts the output back. When that count is refused or carries a risk, and
        the count snapped DOWN is strictly better, the job runs the smaller count, and the
        pipeline drops the last frames as it did before. From best to worst: clean, an
        ``unmeasured`` risk (a fence that moves with the frame count), an ``over`` risk, a
        422. Brave mode returns a risk in place of the 422. When the snapped-down count
        is no better, or is refused itself, the snapped-up count runs with its risk, or its 422
        propagates. The check asks "will it ever fit" (``total_only``). The create-time
        preflight and the run use this one method, but they count the clip frames in two ways
        (stored count with resample math, and the cropped file itself), so the counts can
        differ by a frame. The run gates the count it reads from the file.

        An executor passes ``allowance`` from the stored annotation, so a mode change never
        gates a queued job again.

        ``source_width`` and ``source_height`` are the size of the file the user picked. The
        run reads the clip after ``center_crop_video``, so the canvas comes from the cropped
        size. The cutout recipe keeps the whole frame and always pads up, so it uses the
        uncropped size and has no snapped-down fallback.
        """
        item = self._catalog.get_ic_lora(ic_lora_id)
        if item is None:
            raise HTTPError(404, "UNKNOWN_IC_LORA")
        job_filter = self._local_job_filter(total_only=True)
        local_model_id = resolve_job_local_model_id(self.models_dir, stored.model)
        width, height, resolution_factor, s = self._recipe_canvas(
            item,
            local_model_id=local_model_id,
            resolution=stored.resolution,
            source_width=source_width,
            source_height=source_height,
            audio_mode=stored.audioMode,
            lora_strength=stored.loras[0].scale,
            cutout=cutout,
            upscale=upscale,
            tile_overlap=stored.tilingOverlap,
            tile_size=stored.tilingSize,
            carry_frames=stored.carryFrames,
            max_window_frames=stored.maxWindowFrames,
        )
        snapped_up = snap_up_to_frame_grid(clip_frames, floor=9)
        snapped_down = snap_to_frame_grid(clip_frames, floor=9)
        stills = catalog_preview_stills(item)

        def decide(frame_count: int) -> Admission:
            # Queued admission gates on total ("will it ever fit"): a running job
            # depresses free, and the queue exists to wait for a free slot. The
            # optional look still is priced as present, so execute never costs more.
            return self._decide_ic_lora_admission(
                width,
                height,
                frame_count=frame_count,
                stage_mode=s.stage_mode,
                resolution_factor=resolution_factor,
                job_filter=job_filter,
                ic_lora_id=item.id,
                stills=stills,
                tiling=s.tiling,
                allowance=allowance,
            )

        def badness(admission: Admission) -> int:
            if isinstance(admission, Refused):
                return 3
            if admission.risk is None:
                return 0
            return 1 if admission.risk.tier == "unmeasured" else 2

        def finish(frame_count: int, admission: Admission) -> RecipeRunPlan:
            if isinstance(admission, Refused):
                message, code = dropped_duration_message_and_code(admission.reject, job_filter)
                raise HTTPError(422, message, code=code)
            return RecipeRunPlan(
                frames=frame_count,
                risk=admission.risk,
                window=admission.load.chunk_pixel_frames,
                carry=ic_lora_carry_frames(picked_carry_tiling(s.tiling, admission.priced)),
            )

        can_snap_down = not cutout and snapped_down < snapped_up
        up_admission = decide(snapped_up)
        up_badness = badness(up_admission)
        if up_badness == 0 or not can_snap_down:
            return finish(snapped_up, up_admission)
        down_admission = decide(snapped_down)
        if badness(down_admission) >= up_badness:
            # No better count. Keep the snapped-up count, so no frame drops: it runs with its
            # risk, or its refusal is the answer. A refused snapped-down count never
            # replaces a runnable snapped-up count.
            return finish(snapped_up, up_admission)
        if isinstance(up_admission, Refused):
            reason, _ = dropped_duration_message_and_code(up_admission.reject, job_filter)
        else:
            reason = f"{up_admission.risk.tier} risk" if up_admission.risk is not None else ""
        logger.warning(
            "[ic-lora] %d frames over budget (%s): running %d, dropping the last %d",
            snapped_up, reason, snapped_down, clip_frames - snapped_down,
        )
        return finish(snapped_down, down_admission)

    def recipe_memory(
        self, recipe_id: str, req: IcLoraRecipeMemoryRequest
    ) -> IcLoraRecipeMemoryResponse:
        """Memory advice for a Home video recipe form. Never gates; create stays authoritative.

        Before a clip is picked, each offered cell reports the longest clip whose job loads, for
        a 16:9 clip of that cell's size. With a clip, the selected cell reports its ask for that
        clip, at the frame count ``recipe_run_frames`` would run.
        """
        recipe = get_ic_lora_recipe(recipe_id)
        item = None if recipe is None else self._catalog.get_ic_lora(recipe.catalog_id)
        if recipe is None or recipe.kind != "video" or item is None:
            raise HTTPError(404, "UNKNOWN_IC_LORA")
        job_filter = self._local_job_filter(total_only=True)
        local_model_id = resolve_job_local_model_id(self.models_dir, req.model)
        base = item.default_settings
        high_res_runs = (
            high_res_flag_satisfied(item.id, flag_enabled=self._high_res_enabled())
            and base.stage_mode is not IcLoraStageMode.STAGE_2_LORA
        )

        def price(
            resolution: IcLoraRecipeResolution, width: int, height: int, frames: int
        ) -> PricedJob | None:
            if high_res_cell(resolution) is not None and not high_res_runs:
                return None
            try:
                canvas_w, canvas_h, factor, s = self._recipe_canvas(
                    item,
                    local_model_id=local_model_id,
                    resolution=resolution,
                    source_width=width,
                    source_height=height,
                    audio_mode=req.audioMode or base.audio_mode,
                    lora_strength=base.lora_strength if req.scale is None else req.scale,
                    cutout=recipe.cutout,
                    upscale=recipe.upscales_to_cell,
                    tile_overlap=req.tilingOverlap,
                    tile_size=req.tilingSize,
                    carry_frames=req.carryFrames,
                    max_window_frames=req.maxWindowFrames,
                )
            except HTTPError:
                return None
            return self._price_ic_lora(
                canvas_w,
                canvas_h,
                frame_count=frames,
                stage_mode=s.stage_mode,
                resolution_factor=factor,
                job_filter=job_filter,
                ic_lora_id=item.id,
                stills=catalog_preview_stills(item),
                tiling=s.tiling,
            )

        def loads(priced: PricedJob | None) -> bool:
            if priced is None:
                return False
            return admits_within_device_memory(priced, job_filter.budget_mode)

        # A clip at the cap snaps down to run when its snapped-up count does not load.
        full_frames = snap_to_frame_grid(MAX_IC_LORA_INPUT_FRAMES, floor=9)
        grid = range(snap_up_to_frame_grid(MAX_IC_LORA_INPUT_FRAMES, floor=9), 8, -8)
        cells: list[IcLoraCellFit] = []
        for resolution in dict.fromkeys(req.cells):
            width, height = _STAND_IN_SOURCE[resolution]
            longest = next(
                (frames for frames in grid if loads(price(resolution, width, height, frames))),
                0,
            )
            cells.append(
                IcLoraCellFit(
                    resolution=resolution,
                    max_frames=min(longest, MAX_IC_LORA_INPUT_FRAMES),
                    full=longest >= full_frames,
                )
            )

        selected = None
        if req.source is not None:
            source = req.source
            up = price(
                req.resolution,
                source.width,
                source.height,
                snap_up_to_frame_grid(source.frames, floor=9),
            )
            down = (
                None
                if recipe.cutout
                else price(
                    req.resolution,
                    source.width,
                    source.height,
                    snap_to_frame_grid(source.frames, floor=9),
                )
            )
            chosen = up if loads(up) or not loads(down) else down
            if chosen is not None:
                selected = job_memory_ask(
                    chosen,
                    route=budget_route("iclora", job_filter.profile),
                    system=job_filter.profile.system,
                    machine_factor=(
                        job_filter_machine_factor(job_filter) if loads(chosen) else None
                    ),
                    conv_vae=job_filter.conv_vae,
                )
        return IcLoraRecipeMemoryResponse(
            memory=memory_capacity(job_filter),
            max_input_frames=MAX_IC_LORA_INPUT_FRAMES,
            cells=cells,
            selected=selected,
        )

    def preflight_queued_synthesized_job(
        self,
        *,
        recipe: IcLoraRecipe,
        item: IcLoraCatalogItem,
        stored: IcLoraSynthesizedStoredParams,
    ) -> GenerationRisk | None:
        """Create-time mirror of a synthesized IC-LoRA executor's budget gate.

        The control file is built at the denoise size, so the canvas and the
        frames come from the stored params alone. No source file is probed, and
        the frame count is already on the frame grid, so nothing snaps.
        """
        canvas_w, canvas_h = denoise_canvas(
            resolve_job_local_model_id(self.models_dir, stored.model), stored, name=item.name
        )
        if recipe.kind == "video":
            raise ValueError(f"IC-LoRA recipe '{recipe.recipe_id}' has no synthesized control file")
        request = SYNTHESIZED_IC_LORA_EXECUTORS[recipe.kind].generate_request(
            recipe, stored, (canvas_w, canvas_h)
        )
        s = _resolve_settings(request, item.default_settings)
        width, height, resolution_factor = ic_lora_output_canvas(
            skip_stage_2=s.skip_stage_2,
            resolution_factor=s.resolution_factor,
            input_width=canvas_w,
            input_height=canvas_h,
            resolution=request.resolution,
            stage_2_ic_lora=s.stage_2_ic_lora,
        )
        # Queued admission gates on total ("will it ever fit"): a running job
        # depresses free, and the queue exists to wait for a free slot. The
        # optional look still is priced as present, so execute never costs more.
        return self._ic_lora_job(
            width,
            height,
            frame_count=compute_num_frames(stored.duration, int(stored.fps)),
            stage_mode=s.stage_mode,
            resolution_factor=resolution_factor,
            ic_lora_id=item.id,
            stills=catalog_preview_stills(item),
            tiling=s.tiling,
            total_only=True,
        ).risk

    def generate(self, req: IcLoraGenerateRequest) -> IcLoraGenerateResponse:
        # Hardware viability first: local downloads are disabled on an unsupported
        # runtime, so a later weights check would report IC_LORA_NOT_DOWNLOADED and
        # hide the real reason. The per-job budget still 422s unsupported on its own.
        self._require_local_generation_possible()
        if req.ic_lora_id is not None:
            # Queued Home passes an explicit size with stage 2 skipped. GenSpace
            # does not. Reject it here so a later GenSpace caller cannot inherit
            # that canvas without opting in on the queued entry.
            if req.resolution is not None and req.skip_stage_2 is not False:
                raise HTTPError(
                    400,
                    "Explicit resolution with skip_stage_2 is only supported on the queued catalog path",
                )
            with self._generation.reserved_generation_start():
                return self._generate_ic_lora(
                    req,
                    generation_id=uuid.uuid4().hex[:8],
                    seed=self._resolve_seed(),
                    output_path=self.config.outputs_dir
                    / f"ic_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.mp4",
                    local_model_id=None,
                    explore_generation=False,
                )

        # The canny/depth/custom paths have no catalog entry to opt into promptless runs.
        if not req.prompt.strip():
            raise HTTPError(400, "Prompt is required")
        with self._generation.reserved_generation_start():
            # Built-in defaults, with any explicit request value applied on top.
            settings = _resolve_settings(req, IcLoraSettings())

            depth_model_path: Path | None
            if req.conditioning_type == "custom":
                # User-supplied IC-LoRA + pre-rendered control video; no preprocessing, so no
                # depth processor needed (depth_model_path stays None).
                if not req.custom_lora_ref or not req.control_video_path:
                    raise HTTPError(400, "Custom IC-LoRA requires custom_lora_ref and control_video_path")
                try:
                    lora_path = resolve_lora_ref(self.models_dir, req.custom_lora_ref)
                except ValueError as exc:
                    raise HTTPError(400, str(exc)) from exc
                if not Path(req.control_video_path).exists():
                    raise HTTPError(400, f"Control video not found: {req.control_video_path}")
                if not is_ic_lora_file(lora_path):
                    logger.warning(
                        "[ic-lora] %s lacks reference_downscale_factor metadata; may not be an IC-LoRA",
                        lora_path.name,
                    )
                depth_model_path = None
            else:
                video_path = Path(req.video_path)
                if not video_path.exists():
                    raise HTTPError(400, f"Video not found: {req.video_path}")
                lora_path, depth_model_path = self._require_ic_lora_model_paths(req.conditioning_type)

            generation_id = uuid.uuid4().hex[:8]
            t_total_start = time.perf_counter()
            logger.info("[ic-lora] Generation started (conditioning=%s)", req.conditioning_type)

            seed = self._resolve_seed()

            try:
                preprocess_time = 0.0
                if req.conditioning_type == "custom":
                    # Pre-rendered control video supplied; derive shape/timing from it.
                    assert req.control_video_path is not None  # validated above
                    control_video_path = req.control_video_path
                    cap = self._video_processor.open_video(control_video_path)
                    if not cap.isOpened():
                        raise HTTPError(400, f"Cannot open control video: {control_video_path}")
                    info = self._video_processor.get_video_info(cap)
                    self._video_processor.release(cap)
                    input_width = int(info["width"])
                    input_height = int(info["height"])
                    frame_count = int(info["frame_count"])
                    fps = float(info["fps"])
                    logger.info("[ic-lora] custom: using provided control video (%d frames)", frame_count)
                else:
                    video_path = Path(req.video_path)
                    cap = self._video_processor.open_video(str(video_path))
                    if not cap.isOpened():
                        raise HTTPError(400, f"Cannot open video: {video_path}")
                    info = self._video_processor.get_video_info(cap)
                    self._video_processor.release(cap)
                    input_width = int(info["width"])
                    input_height = int(info["height"])
                    frame_count = int(info["frame_count"])
                    fps = float(info["fps"])
                    control_video_path = ""

                width, height, resolution_factor = ic_lora_output_canvas(
                    skip_stage_2=settings.skip_stage_2,
                    resolution_factor=settings.resolution_factor,
                    input_width=input_width,
                    input_height=input_height,
                    resolution=req.resolution,
                )
                budget_frames = frame_count
                if req.fps_override is not None and fps > 0 and req.fps_override < fps:
                    budget_frames = self._frame_count_after_fps_override(
                        frame_count, fps, req.fps_override
                    )
                admitted = self._ic_lora_job(
                    width,
                    height,
                    frame_count=snap_to_frame_grid(budget_frames, floor=9),
                    stage_mode=settings.stage_mode,
                    resolution_factor=resolution_factor,
                    allow_evict_retry=True,
                )
                job = admitted.load
                # Before the pipeline claims the GPU. canny/depth get a conditioning system
                # prompt; a fully custom IC-LoRA falls back to the generic rewrite.
                nag_decision = self._decide_nag(req, model_id=None)
                resolved_prompt = self._prompt_enhancement.resolve_for_generation(
                    req.prompt,
                    provenance=req.prompt_provenance,
                    generation_seed=seed,
                    conditioning_type=None if req.conditioning_type == "custom" else req.conditioning_type,
                    negative_prompt=nag_decision.enhancer_negative,
                )
                prompt = resolved_prompt.prompt
                t_load_start = time.perf_counter()
                with log_heartbeat("ic-lora model load"):
                    ic_state = self._pipelines.load_ic_lora(
                        str(lora_path),
                        str(depth_model_path) if depth_model_path is not None else None,
                        settings.lora_strength,
                        mode=job.mode,
                    )
                t_load_end = time.perf_counter()
                logger.info("[ic-lora] Pipeline load: %.2fs", t_load_end - t_load_start)

                self._generation.start_generation(generation_id, ran=admitted.envelope)
                self._generation.update_progress("loading_model", 5, 0, 1)

                encoding_method = "local" if self._text.should_use_local_encoding() else "api"
                t_text_start = time.perf_counter()
                self._text.prepare_text_encoding(prompt, enhance_prompt=resolved_prompt.enhance_via_api)
                t_text_end = time.perf_counter()
                logger.info("[ic-lora] Text encoding (%s): %.2fs", encoding_method, t_text_end - t_text_start)
                nag = self._resolve_nag(req, model_id=None)
                self._generation.raise_if_cancelled()

                if req.conditioning_type != "custom":
                    cond = req.conditioning_type  # narrowed to Literal["canny", "depth"]
                    video_path = Path(req.video_path)
                    cap = self._video_processor.open_video(str(video_path))
                    if not cap.isOpened():
                        raise HTTPError(400, f"Cannot open video: {video_path}")
                    info = self._video_processor.get_video_info(cap)

                    cache_key = ConditioningCacheKey(str(video_path), cond)
                    cached = ic_state.conditioning_cache.get(cache_key)

                    if cached is not None:
                        self._video_processor.release(cap)
                        control_video_path = cached.control_video_path
                        frame_count = cached.frame_count
                        fps = cached.fps
                        logger.info("[ic-lora] Conditioning cache hit for %s/%s", video_path.name, cond)
                    else:
                        t_preprocess_start = time.perf_counter()

                        frame_count = int(info["frame_count"])
                        fps = float(info["fps"])

                        control_video_path = str(
                            self.config.outputs_dir / f"_control_{cond}_{uuid.uuid4().hex[:8]}.mp4"
                        )
                        writer = self._video_processor.create_writer(
                            control_video_path,
                            fourcc="mp4v",
                            fps=fps,
                            size=(int(info["width"]), int(info["height"])),
                        )

                        frame_idx = 0
                        while frame_idx < frame_count:
                            self._generation.raise_if_cancelled()
                            frame = self._video_processor.read_frame(cap)
                            if frame is None:
                                break
                            control_frame = self._build_conditioning_frame(frame, cond, ic_state)
                            writer.write(control_frame)
                            frame_idx += 1

                        self._video_processor.release(cap)
                        self._video_processor.release(writer)
                        preprocess_time = time.perf_counter() - t_preprocess_start
                        logger.info(
                            "[ic-lora] Preprocessing (%s, %d frames): %.2fs",
                            cond, frame_idx, preprocess_time,
                        )

                        ic_state.conditioning_cache.put(
                            cache_key, ConditioningCacheEntry(control_video_path, frame_count, fps)
                        )

                if req.fps_override is not None and req.fps_override < fps:
                    control_video_path, frame_count, fps = self._resample_control_fps(
                        control_video_path, frame_count, fps, req.fps_override
                    )

                images: list[ImageConditioningInput] = [
                    ImageConditioningInput(path=img.path, frame_idx=int(img.frame), strength=float(img.strength))
                    for img in req.images
                ]

                self._generation.update_progress("inference", 15, 0, 1)

                output_path = (
                    self.config.outputs_dir / f"ic_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.mp4"
                )

                # The budget priced the snapped count. The control video can be longer.
                frame_count = snap_to_frame_grid(frame_count, floor=9)
                logger.info(
                    "[ic-lora] run: lora=%s strength=%.2f skip_stage_2=%s res_factor=%.2f audio=%s enhance=%s "
                    "seed=%d -> target %dx%d x %d frames @%gfps | prompt_len=%d",
                    lora_path.name, settings.lora_strength, settings.skip_stage_2, resolution_factor,
                    settings.audio_mode, resolved_prompt.enhancement, seed, width, height, frame_count, fps, len(prompt),
                )
                t_inference_start = time.perf_counter()
                with log_heartbeat("ic-lora inference"):
                    ic_state.pipeline.generate(
                        prompt=prompt,
                        seed=seed,
                        height=height,
                        width=width,
                        num_frames=frame_count,
                        frame_rate=fps,
                        images=images,
                        video_conditioning=[(control_video_path, settings.conditioning_strength)],
                        output_path=str(output_path),
                        skip_stage_2=settings.skip_stage_2,
                        use_lora_in_stage_2=settings.use_lora_in_stage_2,
                        resolution_factor=resolution_factor,
                        chunk_pixel_frames=job.chunk_pixel_frames,
                        # "source" mode muxes the input clip's soundtrack (None if it has none);
                        # "off" mutes; "generated" keeps the model audio.
                        source_audio_path=(
                            req.video_path
                            if settings.audio_mode == "source" and Path(req.video_path).exists()
                            else None
                        ),
                        mute_audio=settings.audio_mode == "off",
                        nag=nag,
                    )
                t_inference_end = time.perf_counter()
                logger.info("[ic-lora] Inference: %.2fs", t_inference_end - t_inference_start)

                t_total_end = time.perf_counter()
                logger.info(
                    "[ic-lora] Total generation: %.2fs (load=%.2fs, text=%.2fs, preprocess=%.2fs, inference=%.2fs)",
                    t_total_end - t_total_start,
                    t_load_end - t_load_start,
                    t_text_end - t_text_start,
                    preprocess_time,
                    t_inference_end - t_inference_start,
                )

                # Priced before preprocessing, so the frames that ran may not match the envelope.
                self._complete_or_drop_cancelled(output_path, ran=None)
                return IcLoraGenerateCompleteResponse(status="complete", video_path=str(output_path))

            except HTTPError:
                self._generation.fail_generation("IC-LoRA generation failed")
                raise
            except Exception as exc:
                self._generation.fail_generation(str(exc))
                if is_cancel_exception(exc):
                    logger.info("Generation cancelled by user")
                    return IcLoraGenerateCancelledResponse(status="cancelled")
                raise HTTPError(500, f"Generation error: {exc}") from exc
            finally:
                self._text.clear_api_embeddings()
