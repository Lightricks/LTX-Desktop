"""Queued Home IC-LoRA recipe. Runs the catalog IC-LoRA path for one recipe row."""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast, get_args

from pydantic import ValidationError

from api_types import (
    IC_LORA_CONTROL_TYPE_ID,
    GenerationRisk,
    IcLoraRecipeInputs,
    IcLoraRecipeStoredParams,
    IcLoraVideoRecipeStoredParams,
    IcLoraGenerateCancelledResponse,
    IcLoraGenerateRequest,
    IcLoraImageInput,
    InputAssetRef,
    LTXLocalModelId,
    LTXVideoGenAspectRatio,
    TargetResolution,
)
from _routes._errors import HTTPError
from runtime_config.budget_admission import RiskAllowance
from runtime_config.ic_lora_high_res import (
    HIGH_RES_NOT_ENABLED,
    high_res_flag_satisfied,
    high_res_cell,
    high_res_stage1,
)
from runtime_config.ic_lora_local_envelope import fit_ic_lora_stage1
from runtime_config.ltx_capabilities import budget_pixels, local_caps
from services.aspect_ratio import center_crop_box, find_closest_aspect_ratio_option
from services.media_probe import video_fps, video_frame_count
from services.features.video.alpha_cutout import (
    CUTOUT_OUTPUT_PLANS,
    bake_alpha_cutout,
    bake_alpha_gif_if_possible,
    require_cutout_outputs,
)
from services.phase_log import PREFIX, logged_phase
from services.features.video.cutout_alignment import (
    conform_matte_to_clip,
    fit_clip_to_canvas,
    prepare_clip_for_model,
)
from services.features.video.lanczos_guide import prepare_lanczos_guide
from services.features.video.ic_lora_conform import (
    conform_output_to_clip,
    pad_clip_to_frame_grid,
)
from services.features.video.job_common import (
    AssetLookup,
    capability_failed_from_http,
    model_from_spec,
    require_existing_asset,
    resolve_generation_seed,
    resolve_job_local_model_id,
    spec_risk_allowance,
)
from services.features.video.text_to_video import (
    VIDEO_OUTPUT_PLAN,
    require_supported_contract_version,
    require_video_output,
)
from services.ffmpeg import FFMPEG_PROTOCOL_WHITELIST, run_ffmpeg, timeout_for_pixels
from services.generation_interrupt import GenerationCancelledError, is_cancel_exception
from services.generation_queue.sessions import JobSession
from services.generation_queue.types import (
    ExecutionOutcome,
    OutputAllocation,
    OutputPlan,
)
from services.records import CapabilityFailedError, GenerationRecord, MediaError

logger = logging.getLogger(__name__)


def _frame_outcome(clip_frames: int, run_frames: int) -> str:
    if run_frames > clip_frames:
        return f"snapped up, {run_frames - clip_frames} frames padded and cut after the run"
    if run_frames < clip_frames:
        return f"over budget, fell back to snapped down, last {clip_frames - run_frames} frames dropped"
    return "no change, already on the frame grid"


class ReservedIcLoraGenerator(Protocol):
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
    ) -> object: ...

    def recipe_run_frames(
        self,
        *,
        ic_lora_id: str,
        source_width: int,
        source_height: int,
        stored: IcLoraRecipeStoredParams,
        clip_frames: int,
        upscale: bool,
        allowance: RiskAllowance | None = None,
    ) -> tuple[int, GenerationRisk | None]:
        """Frame count the run uses for a clip: snapped up, or down when up is over budget.

        The risk is the one the create-time preflight already stored. The executor ignores it.
        """
        ...


def ic_lora_video_filters(
    box: tuple[int, int, int, int] | None, output_fps: float | None
) -> str | None:
    """ffmpeg -vf chain. ``fps`` picks the nearest source frame, so the file the
    catalog path reads is already at the selected rate."""
    filters: list[str] = []
    if output_fps is not None:
        filters.append(f"fps={output_fps:g}")
    if box is not None:
        crop_w, crop_h, x, y = box
        filters.append(f"crop={crop_w}:{crop_h}:{x}:{y}")
    if not filters:
        return None
    return ",".join(filters)


def center_crop_video(
    src: Path,
    dest: Path,
    width: int,
    height: int,
    aspect: str,
    *,
    keep_audio: bool,
    output_fps: float | None = None,
    crop: bool = True,
) -> Path:
    box = center_crop_box(width, height, aspect) if crop else None
    video_filter = ic_lora_video_filters(box, output_fps)
    if video_filter is None:
        return src
    args = [
        "-y",
        "-protocol_whitelist",
        FFMPEG_PROTOCOL_WHITELIST,
        "-i",
        str(src),
        "-vf",
        video_filter,
        "-pix_fmt",
        "yuv420p",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-color_range",
        "tv",
    ]
    # Re-encode. Stream-copy fails when the source audio (WebM Vorbis/Opus) cannot
    # live in the mp4 crop.
    args += ["-c:a", "aac"] if keep_audio else ["-an"]
    args += ["-protocol_whitelist", FFMPEG_PROTOCOL_WHITELIST, str(dest)]
    run_ffmpeg(args, timeout=timeout_for_pixels(width, height, video_frame_count(src)))
    return dest


class IcLoraRecipeExecutor:
    def __init__(
        self,
        ic_lora: ReservedIcLoraGenerator,
        assets: AssetLookup,
        models_dir: Callable[[], Path],
        *,
        catalog_id: str,
        cutout: bool = False,
        upscales_to_cell: bool = False,
        high_res_enabled: Callable[[], bool] | None = None,
    ) -> None:
        self._ic_lora = ic_lora
        self._assets = assets
        self._models_dir = models_dir
        self._catalog_id = catalog_id
        self._cutout = cutout
        self._upscales_to_cell = upscales_to_cell
        self._high_res_enabled = high_res_enabled or (lambda: False)

    def _require_high_res(self) -> None:
        """The 1440p and 2160p cells need the highResVideoToVideo flag, for most recipes."""
        if not high_res_flag_satisfied(self._catalog_id, flag_enabled=self._high_res_enabled()):
            logger.warning(
                "%s high-res cell refused: recipe=%s highResVideoToVideo=%s",
                PREFIX, self._catalog_id, self._high_res_enabled(),
            )
            raise CapabilityFailedError(HIGH_RES_NOT_ENABLED, code=HIGH_RES_NOT_ENABLED)

    def validate_params(
        self, spec: Mapping[str, object], *, contract_version: int
    ) -> None:
        require_supported_contract_version(contract_version)
        try:
            model_from_spec(spec, "params", IcLoraVideoRecipeStoredParams)
            model_from_spec(spec, "inputs", IcLoraRecipeInputs)
        except (ValidationError, ValueError) as exc:
            raise CapabilityFailedError(
                str(exc), code="INVALID_GENERATION_SPEC"
            ) from exc

    def _reference_image_path(self, ref: InputAssetRef | None) -> str | None:
        if ref is None:
            return None
        image = require_existing_asset(
            self._assets,
            ref.assetId,
            media_kind="image",
            unavailable="reference image is no longer available",
        )
        return image.path

    def plan_outputs(self, generation: GenerationRecord) -> tuple[OutputPlan, ...]:
        self.validate_params(
            generation.spec, contract_version=generation.contract_version
        )
        return CUTOUT_OUTPUT_PLANS if self._cutout else (VIDEO_OUTPUT_PLAN,)

    def execute(
        self,
        generation: GenerationRecord,
        outputs: tuple[OutputAllocation, ...],
        session: JobSession,
    ) -> ExecutionOutcome:
        self.validate_params(
            generation.spec, contract_version=generation.contract_version
        )
        params = model_from_spec(generation.spec, "params", IcLoraVideoRecipeStoredParams)
        inputs = model_from_spec(generation.spec, "inputs", IcLoraRecipeInputs)
        video = require_existing_asset(
            self._assets,
            inputs.video.assetId,
            media_kind="video",
            unavailable="video is no longer available",
        )
        if video.metadata.mediaType != "video":
            raise CapabilityFailedError(
                "INVALID_VIDEO_ASSET", code="INVALID_GENERATION_SPEC"
            )
        reference_image_path = self._reference_image_path(inputs.image)
        meta = video.metadata.metadata
        local_model_id = resolve_job_local_model_id(self._models_dir(), params.model)
        seed = resolve_generation_seed(params.seed)
        # A cutout recipe keeps the model matte as the second output and bakes the
        # WebM with alpha and a GIF from it. Any other recipe writes its one output directly.
        cutout_dest: Path | None = None
        gif_dest: Path | None = None
        if self._cutout:
            cutout, matte, gif = require_cutout_outputs(outputs)
            cutout_dest = Path(cutout.dest_path)
            gif_dest = Path(gif.dest_path)
            dest = Path(matte.dest_path)
        else:
            dest = Path(require_video_output(outputs).dest_path)
        source = Path(video.path)
        cropped: Path | None = None
        padded: Path | None = None
        guide: Path | None = None
        try:
            aspect = cast(
                LTXVideoGenAspectRatio,
                find_closest_aspect_ratio_option(
                    meta.width,
                    meta.height,
                    get_args(LTXVideoGenAspectRatio),
                ),
            )
            cropped = dest.with_name(f"{dest.stem}-crop.mp4")
            source_fps = meta.fps if meta.fps is not None else video_fps(Path(video.path))
            # A selected rate at or above the source is a no-op. 23.976 labelled
            # 24 must not be resampled up.
            output_fps = params.fps if params.fps + 1e-3 < source_fps else None
            if self._upscales_to_cell:
                # The Lanczos guide below scales, crops and resamples in one lossless pass.
                # A crop file here would add a lossy encode before it.
                source = Path(video.path)
            else:
                source = center_crop_video(
                    Path(video.path),
                    cropped,
                    meta.width,
                    meta.height,
                    aspect,
                    keep_audio=params.audioMode == "source" and meta.audioStreamCount > 0,
                    output_fps=output_fps,
                    # A cutout keeps the whole frame. The clip is fitted into the canvas below.
                    crop=not self._cutout,
                )
            # Inside the try so a resolution with no local cell becomes a capability
            # failure, the same as image-to-video's pixel lookup.
            high_res = high_res_cell(params.resolution)
            if high_res is not None:
                self._require_high_res()
            if high_res is not None:
                width, height = stage_canvas = high_res_stage1(
                    meta.width,
                    meta.height,
                    high_res,
                    upscale=self._upscales_to_cell,
                )
            else:
                local_resolution = params.resolution
                assert local_resolution != "1440p" and local_resolution != "2160p"
                width, height = budget_pixels(
                    local_caps(local_model_id),
                    local_resolution,
                    aspect,
                    mode="video",
                )
                stage_canvas = fit_ic_lora_stage1(width, height)
            if self._cutout:
                logger.info(
                    "%s run %s: cell=%s source=%dx%d %.3gfps %.0f MB, stage-1 canvas %dx%d, "
                    "high_res=%s",
                    PREFIX, generation.id, params.resolution, meta.width, meta.height,
                    meta.fps or 0.0, (meta.sizeBytes or 0) / 1e6, *stage_canvas,
                    high_res is not None,
                )
            if self._upscales_to_cell and cutout_dest is None:
                # The model card feeds a Lanczos scaled clip. The pipeline would scale with
                # a bilinear filter, so the clip is written at the canvas size here. The frame
                # count, the grid pad and the output conform below all read this clip.
                guide = dest.with_name(f"{dest.stem}-guide.mp4")
                with logged_phase(
                    logger,
                    f"guide clip {stage_canvas[0]}x{stage_canvas[1]} (lanczos, source {meta.width}x{meta.height})",
                    heartbeat=True,
                ):
                    source = prepare_lanczos_guide(
                        source,
                        guide,
                        width=stage_canvas[0],
                        height=stage_canvas[1],
                        keep_audio=params.audioMode == "source" and meta.audioStreamCount > 0,
                        output_fps=output_fps,
                    )
            model_input = source
            fit = None
            clip_frames = video_frame_count(source)
            keep_source_audio = params.audioMode == "source" and meta.audioStreamCount > 0
            allowance = spec_risk_allowance(generation.spec)
            if cutout_dest is None:
                # The pipeline snaps a frame count down. Repeat the last frame up to the
                # next valid count, and cut the output back to `clip_frames` after the run.
                # When that count is over the budget, the budget picks the count below instead.
                # The stored risk allowance, not the live mode, gates the check, so a mode
                # change does not refuse a job that create already admitted.
                target_frames, _ = self._ic_lora.recipe_run_frames(
                    ic_lora_id=self._catalog_id,
                    source_width=meta.width,
                    source_height=meta.height,
                    stored=params,
                    clip_frames=clip_frames,
                    upscale=self._upscales_to_cell,
                    allowance=allowance,
                )
                logger.info(
                    "[ic-lora] recipe %s frames: clip=%d run=%d (%s)",
                    self._catalog_id, clip_frames, target_frames,
                    _frame_outcome(clip_frames, target_frames),
                )
                padded = dest.with_name(f"{dest.stem}-grid.mp4")
                model_input = pad_clip_to_frame_grid(
                    source,
                    padded,
                    frames=clip_frames,
                    target=target_frames,
                    keep_audio=keep_source_audio,
                )
            if cutout_dest is not None:
                # The model denoises one stage canvas, and the pipeline center-crops the clip
                # to fill it. Fit the clip inside that canvas, so the matte covers the same view
                # as `source`. Repeat the last frame, so the 8k+1 snap keeps every frame.
                fit = fit_clip_to_canvas(meta.width, meta.height, *stage_canvas)
                padded = dest.with_name(f"{dest.stem}-fit.mp4")
                with logged_phase(
                    logger,
                    f"guide clip {fit.canvas_width}x{fit.canvas_height} (clip {fit.width}x{fit.height})",
                    heartbeat=True,
                ):
                    model_input = prepare_clip_for_model(
                        source,
                        padded,
                        fit=fit,
                        keep_audio=params.audioMode == "source" and meta.audioStreamCount > 0,
                    )
                logger.info(
                    "%s guide clip file %.0f MB", PREFIX, Path(model_input).stat().st_size / 1e6
                )
            request = IcLoraGenerateRequest(
                ic_lora_id=self._catalog_id,
                input_path=str(model_input),
                prompt=params.prompt,
                negative_prompt=params.negativePrompt,
                nag_scale=params.nagScale,
                nag_alpha=params.nagAlpha,
                nag_tau=params.nagTau,
                prompt_provenance=params.promptProvenance,
                conditioning_type="custom",
                resolution=TargetResolution(width=width, height=height),
                audio_mode=params.audioMode,
                lora_strength=params.loras[0].scale,
                tile_overlap=params.tilingOverlap,
                tile_size=params.tilingSize,
                carry_frames=params.carryFrames,
                max_window_frames=params.maxWindowFrames,
                variant_id=params.loras[0].variantId,
                control_values=(
                    {IC_LORA_CONTROL_TYPE_ID: params.controlType}
                    if params.controlType is not None
                    else {}
                ),
                # No request frame. The handler uses the catalog pin
                # (-1 for the Layout look still).
                images=(
                    [IcLoraImageInput(path=reference_image_path)]
                    if reference_image_path is not None
                    else []
                ),
            )
            with logged_phase(logger, "model run"):
                result = self._ic_lora.generate_local_reserved(
                    request,
                    generation_id=generation.id,
                    output_path=dest,
                    local_model_id=local_model_id,
                    seed=seed,
                    explicit_generation_seed=params.seed is not None,
                    high_res=high_res is not None,
                    allowance=allowance,
                )
            if isinstance(result, IcLoraGenerateCancelledResponse):
                raise GenerationCancelledError()
            if cutout_dest is None:
                conform_output_to_clip(
                    dest,
                    frames=clip_frames,
                    source_audio=Path(video.path) if keep_source_audio else None,
                )
            if cutout_dest is not None and fit is not None and gif_dest is not None:
                with logged_phase(logger, "matte conform", heartbeat=True):
                    conform_matte_to_clip(
                        dest, frames=video_frame_count(source), fps=video_fps(dest), fit=fit
                    )
                # `source` is the clip the conformed matte is aligned with. Keep it until here.
                with logged_phase(logger, "webm bake", heartbeat=True):
                    bake_alpha_cutout(source, dest, cutout_dest)
                logger.info("%s webm file %.0f MB", PREFIX, cutout_dest.stat().st_size / 1e6)
                with logged_phase(logger, "gif bake", heartbeat=True):
                    bake_alpha_gif_if_possible(source, dest, gif_dest)
            # Modes unknown: the catalog IC-LoRA path does not surface its
            # load/offload decisions. Nulls mean unknown — a follow-up can
            # thread them through.
            return ExecutionOutcome()
        except Exception as exc:
            if is_cancel_exception(exc):
                raise GenerationCancelledError() from exc
            if isinstance(exc, KeyError):
                raise CapabilityFailedError(
                    "INVALID_LOCAL_RESOLUTION", code="INVALID_LOCAL_RESOLUTION"
                ) from exc
            if isinstance(exc, HTTPError):
                raise capability_failed_from_http(exc) from exc
            if isinstance(exc, CapabilityFailedError):
                raise
            if isinstance(exc, MediaError):
                raise CapabilityFailedError(str(exc)) from exc
            if isinstance(exc, subprocess.CalledProcessError):
                raise CapabilityFailedError("crop failed") from exc
            raise
        finally:
            # Unlink the temp path itself. `source` is only reassigned after a
            # successful crop, so a failed ffmpeg would otherwise leave the file.
            if cropped is not None:
                cropped.unlink(missing_ok=True)
            if padded is not None:
                padded.unlink(missing_ok=True)
            if guide is not None:
                guide.unlink(missing_ok=True)
