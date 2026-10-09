"""IC-LoRA endpoints orchestration handler."""

from __future__ import annotations

import base64
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from threading import RLock

from PIL import UnidentifiedImageError

from api_types import (
    ConditioningType,
    IcLoraCatalogItem,
    IcLoraExtractRequest,
    IcLoraExtractResponse,
    IcLoraGenerateCancelledResponse,
    IcLoraGenerateCompleteResponse,
    IcLoraGenerateRequest,
    IcLoraGenerateResponse,
    IcLoraImageInput,
    IcLoraSettings,
    ImageConditioningInput,
    LTXLocalModelId,
)
from _routes._errors import HTTPError
from frame_math import compute_num_frames, snap_to_frame_grid
from handlers.base import StateHandlerBase
from handlers.generation_handler import GenerationHandler
from handlers.pipelines_handler import PipelinesHandler
from handlers.prompt_enhancement_handler import PromptEnhancementHandler
from handlers.text_handler import TextHandler
from ic_lora_preprocessing import MediaArtifact, OutpaintParams, PreprocessingContext, run_preprocessing
from runtime_config.ic_lora_job_budget import (
    IcLoraJobLoad,
    IcLoraJobReject,
    catalog_output_fps,
    decide_ic_lora_job,
)
from runtime_config.ic_lora_local_envelope import (
    IC_LORA_SOURCE_TOO_LARGE,
    IC_LORA_V1_ENVELOPE_MESSAGE,
    effective_denoise_stage1,
    ic_lora_output_canvas,
)
from runtime_config.ic_lora_stage_mode import IcLoraStageMode
from runtime_config.ic_lora_tiling import IcLoraTiling
from runtime_config.video_job_budget import (
    LOCAL_GENERATION_UNSUPPORTED,
    LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
    VIDEO_JOB_TOO_LARGE,
    VIDEO_JOB_TOO_LARGE_MESSAGE,
)
from runtime_config.ltx_capabilities import local_caps, supports
from runtime_config.model_download_specs import (
    DEPTH_PROCESSOR_CP_ID,
    catalog_item_visible_for_installed_ltx,
    find_installed_ic_lora_path,
    get_existing_cp_path,
    get_ltx_model_spec,
    ltx_catalog_family_for_model,
    resolve_active_ltx_model_id,
    resolve_ic_lora_path,
)
from runtime_config.models_scanner import is_ic_lora_file, resolve_lora_ref
from runtime_config.runtime_config import RuntimeConfig
from runtime_config.video_job_budget import memory_gb_for_job
from state.conditioning_cache import ConditioningCacheEntry, ConditioningCacheKey
from services.interfaces import VideoProcessor
from services.lora_catalog import LoraCatalogProvider
from services.services_utils import FrameArray
from server_utils.heartbeat import log_heartbeat
from server_utils.oriented_image import open_oriented_image
from services.generation_interrupt import GenerationCancelledError, is_cancel_exception
from state.app_state_types import AppState, ICLoraState

logger = logging.getLogger(__name__)


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


def _resolve_settings(req: IcLoraGenerateRequest, base: IcLoraSettings) -> IcLoraSettings:
    """Overlay the request's explicit settings onto a base; None means 'keep the base'."""
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
    return IcLoraSettings(
        skip_stage_2=skip_stage_2,
        use_lora_in_stage_2=False,
        # Stage 2 does not run when skipped, so the flag would only change the cache key.
        stage_2_ic_lora=base.stage_2_ic_lora and not skip_stage_2,
        # Catalog only. A request cannot change the tile size.
        tiling=base.tiling,
        resolution_factor=base.resolution_factor if req.resolution_factor is None else req.resolution_factor,
        audio_mode=base.audio_mode if req.audio_mode is None else req.audio_mode,
        lora_strength=base.lora_strength if req.lora_strength is None else req.lora_strength,
        conditioning_strength=(
            base.conditioning_strength if req.conditioning_strength is None else req.conditioning_strength
        ),
    )


def _ic_lora_output_fps(ic_lora: IcLoraCatalogItem) -> int:
    """The fps the IC-LoRA's control video runs at — the image_to_frames step's fps param."""
    return catalog_output_fps(ic_lora)


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
    ) -> None:
        super().__init__(state, lock, config)
        self._generation = generation_handler
        self._pipelines = pipelines_handler
        self._text = text_handler
        self._prompt_enhancement = prompt_enhancement_handler
        self._video_processor = video_processor
        self._catalog = lora_catalog

    def _ic_lora_job(
        self,
        canvas_width: int,
        canvas_height: int,
        *,
        frame_count: int,
        stage_mode: IcLoraStageMode,
        resolution_factor: float,
        stills: int = 0,
        tiling: IcLoraTiling | None = None,
    ) -> IcLoraJobLoad:
        """422, or pick full vs stream and the chunk window for this job.

        Does not change process mode. The pipeline runs the window this returns."""
        stage1_w, stage1_h = effective_denoise_stage1(
            canvas_width,
            canvas_height,
            skip_stage_2=stage_mode is IcLoraStageMode.SINGLE_STAGE,
            resolution_factor=resolution_factor,
        )
        darwin = self.config.darwin_unified_memory
        memory_gb = memory_gb_for_job(
            vram_gb=self.config.vram_gb,
            available_ram_gb=self.config.available_ram_gb,
            darwin=darwin,
        )
        decision = decide_ic_lora_job(
            stage1_w,
            stage1_h,
            frame_count,
            memory_gb=memory_gb,
            process_mode=self.config.local_generations_mode,
            stage_mode=stage_mode,
            darwin=darwin,
            stills=stills,
            tiling=tiling,
        )
        outcome = "reject" if isinstance(decision, IcLoraJobReject) else decision.mode
        logger.info(
            "[ic-lora] job budget seq=%.0f full=%.1fGiB stream=%.1fGiB -> %s",
            decision.seq,
            decision.full_gib,
            decision.stream_gib,
            outcome,
        )
        if isinstance(decision, IcLoraJobReject):
            if decision.reason == "spatial":
                raise HTTPError(422, IC_LORA_V1_ENVELOPE_MESSAGE, code=IC_LORA_SOURCE_TOO_LARGE)
            if decision.reason == "unsupported":
                raise HTTPError(
                    422,
                    LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
                    code=LOCAL_GENERATION_UNSUPPORTED,
                )
            raise HTTPError(422, VIDEO_JOB_TOO_LARGE_MESSAGE, code=VIDEO_JOB_TOO_LARGE)
        return decision

    def _probe_ic_lora_source(
        self, path: str, kind: str, requested_frames: int
    ) -> tuple[int, int, int]:
        """Width, height, and snapped frame count from the user file — before preprocess."""
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
        return width, height, snap_to_frame_grid(src_frames, floor=9)

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
    def _frame_count_after_fps_override(frame_count: int, src_fps: float, target_fps: float) -> int:
        duration = frame_count / src_fps
        return snap_to_frame_grid(round(duration * target_fps), floor=1)

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

    def _complete_or_drop_cancelled(self, output_path: Path) -> None:
        # Denoiser interrupt cannot abort VAE decode / ffmpeg; a Stop after the last
        # denoise step still finishes encode, then this check drops the file.
        if self._generation.is_generation_cancelled():
            output_path.unlink(missing_ok=True)
            raise GenerationCancelledError()
        self._generation.update_progress("complete", 100, 1, 1)
        self._generation.complete_generation(str(output_path))

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
    ) -> IcLoraGenerateResponse:
        """Local-only Home/Remote entry. The checkpoint id is required.

        The caller already holds the GPU slot and allocated ``output_path``.
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
            lora_path = resolve_ic_lora_path(self.models_dir, ic_lora.id, variant.filename)
            if not lora_path.exists():
                raise HTTPError(409, "IC_LORA_NOT_DOWNLOADED")
        else:
            lora_path = find_installed_ic_lora_path(
                self.models_dir, ic_lora.id, ic_lora.download.candidate_filenames()
            )
            if lora_path is None:
                raise HTTPError(409, "IC_LORA_NOT_DOWNLOADED")

        control_values = self._resolve_control_values(req, ic_lora)
        duration = int(control_values.get("duration", 5))
        requested_frames = compute_num_frames(duration, _ic_lora_output_fps(ic_lora))

        # IC-LoRA defaults, with any explicit UI override (e.g. skip_stage_2) applied on top.
        s = _resolve_settings(req, ic_lora.default_settings)
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
            )
            job = self._ic_lora_job(
                width,
                height,
                frame_count=probe_frames,
                stage_mode=stage_mode,
                resolution_factor=resolution_factor,
                stills=stills,
                tiling=s.tiling,
            )
            resolved_prompt = self._prompt_enhancement.resolve_for_generation(
                req.prompt,
                provenance=req.prompt_provenance,
                generation_seed=seed,
                explicit_generation_seed=explicit_generation_seed,
                ic_lora=ic_lora,
                local_model_id=local_model_id,
                explore_generation=explore_generation,
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
                tiling=s.tiling,
            )
            self._generation.start_generation(generation_id)
            self._generation.update_progress("loading_model", 5, 0, 1)

            input_artifact = MediaArtifact(path=req.input_path, kind=ic_lora.input.kind)
            ctx = PreprocessingContext(
                num_frames=requested_frames,
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
                )
                try:
                    new_job = self._ic_lora_job(
                        width,
                        height,
                        frame_count=frame_count,
                        stage_mode=stage_mode,
                        resolution_factor=resolution_factor,
                        stills=stills,
                        tiling=s.tiling,
                    )
                except HTTPError:
                    self._drop_preprocess_artifact(control_video_path, req.input_path)
                    self._drop_preprocess_artifact(control.mask_path, req.input_path)
                    raise
                reload = new_job.mode != job.mode
                job = new_job
                if reload:
                    ic_state = self._pipelines.load_ic_lora(
                        str(lora_path),
                        None,
                        s.lora_strength,
                        mode=job.mode,
                        ltx_model_id=local_model_id,
                        stage_2_ic_lora=s.stage_2_ic_lora,
                        tiling=s.tiling,
                    )

            self._text.prepare_text_encoding(
                prompt,
                enhance_prompt=enhance_via_api,
                model_id=local_model_id,
            )
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
                )
            self._complete_or_drop_cancelled(output_path)
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
            self._text.clear_api_embeddings()

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
                job = self._ic_lora_job(
                    width,
                    height,
                    frame_count=snap_to_frame_grid(budget_frames, floor=9),
                    stage_mode=settings.stage_mode,
                    resolution_factor=resolution_factor,
                )
                # Before the pipeline claims the GPU. canny/depth get a conditioning system
                # prompt; a fully custom IC-LoRA falls back to the generic rewrite.
                resolved_prompt = self._prompt_enhancement.resolve_for_generation(
                    req.prompt,
                    provenance=req.prompt_provenance,
                    generation_seed=seed,
                    conditioning_type=None if req.conditioning_type == "custom" else req.conditioning_type,
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

                self._generation.start_generation(generation_id)
                self._generation.update_progress("loading_model", 5, 0, 1)

                encoding_method = "local" if self._text.should_use_local_encoding() else "api"
                t_text_start = time.perf_counter()
                self._text.prepare_text_encoding(prompt, enhance_prompt=resolved_prompt.enhance_via_api)
                t_text_end = time.perf_counter()
                logger.info("[ic-lora] Text encoding (%s): %.2fs", encoding_method, t_text_end - t_text_start)
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

                self._complete_or_drop_cancelled(output_path)
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
