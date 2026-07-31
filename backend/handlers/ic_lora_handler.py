"""IC-LoRA endpoints orchestration handler."""

from __future__ import annotations

import base64
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from api_types import (
    ConditioningType,
    IcLoraExtractRequest,
    IcLoraExtractResponse,
    IcLoraGenerateCancelledResponse,
    IcLoraGenerateCompleteResponse,
    IcLoraGenerateRequest,
    IcLoraGenerateResponse,
    ImageConditioningInput,
    IcLoraCatalogItem,
    IcLoraSettings,
)
from _routes._errors import HTTPError
from frame_math import compute_num_frames, snap_to_frame_grid
from handlers.base import StateHandlerBase
from handlers.generation_handler import GenerationHandler
from handlers.pipelines_handler import PipelinesHandler
from handlers.text_handler import TextHandler
from ic_lora_preprocessing import MediaArtifact, OutpaintParams, PreprocessingContext, run_preprocessing
from runtime_config.model_download_specs import (
    DEPTH_PROCESSOR_CP_ID,
    find_installed_ic_lora_path,
    get_downloaded_ltx_model_id,
    get_existing_cp_path,
    get_ltx_model_spec,
    resolve_ic_lora_path,
)
from runtime_config.models_scanner import is_ic_lora_file, resolve_lora_ref
from runtime_config.runtime_config import RuntimeConfig
from state.conditioning_cache import ConditioningCacheEntry, ConditioningCacheKey
from services.interfaces import VideoProcessor
from services.lora_catalog import LoraCatalogProvider
from services.services_utils import FrameArray
from server_utils.heartbeat import log_heartbeat
from state.app_state_types import AppState, ICLoraState

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)

_DEFAULT_IC_LORA_FPS = 24

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


def _resolve_settings(req: IcLoraGenerateRequest, base: IcLoraSettings) -> IcLoraSettings:
    """Overlay the request's explicit settings onto a base; None means 'keep the base'."""
    return IcLoraSettings(
        skip_stage_2=base.skip_stage_2 if req.skip_stage_2 is None else req.skip_stage_2,
        use_lora_in_stage_2=(
            base.use_lora_in_stage_2 if req.use_lora_in_stage_2 is None else req.use_lora_in_stage_2
        ),
        resolution_factor=base.resolution_factor if req.resolution_factor is None else req.resolution_factor,
        audio_mode=base.audio_mode if req.audio_mode is None else req.audio_mode,
        lora_strength=base.lora_strength if req.lora_strength is None else req.lora_strength,
        conditioning_strength=(
            base.conditioning_strength if req.conditioning_strength is None else req.conditioning_strength
        ),
    )


def _ic_lora_output_fps(ic_lora: IcLoraCatalogItem) -> int:
    """The fps the IC-LoRA's control video runs at — the image_to_frames step's fps param."""
    for step in ic_lora.preprocessing:
        if step.utility == "image_to_frames":
            return int(step.params.get("fps", _DEFAULT_IC_LORA_FPS))  # type: ignore[arg-type]
    return _DEFAULT_IC_LORA_FPS


class IcLoraHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        generation_handler: GenerationHandler,
        pipelines_handler: PipelinesHandler,
        text_handler: TextHandler,
        video_processor: VideoProcessor,
        lora_catalog: LoraCatalogProvider,
        config: RuntimeConfig,
    ) -> None:
        super().__init__(state, lock, config)
        self._generation = generation_handler
        self._pipelines = pipelines_handler
        self._text = text_handler
        self._video_processor = video_processor
        self._catalog = lora_catalog

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
        model_id = get_downloaded_ltx_model_id(self.models_dir)
        if model_id is None:
            raise HTTPError(409, "NO_DOWNLOADED_LTX_MODEL")
        ic_loras_spec = get_ltx_model_spec(model_id).ic_loras_spec
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
            ic_state = self._pipelines.load_ic_lora(
                str(lora_path),
                str(depth_model_path),
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

    def _resample_control_fps(
        self, control_video_path: str, frame_count: int, src_fps: float, target_fps: float
    ) -> tuple[str, int, float]:
        """Decimate the control video to target_fps so fewer frames are generated.

        Fewer frames = shorter sequence = less compute/VRAM (fits longer real-time clips),
        at the cost of choppier motion. Keeps the output duration ~constant and the frame
        count valid for the pipeline ((n-1) % 8 == 0).
        """
        duration = frame_count / src_fps
        new_count = snap_to_frame_grid(round(duration * target_fps), floor=1)
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

    def _generate_ic_lora(self, req: IcLoraGenerateRequest) -> IcLoraGenerateResponse:
        assert req.ic_lora_id is not None  # branch guard in generate()
        with self._generation.reserved_generation_start():
            ic_lora = self._catalog.get_ic_lora(req.ic_lora_id)
            if ic_lora is None:
                raise HTTPError(404, "UNKNOWN_IC_LORA")
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
            generation_id = uuid.uuid4().hex[:8]
            logger.info("[ic-lora] IC-LoRA generation started (ic_lora=%s)", ic_lora.id)
            lease = None
            try:
                lease = self._generation.acquire_local_metal_lease(
                    generation_id=generation_id,
                    workload=f"ic_lora:{ic_lora.id}",
                    reason="Torch local IC-LoRA generation",
                )
                # IC-LoRA preprocessing builds the control video itself; no depth processor needed.
                ic_state = self._pipelines.load_ic_lora(str(lora_path), None, s.lora_strength)
                self._generation.start_generation(generation_id)
                self._generation.update_progress("loading_model", 5, 0, 1)

                use_api = not self._text.should_use_local_encoding()
                # Never enhance an empty prompt — there's nothing to expand and the enhancer would hallucinate.
                enhance_prompt = use_api and self.state.app_settings.prompt_enhancer_enabled_t2v and has_prompt
                self._text.prepare_text_encoding(req.prompt, enhance_prompt=enhance_prompt)

                input_artifact = MediaArtifact(path=req.input_path, kind=ic_lora.input.kind)
                ctx = PreprocessingContext(
                    num_frames=requested_frames,
                    outputs_dir=self.config.outputs_dir,
                    video_processor=self._video_processor,
                    outpaint=self._outpaint_from_request(req),
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

                self._generation.update_progress("inference", 15, 0, 1)
                if s.resolution_factor == 0 and s.skip_stage_2:
                    # "Source dimensions" (resolution_factor sentinel 0): land the output at the
                    # source resolution instead of the 768 bucket. skip_stage_2 renders at
                    # canvas//2, so pass 2*source and render at factor 1.0; the pipeline snaps to
                    # /128, so final dims are multiples of 64 (source dims that aren't are snapped
                    # to the nearest, e.g. 540 -> 512).
                    width = max(128, 2 * input_width)
                    height = max(128, 2 * input_height)
                    resolution_factor = 1.0
                elif s.use_lora_in_stage_2 and not s.skip_stage_2:
                    # IC-LoRA kept in Stage 2 → mirror the t2v two-stage flow: land the output at
                    # the chosen resolution (or source if unset). Snap each edge down to a multiple
                    # of 128 (Stage 1 runs at canvas//2 and its latent must be even for the 2x
                    # patchify) and never upscale above source.
                    target_w = req.resolution.width if req.resolution else input_width
                    target_h = req.resolution.height if req.resolution else input_height
                    width = max(128, (min(target_w, input_width) // 128) * 128)
                    height = max(128, (min(target_h, input_height) // 128) * 128)
                    # The pipeline only consumes resolution_factor on the skip_stage_2 path, so it
                    # has no effect here (two-stage lands at the width/height above). Pass 1.0.
                    resolution_factor = 1.0
                else:
                    width = 768
                    height = max(round(width * input_height / input_width / 128) * 128, 128)
                    resolution_factor = s.resolution_factor
                output_path = (
                    self.config.outputs_dir
                    / f"ic_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.mp4"
                )
                # Optional reference image (e.g. a photoreal seed for 3D-render): conditions the
                # first frame. Only honoured when the catalog entry opts in; ignored otherwise.
                reference_images: list[ImageConditioningInput] = (
                    [
                        ImageConditioningInput(path=img.path, frame_idx=int(img.frame), strength=float(img.strength))
                        for img in req.images
                    ]
                    if ic_lora.allows_reference_image
                    else []
                )
                seed = self._resolve_seed()
                with log_heartbeat("ic-lora inference"):
                    ic_state.pipeline.generate(
                        prompt=req.prompt,
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
                        source_audio_path=(
                            req.input_path
                            if s.audio_mode == "source" and ic_lora.input.kind == "video"
                            else None
                        ),
                        mute_audio=s.audio_mode == "off",
                        conditioning_mask_path=control.mask_path,
                    )
                self._generation.update_progress("complete", 100, 1, 1)
                self._generation.complete_generation(str(output_path))
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
                self._generation.fail_generation(str(exc))
                if "cancelled" in str(exc).lower():
                    return IcLoraGenerateCancelledResponse(status="cancelled")
                raise HTTPError(500, f"Generation error: {exc}") from exc
            finally:
                self._text.clear_api_embeddings()
                if lease is not None:
                    self._pipelines.cleanup_runtime_caches()
                    lease.close()

    def generate(self, req: IcLoraGenerateRequest) -> IcLoraGenerateResponse:
        if req.ic_lora_id is not None:
            return self._generate_ic_lora(req)

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

            lease = None
            try:
                lease = self._generation.acquire_local_metal_lease(
                    generation_id=generation_id,
                    workload=f"ic_lora:{req.conditioning_type}",
                    reason="Torch local IC-LoRA generation",
                )
                t_load_start = time.perf_counter()
                with log_heartbeat("ic-lora model load"):
                    ic_state = self._pipelines.load_ic_lora(
                        str(lora_path),
                        str(depth_model_path) if depth_model_path is not None else None,
                        settings.lora_strength,
                    )
                t_load_end = time.perf_counter()
                logger.info("[ic-lora] Pipeline load: %.2fs", t_load_end - t_load_start)

                self._generation.start_generation(generation_id)
                self._generation.update_progress("loading_model", 5, 0, 1)

                s = self.state.app_settings
                use_api = not self._text.should_use_local_encoding()
                encoding_method = "api" if use_api else "local"
                t_text_start = time.perf_counter()
                # The prompt is guaranteed non-empty here (guarded at the top of generate()).
                enhance_prompt = use_api and s.prompt_enhancer_enabled_t2v
                self._text.prepare_text_encoding(req.prompt, enhance_prompt=enhance_prompt)
                t_text_end = time.perf_counter()
                logger.info("[ic-lora] Text encoding (%s): %.2fs", encoding_method, t_text_end - t_text_start)

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
                    cond = req.conditioning_type  # narrowed to Literal["canny", "depth"]
                    video_path = Path(req.video_path)
                    cap = self._video_processor.open_video(str(video_path))
                    if not cap.isOpened():
                        raise HTTPError(400, f"Cannot open video: {video_path}")
                    info = self._video_processor.get_video_info(cap)
                    input_width = int(info["width"])
                    input_height = int(info["height"])

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

                if settings.use_lora_in_stage_2 and not settings.skip_stage_2:
                    # IC-LoRA kept in Stage 2 → mirror the t2v two-stage flow: output at the chosen
                    # resolution (or source if unset), snapped down to a multiple of 128 (Stage 1
                    # runs at canvas//2; its latent must be even for the 2x patchify), never upscaling.
                    target_w = req.resolution.width if req.resolution else input_width
                    target_h = req.resolution.height if req.resolution else input_height
                    width = max(128, (min(target_w, input_width) // 128) * 128)
                    height = max(128, (min(target_h, input_height) // 128) * 128)
                else:
                    width = 768
                    height = max(round(width * input_height / input_width / 128) * 128, 128)

                output_path = (
                    self.config.outputs_dir / f"ic_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.mp4"
                )

                if (frame_count - 1) % 8 != 0:
                    logger.warning(
                        "[ic-lora] %d frames: (frames-1) %% 8 != 0 — pipeline may pad/trim and output could glitch.",
                        frame_count,
                    )
                seed = self._resolve_seed()
                logger.info(
                    "[ic-lora] run: lora=%s strength=%.2f skip_stage_2=%s res_factor=%.2f audio=%s enhance=%s "
                    "seed=%d -> target %dx%d x %d frames @%gfps | prompt_len=%d",
                    lora_path.name, settings.lora_strength, settings.skip_stage_2, settings.resolution_factor,
                    settings.audio_mode, enhance_prompt, seed, width, height, frame_count, fps, len(req.prompt),
                )
                t_inference_start = time.perf_counter()
                with log_heartbeat("ic-lora inference"):
                    ic_state.pipeline.generate(
                        prompt=req.prompt,
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
                        resolution_factor=settings.resolution_factor,
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

                self._generation.update_progress("complete", 100, 1, 1)
                self._generation.complete_generation(str(output_path))
                return IcLoraGenerateCompleteResponse(status="complete", video_path=str(output_path))

            except HTTPError:
                self._generation.fail_generation("IC-LoRA generation failed")
                raise
            except Exception as exc:
                self._generation.fail_generation(str(exc))
                if "cancelled" in str(exc).lower():
                    return IcLoraGenerateCancelledResponse(status="cancelled")
                raise HTTPError(500, f"Generation error: {exc}") from exc
            finally:
                self._text.clear_api_embeddings()
                if lease is not None:
                    self._pipelines.cleanup_runtime_caches()
                    lease.close()
