"""Video generation orchestration handler."""

from __future__ import annotations

import logging
import os

from collections.abc import Callable
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from PIL import Image

from frame_math import (
    AutoDurationSpec,
    compute_num_frames,
    effective_a2v_audio_seconds,
    num_frames_for_audio_duration,
    snap_up_to_multiple,
)
from api_types import (
    AudioToVideoParams,
    CreateAudioToVideoParams,
    OfferingId,
    GenerateVideoCancelledResponse,
    GenerateVideoCompleteResponse,
    GenerateVideoModelsSpecsResponse,
    GenerateVideoRequest,
    GenerateVideoResponse,
    ImageConditioningInput,
    LoraEntry,
    LTXLocalModelId,
    LTXLocalA2VResolution,
    LTXVideoGenFps,
    LTXVideoGenResolution,
    VideoCameraMotion,
)
from runtime_config.ltx_capabilities import (
    CanvasMode,
    LtxAspectRatio,
    a2v_pixels_for,
    budget_pixels,
    budget_size,
    local_canvas,
    local_caps,
    pixels_for,
    supports,
)
from runtime_config.low_performance_machine import detect_low_performance_machine
from runtime_config.models_scanner import resolve_lora_ref
from runtime_config.runtime_policy import LocalGenerationMode
from runtime_config.video_job_budget import (
    LOCAL_GENERATION_UNSUPPORTED,
    LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
    VIDEO_JOB_TOO_LARGE,
    VIDEO_JOB_TOO_LARGE_MESSAGE,
    VideoJobReject,
    decide_video_job,
    memory_gb_for_job,
)
from _routes._errors import HTTPError
from api_model_specs import (
    FORCED_API_MODEL_MAP,
    build_generate_video_model_specs_response,
    forced_api_resolution_map,
    get_local_video_generation_model_specs,
    supported_duration_range,
    validate_generate_video_request,
)
from handlers.base import StateHandlerBase
from server_utils.heartbeat import log_heartbeat
from handlers.generation_handler import GenerationHandler
from handlers.pipelines_handler import PipelinesHandler
from handlers.prompt_enhancement_handler import PromptEnhancementHandler
from handlers.text_handler import TextHandler
from runtime_config.offerings import resolve_offering_local_model_id
from runtime_config.model_download_specs import is_duration_head_ready, resolve_active_ltx_model_id
from server_utils.media_validation import (
    normalize_optional_path,
    validate_audio_file,
    validate_image_file,
)
from server_utils.oriented_image import open_oriented_rgb, oriented_image_file
from services.denoising_progress import distilled_total_steps
from services.generation_interrupt import GenerationCancelledError, is_cancel_exception
from services.interfaces import LTXAPIClient
from services.ltx_api_client.ltx_api_client import LTXAPIClientError
from services.ltx_api_client.ltx_api_errors import map_ltx_api_client_error, missing_ltx_api_key_error
from state.app_state_types import AppState
from state.app_settings import should_video_generate_with_ltx_api

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)


def _upload_oriented_image(client: LTXAPIClient, *, api_key: str, path: Path) -> str:
    with oriented_image_file(path) as upload_path:
        return client.upload_file(api_key=api_key, file_path=str(upload_path))


def _lora_catalog_ids(loras: list[LoraEntry]) -> list[str]:
    """The catalog ids among the selected LoRAs, for catalog-aware prompt enhancement.

    A custom LoRA carries none and stays weights-only: its filename is deliberately not
    matched against the catalog, which would attach a stranger's trigger phrase to it.
    """
    return [entry.catalogId for entry in loras if entry.catalogId is not None]


FORCED_API_RESOLUTION_MAP: dict[str, dict[str, dict[str, str]]] = forced_api_resolution_map()


class VideoGenerationHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        generation_handler: GenerationHandler,
        pipelines_handler: PipelinesHandler,
        text_handler: TextHandler,
        prompt_enhancement_handler: PromptEnhancementHandler,
        ltx_api_client: LTXAPIClient,
        config: RuntimeConfig,
    ) -> None:
        super().__init__(state, lock, config)
        self._generation = generation_handler
        self._pipelines = pipelines_handler
        self._text = text_handler
        self._prompt_enhancement = prompt_enhancement_handler
        self._ltx_api_client = ltx_api_client

    def _active_ltx_model_id(self) -> LTXLocalModelId | None:
        return resolve_active_ltx_model_id(
            self.models_dir, self.state.app_settings.active_ltx_model_id
        )

    def _duration_head_ready_for(self, model_id: LTXLocalModelId | None) -> bool:
        return model_id is not None and is_duration_head_ready(self.models_dir, model_id)

    def _duration_head_ready(self) -> bool:
        return self._duration_head_ready_for(self._active_ltx_model_id())

    def _local_pixels(
        self,
        resolution: LTXVideoGenResolution,
        aspect: LtxAspectRatio,
        *,
        invalid_code: str,
        local_model_id: LTXLocalModelId | None = None,
        mode: CanvasMode = "video",
    ) -> tuple[int, int]:
        model_id = local_model_id or self._active_ltx_model_id()
        if model_id is None:
            raise HTTPError(409, "NO_DOWNLOADED_LTX_MODEL")
        lookup = a2v_pixels_for if mode == "a2v" else pixels_for
        try:
            return lookup(local_caps(model_id), resolution, aspect)
        except KeyError as exc:
            raise HTTPError(400, invalid_code) from exc

    def _local_job_filter(self) -> tuple[float | None, LocalGenerationMode, bool]:
        darwin = self.config.darwin_unified_memory
        return (
            memory_gb_for_job(
                vram_gb=self.config.vram_gb,
                available_ram_gb=self.config.available_ram_gb,
                darwin=darwin,
            ),
            self.config.local_generations_mode,
            darwin,
        )

    def get_model_specs(self) -> GenerateVideoModelsSpecsResponse:
        memory_gb, process_mode, darwin = self._local_job_filter()
        return build_generate_video_model_specs_response(
            local_model_id=self._active_ltx_model_id(),
            models_dir=self.models_dir,
            duration_head_ready=self._duration_head_ready(),
            memory_gb=memory_gb,
            process_mode=process_mode,
            darwin=darwin,
            low_performance_machine=detect_low_performance_machine(self.config),
        )

    def derive_local_a2v_params(
        self, params: CreateAudioToVideoParams, audio_duration_ms: int
    ) -> AudioToVideoParams:
        num_frames = self._local_a2v_num_frames(
            resolution=params.resolution,
            model=params.model,
            fps=params.fps,
            audio_duration_ms=audio_duration_ms,
        )
        return AudioToVideoParams(
            **params.model_dump(mode="python"),
            numFrames=num_frames,
        )

    def validate_local_a2v_request(
        self, params: AudioToVideoParams, audio_duration_ms: int
    ) -> None:
        num_frames = self._local_a2v_num_frames(
            resolution=params.resolution,
            model=params.model,
            fps=params.fps,
            audio_duration_ms=audio_duration_ms,
        )
        if params.numFrames != num_frames:
            raise HTTPError(
                422,
                "Audio-to-video frame count no longer matches the selected audio.",
                code="INVALID_VIDEO_GENERATION_SPEC",
            )

    def _prefix_a2v_audio_seconds(
        self,
        *,
        model_id: LTXLocalModelId,
        resolution: LTXVideoGenResolution,
        fps: LTXVideoGenFps,
        audio_duration_seconds: float,
    ) -> float | None:
        """Opening slice this resolution will hear.

        A clip past this cell but inside a longer one is cut to the cell max.
        A clip past every cell is None, so a caller cannot treat it as the
        whole clip. Missing specs leave the length unchanged.
        """
        memory_gb, process_mode, darwin = self._local_job_filter()
        item = next(
            (
                candidate
                for candidate in get_local_video_generation_model_specs(
                    model_id,
                    duration_head_ready=self._duration_head_ready_for(model_id),
                    memory_gb=memory_gb,
                    process_mode=process_mode,
                    darwin=darwin,
                )
                if candidate.pipeline == "fast"
            ),
            None,
        )
        duration_map = None if item is None else item.spec.a2v_supported_resolutions_durations
        if duration_map is None:
            return audio_duration_seconds
        resolution_spec = duration_map.get(resolution)
        if resolution_spec is None:
            return audio_duration_seconds
        durations = resolution_spec.fps_to_durations.get(fps, [])
        if not durations:
            return audio_duration_seconds
        longest = 0
        for cell in duration_map.values():
            cell_durations = cell.fps_to_durations.get(fps, [])
            if cell_durations:
                longest = max(longest, max(cell_durations))
        return effective_a2v_audio_seconds(
            audio_duration_seconds,
            cell_max_seconds=float(max(durations)),
            longest_cell_seconds=float(longest),
        )

    def _local_a2v_num_frames(
        self,
        *,
        resolution: LTXLocalA2VResolution,
        model: OfferingId,
        fps: LTXVideoGenFps,
        audio_duration_ms: int,
    ) -> int:
        # Envelope of the checkpoint that will actually run, not the catalog
        # preferred row (2.3 1.1 while only 1.0 is on disk).
        a2v_model_id = resolve_offering_local_model_id(self.models_dir, model)
        if a2v_model_id is None:
            raise HTTPError(409, "LTX_MODEL_NOT_INSTALLED")
        memory_gb, process_mode, darwin = self._local_job_filter()
        item = next(
            (
                candidate
                for candidate in get_local_video_generation_model_specs(
                    a2v_model_id,
                    duration_head_ready=self._duration_head_ready_for(a2v_model_id),
                    memory_gb=memory_gb,
                    process_mode=process_mode,
                    darwin=darwin,
                )
                if candidate.pipeline == "fast"
            ),
            None,
        )
        if item is None or item.spec.a2v_supported_resolutions_durations is None:
            raise HTTPError(
                422,
                "Audio-to-video is not available for the selected offering.",
                code="INVALID_VIDEO_GENERATION_SPEC",
            )
        resolution_spec = item.spec.a2v_supported_resolutions_durations.get(resolution)
        if resolution_spec is None:
            raise HTTPError(
                422,
                "Audio-to-video is not available at the selected resolution.",
                code="INVALID_VIDEO_GENERATION_SPEC",
            )
        durations = resolution_spec.fps_to_durations.get(fps, [])
        if not durations:
            raise HTTPError(
                422,
                "Audio-to-video is not available at the selected frame rate and duration.",
                code="INVALID_VIDEO_GENERATION_SPEC",
            )
        audio_duration_seconds = audio_duration_ms / 1000
        if audio_duration_seconds < 2.0:
            raise HTTPError(
                422,
                "Audio is too short for audio-to-video.",
                code="INVALID_VIDEO_GENERATION_SPEC",
            )
        max_duration = max(durations)
        heard = self._prefix_a2v_audio_seconds(
            model_id=a2v_model_id,
            resolution=resolution,
            fps=fps,
            audio_duration_seconds=audio_duration_seconds,
        )
        if heard is None or heard >= max_duration + 0.1:
            raise HTTPError(
                422,
                "Audio is too long for audio-to-video.",
                code="INVALID_VIDEO_GENERATION_SPEC",
            )
        audio_duration_seconds = heard
        max_frames = compute_num_frames(max_duration, fps)
        num_frames = num_frames_for_audio_duration(
            audio_duration_seconds,
            fps,
            max_frames=max_frames,
        )
        if num_frames is None:
            raise HTTPError(
                422,
                "Audio is too long for audio-to-video.",
                code="INVALID_VIDEO_GENERATION_SPEC",
            )
        return num_frames

    def generate(self, req: GenerateVideoRequest) -> GenerateVideoResponse:
        self._require_local_generation_possible(api_key=self.state.app_settings.ltx_api_key)
        use_api_specs = should_video_generate_with_ltx_api(
            force_api_generations=self.config.force_api_generations,
            settings=self.state.app_settings,
        )
        self._validate_generate_video_request(req, use_api_specs=use_api_specs)

        if use_api_specs:
            return self._generate_forced_api(req)

        with self._generation.reserved_generation_start():
            return self._generate_local_reserved_validated(
                req, generation_id=self._make_generation_id()
            )

    def generate_local_reserved(
        self,
        req: GenerateVideoRequest,
        *,
        generation_id: str,
        output_path: Path | None = None,
        prompt_wrap: Callable[[str], str] | None = None,
        a2v_num_frames: int | None = None,
        a2v_audio_duration_seconds: float | None = None,
        local_model_id: LTXLocalModelId,
        explore_generation: bool = True,
    ) -> GenerateVideoResponse:
        # Local-only Home/Remote entry: checkpoint is required. GenSpace
        # ``generate()`` still uses Settings via ``_generate_local_reserved_validated``.
        self._require_local_generation_possible()
        self._validate_generate_video_request(
            req, use_api_specs=False, local_model_id=local_model_id
        )
        return self._generate_local_reserved_validated(
            req,
            generation_id=generation_id,
            output_path=output_path,
            prompt_wrap=prompt_wrap,
            a2v_num_frames=a2v_num_frames,
            a2v_audio_duration_seconds=a2v_audio_duration_seconds,
            local_model_id=local_model_id,
            explore_generation=explore_generation,
        )

    def _validate_generate_video_request(
        self,
        req: GenerateVideoRequest,
        *,
        use_api_specs: bool,
        local_model_id: LTXLocalModelId | None = None,
    ) -> None:
        job_model_id = None if use_api_specs else (local_model_id or self._active_ltx_model_id())
        validation_error = validate_generate_video_request(
            req,
            use_api_specs=use_api_specs,
            local_model_id=job_model_id,
            duration_head_ready=False if use_api_specs else self._duration_head_ready_for(job_model_id),
        )
        if validation_error is not None:
            raise HTTPError(422, validation_error, code="INVALID_VIDEO_GENERATION_SPEC")

    def _generate_local_reserved_validated(
        self,
        req: GenerateVideoRequest,
        *,
        generation_id: str,
        output_path: Path | None = None,
        prompt_wrap: Callable[[str], str] | None = None,
        a2v_num_frames: int | None = None,
        a2v_audio_duration_seconds: float | None = None,
        local_model_id: LTXLocalModelId | None = None,
        explore_generation: bool = False,
    ) -> GenerateVideoResponse:
        resolution = req.resolution
        duration = req.duration
        fps = req.fps
        job_model_id = local_model_id or self._active_ltx_model_id()

        audio_path = normalize_optional_path(req.audioPath)
        if audio_path:
            if duration is None:
                raise HTTPError(
                    422,
                    "Automatic duration cannot be combined with audio-to-video",
                    code="INVALID_VIDEO_GENERATION_SPEC",
                )
            return self._generate_a2v(
                req,
                duration,
                fps,
                audio_path=audio_path,
                generation_id=generation_id,
                output_path=output_path,
                num_frames=a2v_num_frames,
                audio_duration_seconds=a2v_audio_duration_seconds,
                local_model_id=job_model_id,
                explore_generation=explore_generation,
            )

        logger.info("Resolution %s - using fast pipeline", resolution)

        width, height = self._local_pixels(
            resolution,
            req.aspectRatio,
            invalid_code="INVALID_LOCAL_RESOLUTION",
            local_model_id=job_model_id,
        )
        width = snap_up_to_multiple(width, 64)
        height = snap_up_to_multiple(height, 64)

        if duration is None:
            memory_gb, process_mode, darwin = self._local_job_filter()
            item = next(
                candidate
                for candidate in get_local_video_generation_model_specs(
                    job_model_id,
                    duration_head_ready=self._duration_head_ready_for(job_model_id),
                    memory_gb=memory_gb,
                    process_mode=process_mode,
                    darwin=darwin,
                )
                if candidate.pipeline == req.model
            )
            min_seconds, max_seconds = supported_duration_range(
                item, resolution=resolution, fps=fps
            )
            num_frames: int | AutoDurationSpec = AutoDurationSpec(
                min_seconds=float(min_seconds),
                max_seconds=float(max_seconds),
            )
        else:
            num_frames = self._compute_num_frames(duration, fps)

        image = None
        last_image = None
        keyframe_images: list[tuple[Image.Image, int, float]] | None = None
        image_path = normalize_optional_path(req.imagePath)
        last_image_path = normalize_optional_path(req.lastImagePath)
        if req.keyframes:
            keyframe_images = [
                (self._prepare_image(keyframe.imagePath, width, height), keyframe.frameIndex, keyframe.strength)
                for keyframe in req.keyframes
            ]
            logger.info("Keyframes: %s", [(keyframe.imagePath, keyframe.frameIndex) for keyframe in req.keyframes])
            opening = min(req.keyframes, key=lambda keyframe: keyframe.frameIndex)
            image_path = normalize_optional_path(opening.imagePath)
            last_image_path = None
            enhance_keyframes = sorted(
                (
                    (
                        normalize_optional_path(keyframe.imagePath) or keyframe.imagePath,
                        keyframe.frameIndex,
                        keyframe.strength,
                    )
                    for keyframe in req.keyframes
                ),
                key=lambda item: item[1],
            )
        else:
            enhance_keyframes = None
            if image_path:
                image = self._prepare_image(image_path, width, height)
                logger.info("Image: %s -> %sx%s", image_path, width, height)
            if last_image_path:
                last_image = self._prepare_image(last_image_path, width, height)
                logger.info("Last image: %s -> %sx%s", last_image_path, width, height)

        seed = req.seed if req.seed is not None else self._resolve_seed()
        loras = self._resolve_loras(req.loras, local_model_id=job_model_id)

        canvas = local_canvas(resolution, mode="video")
        budget_width, budget_height = budget_size(canvas)
        # The shared duration list is the 16:9 anchor. Only a wider frame can exceed it.
        if job_model_id is None:
            raise HTTPError(409, "NO_DOWNLOADED_LTX_MODEL")
        selected_width, selected_height = budget_pixels(
            local_caps(job_model_id), resolution, req.aspectRatio, mode="video"
        )
        if selected_width * selected_height > budget_width * budget_height:
            budget_width, budget_height = selected_width, selected_height
        load_mode = self._video_job_load_mode(
            budget_width, budget_height, self._budget_frame_count(num_frames, fps)
        )

        try:
            # Before the pipeline loads and before the generation is marked running: local
            # enhancement needs the VRAM a resident pipeline holds, and evicting a pipeline is
            # refused once a generation is running. Budget 422 already happened so Gemma is
            # not loaded for a job this machine cannot run.
            resolved_prompt = self._prompt_enhancement.resolve_for_generation(
                req.prompt,
                provenance=req.promptProvenance,
                generation_seed=seed,
                explicit_generation_seed=req.seed is not None,
                image_path=image_path,
                last_image_path=last_image_path,
                keyframes=enhance_keyframes,
                duration=req.duration,
                fps=req.fps,
                lora_catalog_ids=_lora_catalog_ids(req.loras),
                local_model_id=job_model_id,
                explore_generation=explore_generation,
            )
            # Recipe-scoped: applied after enhancement so the LLM rewrites the user
            # scene, then the fixed scaffold wraps the result. None (and thus
            # a no-op) for the plain text-to-video feature.
            #
            # skip_recipe_wrap is owned by resolve_for_generation: auto-off + a typed
            # prompt skips the wrap so Generate sends the scene as typed. A manual
            # Enhance (provenance "enhanced") still wraps so the style lock lands on
            # the rewrite the user approved.
            generation_prompt = resolved_prompt.prompt
            enhance_via_api = resolved_prompt.enhance_via_api
            if prompt_wrap is not None and not resolved_prompt.skip_recipe_wrap:
                generation_prompt = prompt_wrap(generation_prompt)
                # The wrap must be the last text mutation: on the API-encoding path
                # resolved_prompt.prompt is the *raw* user scene with
                # enhance_via_api=True, which would let /prompt-embedding rewrite the
                # scaffold we just prepended. Encode the wrapped prompt as-is instead
                # so the style lock survives. Home auto-enhance rewrites locally
                # before this wrap when the enhancer checkpoint is installed.
                if (
                    resolved_prompt.enhance_via_api
                    and not resolved_prompt.enhanced_locally
                    and req.promptProvenance != "enhanced"
                ):
                    logger.info(
                        "Style recipe sent the typed scene: no local prompt enhancer "
                        "is installed, and API enhance would rewrite the scaffold"
                    )
                enhance_via_api = False
            self._generation.raise_if_cancelled()
            self._pipelines.load_gpu_pipeline(
                "fast", loras=loras, mode=load_mode, ltx_model_id=job_model_id
            )
            self._generation.start_generation(generation_id)

            written_path = self.generate_video(
                prompt=generation_prompt,
                enhance_via_api=enhance_via_api,
                image=image,
                last_image=last_image,
                keyframe_images=keyframe_images,
                height=height,
                width=width,
                num_frames=num_frames,
                fps=fps,
                seed=seed,
                camera_motion=req.cameraMotion,
                negative_prompt=req.negativePrompt,
                loras=loras,
                output_path=output_path,
                load_mode=load_mode,
                ltx_model_id=job_model_id,
                skip_stage_2=canvas.skip_stage_2,
            )

            self._generation.complete_generation(written_path)
            return GenerateVideoCompleteResponse(status="complete", video_path=written_path)

        except HTTPError as e:
            self._generation.fail_generation(e.detail)
            raise
        except Exception as e:
            self._generation.fail_generation(str(e))
            if is_cancel_exception(e):
                logger.info("Generation cancelled by user")
                return GenerateVideoCancelledResponse(status="cancelled")

            raise HTTPError(500, str(e)) from e

    def _resolve_loras(
        self,
        loras: list[LoraEntry],
        *,
        local_model_id: LTXLocalModelId | None = None,
    ) -> list[tuple[str, float]]:
        if loras:
            model_id = local_model_id or self._active_ltx_model_id()
            if model_id is None:
                raise HTTPError(409, "NO_DOWNLOADED_LTX_MODEL")
            if not supports(local_caps(model_id), "user_loras"):
                raise HTTPError(
                    409,
                    "User LoRAs are not supported for the selected offering.",
                    code="UNSUPPORTED_USER_LORAS",
                )
        try:
            return [(str(resolve_lora_ref(self.models_dir, e.ref)), e.scale) for e in loras]
        except ValueError as exc:
            raise HTTPError(400, str(exc)) from exc

    def generate_video(
        self,
        prompt: str,
        enhance_via_api: bool,
        image: Image.Image | None,
        height: int,
        width: int,
        num_frames: int | AutoDurationSpec,
        fps: float,
        seed: int,
        camera_motion: VideoCameraMotion,
        negative_prompt: str,
        loras: list[tuple[str, float]] | None = None,
        last_image: Image.Image | None = None,
        keyframe_images: list[tuple[Image.Image, int, float]] | None = None,
        output_path: Path | None = None,
        *,
        load_mode: LocalGenerationMode,
        ltx_model_id: LTXLocalModelId | None = None,
        skip_stage_2: bool = False,
    ) -> str:
        t_total_start = time.perf_counter()
        gen_mode = "keyframes" if keyframe_images else "i2v" if image is not None else "t2v"
        frames_log = (
            f"auto {num_frames.min_seconds:g}-{num_frames.max_seconds:g}s"
            if isinstance(num_frames, AutoDurationSpec)
            else f"{num_frames} frames"
        )
        logger.info("[%s] Generation started (model=fast, %dx%d, %s, %d fps)", gen_mode, width, height, frames_log, int(fps))

        self._generation.raise_if_cancelled()

        if keyframe_images:
            images, temp_image_paths = self._keyframe_conditionings(keyframe_images)
        else:
            images, temp_image_paths = self._image_conditionings(
                first=image, last=last_image, num_frames=num_frames
            )

        if output_path is None:
            output_path = self._make_output_path()

        # Appended after any rewrite the caller already applied, so the enhancer can't
        # paraphrase the camera directive away.
        enhanced_prompt = prompt + self.config.camera_motion_prompts.get(camera_motion, "")

        try:
            self._generation.update_progress("loading_model", 5)
            t_load_start = time.perf_counter()
            pipeline_state = self._pipelines.load_gpu_pipeline(
                "fast", loras=loras, mode=load_mode, ltx_model_id=ltx_model_id
            )
            t_load_end = time.perf_counter()
            logger.info("[%s] Pipeline load: %.2fs", gen_mode, t_load_end - t_load_start)

            self._generation.update_progress("encoding_text", 10)
            encoding_method = (
                "api" if not self._text.should_use_local_encoding(ltx_model_id) else "local"
            )
            t_text_start = time.perf_counter()
            self._text.prepare_text_encoding(
                enhanced_prompt, enhance_prompt=enhance_via_api, model_id=ltx_model_id
            )
            t_text_end = time.perf_counter()
            logger.info("[%s] Text encoding (%s): %.2fs", gen_mode, encoding_method, t_text_end - t_text_start)

            self._generation.raise_if_cancelled()
            self._generation.update_progress("inference", 15)

            # Guard for the /64 two-stage grid. Half-way values round up: Python's round() is
            # half-to-even, which turned a 544 height into 512 and silently shipped a frame 32px
            # shorter (and off its stated aspect ratio) rather than the nearest legal size.
            height = snap_up_to_multiple(height, 64)
            width = snap_up_to_multiple(width, 64)

            t_inference_start = time.perf_counter()
            with log_heartbeat(f"{gen_mode} inference"):
                pipeline_state.pipeline.generate(
                    prompt=enhanced_prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=fps,
                    images=images,
                    output_path=str(output_path),
                    guide_all_images=bool(keyframe_images),
                    skip_stage_2=skip_stage_2,
                )
            t_inference_end = time.perf_counter()
            logger.info("[%s] Inference: %.2fs", gen_mode, t_inference_end - t_inference_start)

            # Denoiser interrupt cannot abort VAE decode / ffmpeg; a Stop after the last
            # denoise step still finishes encode, then this check drops the file.
            if self._generation.is_generation_cancelled():
                if output_path.exists():
                    output_path.unlink()
                raise GenerationCancelledError()

            t_total_end = time.perf_counter()
            logger.info("[%s] Total generation: %.2fs (load=%.2fs, text=%.2fs, inference=%.2fs)",
                        gen_mode, t_total_end - t_total_start,
                        t_load_end - t_load_start, t_text_end - t_text_start, t_inference_end - t_inference_start)

            self._generation.update_progress("complete", 100)
            return str(output_path)
        finally:
            self._text.clear_api_embeddings()
            self._unlink_temp_paths(temp_image_paths)

    def _generate_a2v(
        self,
        req: GenerateVideoRequest,
        duration: int,
        fps: int,
        *,
        audio_path: str,
        generation_id: str,
        output_path: Path | None = None,
        num_frames: int | None = None,
        audio_duration_seconds: float | None = None,
        local_model_id: LTXLocalModelId | None = None,
        explore_generation: bool = False,
    ) -> GenerateVideoResponse:
        model_id = local_model_id
        if model_id is None:
            raise HTTPError(409, "NO_DOWNLOADED_LTX_MODEL")
        if not supports(local_caps(model_id), "a2v"):
            raise HTTPError(
                409,
                "Audio-to-video is not supported for the selected offering.",
                code="UNSUPPORTED_A2V",
            )
        validated_audio_path = validate_audio_file(audio_path)
        audio_path_str = str(validated_audio_path)

        width, height = self._local_pixels(
            req.resolution,
            req.aspectRatio,
            invalid_code="INVALID_LOCAL_A2V_RESOLUTION",
            local_model_id=model_id,
            mode="a2v",
        )
        width = snap_up_to_multiple(width, 64)
        height = snap_up_to_multiple(height, 64)

        if num_frames is None:
            num_frames = self._compute_num_frames(duration, fps)
        if audio_duration_seconds is not None:
            heard = self._prefix_a2v_audio_seconds(
                model_id=model_id,
                resolution=req.resolution,
                fps=req.fps,
                audio_duration_seconds=audio_duration_seconds,
            )
            if heard is None:
                raise HTTPError(
                    422,
                    "Audio is too long for audio-to-video.",
                    code="INVALID_VIDEO_GENERATION_SPEC",
                )
            audio_duration_seconds = heard

        image = None
        image_path = normalize_optional_path(req.imagePath)
        if image_path:
            image = self._prepare_image(image_path, width, height)

        last_image = None
        last_image_path = normalize_optional_path(req.lastImagePath)
        if last_image_path:
            last_image = self._prepare_image(last_image_path, width, height)

        seed = req.seed if req.seed is not None else self._resolve_seed()
        loras = self._resolve_loras(req.loras, local_model_id=model_id)

        temp_image_paths: list[str] = []

        try:
            neg = req.negativePrompt if req.negativePrompt else self.config.default_negative_prompt

            images, temp_image_paths = self._image_conditionings(
                first=image, last=last_image, num_frames=num_frames
            )

            load_mode = self._video_job_load_mode(width, height, num_frames)
            # Same ordering as Fast: budget 422 before Gemma so an over-size cell
            # does not evict a warmed pipeline for a rewrite that will not run.
            resolved_prompt = self._prompt_enhancement.resolve_for_generation(
                req.prompt,
                provenance=req.promptProvenance,
                generation_seed=seed,
                explicit_generation_seed=req.seed is not None,
                image_path=image_path,
                last_image_path=last_image_path,
                lora_catalog_ids=_lora_catalog_ids(req.loras),
                local_model_id=model_id,
                explore_generation=explore_generation,
                audio_path=audio_path_str,
            )
            enhanced_prompt = resolved_prompt.prompt + self.config.camera_motion_prompts.get(req.cameraMotion, "")

            self._generation.raise_if_cancelled()
            a2v_state = self._pipelines.load_a2v_pipeline(
                loras=loras, mode=load_mode, ltx_model_id=model_id
            )
            self._generation.start_generation(generation_id)

            if output_path is None:
                output_path = self._make_output_path()

            self._generation.update_progress("loading_model", 5)
            self._generation.update_progress("encoding_text", 10)
            self._text.prepare_text_encoding(
                enhanced_prompt,
                enhance_prompt=resolved_prompt.enhance_via_api,
                model_id=model_id,
            )
            self._generation.raise_if_cancelled()
            self._generation.update_progress("inference", 15)

            with log_heartbeat("a2v inference"):
                a2v_state.pipeline.generate(
                    prompt=enhanced_prompt,
                    negative_prompt=neg,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=fps,
                    num_inference_steps=distilled_total_steps(),
                    images=images,
                    audio_path=audio_path_str,
                    audio_start_time=0.0,
                    audio_max_duration=(
                        None
                        if audio_duration_seconds is None
                        else max(num_frames / fps, audio_duration_seconds)
                    ),
                    output_path=str(output_path),
                )

            # Denoiser interrupt cannot abort VAE decode / ffmpeg; a Stop after the last
            # denoise step still finishes encode, then this check drops the file.
            if self._generation.is_generation_cancelled():
                if output_path.exists():
                    output_path.unlink()
                raise GenerationCancelledError()

            self._generation.update_progress("complete", 100)
            self._generation.complete_generation(str(output_path))
            return GenerateVideoCompleteResponse(status="complete", video_path=str(output_path))

        except HTTPError as e:
            self._generation.fail_generation(e.detail)
            raise
        except Exception as e:
            self._generation.fail_generation(str(e))
            if is_cancel_exception(e):
                logger.info("Generation cancelled by user")
                return GenerateVideoCancelledResponse(status="cancelled")
            raise HTTPError(500, str(e)) from e
        finally:
            self._text.clear_api_embeddings()
            self._unlink_temp_paths(temp_image_paths)

    def _image_conditionings(
        self,
        *,
        first: Image.Image | None,
        last: Image.Image | None,
        num_frames: int | AutoDurationSpec,
    ) -> tuple[list[ImageConditioningInput], list[str]]:
        temp_paths: list[str] = []
        images: list[ImageConditioningInput] = []
        if first is not None:
            path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
            first.save(path)
            temp_paths.append(path)
            images.append(ImageConditioningInput(path=path, frame_idx=0, strength=1.0))
        if last is not None:
            if first is None:
                raise HTTPError(
                    422,
                    "Last frame requires a first-frame image",
                    code="INVALID_VIDEO_GENERATION_SPEC",
                )
            if not isinstance(num_frames, int):
                raise HTTPError(
                    422,
                    "Last frame cannot be combined with automatic duration",
                    code="INVALID_VIDEO_GENERATION_SPEC",
                )
            path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
            last.save(path)
            temp_paths.append(path)
            images.append(ImageConditioningInput(path=path, frame_idx=num_frames - 1, strength=1.0))
        return images, temp_paths

    def _keyframe_conditionings(
        self,
        frames: list[tuple[Image.Image, int, float]],
    ) -> tuple[list[ImageConditioningInput], list[str]]:
        temp_paths: list[str] = []
        images: list[ImageConditioningInput] = []
        for image, frame_idx, strength in frames:
            path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
            image.save(path)
            temp_paths.append(path)
            images.append(ImageConditioningInput(path=path, frame_idx=frame_idx, strength=strength))
        return images, temp_paths

    @staticmethod
    def _unlink_temp_paths(paths: list[str]) -> None:
        for path in paths:
            if os.path.exists(path):
                os.unlink(path)

    def _prepare_image(self, image_path: str, width: int, height: int) -> Image.Image:
        validated_path = validate_image_file(image_path)
        try:
            img = open_oriented_rgb(validated_path)
        except Exception:
            raise HTTPError(400, f"Invalid image file: {image_path}") from None
        img_w, img_h = img.size
        target_ratio = width / height
        img_ratio = img_w / img_h
        if img_ratio > target_ratio:
            new_h = height
            new_w = int(img_w * (height / img_h))
        else:
            new_w = width
            new_h = int(img_h * (width / img_w))
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        left = (new_w - width) // 2
        top = (new_h - height) // 2
        return resized.crop((left, top, left + width, top + height))

    @staticmethod
    def _make_generation_id() -> str:
        return uuid.uuid4().hex[:8]

    def _budget_frame_count(self, num_frames: int | AutoDurationSpec, fps: int) -> int:
        if isinstance(num_frames, AutoDurationSpec):
            return compute_num_frames(int(num_frames.max_seconds), fps)
        return num_frames

    def _video_job_load_mode(
        self, width: int, height: int, frames: int
    ) -> LocalGenerationMode:
        """422 or pick full vs stream for this job. Does not change process mode."""
        darwin = self.config.darwin_unified_memory
        memory_gb = memory_gb_for_job(
            vram_gb=self.config.vram_gb,
            available_ram_gb=self.config.available_ram_gb,
            darwin=darwin,
        )
        decision = decide_video_job(
            width,
            height,
            frames,
            memory_gb=memory_gb,
            process_mode=self.config.local_generations_mode,
            darwin=darwin,
        )
        outcome = "reject" if isinstance(decision, VideoJobReject) else decision.mode
        logger.info(
            "[video] job budget seq=%.0f full=%.1fGiB stream=%.1fGiB -> %s",
            decision.seq,
            decision.full_gib,
            decision.stream_gib,
            outcome,
        )
        if isinstance(decision, VideoJobReject):
            if decision.reason == "unsupported":
                raise HTTPError(
                    422,
                    LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
                    code=LOCAL_GENERATION_UNSUPPORTED,
                )
            raise HTTPError(422, VIDEO_JOB_TOO_LARGE_MESSAGE, code=VIDEO_JOB_TOO_LARGE)
        return decision.mode

    @staticmethod
    def _compute_num_frames(duration: int, fps: int) -> int:
        return compute_num_frames(duration, fps)

    def _make_output_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.config.outputs_dir / f"ltx2_video_{timestamp}_{self._make_generation_id()}.mp4"

    def _generate_forced_api(self, req: GenerateVideoRequest) -> GenerateVideoResponse:
        with self._generation.reserved_generation_start():

            generation_id = self._make_generation_id()
            self._generation.start_api_generation(generation_id)

            audio_path = normalize_optional_path(req.audioPath)
            image_path = normalize_optional_path(req.imagePath)
            last_image_path = normalize_optional_path(req.lastImagePath)
            has_input_audio = bool(audio_path)
            has_input_image = bool(image_path)

            try:
                self._generation.update_progress("validating_request", 5, None, None)

                api_key = self.state.app_settings.ltx_api_key.strip()
                logger.info("Forced API generation route selected (key_present=%s)", bool(api_key))
                if not api_key:
                    raise missing_ltx_api_key_error()

                requested_model = req.model
                api_model_id = FORCED_API_MODEL_MAP.get(requested_model)
                if api_model_id is None:
                    raise HTTPError(500, "INVALID_FORCED_API_MODEL_CONFIG")

                resolution_label = req.resolution
                resolution_by_aspect = FORCED_API_RESOLUTION_MAP.get(requested_model, {}).get(resolution_label)
                if resolution_by_aspect is None:
                    raise HTTPError(500, "INVALID_FORCED_API_RESOLUTION_CONFIG")

                aspect_ratio = req.aspectRatio
                api_resolution = resolution_by_aspect.get(aspect_ratio)
                if api_resolution is None:
                    raise HTTPError(400, "INVALID_FORCED_API_ASPECT_RATIO")

                prompt = req.prompt
                enhance_prompt = self._prompt_enhancement.api_enhance_prompt(
                    prompt,
                    provenance=req.promptProvenance,
                    image_path=image_path,
                    audio_path=audio_path,
                )

                self._generation.raise_if_cancelled()

                if has_input_audio:
                    validated_audio_path = validate_audio_file(audio_path)
                    validated_image_path: Path | None = None
                    if image_path is not None:
                        validated_image_path = validate_image_file(image_path)

                    self._generation.update_progress("uploading_audio", 20, None, None)
                    audio_uri = self._ltx_api_client.upload_file(
                        api_key=api_key,
                        file_path=str(validated_audio_path),
                    )
                    image_uri: str | None = None
                    last_frame_uri: str | None = None
                    if validated_image_path is not None:
                        self._generation.update_progress("uploading_image", 35, None, None)
                        image_uri = _upload_oriented_image(
                            self._ltx_api_client,
                            api_key=api_key,
                            path=validated_image_path,
                        )
                    if last_image_path is not None:
                        self._generation.update_progress("uploading_image", 45, None, None)
                        last_frame_uri = _upload_oriented_image(
                            self._ltx_api_client,
                            api_key=api_key,
                            path=validate_image_file(last_image_path),
                        )
                    self._generation.update_progress("inference", 55, None, None)
                    video_bytes = self._ltx_api_client.generate_audio_to_video(
                        api_key=api_key,
                        prompt=prompt,
                        audio_uri=audio_uri,
                        image_uri=image_uri,
                        last_frame_uri=last_frame_uri,
                        model=api_model_id,
                        resolution=api_resolution,
                        enhance_prompt=enhance_prompt,
                    )
                    self._generation.update_progress("downloading_output", 85, None, None)
                elif has_input_image:
                    validated_image_path = validate_image_file(image_path)

                    duration = req.duration
                    fps = req.fps

                    generate_audio = req.audio
                    self._generation.update_progress("uploading_image", 20, None, None)
                    image_uri = _upload_oriented_image(
                        self._ltx_api_client,
                        api_key=api_key,
                        path=validated_image_path,
                    )
                    last_frame_uri: str | None = None
                    if last_image_path is not None:
                        self._generation.update_progress("uploading_image", 35, None, None)
                        last_frame_uri = _upload_oriented_image(
                            self._ltx_api_client,
                            api_key=api_key,
                            path=validate_image_file(last_image_path),
                        )
                    self._generation.update_progress("inference", 55, None, None)
                    video_bytes = self._ltx_api_client.generate_image_to_video(
                        api_key=api_key,
                        prompt=prompt,
                        image_uri=image_uri,
                        last_frame_uri=last_frame_uri,
                        model=api_model_id,
                        resolution=api_resolution,
                        duration=None if duration is None else float(duration),
                        fps=float(fps),
                        generate_audio=generate_audio,
                        camera_motion=req.cameraMotion,
                        enhance_prompt=enhance_prompt,
                    )
                    self._generation.update_progress("downloading_output", 85, None, None)
                else:
                    duration = req.duration
                    fps = req.fps

                    generate_audio = req.audio
                    self._generation.update_progress("inference", 55, None, None)
                    video_bytes = self._ltx_api_client.generate_text_to_video(
                        api_key=api_key,
                        prompt=prompt,
                        model=api_model_id,
                        resolution=api_resolution,
                        duration=None if duration is None else float(duration),
                        fps=float(fps),
                        generate_audio=generate_audio,
                        camera_motion=req.cameraMotion,
                        enhance_prompt=enhance_prompt,
                    )
                    self._generation.update_progress("downloading_output", 85, None, None)

                self._generation.raise_if_cancelled()

                output_path = self._write_forced_api_video(video_bytes)
                if self._generation.is_generation_cancelled():
                    output_path.unlink(missing_ok=True)
                    raise GenerationCancelledError()

                self._generation.update_progress("complete", 100, None, None)
                self._generation.complete_generation(str(output_path))
                return GenerateVideoCompleteResponse(status="complete", video_path=str(output_path))
            except HTTPError as e:
                self._generation.fail_generation(e.detail)
                raise
            except LTXAPIClientError as e:
                mapped_error = map_ltx_api_client_error(e)
                self._generation.fail_generation(mapped_error.detail)
                raise mapped_error from e
            except Exception as e:
                self._generation.fail_generation(str(e))
                if is_cancel_exception(e):
                    logger.info("Generation cancelled by user")
                    return GenerateVideoCancelledResponse(status="cancelled")
                raise HTTPError(500, str(e)) from e

    def _write_forced_api_video(self, video_bytes: bytes) -> Path:
        output_path = self._make_output_path()
        output_path.write_bytes(video_bytes)
        return output_path
