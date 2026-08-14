"""Video generation orchestration handler."""

from __future__ import annotations

import logging
import os

from frame_math import AutoDurationSpec, compute_num_frames, snap_up_to_multiple
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from PIL import Image

from api_types import (
    GenerateVideoCancelledResponse,
    GenerateVideoCompleteResponse,
    GenerateVideoModelsSpecsResponse,
    GenerateVideoRequest,
    GenerateVideoResponse,
    ImageConditioningInput,
    LoraEntry,
    LTXLocalModelId,
    LTXVideoGenResolution,
    VideoCameraMotion,
)
from runtime_config.ltx_capabilities import LtxAspectRatio, api_caps, local_caps, pixels_for, supports
from runtime_config.models_scanner import resolve_lora_ref
from _routes._errors import HTTPError
from api_model_specs import (
    FORCED_API_MODEL_MAP,
    build_generate_video_model_specs_response,
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
from runtime_config.model_download_specs import is_duration_head_ready, resolve_active_ltx_model_id
from server_utils.media_validation import (
    normalize_optional_path,
    validate_audio_file,
    validate_image_file,
)
from services.interfaces import LTXAPIClient
from services.ltx_api_client.ltx_api_client import LTXAPIClientError
from state.app_state_types import AppState
from state.app_settings import should_video_generate_with_ltx_api

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)


def _wxh(size: tuple[int, int]) -> str:
    return f"{size[0]}x{size[1]}"


def _forced_api_resolution_map() -> dict[str, dict[str, str]]:
    caps = api_caps("fast")
    return {
        resolution: {
            "16:9": _wxh(pixels_for(caps, resolution, "16:9")),
            "9:16": _wxh(pixels_for(caps, resolution, "9:16")),
        }
        for resolution in caps.resolution_pixels_16_9
    }


FORCED_API_RESOLUTION_MAP: dict[str, dict[str, str]] = _forced_api_resolution_map()
FORCED_API_ALLOWED_ASPECT_RATIOS = {"16:9", "9:16"}
_LTX_INSUFFICIENT_FUNDS_MESSAGE = "Your LTX API credits are insufficient for this generation. Buy more credits and try again."


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

    def _resolve_prompt_enhancement(
        self, prompt: str, *, image_path: str | None
    ) -> tuple[str, bool]:
        """Apply the enhancer setting, returning ``(prompt, enhance_via_api)``.

        The two encoding paths enhance in different places: API encoding rewrites server-side
        inside the same /prompt-embedding call, so it only needs the flag forwarded. Local
        encoding has no such step, so the rewrite happens here — without it the model sees the
        prompt as typed, which for a version captioned in 150-220 word audio-visual paragraphs
        (2.5) lands far outside its training distribution and it invents the rest.

        Must be called before the pipeline is loaded and before start_generation: the enhancer
        needs the VRAM a resident pipeline holds, and PipelinesHandler refuses to evict one
        while a generation is running.
        """
        settings = self.state.app_settings
        enabled = (
            settings.prompt_enhancer_enabled_i2v if image_path is not None
            else settings.prompt_enhancer_enabled_t2v
        )
        if not enabled:
            return prompt, False
        if not self._text.should_use_local_encoding():
            return prompt, True
        return self._prompt_enhancement.enhance_for_generation(prompt, image_path=image_path), False

    def _active_ltx_model_id(self) -> LTXLocalModelId | None:
        return resolve_active_ltx_model_id(
            self.models_dir, self.state.app_settings.active_ltx_model_id
        )

    def _duration_head_ready(self) -> bool:
        model_id = self._active_ltx_model_id()
        return model_id is not None and is_duration_head_ready(self.models_dir, model_id)

    def _local_pixels(
        self,
        resolution: LTXVideoGenResolution,
        aspect: LtxAspectRatio,
        *,
        invalid_code: str,
    ) -> tuple[int, int]:
        model_id = self._active_ltx_model_id()
        if model_id is None:
            raise HTTPError(409, "NO_DOWNLOADED_LTX_MODEL")
        try:
            return pixels_for(local_caps(model_id), resolution, aspect)
        except KeyError as exc:
            raise HTTPError(400, invalid_code) from exc

    def get_model_specs(self) -> GenerateVideoModelsSpecsResponse:
        return build_generate_video_model_specs_response(
            local_model_id=self._active_ltx_model_id(),
            duration_head_ready=self._duration_head_ready(),
        )

    def generate(self, req: GenerateVideoRequest) -> GenerateVideoResponse:
        use_api_specs = should_video_generate_with_ltx_api(
            force_api_generations=self.config.force_api_generations,
            settings=self.state.app_settings,
        )
        validation_error = validate_generate_video_request(
            req,
            use_api_specs=use_api_specs,
            local_model_id=None if use_api_specs else self._active_ltx_model_id(),
            duration_head_ready=False if use_api_specs else self._duration_head_ready(),
        )
        if validation_error is not None:
            raise HTTPError(422, validation_error, code="INVALID_VIDEO_GENERATION_SPEC")

        if use_api_specs:
            return self._generate_forced_api(req)

        with self._generation.reserved_generation_start():

            resolution = req.resolution
            duration = req.duration
            fps = req.fps

            audio_path = normalize_optional_path(req.audioPath)
            if audio_path:
                if duration is None:
                    raise HTTPError(
                        422,
                        "Automatic duration cannot be combined with audio-to-video",
                        code="INVALID_VIDEO_GENERATION_SPEC",
                    )
                return self._generate_a2v(req, duration, fps, audio_path=audio_path)

            logger.info("Resolution %s - using fast pipeline", resolution)

            width, height = self._local_pixels(
                resolution, req.aspectRatio, invalid_code="INVALID_LOCAL_RESOLUTION"
            )

            if duration is None:
                item = next(
                    candidate
                    for candidate in get_local_video_generation_model_specs(
                        self._active_ltx_model_id(),
                        duration_head_ready=self._duration_head_ready(),
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
            image_path = normalize_optional_path(req.imagePath)
            if image_path:
                image = self._prepare_image(image_path, width, height)
                logger.info("Image: %s -> %sx%s", image_path, width, height)

            generation_id = self._make_generation_id()
            seed = req.seed if req.seed is not None else self._resolve_seed()
            loras = self._resolve_loras(req.loras)

            # Before the pipeline loads and before the generation is marked running: local
            # enhancement needs the VRAM a resident pipeline holds, and evicting a pipeline is
            # refused once a generation is running.
            prompt, enhance_via_api = self._resolve_prompt_enhancement(
                req.prompt, image_path=image_path
            )

            try:
                self._pipelines.load_gpu_pipeline("fast", loras=loras)
                self._generation.start_generation(generation_id)

                output_path = self.generate_video(
                    prompt=prompt,
                    enhance_via_api=enhance_via_api,
                    image=image,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    fps=fps,
                    seed=seed,
                    camera_motion=req.cameraMotion,
                    negative_prompt=req.negativePrompt,
                    loras=loras,
                )

                self._generation.complete_generation(output_path)
                return GenerateVideoCompleteResponse(status="complete", video_path=output_path)

            except HTTPError as e:
                self._generation.fail_generation(e.detail)
                raise
            except Exception as e:
                self._generation.fail_generation(str(e))
                if "cancelled" in str(e).lower():
                    logger.info("Generation cancelled by user")
                    return GenerateVideoCancelledResponse(status="cancelled")

                raise HTTPError(500, str(e)) from e

    def _resolve_loras(self, loras: list[LoraEntry]) -> list[tuple[str, float]]:
        if loras:
            model_id = self._active_ltx_model_id()
            if model_id is None:
                raise HTTPError(409, "NO_DOWNLOADED_LTX_MODEL")
            if not supports(local_caps(model_id), "user_loras"):
                raise HTTPError(
                    409,
                    "User LoRAs are not supported for the active LTX model.",
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
    ) -> str:
        t_total_start = time.perf_counter()
        gen_mode = "i2v" if image is not None else "t2v"
        frames_log = (
            f"auto {num_frames.min_seconds:g}-{num_frames.max_seconds:g}s"
            if isinstance(num_frames, AutoDurationSpec)
            else f"{num_frames} frames"
        )
        logger.info("[%s] Generation started (model=fast, %dx%d, %s, %d fps)", gen_mode, width, height, frames_log, int(fps))

        if self._generation.is_generation_cancelled():
            raise RuntimeError("Generation was cancelled")

        total_steps = 8

        images: list[ImageConditioningInput] = []
        temp_image_path: str | None = None
        if image is not None:
            temp_image_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
            image.save(temp_image_path)
            images = [ImageConditioningInput(path=temp_image_path, frame_idx=0, strength=1.0)]

        output_path = self._make_output_path()

        # Appended after any rewrite the caller already applied, so the enhancer can't
        # paraphrase the camera directive away.
        enhanced_prompt = prompt + self.config.camera_motion_prompts.get(camera_motion, "")

        try:
            self._generation.update_progress("loading_model", 5, 0, total_steps)
            t_load_start = time.perf_counter()
            pipeline_state = self._pipelines.load_gpu_pipeline("fast", loras=loras)
            t_load_end = time.perf_counter()
            logger.info("[%s] Pipeline load: %.2fs", gen_mode, t_load_end - t_load_start)

            self._generation.update_progress("encoding_text", 10, 0, total_steps)
            encoding_method = "api" if not self._text.should_use_local_encoding() else "local"
            t_text_start = time.perf_counter()
            self._text.prepare_text_encoding(enhanced_prompt, enhance_prompt=enhance_via_api)
            t_text_end = time.perf_counter()
            logger.info("[%s] Text encoding (%s): %.2fs", gen_mode, encoding_method, t_text_end - t_text_start)

            self._generation.update_progress("inference", 15, 0, total_steps)

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
                )
            t_inference_end = time.perf_counter()
            logger.info("[%s] Inference: %.2fs", gen_mode, t_inference_end - t_inference_start)

            if self._generation.is_generation_cancelled():
                if output_path.exists():
                    output_path.unlink()
                raise RuntimeError("Generation was cancelled")

            t_total_end = time.perf_counter()
            logger.info("[%s] Total generation: %.2fs (load=%.2fs, text=%.2fs, inference=%.2fs)",
                        gen_mode, t_total_end - t_total_start,
                        t_load_end - t_load_start, t_text_end - t_text_start, t_inference_end - t_inference_start)

            self._generation.update_progress("complete", 100, total_steps, total_steps)
            return str(output_path)
        finally:
            self._text.clear_api_embeddings()
            if temp_image_path and os.path.exists(temp_image_path):
                os.unlink(temp_image_path)

    def _generate_a2v(
        self, req: GenerateVideoRequest, duration: int, fps: int, *, audio_path: str
    ) -> GenerateVideoResponse:
        model_id = self._active_ltx_model_id()
        if model_id is None:
            raise HTTPError(409, "NO_DOWNLOADED_LTX_MODEL")
        if not supports(local_caps(model_id), "a2v"):
            raise HTTPError(
                409,
                "Audio-to-video is not supported for the active LTX model.",
                code="UNSUPPORTED_A2V",
            )
        validated_audio_path = validate_audio_file(audio_path)
        audio_path_str = str(validated_audio_path)

        width, height = self._local_pixels(
            req.resolution, req.aspectRatio, invalid_code="INVALID_LOCAL_A2V_RESOLUTION"
        )

        num_frames = self._compute_num_frames(duration, fps)

        image = None
        temp_image_path: str | None = None
        image_path = normalize_optional_path(req.imagePath)
        if image_path:
            image = self._prepare_image(image_path, width, height)

        seed = req.seed if req.seed is not None else self._resolve_seed()
        loras = self._resolve_loras(req.loras)

        generation_id = self._make_generation_id()

        try:
            neg = req.negativePrompt if req.negativePrompt else self.config.default_negative_prompt

            images: list[ImageConditioningInput] = []
            if image is not None:
                temp_image_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
                image.save(temp_image_path)
                images = [ImageConditioningInput(path=temp_image_path, frame_idx=0, strength=1.0)]

            # Same ordering rule as the fast path: enhance before the pipeline takes the GPU
            # (and so before start_generation, which requires a pipeline to already be loaded).
            a2v_base_prompt, a2v_enhance = self._resolve_prompt_enhancement(
                req.prompt, image_path=image_path
            )
            enhanced_prompt = a2v_base_prompt + self.config.camera_motion_prompts.get(req.cameraMotion, "")

            a2v_state = self._pipelines.load_a2v_pipeline(loras=loras)
            self._generation.start_generation(generation_id)

            output_path = self._make_output_path()

            total_steps = 11  # distilled: 8 steps (stage 1) + 3 steps (stage 2)

            self._generation.update_progress("loading_model", 5, 0, total_steps)
            self._generation.update_progress("encoding_text", 10, 0, total_steps)
            self._text.prepare_text_encoding(enhanced_prompt, enhance_prompt=a2v_enhance)
            self._generation.update_progress("inference", 15, 0, total_steps)

            a2v_state.pipeline.generate(
                prompt=enhanced_prompt,
                negative_prompt=neg,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=fps,
                num_inference_steps=total_steps,
                images=images,
                audio_path=audio_path_str,
                audio_start_time=0.0,
                audio_max_duration=None,
                output_path=str(output_path),
            )

            if self._generation.is_generation_cancelled():
                if output_path.exists():
                    output_path.unlink()
                raise RuntimeError("Generation was cancelled")

            self._generation.update_progress("complete", 100, total_steps, total_steps)
            self._generation.complete_generation(str(output_path))
            return GenerateVideoCompleteResponse(status="complete", video_path=str(output_path))

        except HTTPError as e:
            self._generation.fail_generation(e.detail)
            raise
        except Exception as e:
            self._generation.fail_generation(str(e))
            if "cancelled" in str(e).lower():
                logger.info("Generation cancelled by user")
                return GenerateVideoCancelledResponse(status="cancelled")
            raise HTTPError(500, str(e)) from e
        finally:
            self._text.clear_api_embeddings()
            if temp_image_path and os.path.exists(temp_image_path):
                os.unlink(temp_image_path)

    def _prepare_image(self, image_path: str, width: int, height: int) -> Image.Image:
        validated_path = validate_image_file(image_path)
        try:
            img = Image.open(validated_path).convert("RGB")
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
            has_input_audio = bool(audio_path)
            has_input_image = bool(image_path)

            try:
                self._generation.update_progress("validating_request", 5, None, None)

                api_key = self.state.app_settings.ltx_api_key.strip()
                logger.info("Forced API generation route selected (key_present=%s)", bool(api_key))
                if not api_key:
                    raise HTTPError(400, "PRO_API_KEY_REQUIRED")

                requested_model = req.model
                api_model_id = FORCED_API_MODEL_MAP.get(requested_model)
                if api_model_id is None:
                    raise HTTPError(500, "INVALID_FORCED_API_MODEL_CONFIG")

                resolution_label = req.resolution
                resolution_by_aspect = FORCED_API_RESOLUTION_MAP.get(resolution_label)
                if resolution_by_aspect is None:
                    raise HTTPError(500, "INVALID_FORCED_API_RESOLUTION_CONFIG")

                aspect_ratio = req.aspectRatio
                if aspect_ratio not in FORCED_API_ALLOWED_ASPECT_RATIOS:
                    raise HTTPError(400, "INVALID_FORCED_API_ASPECT_RATIO")

                api_resolution = resolution_by_aspect[aspect_ratio]

                prompt = req.prompt

                if self._generation.is_generation_cancelled():
                    raise RuntimeError("Generation was cancelled")

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
                    if validated_image_path is not None:
                        self._generation.update_progress("uploading_image", 35, None, None)
                        image_uri = self._ltx_api_client.upload_file(
                            api_key=api_key,
                            file_path=str(validated_image_path),
                        )
                    self._generation.update_progress("inference", 55, None, None)
                    video_bytes = self._ltx_api_client.generate_audio_to_video(
                        api_key=api_key,
                        prompt=prompt,
                        audio_uri=audio_uri,
                        image_uri=image_uri,
                        model=api_model_id,
                        resolution=api_resolution,
                    )
                    self._generation.update_progress("downloading_output", 85, None, None)
                elif has_input_image:
                    validated_image_path = validate_image_file(image_path)

                    duration = req.duration
                    fps = req.fps

                    generate_audio = req.audio
                    self._generation.update_progress("uploading_image", 20, None, None)
                    image_uri = self._ltx_api_client.upload_file(
                        api_key=api_key,
                        file_path=str(validated_image_path),
                    )
                    self._generation.update_progress("inference", 55, None, None)
                    video_bytes = self._ltx_api_client.generate_image_to_video(
                        api_key=api_key,
                        prompt=prompt,
                        image_uri=image_uri,
                        model=api_model_id,
                        resolution=api_resolution,
                        duration=None if duration is None else float(duration),
                        fps=float(fps),
                        generate_audio=generate_audio,
                        camera_motion=req.cameraMotion,
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
                    )
                    self._generation.update_progress("downloading_output", 85, None, None)

                if self._generation.is_generation_cancelled():
                    raise RuntimeError("Generation was cancelled")

                output_path = self._write_forced_api_video(video_bytes)
                if self._generation.is_generation_cancelled():
                    output_path.unlink(missing_ok=True)
                    raise RuntimeError("Generation was cancelled")

                self._generation.update_progress("complete", 100, None, None)
                self._generation.complete_generation(str(output_path))
                return GenerateVideoCompleteResponse(status="complete", video_path=str(output_path))
            except HTTPError as e:
                self._generation.fail_generation(e.detail)
                raise
            except LTXAPIClientError as e:
                mapped_error = self._map_ltx_api_generation_error(e)
                self._generation.fail_generation(mapped_error.detail)
                raise mapped_error from e
            except Exception as e:
                self._generation.fail_generation(str(e))
                if "cancelled" in str(e).lower():
                    logger.info("Generation cancelled by user")
                    return GenerateVideoCancelledResponse(status="cancelled")
                raise HTTPError(500, str(e)) from e

    def _write_forced_api_video(self, video_bytes: bytes) -> Path:
        output_path = self._make_output_path()
        output_path.write_bytes(video_bytes)
        return output_path

    @staticmethod
    def _map_ltx_api_generation_error(exc: LTXAPIClientError) -> HTTPError:
        if exc.status_code == 402 and exc.provider_error_type == "insufficient_funds_error":
            return HTTPError(402, _LTX_INSUFFICIENT_FUNDS_MESSAGE, code="LTX_INSUFFICIENT_FUNDS")
        return HTTPError(exc.status_code, exc.detail)
