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
)
from _routes._errors import HTTPError
from handlers.base import StateHandlerBase
from handlers.generation_handler import GenerationHandler
from handlers.pipelines_handler import PipelinesHandler
from handlers.text_handler import TextHandler
from runtime_config.model_download_specs import resolve_model_path
from runtime_config.runtime_config import RuntimeConfig
from state.conditioning_cache import ConditioningCacheEntry, ConditioningCacheKey
from services.interfaces import VideoProcessor
from services.services_utils import FrameArray
from state.app_state_types import AppState, ICLoraState

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)


class IcLoraHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        generation_handler: GenerationHandler,
        pipelines_handler: PipelinesHandler,
        text_handler: TextHandler,
        video_processor: VideoProcessor,
        config: RuntimeConfig,
    ) -> None:
        super().__init__(state, lock, config)
        self._generation = generation_handler
        self._pipelines = pipelines_handler
        self._text = text_handler
        self._video_processor = video_processor

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
                if ic_state is None:
                    raise HTTPError(500, "Depth conditioning requires loaded IC-LoRA resources")
                return self._video_processor.apply_depth(frame, ic_state.depth_pipeline)
            case _:
                raise HTTPError(400, f"Unsupported conditioning_type: {conditioning_type}")

    def _require_ic_lora_model_paths(self) -> tuple[Path, Path]:
        lora_path = resolve_model_path(self.models_dir, self.config.model_download_specs,"ic_lora")
        depth_model_path = resolve_model_path(self.models_dir, self.config.model_download_specs,"depth_processor")
        if not lora_path.exists():
            raise HTTPError(400, f"IC-LoRA model not found: {lora_path}")
        if not depth_model_path.exists():
            raise HTTPError(400, f"Depth processor model not found: {depth_model_path}")
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
            lora_path, depth_model_path = self._require_ic_lora_model_paths()
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

    def _resolve_seed(self) -> int:
        settings = self.state.app_settings
        if settings.seed_locked:
            return settings.locked_seed
        if self.config.dev_mode:
            return 1000
        return int(time.time()) % 2147483647

    def generate(self, req: IcLoraGenerateRequest) -> IcLoraGenerateResponse:
        if self._generation.is_generation_running():
            raise HTTPError(409, "Generation already in progress")

        video_path = Path(req.video_path)
        if not video_path.exists():
            raise HTTPError(400, f"Video not found: {req.video_path}")
        lora_path, depth_model_path = self._require_ic_lora_model_paths()

        generation_id = uuid.uuid4().hex[:8]
        t_total_start = time.perf_counter()
        logger.info("[ic-lora] Generation started (conditioning=%s)", req.conditioning_type)

        try:
            t_load_start = time.perf_counter()
            ic_state = self._pipelines.load_ic_lora(
                str(lora_path),
                str(depth_model_path),
            )
            t_load_end = time.perf_counter()
            logger.info("[ic-lora] Pipeline load: %.2fs", t_load_end - t_load_start)

            self._generation.start_generation(generation_id)
            self._generation.update_progress("loading_model", 5, 0, 1)

            s = self.state.app_settings
            use_api = not self._text.should_use_local_encoding()
            encoding_method = "api" if use_api else "local"
            t_text_start = time.perf_counter()
            self._text.prepare_text_encoding(req.prompt, enhance_prompt=use_api and s.prompt_enhancer_enabled_t2v)
            t_text_end = time.perf_counter()
            logger.info("[ic-lora] Text encoding (%s): %.2fs", encoding_method, t_text_end - t_text_start)

            cap = self._video_processor.open_video(str(video_path))
            if not cap.isOpened():
                raise HTTPError(400, f"Cannot open video: {video_path}")
            info = self._video_processor.get_video_info(cap)
            input_width = int(info["width"])
            input_height = int(info["height"])

            cache_key = ConditioningCacheKey(str(video_path), req.conditioning_type)
            cached = ic_state.conditioning_cache.get(cache_key)

            t_preprocess_start = 0.0
            t_preprocess_end = 0.0

            if cached is not None:
                self._video_processor.release(cap)
                control_video_path = cached.control_video_path
                frame_count = cached.frame_count
                fps = cached.fps
                logger.info("[ic-lora] Conditioning cache hit for %s/%s", video_path.name, req.conditioning_type)
            else:
                t_preprocess_start = time.perf_counter()

                frame_count = int(info["frame_count"])
                fps = float(info["fps"])

                control_video_path = str(
                    self.config.outputs_dir / f"_control_{req.conditioning_type}_{uuid.uuid4().hex[:8]}.mp4"
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
                    control_frame = self._build_conditioning_frame(frame, req.conditioning_type, ic_state)
                    writer.write(control_frame)
                    frame_idx += 1

                self._video_processor.release(cap)
                self._video_processor.release(writer)
                t_preprocess_end = time.perf_counter()
                logger.info(
                    "[ic-lora] Preprocessing (%s, %d frames): %.2fs",
                    req.conditioning_type, frame_idx, t_preprocess_end - t_preprocess_start,
                )

                ic_state.conditioning_cache.put(
                    cache_key, ConditioningCacheEntry(control_video_path, frame_count, fps)
                )

            images: list[ImageConditioningInput] = [
                ImageConditioningInput(path=img.path, frame_idx=int(img.frame), strength=float(img.strength))
                for img in req.images
            ]

            self._generation.update_progress("inference", 15, 0, 1)

            width = 768
            height = round(width * input_height / input_width / 128) * 128
            height = max(height, 128)

            output_path = (
                self.config.outputs_dir / f"ic_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.mp4"
            )

            t_inference_start = time.perf_counter()
            ic_state.pipeline.generate(
                prompt=req.prompt,
                seed=self._resolve_seed(),
                height=height,
                width=width,
                num_frames=frame_count,
                frame_rate=fps,
                images=images,
                video_conditioning=[(control_video_path, req.conditioning_strength)],
                output_path=str(output_path),
            )
            t_inference_end = time.perf_counter()
            logger.info("[ic-lora] Inference: %.2fs", t_inference_end - t_inference_start)

            t_total_end = time.perf_counter()
            preprocess_time = (t_preprocess_end - t_preprocess_start) if cached is None else 0.0
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
