"""IC-LoRA endpoints orchestration handler."""

from __future__ import annotations

import base64
import logging
import uuid
from pathlib import Path
from threading import RLock

from server_utils.output_naming import make_output_path
from api_types import (
    IcLoraDownloadRequest,
    IcLoraDownloadResponse,
    IcLoraExtractRequest,
    IcLoraExtractResponse,
    IcLoraGenerateRequest,
    IcLoraGenerateResponse,
    IcLoraListResponse,
    IcLoraModel,
    ImageConditioningInput,
)
from _routes._errors import HTTPError
from handlers.base import StateHandlerBase
from handlers.generation_handler import GenerationHandler
from handlers.pipelines_handler import PipelinesHandler
from handlers.text_handler import TextHandler
from services.interfaces import IcLoraModelDownloader, VideoProcessor
from state.app_state_types import AppState

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
        ic_lora_model_downloader: IcLoraModelDownloader,
        ic_lora_dir: Path,
        outputs_dir: Path,
    ) -> None:
        super().__init__(state, lock)
        self._generation = generation_handler
        self._pipelines = pipelines_handler
        self._text = text_handler
        self._video_processor = video_processor
        self._ic_lora_model_downloader = ic_lora_model_downloader
        self._ic_lora_dir = ic_lora_dir
        self._outputs_dir = outputs_dir

    def list_models(self) -> IcLoraListResponse:
        models = self._ic_lora_model_downloader.list_models(self._ic_lora_dir)
        return IcLoraListResponse(models=[IcLoraModel(**model) for model in models], directory=str(self._ic_lora_dir))

    def download_model(self, req: IcLoraDownloadRequest) -> IcLoraDownloadResponse:
        try:
            payload = self._ic_lora_model_downloader.download_model(req.model, self._ic_lora_dir)
            return IcLoraDownloadResponse(**payload)
        except ValueError as exc:
            logger.warning("IC-LoRA download request rejected for model '%s': %s", req.model, exc)
            raise HTTPError(400, str(exc))
        except Exception as exc:
            raise HTTPError(500, f"Download failed: {exc}") from exc

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

        if req.conditioning_type == "canny":
            result = self._video_processor.apply_canny(frame)
        elif req.conditioning_type == "depth":
            result = self._video_processor.apply_depth(frame)
        else:
            result = frame

        conditioning = self._video_processor.encode_frame_jpeg(result, quality=85)
        original = self._video_processor.encode_frame_jpeg(frame, quality=85)

        return IcLoraExtractResponse(
            conditioning="data:image/jpeg;base64," + base64.b64encode(conditioning).decode("utf-8"),
            original="data:image/jpeg;base64," + base64.b64encode(original).decode("utf-8"),
            conditioning_type=req.conditioning_type,
            frame_time=req.frame_time,
        )

    def generate(self, req: IcLoraGenerateRequest) -> IcLoraGenerateResponse:
        if self._generation.is_generation_running():
            raise HTTPError(409, "Generation already in progress")

        video_path = Path(req.video_path)
        lora_path = Path(req.lora_path)
        if not video_path.exists():
            raise HTTPError(400, f"Video not found: {req.video_path}")
        if not lora_path.exists():
            raise HTTPError(400, f"LoRA not found: {req.lora_path}")

        generation_id = uuid.uuid4().hex[:8]

        try:
            ic_state = self._pipelines.load_ic_lora(str(lora_path))
            self._generation.start_generation(generation_id)
            self._generation.update_progress("loading_model", 5, 0, 1)

            s = self.state.app_settings
            use_api = not self._text.should_use_local_encoding()
            self._text.prepare_text_encoding(req.prompt, enhance_prompt=use_api and s.prompt_enhancer_enabled_t2v)

            cap = self._video_processor.open_video(str(video_path))
            info = self._video_processor.get_video_info(cap)
            if not cap.isOpened():
                raise HTTPError(400, f"Cannot open video: {video_path}")

            control_video_path = str(self._outputs_dir / f"_control_{req.conditioning_type}_{uuid.uuid4().hex[:8]}.mp4")
            writer = self._video_processor.create_writer(
                control_video_path,
                fourcc="mp4v",
                fps=float(info["fps"]),
                size=(int(info["width"]), int(info["height"])),
            )

            frame_idx = 0
            max_frames = min(int(info["frame_count"]), req.num_frames * 2)
            while frame_idx < max_frames:
                frame = self._video_processor.read_frame(cap)
                if frame is None:
                    break
                if req.conditioning_type == "canny":
                    control_frame = self._video_processor.apply_canny(frame)
                elif req.conditioning_type == "depth":
                    control_frame = self._video_processor.apply_depth(frame)
                else:
                    control_frame = frame
                writer.write(control_frame)
                frame_idx += 1

            self._video_processor.release(cap)
            self._video_processor.release(writer)

            images: list[ImageConditioningInput] = [
                ImageConditioningInput(path=img.path, frame_idx=int(img.frame), strength=float(img.strength))
                for img in req.images
            ]

            self._generation.update_progress("inference", 15, 0, 1)

            height = round(req.height / 64) * 64
            width = round(req.width / 64) * 64
            output_path = make_output_path(self._outputs_dir, model="ic-lora", prompt=req.prompt, ext="mp4")

            ic_state.pipeline.generate(
                prompt=req.prompt,
                seed=req.seed,
                height=height,
                width=width,
                num_frames=req.num_frames,
                frame_rate=req.frame_rate,
                images=images,
                video_conditioning=[(control_video_path, req.conditioning_strength)],
                output_path=str(output_path),
            )

            try:
                Path(control_video_path).unlink()
            except Exception:
                logger.warning("Could not remove temporary control video: %s", control_video_path, exc_info=True)

            self._generation.update_progress("complete", 100, 1, 1)
            self._generation.complete_generation(str(output_path))
            return IcLoraGenerateResponse(status="complete", video_path=str(output_path))

        except HTTPError:
            self._generation.fail_generation("IC-LoRA generation failed")
            raise
        except Exception as exc:
            self._generation.fail_generation(str(exc))
            if "cancelled" in str(exc).lower():
                return IcLoraGenerateResponse(status="cancelled")
            raise HTTPError(500, f"Generation error: {exc}") from exc
        finally:
            self._text.clear_api_embeddings()
