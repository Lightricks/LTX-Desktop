"""Image generation orchestration handler."""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from PIL import Image

from _routes._errors import HTTPError
from api_types import GenerateImageRequest, GenerateImageResponse
from handlers.base import StateHandlerBase
from handlers.generation_handler import GenerationHandler
from handlers.pipelines_handler import PipelinesHandler
from server_utils.output_naming import make_output_path
from services.interfaces import ImageAPIClient
from state.app_state_types import AppState

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)


class ImageGenerationHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        generation_handler: GenerationHandler,
        pipelines_handler: PipelinesHandler,
        outputs_dir: Path,
        config: RuntimeConfig,
        image_api_client: ImageAPIClient,
    ) -> None:
        super().__init__(state, lock)
        self._generation = generation_handler
        self._pipelines = pipelines_handler
        self._outputs_dir = outputs_dir
        self._config = config
        self._image_api_client = image_api_client

    def generate(self, req: GenerateImageRequest) -> GenerateImageResponse:
        if self._generation.is_generation_running():
            raise HTTPError(409, "Generation already in progress")

        width = (req.width // 16) * 16
        height = (req.height // 16) * 16
        num_images = max(1, min(12, req.numImages))

        generation_id = uuid.uuid4().hex[:8]
        settings = self.state.app_settings.model_copy(deep=True)
        if settings.seed_locked:
            seed = settings.locked_seed
            logger.info("Using locked seed for image: %s", seed)
        else:
            seed = int(time.time()) % 2147483647

        if self._config.force_api_generations:
            return self._generate_via_api(
                prompt=req.prompt,
                width=width,
                height=height,
                num_inference_steps=req.numSteps,
                seed=seed,
                num_images=num_images,
            )

        try:
            image_model = settings.image_model
            self._pipelines.load_image_model_to_gpu(image_model)
            self._generation.start_generation(generation_id)
            output_paths = self.generate_image(
                prompt=req.prompt,
                width=width,
                height=height,
                num_inference_steps=req.numSteps,
                seed=seed,
                num_images=num_images,
                lora_path=req.loraPath,
                lora_weight=req.loraWeight,
                source_image_path=req.sourceImagePath,
                strength=req.strength,
                image_model=image_model,
            )
            self._generation.complete_generation(output_paths)
            return GenerateImageResponse(status="complete", image_paths=output_paths)
        except Exception as e:
            self._generation.fail_generation(str(e))
            if "cancelled" in str(e).lower():
                logger.info("Image generation cancelled by user")
                return GenerateImageResponse(status="cancelled")
            raise HTTPError(500, str(e)) from e

    def generate_image(
        self,
        prompt: str,
        width: int,
        height: int,
        num_inference_steps: int,
        seed: int | None,
        num_images: int,
        lora_path: str | None = None,
        lora_weight: float = 1.0,
        source_image_path: str | None = None,
        strength: float = 0.65,
        image_model: str = "flux-klein-9b",
    ) -> list[str]:
        if self._generation.is_generation_cancelled():
            raise RuntimeError("Generation was cancelled")

        self._generation.update_progress("preparing_gpu", 3, 0, num_inference_steps)
        pipeline = self._pipelines.load_image_model_to_gpu(
            model_name=image_model,
            on_phase=lambda phase: self._generation.update_progress(phase, 5, 0, num_inference_steps),
        )

        if lora_path:
            logger.info("Loading LoRA: %s (weight=%.2f)", lora_path, lora_weight)
            self._generation.update_progress("loading_lora", 10, 0, num_inference_steps)
            pipeline.load_lora(lora_path, weight=lora_weight)
        else:
            pipeline.unload_lora()

        # Load and prepare source image for img2img
        source_image = None
        if source_image_path:
            self._generation.update_progress("encoding_image", 12, 0, num_inference_steps)
            source_image = Image.open(source_image_path).convert("RGB")
            width = (source_image.width // 16) * 16
            height = (source_image.height // 16) * 16
            source_image = source_image.resize((width, height), Image.Resampling.LANCZOS)

        self._generation.update_progress("inference", 15, 0, num_inference_steps)

        if seed is None:
            seed = int(time.time()) % 2147483647

        is_flux = "flux" in image_model
        is_edit = source_image is not None
        if is_flux:
            model_label = "flux-klein-edit" if is_edit else "flux-klein"
        else:
            model_label = "zit-edit" if is_edit else "zit"

        # FLUX Klein uses guidance_scale=4.0; ZIT uses 0.0
        guidance = 4.0 if is_flux else 0.0
        outputs: list[str] = []

        for i in range(num_images):
            if self._generation.is_generation_cancelled():
                raise RuntimeError("Generation was cancelled")

            progress = 15 + int((i / num_images) * 80)
            self._generation.update_progress("inference", progress, i, num_images)

            if source_image is not None:
                result = pipeline.img2img(
                    prompt=prompt,
                    image=source_image,
                    strength=strength,
                    height=height,
                    width=width,
                    guidance_scale=guidance,
                    num_inference_steps=num_inference_steps,
                    seed=seed + i,
                )
            else:
                result = pipeline.generate(
                    prompt=prompt,
                    height=height,
                    width=width,
                    guidance_scale=guidance,
                    num_inference_steps=num_inference_steps,
                    seed=seed + i,
                )

            output_path = make_output_path(self._outputs_dir, model=model_label, prompt=prompt, ext="png")
            result.images[0].save(str(output_path))
            outputs.append(str(output_path))

        if self._generation.is_generation_cancelled():
            raise RuntimeError("Generation was cancelled")

        self._generation.update_progress("complete", 100, num_images, num_images)
        return outputs

    def _generate_via_api(
        self,
        *,
        prompt: str,
        width: int,
        height: int,
        num_inference_steps: int,
        seed: int,
        num_images: int,
    ) -> GenerateImageResponse:
        generation_id = uuid.uuid4().hex[:8]
        output_paths: list[Path] = []
        settings = self.state.app_settings.model_copy(deep=True)

        try:
            self._generation.start_api_generation(generation_id)
            self._generation.update_progress("validating_request", 5, None, None)

            if not settings.replicate_api_key.strip():
                raise HTTPError(500, "REPLICATE_API_KEY_NOT_CONFIGURED")

            for idx in range(num_images):
                if self._generation.is_generation_cancelled():
                    raise RuntimeError("Generation was cancelled")

                inference_progress = 15 + int((idx / num_images) * 60)
                self._generation.update_progress("inference", inference_progress, None, None)
                image_bytes = self._image_api_client.generate_text_to_image(
                    api_key=settings.replicate_api_key,
                    model=settings.image_model,
                    prompt=prompt,
                    width=width,
                    height=height,
                    seed=seed + idx,
                    num_inference_steps=num_inference_steps,
                )

                if self._generation.is_generation_cancelled():
                    raise RuntimeError("Generation was cancelled")

                download_progress = 75 + int(((idx + 1) / num_images) * 20)
                self._generation.update_progress("downloading_output", download_progress, None, None)

                output_path = make_output_path(self._outputs_dir, model=settings.image_model, prompt=prompt, ext="png")
                output_path.write_bytes(image_bytes)
                output_paths.append(output_path)

            self._generation.update_progress("complete", 100, None, None)
            self._generation.complete_generation([str(path) for path in output_paths])
            return GenerateImageResponse(status="complete", image_paths=[str(path) for path in output_paths])
        except HTTPError as e:
            self._generation.fail_generation(e.detail)
            raise
        except Exception as e:
            self._generation.fail_generation(str(e))
            if "cancelled" in str(e).lower():
                for path in output_paths:
                    path.unlink(missing_ok=True)
                logger.info("Image generation cancelled by user")
                return GenerateImageResponse(status="cancelled")
            raise HTTPError(500, str(e)) from e
