"""Image generation orchestration handler."""

from __future__ import annotations

import io
import logging
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from PIL import Image as PILImage

from _routes._errors import HTTPError
from api_types import (
    GenerateImageCancelledResponse,
    GenerateImageCompleteResponse,
    GenerateImageRequest,
    GenerateImageResponse,
)
from handlers.base import StateHandlerBase
from handlers.generation_handler import GenerationHandler
from handlers.pipelines_handler import PipelinesHandler
from services.interfaces import ZitAPIClient
from server_utils.media_validation import validate_image_file
from services.services_utils import clamp_strength, compute_edit_dimensions, effective_edit_steps
from state.app_settings import should_image_generate_with_fal_api
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
        config: RuntimeConfig,
        zit_api_client: ZitAPIClient,
    ) -> None:
        super().__init__(state, lock, config)
        self._generation = generation_handler
        self._pipelines = pipelines_handler
        self._zit_api_client = zit_api_client

    def generate(self, req: GenerateImageRequest) -> GenerateImageResponse:
        with self._generation.reserved_generation_start():

            width = (req.width // 16) * 16
            height = (req.height // 16) * 16
            num_images = max(1, min(12, req.numImages))

            seed = self._resolve_seed()
            use_fal_api = should_image_generate_with_fal_api(
                force_api_generations=self.config.force_api_generations,
                settings=self.state.app_settings,
            )

            if req.imagePath is not None:
                return self._edit(
                    req=req,
                    seed=seed,
                    num_images=num_images,
                    use_fal_api=use_fal_api,
                )

            if use_fal_api:
                return self._generate_via_api(
                    prompt=req.prompt,
                    width=width,
                    height=height,
                    num_inference_steps=req.numSteps,
                    seed=seed,
                    num_images=num_images,
                )

            generation_id = uuid.uuid4().hex[:8]
            lease = None
            try:
                lease = self._generation.acquire_local_metal_lease(
                    generation_id=generation_id,
                    workload="image_generation",
                    reason="Torch local image generation",
                )
                self._pipelines.load_image_generation_pipeline_to_gpu()
                self._generation.start_generation(generation_id)
                output_paths = self.generate_image(
                    prompt=req.prompt,
                    width=width,
                    height=height,
                    num_inference_steps=req.numSteps,
                    seed=seed,
                    num_images=num_images,
                )
                self._generation.complete_generation(output_paths)
                return GenerateImageCompleteResponse(status="complete", image_paths=output_paths)
            except Exception as e:
                self._generation.fail_generation(str(e))
                if "cancelled" in str(e).lower():
                    logger.info("Image generation cancelled by user")
                    return GenerateImageCancelledResponse(status="cancelled")
                raise HTTPError(500, str(e)) from e
            finally:
                if lease is not None:
                    self._pipelines.cleanup_runtime_caches()
                    lease.close()

    def _edit(
        self,
        *,
        req: GenerateImageRequest,
        seed: int,
        num_images: int,
        use_fal_api: bool,
    ) -> GenerateImageResponse:
        assert req.imagePath is not None
        validate_image_file(req.imagePath)

        # Only strength == 0.0 can drive the pipeline's effective step count to 0; checked
        # unclamped here, before clamping, so the guard can't be masked — applied to both
        # providers so a request that FAL would silently mishandle fails the same way locally.
        if effective_edit_steps(req.numSteps, req.strength) < 1:
            raise HTTPError(422, "EDIT_STRENGTH_TOO_LOW_FOR_STEPS")
        strength = clamp_strength(req.strength)

        if use_fal_api:
            return self._edit_via_api(
                prompt=req.prompt,
                image_path=req.imagePath,
                strength=strength,
                num_inference_steps=req.numSteps,
                seed=seed,
                num_images=num_images,
            )

        generation_id = uuid.uuid4().hex[:8]
        lease = None
        try:
            lease = self._generation.acquire_local_metal_lease(
                generation_id=generation_id,
                workload="image_edit",
                reason="Torch local image editing",
            )
            self._pipelines.load_image_generation_pipeline_to_gpu()
            self._generation.start_generation(generation_id)
            output_paths = self.edit_image(
                prompt=req.prompt,
                image_path=req.imagePath,
                strength=strength,
                num_inference_steps=req.numSteps,
                seed=seed,
                num_images=num_images,
            )
            self._generation.complete_generation(output_paths)
            return GenerateImageCompleteResponse(status="complete", image_paths=output_paths)
        except Exception as e:
            self._generation.fail_generation(str(e))
            if "cancelled" in str(e).lower():
                logger.info("Image edit cancelled by user")
                return GenerateImageCancelledResponse(status="cancelled")
            raise HTTPError(500, str(e)) from e
        finally:
            if lease is not None:
                self._pipelines.cleanup_runtime_caches()
                lease.close()

    def edit_image(
        self,
        prompt: str,
        image_path: str,
        strength: float,
        num_inference_steps: int,
        seed: int,
        num_images: int,
    ) -> list[str]:
        if self._generation.is_generation_cancelled():
            raise RuntimeError("Generation was cancelled")

        self._generation.update_progress("loading_model", 5, 0, num_inference_steps)
        image_generation_pipeline = self._pipelines.load_image_generation_pipeline_to_gpu()
        self._generation.update_progress("inference", 15, 0, num_inference_steps)

        source = self._load_edit_source(image_path)

        def generate_one(seed_i: int) -> PILImage.Image:
            result = image_generation_pipeline.edit(
                prompt=prompt,
                image=source,
                strength=strength,
                num_inference_steps=num_inference_steps,
                seed=seed_i,
            )
            return result.images[0]

        return self._run_local_batch(num_images, seed, "zit_edit", generate_one)

    def generate_image(
        self,
        prompt: str,
        width: int,
        height: int,
        num_inference_steps: int,
        seed: int,
        num_images: int,
    ) -> list[str]:
        if self._generation.is_generation_cancelled():
            raise RuntimeError("Generation was cancelled")

        self._generation.update_progress("loading_model", 5, 0, num_inference_steps)
        image_generation_pipeline = self._pipelines.load_image_generation_pipeline_to_gpu()
        self._generation.update_progress("inference", 15, 0, num_inference_steps)

        def generate_one(seed_i: int) -> PILImage.Image:
            result = image_generation_pipeline.generate(
                prompt=prompt,
                height=height,
                width=width,
                guidance_scale=0.0,
                num_inference_steps=num_inference_steps,
                seed=seed_i,
            )
            return result.images[0]

        return self._run_local_batch(num_images, seed, "zit_image", generate_one)

    def _run_local_batch(
        self,
        num_images: int,
        seed: int,
        filename_prefix: str,
        generate_one: Callable[[int], PILImage.Image],
    ) -> list[str]:
        outputs: list[Path] = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            for i in range(num_images):
                if self._generation.is_generation_cancelled():
                    raise RuntimeError("Generation was cancelled")

                progress = 15 + int((i / num_images) * 80)
                self._generation.update_progress("inference", progress, i, num_images)

                image = generate_one(seed + i)
                output_path = self.config.outputs_dir / f"{filename_prefix}_{timestamp}_{uuid.uuid4().hex[:8]}.png"
                image.save(str(output_path))
                outputs.append(output_path)

            if self._generation.is_generation_cancelled():
                raise RuntimeError("Generation was cancelled")
        except Exception:
            for path in outputs:
                path.unlink(missing_ok=True)
            raise

        self._generation.update_progress("complete", 100, num_images, num_images)
        return [str(path) for path in outputs]

    def _load_edit_source(self, image_path: str) -> PILImage.Image:
        # Downscale to the same /16 target the local edit path uses before the pipeline
        # runs — keeps the base64 upload on the FAL path from inflating with source size.
        with PILImage.open(image_path) as raw:
            source = raw.convert("RGB")
        target_w, target_h = compute_edit_dimensions(source.width, source.height)
        return source.resize((target_w, target_h), PILImage.Resampling.LANCZOS)

    def _edit_via_api(
        self,
        *,
        prompt: str,
        image_path: str,
        strength: float,
        num_inference_steps: int,
        seed: int,
        num_images: int,
    ) -> GenerateImageResponse:
        source = self._load_edit_source(image_path)
        buffer = io.BytesIO()
        source.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        def call_provider(api_key: str, seed_i: int) -> bytes:
            return self._zit_api_client.generate_image_to_image(
                api_key=api_key,
                prompt=prompt,
                image_bytes=image_bytes,
                strength=strength,
                seed=seed_i,
                num_inference_steps=num_inference_steps,
            )

        return self._run_api_batch(seed, num_images, "zit_api_edit", call_provider)

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
        def call_provider(api_key: str, seed_i: int) -> bytes:
            return self._zit_api_client.generate_text_to_image(
                api_key=api_key,
                prompt=prompt,
                width=width,
                height=height,
                seed=seed_i,
                num_inference_steps=num_inference_steps,
            )

        return self._run_api_batch(seed, num_images, "zit_api_image", call_provider)

    def _run_api_batch(
        self,
        seed: int,
        num_images: int,
        filename_prefix: str,
        call_provider: Callable[[str, int], bytes],
    ) -> GenerateImageResponse:
        generation_id = uuid.uuid4().hex[:8]
        output_paths: list[Path] = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        settings = self.state.app_settings.model_copy(deep=True)

        try:
            self._generation.start_api_generation(generation_id)
            self._generation.update_progress("validating_request", 5, None, None)

            if not settings.fal_api_key.strip():
                raise HTTPError(500, "FAL_API_KEY_NOT_CONFIGURED")

            for idx in range(num_images):
                if self._generation.is_generation_cancelled():
                    raise RuntimeError("Generation was cancelled")

                inference_progress = 15 + int((idx / num_images) * 60)
                self._generation.update_progress("inference", inference_progress, None, None)
                result_bytes = call_provider(settings.fal_api_key, seed + idx)

                if self._generation.is_generation_cancelled():
                    raise RuntimeError("Generation was cancelled")

                download_progress = 75 + int(((idx + 1) / num_images) * 20)
                self._generation.update_progress("downloading_output", download_progress, None, None)

                output_path = self.config.outputs_dir / f"{filename_prefix}_{timestamp}_{uuid.uuid4().hex[:8]}.png"
                output_path.write_bytes(result_bytes)
                output_paths.append(output_path)

            self._generation.update_progress("complete", 100, None, None)
            self._generation.complete_generation([str(p) for p in output_paths])
            return GenerateImageCompleteResponse(status="complete", image_paths=[str(p) for p in output_paths])
        except HTTPError as e:
            self._generation.fail_generation(e.detail)
            for path in output_paths:
                path.unlink(missing_ok=True)
            raise
        except Exception as e:
            self._generation.fail_generation(str(e))
            for path in output_paths:
                path.unlink(missing_ok=True)
            if "cancelled" in str(e).lower():
                logger.info("Image generation cancelled by user")
                return GenerateImageCancelledResponse(status="cancelled")
            raise HTTPError(500, str(e)) from e
