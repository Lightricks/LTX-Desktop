"""Z-Image-Turbo image generation pipeline wrapper."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from diffusers.pipelines.auto_pipeline import ZImagePipeline  # type: ignore[reportUnknownVariableType]
from PIL.Image import Image as PILImage

from services.generation_interrupt import diffusers_step_callback
from services.services_utils import (
    ImagePipelineOutputLike,
    PILImageType,
    clamp_strength,
    get_device_type,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _ZImageOutput:
    images: Sequence[PILImageType]


class ZitImageGenerationPipeline:
    @staticmethod
    def create(
        model_path: str,
        device: str | None = None,
    ) -> "ZitImageGenerationPipeline":
        return ZitImageGenerationPipeline(model_path=model_path, device=device)

    def __init__(self, model_path: str, device: str | None = None) -> None:
        self._device: str | None = None
        self._cpu_offload_active = False
        self._img2img: Any = None
        self.pipeline = ZImagePipeline.from_pretrained(  # type: ignore[reportUnknownMemberType]
            model_path,
            torch_dtype=torch.bfloat16,
        )
        if device is not None:
            self.to(device)

    def _resolve_generator_device(self) -> str:
        # The configured runtime device is authoritative. With enable_model_cpu_offload()
        # the pipeline's _execution_device can read as "cpu", but the generator must live
        # on the actual compute device. Offload is CUDA-only; if we ever lose `_device`
        # while offload is active, CUDA is the only remaining compute device.
        if self._device is not None:
            return self._device
        if self._cpu_offload_active:
            return "cuda"

        execution_device = getattr(self.pipeline, "_execution_device", None)
        return get_device_type(execution_device)

    def _log_run(self, kind: str, *, width: int | None = None, height: int | None = None) -> None:
        size = f"{width}x{height} " if width is not None and height is not None else ""
        logger.info(
            "ZIT %s %sdevice=%s offload=%s",
            kind,
            size,
            self._device,
            self._cpu_offload_active,
        )

    @staticmethod
    def _normalize_output(output: object) -> ImagePipelineOutputLike:
        images = getattr(output, "images", None)
        if not isinstance(images, Sequence):
            raise RuntimeError("Unexpected ZIT pipeline output format: missing images sequence")

        images_list = cast(Sequence[object], images)
        validated_images: list[PILImageType] = []
        for image in images_list:
            if not isinstance(image, PILImage):
                raise RuntimeError("Unexpected ZIT pipeline output format: images must be PIL.Image instances")
            validated_images.append(image)

        return _ZImageOutput(images=validated_images)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        height: int,
        width: int,
        guidance_scale: float,
        num_inference_steps: int,
        seed: int,
    ) -> ImagePipelineOutputLike:
        # ZImagePipeline ignores guidance_scale, so we drop it explicitly.
        _ = guidance_scale
        self._log_run("generate", width=width, height=height)
        generator = torch.Generator(device=self._resolve_generator_device()).manual_seed(seed)
        pipeline = cast(Any, self.pipeline)
        output = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=0.0,
            num_inference_steps=num_inference_steps,
            generator=generator,
            output_type="pil",
            return_dict=True,
            callback_on_step_end=diffusers_step_callback,
        )
        return self._normalize_output(output)

    def _ensure_img2img_pipeline(self) -> Any:
        if self._img2img is not None:
            return self._img2img
        try:
            from diffusers import ZImageImg2ImgPipeline  # type: ignore[attr-defined]
        except Exception as e:
            raise RuntimeError("DIFFUSERS_IMG2IMG_UNAVAILABLE") from e
        # Reuse the loaded components so no second copy of the weights lands in VRAM.
        pipeline_any = cast(Any, self.pipeline)
        self._img2img = ZImageImg2ImgPipeline(**pipeline_any.components)  # type: ignore[reportUnknownMemberType]
        return self._img2img

    @torch.inference_mode()
    def edit(
        self,
        prompt: str,
        image: PILImageType,
        strength: float,
        num_inference_steps: int,
        seed: int,
    ) -> ImagePipelineOutputLike:
        img2img = self._ensure_img2img_pipeline()
        self._log_run("edit")

        generator = torch.Generator(device=self._resolve_generator_device()).manual_seed(seed)
        output = img2img(
            prompt=prompt,
            image=image,
            strength=clamp_strength(strength),
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,  # Turbo is guidance-free; img2img defaults to 5.0.
            generator=generator,
            output_type="pil",
            return_dict=True,
            callback_on_step_end=diffusers_step_callback,
        )
        return self._normalize_output(output)

    def to(self, device: str) -> None:
        runtime_device = get_device_type(device)
        if runtime_device == "cuda":
            self.pipeline.enable_model_cpu_offload()  # type: ignore[reportUnknownMemberType]
            self._cpu_offload_active = True
        else:
            # MPS is unified memory: cpu-offload keeps a host copy *and* a GPU copy of
            # the active module in the same RAM pool, which raised peak and jetsam'd Macs
            # during Z-Image step 0. Resident placement matches how the weights already
            # sit in RAM. CUDA still offloads (discrete VRAM ≠ host RAM).
            self._cpu_offload_active = False
            self.pipeline.to(runtime_device)  # type: ignore[reportUnknownMemberType]
        self._device = runtime_device
        # The base pipeline's offload hooks/device placement may have changed; drop the
        # cached img2img wrapper so it's rebuilt fresh (cheap — no weight reload) against
        # the current state next time edit() runs, instead of risking stale device info.
        self._img2img = None
