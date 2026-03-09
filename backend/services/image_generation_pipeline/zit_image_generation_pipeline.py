"""Z-Image-Turbo image generation pipeline wrapper."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from diffusers.pipelines.auto_pipeline import ZImageImg2ImgPipeline, ZImagePipeline  # type: ignore[reportUnknownVariableType]
from PIL.Image import Image as PILImage

from services.services_utils import ImagePipelineOutputLike, PILImageType, get_device_type


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
        self._lora_loaded: str | None = None
        self.pipeline = ZImagePipeline.from_pretrained(  # type: ignore[reportUnknownMemberType]
            model_path,
            torch_dtype=torch.bfloat16,
        )
        # Create img2img pipeline sharing the same model components — no extra VRAM.
        pipeline_any = cast(Any, self.pipeline)
        self._img2img_pipeline: Any = ZImageImg2ImgPipeline(  # type: ignore[reportUnknownMemberType]
            scheduler=pipeline_any.scheduler,
            vae=pipeline_any.vae,
            text_encoder=pipeline_any.text_encoder,
            tokenizer=pipeline_any.tokenizer,
            transformer=pipeline_any.transformer,
        )
        if device is not None:
            self.to(device)

    def _resolve_generator_device(self) -> str:
        if self._cpu_offload_active:
            return "cuda"
        if self._device is not None:
            return self._device

        execution_device = getattr(self.pipeline, "_execution_device", None)
        return get_device_type(execution_device)

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
        )
        return self._normalize_output(output)

    @torch.inference_mode()
    def img2img(
        self,
        prompt: str,
        image: PILImageType,
        strength: float,
        height: int,
        width: int,
        guidance_scale: float,
        num_inference_steps: int,
        seed: int,
    ) -> ImagePipelineOutputLike:
        _ = guidance_scale
        generator = torch.Generator(device=self._resolve_generator_device()).manual_seed(seed)
        output = self._img2img_pipeline(
            prompt=prompt,
            image=image,
            strength=strength,
            height=height,
            width=width,
            guidance_scale=0.0,
            num_inference_steps=num_inference_steps,
            generator=generator,
            output_type="pil",
            return_dict=True,
        )
        return self._normalize_output(output)

    def to(self, device: str) -> None:
        runtime_device = get_device_type(device)
        if runtime_device in ("cuda", "mps"):
            self.pipeline.enable_model_cpu_offload()  # type: ignore[reportUnknownMemberType]
            self._img2img_pipeline.enable_model_cpu_offload()  # type: ignore[reportUnknownMemberType]
            self._cpu_offload_active = True
        else:
            self._cpu_offload_active = False
            self.pipeline.to(runtime_device)  # type: ignore[reportUnknownMemberType]
            self._img2img_pipeline.to(runtime_device)  # type: ignore[reportUnknownMemberType]
        self._device = runtime_device

    def load_lora(self, lora_path: str, weight: float = 1.0) -> None:
        if self._lora_loaded == lora_path:
            return
        if self._lora_loaded is not None:
            self.unload_lora()
        pipeline = cast(Any, self.pipeline)
        pipeline.load_lora_weights(lora_path, adapter_name="user_lora")
        pipeline.set_adapters(["user_lora"], adapter_weights=[weight])
        self._lora_loaded = lora_path

    def unload_lora(self) -> None:
        if self._lora_loaded is None:
            return
        pipeline = cast(Any, self.pipeline)
        pipeline.unload_lora_weights()
        self._lora_loaded = None
