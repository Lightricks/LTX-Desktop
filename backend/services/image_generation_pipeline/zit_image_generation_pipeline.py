"""Z-Image-Turbo image generation pipeline wrapper."""

# ZImagePipeline / ZImageTransformer2DModel are untyped in diffusers, so values
# derived from them read as "unknown" under pyright's strict mode. Suppress the
# unknown-type family for this file rather than scattering per-line ignores.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from diffusers import BitsAndBytesConfig  # type: ignore[reportPrivateImportUsage]
from diffusers import ZImageTransformer2DModel  # type: ignore[reportPrivateImportUsage]
from diffusers.pipelines.auto_pipeline import ZImagePipeline  # type: ignore[reportUnknownVariableType]
from PIL.Image import Image as PILImage
from PIL.Image import Resampling
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

from services.generation_interrupt import diffusers_step_callback
from services.image_generation_pipeline.variation_boost import (
    boost_step_count,
    clamp_variation,
    log_boost,
    perturb_embeds,
    restoring_callback,
)
from services.services_utils import (
    ImagePipelineOutputLike,
    PILImageType,
    clamp_strength,
    empty_device_cache,
    get_device_type,
    sync_device,
)

logger = logging.getLogger(__name__)

# Cached beside the model files themselves (never touches the originals) so
# subsequent loads read the already-quantized weights instead of re-quantizing
# the full bf16 originals. Cuts cold-load time roughly 8x and makes the load far
# less likely to stall on the OS file cache evicting the large source shards
# under RAM pressure (e.g. from a video model).
_NF4_TRANSFORMER_CACHE_DIRNAME = "_nf4_transformer_cache"
_NF4_TEXT_ENCODER_CACHE_DIRNAME = "_nf4_text_encoder_cache"


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

    @staticmethod
    def _load_quantized_transformer(model_path: str) -> ZImageTransformer2DModel:
        cache_dir = Path(model_path) / _NF4_TRANSFORMER_CACHE_DIRNAME
        if (cache_dir / "config.json").exists():
            try:
                return ZImageTransformer2DModel.from_pretrained(  # type: ignore[reportUnknownMemberType]
                    cache_dir, torch_dtype=torch.bfloat16
                )
            except Exception:
                logger.warning(
                    "Failed to load cached NF4 Z-Image transformer from %s; re-quantizing from source.",
                    cache_dir,
                    exc_info=True,
                )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        transformer = ZImageTransformer2DModel.from_pretrained(  # type: ignore[reportUnknownMemberType]
            model_path,
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        )
        try:
            transformer.save_pretrained(cache_dir)  # type: ignore[reportUnknownMemberType]
        except OSError:
            logger.warning("Failed to cache NF4 Z-Image transformer to %s", cache_dir, exc_info=True)
        return transformer

    @staticmethod
    def _load_quantized_text_encoder(model_path: str) -> Qwen3Model:
        cache_dir = Path(model_path) / _NF4_TEXT_ENCODER_CACHE_DIRNAME
        if (cache_dir / "config.json").exists():
            try:
                return Qwen3Model.from_pretrained(cache_dir, torch_dtype=torch.bfloat16)  # type: ignore[reportUnknownMemberType]
            except Exception:
                logger.warning(
                    "Failed to load cached NF4 Z-Image text encoder from %s; re-quantizing from source.",
                    cache_dir,
                    exc_info=True,
                )

        quantization_config = TransformersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        text_encoder = Qwen3Model.from_pretrained(  # type: ignore[reportUnknownMemberType]
            model_path,
            subfolder="text_encoder",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        )
        try:
            text_encoder.save_pretrained(cache_dir)  # type: ignore[reportUnknownMemberType]
        except OSError:
            logger.warning("Failed to cache NF4 Z-Image text encoder to %s", cache_dir, exc_info=True)
        return text_encoder

    def __init__(self, model_path: str, device: str | None = None) -> None:
        self._device: str | None = None
        self._cpu_offload_active = False
        self._img2img: Any = None

        # NF4 quantization only benefits CUDA: it shrinks the ~24.6 GB bf16
        # transformer to ~7 GB and the ~8 GB text encoder to ~3 GB. Loaded as
        # raw bf16 the transformer alone fills a 24 GB card, so
        # enable_model_cpu_offload()'s whole-module move overflowed into shared
        # system RAM and thrashed forever (uninterruptible — cancel is only
        # polled between denoise steps). Quantized, both components fit as whole
        # modules with room for activations. bitsandbytes has no MPS backend, so
        # other devices keep the plain bf16 components.
        if get_device_type(device) == "cuda":
            transformer = self._load_quantized_transformer(model_path)
            text_encoder = self._load_quantized_text_encoder(model_path)
            self.pipeline = ZImagePipeline.from_pretrained(  # type: ignore[reportUnknownMemberType]
                model_path,
                transformer=transformer,
                text_encoder=text_encoder,
                torch_dtype=torch.bfloat16,
            )
        else:
            self.pipeline = ZImagePipeline.from_pretrained(  # type: ignore[reportUnknownMemberType]
                model_path,
                torch_dtype=torch.bfloat16,
            )

        if device is not None:
            self.to(device)

    def _resolve_generator_device(self) -> str:
        # The configured runtime device is authoritative. With enable_model_cpu_offload()
        # the pipeline's _execution_device can read as "cpu", but the generator must live
        # on the actual compute device. This previously returned "cuda" whenever offload
        # was active — but offload is enabled on MPS too, so on a Mac it built a CUDA
        # generator and failed ("Cannot get CUDA generator without ATen_cuda library").
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

    # Quantizing the transformer shrinks its *weights* (~24.6 GB -> ~7 GB), but
    # activation memory still scales with pixel count regardless of quantization.
    # Cap generation at ~1 MP (matching the Krea 2 path) so a large requested
    # resolution can't push the denoise activations back into shared system RAM
    # and re-introduce the thrash this quantization work fixes; the result is
    # Lanczos-upscaled to the requested size.
    _MAX_GENERATION_PIXELS = 1344 * 768

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        height: int,
        width: int,
        guidance_scale: float,
        num_inference_steps: int,
        seed: int,
        variation: float = 0.0,
    ) -> ImagePipelineOutputLike:
        # ZImagePipeline ignores guidance_scale, so we drop it explicitly.
        _ = guidance_scale
        requested_width, requested_height = width, height
        if width * height > self._MAX_GENERATION_PIXELS:
            scale = (self._MAX_GENERATION_PIXELS / (width * height)) ** 0.5
            width = max(16, int(width * scale) // 16 * 16)
            height = max(16, int(height * scale) // 16 * 16)

        self._log_run("generate", width=width, height=height)
        device = self._resolve_generator_device()
        generator = torch.Generator(device=device).manual_seed(seed)
        pipeline = cast(Any, self.pipeline)
        prompt_kwargs, callback, callback_inputs = self._variation_inputs(
            pipeline, prompt, seed, variation, num_inference_steps
        )
        try:
            output = pipeline(
                **prompt_kwargs,
                height=height,
                width=width,
                guidance_scale=0.0,
                num_inference_steps=num_inference_steps,
                generator=generator,
                output_type="pil",
                return_dict=True,
                callback_on_step_end=callback,
                callback_on_step_end_tensor_inputs=callback_inputs,
            )
        finally:
            # Leftover allocator cache would starve the next run.
            sync_device(device)
            empty_device_cache(device)

        result = self._normalize_output(output)
        if (width, height) != (requested_width, requested_height):
            result = _ZImageOutput(
                images=[
                    img.resize((requested_width, requested_height), Resampling.LANCZOS) for img in result.images
                ]
            )
        return result

    # Variation boost (see variation_boost.py). Tuned 2026-09-27 on the 3090 (4 steps):
    # noising only the FIRST step keeps prompt adherence up to 1.0 with big
    # layout/pose/camera changes; noising 2 of 4 steps broke the prompt from ~0.75 up.
    _VARIATION_MAX_NOISE = 1.0
    _VARIATION_MASK_FRACTION = 0.5
    _VARIATION_BOOST_FRACTION = 0.25

    def _variation_inputs(
        self,
        pipeline: Any,
        prompt: str,
        seed: int,
        variation: float,
        num_inference_steps: int,
    ) -> tuple[dict[str, Any], Any, list[str]]:
        variation = clamp_variation(variation)
        if variation <= 0.0:
            return {"prompt": prompt}, diffusers_step_callback, ["latents"]

        clean, _ = pipeline.encode_prompt(
            prompt=prompt, device=pipeline._execution_device, do_classifier_free_guidance=False
        )
        noisy = perturb_embeds(
            list(clean),
            seed=seed,
            variation=variation,
            max_noise=self._VARIATION_MAX_NOISE,
            mask_fraction=self._VARIATION_MASK_FRACTION,
        )
        boost_steps = boost_step_count(num_inference_steps, self._VARIATION_BOOST_FRACTION)
        log_boost("Z-Image", variation, boost_steps, num_inference_steps)
        return {"prompt_embeds": noisy}, restoring_callback(clean, boost_steps), ["prompt_embeds"]

    def _ensure_img2img_pipeline(self) -> Any:
        if self._img2img is not None:
            return self._img2img
        try:
            from diffusers import ZImageImg2ImgPipeline  # type: ignore[attr-defined]
        except Exception as e:
            raise RuntimeError("DIFFUSERS_IMG2IMG_UNAVAILABLE") from e
        # Reuse the loaded components so no second copy of the weights lands in VRAM.
        # With the CUDA path these are the quantized transformer/text encoder.
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

        device = self._resolve_generator_device()
        generator = torch.Generator(device=device).manual_seed(seed)
        try:
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
        finally:
            sync_device(device)
            empty_device_cache(device)
        return self._normalize_output(output)

    def to(self, device: str) -> None:
        runtime_device = get_device_type(device)
        if runtime_device == "cuda":
            # enable_model_cpu_offload() installs accelerate hooks. Re-invoking it while
            # offload is already active for this device leaves modules holding
            # accelerate's wrapped forward but no _hf_hook attribute, which then blows up
            # at inference: "'Qwen3Model' object has no attribute '_hf_hook'". Install it
            # once; a park-to-CPU cycle takes the else-branch below and clears the flag,
            # so moving back to the accelerator re-installs correctly.
            if not (self._cpu_offload_active and self._device == runtime_device):
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
