"""LTX fast video pipeline wrapper."""

from __future__ import annotations

from collections.abc import Iterator
import os
from typing import Final

import torch

from api_types import ImageConditioningInput
from frame_math import AutoDurationSpec
from services.ltx_pipeline_common import (
    auto_tiling_config,
    build_model_paths,
    encode_video_output,
    offload_mode_for_prefetch_count,
    video_chunks_number,
)
from services.services_utils import AudioOrNone, PipelineTilingType, TilingConfigType, device_supports_fp8


class LTXFastVideoPipeline:
    pipeline_kind: Final = "fast"

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        streaming_prefetch_count: int | None,
        loras: list[tuple[str, float]] | None = None,
        *,
        video_vae_path: str | None = None,
        audio_vae_path: str | None = None,
        duration_head_path: str | None = None,
    ) -> "LTXFastVideoPipeline":
        return LTXFastVideoPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            device=device,
            streaming_prefetch_count=streaming_prefetch_count,
            loras=loras or [],
            video_vae_path=video_vae_path,
            audio_vae_path=audio_vae_path,
            duration_head_path=duration_head_path,
        )

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        streaming_prefetch_count: int | None,
        loras: list[tuple[str, float]] | None = None,
        *,
        video_vae_path: str | None = None,
        audio_vae_path: str | None = None,
        duration_head_path: str | None = None,
    ) -> None:
        from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
        from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
        from ltx_core.quantization.fp8_cast import build_policy as build_fp8_cast_policy
        from ltx_pipelines.distilled import DistilledPipeline

        self._checkpoint_path = checkpoint_path
        self._gemma_root = gemma_root
        self._upsampler_path = upsampler_path
        self._video_vae_path = video_vae_path
        self._audio_vae_path = audio_vae_path
        self._duration_head_path = duration_head_path
        self._device = device
        self._offload_mode = offload_mode_for_prefetch_count(streaming_prefetch_count, device)
        self._quantization = build_fp8_cast_policy(checkpoint_path) if device_supports_fp8(device) else None
        self._loras = loras or []

        lora_entries = [
            LoraPathStrengthAndSDOps(path=path, strength=scale, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)
            for path, scale in self._loras
        ]

        self.pipeline = DistilledPipeline(
            model_paths=build_model_paths(
                checkpoint_path,
                gemma_root,
                video_vae_path=video_vae_path,
                audio_vae_path=audio_vae_path,
                duration_head_path=duration_head_path,
            ),
            spatial_upsampler_path=upsampler_path,
            loras=lora_entries,
            device=device,
            quantization=self._quantization,
            offload_mode=self._offload_mode,
        )

    def _run_inference(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int | AutoDurationSpec,
        frame_rate: float,
        images: list[ImageConditioningInput],
        tiling_config: PipelineTilingType,
    ) -> tuple[torch.Tensor | Iterator[torch.Tensor], AudioOrNone, int, TilingConfigType | None]:
        from ltx_pipelines.utils.args import ImageConditioningInput as _LtxImageInput
        from ltx_pipelines.utils.types import AutoDuration

        pipeline_num_frames: int | AutoDuration = (
            AutoDuration(min_seconds=num_frames.min_seconds, max_seconds=num_frames.max_seconds)
            if isinstance(num_frames, AutoDurationSpec)
            else num_frames
        )

        video, audio, resolved_frames, resolved_tiling = self.pipeline(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=pipeline_num_frames,
            frame_rate=frame_rate,
            images=[_LtxImageInput(img.path, img.frame_idx, img.strength) for img in images],
            tiling_config=tiling_config,
        )
        return video, audio, resolved_frames, resolved_tiling

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int | AutoDurationSpec,
        frame_rate: float,
        images: list[ImageConditioningInput],
        output_path: str,
    ) -> None:
        video, audio, resolved_frames, resolved_tiling = self._run_inference(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            tiling_config=auto_tiling_config(),
        )
        chunks = video_chunks_number(resolved_frames, resolved_tiling)
        encode_video_output(video=video, audio=audio, fps=int(frame_rate), output_path=output_path, video_chunks_number_value=chunks)

    @torch.inference_mode()
    def warmup(self, output_path: str) -> None:
        warmup_frames = 9

        try:
            video, audio, resolved_frames, resolved_tiling = self._run_inference(
                prompt="test warmup",
                seed=42,
                height=256,
                width=384,
                num_frames=warmup_frames,
                frame_rate=8,
                images=[],
                tiling_config=auto_tiling_config(),
            )
            chunks = video_chunks_number(resolved_frames, resolved_tiling)
            encode_video_output(video=video, audio=audio, fps=8, output_path=output_path, video_chunks_number_value=chunks)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def compile_transformer(self) -> None:
        from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
        from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
        from ltx_core.model.transformer.compiling import CompilationConfig
        from ltx_pipelines.distilled import DistilledPipeline

        lora_entries = [
            LoraPathStrengthAndSDOps(path=path, strength=scale, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)
            for path, scale in self._loras
        ]

        self.pipeline = DistilledPipeline(
            model_paths=build_model_paths(
                self._checkpoint_path,
                self._gemma_root,
                video_vae_path=self._video_vae_path,
                audio_vae_path=self._audio_vae_path,
                duration_head_path=self._duration_head_path,
            ),
            spatial_upsampler_path=self._upsampler_path,
            loras=lora_entries,
            device=self._device,
            quantization=self._quantization,
            offload_mode=self._offload_mode,
            compilation_config=CompilationConfig(),
        )
