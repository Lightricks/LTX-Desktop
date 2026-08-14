"""LTX A2V (Audio-to-Video) pipeline wrapper."""

from __future__ import annotations

from collections.abc import Iterator

import torch

from api_types import ImageConditioningInput
from services.ltx_pipeline_common import (
    build_model_paths,
    encode_video_output,
    resolve_tiling_config,
    video_chunks_number,
)
from services.services_utils import AudioOrNone, TilingConfigType, device_supports_fp8


class LTXa2vPipeline:
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
    ) -> "LTXa2vPipeline":
        return LTXa2vPipeline(
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

        from services.a2v_pipeline.distilled_a2v_pipeline import DistilledA2VPipeline

        lora_entries = [
            LoraPathStrengthAndSDOps(path=path, strength=scale, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)
            for path, scale in (loras or [])
        ]

        self.pipeline = DistilledA2VPipeline(
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
            quantization=build_fp8_cast_policy(checkpoint_path) if device_supports_fp8(device) else None,
            streaming_prefetch_count=streaming_prefetch_count,
        )

    def _run_inference(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        images: list[ImageConditioningInput],
        audio_path: str,
        audio_start_time: float,
        audio_max_duration: float | None,
        tiling_config: TilingConfigType,
    ) -> tuple[torch.Tensor | Iterator[torch.Tensor], AudioOrNone]:
        return self.pipeline(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=[(img.path, img.frame_idx, img.strength) for img in images],
            audio_path=audio_path,
            audio_start_time=audio_start_time,
            audio_max_duration=audio_max_duration,
            tiling_config=tiling_config,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        images: list[ImageConditioningInput],
        audio_path: str,
        audio_start_time: float,
        audio_max_duration: float | None,
        output_path: str,
    ) -> None:
        tiling_config = resolve_tiling_config(
            self.pipeline.video_decoder.checkpoint_path,
            height=height,
            width=width,
            num_frames=num_frames,
            device=self.pipeline.device,
        )
        video, audio = self._run_inference(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            images=images,
            audio_path=audio_path,
            audio_start_time=audio_start_time,
            audio_max_duration=audio_max_duration,
            tiling_config=tiling_config,
        )
        chunks = video_chunks_number(num_frames, tiling_config)
        encode_video_output(video=video, audio=audio, fps=int(frame_rate), output_path=output_path, video_chunks_number_value=chunks)
