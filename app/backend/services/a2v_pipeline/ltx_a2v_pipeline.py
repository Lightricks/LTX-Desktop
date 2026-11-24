"""LTX A2V (Audio-to-Video) pipeline wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from api_types import ImageConditioningInput
from services.fast_video_pipeline.nag_config import NagConfig
from services.fast_video_pipeline.windowed_run import encode_result, measured_windows, plan_windows
from services.ic_lora_pipeline.ic_lora_inference import InferenceResult, inference_result_from_output
from services.ltx_pipeline_common import (
    build_model_paths,
    resolve_tiling_config,
    to_ltx_images,
)
from services.services_utils import TilingConfigType, device_supports_fp8

if TYPE_CHECKING:
    from ltx_pipelines.chunks import ChunkConfig
    from ltx_pipelines.utils.types import OffloadMode


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
        offload_mode: OffloadMode,
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
            offload_mode=offload_mode,
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
        offload_mode: OffloadMode,
        video_vae_path: str | None = None,
        audio_vae_path: str | None = None,
        duration_head_path: str | None = None,
    ) -> None:
        from ltx_core.loader import LoraPathStrengthAndSDOps
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
            offload_mode=offload_mode,
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
        nag: NagConfig | None = None,
    ) -> InferenceResult:
        output = self.pipeline(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=to_ltx_images(images),
            audio_path=audio_path,
            audio_start_time=audio_start_time,
            audio_max_duration=audio_max_duration,
            tiling_config=tiling_config,
            nag=nag,
        )
        return inference_result_from_output(output)

    def _stream_windows(
        self,
        *,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        audio_path: str,
        audio_start_time: float,
        audio_max_duration: float | None,
        tiling_config: TilingConfigType,
        chunk_config: ChunkConfig,
        nag: NagConfig | None = None,
    ) -> InferenceResult:
        chunks = self.pipeline.stream_chunks(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=to_ltx_images(images),
            audio_path=audio_path,
            chunk_config=chunk_config,
            audio_start_time=audio_start_time,
            audio_max_duration=audio_max_duration,
            tiling_config=tiling_config,
            nag=nag,
        )
        return InferenceResult(chunks, num_frames, tiling_config, (chunks,))

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
        chunk_pixel_frames: int | None = None,
        nag: NagConfig | None = None,
    ) -> None:
        from services.denoising_progress import distilled_total_steps, track_denoising

        chunk_config, chunk_count = plan_windows(
            chunk_pixel_frames, width=width, height=height, num_frames=num_frames, frame_rate=frame_rate
        )
        # Decode tiles are sized for one window, which is what the decoder sees at a time.
        tiling_config = resolve_tiling_config(
            self.pipeline.video_decoder.checkpoint_path,
            height=height,
            width=width,
            num_frames=num_frames if chunk_config is None else min(num_frames, chunk_config.chunk_pixel_frames),
            device=self.pipeline.device,
        )
        measured = measured_windows(
            chunk_config, width=width, height=height, window_count=chunk_count,
            num_frames=num_frames, frame_rate=frame_rate, device=self.pipeline.device,
        )
        with track_denoising(distilled_total_steps() * chunk_count), measured:
            if chunk_config is None:
                result = self._run_inference(
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
                    nag=nag,
                )
            else:
                result = self._stream_windows(
                    prompt=prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=frame_rate,
                    images=images,
                    audio_path=audio_path,
                    audio_start_time=audio_start_time,
                    audio_max_duration=audio_max_duration,
                    tiling_config=tiling_config,
                    chunk_config=chunk_config,
                    nag=nag,
                )
            encode_result(result, fps=int(frame_rate), output_path=output_path, streamed=chunk_config is not None)
