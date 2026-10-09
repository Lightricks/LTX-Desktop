"""LTX IC-LoRA pipeline wrapper."""

from __future__ import annotations

import json
import logging
import re
import struct
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from numpy.typing import NDArray

from api_types import ImageConditioningInput
from runtime_config.ic_lora_tiling import IC_LORA_TILE_OVERLAP, IcLoraTiling
from services.ic_lora_pipeline.ic_lora_inference import (
    InferenceResult,
    PipelineKwargs,
    encode_result,
    plan_chunks,
)
from services.ltx_pipeline_common import (
    auto_tiling_config,
    build_model_paths,
    offload_mode_for_prefetch_count,
)
from services.services_utils import PipelineTilingType, device_supports_fp8

if TYPE_CHECKING:
    from ltx_pipelines.chunks import ChunkConfig

logger = logging.getLogger(__name__)


class LTXIcLoraPipeline:
    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        lora_path: str,
        device: torch.device,
        streaming_prefetch_count: int | None,
        lora_strength: float = 1.0,
        *,
        video_vae_path: str | None = None,
        audio_vae_path: str | None = None,
        duration_head_path: str | None = None,
        stage_2_ic_lora: bool = False,
        tiling: IcLoraTiling | None = None,
    ) -> "LTXIcLoraPipeline":
        return LTXIcLoraPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            lora_path=lora_path,
            device=device,
            streaming_prefetch_count=streaming_prefetch_count,
            lora_strength=lora_strength,
            video_vae_path=video_vae_path,
            audio_vae_path=audio_vae_path,
            duration_head_path=duration_head_path,
            stage_2_ic_lora=stage_2_ic_lora,
            tiling=tiling,
        )

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        lora_path: str,
        device: torch.device,
        streaming_prefetch_count: int | None,
        lora_strength: float = 1.0,
        *,
        video_vae_path: str | None = None,
        audio_vae_path: str | None = None,
        duration_head_path: str | None = None,
        stage_2_ic_lora: bool = False,
        tiling: IcLoraTiling | None = None,
    ) -> None:
        from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
        from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
        from ltx_core.quantization.fp8_cast import build_policy as build_fp8_cast_policy
        from ltx_pipelines.ic_lora import DEFAULT_IC_LORA_STAGES
        from services.ic_lora_pipeline.tiled_ic_lora_pipeline import TiledICLoraPipeline

        self._device = device
        self._stage_2_ic_lora = stage_2_ic_lora
        self._tiling = tiling
        self._validate_lora_keys(lora_path)
        lora_entry = LoraPathStrengthAndSDOps(path=lora_path, strength=lora_strength, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)
        self._checkpoint_path = checkpoint_path
        self._quantization = build_fp8_cast_policy(checkpoint_path) if device_supports_fp8(device) else None
        offload_mode = offload_mode_for_prefetch_count(streaming_prefetch_count, device)
        self.pipeline = TiledICLoraPipeline(
            model_paths=build_model_paths(
                checkpoint_path,
                gemma_root,
                video_vae_path=video_vae_path,
                audio_vae_path=audio_vae_path,
                duration_head_path=duration_head_path,
            ),
            spatial_upsampler_path=upsampler_path,
            loras=[lora_entry],
            device=device,
            quantization=self._quantization,
            offload_mode=offload_mode,
            # Stage 2 keeps the IC-LoRA and the reference. The default recipe drops both.
            stages=(
                (DEFAULT_IC_LORA_STAGES[0], replace(DEFAULT_IC_LORA_STAGES[1], apply_ic_lora=True))
                if stage_2_ic_lora
                else None
            ),
        )

    def _tile_stage_1(self, stage_width: int, stage_height: int, tiling: IcLoraTiling) -> None:
        """Tile the transformer on stage 1 at the tile size that fits this canvas.

        The tile follows the canvas orientation, which is known only here, so the stage
        list is set per job. A pipeline serves one job at a time.
        """
        from ltx_pipelines.ic_lora import spatial_tiled_ic_lora_stages

        tile_width, tile_height = tiling.tile_size(stage_width, stage_height)
        logger.info("[ic-lora] tiled: %dx%d tiles on %dx%d", tile_width, tile_height, stage_width, stage_height)
        self.pipeline.stages = spatial_tiled_ic_lora_stages(
            tile_height=tile_height,
            tile_width=tile_width,
            overlap_fraction=IC_LORA_TILE_OVERLAP,
            apply_ic_lora=(True, False),
        )

    @staticmethod
    def _validate_lora_keys(lora_path: str) -> None:
        """Warn at load time if the LoRA won't actually apply.

        A LoRA in an unexpected key format (PEFT / trainer-native instead of the
        published `…transformer_blocks.N.attn*.lora_A/B` shape) maps onto ~0 model
        modules and loads as an identity no-op — producing zero effect with no error.
        This is a header-only read (8-byte length + JSON); it does not load tensor data.
        """
        try:
            file_size = Path(lora_path).stat().st_size
            with open(lora_path, "rb") as f:
                (header_size,) = struct.unpack("<Q", f.read(8))
                # safetensors header length is read from the file's own <Q; guard it against the
                # actual size so a corrupt/non-safetensors file can't ask us to read absurd bytes.
                if header_size <= 0 or header_size > file_size - 8:
                    raise ValueError(f"invalid safetensors header size {header_size} (file {file_size}B)")
                header: dict[str, object] = json.loads(f.read(header_size).decode("utf-8"))
        except Exception as exc:
            logger.warning("[ic-lora] could not read LoRA keys from %s: %s", lora_path, exc)
            return

        name = Path(lora_path).name
        keys = [k for k in header if k != "__metadata__"]
        adapter_keys = [k for k in keys if "lora_A" in k or "lora_B" in k]
        targeted = [k for k in adapter_keys if re.search(r"transformer_blocks\.\d+\.(attn|ff)", k)]
        if not adapter_keys:
            logger.warning(
                "[ic-lora] %s has no lora_A/lora_B tensors — it will load as a no-op (no effect).", name
            )
        elif not targeted:
            logger.warning(
                "[ic-lora] %s: %d adapter tensors but none target transformer attn/ff blocks "
                "(unexpected key format) — effect likely won't apply.",
                name, len(adapter_keys),
            )
        else:
            logger.info(
                "[ic-lora] %s: %d adapter tensors targeting %d transformer modules.",
                name, len(adapter_keys), len(targeted),
            )

    def _run_inference(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        video_conditioning: list[tuple[str, float]],
        tiling_config: PipelineTilingType,
        skip_stage_2: bool,
        conditioning_attention_mask: torch.Tensor | None,
        chunk_config: ChunkConfig | None = None,
    ) -> InferenceResult:
        """Run the pipeline. A chunked job streams: ``__call__`` would collect every
        decoded window on the GPU first."""
        from ltx_pipelines.utils.types import ImageConditioningInput as _LtxImageInput

        kwargs = PipelineKwargs(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=[_LtxImageInput(img.path, img.frame_idx, img.strength) for img in images],
            video_conditioning=video_conditioning,
            tiling_config=tiling_config,
            skip_stage_2=skip_stage_2,
            conditioning_attention_mask=conditioning_attention_mask,
        )
        if chunk_config is None:
            result = self.pipeline(**kwargs)
            return InferenceResult(result.video, result.audio, result.num_frames, result.tiling_config)
        from ltx_pipelines.chunks import split_decoded_chunks

        chunks, resolved_frames, resolved_tiling = self.pipeline.stream_chunks(
            **kwargs, chunk_config=chunk_config
        )
        video, audio, audio_sampling_rate = split_decoded_chunks(chunks)
        streams = (video, chunks) if audio is None else (video, audio, chunks)
        return InferenceResult(video, audio, resolved_frames, resolved_tiling, audio_sampling_rate, streams)

    def _load_mask_tensor(self, path: str, num_frames: int) -> torch.Tensor:
        """Load an outpaint mask video as a (1, 1, F, H, W) float tensor in [0, 1].

        The mask is static across time (the preprocessor writes a single frame), so we read
        one frame and broadcast it to num_frames via expand — a view, not F materialized
        copies — to avoid a VRAM spike on long/high-res clips. F is pinned to num_frames so
        it matches the reference video the upstream pipeline downsamples it against.
        """
        import cv2

        cap = cv2.VideoCapture(path)
        try:
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok:
            raise ValueError(f"could not read outpaint mask video: {path}")
        gray: NDArray[np.float32] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0  # (H, W)
        single = torch.as_tensor(gray, dtype=torch.float32).to(self._device)  # (H, W)
        return single[None, None, None].expand(1, 1, num_frames, -1, -1)  # (1, 1, F, H, W) broadcast view

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        video_conditioning: list[tuple[str, float]],
        output_path: str,
        skip_stage_2: bool = False,
        use_lora_in_stage_2: bool = False,
        resolution_factor: float = 2.0,
        source_audio_path: str | None = None,
        mute_audio: bool = False,
        conditioning_mask_path: str | None = None,
        chunk_pixel_frames: int | None = None,
    ) -> None:
        if use_lora_in_stage_2:
            logger.warning("[ic-lora] use_lora_in_stage_2 is deprecated and ignored (LTXP-514)")
        # A still at a negative frame index is an IC-LoRA reference token. A stage
        # without the IC-LoRA has no place for it, so fail before any GPU work.
        if any(img.frame_idx < 0 for img in images) and not skip_stage_2 and not self._stage_2_ic_lora:
            raise ValueError("A reference still at a negative frame index needs the IC-LoRA on stage 2")
        if self._tiling is not None:
            if not skip_stage_2:
                raise ValueError("A tiled IC-LoRA runs a single stage")
            if any(img.frame_idx < 0 for img in images):
                # Upstream keeps such a token whole in every tile instead of cropping it.
                raise ValueError("A tiled IC-LoRA cannot take a reference still")
        # Stage-1-only outputs height//2 x width//2. Scale the passed canvas by
        # resolution_factor so the result lands at the desired fraction of native
        # (2.0 = native target, 1.0 = half). Round to a multiple of 128: stage-1 runs
        # at canvas//2, and its latent (px//32) must be even for the 2x patchify — so
        # the full canvas must be divisible by 128 (e.g. 1.5x on a 2:1 clip otherwise
        # yields stage-1 height 288 -> odd latent 9 -> patchify crash).
        if skip_stage_2:
            width = max(128, round(width * resolution_factor / 128) * 128)
            height = max(128, round(height * resolution_factor / 128) * 128)
        # Outpainting: a pixel-space keep=1/pad=0 mask tells the model to preserve the
        # reference where it's kept and generate freely in the new canvas region.
        conditioning_attention_mask = (
            self._load_mask_tensor(conditioning_mask_path, num_frames)
            if conditioning_mask_path is not None
            else None
        )
        if self._tiling is not None:
            self._tile_stage_1(width // 2, height // 2, self._tiling)
        # Stage-1 (the heavy diffusion) runs at half the passed canvas — this is the main
        # cost driver, so log it to make slow runs explicable.
        logger.info(
            "[ic-lora] stage-1 %dx%d, %d frames (skip_stage_2=%s, resolution_factor=%.2f)",
            width // 2, height // 2, num_frames, skip_stage_2, resolution_factor,
        )
        chunk_config, chunk_count = plan_chunks(
            chunk_pixel_frames,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=frame_rate,
        )
        if chunk_config is not None:
            logger.info(
                "[ic-lora] chunked: %d windows of %d frames (carry %d)",
                chunk_count,
                chunk_config.chunk_pixel_frames,
                chunk_config.next_video_carry_frames,
            )
        cuda = torch.cuda.is_available() and self._device.type == "cuda"
        base_gb = 0.0
        if cuda:
            torch.cuda.reset_peak_memory_stats(self._device)
            base_gb = torch.cuda.memory_allocated(self._device) / 1e9
            free_gb = torch.cuda.mem_get_info(self._device)[0] / 1e9
            if free_gb < 3.0:
                logger.warning(
                    "[ic-lora] only %.1f GB VRAM free at inference start — likely to crawl or OOM; "
                    "reduce frames (shorter clip / lower FPS) or RES FACTOR.",
                    free_gb,
                )
        started = time.monotonic()
        try:
            from services.denoising_progress import distilled_total_steps, track_denoising

            with track_denoising(distilled_total_steps(stage_2=not skip_stage_2) * chunk_count):
                result = self._run_inference(
                    prompt=prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=frame_rate,
                    images=images,
                    video_conditioning=video_conditioning,
                    tiling_config=auto_tiling_config(),
                    skip_stage_2=skip_stage_2,
                    conditioning_attention_mask=conditioning_attention_mask,
                    chunk_config=chunk_config,
                )
                encode_result(
                    result,
                    output_path=output_path,
                    num_frames=num_frames,
                    frame_rate=frame_rate,
                    mute_audio=mute_audio,
                    source_audio_path=source_audio_path,
                    streamed=chunk_config is not None,
                    device=self._device,
                )
        except torch.cuda.OutOfMemoryError:
            logger.error(
                "[ic-lora] CUDA OOM at stage-1 %dx%d x %d frames — lower RES FACTOR or trim the clip.",
                width // 2, height // 2, num_frames,
            )
            raise
        # The tile and the window count tell a tiled run from a plain one, so run times can be
        # compared. This line has no CUDA condition, so a Mac run is measured too.
        tile = "none"
        if self._tiling is not None:
            tile_width, tile_height = self._tiling.tile_size(width // 2, height // 2)
            tile = f"{tile_width}x{tile_height}"
        logger.info(
            "[ic-lora] run time %.0fs (%d frames, tile %s, %d window(s))",
            time.monotonic() - started, num_frames, tile, chunk_count,
        )
        if cuda:
            peak_alloc = torch.cuda.max_memory_allocated(self._device) / 1e9
            peak_resv = torch.cuda.max_memory_reserved(self._device) / 1e9
            # base = weights resident at start; (peak_alloc - base) = activation demand;
            # reserved = what nvidia-smi shows (allocator pool, usually higher than alloc).
            logger.info(
                "[ic-lora] VRAM: base %.1f / peak alloc %.1f / peak reserved %.1f GB (tile %s, %d window(s))",
                base_gb, peak_alloc, peak_resv, tile, chunk_count,
            )
