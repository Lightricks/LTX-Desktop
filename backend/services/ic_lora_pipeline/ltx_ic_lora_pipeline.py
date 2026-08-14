"""LTX IC-LoRA pipeline wrapper."""

from __future__ import annotations

import json
import logging
import re
import struct
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from api_types import ImageConditioningInput
from services.ltx_pipeline_common import (
    auto_tiling_config,
    build_model_paths,
    encode_video_output,
    offload_mode_for_prefetch_count,
    video_chunks_number,
)
from services.services_utils import AudioOrNone, PipelineTilingType, TilingConfigType, device_supports_fp8

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
    ) -> None:
        from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
        from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
        from ltx_core.quantization.fp8_cast import build_policy as build_fp8_cast_policy
        from ltx_pipelines.ic_lora import ICLoraPipeline
        from ltx_pipelines.utils.types import OffloadMode

        self._device = device
        self._validate_lora_keys(lora_path)
        lora_entry = LoraPathStrengthAndSDOps(path=lora_path, strength=lora_strength, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)
        self._checkpoint_path = checkpoint_path
        self._quantization = build_fp8_cast_policy(checkpoint_path) if device_supports_fp8(device) else None
        offload_mode = offload_mode_for_prefetch_count(streaming_prefetch_count, device)
        self.pipeline = ICLoraPipeline(
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
        )
        # use_lora_in_stage_2 conditions stage 2 on the full-res reference video (~doubles
        # the attention sequence). A fully-resident stage 2 (offload_mode NONE, i.e. no
        # streaming_prefetch_count was set) can overflow VRAM even on high-VRAM cards.
        # Tracked here so _ensure_stage_2_streams_for_lora can force streaming, once, the
        # first time use_lora_in_stage_2 is actually requested -- construction-time, not
        # per-call, per services.patches.ic_lora_stage2_lora's note on this. See there.
        self._stage_2_streams = offload_mode != OffloadMode.NONE

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

    def _ensure_stage_2_streams_for_lora(self) -> None:
        """Force stage 2 onto the streaming path the first time use_lora_in_stage_2 is
        requested, and keep it that way for this pipeline's lifetime (construction-time
        override, not a per-call one -- see the note in services.patches.ic_lora_stage2_lora).
        No-op if stage 2 already streams (streaming_prefetch_count was already set)."""
        if self._stage_2_streams:
            return
        from ltx_pipelines.utils.blocks import DiffusionStage

        self.pipeline.stage_2 = DiffusionStage.from_checkpoint(  # type: ignore[reportUnknownMemberType]
            self._checkpoint_path,
            self.pipeline.dtype,
            self._device,
            quantization=self._quantization,
            offload_mode=offload_mode_for_prefetch_count(1, self._device),
        )
        self._stage_2_streams = True

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
    ) -> tuple[torch.Tensor | Iterator[torch.Tensor], AudioOrNone, TilingConfigType | None]:
        from ltx_pipelines.utils.args import ImageConditioningInput as _LtxImageInput

        video, audio, resolved_tiling = self.pipeline(
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
        return video, audio, resolved_tiling

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

    def _load_source_audio(self, path: str) -> AudioOrNone:
        from dataclasses import replace

        from ltx_pipelines.utils.media_io import decode_audio_from_file

        src = decode_audio_from_file(path, self._device)
        if src is None:
            return None
        # decode_audio_from_file returns (1, channels, samples), but encode_video's writer
        # expects a 2D (channels, samples) tensor with exactly 2 channels. Feeding the 3D
        # tensor flattens it in planar order (LL..RR) and it gets misread as interleaved
        # stereo (LRLR..) -> the audio plays sped up and loops. Drop the batch dim and
        # normalize to stereo.
        waveform = src.waveform
        if waveform.ndim == 3:
            waveform = waveform.squeeze(0)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]
        return replace(src, waveform=waveform)

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
    ) -> None:
        # When Stage 2 runs, optionally keep the IC-LoRA active in it (so the
        # transformation survives refinement). Honoured by
        # services.patches.ic_lora_stage2_lora; a no-op attribute otherwise.
        if use_lora_in_stage_2:
            self._ensure_stage_2_streams_for_lora()
        self.pipeline.use_lora_in_stage_2 = use_lora_in_stage_2  # type: ignore[attr-defined]
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
        # Stage-1 (the heavy diffusion) runs at half the passed canvas — this is the main
        # cost driver, so log it to make slow runs explicable.
        logger.info(
            "[ic-lora] stage-1 %dx%d, %d frames (skip_stage_2=%s, resolution_factor=%.2f)",
            width // 2, height // 2, num_frames, skip_stage_2, resolution_factor,
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
        try:
            video, audio, resolved_tiling = self._run_inference(
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
            )
            # Audio: "off" drops it entirely; "source" muxes the input clip's soundtrack and
            # discards the prompt-synthesized model audio; otherwise keep the generated audio.
            if mute_audio:
                audio = None
            elif source_audio_path is not None:
                audio = self._load_source_audio(source_audio_path)
                if audio is not None:
                    from dataclasses import replace

                    v_dur = num_frames / frame_rate
                    # Trim the source soundtrack to the generated video's length. A longer
                    # source (e.g. an 8s clip rendered to 5s) would otherwise keep playing
                    # over a frozen last frame.
                    max_samples = int(round(v_dur * audio.sampling_rate))
                    if audio.waveform.shape[-1] > max_samples:
                        audio = replace(audio, waveform=audio.waveform[..., :max_samples])
                    a_dur = audio.waveform.shape[-1] / audio.sampling_rate
                    logger.info(
                        "[ic-lora] source audio: %dHz %dch %.2fs (video %.2fs)%s",
                        audio.sampling_rate, audio.waveform.shape[0], a_dur, v_dur,
                        "" if abs(a_dur - v_dur) < 0.1 else "  MISMATCH",
                    )
            chunks = video_chunks_number(num_frames, resolved_tiling)
            # round(), not int(): avoids truncating e.g. 23.976 -> 23 (encode_video int()s again).
            encode_video_output(video=video, audio=audio, fps=round(frame_rate), output_path=output_path, video_chunks_number_value=chunks)
        except torch.cuda.OutOfMemoryError:
            logger.error(
                "[ic-lora] CUDA OOM at stage-1 %dx%d x %d frames — lower RES FACTOR or trim the clip.",
                width // 2, height // 2, num_frames,
            )
            raise
        if cuda:
            peak_alloc = torch.cuda.max_memory_allocated(self._device) / 1e9
            peak_resv = torch.cuda.max_memory_reserved(self._device) / 1e9
            # base = weights resident at start; (peak_alloc - base) = activation demand;
            # reserved = what nvidia-smi shows (allocator pool, usually higher than alloc).
            logger.info(
                "[ic-lora] VRAM: base %.1f / peak alloc %.1f / peak reserved %.1f GB",
                base_gb, peak_alloc, peak_resv,
            )
