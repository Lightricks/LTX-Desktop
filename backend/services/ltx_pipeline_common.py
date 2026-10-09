"""Shared helpers and primitives for LTX video pipeline wrappers."""

from __future__ import annotations

from collections.abc import Iterator
import logging
import platform
from typing import TYPE_CHECKING

import torch

from runtime_config.runtime_policy import CUDA_CPU_OFFLOAD_RAM_FLOOR_GB, should_disk_stream_cuda_weights
from services.services_utils import AudioOrNone, PipelineTilingType, TilingConfigType

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_pipelines.utils.types import OffloadMode


def auto_tiling_config() -> PipelineTilingType:
    """Let the pipeline derive decode tiling from the VAE it will decode with.

    A conv VAE (2.3 monolith) and a diffusion VAE (2.5 split) need different tile
    overlaps, so a fixed layout that one accepts the other rejects.
    """
    from ltx_core.model.video_vae import AUTO_TILING

    return AUTO_TILING


def host_available_bytes() -> int:
    """Currently available system RAM in bytes (unified memory on Apple Silicon)."""
    import psutil

    return int(psutil.virtual_memory().available)


def host_total_gib() -> int | None:
    """Total system RAM in GiB, or None if it cannot be queried."""
    try:
        import psutil

        return int(psutil.virtual_memory().total // (1024**3))
    except Exception:
        logger.warning("Failed to query total system RAM", exc_info=True)
        return None


def diffvae_activation_budget_bytes(device: torch.device | None = None) -> int:
    """Bytes DiffVAE decode tiling may treat as free activation memory.

    ltx-pipelines only queries the CUDA allocator. On MPS/CPU that path yields 0,
    so AUTO_TILING raises ``Cannot fit a DiffVAE decode tile`` before decode.
    CUDA keeps the upstream allocator budget; everywhere else uses available RAM.
    """
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        from ltx_core.devices import cuda_activation_budget_bytes

        return int(cuda_activation_budget_bytes(device))
    return host_available_bytes()


def resolve_diffvae_free_bytes(device: torch.device | None, free_bytes: int | None) -> int | None:
    """Fill a DiffVAE tiling budget when upstream would treat non-CUDA as 0."""
    if free_bytes is not None and free_bytes > 0:
        return free_bytes
    if device is not None and device.type == "cuda":
        return free_bytes
    return host_available_bytes()


def resolve_tiling_config(
    vae_checkpoint_path: str,
    *,
    height: int,
    width: int,
    num_frames: int,
    device: torch.device | None = None,
) -> TilingConfigType:
    """Same recommendation ``AUTO_TILING`` resolves to, for pipelines that decode themselves."""
    from ltx_pipelines.utils.helpers import get_device, tiling_config_for_vae

    if device is None:
        device = get_device()
    return tiling_config_for_vae(
        vae_checkpoint_path,
        height=height,
        width=width,
        num_frames=num_frames,
        device=device,
        free_bytes=diffvae_activation_budget_bytes(device),
    )


def build_model_paths(
    checkpoint_path: str,
    gemma_root: str | None,
    *,
    video_vae_path: str | None = None,
    audio_vae_path: str | None = None,
    duration_head_path: str | None = None,
) -> ModelPaths:
    """Build ``ModelPaths`` for monolith (2.3) or split (2.5) checkpoint layouts.

    When both VAE paths are provided, uses ``from_split`` (LTX 2.5). Otherwise uses
    ``from_monolith`` where the fat checkpoint also supplies the VAEs and DurationHead.
    Split 2.5 DurationHead is a separate safetensors; omit ``duration_head_path`` and
    AutoDuration fails closed in the pipeline.
    """
    from ltx_pipelines.utils.model_paths import ModelPaths

    if video_vae_path is not None and audio_vae_path is not None:
        return ModelPaths.from_split(
            transformer_path=checkpoint_path,
            text_encoder_path=gemma_root,
            video_vae_path=video_vae_path,
            audio_vae_path=audio_vae_path,
            duration_head_path=duration_head_path,
        )
    return ModelPaths.from_monolith(checkpoint_path, gemma_root, video_vae_path=video_vae_path)


def default_guiders() -> tuple[MultiModalGuiderParams, MultiModalGuiderParams]:
    from ltx_core.components.guiders import MultiModalGuiderParams

    return MultiModalGuiderParams(cfg_scale=3.0), MultiModalGuiderParams(cfg_scale=3.0)


def video_chunks_number(num_frames: int, tiling_config: TilingConfigType | None) -> int:
    from ltx_core.model.video_vae import get_video_chunks_number

    return int(get_video_chunks_number(num_frames, tiling_config))


def offload_mode_for_prefetch_count(streaming_prefetch_count: int | None, device: torch.device) -> OffloadMode:
    """Translate the desktop's streaming_prefetch_count knob to ltx_pipelines' OffloadMode.

    ltx_pipelines moved weight streaming from a per-call prefetch-count int to a
    construction-time OffloadMode enum (NONE/CPU/DISK). Desktop's runtime policy
    (runtime_config/runtime_policy.py) distinguishes fully resident (None) vs streaming
    (an int); which *kind* of streaming depends on the device's memory model:

    - CUDA: system RAM is separate from VRAM, so OffloadMode.CPU pins the blocks in host
      RAM and streams them to the smaller VRAM — the fast streaming path. On Linux,
      that pin is unswappable and needs ~36 GB host RAM; 32 GB machines (the advertised
      local-gen floor) must use OffloadMode.DISK instead (LTX-Desktop#163).
    - MPS (Apple Silicon): CPU-pinned weights live in the *same* unified RAM as the GPU,
      so OffloadMode.CPU (which pins every block, ~46 GB for the bf16 transformer) OOMs.
      OffloadMode.DISK mmaps blocks from the checkpoint through a small pinned buffer
      (~5 GB), the only memory-safe streaming path on unified memory. This is the "mmap
      streaming" the upstream MPS-support work validated on an M4 Pro.
    """
    from ltx_pipelines.utils.types import OffloadMode

    if streaming_prefetch_count is None:
        return OffloadMode.NONE
    if device.type == "mps":
        return OffloadMode.DISK
    if device.type == "cuda":
        system = platform.system()
        ram_gb = host_total_gib() if system == "Linux" else None
        if should_disk_stream_cuda_weights(system, ram_gb):
            if ram_gb is None:
                logger.info(
                    "Using OffloadMode.DISK on Linux CUDA: host RAM unknown; "
                    "failing closed below the %s GiB pinned-CPU floor (LTX-Desktop#163).",
                    CUDA_CPU_OFFLOAD_RAM_FLOOR_GB,
                )
            else:
                logger.info(
                    "Using OffloadMode.DISK on Linux CUDA: host RAM %s GiB is below the "
                    "%s GiB floor for pinned CPU streaming (LTX-Desktop#163).",
                    ram_gb,
                    CUDA_CPU_OFFLOAD_RAM_FLOOR_GB,
                )
            return OffloadMode.DISK
    return OffloadMode.CPU


def encode_video_output(
    video: torch.Tensor | Iterator[torch.Tensor],
    audio: AudioOrNone,
    fps: int,
    output_path: str,
    video_chunks_number_value: int,
) -> None:
    from ltx_pipelines.utils.media_io import encode_video

    encode_video(
        video=video,
        fps=fps,
        audio=audio,
        output_path=output_path,
        video_chunks_number=video_chunks_number_value,
    )
