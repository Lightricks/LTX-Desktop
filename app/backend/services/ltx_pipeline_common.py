"""Shared helpers and primitives for LTX video pipeline wrappers."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import logging
from typing import TYPE_CHECKING

import torch

from runtime_config.runtime_policy import CUDA_CPU_OFFLOAD_RAM_FLOOR_GB, should_disk_stream_cuda_weights
from services.services_utils import PipelineTilingType, TilingConfigType
from services.storage_info.storage_info import DriveType

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from api_types import ImageConditioningInput
    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_core.model.video_vae.transformer.config import DiffVAEMode
    from ltx_pipelines.utils.media_io import PlaybackSegment
    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_pipelines.utils.types import ImageConditioningInput as LtxImageInput
    from ltx_pipelines.utils.types import OffloadMode


def to_ltx_images(images: Sequence[ImageConditioningInput], *, guide_all: bool = False) -> list[LtxImageInput]:
    """Desktop stills as ltx-pipelines inputs, with the CRF left to the checkpoint default.

    A still at frame 0 replaces that frame and a later one guides. A negative index (the
    IC-LoRA reference token) is a guide. ``guide_all`` also guides frame 0, which multi-keyframe
    interpolation needs so that the first still stays a keyframe instead of a replacement.
    """
    from ltx_pipelines.utils.types import ImageConditioningInput as LtxImageInput

    if guide_all:
        return [LtxImageInput.guide(img.path, img.frame_idx, img.strength) for img in images]
    return [LtxImageInput.at(img.path, img.frame_idx, img.strength) for img in images]


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
    diffvae_optimization: DiffVAEMode | None = None,
) -> TilingConfigType:
    """Same recommendation ``AUTO_TILING`` resolves to, for pipelines that decode themselves.

    ``height`` and ``width`` are the size the decoder sees, not the pipeline canvas.
    """
    from ltx_core.model.video_vae.transformer.config import DiffVAEMode
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
        # CHUNKED_EAGER is upstream's own default for ``tiling_config_for_vae``. It is spelled
        # out here because the parameter is optional on this wrapper, not to change the mode.
        diffvae_optimization=DiffVAEMode.CHUNKED_EAGER if diffvae_optimization is None else diffvae_optimization,
    )


def is_decode_tile_shortage(exc: BaseException) -> bool:
    """True when the failure is the DiffVAE resolver fitting no tile in budget.

    Matches the upstream ``recommended_decode_tiling_config`` refusal by its
    message prefix — the only signal it offers (a bare ValueError). Any other
    ValueError (bad geometry, unreadable checkpoint) must propagate untouched:
    those do not get better with a bigger budget.
    """
    return isinstance(exc, ValueError) and str(exc).startswith("Cannot fit a DiffVAE decode tile")


def stable_cuda_decode_budget_bytes(
    *, fraction: float, device_total_bytes: int, raw_free_bytes: int, reserved_bytes: int
) -> int:
    """Decode-phase CUDA budget: the cap, minus what other processes hold.

    The resolver asks too early (right after text encoding, transformer still
    resident) and reads a transient low-water mark. Post-evict decode enjoys
    everything no other process holds — raw device free plus our own reserved
    pool, which the evict frees — capped by our per-process fraction. Pure
    arithmetic over torch reads the caller supplies, so it unit-tests on CPU.
    """
    return max(0, min(int(fraction * device_total_bytes), raw_free_bytes + reserved_bytes))


def resolve_cuda_decode_tiling_retry(
    vae_checkpoint_path: str,
    *,
    height: int,
    width: int,
    num_frames: int,
    chunk_pixel_frames: int | None,
    device: torch.device,
    keyframes: bool,
    diffvae_optimization: DiffVAEMode,
) -> TilingConfigType:
    """Decode tiling resolved against the stable budget — the R8 retry.

    Call only after the momentary-budget resolution refused with a decode-tile
    shortage (see ``is_decode_tile_shortage``): the first attempt already proved
    the transient number too small, so this one prices against what decode will
    actually enjoy post-evict. A refusal here is a genuine floor (the R5 family)
    and propagates to the caller as today's 400.

    Resolves per-window frames like upstream (frame-grid snap plus the chunk
    cap); decode re-resolves post-evict anyway (``diffvae_decode_vram`` patch),
    so this config only survives the early gate and tiles the reference encode.
    CUDA-only by contract — the caller keeps every other device on AUTO_TILING.
    """
    from ltx_pipelines.utils.helpers import snap_frames_to_grid, tiling_config_for_vae

    resolve_frames = snap_frames_to_grid(num_frames)
    if chunk_pixel_frames is not None:
        resolve_frames = min(resolve_frames, chunk_pixel_frames)
    raw_free, device_total = torch.cuda.mem_get_info(device)
    stable = stable_cuda_decode_budget_bytes(
        fraction=torch.cuda.get_per_process_memory_fraction(device),
        device_total_bytes=device_total,
        raw_free_bytes=raw_free,
        reserved_bytes=torch.cuda.memory_reserved(device),
    )
    tiling = tiling_config_for_vae(
        vae_checkpoint_path,
        height=height,
        width=width,
        num_frames=resolve_frames,
        diffvae_optimization=diffvae_optimization,
        device=device,
        free_bytes=stable,
        keyframes=keyframes,
    )
    logger.info(
        "[ic-lora] decode-tiling retry against a %.2f GiB stable budget: %s",
        stable / 1024**3,
        tiling,
    )
    return tiling


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


def _disk_stream_reason(
    system: str, ram_gb: int | None, cpu_ram_floor_gb: int | None
) -> str:
    if system == "Windows":
        if cpu_ram_floor_gb is None:
            return "checkpoint volume is NVMe on Windows CUDA (edit pipeline)"
        if ram_gb is None:
            return (
                "host RAM unknown on Windows CUDA NVMe; failing closed below the "
                f"{cpu_ram_floor_gb} GiB CPU-streaming floor"
            )
        return (
            f"host RAM {ram_gb} GiB on Windows CUDA NVMe is below the "
            f"{cpu_ram_floor_gb} GiB CPU-streaming floor"
        )
    floor_gb = (
        CUDA_CPU_OFFLOAD_RAM_FLOOR_GB
        if cpu_ram_floor_gb is None
        else cpu_ram_floor_gb
    )
    if ram_gb is None:
        return (
            "host RAM unknown on Linux CUDA; failing closed below the "
            f"{floor_gb} GiB pinned-CPU floor (LTX-Desktop#163)"
        )
    return (
        f"host RAM {ram_gb} GiB on Linux CUDA is below the "
        f"{floor_gb} GiB floor for pinned CPU streaming (LTX-Desktop#163)"
    )


def offload_mode_for_prefetch_count(
    streaming_prefetch_count: int | None,
    device: torch.device,
    *,
    drive_type: DriveType | None,
    system: str,
    ram_bytes: int | None,
    cpu_ram_floor_gb: int | None = CUDA_CPU_OFFLOAD_RAM_FLOOR_GB,
) -> OffloadMode:
    """Translate the desktop's streaming_prefetch_count knob to ltx_pipelines' OffloadMode.

    Pure over explicit inputs: prefetch distinguishes fully resident (None) vs
    streaming (an int); device picks the memory model; system/RAM/drive pick the
    CUDA streaming kind. Callers pass frozen startup facts — no platform or
    psutil reads inside, so the verdict is stable and trivially testable.

    - CUDA: system RAM is separate from VRAM, so OffloadMode.CPU holds the blocks in host
      RAM and streams them to the smaller VRAM — the fast streaming path when RAM fits. On
      Linux, that buffer is page-locked (unswappable) and needs ~36 GB host RAM; 32 GB
      machines (the advertised local-gen floor) must use OffloadMode.DISK instead
      (LTX-Desktop#163). On Windows, non-NVMe and unknown drives stay on OffloadMode.CPU
      (SATA takes ~130 s for ~70 GB of reads per generation). On Windows NVMe, ltx-core
      1.6.0 has no os.preadv and opens an unbuffered handle per 8 MiB chunk, making disk
      streaming slower than CPU offload on Fast and IC-LoRA once RAM clears
      ``cpu_ram_floor_gb`` (40 GB for Fast, 64 GB for IC-LoRA); below that floor (or when
      ``cpu_ram_floor_gb`` is None, as on Edit where full-resolution stage-1 denoise hides
      the reads and disk stays faster), NVMe streams from DISK.
    - MPS (Apple Silicon): CPU-pinned weights live in the *same* unified RAM as the GPU,
      so OffloadMode.CPU (which pins every block, ~46 GB for the bf16 transformer) OOMs.
      OffloadMode.DISK reads blocks from the checkpoint with positional reads into block-sized
      host buffers (2 plus the prefetch depth) and two 64 MiB staging buckets, all unpinned
      on MPS, the only memory-safe streaming path on unified memory. Measurements taken
      before ltx-core 1.6.0 describe the older mmap path.
    """
    from ltx_pipelines.utils.types import OffloadMode

    if streaming_prefetch_count is None:
        return OffloadMode.NONE
    if device.type == "mps":
        return OffloadMode.DISK
    if device.type == "cuda":
        ram_gb = (
            (ram_bytes + 1024**3 - 1) // (1024**3)
            if ram_bytes is not None
            else None
        )
        if should_disk_stream_cuda_weights(
            system,
            ram_gb,
            drive_type=drive_type,
            cpu_ram_floor_gb=cpu_ram_floor_gb,
        ):
            logger.info(
                "Using OffloadMode.DISK: %s",
                _disk_stream_reason(system, ram_gb, cpu_ram_floor_gb),
            )
            return OffloadMode.DISK
    return OffloadMode.CPU


def encode_segments_output(
    segments: Iterator[PlaybackSegment],
    fps: int,
    output_path: str,
    video_chunks_number_value: int | None,
) -> None:
    from ltx_pipelines.utils.media_io import encode_media_segments

    encode_media_segments(
        segments,
        fps=fps,
        output_path=output_path,
        video_chunks_number=video_chunks_number_value,
    )
