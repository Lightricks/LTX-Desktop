"""Runtime policy decisions for local generation mode."""

from __future__ import annotations

from typing import Literal

from inference.device import (
    CUDA_CPU_OFFLOAD_RAM_FLOOR_GB as CUDA_CPU_OFFLOAD_RAM_FLOOR_GB,
    CUDA_FULL_VRAM_GB as CUDA_FULL_VRAM_GB,
    CUDA_VRAM_FLOOR_GB as CUDA_VRAM_FLOOR_GB,
    LINUX_VRAM_FLOOR_GB as LINUX_VRAM_FLOOR_GB,
    decide_cuda_local_generation_mode,
    should_disk_stream_cuda_weights as should_disk_stream_cuda_weights,
    windows_checkpoint_streams_from_disk as windows_checkpoint_streams_from_disk,
)

LocalGenerationMode = Literal[
    "full_models_loading",
    "streaming_models_loading",
    "unsupported",
]

# Documented Darwin minimum (README, compat matrix). Keep product copy here.
DARWIN_ADVERTISED_FLOOR_GB = 32

# Documented macOS minimum for local generation (README). The bundled prebuilt
# mpsgraph_zc extension is built for macOS 26 and cannot load on older systems.
# On macOS 15, mps-sdpa falls back to a backend that leaks Metal memory and
# runs out of memory on longer clips (docs/mps-attention-memory-leak.md).
# LTX-Desktop#171 reports a failure at the first attention call on macOS 14.8.2.
# The cause is NOT confirmed. One hypothesis is that the MPSGraph attention API
# needs macOS 15. Do not treat it as verified evidence for this floor.
DARWIN_MIN_MACOS_MAJOR = 26

# Unpublished API-only cutoff (integer GiB total RAM). Do not use in user-facing
# copy — ``DARWIN_ADVERTISED_FLOOR_GB`` is the documented minimum.
DARWIN_STREAMING_FLOOR_GB = 24


def decide_local_generation_mode(
    system: str,
    cuda_available: bool,
    vram_gb: float | None,
    mps_available: bool = False,
    ram_gb: int | None = None,
    fp8_capable: bool = True,
    macos_major: int | None = None,
) -> LocalGenerationMode:
    """Pick the local-generation mode for this runtime.

    - "unsupported": local generation is not viable; caller must route to the API.
    - "streaming_models_loading": enough memory to run, but model weights must be
      streamed from pinned host RAM (Windows fp8 9-30 GB, other CUDA 15-30 GB).
    - "full_models_loading": enough memory to hold the whole model resident, so
      streaming is skipped to avoid unnecessary host-RAM pressure.

    On CUDA (Windows/Linux) the memory figure is discrete (total) VRAM in GiB —
    exact bytes-derived where available, integer-floored otherwise (both compare
    identically against these integer floors) — and
    "full_models_loading" (>=31 GB) holds the fp8-halved (~23 GB) transformer
    resident. That branch is ``inference.device.decide_cuda_local_generation_mode``,
    so a host that copies only ``inference/`` can call it without this module.
    Pass ``fp8_capable=False`` (ROCm today — see
    ``runtime_config.accelerator.accelerator_backend``) to stay on the streaming
    path regardless of VRAM: without fp8 the full ~42-46 GB bf16 transformer
    would try to stay resident at that floor and OOM. On Apple Silicon (Darwin)
    there is no discrete VRAM — the GPU shares system RAM — so ``ram_gb`` is the
    machine's *total* RAM (SKU), integer GiB, matching CUDA's total-VRAM floor.
    Gated on MPS being available (i.e. Apple Silicon, not an Intel Mac, which
    has no MPS backend and stays unsupported). ``fp8_capable`` is ignored on
    Darwin (no fp8 path on MPS). ``macos_major`` below ``DARWIN_MIN_MACOS_MAJOR``
    is unsupported. ``None`` means the version is unknown, and it does not block.

    Darwin always streams (from disk via OffloadMode.DISK, not CUDA's pinned-host
    OffloadMode.CPU — see ``offload_mode_for_prefetch_count``). The original
    reason was corruption: on an M5 Max 128 GB, full-resident plus a LoRA
    decoded to green/gray frames. That was an unfenced MPS transfer in upstream
    LoRA fusion (a fused H2D+cast issued non-blocking; the matmul read pre-blit
    garbage), now fenced by ``ltx_core.devices.allow_async_transfer``
    (ltx-2-internal#843 + desktop #304) — full resident is correct again.
    Streaming stays anyway: the interleaved F3 A/B measured full resident only
    ~8% faster (138.6 s vs 149.9 s mean for 540p/5s) while eating ~3.5x the MPS
    pool (~42 vs ~10 GiB) that unified memory shares with the OS and apps — and
    full resident cannot run at all on smaller Macs, while streaming serves them
    the same cells. Memory universality plus a negligible speed delta keeps
    streaming as the policy. Do not re-add a high-RAM full path without a
    controlled A/B showing a meaningful win.
    """
    if system == "Darwin":
        if not mps_available:
            return "unsupported"
        if macos_major is not None and macos_major < DARWIN_MIN_MACOS_MAJOR:
            return "unsupported"
        if ram_gb is None:
            return "unsupported"
        if ram_gb < DARWIN_STREAMING_FLOOR_GB:
            return "unsupported"
        # Always stream on MPS — see docstring. High RAM used to pick
        # full_models_loading (>=85 GB); the corruption that ruled that out is
        # fixed, but streaming still dominates it on speed and memory.
        return "streaming_models_loading"

    if system in ("Windows", "Linux"):
        return decide_cuda_local_generation_mode(system, cuda_available, vram_gb, fp8_capable)

    # Fail closed for non-target platforms unless explicitly relaxed.
    return "unsupported"


def streaming_prefetch_count_for_mode(mode: LocalGenerationMode) -> int | None:
    """Return the streaming_prefetch_count to pass to a local pipeline.

    Must not be called when local generation is unsupported — callers should
    route through the API instead.
    """
    if mode == "unsupported":
        raise AssertionError(
            "streaming_prefetch_count_for_mode called with 'unsupported' mode; "
            "callers must route to the API instead of constructing a local pipeline."
        )
    if mode == "full_models_loading":
        return None
    if mode == "streaming_models_loading":
        return 2
    raise AssertionError(f"Unexpected LocalGenerationMode: {mode!r}")
