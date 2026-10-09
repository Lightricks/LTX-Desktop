"""Runtime policy decisions for local generation mode."""

from __future__ import annotations

from typing import Literal

LocalGenerationMode = Literal[
    "full_models_loading",
    "streaming_models_loading",
    "unsupported",
]

# Documented Darwin minimum (README, compat matrix). Keep product copy here.
DARWIN_ADVERTISED_FLOOR_GB = 32

# Unpublished API-only cutoff (integer GiB total RAM). Do not use in user-facing
# copy — ``DARWIN_ADVERTISED_FLOOR_GB`` is the documented minimum.
DARWIN_STREAMING_FLOOR_GB = 24

# CUDA discrete VRAM floors (integer GiB, ``total_memory // 1024**3``). Below the
# streaming floor local gen is not viable. At/above the full floor, fp8-capable
# cards can hold the halved transformer resident.
CUDA_VRAM_FLOOR_GB = 15
CUDA_FULL_VRAM_GB = 31

# CUDA OffloadMode.CPU pre-pins every transformer block in host RAM. ltx-pipelines
# documents that path as ~36 GB RAM + ~5 GB VRAM. On Linux those buffers are
# page-locked (unswappable), so a 32 GB machine — the advertised local-gen floor —
# dies in the kernel OOM killer with VRAM still empty (LTX-Desktop#163). Below this
# total-RAM floor, stream from disk instead (OffloadMode.DISK, ~5 GB host).
CUDA_CPU_OFFLOAD_RAM_FLOOR_GB = 40


def decide_local_generation_mode(
    system: str,
    cuda_available: bool,
    vram_gb: int | None,
    mps_available: bool = False,
    ram_gb: int | None = None,
    fp8_capable: bool = True,
) -> LocalGenerationMode:
    """Pick the local-generation mode for this runtime.

    - "unsupported": local generation is not viable; caller must route to the API.
    - "streaming_models_loading": enough memory to run, but model weights must be
      streamed from pinned host RAM (15-30 GB range on CUDA).
    - "full_models_loading": enough memory to hold the whole model resident, so
      streaming is skipped to avoid unnecessary host-RAM pressure.

    On CUDA (Windows/Linux) the memory figure is discrete (total) VRAM, and
    "full_models_loading" (>=31 GB) holds the fp8-halved (~23 GB) transformer
    resident. Pass ``fp8_capable=False`` (ROCm today — see
    ``runtime_config.accelerator.accelerator_backend``) to stay on the streaming
    path regardless of VRAM: without fp8 the full ~42-46 GB bf16 transformer
    would try to stay resident at that floor and OOM. On Apple Silicon (Darwin)
    there is no discrete VRAM — the GPU shares system RAM — so ``ram_gb`` is the
    machine's *total* RAM (SKU), integer GiB, matching CUDA's total-VRAM floor.
    Gated on MPS being available (i.e. Apple Silicon, not an Intel Mac, which
    has no MPS backend and stays unsupported). ``fp8_capable`` is ignored on
    Darwin (no fp8 path on MPS).

    Darwin always streams (mmap via OffloadMode.DISK, not CUDA's pinned-host
    OffloadMode.CPU — see ``offload_mode_for_prefetch_count``). The original
    reason was corruption: on an M5 Max 128 GB, full-resident plus a LoRA
    decoded to green/gray frames. That was an unfenced MPS transfer in upstream
    LoRA fusion (a fused H2D+cast issued non-blocking; the matmul read pre-blit
    garbage), now fenced by ``ltx_core.devices.allow_async_transfer`` — full
    resident is correct again. Streaming stays anyway: the interleaved F3 A/B
    measured full resident only ~8% faster (138.6 s vs 149.9 s mean for
    540p/5s) while eating ~3.5x the MPS pool (~42 vs ~10 GiB) that unified
    memory shares with the OS and apps — and full resident cannot run at all
    on smaller Macs, while streaming serves them the same cells. Memory
    universality plus a negligible speed delta keeps streaming as the policy.
    Do not re-add a high-RAM full path without a controlled A/B showing a
    meaningful win.
    """
    if system == "Darwin":
        if not mps_available:
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
        if not cuda_available:
            return "unsupported"
        if vram_gb is None:
            return "unsupported"
        if vram_gb < CUDA_VRAM_FLOOR_GB:
            return "unsupported"
        # full_models_loading's CUDA_FULL_VRAM_GB floor assumes the fp8-halved (~23 GB)
        # transformer (see module docstring). Without fp8 (ROCm today — see
        # runtime_config.accelerator.accelerator_backend), holding the full bf16
        # (~42-46 GB) transformer resident instead would OOM at this floor, so stay on
        # the streaming path regardless of VRAM until a real bf16-full-resident floor is
        # established on non-CUDA hardware.
        # Originally contributed by boxwrench in https://github.com/Lightricks/LTX-Desktop/pull/160
        if not fp8_capable:
            return "streaming_models_loading"
        if vram_gb < CUDA_FULL_VRAM_GB:
            return "streaming_models_loading"
        return "full_models_loading"

    # Fail closed for non-target platforms unless explicitly relaxed.
    return "unsupported"


def should_disk_stream_cuda_weights(system: str, ram_gb: int | None) -> bool:
    """Whether CUDA streaming must mmap weights from disk instead of pinning them in RAM.

    Windows already allocates pageable host buffers (pinned_pool_fix / LTX-Desktop#141)
    and stays on OffloadMode.CPU. Linux actually succeeds at cudaHostRegister, so the
    32 GB product floor cannot hold the pinned transformer.
    """
    if system != "Linux":
        return False
    if ram_gb is None:
        return True
    return ram_gb < CUDA_CPU_OFFLOAD_RAM_FLOOR_GB


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
