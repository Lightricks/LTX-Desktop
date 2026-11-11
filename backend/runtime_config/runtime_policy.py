"""Runtime policy decisions for local generation mode."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    # Runtime import would cycle: api_types initializes through this module
    # while services/__init__ needs api_types (see runtime_config.py).
    from services.storage_info.storage_info import DriveType

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

# CUDA discrete VRAM floors, compared against exact GiB (bytes-derived via
# ``MemoryBudget.total_vram_gib`` at the production path; plain integers compare
# identically against integer floors). Below the streaming floor local gen is not
# viable. At/above the full floor, fp8-capable cards can hold the halved transformer
# resident. The streaming floor is 9: an early-out, not a policy — derived, not
# picked. The per-cell model is the real gate (it refuses unfitting cells against
# free VRAM per request), so the floor only skips grid evaluation on hardware that
# cannot run the smallest cell: 8.00 GiB (540p/5s, Phase 2 bracket (7.9, 8.0]) +
# ~1.00 GiB typical desktop occupancy (Phase 0). Cards report below nominal (16 GB
# cards report 15.99 — see ``test_gpu_raw_capacity``), so 10 GB-class cards at
# ~9.99 stream and the model offers them ~8 cells, while 8 GB-class cards at ~7.99
# stay excluded. Per-cell budgets (not this floor) decide what each card offers.
# The 1.00 GiB desktop figure is an estimate (no Phase 0 survey was captured) —
# do not tighten this floor toward 8.00 on its strength. fp8 cards only: without
# fp8 the bf16 path is unmeasured, so non-fp8 CUDA keeps the 15 floor below.
CUDA_VRAM_FLOOR_GB = 9
# CUDA_VRAM_FLOOR_GB is Windows-fp8-measured only (5090 laws; the bf16
# path is unmeasured and the legacy curve under-asks 9-14 GB cards). Linux
# and non-fp8 Windows ride the legacy curve below 15 and keep the old floor.
LINUX_VRAM_FLOOR_GB = 15
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
    resident. Pass ``fp8_capable=False`` (ROCm today — see
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
        if not cuda_available:
            return "unsupported"
        if vram_gb is None:
            return "unsupported"
        floor_gb = (
            CUDA_VRAM_FLOOR_GB
            if system == "Windows" and fp8_capable
            else LINUX_VRAM_FLOOR_GB
        )
        if vram_gb < floor_gb:
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


def windows_checkpoint_streams_from_disk(
    system: str, drive_type: DriveType | None
) -> bool:
    """Whether Windows can stream checkpoint weights from disk (NVMe only).

    Single source for the Windows NVMe disk-streaming gate:
    ``should_disk_stream_cuda_weights`` and ``HardwareProfile.streams_from_disk``
    both read this rule, so the lower NVMe RAM floors only open when the drive
    can actually take the disk path.
    """
    return system == "Windows" and drive_type == "nvme"


def should_disk_stream_cuda_weights(
    system: str,
    ram_gb: int | None,
    *,
    drive_type: DriveType | None,
    cpu_ram_floor_gb: int | None = CUDA_CPU_OFFLOAD_RAM_FLOOR_GB,
) -> bool:
    """Whether CUDA streaming must read weights from disk instead of pinning them in RAM.

    Linux actually succeeds at cudaHostRegister, so below the CPU-offload RAM
    floor (or when RAM is unknown) the pinned transformer cannot fit and weights
    stream from disk instead (LTX-Desktop#163). Drive class does not route Linux.
    Windows needs an NVMe checkpoint drive to stream from disk at all: SATA
    (~130 s for the ~70 GB per-generation reads) does not earn the switch, so
    non-NVMe and unknown drives stay on OffloadMode.CPU with its pageable host
    buffers (pinned_pool_fix / LTX-Desktop#141). On Windows NVMe, ltx-core 1.6.0
    has no os.preadv and opens a fresh unbuffered handle per 8 MiB chunk, making
    disk streaming slower than CPU offload on Fast and IC-LoRA once host RAM
    clears ``cpu_ram_floor_gb`` (40 GB for Fast, 64 GB for IC-LoRA). Below that
    floor — or when ``cpu_ram_floor_gb`` is None (Edit, where full-resolution
    stage-1 denoise hides the reads and disk stays faster) — NVMe streams from
    disk.
    """
    if system == "Windows":
        if not windows_checkpoint_streams_from_disk(system, drive_type):
            return False
        if cpu_ram_floor_gb is None or ram_gb is None:
            return True
        return ram_gb < cpu_ram_floor_gb
    if system != "Linux":
        return False
    if ram_gb is None:
        return True
    floor_gb = (
        CUDA_CPU_OFFLOAD_RAM_FLOOR_GB
        if cpu_ram_floor_gb is None
        else cpu_ram_floor_gb
    )
    return ram_gb < floor_gb


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
