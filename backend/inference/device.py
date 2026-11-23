"""Device type, fp8 capability, and CUDA RAM/VRAM floors for local inference."""

from __future__ import annotations

import logging
from typing import Literal

import torch

logger = logging.getLogger(__name__)

DriveType = Literal["nvme", "ssd", "hdd"]

# CUDA discrete VRAM floors (integer GiB, ``total_memory // 1024**3``).
# At/above the full floor, fp8-capable cards can hold the halved transformer
# resident.
#
# The streaming floor is an early exit, not the policy. The per-cell model is
# the real gate: it refuses a cell that does not fit free VRAM. The floor only
# skips that check on hardware that cannot run the smallest cell. 10 GB-class
# cards report just under 10 GiB and stream. 8 GB-class cards report just under
# 8 GiB and stay excluded. Do not tighten this floor toward 8. The gap above
# the smallest cell is an estimate of ordinary desktop occupancy, and shaving
# it would exclude cards the per-cell model still runs. Measured for Windows
# fp8 only. Without fp8 the bf16 path is unmeasured, so Linux and non-fp8 CUDA
# keep the 15 floor.
CUDA_VRAM_FLOOR_GB = 9
LINUX_VRAM_FLOOR_GB = 15
CUDA_FULL_VRAM_GB = 31

# CUDA OffloadMode.CPU pre-pins every transformer block in host RAM. ltx-pipelines
# documents that path as ~36 GB RAM + ~5 GB VRAM. On Linux those buffers are
# page-locked (unswappable), so a 32 GB machine — the advertised local-gen floor —
# dies in the kernel OOM killer with VRAM still empty (LTX-Desktop#163). Below this
# total-RAM floor, stream from disk instead (OffloadMode.DISK, ~5 GB host).
CUDA_CPU_OFFLOAD_RAM_FLOOR_GB = 40

# First compute capability with native fp8 (E4M3/E5M2): Ada Lovelace.
# Ampere (8.x) and Turing (7.5) have no hardware fp8.
_FP8_MIN_COMPUTE_CAPABILITY = (8, 9)


def get_device_type(device: str | torch.device | object | None) -> str:
    if device is None:
        return "cpu"

    device_type = getattr(device, "type", None)
    if isinstance(device_type, str):
        return device_type

    if isinstance(device, str):
        try:
            return str(torch.device(device).type)
        except Exception:
            logger.warning("Could not parse device string '%s', using it as-is", device, exc_info=True)
            return device

    return "cpu"


def _device_index(device: str | torch.device | object | None) -> int:
    """CUDA device ordinal for capability probes. Unindexed shapes mean 0."""
    # NB: torch stubs type .index as int, but at runtime an unindexed device
    # (torch.device("cuda")) carries None — hence `or 0`, not `is None`.
    if isinstance(device, torch.device):
        return device.index or 0
    if isinstance(device, str):
        try:
            return torch.device(device).index or 0
        except Exception:
            return 0
    index = getattr(device, "index", None)
    return index if isinstance(index, int) else 0


def device_supports_fp8(device: str | torch.device | object | None) -> bool:
    # ROCm PyTorch also reports device.type == "cuda" (see runtime_config.accelerator),
    # so this must not be a plain device-type check: ROCm has no fp8_cast kernel support
    # here yet, and would otherwise be silently misidentified as CUDA/NVIDIA.
    # Pre-Ada NVIDIA (compute capability < 8.9) has no hardware fp8 either.
    # Originally contributed by boxwrench in https://github.com/Lightricks/LTX-Desktop/pull/160
    if get_device_type(device) != "cuda":
        return False
    if getattr(torch.version, "hip", None):
        return False
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.get_device_capability(_device_index(device)) >= _FP8_MIN_COMPUTE_CAPABILITY
    except Exception:
        logger.warning("fp8 capability probe failed; treating device as non-fp8", exc_info=True)
        return False


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


def decide_cuda_local_generation_mode(
    system: str,
    cuda_available: bool,
    vram_gb: float | None,
    fp8_capable: bool = True,
) -> Literal["full_models_loading", "streaming_models_loading", "unsupported"]:
    """Windows and Linux load mode from discrete VRAM.

    ``runtime_config.runtime_policy.decide_local_generation_mode`` calls this for
    those systems and keeps the Darwin rules. A host that copies only
    ``inference/`` calls this directly.

    - "unsupported": CUDA is missing, VRAM is unknown, or VRAM is below the floor
      (Windows fp8 uses ``CUDA_VRAM_FLOOR_GB``; Linux and non-fp8 use
      ``LINUX_VRAM_FLOOR_GB``).
    - "streaming_models_loading": enough to run, but below ``CUDA_FULL_VRAM_GB``,
      or no hardware fp8. The full floor assumes the fp8-halved transformer.
    - "full_models_loading": fp8-capable and at or above ``CUDA_FULL_VRAM_GB``.
    """
    if system not in ("Windows", "Linux"):
        return "unsupported"
    if not cuda_available or vram_gb is None:
        return "unsupported"
    floor_gb = CUDA_VRAM_FLOOR_GB if system == "Windows" and fp8_capable else LINUX_VRAM_FLOOR_GB
    if vram_gb < floor_gb:
        return "unsupported"
    # Without fp8 (ROCm today), holding the full bf16 transformer resident would
    # OOM at CUDA_FULL_VRAM_GB. Stay on streaming until a bf16 floor exists.
    # Originally contributed by boxwrench in https://github.com/Lightricks/LTX-Desktop/pull/160
    if not fp8_capable or vram_gb < CUDA_FULL_VRAM_GB:
        return "streaming_models_loading"
    return "full_models_loading"
