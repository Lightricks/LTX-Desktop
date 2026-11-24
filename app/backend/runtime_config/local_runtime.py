"""Startup local-runtime resolution for generation mode and hardware capacity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from runtime_config.runtime_policy import LocalGenerationMode, decide_local_generation_mode
from runtime_config.video_job_budget import MemoryBudget
from services.gpu_info.gpu_info import GpuInfo
from services.storage_info.storage_info import DriveType, StorageInfo


class LocalRuntimeConfigKwargs(TypedDict):
    local_generations_mode: LocalGenerationMode
    vram_gb: int | None
    available_ram_gb: int | None
    darwin_unified_memory: bool
    system: str
    vram_bytes: int | None
    usable_vram_bytes: int | None
    total_ram_bytes: int | None
    checkpoint_drive_type: DriveType | None
    gpu_device_name: str | None
    fp8_capable: bool


@dataclass(frozen=True)
class LocalRuntime:
    mode: LocalGenerationMode
    vram_gb: int | None
    available_ram_gb: int | None
    darwin: bool
    system: str
    vram_bytes: int | None
    total_ram_bytes: int | None
    checkpoint_drive_type: DriveType | None
    gpu_device_name: str | None
    fp8_capable: bool
    usable_vram_bytes: int | None = None

    def runtime_config_kwargs(
        self,
        *,
        local_generations_mode: LocalGenerationMode | None = None,
    ) -> LocalRuntimeConfigKwargs:
        return {
            "local_generations_mode": self.mode if local_generations_mode is None else local_generations_mode,
            "vram_gb": self.vram_gb,
            "available_ram_gb": self.available_ram_gb,
            "darwin_unified_memory": self.darwin,
            "system": self.system,
            "vram_bytes": self.vram_bytes,
            "usable_vram_bytes": self.usable_vram_bytes,
            "total_ram_bytes": self.total_ram_bytes,
            "checkpoint_drive_type": self.checkpoint_drive_type,
            "gpu_device_name": self.gpu_device_name,
            "fp8_capable": self.fp8_capable,
        }


def resolve_local_runtime(
    gpu_info: GpuInfo,
    *,
    system: str,
    fp8_capable: bool,
    storage_info: StorageInfo,
    macos_major: int | None = None,
) -> LocalRuntime:
    cuda_available = gpu_info.get_cuda_available()
    mps_available = gpu_info.get_mps_available()
    vram_gb = gpu_info.get_vram_total_gb()
    vram_bytes = gpu_info.get_cuda_total_vram_bytes()
    usable_vram_bytes = gpu_info.get_free_cuda_vram_bytes()
    total_ram_bytes = gpu_info.get_total_ram_bytes()
    # Darwin has no discrete VRAM (unified memory). Eligibility uses *total* RAM
    # (``get_vram_total_gb`` / SKU). Job sizing still uses free RAM at launch.
    available_ram_gb = gpu_info.get_available_ram_gb() if system == "Darwin" else None
    ram_gb = vram_gb if system == "Darwin" else None

    # Server-owned source of truth for mode selection. The floor sees exact total
    # VRAM with the integer as fallback. Exactness changes no verdict against
    # integer floors — it keeps fractional floors expressible. Free VRAM is
    # deliberately None here: process mode is a hardware class, not momentary
    # occupancy.
    vram_gib = MemoryBudget(
        total_vram_bytes=vram_bytes,
        free_vram_bytes=None,
        total_ram_bytes=total_ram_bytes,
        memory_gb=vram_gb,
        # Mode resolution is a hardware class: no pool sample here. Request
        # filters sample the live pool per request via GpuInfo.get_mps_pool_bytes().
        mps_pool_bytes=None,
    ).total_vram_gib
    mode = decide_local_generation_mode(
        system=system,
        cuda_available=cuda_available,
        vram_gb=vram_gib,
        mps_available=mps_available,
        ram_gb=ram_gb,
        fp8_capable=fp8_capable,
        macos_major=macos_major,
    )
    return LocalRuntime(
        mode=mode,
        vram_gb=vram_gb,
        available_ram_gb=available_ram_gb,
        darwin=system == "Darwin",
        system=system,
        vram_bytes=vram_bytes,
        usable_vram_bytes=usable_vram_bytes,
        total_ram_bytes=total_ram_bytes,
        checkpoint_drive_type=storage_info.get_drive_type(),
        gpu_device_name=gpu_info.get_device_name(),
        fp8_capable=fp8_capable,
    )
