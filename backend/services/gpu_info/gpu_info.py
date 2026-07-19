"""GPU info service protocol definitions."""

from __future__ import annotations

from typing import Protocol, TypedDict


class GpuTelemetryPayload(TypedDict):
    name: str
    vram: int
    vramUsed: int


class GpuInfo(Protocol):
    def get_gpu_info(self) -> GpuTelemetryPayload:
        ...

    def get_cuda_available(self) -> bool:
        ...

    def get_mps_available(self) -> bool:
        ...

    def get_gpu_available(self) -> bool:
        ...

    def get_device_name(self) -> str | None:
        ...

    def get_vram_total_gb(self) -> int | None:
        ...

    def get_available_ram_gb(self) -> int | None:
        ...
