"""Fake GPU info service for tests."""

from __future__ import annotations

from services.interfaces import GpuTelemetryPayload


class FakeGpuInfo:
    def __init__(self) -> None:
        self.cuda_available = False
        self.mps_available = False
        self.gpu_name: str | None = None
        self.vram_gb: int | None = None
        self.available_ram_gb: int | None = None
        self.gpu_info: GpuTelemetryPayload = {"name": "Unknown", "vram": 0, "vramUsed": 0}

    def get_gpu_info(self) -> GpuTelemetryPayload:
        return self.gpu_info

    def get_cuda_available(self) -> bool:
        return self.cuda_available

    def get_mps_available(self) -> bool:
        return self.mps_available

    def get_gpu_available(self) -> bool:
        return self.cuda_available or self.mps_available

    def get_device_name(self) -> str | None:
        return self.gpu_name

    def get_vram_total_gb(self) -> int | None:
        return self.vram_gb

    def get_available_ram_gb(self) -> int | None:
        return self.available_ram_gb
