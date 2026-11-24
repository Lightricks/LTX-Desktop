"""Fake GPU info service for tests."""

from __future__ import annotations

from collections.abc import Callable

from runtime_config.device_calibration import RunMemoryBaseline, RunMemoryEnd
from services.gpu_info import run_peak
from services.interfaces import GpuTelemetryPayload


class FakeGpuInfo:
    def __init__(self) -> None:
        self.cuda_available = False
        self.mps_available = False
        self.gpu_name: str | None = None
        self.vram_gb: int | None = None
        self.vram_bytes: int | None = None
        self.available_ram_gb: int | None = None
        self.total_ram_bytes: int | None = None
        self.gpu_info: GpuTelemetryPayload = {"name": "Unknown", "vram": 0, "vramUsed": 0}
        self.free_vram_bytes: int | None = None
        self.live_allocated_bytes: int | None = None
        self.reserved_bytes: int | None = None
        self.mps_pool_bytes: int | None = None
        # Dynamic overrides for mid-call flips (evict-retry re-samples inside
        # one gate call, so a static field cannot model the freed memory).
        self.free_vram_bytes_fn: Callable[[], int | None] | None = None
        self.live_allocated_bytes_fn: Callable[[], int | None] | None = None
        self.swap_used_bytes: int | None = None
        # What the run watch reports at the end; all-None reads as no evidence.
        self.run_end = RunMemoryEnd(
            peak_reserved_bytes=None, min_free_vram_bytes=None, swap_used_bytes=None
        )

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

    def get_cuda_total_vram_bytes(self) -> int | None:
        return self.vram_bytes

    def get_available_ram_gb(self) -> int | None:
        return self.available_ram_gb

    def get_total_ram_bytes(self) -> int | None:
        return self.total_ram_bytes

    def get_free_cuda_vram_bytes(self) -> int | None:
        if self.free_vram_bytes_fn is not None:
            return self.free_vram_bytes_fn()
        return self.free_vram_bytes

    def get_cuda_live_allocated_bytes(self) -> int | None:
        if self.live_allocated_bytes_fn is not None:
            return self.live_allocated_bytes_fn()
        return self.live_allocated_bytes

    def get_cuda_reserved_bytes(self) -> int | None:
        return self.reserved_bytes

    def get_mps_pool_bytes(self) -> int | None:
        return self.mps_pool_bytes

    def get_swap_used_bytes(self) -> int | None:
        return self.swap_used_bytes

    def begin_run_memory_watch(self) -> RunMemoryBaseline:
        run_peak.begin_run()
        return RunMemoryBaseline(
            free_vram_bytes=self.get_free_cuda_vram_bytes(),
            reserved_bytes=self.reserved_bytes,
            swap_used_bytes=self.swap_used_bytes,
        )

    def end_run_memory_watch(self) -> RunMemoryEnd:
        run_peak.end_run()
        return self.run_end
