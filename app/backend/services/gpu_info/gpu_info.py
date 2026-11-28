"""GPU info service protocol definitions."""

from __future__ import annotations

from typing import Protocol, TypedDict

from runtime_config.device_calibration import RunMemoryBaseline, RunMemoryEnd


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

    def get_cuda_total_vram_bytes(self) -> int | None:
        ...

    def get_available_ram_gb(self) -> int | None:
        ...

    def get_total_ram_bytes(self) -> int | None:
        ...

    def get_free_cuda_vram_bytes(self) -> int | None:
        """Free VRAM bytes on CUDA device 0, or None when unknowable.

        Fail-open probe for availability decisions: None means callers fall
        back to total VRAM. Sampled live per call — never frozen into config.
        """

    def get_cuda_live_allocated_bytes(self) -> int | None:
        """Live allocated bytes on CUDA device 0, or None when unknowable.

        Fail-None probe for the evict-retry guard: None means callers skip
        the retry.
        """

    def get_cuda_reserved_bytes(self) -> int | None:
        """Allocator-reserved bytes on CUDA device 0, or None when unknowable.

        Fail-None logging probe for the gate-time VRAM triple: reserved minus
        allocated is dead torch cache, which discriminates reclaimable
        occupancy from other-apps occupancy on the next triage.
        """

    def get_mps_pool_bytes(self) -> int | None:
        """MPS allocator pool (recommended_max) in bytes, or None when unknowable.

        Fail-None probe: Mac-profiled callers fail closed to unsupported.
        Sampled live per call — never frozen into config.
        """

    def get_swap_used_bytes(self) -> int | None:
        """Swap in use (pagefile on Windows), or None when unknowable."""

    def begin_run_memory_watch(self) -> RunMemoryBaseline:
        """Sample free, reserved, and swap, and start the run peak. Fail-None per field."""
        ...

    def end_run_memory_watch(self) -> RunMemoryEnd:
        """Close the run peak and sample swap. Fail-None per field."""
        ...
