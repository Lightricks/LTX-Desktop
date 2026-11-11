"""GPU info query service implementation."""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
from typing import Protocol, cast

import psutil
import torch

from runtime_config.device_calibration import RunMemoryBaseline, RunMemoryEnd
from services.gpu_info import run_peak
from services.gpu_info.gpu_info import GpuTelemetryPayload

logger = logging.getLogger(__name__)


def platform_label() -> str:
    """OS/arch for startup logs. Includes macOS version on Darwin (15+ is required
    for fused MPSGraph SDPA; 14 falls back to stock and OOMs on video lengths)."""
    system = platform.system()
    machine = platform.machine()
    if system == "Darwin":
        mac_ver = platform.mac_ver()[0]
        if mac_ver:
            return f"{system} {mac_ver} ({machine})"
    return f"{system} ({machine})"


class _CudaDeviceProperties(Protocol):
    total_memory: int


class GpuInfoImpl:
    """Wraps CUDA and MPS runtime queries."""

    def _get_macos_chip_name(self) -> str | None:
        if platform.system() != "Darwin":
            return None

        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            chip = result.stdout.strip()
            return chip if chip else None
        except Exception:
            logger.warning("Failed to read macOS chip name", exc_info=True)
            return None

    def _get_system_ram_mb(self) -> int:
        try:
            if sys.platform == "win32":
                return 0
            return int((os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) // (1024 * 1024))
        except Exception:
            logger.warning("Failed to query system RAM", exc_info=True)
            return 0

    def get_gpu_info(self) -> GpuTelemetryPayload:
        if self.get_cuda_available():
            try:
                import pynvml  # type: ignore[reportMissingModuleSource]

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                raw_name = pynvml.nvmlDeviceGetName(handle)
                name = raw_name.decode("utf-8", errors="replace") if isinstance(raw_name, bytes) else str(raw_name)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                pynvml.nvmlShutdown()
                return {
                    "name": name,
                    "vram": memory.total // (1024 * 1024),
                    "vramUsed": memory.used // (1024 * 1024),
                }
            except Exception:
                logger.warning("Failed to query NVML GPU memory/name; falling back to torch metadata", exc_info=True)
                device_name = self.get_device_name() or "Unknown"
                total_vram_gb = self.get_vram_total_gb() or 0
                return {
                    "name": device_name,
                    "vram": total_vram_gb * 1024,
                    "vramUsed": 0,
                }

        if self.get_mps_available():
            chip = self._get_macos_chip_name()
            name = f"{chip} (MPS)" if chip else "Apple Silicon (MPS)"
            return {
                "name": name,
                "vram": self._get_system_ram_mb(),
                "vramUsed": 0,
            }

        return {"name": "Unknown", "vram": 0, "vramUsed": 0}

    def get_cuda_available(self) -> bool:
        try:
            return bool(torch.cuda.is_available())
        except Exception:
            logger.warning("Failed to query CUDA availability", exc_info=True)
            return False

    def get_mps_available(self) -> bool:
        try:
            return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        except Exception:
            logger.warning("Failed to query MPS availability", exc_info=True)
            return False

    def get_gpu_available(self) -> bool:
        return self.get_cuda_available() or self.get_mps_available()

    def get_device_name(self) -> str | None:
        if self.get_cuda_available():
            try:
                return str(torch.cuda.get_device_name(0))
            except Exception:
                logger.warning("Failed to query CUDA device name", exc_info=True)
                return None

        if self.get_mps_available():
            chip = self._get_macos_chip_name()
            return f"{chip} (MPS)" if chip else "Apple Silicon (MPS)"

        return None

    def get_vram_total_gb(self) -> int | None:
        if self.get_cuda_available():
            try:
                properties = cast(
                    _CudaDeviceProperties,
                    torch.cuda.get_device_properties(0),  # type: ignore[reportUnknownMemberType]
                )
                return int(properties.total_memory // (1024**3))
            except Exception:
                logger.warning("Failed to query CUDA total VRAM", exc_info=True)
                return None

        if self.get_mps_available():
            try:
                if sys.platform == "win32":
                    return None
                return int((os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) // (1024**3))
            except Exception:
                logger.warning("Failed to query MPS total memory", exc_info=True)
                return None

        return None

    def get_cuda_total_vram_bytes(self) -> int | None:
        """Exact CUDA device 0 total VRAM in bytes. None if CUDA is unavailable or the query fails."""
        if not self.get_cuda_available():
            return None
        try:
            properties = cast(
                _CudaDeviceProperties,
                torch.cuda.get_device_properties(0),  # type: ignore[reportUnknownMemberType]
            )
            return int(properties.total_memory)
        except Exception:
            logger.warning("Failed to query CUDA total VRAM bytes", exc_info=True)
            return None

    def get_available_ram_gb(self) -> int | None:
        """Free/available system RAM, not total — used for Darwin per-job sizing.

        Local-generation *eligibility* on Apple Silicon uses total RAM
        (``get_vram_total_gb``), not this figure.
        """
        try:
            return int(psutil.virtual_memory().available // (1024**3))
        except Exception:
            logger.warning("Failed to query available system RAM", exc_info=True)
            return None

    def get_total_ram_bytes(self) -> int | None:
        """Exact total system RAM in bytes. Works on Windows (psutil, not sysconf)."""
        try:
            return int(psutil.virtual_memory().total)
        except Exception:
            logger.warning("Failed to query total system RAM", exc_info=True)
            return None

    def get_free_cuda_vram_bytes(self) -> int | None:
        """Free VRAM bytes on CUDA device 0, or None when unknowable.

        Fail-open probe for availability decisions: None (no CUDA, driver
        oddities, any exception) means callers fall back to total VRAM.
        Device 0 matches the total-VRAM probe; mixed-GPU boxes are
        approximate. Sampled live per call — never frozen into config — so a
        closing browser is visible on the next request.
        """
        try:
            if not torch.cuda.is_available():
                return None
            free_bytes, _total_bytes = torch.cuda.mem_get_info(0)
            return int(free_bytes)
        except Exception:
            logger.debug("free VRAM probe failed; callers fall back to total", exc_info=True)
            return None

    def get_cuda_live_allocated_bytes(self) -> int | None:
        """Live allocated bytes on CUDA device 0, or None when unknowable.

        Fail-None probe for the evict-retry guard: None (no CUDA, any
        exception) means callers skip the retry. Device 0 matches the
        free-VRAM probe. At the inter-job moment nothing else is live, so
        this approximates the warm weights — the most an evict could free.
        """
        try:
            if not torch.cuda.is_available():
                return None
            return int(torch.cuda.memory_allocated(0))
        except Exception:
            logger.debug(
                "live-allocated probe failed; callers skip the evict retry", exc_info=True
            )
            return None

    def get_cuda_reserved_bytes(self) -> int | None:
        """Allocator-reserved bytes on CUDA device 0, or None when unknowable.

        Fail-None logging probe for the gate-time VRAM triple. Device 0
        matches the free-VRAM probe.
        """
        try:
            if not torch.cuda.is_available():
                return None
            return int(torch.cuda.memory_reserved(0))
        except Exception:
            logger.debug(
                "reserved probe failed; budget lines log it as unknown", exc_info=True
            )
            return None

    def get_mps_pool_bytes(self) -> int | None:
        """MPS allocator pool (recommended_max) in bytes, or None when unknowable.

        Fail-None probe. Unlike ``get_free_cuda_vram_bytes`` (where None
        falls back to total), None here is terminal: Mac-profiled callers
        fail closed to unsupported. Sampled live per call — never frozen
        into config — so a shrinking pool is visible on the next request.
        """
        try:
            mps = getattr(torch.backends, "mps", None)
            if mps is None or not bool(mps.is_available()):
                return None
            return int(torch.mps.recommended_max_memory())
        except Exception:
            logger.debug("MPS pool probe failed; Mac callers fail closed", exc_info=True)
            return None

    def get_swap_used_bytes(self) -> int | None:
        try:
            return int(psutil.swap_memory().used)
        except Exception:
            logger.debug("swap probe failed; the run is not recorded", exc_info=True)
            return None

    def begin_run_memory_watch(self) -> RunMemoryBaseline:
        baseline = RunMemoryBaseline(
            free_vram_bytes=self.get_free_cuda_vram_bytes(),
            reserved_bytes=self.get_cuda_reserved_bytes(),
            swap_used_bytes=self.get_swap_used_bytes(),
        )
        run_peak.begin_run()
        return baseline

    def end_run_memory_watch(self) -> RunMemoryEnd:
        peak_reserved, min_free = run_peak.end_run()
        return RunMemoryEnd(
            peak_reserved_bytes=peak_reserved,
            min_free_vram_bytes=min_free,
            swap_used_bytes=self.get_swap_used_bytes(),
        )
