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

from services.gpu_info.gpu_info import GpuTelemetryPayload

logger = logging.getLogger(__name__)


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

    def get_available_ram_gb(self) -> int | None:
        """Free/available system RAM, not total -- used to gate local generation on
        Apple Silicon (unified memory shared with the OS/app, so total RAM alone
        overstates real headroom)."""
        try:
            return int(psutil.virtual_memory().available // (1024**3))
        except Exception:
            logger.warning("Failed to query available system RAM", exc_info=True)
            return None
