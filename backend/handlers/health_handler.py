"""Health and hardware info handlers."""

from __future__ import annotations

from datetime import UTC, datetime
import os
from threading import RLock
from typing import TYPE_CHECKING

import psutil

from api_types import (
    GpuInfoResponse,
    GpuTelemetry,
    HealthResponse,
    ModelStatusItem,
    MpsMemoryResponse,
    RuntimeTelemetryResponse,
)
from handlers.base import StateHandlerBase
from handlers.models_handler import ModelsHandler
from services.fast_video_pipeline.mlx_profile import get_mlx_profile_snapshot
from services.interfaces import GpuInfo
from services.local_metal_lease import get_local_metal_lease_snapshot
from state.app_state_types import AppState, GpuSlot, VideoPipelineState

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

_BYTES_PER_MIB = 1024 * 1024


class HealthHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        models_handler: ModelsHandler,
        gpu_info: GpuInfo,
        config: RuntimeConfig,
    ) -> None:
        super().__init__(state, lock, config)
        self._models = models_handler
        self._gpu_info = gpu_info

    def get_health(self) -> HealthResponse:
        active_model: str | None = None
        models_loaded = False

        with self._lock:
            match self.state.gpu_slot:
                case GpuSlot(active_pipeline=VideoPipelineState(pipeline=pipeline)):
                    active_model = pipeline.pipeline_kind
                    models_loaded = True
                case _:
                    pass

        downloaded_checkpoints = self._models.get_downloaded_checkpoints()

        return HealthResponse(
            status="ok",
            models_loaded=models_loaded,
            active_model=active_model,
            gpu_info=GpuTelemetry(**self._gpu_info.get_gpu_info()),
            sage_attention=self.config.use_sage_attention,
            models_status=[
                ModelStatusItem(
                    id="fast",
                    name="LTX-2 Fast",
                    loaded=models_loaded,
                    downloaded=any(cp_id.startswith("ltx-") for cp_id in downloaded_checkpoints),
                ),
            ],
        )

    def get_gpu_info(self) -> GpuInfoResponse:
        return GpuInfoResponse(
            cuda_available=self._gpu_info.get_cuda_available(),
            mps_available=self._gpu_info.get_mps_available(),
            gpu_available=self._gpu_info.get_gpu_available(),
            gpu_name=self._gpu_info.get_device_name(),
            vram_gb=self._gpu_info.get_vram_total_gb(),
            gpu_info=GpuTelemetry(**self._gpu_info.get_gpu_info()),
        )

    def get_mps_memory(self) -> MpsMemoryResponse:
        """Read-only Apple Silicon MPS memory snapshot (torch-tracked / driver-allocated /
        recommended-max, MiB). ``available`` is False off MPS. No side effects; torch is
        imported lazily so the call is cheap and safe on non-MPS hosts."""
        import sys

        import torch

        if sys.platform != "darwin" or not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            return MpsMemoryResponse(available=False)
        try:
            return MpsMemoryResponse(
                available=True,
                allocated_mib=round(torch.mps.current_allocated_memory() / _BYTES_PER_MIB),
                driver_mib=round(torch.mps.driver_allocated_memory() / _BYTES_PER_MIB),
                recommended_max_mib=round(torch.mps.recommended_max_memory() / _BYTES_PER_MIB),
            )
        except Exception:  # noqa: BLE001
            return MpsMemoryResponse(available=False)

    def get_runtime_telemetry(self) -> RuntimeTelemetryResponse:
        """Return one cheap process/system/accelerator memory sample."""
        process_rss_mib = round(psutil.Process(os.getpid()).memory_info().rss / _BYTES_PER_MIB)
        virtual_memory = psutil.virtual_memory()

        active_engine = None
        active_pipeline = None
        with self._lock:
            if self.state.gpu_slot is not None:
                active = self.state.gpu_slot.active_pipeline
                if isinstance(active, VideoPipelineState):
                    active_engine = active.runtime_engine
                    active_pipeline = active.pipeline.pipeline_kind
                else:
                    active_engine = "torch"
                    active_pipeline = type(active).__name__

        mlx_profile = get_mlx_profile_snapshot()

        mps = self.get_mps_memory()
        lease = get_local_metal_lease_snapshot()
        return RuntimeTelemetryResponse(
            sampled_at=datetime.now(UTC).isoformat(),
            active_engine=active_engine,
            active_pipeline=active_pipeline,
            process_rss_mib=process_rss_mib,
            system_total_mib=round(virtual_memory.total / _BYTES_PER_MIB),
            system_available_mib=round(virtual_memory.available / _BYTES_PER_MIB),
            mlx_active_mib=mlx_profile.active_mib if mlx_profile else None,
            mlx_cache_mib=mlx_profile.cache_mib if mlx_profile else None,
            mlx_peak_mib=mlx_profile.peak_mib if mlx_profile else None,
            mlx_profile_status=mlx_profile.status if mlx_profile else None,
            mlx_profile_phase=mlx_profile.phase if mlx_profile else None,
            mlx_profile_path=mlx_profile.profile_path if mlx_profile else None,
            mlx_profile_sampled_at=mlx_profile.sampled_at if mlx_profile else None,
            mlx_runtime_identity=mlx_profile.runtime_identity if mlx_profile else None,
            mps_allocated_mib=mps.allocated_mib,
            mps_driver_mib=mps.driver_mib,
            mps_recommended_max_mib=mps.recommended_max_mib,
            local_metal_lease_status=lease["status"],
            local_metal_lease_reason=lease["reason"],
            local_metal_lease_waited_seconds=lease["waited_seconds"],
            local_metal_lease_owner=lease["owner"],
        )
