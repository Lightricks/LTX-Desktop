"""Device calibration: watch a local GPU run's memory and record it when it ran clean."""

from __future__ import annotations

import logging
from threading import RLock
from typing import TYPE_CHECKING

from handlers.base import StateHandlerBase
from runtime_config.device_calibration import (
    CALIBRATION_EPOCH,
    CleanVerdict,
    DeviceCalibration,
    DeviceIdentity,
    FrontierPoint,
    JobEnvelope,
    RunMemoryBaseline,
    RunMemoryEnd,
    clean_run_verdict,
    peak_growth_bytes,
)
from services.records import UnavailableError
from services.store import CalibrationStore
from services.unavailable_store import UnavailableStore
from state.app_state_types import AppState

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig
    from services.gpu_info.gpu_info import GpuInfo

logger = logging.getLogger(__name__)


def device_identity(config: RuntimeConfig) -> DeviceIdentity | None:
    """None when the GPU name or total memory is unknown: nothing is loaded or recorded."""
    match config.system:
        case "Darwin":
            total = config.total_ram_bytes
        case "Windows" | "Linux":
            total = config.vram_bytes
        case _:
            return None
    if not config.gpu_device_name or total is None or total <= 0:
        return None
    return DeviceIdentity(
        system=config.system, gpu_model=config.gpu_device_name, total_memory_bytes=total
    )


class CalibrationHandler(StateHandlerBase):
    def __init__(
        self, *, state: AppState, lock: RLock, config: RuntimeConfig, gpu_info: GpuInfo
    ) -> None:
        super().__init__(state, lock, config, gpu_info=gpu_info)
        self._store: CalibrationStore = UnavailableStore()

    def bind(self, store: CalibrationStore) -> None:
        self._store = store

    def load(self) -> None:
        identity = device_identity(self.config)
        if identity is None:
            return
        try:
            frontier = self._store.load_device_frontier(identity, CALIBRATION_EPOCH)
            machine_factor = self._store.load_device_machine_factor(identity, CALIBRATION_EPOCH)
        except UnavailableError:
            logger.warning("[calibration] frontier load failed; lab tables only", exc_info=True)
            return
        with self._lock:
            self.state.device_calibration = DeviceCalibration(
                identity=identity, frontier=frontier, machine_factor=machine_factor
            )

    def open_run(self) -> RunMemoryBaseline | None:
        if device_identity(self.config) is None:
            return None
        return self._gpu.begin_run_memory_watch()

    def close_run(self) -> RunMemoryEnd:
        return self._gpu.end_run_memory_watch()

    def record_if_clean(
        self,
        ran: JobEnvelope,
        baseline: RunMemoryBaseline,
        end: RunMemoryEnd,
        *,
        generation_id: str,
        wall_s: float | None = None,
        conv_vae: bool = False,
    ) -> CleanVerdict:
        """Widen the frontier with a clean run. Never raises: a lost record keeps the old warning."""
        identity = device_identity(self.config)
        if identity is None:
            return "no_evidence"
        verdict = clean_run_verdict(identity, baseline, end)
        if verdict != "clean":
            logger.info("[calibration] %s not recorded: %s", generation_id, verdict)
            return verdict
        try:
            points = self._store.widen_device_frontier(
                identity,
                CALIBRATION_EPOCH,
                ran.bucket,
                FrontierPoint(
                    point=ran.point,
                    peak_growth_bytes=peak_growth_bytes(baseline, end),
                    wall_s=wall_s,
                ),
                priced_gib=ran.priced_gib,
                generation_id=generation_id,
                conv_vae=conv_vae,
            )
        except UnavailableError:
            logger.warning("[calibration] %s record failed", generation_id, exc_info=True)
            return verdict
        # A failed refresh must not hide a frontier point that is already stored.
        try:
            machine_factor = self._store.load_device_machine_factor(identity, CALIBRATION_EPOCH)
        except UnavailableError:
            logger.warning(
                "[calibration] %s machine factor refresh failed", generation_id, exc_info=True
            )
            machine_factor = None
        with self._lock:
            if points is not None:
                self.state.device_calibration = self.state.device_calibration.with_frontier(
                    ran.bucket, points, machine_factor=machine_factor
                )
            elif machine_factor is not None:
                self.state.device_calibration = self.state.device_calibration.with_machine_factor(
                    machine_factor
                )
        logger.info(
            "[calibration] %s clean: %s seq=%.0f canvas=%d windows=%d (%s)",
            generation_id,
            ran.bucket.feature,
            ran.point.seq,
            ran.point.canvas_px,
            ran.point.windows,
            "widened" if points is not None else "already covered",
        )
        return verdict
