"""What this device has already run cleanly, and whether a finished run counts.

A clean run of shape (seq, canvas_px, windows) proves every job in the same
bucket whose three numbers are all at or under the run's. Two runs do not
prove the box they span (a long 1080p clip and a short 4K clip say nothing
about a long 4K clip), so each bucket keeps its individual run points and
drops only the points another point covers: a Pareto frontier.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final, Literal

from runtime_config.budget_decision import LoadMode
from runtime_config.budget_evidence import JobShape
from runtime_config.measured_cells import BudgetFeature

# Bump with ltx-core / ltx-pipelines or a law refit: old proofs stop counting.
CALIBRATION_EPOCH: Final = 2
# Unmeasured (D7): the 1080p retake ladder ran clean about 1.3 GiB under free
# and paged about 0.4 GiB over, so the line sits between them.
PAGING_MARGIN_BYTES: Final = 1 * 1024**3
MIN_FREE_FLOOR_BYTES: Final = 512 * 1024**2
SWAP_GROWTH_TOLERANCE_BYTES: Final = 256 * 1024**2

CalibrationSystem = Literal["Windows", "Linux", "Darwin"]


@dataclass(frozen=True, slots=True)
class DeviceIdentity:
    system: CalibrationSystem
    gpu_model: str
    # CUDA VRAM bytes; total RAM on Darwin (unified memory).
    total_memory_bytes: int


@dataclass(frozen=True, slots=True)
class CalibrationBucket:
    model_id: str
    feature: BudgetFeature
    adapter_id: str
    stage_mode: str
    load_mode: LoadMode
    tiled: bool
    chunked: bool


@dataclass(frozen=True, slots=True)
class EnvelopePoint:
    seq: float
    canvas_px: int
    windows: int
    tiles: int = field(default=1, compare=False)

    def within(self, other: EnvelopePoint) -> bool:
        """Every axis at or under ``other``'s, so an equal re-run is covered."""
        return (
            self.seq <= other.seq
            and self.canvas_px <= other.canvas_px
            and self.windows <= other.windows
        )


@dataclass(frozen=True, slots=True)
class JobEnvelope:
    bucket: CalibrationBucket
    point: EnvelopePoint
    priced_gib: float


@dataclass(frozen=True, slots=True)
class FrontierPoint:
    point: EnvelopePoint
    # Reserved growth over the run; None on Darwin, which has no peak probe.
    peak_growth_bytes: int | None
    wall_s: float | None = field(default=None, compare=False)


@dataclass(frozen=True, slots=True)
class DeviceCalibration:
    identity: DeviceIdentity | None
    frontier: Mapping[CalibrationBucket, tuple[FrontierPoint, ...]]
    machine_factor: float | None = None

    def covering(self, envelope: JobEnvelope) -> FrontierPoint | None:
        """A stored point the job is within, preferring the smallest measured growth."""
        covering = [
            stored
            for stored in self.frontier.get(envelope.bucket, ())
            if envelope.point.within(stored.point)
        ]
        if not covering:
            return None
        return min(
            covering,
            key=lambda stored: (
                stored.peak_growth_bytes is None,
                stored.peak_growth_bytes or 0,
            ),
        )

    def with_frontier(
        self,
        bucket: CalibrationBucket,
        points: tuple[FrontierPoint, ...],
        *,
        machine_factor: float | None = None,
    ) -> DeviceCalibration:
        return DeviceCalibration(
            identity=self.identity,
            frontier={**self.frontier, bucket: points},
            machine_factor=self.machine_factor if machine_factor is None else machine_factor,
        )

    def with_machine_factor(self, machine_factor: float | None) -> DeviceCalibration:
        return DeviceCalibration(
            identity=self.identity,
            frontier=self.frontier,
            machine_factor=machine_factor,
        )


EMPTY_CALIBRATION: Final = DeviceCalibration(identity=None, frontier={})


def merge_frontier(
    points: tuple[FrontierPoint, ...], new: FrontierPoint
) -> tuple[FrontierPoint, ...] | None:
    """None when ``new`` is within a stored point. Else ``new`` plus the points not within it."""
    if any(new.point.within(stored.point) for stored in points):
        return None
    return (new, *(stored for stored in points if not stored.point.within(new.point)))


def envelope_for(shape: JobShape, load_mode: LoadMode, priced_gib: float) -> JobEnvelope:
    return JobEnvelope(
        bucket=CalibrationBucket(
            model_id=shape.model_id,
            feature=shape.feature,
            adapter_id=shape.adapter_id,
            stage_mode=shape.stage_mode,
            load_mode=load_mode,
            tiled=shape.tiled,
            chunked=shape.chunked,
        ),
        point=EnvelopePoint(
            seq=shape.seq,
            canvas_px=shape.canvas_px,
            windows=shape.windows,
            tiles=shape.tiles,
        ),
        priced_gib=priced_gib,
    )


@dataclass(frozen=True, slots=True)
class RunMemoryBaseline:
    free_vram_bytes: int | None
    reserved_bytes: int | None
    swap_used_bytes: int | None


@dataclass(frozen=True, slots=True)
class RunMemoryEnd:
    peak_reserved_bytes: int | None
    min_free_vram_bytes: int | None
    swap_used_bytes: int | None


CleanVerdict = Literal["clean", "vram_growth", "free_floor", "swap_growth", "no_evidence"]


def clean_run_verdict(
    identity: DeviceIdentity, baseline: RunMemoryBaseline, end: RunMemoryEnd
) -> CleanVerdict:
    """Did this run have room? A missing probe fails closed as ``no_evidence``.

    CUDA: reserved growth stays under start-free minus the paging margin,
    free never hits the floor (another app growing mid-run), and host swap
    does not grow when swap readings are present (catches CPU-offload pagefile
    thrashing while treating missing swap probes as clean). Darwin: swap did
    not grow, which on unified memory is the only signal.
    """
    if identity.system != "Darwin":
        growth = peak_growth_bytes(baseline, end)
        if growth is None or baseline.free_vram_bytes is None or end.min_free_vram_bytes is None:
            return "no_evidence"
        if growth > baseline.free_vram_bytes - PAGING_MARGIN_BYTES:
            return "vram_growth"
        if end.min_free_vram_bytes < MIN_FREE_FLOOR_BYTES:
            return "free_floor"
        if (
            baseline.swap_used_bytes is not None
            and end.swap_used_bytes is not None
            and end.swap_used_bytes - baseline.swap_used_bytes > SWAP_GROWTH_TOLERANCE_BYTES
        ):
            return "swap_growth"
        return "clean"
    if baseline.swap_used_bytes is None or end.swap_used_bytes is None:
        return "no_evidence"
    if end.swap_used_bytes - baseline.swap_used_bytes > SWAP_GROWTH_TOLERANCE_BYTES:
        return "swap_growth"
    return "clean"


def peak_growth_bytes(baseline: RunMemoryBaseline, end: RunMemoryEnd) -> int | None:
    if baseline.reserved_bytes is None or end.peak_reserved_bytes is None:
        return None
    return max(0, end.peak_reserved_bytes - baseline.reserved_bytes)
