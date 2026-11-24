"""How sure the Home memory meter may be about a priced job's ask.

The meter draws ``PricedJob`` asks, so its confidence is read off the same evidence admission
uses. It never changes a verdict.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, assert_never

from api_types import GenerationRisk, JobMemoryAsk, MemoryCapacity, MemoryConfidence
from runtime_config.budget_admission import compared_available_gib
from runtime_config.budget_decision import IcLoraJobLoad, VideoJobLoad
from runtime_config.budget_evidence import Extrapolated, Fitted, GenericCurve
from runtime_config.eta import EtaSystem, default_machine_factor, estimate_job_shape_seconds
from runtime_config.measured_cells import BudgetRoute

if TYPE_CHECKING:
    from runtime_config.budget_admission import PricedJob
    from runtime_config.video_job_budget import LocalJobFilter

_ASK_ROUNDING = 1000

# Where a single pass past the fitted range may really land, as ratios of its ask. Drawn as a soft
# trailing edge, never as numbers. Low: past-fit passes land under the ask by about what fitted
# ones do, which on a Mac includes the edge-basis margin above the smallest passing cap. High: a
# stream law that runs past its fit can stop holding, and an under-ask is the failure that matters
# (an OOM, or Windows paging partway through the job), so the top is the worst such miss seen on
# either route. Narrowing either side needs a past-fit run on that route.
ESTIMATE_BAND_RATIOS: dict[BudgetRoute, tuple[float, float]] = {
    "windows_measured": (0.975, 1.07),
    "mac_measured": (0.95, 1.07),
}


def meter_confidence(priced: PricedJob, *, route: BudgetRoute) -> MemoryConfidence:
    """Solid, estimate, or rough for the meter.

    Confidence follows the route. The legacy curve is rough whatever its tag: a 1440p cell on it
    is tagged unmeasured, but its number is still the generic curve's. A job in parts peaks by its
    third part and stays flat after it, so its ask is as sure as a fitted one even past the
    longest run. A single pass past the measured range has no such plateau to lean on. A clean
    run on this device covers the shape.
    """
    if route == "legacy":
        return "rough"
    evidence = priced.evidence
    match evidence:
        case GenericCurve():
            return "rough"
        case Fitted():
            return "solid"
        case Extrapolated():
            if priced.covered or priced.shape.windows >= 2:
                return "solid"
            return "estimate"
        case _:
            assert_never(evidence)


def estimate_band_gib(ask_gib: float, *, route: BudgetRoute) -> tuple[float, float] | None:
    """The soft band of an estimate. None on the legacy route, which never estimates."""
    ratios = ESTIMATE_BAND_RATIOS.get(route)
    if ratios is None:
        return None
    low, high = ratios
    return ask_gib * low, ask_gib * high


def job_filter_machine_factor(job_filter: LocalJobFilter) -> float | None:
    """Calibrated or hardware-class speed factor for ``job_filter``."""
    if job_filter.calibration.machine_factor is not None:
        return job_filter.calibration.machine_factor
    identity = job_filter.calibration.identity
    if identity is not None:
        factor = default_machine_factor(identity.gpu_model)
        if factor is not None:
            return factor
    return default_machine_factor(job_filter.profile.device_name)


def job_memory_ask(
    priced: PricedJob,
    *,
    route: BudgetRoute,
    system: EtaSystem | None = None,
    risk: GenerationRisk | None = None,
    machine_factor: float | None = None,
    conv_vae: bool = False,
) -> JobMemoryAsk:
    """The meter's reading of a priced job. The ask rounds up, so over never reads as fitting."""
    ask_gib = math.ceil(priced.decision.stream_gib * _ASK_ROUNDING) / _ASK_ROUNDING
    confidence = meter_confidence(priced, route=route)
    band = estimate_band_gib(ask_gib, route=route) if confidence == "estimate" else None
    load_mode = (
        priced.decision.mode
        if isinstance(priced.decision, (VideoJobLoad, IcLoraJobLoad))
        else "streaming_models_loading"
    )
    resolved_system: EtaSystem = (
        system
        if system is not None
        else ("Darwin" if route == "mac_measured" else "Windows")
    )
    estimated_seconds = (
        estimate_job_shape_seconds(
            priced.shape,
            load_mode,
            system=resolved_system,
            machine_factor=machine_factor,
            conv_vae=conv_vae,
        )
        if machine_factor is not None
        else None
    )
    return JobMemoryAsk(
        ask_gib=ask_gib,
        confidence=confidence,
        windows=priced.shape.windows,
        tight=priced.tight,
        band_low_gib=None if band is None else band[0],
        band_high_gib=None if band is None else band[1],
        risk=risk,
        estimated_seconds=estimated_seconds,
    )


def memory_capacity(job_filter: LocalJobFilter) -> MemoryCapacity:
    """The figure the meter's end mark stands for: the one admission compares asks against.

    The legacy curve compares the integer GB fallback when exact VRAM bytes are missing, and its
    rough scale only needs that much.
    """
    darwin = job_filter.profile.system == "Darwin"
    compared = compared_available_gib(job_filter)
    return MemoryCapacity(
        device="mac" if darwin else "gpu",
        capacity_gib=(
            compared if compared is not None or darwin else job_filter.budget.total_vram_gib
        ),
    )
