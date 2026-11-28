"""How much to trust a priced verdict, and the job shape it priced.

Pricing tags every verdict with its evidence; admission reads the tag. A
``Fitted`` verdict sits inside the lab-measured range, ``GenericCurve`` comes
from the legacy curve (no measured route), and ``Extrapolated`` prices a shape
no lab run covers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from runtime_config.measured_cells import BudgetFeature
from runtime_config.reject_reasons import UnmeasuredBasis


@dataclass(frozen=True, slots=True)
class Fitted:
    pass


@dataclass(frozen=True, slots=True)
class GenericCurve:
    pass


@dataclass(frozen=True, slots=True)
class Extrapolated:
    basis: UnmeasuredBasis


PriceEvidence = Fitted | GenericCurve | Extrapolated


@dataclass(frozen=True, slots=True)
class JobShape:
    """What the priced job runs, minus the load mode. Device calibration keys on it."""

    feature: BudgetFeature
    model_id: str
    # IC-LoRA adapter id; "" for Fast and edit (their laws ignore LoRAs).
    adapter_id: str
    # IcLoraStageMode value; "" for Fast and edit.
    stage_mode: str
    tiled: bool
    chunked: bool
    # Heaviest pass tokens, per window when chunked.
    seq: float
    # Output canvas (Fast, edit) or heavy-pass canvas (IC-LoRA), in pixels.
    canvas_px: int
    windows: int
    tiles: int = field(default=1, compare=False)
