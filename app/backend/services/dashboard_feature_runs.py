"""Retake, extend, and IC-LoRA runs for the dashboard render-time grid.

Nothing here is stored. The spec of these features has no aspect ratio, and a
retake or extend keeps the source size. The size and aspect ratio come from the
output video and snap to the nearest ones the app offers. The length is the
edited span for retake and extend, and the output length for IC-LoRA. Both snap
to a two-second step so near-equal lengths share a cell.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from typing import cast, get_args

from api_types import IcLoraRecipeResolution, LTXVideoGenAspectRatio
from services.aspect_ratio import find_closest_aspect_ratio_option

FEATURE_RENDER_GROUPS = frozenset({"retake", "extend", "ic-lora"})
LENGTH_STEP_S = 2.0
_ASPECT_RATIOS = get_args(LTXVideoGenAspectRatio)
# Local edits and IC-LoRA stay inside the 1080p envelope. The label is the short side.
_RESOLUTION_HEIGHTS = tuple(
    sorted(int(label.removesuffix("p")) for label in get_args(IcLoraRecipeResolution))
)


@dataclass(frozen=True, slots=True)
class FeatureRun:
    feature: str
    model: str
    resolution: str
    aspect_ratio: str
    length_s: float
    run_ms: float


def video_metadata(metadata: str) -> dict[str, object] | None:
    """The inner video fields of an asset metadata JSON, or None when it is not a video."""
    try:
        parsed: object = json.loads(metadata)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    inner = cast(dict[str, object], parsed).get("metadata")
    if not isinstance(inner, dict):
        return None
    return cast(dict[str, object], inner)


def nearest_resolution(width: int, height: int) -> str:
    """The offered label whose height is closest to the short side. A tie picks the lower one."""
    short = min(width, height)
    best = min(_RESOLUTION_HEIGHTS, key=lambda option: (abs(option - short), option))
    return f"{best}p"


def length_bucket_s(length_s: float) -> float:
    """Nearest step, never below one step. An exact half rounds up."""
    return max(LENGTH_STEP_S, LENGTH_STEP_S * math.floor(length_s / LENGTH_STEP_S + 0.5))


def _positive(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value > 0 else None


def _pixels(value: object) -> int | None:
    """A whole pixel count of at least 1. A row that has none is skipped, not fatal."""
    size = _positive(value)
    return None if size is None or int(size) < 1 else int(size)


def _feature_run(
    row: sqlite3.Row, output: dict[str, object] | None
) -> FeatureRun | None:
    if output is None or row["run_ms"] is None:
        return None
    width = _pixels(output.get("width"))
    height = _pixels(output.get("height"))
    if width is None or height is None:
        return None
    if row["feature_group"] == "ic-lora":
        duration_ms = _positive(output.get("durationMs"))
        length = None if duration_ms is None else duration_ms / 1000
        # The recipe stores the label it ran at. The output is snapped to the 64 grid.
        resolution = str(row["resolution_label"] or "") or nearest_resolution(
            width, height
        )
    else:
        length = _positive(row["edited_span_s"])
        resolution = nearest_resolution(width, height)
    if length is None:
        return None
    return FeatureRun(
        feature=str(row["feature"]),
        model=str(row["model"] or ""),
        resolution=resolution,
        aspect_ratio=find_closest_aspect_ratio_option(width, height, _ASPECT_RATIOS),
        length_s=length_bucket_s(length),
        run_ms=float(row["run_ms"]),
    )


def is_feature_candidate(row: sqlite3.Row) -> bool:
    return row["status"] == "succeeded" and row["feature_group"] in FEATURE_RENDER_GROUPS


def feature_runs(
    rows: list[sqlite3.Row], outputs: dict[str, dict[str, object]]
) -> list[FeatureRun]:
    """Finished runs that have an output video to read the size from."""
    runs: list[FeatureRun] = []
    for row in rows:
        if not is_feature_candidate(row):
            continue
        run = _feature_run(row, outputs.get(str(row["id"])))
        if run is not None:
            runs.append(run)
    return runs
