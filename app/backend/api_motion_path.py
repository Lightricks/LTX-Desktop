"""The path a Motion Track dot follows. Both the create body and the stored spec hold it."""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

MOTION_PATH_MAX_POINTS = 7
# A run has up to this many paths. Each is a dot in the control clip.
MOTION_PATH_MAX_PATHS = 3
MOTION_PATH_MAX_HOLD_SEC = 3
# Every coordinate, anchors and handles, is within +/-10. The editor keeps anchors inside the
# frame, and handles can extend past it to shape a curve.
MOTION_PATH_COORD_BOUND = 10
MotionPathEasing: TypeAlias = Literal[
    "linear",
    "easeIn",
    "easeOut",
    "easeInOut",
    "easeInBack",
    "easeOutBack",
    "easeInOutBack",
    "gentle",
    "quick",
    "bouncy",
    "slow",
]


class MotionPathVec2(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    x: float = Field(
        allow_inf_nan=False, ge=-MOTION_PATH_COORD_BOUND, le=MOTION_PATH_COORD_BOUND
    )
    y: float = Field(
        allow_inf_nan=False, ge=-MOTION_PATH_COORD_BOUND, le=MOTION_PATH_COORD_BOUND
    )


class MotionPathPoint(MotionPathVec2):
    """A path anchor in frame-normalized coordinates. Handles are offsets from the anchor."""

    handleIn: MotionPathVec2 | None = None
    handleOut: MotionPathVec2 | None = None


class MotionPath(BaseModel):
    """The path the Motion Track dot follows. Each segment is a cubic Bézier."""

    model_config = ConfigDict(strict=True, extra="forbid")

    points: list[MotionPathPoint] = Field(min_length=2, max_length=MOTION_PATH_MAX_POINTS)
    easing: MotionPathEasing = "easeInOut"
    # Pauses at the path ends, in seconds. Create refuses holds that fill the clip.
    startHoldSec: float = Field(default=0, allow_inf_nan=False, ge=0, le=MOTION_PATH_MAX_HOLD_SEC)
    endHoldSec: float = Field(default=0, allow_inf_nan=False, ge=0, le=MOTION_PATH_MAX_HOLD_SEC)
