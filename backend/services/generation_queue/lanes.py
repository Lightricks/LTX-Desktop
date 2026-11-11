"""Execution lanes: capacity pools that run queued generations side by side.

A lane is keyed by resource or provider, never by feature. Jobs in one lane run
in queue order, up to the lane's concurrency. Jobs in different lanes never wait
for each other. The lane of a job is computed once, when the job is inserted.

Invariant: every executor runs correctly on the local lane. A lane that is not
registered (an old row, a removed provider, or parallel lanes turned off) falls back to local.

The local lane and the fal lane are registered. Another provider lane adds one LaneSpec.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from services.records import LOCAL_LANE

LaneId = str

FAL_LANE: LaneId = "fal"


@dataclass(frozen=True)
class LaneSpec:
    id: LaneId
    # Jobs of this lane that run at the same time.
    concurrency: int
    # The lane holds the GPU. Only a GPU lane evicts GPU caches and samples GPU hardware.
    uses_gpu: bool
    # A running job can be stopped. False for a provider job that is billed once started.
    cancellable_while_running: bool
    # Seconds a stopping app waits for a running job. None waits as long as the caller allows.
    stop_grace_seconds: float | None = None
    # Asks the provider to drop a job that was running when the app stopped. Gets the
    # provider job id. Best effort and safe to repeat. None for a lane with no provider.
    cancel_orphan: Callable[[str], None] | None = None

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError(f"lane {self.id} needs a concurrency of at least 1")


LOCAL_LANE_SPEC = LaneSpec(
    id=LOCAL_LANE,
    concurrency=1,
    uses_gpu=True,
    cancellable_while_running=True,
)

# One fal queue request per job. A cancel frees the lane at once. The provider cancel
# callback needs the fal client, so AppHandler adds it with dataclasses.replace.
FAL_LANE_SPEC = LaneSpec(
    id=FAL_LANE,
    concurrency=1,
    uses_gpu=False,
    cancellable_while_running=True,
    stop_grace_seconds=5.0,
)


class LaneRegistry:
    def __init__(self, lanes: Iterable[LaneSpec] = (LOCAL_LANE_SPEC,)) -> None:
        by_id = {lane.id: lane for lane in lanes}
        if LOCAL_LANE not in by_id:
            raise ValueError("the local lane must be registered")
        self._lanes: Mapping[LaneId, LaneSpec] = by_id

    @property
    def lanes(self) -> tuple[LaneSpec, ...]:
        return tuple(self._lanes.values())

    @property
    def ids(self) -> tuple[LaneId, ...]:
        return tuple(self._lanes)

    @property
    def uncancellable_ids(self) -> tuple[LaneId, ...]:
        """Lanes whose running job a cancel cannot stop."""
        return tuple(
            lane.id for lane in self._lanes.values() if not lane.cancellable_while_running
        )

    def spec(self, lane_id: LaneId) -> LaneSpec | None:
        return self._lanes.get(lane_id)

    def resolve(self, lane_id: LaneId) -> LaneId:
        """The lane a job runs on: its own when registered, else local."""
        return lane_id if lane_id in self._lanes else LOCAL_LANE
