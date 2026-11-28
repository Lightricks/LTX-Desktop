"""A second lane for tests. Production registers only the local lane."""

from __future__ import annotations

from services.generation_queue.lanes import LaneSpec

TEST_LANE = "test-remote"

# A provider-like lane: no GPU, one job at a time, a short stop grace.
TEST_LANE_SPEC = LaneSpec(
    id=TEST_LANE,
    concurrency=1,
    uses_gpu=False,
    cancellable_while_running=True,
    stop_grace_seconds=5.0,
)

# The feature name tests give to jobs that run on the test lane.
TEST_FEATURE = "remote-feature"
