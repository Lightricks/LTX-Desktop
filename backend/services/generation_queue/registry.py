from __future__ import annotations

from collections.abc import Mapping

from services.generation_queue.lanes import LOCAL_LANE, LaneId, LaneRegistry
from services.generation_queue.types import GenerationExecutor, LaneAwareExecutor


class ExecutorRegistry:
    def __init__(
        self,
        executors: Mapping[str, GenerationExecutor],
        lanes: LaneRegistry | None = None,
    ) -> None:
        self._executors = dict(executors)
        self._lanes = lanes if lanes is not None else LaneRegistry()

    @property
    def lanes(self) -> LaneRegistry:
        return self._lanes

    def get(self, feature: str) -> GenerationExecutor:
        executor = self._executors.get(feature)
        if executor is None:
            raise ValueError(f"unknown generation feature: {feature}")
        return executor

    def validate_params(
        self, feature: str, spec: Mapping[str, object], *, contract_version: int
    ) -> None:
        self.get(feature).validate_params(spec, contract_version=contract_version)

    def lane_for(self, feature: str, spec: Mapping[str, object]) -> LaneId:
        """The lane a new job of this feature runs on. Local unless the executor says otherwise."""
        executor = self.get(feature)
        lane = executor.lane_for(spec) if isinstance(executor, LaneAwareExecutor) else LOCAL_LANE
        return self._lanes.resolve(lane)
