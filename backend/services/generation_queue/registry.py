from __future__ import annotations

from collections.abc import Mapping

from services.generation_queue.types import GenerationExecutor


class ExecutorRegistry:
    def __init__(self, executors: Mapping[str, GenerationExecutor]) -> None:
        self._executors = dict(executors)

    def get(self, feature: str) -> GenerationExecutor:
        executor = self._executors.get(feature)
        if executor is None:
            raise ValueError(f"unknown generation feature: {feature}")
        return executor

    def validate_params(
        self, feature: str, spec: Mapping[str, object], *, contract_version: int
    ) -> None:
        self.get(feature).validate_params(spec, contract_version=contract_version)
