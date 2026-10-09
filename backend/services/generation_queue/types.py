from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from api_types import MediaKind
from services.records import GenerationRecord


@dataclass(frozen=True)
class OutputPlan:
    slot: str
    media_kind: MediaKind
    mime_type: str
    name: str
    # An optional output may be missing when the executor returns. It is then left out.
    optional: bool = False


@dataclass(frozen=True)
class OutputAllocation:
    plan: OutputPlan
    asset_id: str
    dest_path: str


class GenerationExecutor(Protocol):
    def validate_params(
        self, spec: Mapping[str, object], *, contract_version: int
    ) -> None: ...

    def plan_outputs(self, generation: GenerationRecord) -> tuple[OutputPlan, ...]: ...

    def execute(
        self,
        generation: GenerationRecord,
        outputs: tuple[OutputAllocation, ...],
    ) -> None: ...
