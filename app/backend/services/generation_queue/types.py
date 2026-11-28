from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from api_types import MediaKind, PipelineOffloadMode, PromptRewriteSource, VideoLoadMode
from services.generation_queue.sessions import JobSession
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


@dataclass(frozen=True)
class ExecutionOutcome:
    """What an executor reports beyond its output files.

    Modes are None when the executor's pipeline does not surface them
    (retake/extend today) — nulls mean unknown, never default them.
    """

    # A prompt the app wrote for a blank one. The runner stores it with the success, so a run
    # that fails or is cancelled keeps its blank prompt and a retry starts from it.
    written_prompt: str | None = None
    load_mode: VideoLoadMode | None = None
    offload_mode: PipelineOffloadMode | None = None
    # Who rewrote the prompt, when a rewrite ran. None means none ran or the executor
    # does not report it.
    prompt_rewrite: PromptRewriteSource | None = None


class GenerationExecutor(Protocol):
    def validate_params(
        self, spec: Mapping[str, object], *, contract_version: int
    ) -> None: ...

    def plan_outputs(self, generation: GenerationRecord) -> tuple[OutputPlan, ...]: ...

    def execute(
        self,
        generation: GenerationRecord,
        outputs: tuple[OutputAllocation, ...],
        session: JobSession,
    ) -> ExecutionOutcome | None: ...


@runtime_checkable
class LaneAwareExecutor(Protocol):
    """An executor that picks the lane its jobs run on. Others run on the local lane.

    A pure function of the stored spec. The queue computes it once, at insert.
    """

    def lane_for(self, spec: Mapping[str, object]) -> str: ...
