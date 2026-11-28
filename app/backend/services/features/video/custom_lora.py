from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from api_types import LoraEntry
from services.features.video.image_to_video import ImageToVideoExecutor
from services.features.video.text_to_video import TextToVideoExecutor
from services.generation_queue.sessions import JobSession
from services.generation_queue.types import ExecutionOutcome, OutputAllocation, OutputPlan
from services.records import GenerationRecord

CUSTOM_LORA_FEATURE = "custom-lora"
CUSTOM_LORA_NOT_DOWNLOADED = "CUSTOM_LORA_NOT_DOWNLOADED"


class _StoredLoras(BaseModel):
    """The ``loras`` of a stored run. Every other param is ignored."""

    model_config = ConfigDict(extra="ignore")

    loras: list[LoraEntry] = Field(default_factory=list[LoraEntry])


def stored_lora(spec: Mapping[str, object]) -> LoraEntry | None:
    """The one LoRA a stored custom LoRA run holds, or ``None`` when the row has none or is junk."""
    params = spec.get("params")
    if not isinstance(params, dict):
        return None
    try:
        stored = _StoredLoras.model_validate(params)
    except ValidationError:
        return None
    return stored.loras[0] if stored.loras else None


class CustomLoraExecutor:
    """Queued custom LoRA. One feature id for both forms.

    A run without a Start Frame stores no inputs and runs on the text-to-video executor. A run
    with a Start Frame runs on the image-to-video executor. The row decides, so a custom LoRA
    keeps one Home screen and one result list.
    """

    def __init__(
        self,
        text_to_video: TextToVideoExecutor,
        image_to_video: ImageToVideoExecutor,
    ) -> None:
        self._text_to_video = text_to_video
        self._image_to_video = image_to_video

    def validate_params(
        self, spec: Mapping[str, object], *, contract_version: int
    ) -> None:
        self._delegate(spec).validate_params(spec, contract_version=contract_version)

    def plan_outputs(self, generation: GenerationRecord) -> tuple[OutputPlan, ...]:
        return self._delegate(generation.spec).plan_outputs(generation)

    def execute(
        self,
        generation: GenerationRecord,
        outputs: tuple[OutputAllocation, ...],
        session: JobSession,
    ) -> ExecutionOutcome | None:
        return self._delegate(generation.spec).execute(generation, outputs, session)

    def _delegate(
        self, spec: Mapping[str, object]
    ) -> TextToVideoExecutor | ImageToVideoExecutor:
        inputs = spec.get("inputs")
        has_start_frame = isinstance(inputs, Mapping) and "startFrame" in inputs
        return self._image_to_video if has_start_frame else self._text_to_video
