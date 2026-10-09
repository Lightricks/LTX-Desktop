from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from api_types import (
    AudioToVideoParams,
    OfferingId,
    GenerateVideoCancelledResponse,
    GenerateVideoRequest,
    GenerateVideoResponse,
    LoraEntry,
    LTXLocalModelId,
    TextToVideoParams,
)
from _routes._errors import HTTPError
from runtime_config.offerings import OFFERING_IDS, local_pipeline_for_offering
from services.features.video.job_common import (
    model_from_spec,
    resolve_job_local_model_id,
)
from services.generation_interrupt import GenerationCancelledError, is_cancel_exception
from services.generation_queue.types import OutputAllocation, OutputPlan
from services.records import CapabilityFailedError, GenerationRecord

FEATURE = "text-to-video"

VIDEO_OUTPUT_PLAN = OutputPlan(
    slot="output",
    media_kind="video",
    mime_type="video/mp4",
    name="output.mp4",
)
_SUPPORTED_CONTRACT_VERSION = 1


class ReservedVideoGenerator(Protocol):
    @property
    def models_dir(self) -> Path: ...

    def validate_local_a2v_request(
        self, params: AudioToVideoParams, audio_duration_ms: int
    ) -> None: ...

    def generate_local_reserved(
        self,
        req: GenerateVideoRequest,
        *,
        generation_id: str,
        output_path: Path | None = None,
        prompt_wrap: Callable[[str], str] | None = None,
        a2v_num_frames: int | None = None,
        a2v_audio_duration_seconds: float | None = None,
        local_model_id: LTXLocalModelId,
    ) -> GenerateVideoResponse: ...


class TextToVideoExecutor:
    """Queued text-to-video. v1 drains the local GPU only.

    Always calls ``generate_local_reserved`` (``use_api_specs=False``). It does
    not follow ``POST /api/generate``'s LTX API setting. Cancel and progress
    for these jobs use the ledger URLs, not ``/api/generate/cancel`` or
    ``/api/generation/progress``.

    ``prompt_wrap`` is the recipe-scoped post-enhancement prompt transform. It is
    ``None`` for the plain text-to-video feature; a LoRA recipe registers the same
    executor with its scaffold wrap so the style scaffold stays out of
    ``generate_local_reserved`` and applies only when ``feature`` is that recipe
    (and Explore auto-enhance is on, or the user already ran Enhance).
    """

    def __init__(
        self,
        video_generation: ReservedVideoGenerator,
        prompt_wrap: Callable[[str], str] | None = None,
        lora_resolver: Callable[[list[LoraEntry]], list[LoraEntry]] | None = None,
    ) -> None:
        self._video_generation = video_generation
        self._prompt_wrap = prompt_wrap
        # Recipe-scoped: fills the LoRA ref from its catalog id at execution. None for plain t2v.
        self._lora_resolver = lora_resolver

    def validate_params(
        self, spec: Mapping[str, object], *, contract_version: int
    ) -> None:
        require_supported_contract_version(contract_version)
        params_from_spec(spec)

    def plan_outputs(self, generation: GenerationRecord) -> tuple[OutputPlan, ...]:
        self._params(generation)
        return (VIDEO_OUTPUT_PLAN,)

    def execute(
        self,
        generation: GenerationRecord,
        outputs: tuple[OutputAllocation, ...],
    ) -> None:
        params = self._params(generation)
        output = require_video_output(outputs)
        try:
            req, local_model_id = generate_request_from_offering_params(
                params, self._video_generation
            )
            if self._lora_resolver is not None:
                # Inside the try so a missing/removed adapter becomes a capability failure.
                req = req.model_copy(update={"loras": self._lora_resolver(req.loras)})
            result = self._video_generation.generate_local_reserved(
                req,
                generation_id=generation.id,
                output_path=Path(output.dest_path),
                prompt_wrap=self._prompt_wrap,
                local_model_id=local_model_id,
            )
        except Exception as exc:
            if is_cancel_exception(exc):
                raise GenerationCancelledError() from exc
            if isinstance(exc, HTTPError):
                raise CapabilityFailedError(exc.detail, code=exc.code) from exc
            raise
        if isinstance(result, GenerateVideoCancelledResponse):
            raise GenerationCancelledError()

    def _params(self, generation: GenerationRecord) -> TextToVideoParams:
        require_supported_contract_version(generation.contract_version)
        return params_from_spec(generation.spec)


def require_supported_contract_version(contract_version: int) -> None:
    if contract_version != _SUPPORTED_CONTRACT_VERSION:
        raise ValueError(
            "text-to-video does not support contract_version "
            f"{contract_version}"
        )


def params_from_spec(spec: Mapping[str, object]) -> TextToVideoParams:
    return model_from_spec(spec, "params", TextToVideoParams)


def require_video_output(
    outputs: tuple[OutputAllocation, ...],
) -> OutputAllocation:
    if len(outputs) != 1 or outputs[0].plan.slot != "output":
        raise ValueError("text-to-video requires a single 'output' allocation")
    return outputs[0]


def offering_payload_from_params(
    params: BaseModel,
) -> tuple[dict[str, object], OfferingId]:
    payload = params.model_dump(mode="python")
    raw_model = payload["model"]
    if not isinstance(raw_model, str) or raw_model not in OFFERING_IDS:
        raise HTTPError(422, "INVALID_VIDEO_GENERATION_SPEC")
    payload["model"] = local_pipeline_for_offering(raw_model)
    return payload, raw_model


def generate_request_from_offering_params(
    params: BaseModel, video_generation: ReservedVideoGenerator
) -> tuple[GenerateVideoRequest, LTXLocalModelId]:
    payload, offering = offering_payload_from_params(params)
    return (
        GenerateVideoRequest.model_validate(payload),
        resolve_job_local_model_id(video_generation.models_dir, offering),
    )
