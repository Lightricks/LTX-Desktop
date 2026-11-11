from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, assert_never

from _routes._errors import HTTPError
from api_types import (
    ImageGenerationStatusResponse,
    ImageProviderBlocker,
    TextToImageParams,
    PromptRewriteSource,
    TextToImageStoredParams,
)
from runtime_config.video_job_budget import (
    LOCAL_GENERATION_UNSUPPORTED,
    LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
)
from services.features.image.grid import floor_to_image_grid
from services.features.video.job_common import capability_failed_from_http, model_from_spec
from services.generation_interrupt import GenerationCancelledError, is_cancel_exception
from services.generation_queue.lanes import FAL_LANE
from services.generation_queue.sessions import JobSession
from services.generation_queue.types import (
    ExecutionOutcome,
    OutputAllocation,
    OutputPlan,
)
from services.records import LOCAL_LANE, GenerationRecord

IMAGE_OUTPUT_PLAN = OutputPlan(
    slot="output",
    media_kind="image",
    mime_type="image/png",
    name="output.png",
)
_SUPPORTED_CONTRACT_VERSION = 1

FAL_API_KEY_NOT_CONFIGURED: Final = "FAL_API_KEY_NOT_CONFIGURED"
IMAGE_MODEL_NOT_DOWNLOADED: Final = "IMAGE_MODEL_NOT_DOWNLOADED"


@dataclass(frozen=True)
class QueuedImageResult:
    # The prompt the provider expanded, to replace the typed one on the finished row.
    written_prompt: str | None = None
    prompt_rewrite: PromptRewriteSource | None = None


class QueuedImageRenderer(Protocol):
    def generate_queued(
        self,
        params: TextToImageStoredParams,
        *,
        generation_id: str,
        output_path: Path,
        session: JobSession,
    ) -> QueuedImageResult: ...


def image_generation_status(
    *,
    force_api: bool,
    default_to_fal: bool,
    has_fal_key: bool,
    local_model_downloaded: Callable[[], bool],
) -> ImageGenerationStatusResponse:
    local_blocker: ImageProviderBlocker | None = None
    if force_api:
        local_blocker = LOCAL_GENERATION_UNSUPPORTED
    elif not local_model_downloaded():
        local_blocker = IMAGE_MODEL_NOT_DOWNLOADED
    return ImageGenerationStatusResponse(
        defaultProvider="fal" if default_to_fal else "local",
        localBlocker=local_blocker,
        falBlocker=None if has_fal_key else FAL_API_KEY_NOT_CONFIGURED,
    )


def _refusal(blocker: ImageProviderBlocker) -> HTTPError:
    """One shape for every create refusal: status, a readable message, and the blocker code.

    409 stays for a missing model. The other two are request errors.
    """
    if blocker == IMAGE_MODEL_NOT_DOWNLOADED:
        return HTTPError(409, "Download the Z-Image Turbo model.", code=blocker)
    if blocker == LOCAL_GENERATION_UNSUPPORTED:
        return HTTPError(422, LOCAL_GENERATION_UNSUPPORTED_MESSAGE, code=blocker)
    if blocker == FAL_API_KEY_NOT_CONFIGURED:
        return HTTPError(422, "Add a fal API key.", code=blocker)
    assert_never(blocker)


def resolve_text_to_image_params(
    params: TextToImageParams, status: ImageGenerationStatusResponse
) -> TextToImageStoredParams:
    """Pick the provider and refuse a job that can never run, before it is queued.

    An omitted provider follows the settings. The stored provider is always explicit, so a
    retry after the settings change keeps the provider the user chose.
    """
    provider = params.provider or status.defaultProvider
    blocker = status.localBlocker if provider == "local" else status.falBlocker
    if blocker is not None:
        raise _refusal(blocker)
    return TextToImageStoredParams(
        **params.model_dump(exclude={"provider", "width", "height"}),
        width=floor_to_image_grid(params.width),
        height=floor_to_image_grid(params.height),
        provider=provider,
    )


class TextToImageExecutor:
    """Queued text-to-image on Z-Image Turbo, local or fal. No image input and no edit."""

    def __init__(self, renderer: QueuedImageRenderer) -> None:
        self._renderer = renderer

    def validate_params(
        self, spec: Mapping[str, object], *, contract_version: int
    ) -> None:
        require_supported_contract_version(contract_version)
        params_from_spec(spec)

    def lane_for(self, spec: Mapping[str, object]) -> str:
        """A fal job waits on the provider, not on the GPU, so it has its own lane."""
        return FAL_LANE if params_from_spec(spec).provider == "fal" else LOCAL_LANE

    def plan_outputs(self, generation: GenerationRecord) -> tuple[OutputPlan, ...]:
        self._params(generation)
        return (IMAGE_OUTPUT_PLAN,)

    def execute(
        self,
        generation: GenerationRecord,
        outputs: tuple[OutputAllocation, ...],
        session: JobSession,
    ) -> ExecutionOutcome:
        params = self._params(generation)
        if len(outputs) != 1 or outputs[0].plan.slot != "output":
            raise ValueError("text-to-image requires a single 'output' allocation")
        try:
            result = self._renderer.generate_queued(
                params,
                generation_id=generation.id,
                output_path=Path(outputs[0].dest_path),
                session=session,
            )
        except Exception as exc:
            if is_cancel_exception(exc):
                raise GenerationCancelledError() from exc
            if isinstance(exc, HTTPError):
                raise capability_failed_from_http(exc) from exc
            raise
        return ExecutionOutcome(
            written_prompt=result.written_prompt, prompt_rewrite=result.prompt_rewrite
        )

    def _params(self, generation: GenerationRecord) -> TextToImageStoredParams:
        require_supported_contract_version(generation.contract_version)
        return params_from_spec(generation.spec)


def require_supported_contract_version(contract_version: int) -> None:
    if contract_version != _SUPPORTED_CONTRACT_VERSION:
        raise ValueError(
            f"text-to-image does not support contract_version {contract_version}"
        )


def params_from_spec(spec: Mapping[str, object]) -> TextToImageStoredParams:
    return model_from_spec(spec, "params", TextToImageStoredParams)
