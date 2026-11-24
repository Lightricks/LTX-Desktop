"""Create-time and execution-time rules for a custom LoRA run.

The client names a saved custom LoRA by id and sends a strength. Create stores an empty ``ref``
and ``customLoraId``, and ``make_custom_lora_resolver`` resolves the weights path at execution
time, so a client never sends a path and SQLite never persists one. A Start Frame is optional:
with one the run is image to video, without one it is text to video.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from _routes._errors import HTTPError
from api_types import (
    CreateCustomLoraRequest,
    ImageToVideoInputs,
    ImageToVideoRecipeParams,
    LoraEntry,
    TextToVideoParams,
)
from runtime_config.custom_loras import read_custom_lora
from runtime_config.models_scanner import resolve_lora_ref
from services.features.video.custom_lora import (
    CUSTOM_LORA_FEATURE,
    CUSTOM_LORA_NOT_DOWNLOADED,
    stored_lora,
)
from services.records import UnavailableError
from services.store import Store

CUSTOM_LORA_NOT_FOUND = "CUSTOM_LORA_NOT_FOUND"
INVALID_GENERATION_SPEC = "INVALID_GENERATION_SPEC"

# Params of the create body that the stored row does not keep: the row keeps ``loras`` instead.
_CLIENT_ONLY_PARAMS = {"customLoraId", "scale"}


class CustomLoraCreateError(Exception):
    """A create-time miss for a custom LoRA run. ``status`` is the HTTP status the route answers."""

    def __init__(self, message: str, *, code: str, status: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


def custom_lora_id_of_spec(spec: Mapping[str, object]) -> str | None:
    """The saved LoRA a stored run uses, or ``None`` when the run names none or the row is junk."""
    lora = stored_lora(spec)
    return lora.customLoraId if lora is not None else None


def custom_lora_in_use(store: Store, custom_lora_id: str) -> bool:
    """Whether a queued, running or cancelling run holds this saved LoRA.

    Deleting the weights while a run is queued or running would fail the run, or fail the delete
    on Windows where the open file cannot be removed.
    """
    try:
        queue = store.list_queue_generations()
    except UnavailableError as exc:
        raise HTTPError(503, "STORE_UNAVAILABLE") from exc
    return any(
        entry.generation.feature == CUSTOM_LORA_FEATURE
        and custom_lora_id_of_spec(entry.generation.spec) == custom_lora_id
        for entry in queue
    )


def resolve_custom_lora_create(
    req: CreateCustomLoraRequest, *, models_dir: Path
) -> tuple[str, TextToVideoParams | ImageToVideoRecipeParams, ImageToVideoInputs | None]:
    """Validate a custom LoRA create and build its queued params.

    Returns ``(feature, params, inputs)``. Every miss raises :class:`CustomLoraCreateError` before
    the caller enqueues anything.
    """
    params = req.params
    record = read_custom_lora(models_dir, params.customLoraId)
    if record is None:
        raise CustomLoraCreateError(
            f"Custom LoRA '{params.customLoraId}' was not found",
            code=CUSTOM_LORA_NOT_FOUND,
            status=404,
        )
    if not record.downloaded:
        raise CustomLoraCreateError(
            f"The weights of '{record.name}' are not on this computer. Add them first.",
            code=CUSTOM_LORA_NOT_DOWNLOADED,
        )
    stored = {
        **params.model_dump(exclude=_CLIENT_ONLY_PARAMS),
        "loras": [
            LoraEntry(
                ref="",
                scale=params.scale,
                displayName=record.name,
                customLoraId=record.id,
            )
        ],
    }
    if req.inputs is not None:
        if req.inputs.endFrame is not None:
            raise CustomLoraCreateError(
                "A custom LoRA does not take an end frame", code=INVALID_GENERATION_SPEC
            )
        return (
            CUSTOM_LORA_FEATURE,
            ImageToVideoRecipeParams.model_validate(stored),
            ImageToVideoInputs(startFrame=req.inputs.startFrame),
        )

    if params.aspectRatio == "auto":
        raise CustomLoraCreateError(
            "aspectRatio 'auto' needs a Start Frame to size from", code=INVALID_GENERATION_SPEC
        )
    return CUSTOM_LORA_FEATURE, TextToVideoParams.model_validate(stored), None


def make_custom_lora_resolver(
    models_dir: Callable[[], Path],
) -> Callable[[list[LoraEntry]], list[LoraEntry]]:
    """Hydrate a custom LoRA's ``ref`` from ``customLoraId`` at execution time.

    The stored spec carries ``customLoraId`` + ``scale`` (empty ``ref``); this resolver checks
    that the weights file is still on disk and fills ``ref`` just before the job runs.
    """

    def resolve(loras: list[LoraEntry]) -> list[LoraEntry]:
        resolved: list[LoraEntry] = []
        base = models_dir()
        for entry in loras:
            if entry.customLoraId is None:
                resolved.append(entry)
                continue
            record = read_custom_lora(base, entry.customLoraId)
            name = (record.name if record is not None else entry.displayName) or "This LoRA"
            if record is None or not record.downloaded:
                raise HTTPError(
                    409,
                    f"The weights of '{name}' are not on this computer. Add them again, then retry.",
                    code=CUSTOM_LORA_NOT_DOWNLOADED,
                )
            try:
                resolve_lora_ref(base, record.ref)
            except ValueError as exc:
                raise HTTPError(
                    409,
                    f"The weights of '{name}' are not on this computer. Add them again, then retry.",
                    code=CUSTOM_LORA_NOT_DOWNLOADED,
                ) from exc
            resolved.append(entry.model_copy(update={"ref": record.ref}))
        return resolved

    return resolve
