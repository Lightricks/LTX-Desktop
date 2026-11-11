"""Remote-only generation HTTP surface.

Desktop generation CRUD stays unfiltered. This module is the remote contract:
only Home/Remote text-to-image, text-to-video, image-to-video, audio-to-video, retake,
extend, custom LoRA, and IC-LoRA recipes, plus get/retry/cancel/delete when the generation's feature is in that
allowlist. Recently used uses the same feature set on both apps, without the image jobs.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Path, Query

from _routes._errors import HTTPError
from api_types import (
    CreateAudioToVideoRequest,
    CreateCustomLoraRequest,
    IcLoraRecipeCreate,
    CreateExtendRequest,
    CreateImageToVideoRequest,
    CreateKeyframesRequest,
    CreateLoraRecipeRequest,
    CreateRetakeRequest,
    CreateTextToImageRequest,
    CreateTextToVideoRequest,
    EditPreflightRequest,
    EditPreflightResponse,
    IcLoraRecipeMemoryRequest,
    IcLoraRecipeMemoryResponse,
    IcLoraTilePreflightRequest,
    IcLoraTilePreflightResponse,
    Generation,
    QueueSnapshot,
    ReorderGenerationQueueRequest,
    StatusResponse,
)
from app_handler import AppHandler
from remote.deps import RemoteContext, get_remote_context
from remote.dto import RemoteGeneration, RemoteQueueSnapshot, to_remote_queue_snapshot
from services.features.image.feature_id import FEATURE as TEXT_TO_IMAGE_FEATURE
from services.features.queued_features import QUEUED_GENERATION_FEATURES
from state import get_state_service

# The jobs Remote can read and control. Recently used asks for QUEUED_GENERATION_FEATURES alone,
# as on Desktop: an image job is Remote-visible but has no Home screen to reopen.
REMOTE_GENERATION_FEATURES = QUEUED_GENERATION_FEATURES | {TEXT_TO_IMAGE_FEATURE}

router = APIRouter(prefix="/api")


def _require_remote_feature(feature: str) -> None:
    if feature not in REMOTE_GENERATION_FEATURES:
        raise HTTPError(404, "GENERATION_NOT_FOUND")


def _require_remote_generation(handler: AppHandler, generation_id: str) -> Generation:
    generation = handler.queued_generations.get_generation(generation_id)
    _require_remote_feature(generation.feature)
    return generation


def _project_queue(snapshot: QueueSnapshot, remote: RemoteContext) -> RemoteQueueSnapshot:
    return to_remote_queue_snapshot(
        snapshot,
        signer=remote.signer,
        device_id=remote.device.id,
        feature_is_allowed=REMOTE_GENERATION_FEATURES.__contains__,
    )


@router.get(
    "/generation-queue",
    response_model=RemoteQueueSnapshot,
    tags=["generations"],
)
def route_get_generation_queue(
    handler: AppHandler = Depends(get_state_service),
    remote: RemoteContext = Depends(get_remote_context),
) -> RemoteQueueSnapshot:
    return to_remote_queue_snapshot(
        handler.queued_generations.get_queue_snapshot(),
        signer=remote.signer,
        device_id=remote.device.id,
        feature_is_allowed=REMOTE_GENERATION_FEATURES.__contains__,
    )


@router.post(
    "/generation-queue/reorder",
    response_model=RemoteQueueSnapshot,
    tags=["generations"],
)
def route_reorder_generation_queue(
    req: ReorderGenerationQueueRequest,
    handler: AppHandler = Depends(get_state_service),
    remote: RemoteContext = Depends(get_remote_context),
) -> RemoteQueueSnapshot:
    _require_remote_generation(handler, req.generation_id)
    if req.before_generation_id is not None:
        _require_remote_generation(handler, req.before_generation_id)
    return _project_queue(
        handler.queued_generations.reorder_queue(
            req.generation_id, req.before_generation_id
        ),
        remote,
    )


@router.post(
    "/generation-queue/done/clear",
    response_model=RemoteQueueSnapshot,
    tags=["generations"],
)
def route_clear_generation_queue_done(
    handler: AppHandler = Depends(get_state_service),
    remote: RemoteContext = Depends(get_remote_context),
) -> RemoteQueueSnapshot:
    return _project_queue(
        handler.queued_generations.clear_done(
            feature_is_allowed=REMOTE_GENERATION_FEATURES.__contains__,
        ),
        remote,
    )


@router.post(
    "/generation-queue/failed/clear",
    response_model=RemoteQueueSnapshot,
    tags=["generations"],
)
def route_clear_generation_queue_failed(
    handler: AppHandler = Depends(get_state_service),
    remote: RemoteContext = Depends(get_remote_context),
) -> RemoteQueueSnapshot:
    return _project_queue(
        handler.queued_generations.clear_failed(
            feature_is_allowed=REMOTE_GENERATION_FEATURES.__contains__,
        ),
        remote,
    )


@router.post(
    "/generation-queue/done/{generation_id}/seen",
    response_model=RemoteQueueSnapshot,
    tags=["generations"],
)
def route_mark_generation_queue_done_seen(
    generation_id: str = Path(min_length=1),
    handler: AppHandler = Depends(get_state_service),
    remote: RemoteContext = Depends(get_remote_context),
) -> RemoteQueueSnapshot:
    _require_remote_generation(handler, generation_id)
    return _project_queue(handler.queued_generations.mark_done_seen(generation_id), remote)


@router.post(
    "/generation-queue/done/{generation_id}/dismiss",
    response_model=RemoteQueueSnapshot,
    tags=["generations"],
)
def route_dismiss_generation_queue_done(
    generation_id: str = Path(min_length=1),
    handler: AppHandler = Depends(get_state_service),
    remote: RemoteContext = Depends(get_remote_context),
) -> RemoteQueueSnapshot:
    _require_remote_generation(handler, generation_id)
    return _project_queue(handler.queued_generations.dismiss_done(generation_id), remote)


@router.post(
    "/generations/text-to-video",
    response_model=RemoteGeneration,
    tags=["generations"],
)
def route_create_text_to_video(
    req: CreateTextToVideoRequest,
    handler: AppHandler = Depends(get_state_service),
    remote: RemoteContext = Depends(get_remote_context),
) -> RemoteGeneration:
    return remote.generation(
        handler.queued_generations.create_text_to_video(
            req, analytics_client="remote"
        )
    )


@router.post(
    "/generations/text-to-image",
    response_model=RemoteGeneration,
    tags=["generations"],
)
def route_create_text_to_image(
    req: CreateTextToImageRequest,
    handler: AppHandler = Depends(get_state_service),
    remote: RemoteContext = Depends(get_remote_context),
) -> RemoteGeneration:
    return remote.generation(
        handler.queued_generations.create_text_to_image(
            req, analytics_client="remote"
        )
    )


@router.post(
    "/generations/image-to-video",
    response_model=RemoteGeneration,
    tags=["generations"],
)
def route_create_image_to_video(
    req: CreateImageToVideoRequest,
    handler: AppHandler = Depends(get_state_service),
    remote: RemoteContext = Depends(get_remote_context),
) -> RemoteGeneration:
    return remote.generation(
        handler.queued_generations.create_image_to_video(
            req, analytics_client="remote"
        )
    )


@router.post(
    "/generations/keyframes",
    response_model=RemoteGeneration,
    tags=["generations"],
)
def route_create_keyframes(
    req: CreateKeyframesRequest,
    handler: AppHandler = Depends(get_state_service),
    remote: RemoteContext = Depends(get_remote_context),
) -> RemoteGeneration:
    return remote.generation(
        handler.queued_generations.create_keyframes(
            req, analytics_client="remote"
        )
    )


@router.post(
    "/generations/recipes/{recipe_id}",
    response_model=RemoteGeneration,
    tags=["generations"],
)
def route_create_lora_recipe(
    req: CreateLoraRecipeRequest,
    recipe_id: str = Path(pattern=r"^[a-z0-9-]+$", max_length=64),
    handler: AppHandler = Depends(get_state_service),
    remote: RemoteContext = Depends(get_remote_context),
) -> RemoteGeneration:
    # create_lora_recipe already rejects unknown/unsupported recipe ids with a
    # typed 422 before enqueue; the resolved LoRA ref is stripped by the DTO.
    return remote.generation(
        handler.queued_generations.create_lora_recipe(
            recipe_id, req, analytics_client="remote"
        )
    )


@router.post(
    "/generations/custom-lora",
    response_model=RemoteGeneration,
    tags=["generations"],
)
def route_create_custom_lora(
    req: CreateCustomLoraRequest,
    handler: AppHandler = Depends(get_state_service),
    remote: RemoteContext = Depends(get_remote_context),
) -> RemoteGeneration:
    # The saved LoRA is named by id. The weights path is stripped by the DTO.
    return remote.generation(
        handler.queued_generations.create_custom_lora(req, analytics_client="remote")
    )


@router.post(
    "/generations/audio-to-video",
    response_model=RemoteGeneration,
    tags=["generations"],
)
def route_create_audio_to_video(
    req: CreateAudioToVideoRequest,
    handler: AppHandler = Depends(get_state_service),
    remote: RemoteContext = Depends(get_remote_context),
) -> RemoteGeneration:
    return remote.generation(
        handler.queued_generations.create_audio_to_video(
            req, analytics_client="remote"
        )
    )


@router.post(
    "/generations/retake",
    response_model=RemoteGeneration,
    tags=["generations"],
)
def route_create_retake(
    req: CreateRetakeRequest,
    handler: AppHandler = Depends(get_state_service),
    remote: RemoteContext = Depends(get_remote_context),
) -> RemoteGeneration:
    return remote.generation(
        handler.queued_generations.create_retake(req, analytics_client="remote")
    )


@router.post(
    "/generations/extend",
    response_model=RemoteGeneration,
    tags=["generations"],
)
def route_create_extend(
    req: CreateExtendRequest,
    handler: AppHandler = Depends(get_state_service),
    remote: RemoteContext = Depends(get_remote_context),
) -> RemoteGeneration:
    return remote.generation(
        handler.queued_generations.create_extend(req, analytics_client="remote")
    )


@router.post(
    "/generations/edit/preflight",
    response_model=EditPreflightResponse,
    tags=["generations"],
)
def route_preflight_edit(
    req: EditPreflightRequest, handler: AppHandler = Depends(get_state_service)
) -> EditPreflightResponse:
    # Path-free DTO (dims + verdicts only): no remote projection needed.
    return handler.queued_generations.preflight_edit(req)


@router.post(
    "/generations/ic-lora-recipes/{recipe_id}/memory",
    response_model=IcLoraRecipeMemoryResponse,
    tags=["generations"],
)
def route_ic_lora_recipe_memory(
    req: IcLoraRecipeMemoryRequest,
    recipe_id: str = Path(pattern=r"^[a-z0-9-]+$", max_length=64),
    handler: AppHandler = Depends(get_state_service),
    _remote: RemoteContext = Depends(get_remote_context),
) -> IcLoraRecipeMemoryResponse:
    # Path-free DTO (sizes and verdicts only): no remote projection needed.
    return handler.ic_lora.recipe_memory(recipe_id, req)


@router.post(
    "/generations/ic-lora-recipes/{recipe_id}",
    response_model=RemoteGeneration,
    tags=["generations"],
)
def route_create_ic_lora_recipe(
    req: IcLoraRecipeCreate,
    recipe_id: str = Path(pattern=r"^[a-z0-9-]+$", max_length=64),
    handler: AppHandler = Depends(get_state_service),
    remote: RemoteContext = Depends(get_remote_context),
) -> RemoteGeneration:
    return remote.generation(
        handler.queued_generations.create_ic_lora_recipe(
            recipe_id, req, analytics_client="remote"
        )
    )


@router.post(
    "/generations/ic-lora-recipes/{recipe_id}/preflight",
    response_model=IcLoraTilePreflightResponse,
    tags=["generations"],
)
def route_preflight_ic_lora_tile_sizes(
    req: IcLoraTilePreflightRequest,
    recipe_id: str = Path(pattern=r"^[a-z0-9-]+$", max_length=64),
    handler: AppHandler = Depends(get_state_service),
) -> IcLoraTilePreflightResponse:
    # Path-free DTO (verdicts only): no remote projection needed.
    return handler.queued_generations.preflight_ic_lora_tile_sizes(recipe_id, req)


@router.get(
    "/generations/recent-features",
    response_model=list[str],
    tags=["generations"],
)
def route_list_recent_features(
    limit: int = Query(default=4, ge=1, le=20),
    handler: AppHandler = Depends(get_state_service),
    _remote: RemoteContext = Depends(get_remote_context),
) -> list[str]:
    return handler.queued_generations.list_recent_features(
        limit=limit,
        allowed=QUEUED_GENERATION_FEATURES,
    )


@router.get(
    "/generations", response_model=list[RemoteGeneration], tags=["generations"]
)
def route_list_generations(
    feature: str,
    handler: AppHandler = Depends(get_state_service),
    remote: RemoteContext = Depends(get_remote_context),
) -> list[RemoteGeneration]:
    _require_remote_feature(feature)
    return [
        remote.generation(generation)
        for generation in handler.queued_generations.list_generations(feature)
    ]


@router.get(
    "/generations/{generation_id}",
    response_model=RemoteGeneration,
    tags=["generations"],
)
def route_get_generation(
    generation_id: str,
    handler: AppHandler = Depends(get_state_service),
    remote: RemoteContext = Depends(get_remote_context),
) -> RemoteGeneration:
    return remote.generation(_require_remote_generation(handler, generation_id))


@router.post(
    "/generations/{generation_id}/retry",
    response_model=RemoteGeneration,
    tags=["generations"],
)
def route_retry_generation(
    generation_id: str,
    handler: AppHandler = Depends(get_state_service),
    remote: RemoteContext = Depends(get_remote_context),
) -> RemoteGeneration:
    _require_remote_generation(handler, generation_id)
    return remote.generation(handler.queued_generations.retry_generation(generation_id))


@router.post(
    "/generations/{generation_id}/cancel",
    response_model=RemoteGeneration,
    tags=["generations"],
)
def route_cancel_generation(
    generation_id: str,
    handler: AppHandler = Depends(get_state_service),
    remote: RemoteContext = Depends(get_remote_context),
) -> RemoteGeneration:
    _require_remote_generation(handler, generation_id)
    return remote.generation(handler.queued_generations.cancel_generation(generation_id))


@router.delete(
    "/generations/{generation_id}",
    response_model=StatusResponse,
    tags=["generations"],
)
def route_delete_generation(
    generation_id: str, handler: AppHandler = Depends(get_state_service)
) -> StatusResponse:
    _require_remote_generation(handler, generation_id)
    handler.queued_generations.delete_generation(generation_id)
    return StatusResponse(status="ok")
