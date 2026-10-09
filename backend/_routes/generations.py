from __future__ import annotations

from fastapi import APIRouter, Depends, Path, Query

from api_types import (
    CreateAudioToVideoRequest,
    CreateIcLoraRecipeRequest,
    CreateExtendRequest,
    CreateImageToVideoRequest,
    CreateLoraRecipeRequest,
    CreateRetakeRequest,
    CreateTextToVideoRequest,
    Generation,
    QueueSnapshot,
    ReorderGenerationQueueRequest,
    StatusResponse,
)

# Recipe ids we actually ship — kept in sync with shared/lora-recipes.json. Fail
# closed on shape here so a malformed id 422s at the route rather than reaching
# create_lora_recipe.
_RECIPE_ID_PATTERN = r"^[a-z0-9-]+$"
from app_handler import AppHandler
from services.features.queued_features import QUEUED_GENERATION_FEATURES
from state import get_state_service

router = APIRouter(prefix="/api")


@router.get(
    "/generation-queue",
    response_model=QueueSnapshot,
    tags=["generations"],
)
def route_get_generation_queue(
    handler: AppHandler = Depends(get_state_service),
) -> QueueSnapshot:
    return handler.queued_generations.get_queue_snapshot()


@router.post(
    "/generation-queue/reorder",
    response_model=QueueSnapshot,
    tags=["generations"],
)
def route_reorder_generation_queue(
    req: ReorderGenerationQueueRequest,
    handler: AppHandler = Depends(get_state_service),
) -> QueueSnapshot:
    return handler.queued_generations.reorder_queue(
        req.generation_id, req.before_generation_id
    )


@router.post(
    "/generation-queue/done/clear",
    response_model=QueueSnapshot,
    tags=["generations"],
)
def route_clear_generation_queue_done(
    handler: AppHandler = Depends(get_state_service),
) -> QueueSnapshot:
    return handler.queued_generations.clear_done()


@router.post(
    "/generation-queue/failed/clear",
    response_model=QueueSnapshot,
    tags=["generations"],
)
def route_clear_generation_queue_failed(
    handler: AppHandler = Depends(get_state_service),
) -> QueueSnapshot:
    return handler.queued_generations.clear_failed()


@router.post(
    "/generation-queue/done/{generation_id}/seen",
    response_model=QueueSnapshot,
    tags=["generations"],
)
def route_mark_generation_queue_done_seen(
    generation_id: str = Path(min_length=1),
    handler: AppHandler = Depends(get_state_service),
) -> QueueSnapshot:
    return handler.queued_generations.mark_done_seen(generation_id)


@router.post(
    "/generation-queue/done/{generation_id}/dismiss",
    response_model=QueueSnapshot,
    tags=["generations"],
)
def route_dismiss_generation_queue_done(
    generation_id: str = Path(min_length=1),
    handler: AppHandler = Depends(get_state_service),
) -> QueueSnapshot:
    return handler.queued_generations.dismiss_done(generation_id)


@router.post(
    "/generations/text-to-video",
    response_model=Generation,
    tags=["generations"],
)
def route_create_text_to_video(
    req: CreateTextToVideoRequest, handler: AppHandler = Depends(get_state_service)
) -> Generation:
    return handler.queued_generations.create_text_to_video(
        req, analytics_client="desktop"
    )


@router.post(
    "/generations/image-to-video",
    response_model=Generation,
    tags=["generations"],
)
def route_create_image_to_video(
    req: CreateImageToVideoRequest, handler: AppHandler = Depends(get_state_service)
) -> Generation:
    return handler.queued_generations.create_image_to_video(
        req, analytics_client="desktop"
    )


@router.post(
    # Bounded under /recipes/ so the `{recipe_id}` wildcard does not swallow the
    # whole `POST /generations/*` namespace (a future static create route would
    # otherwise never match, and a typo'd id would 422 instead of 404).
    "/generations/recipes/{recipe_id}",
    response_model=Generation,
    tags=["generations"],
)
def route_create_lora_recipe(
    req: CreateLoraRecipeRequest,
    recipe_id: str = Path(pattern=_RECIPE_ID_PATTERN, max_length=64),
    handler: AppHandler = Depends(get_state_service),
) -> Generation:
    return handler.queued_generations.create_lora_recipe(
        recipe_id, req, analytics_client="desktop"
    )


@router.post(
    "/generations/audio-to-video",
    response_model=Generation,
    tags=["generations"],
)
def route_create_audio_to_video(
    req: CreateAudioToVideoRequest, handler: AppHandler = Depends(get_state_service)
) -> Generation:
    return handler.queued_generations.create_audio_to_video(
        req, analytics_client="desktop"
    )


@router.post(
    "/generations/retake",
    response_model=Generation,
    tags=["generations"],
)
def route_create_retake(
    req: CreateRetakeRequest, handler: AppHandler = Depends(get_state_service)
) -> Generation:
    return handler.queued_generations.create_retake(
        req, analytics_client="desktop"
    )


@router.post(
    "/generations/ic-lora-recipes/{recipe_id}",
    response_model=Generation,
    tags=["generations"],
)
def route_create_ic_lora_recipe(
    req: CreateIcLoraRecipeRequest,
    recipe_id: str = Path(pattern=_RECIPE_ID_PATTERN),
    handler: AppHandler = Depends(get_state_service),
) -> Generation:
    return handler.queued_generations.create_ic_lora_recipe(
        recipe_id, req, analytics_client="desktop"
    )


@router.post(
    "/generations/extend",
    response_model=Generation,
    tags=["generations"],
)
def route_create_extend(
    req: CreateExtendRequest, handler: AppHandler = Depends(get_state_service)
) -> Generation:
    return handler.queued_generations.create_extend(
        req, analytics_client="desktop"
    )


@router.get("/generations", response_model=list[Generation], tags=["generations"])
def route_list_generations(
    feature: str, handler: AppHandler = Depends(get_state_service)
) -> list[Generation]:
    return handler.queued_generations.list_generations(feature)


@router.get(
    "/generations/recent-features",
    response_model=list[str],
    tags=["generations"],
)
def route_list_recent_features(
    limit: int = Query(default=4, ge=1, le=20),
    handler: AppHandler = Depends(get_state_service),
) -> list[str]:
    """Distinct Explore features, newest generation first. Same list as remote."""
    return handler.queued_generations.list_recent_features(
        limit=limit,
        allowed=QUEUED_GENERATION_FEATURES,
    )


@router.get(
    "/generations/{generation_id}",
    response_model=Generation,
    tags=["generations"],
)
def route_get_generation(
    generation_id: str, handler: AppHandler = Depends(get_state_service)
) -> Generation:
    return handler.queued_generations.get_generation(generation_id)


@router.post(
    "/generations/{generation_id}/retry",
    response_model=Generation,
    tags=["generations"],
)
def route_retry_generation(
    generation_id: str, handler: AppHandler = Depends(get_state_service)
) -> Generation:
    return handler.queued_generations.retry_generation(generation_id)


@router.post(
    "/generations/{generation_id}/cancel",
    response_model=Generation,
    tags=["generations"],
)
def route_cancel_generation(
    generation_id: str, handler: AppHandler = Depends(get_state_service)
) -> Generation:
    return handler.queued_generations.cancel_generation(generation_id)


@router.delete(
    "/generations/{generation_id}",
    response_model=StatusResponse,
    tags=["generations"],
)
def route_delete_generation(
    generation_id: str, handler: AppHandler = Depends(get_state_service)
) -> StatusResponse:
    handler.queued_generations.delete_generation(generation_id)
    return StatusResponse(status="ok")
