"""Route handlers for GET/POST /api/generation-seed.

The seed and its lock are the app settings `locked_seed` / `seed_locked`. This is the one
surface the remote app can reach for them, since it does not expose /api/settings.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app_handler import AppHandler
from state import get_state_service
from state.app_settings import GenerationSeed, GenerationSeedUpdate, UpdateSettingsRequest

router = APIRouter(prefix="/api")
phone_generation_seed_router = APIRouter(prefix="/api")


def _current(handler: AppHandler) -> GenerationSeed:
    settings = handler.settings.get_settings_snapshot()
    return GenerationSeed(seed=settings.locked_seed, locked=settings.seed_locked)


@router.get("/generation-seed", response_model=GenerationSeed, tags=["generation-seed"])
@phone_generation_seed_router.get("/generation-seed", response_model=GenerationSeed, tags=["generation-seed"])
def route_get_generation_seed(handler: AppHandler = Depends(get_state_service)) -> GenerationSeed:
    return _current(handler)


@router.post("/generation-seed", response_model=GenerationSeed, tags=["generation-seed"])
@phone_generation_seed_router.post("/generation-seed", response_model=GenerationSeed, tags=["generation-seed"])
def route_post_generation_seed(
    body: GenerationSeedUpdate,
    handler: AppHandler = Depends(get_state_service),
) -> GenerationSeed:
    patch: dict[str, int | bool] = {}
    if body.seed is not None:
        patch["locked_seed"] = body.seed
    if body.locked is not None:
        patch["seed_locked"] = body.locked
    if patch:
        handler.settings.update_settings(UpdateSettingsRequest.model_validate(patch))
    return _current(handler)
