"""Route handlers for /api/feature-flags. Mounted on both Desktop and Remote."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import FeatureFlags, FeatureFlagsPatch
from app_handler import AppHandler
from state import get_state_service

router = APIRouter(prefix="/api", tags=["feature-flags"])


@router.get("/feature-flags", response_model=FeatureFlags)
def route_get_feature_flags(handler: AppHandler = Depends(get_state_service)) -> FeatureFlags:
    return handler.feature_flags.get_flags()


@router.patch("/feature-flags", response_model=FeatureFlags)
def route_update_feature_flags(
    req: FeatureFlagsPatch,
    handler: AppHandler = Depends(get_state_service),
) -> FeatureFlags:
    return handler.feature_flags.update_flags(req)
