"""Route handler for GET /api/ui-preferences. Mounted on both Desktop and Remote."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import UiPreferences
from app_handler import AppHandler
from state import get_state_service

router = APIRouter(prefix="/api", tags=["ui-preferences"])


@router.get("/ui-preferences", response_model=UiPreferences)
def route_get_ui_preferences(handler: AppHandler = Depends(get_state_service)) -> UiPreferences:
    settings = handler.settings.get_settings_snapshot()
    return UiPreferences(advancedControls=settings.advanced_controls)
