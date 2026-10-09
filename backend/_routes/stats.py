from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Query

from app_handler import AppHandler
from handlers.dashboard_handler import DashboardHandler
from services.dashboard_stats import DashboardSnapshot
from state import get_state_service
from state.app_settings import DashboardSelection, UpdateSettingsRequest

router = APIRouter(prefix="/api")
phone_stats_router = APIRouter(prefix="/api")


def _dashboard(handler: AppHandler) -> DashboardHandler:
    return handler.dashboard


@router.get("/stats/dashboard", response_model=DashboardSnapshot, tags=["stats"])
@phone_stats_router.get("/stats/dashboard", response_model=DashboardSnapshot, tags=["stats"])
def route_dashboard(
    tz: str = Query(min_length=1),
    window: Literal["7d", "30d", "all"] = Query(alias="range"),
    handler: AppHandler = Depends(get_state_service),
) -> DashboardSnapshot:
    return _dashboard(handler).get(window=window, tz=tz)


@router.get("/stats/activity-dashboard-selections", response_model=DashboardSelection, tags=["stats"])
@phone_stats_router.get("/stats/activity-dashboard-selections", response_model=DashboardSelection, tags=["stats"])
def route_get_dashboard_selection(
    handler: AppHandler = Depends(get_state_service),
) -> DashboardSelection:
    return handler.settings.get_settings_snapshot().activity_dashboard_selections


@router.post("/stats/activity-dashboard-selections", response_model=DashboardSelection, tags=["stats"])
@phone_stats_router.post("/stats/activity-dashboard-selections", response_model=DashboardSelection, tags=["stats"])
def route_post_dashboard_selection(
    body: DashboardSelection,
    handler: AppHandler = Depends(get_state_service),
) -> DashboardSelection:
    handler.settings.update_settings(
        UpdateSettingsRequest.model_validate({"activity_dashboard_selections": body.model_dump()})
    )
    return handler.settings.get_settings_snapshot().activity_dashboard_selections
