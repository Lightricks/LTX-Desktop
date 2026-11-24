from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Protocol, TypeVar

from services.analytics import AnalyticsService, pixel_resolution

__all__ = [
    "AnalyticsService",
    "desktop_generation_details",
    "image_execution",
    "pixel_resolution",
    "run_tracked_generation",
    "video_execution",
]
from services.services_utils import JSONValue
from state.app_settings import (
    AppSettings,
    should_image_generate_with_fal_api,
    should_video_generate_with_ltx_api,
)


class _Cancellable(Protocol):
    @property
    def status(self) -> str: ...


ResponseT = TypeVar("ResponseT", bound=_Cancellable)


def video_execution(*, force_api_generations: bool, settings: AppSettings) -> str:
    if should_video_generate_with_ltx_api(
        force_api_generations=force_api_generations,
        settings=settings,
    ):
        return "ltx_api"
    return "local"


def image_execution(*, force_api_generations: bool, settings: AppSettings) -> str:
    if should_image_generate_with_fal_api(
        force_api_generations=force_api_generations,
        settings=settings,
    ):
        return "fal_api"
    return "local"


def desktop_generation_details(
    *,
    surface: str,
    feature: str,
    kind: str,
    execution: str,
    **metadata: JSONValue,
) -> dict[str, JSONValue]:
    return {
        "generation_id": str(uuid.uuid4()),
        "client": "desktop",
        "surface": surface,
        "feature": feature,
        "kind": kind,
        "execution": execution,
        **metadata,
    }


def run_tracked_generation(
    analytics: AnalyticsService,
    details: dict[str, JSONValue],
    operation: Callable[[], ResponseT],
) -> ResponseT:
    with analytics.generation(details) as tracker:
        response = operation()
        if response.status == "cancelled":
            tracker.finish("cancelled")
        return response
