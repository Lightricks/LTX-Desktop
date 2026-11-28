"""Dependency wiring helpers for AppState singleton."""

from __future__ import annotations

from app_handler import AppHandler

_app_handler: AppHandler | None = None


def init_state_service(state_service: AppHandler) -> None:
    global _app_handler
    _app_handler = state_service


def get_state_service() -> AppHandler:
    assert _app_handler is not None, "AppHandler is not initialized"
    return _app_handler


def set_state_service_for_tests(state_service: AppHandler) -> None:
    init_state_service(state_service)
