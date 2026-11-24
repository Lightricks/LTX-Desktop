"""Centralized logging policy for request and background exception paths."""

from __future__ import annotations

import logging

from fastapi import Request

from _routes._errors import HTTPError

logger = logging.getLogger(__name__)


def log_http_error(request: Request, exc: HTTPError) -> None:
    """Log typed HTTP errors with policy-based traceback behavior."""
    if 500 <= exc.status_code <= 599:
        logger.error(
            "HTTP error on %s %s: [%s] %s",
            request.method,
            request.url.path,
            exc.status_code,
            exc.detail,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return

    logger.warning(
        "HTTP error on %s %s: [%s] %s",
        request.method,
        request.url.path,
        exc.status_code,
        exc.detail,
    )


def log_unhandled_exception(request: Request, exc: Exception) -> None:
    """Log unhandled request exceptions with full traceback."""
    logger.error(
        "Unhandled error on %s %s",
        request.method,
        request.url.path,
        exc_info=(type(exc), exc, exc.__traceback__),
    )


def log_background_exception(task_name: str, exc: Exception) -> None:
    """Log unhandled background task exceptions with full traceback."""
    logger.error(
        "Unhandled background error in task '%s'",
        task_name,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
