"""Typed HTTP error utilities for route functions."""

from __future__ import annotations

import re

from api_types import HTTPErrorResponse

_MACHINE_CODE_PATTERN = re.compile(r"^[A-Z0-9]+(?:_[A-Z0-9]+)*$")
_FALLBACK_MESSAGE = "An unexpected error occurred"


def _normalize_message(detail: object) -> str:
    message = str(detail).strip()
    return message or _FALLBACK_MESSAGE


def _default_code(status_code: int, message: str) -> str:
    if _MACHINE_CODE_PATTERN.fullmatch(message):
        return message
    return f"HTTP_{status_code}"


def build_http_error_response(
    status_code: int,
    detail: object,
    *,
    code: str | None = None,
) -> HTTPErrorResponse:
    message = _normalize_message(detail)
    return HTTPErrorResponse(
        code=code or _default_code(status_code, message),
        message=message,
    )


class HTTPError(Exception):
    """Raised by route functions to signal an HTTP error response."""

    def __init__(self, status_code: int, detail: str, code: str | None = None) -> None:
        self.status_code = status_code
        self.response = build_http_error_response(status_code, detail, code=code)
        self.detail = self.response.message
        self.code = self.response.code
        super().__init__(self.detail)
