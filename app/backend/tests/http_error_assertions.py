"""Helpers for asserting standardized HTTP error responses."""

from __future__ import annotations


def assert_http_error(
    response,
    *,
    status_code: int,
    code: str,
    message: str | None = None,
) -> dict[str, str]:
    assert response.status_code == status_code
    payload = response.json()
    assert payload == {
        "code": code,
        "message": message if message is not None else code,
    }
    return payload
