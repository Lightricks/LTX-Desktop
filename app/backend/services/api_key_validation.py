"""Cheap authenticated probes so Settings only stores Fal and Gemini keys the provider accepts.

LTX has no key check that avoids a side effect, so a non-empty LTX key is stored as-is.
A rejected LTX key fails on the first real request and the generation dialog opens Settings.
"""

from __future__ import annotations

from typing import NoReturn

from _routes._errors import HTTPError
from services.gemini_text_client import GEMINI_API_BASE_URL
from services.http_client.http_client import HTTPClient, HttpTransportError

# Auth on this route is optional, but a bad Authorization header is rejected (401/403)
# rather than ignored. limit=1 keeps the probe from downloading the catalog.
# Do not point this at /v1/keys or /v1/models/usage — those need an admin key.
FAL_KEY_PROBE_URL = "https://api.fal.ai/v1/models?limit=1"

_AUTH_REJECTED = frozenset({401, 403})
_INVALID_KEY_MESSAGE = "This key isn’t valid."
_UNVERIFIED_MESSAGE = "Couldn’t verify this key. Try again in a moment."


def _raise_transport(provider: str, exc: HttpTransportError) -> NoReturn:
    raise HTTPError(504, f"Couldn’t reach {provider}. Try again in a moment.") from exc


def _raise_unverified(*, code: str) -> NoReturn:
    # Distinct from *_INVALID_API_KEY: the probe did not finish, so the key was not rejected.
    raise HTTPError(502, _UNVERIFIED_MESSAGE, code=code)


def validate_gemini_api_key(http: HTTPClient, *, api_key: str) -> None:
    """One-page models.list — Gemini returns 400/401 for junk keys."""
    try:
        response = http.get(
            f"{GEMINI_API_BASE_URL}/models?pageSize=1",
            headers={"x-goog-api-key": api_key},
            timeout=15,
        )
    except HttpTransportError as exc:
        _raise_transport("Gemini", exc)
    if response.status_code == 200:
        return
    # Google uses 400 for API_KEY_INVALID, and 401/403 for rejected credentials.
    if response.status_code in _AUTH_REJECTED or response.status_code == 400:
        raise HTTPError(400, _INVALID_KEY_MESSAGE, code="GEMINI_INVALID_API_KEY")
    _raise_unverified(code="GEMINI_API_KEY_UNVERIFIED")


def validate_fal_api_key(http: HTTPClient, *, api_key: str) -> None:
    """GET /v1/models — 401/403 means the key was rejected. 200 means it was accepted."""
    try:
        response = http.get(
            FAL_KEY_PROBE_URL,
            headers={"Authorization": f"Key {api_key}"},
            timeout=15,
        )
    except HttpTransportError as exc:
        _raise_transport("FAL", exc)
    if response.status_code == 200:
        return
    if response.status_code in _AUTH_REJECTED:
        raise HTTPError(400, _INVALID_KEY_MESSAGE, code="FAL_INVALID_API_KEY")
    _raise_unverified(code="FAL_API_KEY_UNVERIFIED")
