"""Shared HuggingFace authentication utilities."""

from __future__ import annotations

import time
from threading import RLock

from _routes._errors import HTTPError
from state.app_state_types import AppState, HfAuthenticated


def require_hf_token(state: AppState, lock: RLock) -> str:
    """Return a valid HuggingFace OAuth token or raise HTTPError(403)."""
    token = optional_hf_token(state, lock)
    if token is None:
        raise HTTPError(403, "HuggingFace authentication required for gated models")
    return token


def optional_hf_token(state: AppState, lock: RLock) -> str | None:
    """Return a valid HuggingFace OAuth token if signed in, else None — never raises.

    For opt-in auth: attach the token when the user is authenticated, but allow the caller
    to proceed anonymously (e.g. public downloads) when they're not.
    """
    with lock:
        match state.hf_auth_state:
            case HfAuthenticated(access_token=token, expires_at=exp):
                if time.time() <= exp:
                    return token
            case _:
                pass
    return None
