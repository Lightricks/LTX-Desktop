"""Admin token guard for privileged settings mutations."""

from __future__ import annotations

import hmac

from fastapi import Request

from _routes._errors import HTTPError


def guard_admin_permission(request: Request) -> None:
    """Raise 403 if the request lacks a valid admin token."""
    admin_token: str = getattr(request.app.state, "admin_token", "")
    provided = request.headers.get("X-Admin-Token", "")
    if not admin_token or not provided or not hmac.compare_digest(provided, admin_token):
        raise HTTPError(403, "Admin token required")
