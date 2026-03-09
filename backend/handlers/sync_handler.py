"""Handler for Palette sync operations."""
from __future__ import annotations

import logging
from typing import Any

from services.palette_sync_client.palette_sync_client import PaletteSyncClient
from state.app_state_types import AppState

logger = logging.getLogger(__name__)

# Fallback pricing (cents) when the Palette credits endpoint doesn't return it.
# Values sourced from the live Palette /api/desktop/credits/check endpoint.
_DEFAULT_PRICING: dict[str, int] = {
    "video_t2v": 10,
    "video_i2v": 16,
    "video_seedance": 5,
    "image": 6,
    "image_edit": 20,
    "audio": 15,
    "text_enhance": 3,
}


class SyncHandler:
    def __init__(self, state: AppState, palette_sync_client: PaletteSyncClient) -> None:
        self._state = state
        self._client = palette_sync_client
        self._cached_user: dict[str, Any] | None = None

    def _try_refresh(self) -> dict[str, Any] | None:
        """Attempt to refresh an expired JWT. Returns user info or None."""
        refresh_token = self._state.app_settings.palette_refresh_token
        if not refresh_token:
            return None
        try:
            result = self._client.refresh_access_token(refresh_token=refresh_token)
            self._state.app_settings.palette_api_key = result["access_token"]
            self._state.app_settings.palette_refresh_token = result["refresh_token"]
            self._cached_user = result["user"]
            return result["user"]
        except Exception:
            return None

    def get_status(self) -> dict[str, Any]:
        api_key = self._state.app_settings.palette_api_key
        if not api_key:
            return {"connected": False, "user": None}
        if self._cached_user is not None:
            return {"connected": True, "user": self._cached_user}
        try:
            user = self._client.validate_connection(api_key=api_key)
            self._cached_user = user
            return {"connected": True, "user": user}
        except Exception as exc:
            # JWT might be expired — try refreshing
            user = self._try_refresh()
            if user is not None:
                return {"connected": True, "user": user}
            self._cached_user = None
            return {"connected": False, "user": None, "error": str(exc)}

    def connect(self, token: str) -> dict[str, Any]:
        """Store an auth token and validate it. Returns status."""
        try:
            user = self._client.validate_connection(api_key=token)
        except Exception as exc:
            return {"connected": False, "error": str(exc)}
        self._state.app_settings.palette_api_key = token
        self._cached_user = user
        return {"connected": True, "user": user}

    def login(self, email: str, password: str) -> dict[str, Any]:
        """Sign in with email/password and store the session tokens."""
        try:
            result = self._client.sign_in_with_email(email=email, password=password)
        except Exception as exc:
            return {"connected": False, "error": str(exc)}
        self._state.app_settings.palette_api_key = result["access_token"]
        self._state.app_settings.palette_refresh_token = result["refresh_token"]
        self._cached_user = result["user"]
        return {"connected": True, "user": result["user"]}

    def disconnect(self) -> dict[str, Any]:
        """Clear the stored auth token and cached user."""
        self._state.app_settings.palette_api_key = ""
        self._state.app_settings.palette_refresh_token = ""
        self._cached_user = None
        return {"connected": False, "user": None}

    def get_credits(self) -> dict[str, Any]:
        api_key = self._state.app_settings.palette_api_key
        if not api_key:
            return {"connected": False, "balance_cents": None, "pricing": None}
        try:
            credits = self._client.get_credits(api_key=api_key)
            result: dict[str, Any] = {"connected": True, **credits}
        except Exception:
            result = {"connected": True, "balance_cents": None, "pricing": None}

        # If the credits endpoint didn't return balance_cents, fall back
        # to the check endpoint which reliably includes the balance.
        if result.get("balance_cents") is None:
            try:
                check = self._client.check_credits(
                    api_key=api_key, generation_type="image", count=1,
                )
                result["balance_cents"] = check.get("balance_cents")
            except Exception:
                pass

        # Ensure pricing is present — fall back to known defaults if the
        # credits endpoint didn't provide it.
        if not result.get("pricing"):
            result["pricing"] = _DEFAULT_PRICING

        return result

    def check_credits(self, generation_type: str, count: int = 1) -> dict[str, Any]:
        api_key = self._state.app_settings.palette_api_key
        if not api_key:
            return {"connected": False, "can_afford": True}
        try:
            return {"connected": True, **self._client.check_credits(
                api_key=api_key, generation_type=generation_type, count=count,
            )}
        except Exception as exc:
            logger.warning("Credit check failed: %s", exc)
            # Fail open — don't block generation if credit check is unavailable
            return {"connected": False, "can_afford": True}

    def deduct_credits(
        self, generation_type: str, count: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        api_key = self._state.app_settings.palette_api_key
        if not api_key:
            return {"deducted": False}
        try:
            result = self._client.deduct_credits(
                api_key=api_key, generation_type=generation_type,
                count=count, metadata=metadata,
            )
            return {"deducted": True, **result}
        except Exception as exc:
            logger.warning("Credit deduction failed: %s", exc)
            return {"deducted": False, "error": str(exc)}

    def list_gallery(
        self, page: int = 1, per_page: int = 50, asset_type: str = "all",
    ) -> dict[str, Any]:
        api_key = self._state.app_settings.palette_api_key
        if not api_key:
            return {"connected": False, "items": []}
        try:
            return {
                "connected": True,
                **self._client.list_gallery(
                    api_key=api_key, page=page, per_page=per_page, asset_type=asset_type,
                ),
            }
        except Exception as exc:
            logger.warning("Palette gallery list failed: %s", exc)
            return {"connected": False, "items": [], "error": str(exc)}

    def list_characters(self) -> dict[str, Any]:
        api_key = self._state.app_settings.palette_api_key
        if not api_key:
            return {"connected": False, "characters": []}
        try:
            return {"connected": True, **self._client.list_characters(api_key=api_key)}
        except Exception as exc:
            logger.warning("Palette characters list failed: %s", exc)
            return {"connected": False, "characters": [], "error": str(exc)}

    def list_styles(self) -> dict[str, Any]:
        api_key = self._state.app_settings.palette_api_key
        if not api_key:
            return {"connected": False, "styles": []}
        try:
            return {"connected": True, **self._client.list_styles(api_key=api_key)}
        except Exception as exc:
            logger.warning("Palette styles list failed: %s", exc)
            return {"connected": False, "styles": [], "error": str(exc)}

    def list_references(self, category: str | None = None) -> dict[str, Any]:
        api_key = self._state.app_settings.palette_api_key
        if not api_key:
            return {"connected": False, "references": []}
        try:
            return {
                "connected": True,
                **self._client.list_references(api_key=api_key, category=category),
            }
        except Exception as exc:
            logger.warning("Palette references list failed: %s", exc)
            return {"connected": False, "references": [], "error": str(exc)}

    def enhance_prompt(self, prompt: str, level: str = "2x") -> dict[str, Any]:
        api_key = self._state.app_settings.palette_api_key
        if not api_key:
            return {"error": "Not connected to Palette"}
        try:
            return self._client.enhance_prompt(api_key=api_key, prompt=prompt, level=level)
        except Exception as exc:
            logger.warning("Palette prompt enhance failed: %s", exc)
            return {"error": str(exc)}
