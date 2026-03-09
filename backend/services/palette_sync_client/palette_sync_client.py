"""Protocol for communicating with Director's Palette cloud API."""

from __future__ import annotations

from typing import Any, Protocol


class PaletteSyncClient(Protocol):
    def validate_connection(self, *, api_key: str) -> dict[str, Any]:
        """Validate API key and return user info. Raises on failure."""
        ...

    def sign_in_with_email(self, *, email: str, password: str) -> dict[str, Any]:
        """Sign in with email/password. Returns access_token, refresh_token, user info."""
        ...

    def refresh_access_token(self, *, refresh_token: str) -> dict[str, Any]:
        """Refresh an expired access token. Returns new access_token, refresh_token."""
        ...

    def get_credits(self, *, api_key: str) -> dict[str, Any]:
        """Return credit balance and pricing for the authenticated user."""
        ...

    def check_credits(
        self, *, api_key: str, generation_type: str, count: int,
    ) -> dict[str, Any]:
        """Check whether the user can afford a generation. Does not deduct."""
        ...

    def deduct_credits(
        self, *, api_key: str, generation_type: str, count: int,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Deduct credits after a successful generation."""
        ...

    def list_gallery(
        self, *, api_key: str, page: int, per_page: int, asset_type: str,
    ) -> dict[str, Any]:
        """List cloud gallery items with pagination."""
        ...

    def list_characters(self, *, api_key: str) -> dict[str, Any]:
        """List characters from all user storyboards."""
        ...

    def list_styles(self, *, api_key: str) -> dict[str, Any]:
        """List user style guides and brands."""
        ...

    def list_references(
        self, *, api_key: str, category: str | None,
    ) -> dict[str, Any]:
        """List reference images with optional category filter."""
        ...

    def enhance_prompt(
        self, *, api_key: str, prompt: str, level: str,
    ) -> dict[str, Any]:
        """Enhance a prompt using Palette's prompt expander."""
        ...
