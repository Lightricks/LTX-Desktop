"""HTTP implementation of PaletteSyncClient."""

from __future__ import annotations

import os
from typing import Any, cast

from services.http_client.http_client import HTTPClient

_DEFAULT_BASE = "https://directorspal.com"
_SUPABASE_URL = os.environ.get(
    "PALETTE_SUPABASE_URL",
    "https://tarohelkwuurakbxjyxm.supabase.co",
)
_SUPABASE_ANON_KEY = os.environ.get(
    "PALETTE_SUPABASE_ANON_KEY",
    # Supabase anon keys are designed for public/client-side use.
    # This default is safe to ship but can be overridden via env var.
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRhcm9oZWxrd3V1cmFrYnhqeXhtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTU4OTQzMDYsImV4cCI6MjA3MTQ3MDMwNn0."
    "uTeDJ0YYVGjP2FTL9oIpCRPeqXBbxNnh8y7CH0gIABs",
)


class PaletteSyncClientImpl:
    def __init__(self, http: HTTPClient, base_url: str = _DEFAULT_BASE) -> None:
        self._http = http
        self._base_url = base_url

    def _headers(self, api_key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {api_key}"}

    def _parse_supabase_user(self, user_data: dict[str, Any]) -> dict[str, Any]:
        """Normalize a Supabase user response into a simple user dict."""
        metadata: dict[str, Any] = user_data.get("user_metadata", {})
        return {
            "id": user_data.get("id"),
            "email": user_data.get("email"),
            "name": metadata.get("full_name") or metadata.get("name") or user_data.get("email"),
        }

    def validate_connection(self, *, api_key: str) -> dict[str, Any]:
        if api_key.startswith("dp_"):
            # dp_ API keys must be validated by the Palette app, which has
            # the api_keys table and hashing logic.
            resp = self._http.get(
                f"{self._base_url}/api/desktop/me",
                headers=self._headers(api_key),
                timeout=10,
            )
            if resp.status_code == 404:
                raise RuntimeError(
                    "dp_ API keys are not supported yet. "
                    "The Directors Palette team needs to deploy the /api/desktop endpoints. "
                    "Use 'Login with Email' instead."
                )
            if resp.status_code != 200:
                raise RuntimeError(f"Palette auth failed: {resp.status_code}")
            return cast(dict[str, Any], resp.json())

        # JWT token — validate directly with Supabase
        resp = self._http.get(
            f"{_SUPABASE_URL}/auth/v1/user",
            headers={
                "Authorization": f"Bearer {api_key}",
                "apikey": _SUPABASE_ANON_KEY,
            },
            timeout=10,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Authentication failed (HTTP {resp.status_code}). Check your token.")
        user_data = cast(dict[str, Any], resp.json())
        return self._parse_supabase_user(user_data)

    def sign_in_with_email(self, *, email: str, password: str) -> dict[str, Any]:
        """Sign in with email/password via Supabase auth."""
        resp = self._http.post(
            f"{_SUPABASE_URL}/auth/v1/token?grant_type=password",
            headers={
                "apikey": _SUPABASE_ANON_KEY,
                "Content-Type": "application/json",
            },
            json_payload={"email": email, "password": password},
            timeout=15,
        )
        if resp.status_code == 400:
            data = cast(dict[str, Any], resp.json())
            msg = data.get("error_description") or data.get("msg") or "Invalid email or password"
            raise RuntimeError(msg)
        if resp.status_code != 200:
            raise RuntimeError(f"Sign-in failed (HTTP {resp.status_code})")
        data = cast(dict[str, Any], resp.json())
        user_data = cast(dict[str, Any], data.get("user", {}))
        return {
            "access_token": data["access_token"],
            "refresh_token": data["refresh_token"],
            "user": self._parse_supabase_user(user_data),
        }

    def refresh_access_token(self, *, refresh_token: str) -> dict[str, Any]:
        """Refresh an expired Supabase JWT."""
        resp = self._http.post(
            f"{_SUPABASE_URL}/auth/v1/token?grant_type=refresh_token",
            headers={
                "apikey": _SUPABASE_ANON_KEY,
                "Content-Type": "application/json",
            },
            json_payload={"refresh_token": refresh_token},
            timeout=10,
        )
        if resp.status_code != 200:
            raise RuntimeError("Session expired. Please log in again.")
        data = cast(dict[str, Any], resp.json())
        user_data = cast(dict[str, Any], data.get("user", {}))
        return {
            "access_token": data["access_token"],
            "refresh_token": data["refresh_token"],
            "user": self._parse_supabase_user(user_data),
        }

    def get_credits(self, *, api_key: str) -> dict[str, Any]:
        try:
            resp = self._http.get(
                f"{self._base_url}/api/desktop/credits",
                headers=self._headers(api_key),
                timeout=10,
            )
            if resp.status_code != 200:
                return {"balance_cents": None}
            return cast(dict[str, Any], resp.json())
        except Exception:
            return {"balance_cents": None}

    def check_credits(
        self, *, api_key: str, generation_type: str, count: int,
    ) -> dict[str, Any]:
        resp = self._http.post(
            f"{self._base_url}/api/desktop/credits/check",
            headers={**self._headers(api_key), "Content-Type": "application/json"},
            json_payload={"generation_type": generation_type, "count": count},
            timeout=10,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Credit check failed: {resp.status_code}")
        return cast(dict[str, Any], resp.json())

    def deduct_credits(
        self, *, api_key: str, generation_type: str, count: int,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"generation_type": generation_type, "count": count}
        if metadata:
            payload["metadata"] = metadata
        resp = self._http.post(
            f"{self._base_url}/api/desktop/credits/deduct",
            headers={**self._headers(api_key), "Content-Type": "application/json"},
            json_payload=payload,
            timeout=10,
        )
        if resp.status_code == 402:
            data = cast(dict[str, Any], resp.json())
            raise RuntimeError(f"Insufficient credits: balance={data.get('balance_cents')}")
        if resp.status_code != 200:
            raise RuntimeError(f"Credit deduction failed: {resp.status_code}")
        return cast(dict[str, Any], resp.json())

    def list_gallery(
        self, *, api_key: str, page: int, per_page: int, asset_type: str,
    ) -> dict[str, Any]:
        params = f"?page={page}&per_page={per_page}&type={asset_type}"
        resp = self._http.get(
            f"{self._base_url}/api/desktop/gallery{params}",
            headers=self._headers(api_key),
            timeout=15,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Palette gallery failed: {resp.status_code}")
        return cast(dict[str, Any], resp.json())

    def list_characters(self, *, api_key: str) -> dict[str, Any]:
        resp = self._http.get(
            f"{self._base_url}/api/desktop/library/characters",
            headers=self._headers(api_key),
            timeout=10,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Palette characters failed: {resp.status_code}")
        return cast(dict[str, Any], resp.json())

    def list_styles(self, *, api_key: str) -> dict[str, Any]:
        resp = self._http.get(
            f"{self._base_url}/api/desktop/library/styles",
            headers=self._headers(api_key),
            timeout=10,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Palette styles failed: {resp.status_code}")
        return cast(dict[str, Any], resp.json())

    def list_references(
        self, *, api_key: str, category: str | None,
    ) -> dict[str, Any]:
        params = f"?category={category}" if category else ""
        resp = self._http.get(
            f"{self._base_url}/api/desktop/library/references{params}",
            headers=self._headers(api_key),
            timeout=10,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Palette references failed: {resp.status_code}")
        return cast(dict[str, Any], resp.json())

    def enhance_prompt(
        self, *, api_key: str, prompt: str, level: str,
    ) -> dict[str, Any]:
        resp = self._http.post(
            f"{self._base_url}/api/desktop/prompt/enhance",
            headers=self._headers(api_key),
            json_payload={"prompt": prompt, "level": level},
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Palette prompt enhance failed: {resp.status_code}")
        return cast(dict[str, Any], resp.json())
