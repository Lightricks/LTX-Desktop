"""HuggingFace OAuth handler — PKCE flow, state management, token exchange."""

from __future__ import annotations

import base64
import hashlib
import hmac
import html
import logging
import secrets
import time
from threading import RLock
from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError

from api_types import HuggingFaceAuthStatusResponse, HuggingFaceLoginResponse, HuggingFaceLogoutResponse
from handlers.base import StateHandlerBase, with_state_lock
from services.interfaces import HTTPClient
from server_utils.secure_files import harden_file_permissions, secure_write_text
from state.app_state_types import (
    AppState,
    HfAuthenticated,
    HfAuthState,
    HfNotAuthenticated,
    HfOAuthPending,
)

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)

_HF_TOKEN_URL = "https://huggingface.co/oauth/token"
_OAUTH_PENDING_TIMEOUT_SECONDS = 600  # 10 minutes

_SUCCESS_HTML = (
    "<html><body style='font-family:system-ui,sans-serif;text-align:center;padding:60px;background:#18181b;color:#e4e4e7'>"
    "<h2>Authentication Successful</h2>"
    "<p style='color:#a1a1aa'>You can close this tab and return to LTX Desktop.</p>"
    "<script>window.close()</script>"
    "</body></html>"
)

_ERROR_HTML_TEMPLATE = (
    "<html><body style='font-family:system-ui,sans-serif;text-align:center;padding:60px;background:#18181b;color:#e4e4e7'>"
    "<h2 style='color:#f87171'>Authentication Failed</h2>"
    "<p style='color:#a1a1aa'>{msg}</p>"
    "<p style='color:#71717a'>Please close this tab and try again in LTX Desktop.</p>"
    "</body></html>"
)


class _PersistedHfToken(BaseModel):
    """Schema for hf_auth_token.json."""
    access_token: str
    expires_at: float


class _TokenExchangePayload(BaseModel):
    access_token: str
    expires_in: int = 28800


class HuggingFaceAuthHandler(StateHandlerBase):
    def __init__(self, state: AppState, lock: RLock, config: RuntimeConfig, http: HTTPClient) -> None:
        super().__init__(state, lock, config)
        self._http = http
        self._token_file = config.app_data_dir / "hf_auth_token.json"

    # ============================================================
    # Auth state mutation — centralised to keep persistence in sync
    # ============================================================

    def _set_hf_auth_state(self, new_state: HfAuthState) -> None:
        """Update in-memory auth state and persist to disk.

        Must be called while holding ``self._lock``.
        Pending state is transient (not persisted).
        """
        self.state.hf_auth_state = new_state
        match new_state:
            case HfAuthenticated(access_token=token, expires_at=exp):
                self._save_token_file(_PersistedHfToken(access_token=token, expires_at=exp))
            case HfOAuthPending():
                pass  # transient — don't touch the file
            case _:
                self._clear_token_file()

    def _save_token_file(self, data: _PersistedHfToken) -> None:
        try:
            secure_write_text(self._token_file, data.model_dump_json())
        except Exception:
            logger.error("Failed to write HF auth token file")

    def _clear_token_file(self) -> None:
        try:
            if self._token_file.exists():
                self._token_file.unlink()
        except Exception:
            logger.error("Failed to clear HF auth token file")

    # ============================================================
    # Persistence — load on startup
    # ============================================================

    def load_token(self) -> None:
        """Load a persisted HF token from disk if it exists and hasn't expired."""
        try:
            if not self._token_file.exists():
                return
            persisted = _PersistedHfToken.model_validate_json(self._token_file.read_text(encoding="utf-8"))
            harden_file_permissions(self._token_file)
            if persisted.expires_at <= time.time():
                self._clear_token_file()
                return
            with self._lock:
                self.state.hf_auth_state = HfAuthenticated(
                    access_token=persisted.access_token,
                    expires_at=persisted.expires_at,
                )
        except Exception:
            logger.error("Failed to load HF auth token file")
            self._clear_token_file()

    # ============================================================
    # Public API
    # ============================================================

    def _redirect_uri(self) -> str:
        return f"{self.config.effective_public_base_url}/api/auth/huggingface/callback"

    @with_state_lock
    def start_login(self) -> HuggingFaceLoginResponse:
        """Generate PKCE + state, store pending, return OAuth parameters."""
        state_token = secrets.token_hex(32)
        code_verifier = secrets.token_urlsafe(64)
        code_challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest())
            .rstrip(b"=")
            .decode()
        )

        self._set_hf_auth_state(HfOAuthPending(
            state=state_token,
            code_verifier=code_verifier,
            created_at=time.time(),
        ))

        return HuggingFaceLoginResponse(
            client_id=self.config.hf_oauth_client_id,
            redirect_uri=self._redirect_uri(),
            scope="openid profile gated-repos",
            state=state_token,
            code_challenge=code_challenge,
            code_challenge_method="S256",
        )

    def handle_callback(self, code: str, state_param: str, error: str) -> str:
        """Handle the OAuth callback. Returns an HTML string for the browser."""
        if error:
            return _ERROR_HTML_TEMPLATE.format(msg=html.escape(error))

        if not code or not state_param:
            return _ERROR_HTML_TEMPLATE.format(msg="Missing code or state parameter")

        success = self._exchange_code(code, state_param)
        if success:
            return _SUCCESS_HTML
        return _ERROR_HTML_TEMPLATE.format(msg="Token exchange failed. Please try again.")

    def _exchange_code(self, code: str, state_param: str) -> bool:
        """Validate state, exchange code for token, store in AppState."""
        redirect_uri = self._redirect_uri()

        with self._lock:
            pending = self.state.hf_auth_state
            if not isinstance(pending, HfOAuthPending):
                return False
            if not hmac.compare_digest(pending.state, state_param):
                self._set_hf_auth_state(HfNotAuthenticated())
                return False
            if time.time() - pending.created_at > _OAUTH_PENDING_TIMEOUT_SECONDS:
                self._set_hf_auth_state(HfNotAuthenticated())
                return False
            code_verifier = pending.code_verifier

        # Token exchange — outside lock (network I/O).
        try:
            resp = self._http.post(
                _HF_TOKEN_URL,
                data={
                    "client_id": self.config.hf_oauth_client_id,
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code",
                    "code_verifier": code_verifier,
                },
                timeout=30,
            )
        except Exception:
            logger.error("HuggingFace token exchange request failed")
            with self._lock:
                self._set_hf_auth_state(HfNotAuthenticated())
            return False

        if resp.status_code != 200:
            logger.error("HuggingFace token exchange returned %s: %s", resp.status_code, resp.text)
            with self._lock:
                self._set_hf_auth_state(HfNotAuthenticated())
            return False

        try:
            data = _TokenExchangePayload.model_validate(resp.json())
        except ValidationError:
            logger.error("HuggingFace token exchange returned an invalid payload")
            with self._lock:
                self._set_hf_auth_state(HfNotAuthenticated())
            return False

        with self._lock:
            self._set_hf_auth_state(HfAuthenticated(
                access_token=data.access_token,
                expires_at=time.time() + data.expires_in,
            ))
        return True

    @with_state_lock
    def get_auth_status(self) -> HuggingFaceAuthStatusResponse:
        match self.state.hf_auth_state:
            case HfAuthenticated(expires_at=exp):
                if time.time() > exp:
                    self._set_hf_auth_state(HfNotAuthenticated())
                    return HuggingFaceAuthStatusResponse(status="not_authenticated")
                return HuggingFaceAuthStatusResponse(status="authenticated")
            case HfOAuthPending():
                return HuggingFaceAuthStatusResponse(status="pending")
            case _:
                return HuggingFaceAuthStatusResponse(status="not_authenticated")

    @with_state_lock
    def logout(self) -> HuggingFaceLogoutResponse:
        self._set_hf_auth_state(HfNotAuthenticated())
        return HuggingFaceLogoutResponse(status="logged_out")
