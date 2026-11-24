"""Integration tests for HuggingFace OAuth auth flow, persistence, and expiry."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from state.app_state_types import HfAuthenticated, HfNotAuthenticated, HfOAuthPending


class TestStartLogin:
    def test_radeon_separate_server_uses_registered_callback(self, test_state, monkeypatch) -> None:
        monkeypatch.setenv("LTX_REQUIRE_ROCM", "1")
        test_state.config.backend_port = 41956
        assert test_state.hf_auth.start_login().redirect_uri == (
            "http://127.0.0.1:41954/api/auth/huggingface/callback"
        )

    def test_returns_correct_fields(self, test_state) -> None:
        resp = test_state.hf_auth.start_login()
        assert resp.client_id == "test-client-id"
        assert "127.0.0.1" in resp.redirect_uri
        assert "/api/auth/huggingface/callback" in resp.redirect_uri
        assert resp.scope == "openid profile gated-repos"
        assert resp.code_challenge_method == "S256"
        assert len(resp.state) > 0
        assert len(resp.code_challenge) > 0

    def test_sets_state_to_pending(self, test_state) -> None:
        test_state.hf_auth.start_login()
        assert isinstance(test_state.state.hf_auth_state, HfOAuthPending)

    def test_second_login_replaces_pending(self, test_state) -> None:
        resp1 = test_state.hf_auth.start_login()
        resp2 = test_state.hf_auth.start_login()
        assert resp1.state != resp2.state
        assert isinstance(test_state.state.hf_auth_state, HfOAuthPending)
        assert test_state.state.hf_auth_state.state == resp2.state


class TestHandleCallback:
    def test_denied_login_is_immediately_retryable(self, test_state) -> None:
        resp = test_state.hf_auth.start_login()
        test_state.hf_auth.handle_callback(code="", state_param=resp.state, error="access_denied")
        assert test_state.hf_auth.get_auth_status().status == "not_authenticated"

    def test_rejects_error_param(self, test_state) -> None:
        html = test_state.hf_auth.handle_callback(code="", state_param="", error="access_denied")
        assert "access_denied" in html
        assert "Authentication Failed" in html

    def test_rejects_missing_code(self, test_state) -> None:
        test_state.hf_auth.start_login()
        html = test_state.hf_auth.handle_callback(code="", state_param="abc", error="")
        assert "Missing code or state" in html

    def test_rejects_wrong_state(self, test_state) -> None:
        test_state.hf_auth.start_login()
        html = test_state.hf_auth.handle_callback(code="some-code", state_param="wrong-state", error="")
        assert "Token exchange failed" in html
        assert isinstance(test_state.state.hf_auth_state, HfNotAuthenticated)

    def test_rejects_when_not_pending(self, test_state) -> None:
        # Don't call start_login — state is HfNotAuthenticated
        html = test_state.hf_auth.handle_callback(code="some-code", state_param="some-state", error="")
        assert "Token exchange failed" in html

    def test_rejects_expired_pending(self, test_state) -> None:
        resp = test_state.hf_auth.start_login()
        # Manually expire the pending state
        pending = test_state.state.hf_auth_state
        assert isinstance(pending, HfOAuthPending)
        test_state.state.hf_auth_state = HfOAuthPending(
            state=pending.state,
            code_verifier=pending.code_verifier,
            created_at=time.time() - 700,  # 700s ago, timeout is 600s
        )
        html = test_state.hf_auth.handle_callback(code="some-code", state_param=resp.state, error="")
        assert "Token exchange failed" in html
        assert isinstance(test_state.state.hf_auth_state, HfNotAuthenticated)

    def test_happy_path_exchange(self, test_state, monkeypatch) -> None:
        """Full callback flow with a faked HF token response."""
        resp = test_state.hf_auth.start_login()

        @dataclass
        class FakeTokenResponse:
            status_code: int = 200
            text: str = ""
            def json(self) -> dict[str, object]:
                return {"access_token": "hf_test_token_123", "expires_in": 3600}

        import handlers.hf_auth_handler as handler_module
        monkeypatch.setattr(handler_module.requests, "post", lambda *_args, **_kwargs: FakeTokenResponse())

        html = test_state.hf_auth.handle_callback(code="valid-code", state_param=resp.state, error="")
        assert "Authentication Successful" in html
        assert isinstance(test_state.state.hf_auth_state, HfAuthenticated)
        assert test_state.state.hf_auth_state.access_token == "hf_test_token_123"

    def test_exchange_failure_sets_not_authenticated(self, test_state, monkeypatch) -> None:
        resp = test_state.hf_auth.start_login()

        @dataclass
        class FakeErrorResponse:
            status_code: int = 401
            text: str = "unauthorized"

        import handlers.hf_auth_handler as handler_module
        monkeypatch.setattr(handler_module.requests, "post", lambda *_args, **_kwargs: FakeErrorResponse())

        html = test_state.hf_auth.handle_callback(code="bad-code", state_param=resp.state, error="")
        assert "Token exchange failed" in html
        assert isinstance(test_state.state.hf_auth_state, HfNotAuthenticated)


class TestAuthStatus:
    def test_not_authenticated_by_default(self, client) -> None:
        # Reset to not-authenticated (conftest sets authenticated for download tests)
        from state import get_state_service
        handler = get_state_service()
        handler.state.hf_auth_state = HfNotAuthenticated()

        r = client.get("/api/auth/huggingface/status")
        assert r.status_code == 200
        assert r.json()["status"] == "not_authenticated"

    def test_authenticated_status(self, client) -> None:
        # conftest already sets HfAuthenticated
        r = client.get("/api/auth/huggingface/status")
        assert r.status_code == 200
        assert r.json()["status"] == "authenticated"

    def test_expired_token_returns_not_authenticated(self, client) -> None:
        from state import get_state_service
        handler = get_state_service()
        handler.state.hf_auth_state = HfAuthenticated(
            access_token="expired-token",
            expires_at=time.time() - 1,  # expired 1 second ago
        )

        r = client.get("/api/auth/huggingface/status")
        assert r.status_code == 200
        assert r.json()["status"] == "not_authenticated"
        assert isinstance(handler.state.hf_auth_state, HfNotAuthenticated)

    def test_pending_status(self, client) -> None:
        from state import get_state_service
        handler = get_state_service()
        handler.state.hf_auth_state = HfOAuthPending(
            state="test", code_verifier="test", created_at=time.time(),
        )

        r = client.get("/api/auth/huggingface/status")
        assert r.status_code == 200
        assert r.json()["status"] == "pending"


class TestLogout:
    def test_logout_clears_state(self, client) -> None:
        r = client.post("/api/auth/huggingface/logout")
        assert r.status_code == 200
        assert r.json()["status"] == "logged_out"

        r = client.get("/api/auth/huggingface/status")
        assert r.json()["status"] == "not_authenticated"


class TestTokenPersistence:
    def _token_file(self, test_state) -> Path:
        return test_state.config.app_data_dir / "hf_auth_token.json"

    def test_authenticated_state_writes_token_file(self, test_state, monkeypatch) -> None:
        resp = test_state.hf_auth.start_login()

        @dataclass
        class FakeTokenResponse:
            status_code: int = 200
            text: str = ""
            def json(self) -> dict[str, object]:
                return {"access_token": "persist_me", "expires_in": 7200}

        import handlers.hf_auth_handler as handler_module
        monkeypatch.setattr(handler_module.requests, "post", lambda *_args, **_kwargs: FakeTokenResponse())

        test_state.hf_auth.handle_callback(code="code", state_param=resp.state, error="")

        token_file = self._token_file(test_state)
        assert token_file.exists()
        data = json.loads(token_file.read_text())
        assert data["access_token"] == "persist_me"
        assert data["expires_at"] > time.time()

    def test_logout_clears_token_file(self, test_state, monkeypatch) -> None:
        resp = test_state.hf_auth.start_login()

        @dataclass
        class FakeTokenResponse:
            status_code: int = 200
            text: str = ""
            def json(self) -> dict[str, object]:
                return {"access_token": "to_be_cleared", "expires_in": 7200}

        import handlers.hf_auth_handler as handler_module
        monkeypatch.setattr(handler_module.requests, "post", lambda *_args, **_kwargs: FakeTokenResponse())

        test_state.hf_auth.handle_callback(code="code", state_param=resp.state, error="")
        assert self._token_file(test_state).exists()

        test_state.hf_auth.logout()
        assert not self._token_file(test_state).exists()

    def test_load_token_restores_valid_token(self, test_state) -> None:
        token_file = self._token_file(test_state)
        token_file.write_text(json.dumps({
            "access_token": "restored_token",
            "expires_at": time.time() + 3600,
        }))

        # Reset state to not authenticated
        test_state.state.hf_auth_state = HfNotAuthenticated()
        test_state.hf_auth.load_token()

        assert isinstance(test_state.state.hf_auth_state, HfAuthenticated)
        assert test_state.state.hf_auth_state.access_token == "restored_token"

    def test_load_token_ignores_expired_file(self, test_state) -> None:
        token_file = self._token_file(test_state)
        token_file.write_text(json.dumps({
            "access_token": "expired_token",
            "expires_at": time.time() - 100,
        }))

        test_state.state.hf_auth_state = HfNotAuthenticated()
        test_state.hf_auth.load_token()

        assert isinstance(test_state.state.hf_auth_state, HfNotAuthenticated)
        assert not token_file.exists()  # file should be cleaned up

    def test_load_token_handles_missing_file(self, test_state) -> None:
        test_state.state.hf_auth_state = HfNotAuthenticated()
        test_state.hf_auth.load_token()  # should not raise
        assert isinstance(test_state.state.hf_auth_state, HfNotAuthenticated)

    def test_load_token_handles_corrupt_file(self, test_state) -> None:
        token_file = self._token_file(test_state)
        token_file.write_text("not valid json {{{")

        test_state.state.hf_auth_state = HfNotAuthenticated()
        test_state.hf_auth.load_token()

        assert isinstance(test_state.state.hf_auth_state, HfNotAuthenticated)
        assert not token_file.exists()  # corrupt file should be cleaned up

    def test_expired_status_check_clears_token_file(self, test_state, monkeypatch) -> None:
        """When get_auth_status detects expiry, the token file should also be cleared."""
        resp = test_state.hf_auth.start_login()

        @dataclass
        class FakeTokenResponse:
            status_code: int = 200
            text: str = ""
            def json(self) -> dict[str, object]:
                return {"access_token": "short_lived", "expires_in": 1}

        import handlers.hf_auth_handler as handler_module
        monkeypatch.setattr(handler_module.requests, "post", lambda *_args, **_kwargs: FakeTokenResponse())

        test_state.hf_auth.handle_callback(code="code", state_param=resp.state, error="")
        assert self._token_file(test_state).exists()

        # Manually expire it
        test_state.state.hf_auth_state = HfAuthenticated(
            access_token="short_lived",
            expires_at=time.time() - 1,
        )

        status = test_state.hf_auth.get_auth_status()
        assert status.status == "not_authenticated"
        assert not self._token_file(test_state).exists()
