"""Integration tests for HuggingFace OAuth auth flow, persistence, and expiry."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from state.app_state_types import HfAuthenticated, HfNotAuthenticated, HfOAuthPending
from tests.fakes.services import FakeResponse


class TestStartLogin:
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

    def test_redirect_uses_public_base_url(self, test_state) -> None:
        test_state.config.public_base_url = "http://127.0.0.1:8123"
        resp = test_state.hf_auth.start_login()
        assert resp.redirect_uri == "http://127.0.0.1:8123/api/auth/huggingface/callback"


class TestHandleCallback:
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

    def test_happy_path_exchange(self, test_state, fake_services) -> None:
        """Full callback flow with a faked HF token response."""
        resp = test_state.hf_auth.start_login()

        fake_services.http.queue(
            "post",
            FakeResponse(json_payload={"access_token": "hf_test_token_123", "expires_in": 3600}),
        )

        html = test_state.hf_auth.handle_callback(code="valid-code", state_param=resp.state, error="")
        assert "Authentication Successful" in html
        assert isinstance(test_state.state.hf_auth_state, HfAuthenticated)
        assert test_state.state.hf_auth_state.access_token == "hf_test_token_123"

    def test_exchange_failure_sets_not_authenticated(self, test_state, fake_services) -> None:
        resp = test_state.hf_auth.start_login()

        fake_services.http.queue("post", FakeResponse(status_code=401, text="unauthorized"))

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

    def test_authenticated_state_writes_token_file(self, test_state, fake_services) -> None:
        resp = test_state.hf_auth.start_login()

        fake_services.http.queue(
            "post",
            FakeResponse(json_payload={"access_token": "persist_me", "expires_in": 7200}),
        )

        test_state.hf_auth.handle_callback(code="code", state_param=resp.state, error="")

        token_file = self._token_file(test_state)
        assert token_file.exists()
        data = json.loads(token_file.read_text())
        assert data["access_token"] == "persist_me"
        assert data["expires_at"] > time.time()

    def test_logout_clears_token_file(self, test_state, fake_services) -> None:
        resp = test_state.hf_auth.start_login()

        fake_services.http.queue(
            "post",
            FakeResponse(json_payload={"access_token": "to_be_cleared", "expires_in": 7200}),
        )

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

    def test_expired_status_check_clears_token_file(self, test_state, fake_services) -> None:
        """When get_auth_status detects expiry, the token file should also be cleared."""
        resp = test_state.hf_auth.start_login()

        fake_services.http.queue(
            "post",
            FakeResponse(json_payload={"access_token": "short_lived", "expires_in": 1}),
        )

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
