"""Tests for shared-secret authentication middleware."""

from __future__ import annotations

import base64

from starlette.testclient import TestClient

from app_factory import create_app


def test_request_without_token_returns_401(test_state):
    app = create_app(handler=test_state, auth_token="test-secret")
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 401
        assert response.json() == {"error": "Unauthorized"}


def test_request_with_correct_bearer_token(test_state):
    app = create_app(handler=test_state, auth_token="test-secret")
    with TestClient(app) as client:
        response = client.get("/health", headers={"Authorization": "Bearer test-secret"})
        assert response.status_code == 200


def test_request_with_correct_basic_auth(test_state):
    app = create_app(handler=test_state, auth_token="test-secret")
    credentials = base64.b64encode(b":test-secret").decode()
    with TestClient(app) as client:
        response = client.get("/health", headers={"Authorization": f"Basic {credentials}"})
        assert response.status_code == 200


def test_request_with_wrong_token_returns_401(test_state):
    app = create_app(handler=test_state, auth_token="test-secret")
    with TestClient(app) as client:
        response = client.get("/health", headers={"Authorization": "Bearer wrong-token"})
        assert response.status_code == 401


def test_health_without_token_returns_401(test_state):
    """Health endpoint is NOT exempt from auth."""
    app = create_app(handler=test_state, auth_token="test-secret")
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 401


def test_no_auth_token_disables_middleware(test_state):
    """When auth_token is empty string, auth is disabled (dev/test mode)."""
    app = create_app(handler=test_state, auth_token="")
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200


def test_websocket_with_token_query_param(test_state):
    app = create_app(handler=test_state, auth_token="test-secret")
    with TestClient(app) as client:
        # WebSocket upgrade without token should fail with 401
        response = client.get(
            "/ws/download/test",
            headers={"upgrade": "websocket", "connection": "upgrade"},
        )
        assert response.status_code == 401

        # WebSocket upgrade with correct token query param
        response = client.get(
            "/ws/download/test?token=test-secret",
            headers={"upgrade": "websocket", "connection": "upgrade"},
        )
        # The route may not exist, but auth should pass (not 401)
        assert response.status_code != 401
