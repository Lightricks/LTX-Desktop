"""Tests for standardized HTTP error responses and OpenAPI registration."""

from __future__ import annotations

from app_factory import create_app
from starlette.testclient import TestClient
from tests.http_error_assertions import assert_http_error


def test_validation_errors_use_http_error_response(client):
    response = client.post("/api/generate", json={})
    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == "HTTP_422"
    assert "Field required" in payload["message"]
    assert 'File "' not in payload["message"]


def test_unhandled_exceptions_use_http_error_response(test_state, monkeypatch):
    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(test_state.health, "get_health", _boom)

    app = create_app(handler=test_state)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/health")
        assert_http_error(response, status_code=500, code="HTTP_500", message="boom")


def test_app_openapi_registers_shared_http_error_response(test_state):
    schema = create_app(handler=test_state).openapi()

    generate_responses = schema["paths"]["/api/generate"]["post"]["responses"]
    assert "402" in generate_responses
    assert "4XX" in generate_responses
    assert "5XX" in generate_responses
    assert "422" not in generate_responses
    assert generate_responses["402"]["content"]["application/json"]["schema"]["$ref"] == "#/components/schemas/LtxInsufficientFundsErrorResponse"
    assert generate_responses["4XX"]["content"]["application/json"]["schema"]["$ref"] == "#/components/schemas/HTTPErrorResponse"

    schemas = schema["components"]["schemas"]
    assert schemas["HTTPErrorResponse"] == {
        "properties": {
            "code": {"title": "Code", "type": "string"},
            "message": {"title": "Message", "type": "string"},
        },
        "required": ["code", "message"],
        "title": "HTTPErrorResponse",
        "type": "object",
    }
    assert schemas["LtxInsufficientFundsErrorResponse"]["properties"]["code"] == {
        "const": "LTX_INSUFFICIENT_FUNDS",
        "title": "Code",
        "type": "string",
    }
