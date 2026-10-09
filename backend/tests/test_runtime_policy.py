"""Tests for /api/runtime-policy endpoint."""

from __future__ import annotations


def test_runtime_policy_true(client, test_state):
    test_state.config.local_generations_mode = "unsupported"

    response = client.get("/api/runtime-policy")
    assert response.status_code == 200
    assert response.json() == {"force_api_generations": True, "local_viable": False}


def test_runtime_policy_false(client, test_state):
    test_state.config.local_generations_mode = "full_models_loading"

    response = client.get("/api/runtime-policy")
    assert response.status_code == 200
    assert response.json() == {"force_api_generations": False, "local_viable": True}


def test_runtime_policy_stream_is_locally_viable(client, test_state):
    test_state.config.local_generations_mode = "streaming_models_loading"

    response = client.get("/api/runtime-policy")
    assert response.status_code == 200
    assert response.json() == {"force_api_generations": False, "local_viable": True}


def test_runtime_policy_dev_force_local_viable(client, test_state, monkeypatch):
    test_state.config.local_generations_mode = "unsupported"
    test_state.config.dev_mode = True
    monkeypatch.setenv("LTX_DEV_FORCE_LOCAL_VIABLE", "1")

    response = client.get("/api/runtime-policy")
    assert response.status_code == 200
    assert response.json() == {"force_api_generations": True, "local_viable": True}
