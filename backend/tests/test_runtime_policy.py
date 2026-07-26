"""Tests for /api/runtime-policy endpoint."""

from __future__ import annotations


def test_runtime_policy_true(client, test_state):
    test_state.config.local_generations_mode = "unsupported"

    response = client.get("/api/runtime-policy")
    assert response.status_code == 200
    data = response.json()
    assert data["force_api_generations"] is True
    assert data["auto_fast_video_engine"] == "cloud"
    assert data["execution_mode"] == "unsupported"
    assert data["provenance"]


def test_runtime_policy_false(client, test_state):
    test_state.config.local_generations_mode = "full_models_loading"

    response = client.get("/api/runtime-policy")
    assert response.status_code == 200
    data = response.json()
    assert data["force_api_generations"] is False
    assert data["auto_fast_video_engine"] == "torch"
    assert data["execution_mode"] == "eager"
    assert data["capability_engines"]["retake"] == "torch"
