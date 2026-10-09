"""Tests for /api/feature-flags on Desktop and Remote."""

from __future__ import annotations

import json

from starlette.testclient import TestClient

from api_types import FeatureFlags, FeatureFlagsPatch
from remote.app import create_remote_app

DEFAULTS = {"customIcLora": False, "advancedIcLoraControls": False}
AUTH = {"Authorization": "Bearer pair-token"}


def test_get_returns_defaults_without_file(client, test_state):
    response = client.get("/api/feature-flags")
    assert response.status_code == 200
    assert response.json() == DEFAULTS
    assert not (test_state.config.app_data_dir / "feature_flags.json").exists()


def test_patch_merges_and_persists_to_file(client, test_state):
    first = client.request("PATCH", "/api/feature-flags", json={"customIcLora": True})
    second = client.request("PATCH", "/api/feature-flags", json={"advancedIcLoraControls": True})

    expected = {"customIcLora": True, "advancedIcLoraControls": True}
    assert first.json() == {**DEFAULTS, "customIcLora": True}
    assert second.json() == expected
    assert client.get("/api/feature-flags").json() == expected
    saved = test_state.config.app_data_dir / "feature_flags.json"
    assert json.loads(saved.read_text(encoding="utf-8")) == expected


def test_patch_rejects_unknown_flag(client):
    response = client.request("PATCH", "/api/feature-flags", json={"nope": True})
    assert response.status_code == 422


def test_patch_null_leaves_flag_unchanged(client):
    client.request("PATCH", "/api/feature-flags", json={"customIcLora": True})
    response = client.request("PATCH", "/api/feature-flags", json={"customIcLora": None})
    assert response.status_code == 200
    assert response.json() == {**DEFAULTS, "customIcLora": True}


def test_unreadable_file_falls_back_to_defaults(client, test_state):
    # A directory at the file path raises OSError (not FileNotFoundError) on read.
    (test_state.config.app_data_dir / "feature_flags.json").mkdir()
    assert client.get("/api/feature-flags").json() == DEFAULTS


def test_corrupt_file_falls_back_to_defaults(client, test_state):
    (test_state.config.app_data_dir / "feature_flags.json").write_text("{not json", encoding="utf-8")
    assert client.get("/api/feature-flags").json() == DEFAULTS


def test_remote_serves_flags_with_bearer_only(test_state):
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as remote:
        unauth = remote.get("/api/feature-flags")
        patched = remote.request("PATCH", "/api/feature-flags", headers=AUTH, json={"customIcLora": True})
        read_back = remote.get("/api/feature-flags", headers=AUTH)
    assert unauth.status_code == 401
    assert patched.status_code == 200
    assert read_back.json() == {**DEFAULTS, "customIcLora": True}


def test_patch_model_covers_every_flag():
    assert set(FeatureFlagsPatch.model_fields) == set(FeatureFlags.model_fields)
