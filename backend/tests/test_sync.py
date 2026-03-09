"""Tests for Palette sync routes."""
from __future__ import annotations


class TestSyncStatus:
    def test_disconnected_by_default(self, client):
        resp = client.get("/api/sync/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["connected"] is False
        assert data["user"] is None

    def test_connected_after_setting_api_key(self, client):
        client.post("/api/settings", json={"paletteApiKey": "dp_valid_key"})
        resp = client.get("/api/sync/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["connected"] is True
        assert data["user"]["email"] == "test@example.com"

    def test_connection_fails_with_invalid_key(self, client, fake_services):
        fake_services.palette_sync_client.raise_on_validate = RuntimeError("Invalid API key")
        client.post("/api/settings", json={"paletteApiKey": "dp_bad_key"})
        resp = client.get("/api/sync/status")
        data = resp.json()
        assert data["connected"] is False


class TestSyncCredits:
    def test_credits_when_connected(self, client):
        client.post("/api/settings", json={"paletteApiKey": "dp_valid_key"})
        resp = client.get("/api/sync/credits")
        assert resp.status_code == 200
        data = resp.json()
        assert data["balance_cents"] == 5000
        assert "pricing" in data

    def test_credits_when_disconnected(self, client):
        resp = client.get("/api/sync/credits")
        assert resp.status_code == 200
        data = resp.json()
        assert data["balance_cents"] is None
        assert data["connected"] is False
