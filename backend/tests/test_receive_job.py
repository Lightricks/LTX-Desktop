"""Tests for receive-job endpoint (jobs from Director's Palette)."""
from __future__ import annotations


class TestReceiveJobConnected:
    def test_receive_job_creates_queue_entry(self, client):
        client.post("/api/settings", json={"paletteApiKey": "dp_valid_key"})
        resp = client.post("/api/sync/receive-job", json={
            "prompt": "A cinematic sunset over the ocean",
            "model": "ltx-fast",
            "settings": {"resolution": "720p", "duration": "4", "fps": "24", "aspect_ratio": "16:9"},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert "id" in data

        # Verify it appeared in the queue
        queue = client.get("/api/queue/status").json()
        assert len(queue["jobs"]) == 1
        job = queue["jobs"][0]
        assert job["id"] == data["id"]
        assert job["params"]["prompt"] == "A cinematic sunset over the ocean"

    def test_receive_job_with_character_reference(self, client):
        client.post("/api/settings", json={"paletteApiKey": "dp_valid_key"})
        resp = client.post("/api/sync/receive-job", json={
            "prompt": "Character walking in park",
            "model": "ltx-fast",
            "character_id": "char-abc-123",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"

        queue = client.get("/api/queue/status").json()
        job = queue["jobs"][0]
        assert job["params"]["character_id"] == "char-abc-123"

    def test_receive_job_with_first_frame_url(self, client, fake_services):
        client.post("/api/settings", json={"paletteApiKey": "dp_valid_key"})
        from tests.fakes.services import FakeResponse
        fake_services.http.queue("get", FakeResponse(status_code=200, content=b"fake-image-data"))
        resp = client.post("/api/sync/receive-job", json={
            "prompt": "Animate this scene",
            "model": "ltx-fast",
            "first_frame_url": "https://example.com/frame.png",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"

        queue = client.get("/api/queue/status").json()
        job = queue["jobs"][0]
        assert "imagePath" in job["params"]


class TestReceiveJobDisconnected:
    def test_receive_job_returns_403_without_api_key(self, client):
        resp = client.post("/api/sync/receive-job", json={
            "prompt": "A cinematic sunset over the ocean",
            "model": "ltx-fast",
        })
        assert resp.status_code == 403
        data = resp.json()
        assert "error" in data


class TestReceiveJobValidation:
    def test_receive_job_rejects_empty_prompt(self, client):
        client.post("/api/settings", json={"paletteApiKey": "dp_valid_key"})
        resp = client.post("/api/sync/receive-job", json={
            "prompt": "   ",
            "model": "ltx-fast",
        })
        assert resp.status_code == 422

    def test_receive_job_rejects_missing_prompt(self, client):
        client.post("/api/settings", json={"paletteApiKey": "dp_valid_key"})
        resp = client.post("/api/sync/receive-job", json={
            "model": "ltx-fast",
        })
        assert resp.status_code == 422
