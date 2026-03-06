"""Tests for queue API routes."""

from __future__ import annotations


def test_submit_video_job(client):
    resp = client.post("/api/queue/submit", json={
        "type": "video",
        "model": "ltx-fast",
        "params": {"prompt": "a cat", "duration": "6", "resolution": "720p", "aspectRatio": "16:9"},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "queued"
    assert "id" in data


def test_submit_image_job(client):
    resp = client.post("/api/queue/submit", json={
        "type": "image",
        "model": "z-image-turbo",
        "params": {"prompt": "a dog", "width": 1024, "height": 1024},
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "queued"


def test_get_queue_status(client):
    client.post("/api/queue/submit", json={
        "type": "video",
        "model": "ltx-fast",
        "params": {"prompt": "test"},
    })
    resp = client.get("/api/queue/status")
    assert resp.status_code == 200
    jobs = resp.json()["jobs"]
    assert len(jobs) == 1
    assert jobs[0]["status"] == "queued"


def test_cancel_job(client):
    submit_resp = client.post("/api/queue/submit", json={
        "type": "video",
        "model": "ltx-fast",
        "params": {"prompt": "test"},
    })
    job_id = submit_resp.json()["id"]
    cancel_resp = client.post(f"/api/queue/cancel/{job_id}")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "cancelled"


def test_clear_finished_jobs(client):
    submit_resp = client.post("/api/queue/submit", json={
        "type": "video",
        "model": "ltx-fast",
        "params": {"prompt": "test"},
    })
    job_id = submit_resp.json()["id"]
    client.post(f"/api/queue/cancel/{job_id}")
    client.post("/api/queue/clear")
    status_resp = client.get("/api/queue/status")
    assert len(status_resp.json()["jobs"]) == 0


def test_seedance_routes_to_api_slot(client):
    resp = client.post("/api/queue/submit", json={
        "type": "video",
        "model": "seedance-1.5-pro",
        "params": {"prompt": "test"},
    })
    assert resp.status_code == 200
    status = client.get("/api/queue/status")
    jobs = status.json()["jobs"]
    assert len(jobs) == 1
    assert jobs[0]["slot"] == "api"


def test_nano_banana_routes_to_api_slot(client):
    resp = client.post("/api/queue/submit", json={
        "type": "image",
        "model": "nano-banana-2",
        "params": {"prompt": "test"},
    })
    assert resp.status_code == 200
    status = client.get("/api/queue/status")
    assert status.json()["jobs"][0]["slot"] == "api"
