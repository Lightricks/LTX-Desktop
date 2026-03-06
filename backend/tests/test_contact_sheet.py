"""Tests for contact sheet generation endpoint."""
from __future__ import annotations


class TestContactSheetGenerate:
    def test_generates_9_jobs(self, client):
        resp = client.post("/api/contact-sheet/generate", json={
            "reference_image_path": "/tmp/ref.png",
            "subject_description": "A woman in a red dress",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["job_ids"]) == 9

        # All IDs should be unique
        assert len(set(data["job_ids"])) == 9

    def test_jobs_appear_in_queue(self, client):
        resp = client.post("/api/contact-sheet/generate", json={
            "reference_image_path": "/tmp/ref.png",
            "subject_description": "A man in a suit",
        })
        job_ids = resp.json()["job_ids"]

        queue = client.get("/api/queue/status").json()
        queue_ids = {j["id"] for j in queue["jobs"]}
        for jid in job_ids:
            assert jid in queue_ids

    def test_prompts_include_subject_and_angles(self, client):
        resp = client.post("/api/contact-sheet/generate", json={
            "reference_image_path": "/tmp/ref.png",
            "subject_description": "A warrior elf",
        })
        job_ids = resp.json()["job_ids"]

        queue = client.get("/api/queue/status").json()
        prompts = [j["params"]["prompt"] for j in queue["jobs"]]

        # Every prompt should contain the subject
        for prompt in prompts:
            assert "A warrior elf" in prompt

        # Check that different camera angles are present
        angle_keywords = [
            "Close-up",
            "Medium shot",
            "Full body",
            "Over-the-shoulder",
            "Low angle",
            "High angle",
            "Profile",
            "Three-quarter",
            "Wide establishing",
        ]
        for keyword in angle_keywords:
            assert any(keyword in p for p in prompts), f"Missing angle keyword: {keyword}"

    def test_prompts_include_style(self, client):
        resp = client.post("/api/contact-sheet/generate", json={
            "reference_image_path": "/tmp/ref.png",
            "subject_description": "A robot",
            "style": "cyberpunk neon",
        })
        job_ids = resp.json()["job_ids"]

        queue = client.get("/api/queue/status").json()
        prompts = [j["params"]["prompt"] for j in queue["jobs"]]

        for prompt in prompts:
            assert "cyberpunk neon" in prompt

    def test_returns_9_job_ids(self, client):
        resp = client.post("/api/contact-sheet/generate", json={
            "reference_image_path": "/tmp/ref.png",
            "subject_description": "A cat",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "job_ids" in data
        assert len(data["job_ids"]) == 9
