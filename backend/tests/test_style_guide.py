"""Tests for style guide grid generation endpoint."""
from __future__ import annotations


class TestStyleGuideGenerate:
    def test_generates_9_jobs(self, client):
        resp = client.post("/api/style-guide/generate", json={
            "style_name": "Impressionism",
            "style_description": "soft brushstrokes, vibrant light",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["job_ids"]) == 9
        assert len(set(data["job_ids"])) == 9

    def test_jobs_appear_in_queue(self, client):
        resp = client.post("/api/style-guide/generate", json={
            "style_name": "Art Deco",
        })
        job_ids = resp.json()["job_ids"]

        queue = client.get("/api/queue/status").json()
        queue_ids = {j["id"] for j in queue["jobs"]}
        for jid in job_ids:
            assert jid in queue_ids

    def test_prompts_include_style_name_and_description(self, client):
        resp = client.post("/api/style-guide/generate", json={
            "style_name": "Film Noir",
            "style_description": "high contrast, dramatic shadows",
        })
        job_ids = resp.json()["job_ids"]

        queue = client.get("/api/queue/status").json()
        prompts = [j["params"]["prompt"] for j in queue["jobs"]]

        for prompt in prompts:
            assert "Film Noir" in prompt
            assert "high contrast, dramatic shadows" in prompt

    def test_prompts_include_diverse_subjects(self, client):
        resp = client.post("/api/style-guide/generate", json={
            "style_name": "Watercolor",
        })
        job_ids = resp.json()["job_ids"]

        queue = client.get("/api/queue/status").json()
        prompts = [j["params"]["prompt"] for j in queue["jobs"]]

        subject_keywords = [
            "Portrait",
            "Cityscape",
            "Nature landscape",
            "Interior room",
            "Food",
            "Vehicle",
            "Animal",
            "Architecture",
            "Abstract",
        ]
        for keyword in subject_keywords:
            assert any(keyword in p for p in prompts), f"Missing subject keyword: {keyword}"

    def test_returns_9_job_ids(self, client):
        resp = client.post("/api/style-guide/generate", json={
            "style_name": "Pop Art",
            "style_description": "bold colors, comic style",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "job_ids" in data
        assert len(data["job_ids"]) == 9

    def test_reference_image_path_passed_to_jobs(self, client):
        resp = client.post("/api/style-guide/generate", json={
            "style_name": "Minimalist",
            "reference_image_path": "/tmp/style_ref.png",
        })
        job_ids = resp.json()["job_ids"]

        queue = client.get("/api/queue/status").json()
        for job in queue["jobs"]:
            assert job["params"]["reference_image_path"] == "/tmp/style_ref.png"

    def test_no_reference_image_when_not_provided(self, client):
        resp = client.post("/api/style-guide/generate", json={
            "style_name": "Gothic",
        })
        job_ids = resp.json()["job_ids"]

        queue = client.get("/api/queue/status").json()
        for job in queue["jobs"]:
            assert "reference_image_path" not in job["params"]
