"""Tests for /api/gallery/local endpoints."""

from __future__ import annotations

from pathlib import Path


def _outputs_dir(test_state) -> Path:
    return test_state.config.outputs_dir


def _create_files(outputs: Path, filenames: list[str]) -> None:
    """Create dummy files in the outputs directory."""
    outputs.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        (outputs / name).write_bytes(b"\x00" * 128)


class TestListLocalAssets:
    def test_empty_gallery(self, client):
        r = client.get("/api/gallery/local")
        assert r.status_code == 200
        data = r.json()
        assert data["items"] == []
        assert data["total"] == 0
        assert data["page"] == 1
        assert data["total_pages"] == 1

    def test_list_with_files(self, client, test_state):
        outputs = _outputs_dir(test_state)
        _create_files(outputs, ["test_image.png", "test_video.mp4", "readme.txt"])

        r = client.get("/api/gallery/local")
        assert r.status_code == 200
        data = r.json()
        # txt file should be excluded
        assert data["total"] == 2
        filenames = {item["filename"] for item in data["items"]}
        assert filenames == {"test_image.png", "test_video.mp4"}

    def test_filter_by_image(self, client, test_state):
        outputs = _outputs_dir(test_state)
        _create_files(outputs, ["photo.png", "photo.jpg", "clip.mp4"])

        r = client.get("/api/gallery/local", params={"type": "image"})
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 2
        types = {item["type"] for item in data["items"]}
        assert types == {"image"}

    def test_filter_by_video(self, client, test_state):
        outputs = _outputs_dir(test_state)
        _create_files(outputs, ["photo.png", "clip.mp4", "clip2.webm"])

        r = client.get("/api/gallery/local", params={"type": "video"})
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 2
        types = {item["type"] for item in data["items"]}
        assert types == {"video"}

    def test_pagination(self, client, test_state):
        outputs = _outputs_dir(test_state)
        _create_files(outputs, [f"img_{i:03d}.png" for i in range(5)])

        r = client.get("/api/gallery/local", params={"page": 1, "per_page": 2})
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 5
        assert data["page"] == 1
        assert data["per_page"] == 2
        assert data["total_pages"] == 3
        assert len(data["items"]) == 2

        # Page 3 should have 1 item
        r2 = client.get("/api/gallery/local", params={"page": 3, "per_page": 2})
        data2 = r2.json()
        assert len(data2["items"]) == 1

    def test_model_name_parsed_from_prefix(self, client, test_state):
        outputs = _outputs_dir(test_state)
        _create_files(outputs, ["zit_image_001.png", "api_image_002.jpg", "ltx_fast_003.mp4", "random_004.png"])

        r = client.get("/api/gallery/local")
        assert r.status_code == 200
        data = r.json()
        by_filename = {item["filename"]: item["model_name"] for item in data["items"]}
        assert by_filename["zit_image_001.png"] == "zit"
        assert by_filename["api_image_002.jpg"] == "api"
        assert by_filename["ltx_fast_003.mp4"] == "ltx-fast"
        assert by_filename["random_004.png"] is None

    def test_asset_has_expected_fields(self, client, test_state):
        outputs = _outputs_dir(test_state)
        _create_files(outputs, ["sample.png"])

        r = client.get("/api/gallery/local")
        data = r.json()
        item = data["items"][0]
        assert "id" in item
        assert item["filename"] == "sample.png"
        assert item["type"] == "image"
        assert item["size_bytes"] == 128
        assert "created_at" in item
        assert "path" in item
        assert "url" in item

    def test_subdirectories_ignored(self, client, test_state):
        outputs = _outputs_dir(test_state)
        _create_files(outputs, ["top.png"])
        subdir = outputs / "subdir"
        subdir.mkdir()
        (subdir / "nested.png").write_bytes(b"\x00" * 64)

        r = client.get("/api/gallery/local")
        data = r.json()
        assert data["total"] == 1
        assert data["items"][0]["filename"] == "top.png"


class TestDeleteLocalAsset:
    def test_delete_existing_asset(self, client, test_state):
        outputs = _outputs_dir(test_state)
        _create_files(outputs, ["to_delete.png"])

        # Get the asset ID first
        r = client.get("/api/gallery/local")
        asset_id = r.json()["items"][0]["id"]

        # Delete it
        r2 = client.delete(f"/api/gallery/local/{asset_id}")
        assert r2.status_code == 200
        assert r2.json()["status"] == "ok"

        # Verify it's gone
        r3 = client.get("/api/gallery/local")
        assert r3.json()["total"] == 0

    def test_delete_nonexistent_asset(self, client):
        r = client.delete("/api/gallery/local/nonexistent_id_1234")
        assert r.status_code == 404
