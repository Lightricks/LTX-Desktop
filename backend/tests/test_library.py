"""Tests for /api/library/* endpoints (characters, styles, references)."""

from __future__ import annotations


class TestCharacters:
    def test_list_empty(self, client):
        r = client.get("/api/library/characters")
        assert r.status_code == 200
        assert r.json()["characters"] == []

    def test_create_character(self, client):
        r = client.post("/api/library/characters", json={
            "name": "Alice",
            "role": "protagonist",
            "description": "A curious adventurer",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "Alice"
        assert data["role"] == "protagonist"
        assert data["description"] == "A curious adventurer"
        assert data["reference_image_paths"] == []
        assert "id" in data
        assert "created_at" in data

    def test_create_character_with_images(self, client):
        r = client.post("/api/library/characters", json={
            "name": "Bob",
            "role": "sidekick",
            "description": "Loyal friend",
            "reference_image_paths": ["/img/bob1.png", "/img/bob2.png"],
        })
        assert r.status_code == 200
        assert r.json()["reference_image_paths"] == ["/img/bob1.png", "/img/bob2.png"]

    def test_list_after_create(self, client):
        client.post("/api/library/characters", json={"name": "Alice"})
        client.post("/api/library/characters", json={"name": "Bob"})
        r = client.get("/api/library/characters")
        assert r.status_code == 200
        names = [c["name"] for c in r.json()["characters"]]
        assert "Alice" in names
        assert "Bob" in names

    def test_update_character(self, client):
        r = client.post("/api/library/characters", json={"name": "Alice", "role": "hero"})
        cid = r.json()["id"]

        r = client.put(f"/api/library/characters/{cid}", json={"role": "villain", "description": "Turned evil"})
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "Alice"
        assert data["role"] == "villain"
        assert data["description"] == "Turned evil"

    def test_update_partial_preserves_fields(self, client):
        r = client.post("/api/library/characters", json={
            "name": "Alice",
            "role": "hero",
            "description": "Brave",
        })
        cid = r.json()["id"]

        r = client.put(f"/api/library/characters/{cid}", json={"role": "mentor"})
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "Alice"
        assert data["role"] == "mentor"
        assert data["description"] == "Brave"

    def test_update_nonexistent_returns_404(self, client):
        r = client.put("/api/library/characters/doesnotexist", json={"name": "X"})
        assert r.status_code == 404

    def test_delete_character(self, client):
        r = client.post("/api/library/characters", json={"name": "Alice"})
        cid = r.json()["id"]

        r = client.delete(f"/api/library/characters/{cid}")
        assert r.status_code == 200

        r = client.get("/api/library/characters")
        assert len(r.json()["characters"]) == 0

    def test_delete_nonexistent_returns_404(self, client):
        r = client.delete("/api/library/characters/doesnotexist")
        assert r.status_code == 404

    def test_create_empty_name_returns_400(self, client):
        r = client.post("/api/library/characters", json={"name": "  "})
        assert r.status_code == 400


class TestStyles:
    def test_list_empty(self, client):
        r = client.get("/api/library/styles")
        assert r.status_code == 200
        assert r.json()["styles"] == []

    def test_create_style(self, client):
        r = client.post("/api/library/styles", json={
            "name": "Noir",
            "description": "Dark and moody",
            "reference_image_path": "/img/noir.png",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "Noir"
        assert data["description"] == "Dark and moody"
        assert data["reference_image_path"] == "/img/noir.png"
        assert "id" in data

    def test_list_after_create(self, client):
        client.post("/api/library/styles", json={"name": "Noir"})
        client.post("/api/library/styles", json={"name": "Pastel"})
        r = client.get("/api/library/styles")
        names = [s["name"] for s in r.json()["styles"]]
        assert "Noir" in names
        assert "Pastel" in names

    def test_delete_style(self, client):
        r = client.post("/api/library/styles", json={"name": "Noir"})
        sid = r.json()["id"]

        r = client.delete(f"/api/library/styles/{sid}")
        assert r.status_code == 200

        r = client.get("/api/library/styles")
        assert len(r.json()["styles"]) == 0

    def test_delete_nonexistent_returns_404(self, client):
        r = client.delete("/api/library/styles/doesnotexist")
        assert r.status_code == 404

    def test_create_empty_name_returns_400(self, client):
        r = client.post("/api/library/styles", json={"name": "  "})
        assert r.status_code == 400


class TestReferences:
    def test_list_empty(self, client):
        r = client.get("/api/library/references")
        assert r.status_code == 200
        assert r.json()["references"] == []

    def test_create_reference(self, client):
        r = client.post("/api/library/references", json={
            "name": "City Park",
            "category": "places",
            "image_path": "/img/park.png",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "City Park"
        assert data["category"] == "places"
        assert data["image_path"] == "/img/park.png"
        assert "id" in data

    def test_list_all_categories(self, client):
        client.post("/api/library/references", json={"name": "Park", "category": "places"})
        client.post("/api/library/references", json={"name": "Sword", "category": "props"})
        client.post("/api/library/references", json={"name": "Hero", "category": "people"})

        r = client.get("/api/library/references")
        assert len(r.json()["references"]) == 3

    def test_filter_by_category(self, client):
        client.post("/api/library/references", json={"name": "Park", "category": "places"})
        client.post("/api/library/references", json={"name": "Sword", "category": "props"})
        client.post("/api/library/references", json={"name": "Beach", "category": "places"})

        r = client.get("/api/library/references?category=places")
        refs = r.json()["references"]
        assert len(refs) == 2
        assert all(ref["category"] == "places" for ref in refs)

        r = client.get("/api/library/references?category=props")
        refs = r.json()["references"]
        assert len(refs) == 1
        assert refs[0]["name"] == "Sword"

    def test_filter_empty_category(self, client):
        client.post("/api/library/references", json={"name": "Park", "category": "places"})

        r = client.get("/api/library/references?category=people")
        assert r.json()["references"] == []

    def test_delete_reference(self, client):
        r = client.post("/api/library/references", json={"name": "Park", "category": "places"})
        rid = r.json()["id"]

        r = client.delete(f"/api/library/references/{rid}")
        assert r.status_code == 200

        r = client.get("/api/library/references")
        assert len(r.json()["references"]) == 0

    def test_delete_nonexistent_returns_404(self, client):
        r = client.delete("/api/library/references/doesnotexist")
        assert r.status_code == 404

    def test_create_empty_name_returns_400(self, client):
        r = client.post("/api/library/references", json={"name": "  ", "category": "other"})
        assert r.status_code == 400

    def test_invalid_category_rejected(self, client):
        r = client.post("/api/library/references", json={"name": "X", "category": "invalid"})
        assert r.status_code == 422
