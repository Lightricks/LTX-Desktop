"""Tests for prompt library and wildcard endpoints."""

from __future__ import annotations

import random

from services.wildcard_parser import WildcardDef, expand_prompt, expand_random


# ============================================================
# Unit tests for wildcard_parser
# ============================================================


class TestExpandPrompt:
    def test_single_wildcard(self):
        result = expand_prompt(
            "A _color_ car",
            [WildcardDef("color", ["red", "blue"])],
        )
        assert sorted(result) == ["A blue car", "A red car"]

    def test_two_wildcards_cartesian_product(self):
        result = expand_prompt(
            "A _color_ _animal_",
            [
                WildcardDef("color", ["red", "blue"]),
                WildcardDef("animal", ["cat", "dog"]),
            ],
        )
        assert sorted(result) == [
            "A blue cat",
            "A blue dog",
            "A red cat",
            "A red dog",
        ]

    def test_no_wildcards_returns_original(self):
        result = expand_prompt("plain prompt", [])
        assert result == ["plain prompt"]

    def test_undefined_wildcard_kept_literal(self):
        result = expand_prompt("A _missing_ thing", [])
        assert result == ["A _missing_ thing"]

    def test_nested_wildcards(self):
        """A wildcard value itself contains another wildcard reference."""
        result = expand_prompt(
            "I see a _thing_",
            [
                WildcardDef("thing", ["_color_ ball"]),
                WildcardDef("color", ["red", "green"]),
            ],
        )
        assert sorted(result) == ["I see a green ball", "I see a red ball"]

    def test_repeated_wildcard(self):
        """Same wildcard used twice in a prompt expands to same value in each slot."""
        result = expand_prompt(
            "_color_ and _color_",
            [WildcardDef("color", ["red", "blue"])],
        )
        # Both slots get replaced by same value since replace is global
        assert sorted(result) == ["blue and blue", "red and red"]


class TestExpandRandom:
    def test_returns_requested_count(self):
        result = expand_random(
            "A _color_ car",
            [WildcardDef("color", ["red", "blue", "green"])],
            count=5,
            rng=random.Random(42),
        )
        assert len(result) == 5
        for r in result:
            assert "_color_" not in r

    def test_single_result_default(self):
        result = expand_random(
            "A _color_ _animal_",
            [
                WildcardDef("color", ["red"]),
                WildcardDef("animal", ["cat"]),
            ],
            rng=random.Random(0),
        )
        assert result == ["A red cat"]

    def test_no_wildcards_returns_original(self):
        result = expand_random("no wildcards here", [], count=3)
        assert result == ["no wildcards here"] * 3

    def test_nested_random(self):
        result = expand_random(
            "A _thing_",
            [
                WildcardDef("thing", ["_color_ ball"]),
                WildcardDef("color", ["red"]),
            ],
            count=1,
            rng=random.Random(0),
        )
        assert result == ["A red ball"]


# ============================================================
# Integration tests for routes
# ============================================================


class TestPromptCRUD:
    def test_save_and_list(self, client):
        r = client.post("/api/prompts", json={"text": "a sunset over mountains", "tags": ["nature"], "category": "landscape"})
        assert r.status_code == 200
        data = r.json()
        assert data["text"] == "a sunset over mountains"
        assert data["tags"] == ["nature"]
        assert data["category"] == "landscape"
        assert data["used_count"] == 0
        prompt_id = data["id"]

        r = client.get("/api/prompts")
        assert r.status_code == 200
        prompts = r.json()["prompts"]
        assert len(prompts) == 1
        assert prompts[0]["id"] == prompt_id

    def test_delete_prompt(self, client):
        r = client.post("/api/prompts", json={"text": "to delete"})
        prompt_id = r.json()["id"]

        r = client.delete(f"/api/prompts/{prompt_id}")
        assert r.status_code == 200

        r = client.get("/api/prompts")
        assert len(r.json()["prompts"]) == 0

    def test_delete_nonexistent_returns_404(self, client):
        r = client.delete("/api/prompts/nonexistent")
        assert r.status_code == 404

    def test_save_prompt_minimal(self, client):
        """Save with only required field (text)."""
        r = client.post("/api/prompts", json={"text": "minimal prompt"})
        assert r.status_code == 200
        data = r.json()
        assert data["tags"] == []
        assert data["category"] == ""


class TestPromptSearch:
    def test_search_by_text(self, client):
        client.post("/api/prompts", json={"text": "a red fox in snow"})
        client.post("/api/prompts", json={"text": "blue ocean waves"})

        r = client.get("/api/prompts", params={"search": "fox"})
        prompts = r.json()["prompts"]
        assert len(prompts) == 1
        assert "fox" in prompts[0]["text"]

    def test_filter_by_tag(self, client):
        client.post("/api/prompts", json={"text": "prompt1", "tags": ["nature"]})
        client.post("/api/prompts", json={"text": "prompt2", "tags": ["urban"]})

        r = client.get("/api/prompts", params={"tag": "nature"})
        prompts = r.json()["prompts"]
        assert len(prompts) == 1
        assert prompts[0]["text"] == "prompt1"

    def test_sort_by_used_count(self, client):
        r1 = client.post("/api/prompts", json={"text": "less used"})
        r2 = client.post("/api/prompts", json={"text": "more used"})
        id2 = r2.json()["id"]

        # Bump usage on second prompt
        client.post(f"/api/prompts/{id2}/usage")
        client.post(f"/api/prompts/{id2}/usage")

        r = client.get("/api/prompts", params={"sort_by": "used_count"})
        prompts = r.json()["prompts"]
        assert prompts[0]["text"] == "more used"
        assert prompts[0]["used_count"] == 2


class TestUsageTracking:
    def test_increment_usage(self, client):
        r = client.post("/api/prompts", json={"text": "track me"})
        prompt_id = r.json()["id"]

        r = client.post(f"/api/prompts/{prompt_id}/usage")
        assert r.status_code == 200
        assert r.json()["used_count"] == 1

        r = client.post(f"/api/prompts/{prompt_id}/usage")
        assert r.json()["used_count"] == 2

    def test_increment_nonexistent_returns_404(self, client):
        r = client.post("/api/prompts/nonexistent/usage")
        assert r.status_code == 404


class TestWildcardCRUD:
    def test_create_and_list(self, client):
        r = client.post("/api/wildcards", json={"name": "color", "values": ["red", "blue", "green"]})
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "color"
        assert data["values"] == ["red", "blue", "green"]
        wc_id = data["id"]

        r = client.get("/api/wildcards")
        assert r.status_code == 200
        wildcards = r.json()["wildcards"]
        assert len(wildcards) == 1
        assert wildcards[0]["id"] == wc_id

    def test_update_wildcard(self, client):
        r = client.post("/api/wildcards", json={"name": "animal", "values": ["cat"]})
        wc_id = r.json()["id"]

        r = client.put(f"/api/wildcards/{wc_id}", json={"values": ["cat", "dog", "bird"]})
        assert r.status_code == 200
        assert r.json()["values"] == ["cat", "dog", "bird"]

    def test_update_nonexistent_returns_404(self, client):
        r = client.put("/api/wildcards/nonexistent", json={"values": ["x"]})
        assert r.status_code == 404

    def test_delete_wildcard(self, client):
        r = client.post("/api/wildcards", json={"name": "size", "values": ["big", "small"]})
        wc_id = r.json()["id"]

        r = client.delete(f"/api/wildcards/{wc_id}")
        assert r.status_code == 200

        r = client.get("/api/wildcards")
        assert len(r.json()["wildcards"]) == 0

    def test_delete_nonexistent_returns_404(self, client):
        r = client.delete("/api/wildcards/nonexistent")
        assert r.status_code == 404


class TestWildcardExpansion:
    def test_expand_all(self, client):
        client.post("/api/wildcards", json={"name": "color", "values": ["red", "blue"]})
        client.post("/api/wildcards", json={"name": "animal", "values": ["cat", "dog"]})

        r = client.post("/api/wildcards/expand", json={
            "prompt": "A _color_ _animal_",
            "mode": "all",
        })
        assert r.status_code == 200
        expanded = r.json()["expanded"]
        assert sorted(expanded) == [
            "A blue cat",
            "A blue dog",
            "A red cat",
            "A red dog",
        ]

    def test_expand_random(self, client):
        client.post("/api/wildcards", json={"name": "color", "values": ["red", "blue", "green"]})

        r = client.post("/api/wildcards/expand", json={
            "prompt": "A _color_ thing",
            "mode": "random",
            "count": 3,
        })
        assert r.status_code == 200
        expanded = r.json()["expanded"]
        assert len(expanded) == 3
        for e in expanded:
            assert "_color_" not in e

    def test_expand_no_wildcards(self, client):
        r = client.post("/api/wildcards/expand", json={
            "prompt": "plain prompt",
            "mode": "all",
        })
        assert r.status_code == 200
        assert r.json()["expanded"] == ["plain prompt"]

    def test_expand_nested_wildcards(self, client):
        """Wildcard value contains another wildcard reference."""
        client.post("/api/wildcards", json={"name": "scene", "values": ["_color_ sunset"]})
        client.post("/api/wildcards", json={"name": "color", "values": ["golden", "crimson"]})

        r = client.post("/api/wildcards/expand", json={
            "prompt": "A _scene_",
            "mode": "all",
        })
        assert r.status_code == 200
        expanded = sorted(r.json()["expanded"])
        assert expanded == ["A crimson sunset", "A golden sunset"]
