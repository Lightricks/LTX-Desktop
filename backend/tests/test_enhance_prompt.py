"""Tests for prompt enhancement route."""
from __future__ import annotations

from tests.fakes.services import FakeResponse


# ── Gemini path ────────────────────────────────────────────────


def test_enhance_prompt_returns_enhanced_text(client, test_state, fake_services):
    """Enhance prompt route should return an enhanced version of the input via Gemini."""
    test_state.state.app_settings.gemini_api_key = "test-gemini-key"

    fake_services.http.queue("post", FakeResponse(
        status_code=200,
        json_payload={
            "candidates": [{"content": {"parts": [{"text": "A cinematic shot of a majestic cat walking gracefully across a sun-drenched room, golden hour lighting, shallow depth of field"}]}}]
        },
    ))

    resp = client.post("/api/enhance-prompt", json={
        "prompt": "cat walking in room",
        "mode": "text-to-video",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "enhancedPrompt" in data
    assert len(data["enhancedPrompt"]) > len("cat walking in room")


def test_enhance_prompt_image_mode(client, test_state, fake_services):
    """Should work with image mode via Gemini."""
    test_state.state.app_settings.gemini_api_key = "test-key"

    fake_services.http.queue("post", FakeResponse(
        status_code=200,
        json_payload={
            "candidates": [{"content": {"parts": [{"text": "A stunning photograph of a cat"}]}}]
        },
    ))

    resp = client.post("/api/enhance-prompt", json={
        "prompt": "cat photo",
        "mode": "text-to-image",
    })
    assert resp.status_code == 200
    assert "enhancedPrompt" in resp.json()


# ── Palette proxy path ─────────────────────────────────────────


def test_enhance_prompt_via_palette(client, test_state, fake_services):
    """When palette_api_key is set, should proxy to Palette prompt-expander."""
    test_state.state.app_settings.palette_api_key = "pal-test-key"

    fake_services.http.queue("post", FakeResponse(
        status_code=200,
        json_payload={"enhanced_prompt": "A breathtaking cinematic scene of a cat"},
    ))

    resp = client.post("/api/enhance-prompt", json={
        "prompt": "cat scene",
        "mode": "text-to-video",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["enhancedPrompt"] == "A breathtaking cinematic scene of a cat"

    # Verify the correct URL and auth header were used
    call = fake_services.http.calls[-1]
    assert call.method == "post"
    assert "/api/prompt-expander" in call.url
    assert call.headers is not None
    assert call.headers["Authorization"] == "Bearer pal-test-key"
    assert call.json_payload is not None
    assert call.json_payload["prompt"] == "cat scene"
    assert call.json_payload["level"] == "2x"


def test_enhance_prompt_palette_takes_priority_over_gemini(client, test_state, fake_services):
    """When both keys are set, Palette should be used instead of Gemini."""
    test_state.state.app_settings.palette_api_key = "pal-key"
    test_state.state.app_settings.gemini_api_key = "gem-key"

    fake_services.http.queue("post", FakeResponse(
        status_code=200,
        json_payload={"enhanced_prompt": "Enhanced via palette"},
    ))

    resp = client.post("/api/enhance-prompt", json={
        "prompt": "test prompt",
        "mode": "text-to-video",
    })
    assert resp.status_code == 200
    assert resp.json()["enhancedPrompt"] == "Enhanced via palette"

    # Should have called Palette, not Gemini
    call = fake_services.http.calls[-1]
    assert "/api/prompt-expander" in call.url


def test_enhance_prompt_palette_error_propagates(client, test_state, fake_services):
    """Palette API errors should propagate as HTTP errors."""
    test_state.state.app_settings.palette_api_key = "pal-key"

    fake_services.http.queue("post", FakeResponse(
        status_code=500,
        text="Internal Server Error",
    ))

    resp = client.post("/api/enhance-prompt", json={
        "prompt": "test",
        "mode": "text-to-video",
    })
    assert resp.status_code == 500


def test_enhance_prompt_palette_expanded_prompt_field(client, test_state, fake_services):
    """Should also accept expandedPrompt field from Palette response."""
    test_state.state.app_settings.palette_api_key = "pal-key"

    fake_services.http.queue("post", FakeResponse(
        status_code=200,
        json_payload={"expandedPrompt": "Expanded prompt text"},
    ))

    resp = client.post("/api/enhance-prompt", json={
        "prompt": "short prompt",
        "mode": "text-to-video",
    })
    assert resp.status_code == 200
    assert resp.json()["enhancedPrompt"] == "Expanded prompt text"


# ── No service configured ──────────────────────────────────────


def test_enhance_prompt_no_service_configured(client):
    """Should return error when neither Palette nor Gemini is configured."""
    resp = client.post("/api/enhance-prompt", json={
        "prompt": "cat walking",
        "mode": "text-to-video",
    })
    assert resp.status_code == 400
    assert "NO_AI_SERVICE_CONFIGURED" in resp.json()["error"]
