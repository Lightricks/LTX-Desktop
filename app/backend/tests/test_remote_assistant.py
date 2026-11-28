"""The Assistant on the LAN remote port: same chats as Desktop, none of Desktop's settings."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from starlette.testclient import TestClient

from http_exception_handlers import UNHANDLED_REMOTE_MESSAGE
from remote.app import create_remote_app
from tests.test_assistant import request_body, say, setup_agent  # noqa: F401  # pyright: ignore[reportUnusedImport]

AUTH = {"Authorization": "Bearer pair-token"}


def phone(test_state) -> TestClient:
    return TestClient(create_remote_app(handler=test_state, remote_token="pair-token"))


@pytest.mark.parametrize("setup_agent", ["gemini"], indirect=True)
def test_phone_reattaching_to_a_desktop_turn_does_not_cross_event_loops(client, test_state, setup_agent):
    provider, _ = setup_agent
    entered, release = Event(), Event()
    body = request_body("Hello")

    def delayed_reply():
        entered.set()
        assert release.wait(10)
        return say("Done.")

    provider.responses.append(delayed_reply)
    with phone(test_state) as remote, ThreadPoolExecutor() as pool:
        desktop = pool.submit(client.post, "/api/assistant/turn", json=body)
        assert entered.wait(10)
        reattached = pool.submit(remote.post, "/api/assistant/turn", json=body, headers=AUTH)
        release.set()
        assert desktop.result(timeout=10).json()["text"] == "Done."
        response = reattached.result(timeout=10)
    assert response.status_code == 200, response.text
    assert response.json()["text"] == "Done."


def test_remote_500_hides_exception_text(test_state):
    app = create_remote_app(handler=test_state, remote_token="pair-token")

    def crash() -> None:
        raise RuntimeError("/Users/someone/Movies/secret.mp4")

    app.add_api_route("/api/crash", crash)
    # Ahead of the SPA mount, which would otherwise claim the path.
    app.router.routes.insert(0, app.router.routes.pop())
    with TestClient(app, raise_server_exceptions=False) as remote:
        response = remote.get("/api/crash", headers=AUTH)
    assert response.status_code == 500
    assert "secret.mp4" not in response.text
    assert response.json()["message"] == UNHANDLED_REMOTE_MESSAGE


def test_phone_cannot_download_or_delete_assistant_models(test_state):
    with phone(test_state) as remote:
        download = remote.post("/api/assistant/models/download", json={"model": "qwen3.5:9b"}, headers=AUTH)
        delete = remote.post("/api/assistant/models/delete", json={"model": "qwen3.5:9b"}, headers=AUTH)
    assert download.status_code in (404, 405)
    assert delete.status_code in (404, 405)


def test_phone_selection_reports_a_gemini_key_without_returning_it(test_state):
    test_state.state.app_settings.gemini_api_key = "secret-gemini-key"
    with phone(test_state) as remote:
        response = remote.get("/api/assistant/selection", headers=AUTH)
    assert response.status_code == 200
    assert response.json()["has_gemini_api_key"] is True
    assert "secret-gemini-key" not in response.text


def test_phone_choosing_gemini_without_a_key_is_refused_and_keeps_the_previous_choice(test_state):
    settings = test_state.state.app_settings
    settings.gemini_api_key, settings.assistant_provider, settings.assistant_local_model = "", "local", "qwen3.5:9b"
    with phone(test_state) as remote:
        response = remote.put("/api/assistant/selection", json={"provider": "gemini"}, headers=AUTH)
    assert response.status_code == 400
    assert response.json()["code"] == "GEMINI_API_KEY_MISSING"
    assert (settings.assistant_provider, settings.assistant_local_model) == ("local", "qwen3.5:9b")


def test_phone_saving_an_uninstalled_local_model_is_refused_even_with_gemini_selected(test_state, fake_services):
    settings = test_state.state.app_settings
    settings.gemini_api_key, settings.assistant_provider = "key", "gemini"
    fake_services.assistant_models.models.pop("qwen3.5:9b")
    with phone(test_state) as remote:
        response = remote.put(
            "/api/assistant/selection", json={"provider": "gemini", "local_model": "qwen3.5:9b"}, headers=AUTH,
        )
    assert response.status_code == 409
    assert response.json()["code"] == "ASSISTANT_LOCAL_MODEL_MISSING"
    assert settings.assistant_provider == "gemini"


def test_phone_choice_shows_in_the_desktop_selection(client, test_state):
    test_state.state.app_settings.gemini_api_key = "key"
    with phone(test_state) as remote:
        response = remote.put(
            "/api/assistant/selection", json={"provider": "local", "local_model": "qwen3.5:9b"}, headers=AUTH,
        )
    assert response.status_code == 200, response.text
    desktop = client.get("/api/assistant/selection").json()
    assert (desktop["provider"], desktop["local_model"]) == ("local", "qwen3.5:9b")


def test_phone_assistant_routes_require_pairing(test_state):
    with phone(test_state) as remote:
        response = remote.post(
            "/api/assistant/conversations/2b1c0d4e-1f2a-4b3c-8d4e-5f6a7b8c9d0e/queue",
            json={"id": "7c1d2e3f-4a5b-4c6d-8e7f-9a0b1c2d3e4f", "text": "Hello"},
        )
    assert response.status_code == 401
