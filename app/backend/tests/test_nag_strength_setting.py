"""The `showNagStrengthControls` setting: Desktop saves it, the phone reads it from the status."""

from __future__ import annotations

import json

from starlette.testclient import TestClient

from remote.app import create_remote_app

AUTH = {"Authorization": "Bearer pair-token"}


def test_setting_is_off_by_default_on_desktop_and_phone(client, test_state) -> None:
    """Bug: the row showing for everyone after an upgrade."""
    assert client.get("/api/settings").json()["showNagStrengthControls"] is False
    assert client.get("/api/prompt-enhancer").json()["showNagStrengthControls"] is False

    with TestClient(create_remote_app(handler=test_state, remote_token="pair-token")) as phone:
        status = phone.get("/api/prompt-enhancer", headers=AUTH)
    assert status.json()["showNagStrengthControls"] is False


def test_turning_the_setting_on_is_saved_and_reaches_the_phone(client, test_state) -> None:
    saved = client.post("/api/settings", json={"showNagStrengthControls": True})
    assert saved.status_code == 200, saved.text

    assert client.get("/api/settings").json()["showNagStrengthControls"] is True
    assert client.get("/api/prompt-enhancer").json()["showNagStrengthControls"] is True
    # Saved to disk, so it survives a restart.
    on_disk = json.loads(test_state.config.settings_file.read_text(encoding="utf-8"))
    assert on_disk["show_nag_strength_controls"] is True

    with TestClient(create_remote_app(handler=test_state, remote_token="pair-token")) as phone:
        status = phone.get("/api/prompt-enhancer", headers=AUTH)
    assert status.json()["showNagStrengthControls"] is True

    client.post("/api/settings", json={"showNagStrengthControls": False})
    with TestClient(create_remote_app(handler=test_state, remote_token="pair-token")) as phone:
        status = phone.get("/api/prompt-enhancer", headers=AUTH)
    assert status.json()["showNagStrengthControls"] is False


def test_phone_cannot_change_the_setting(test_state) -> None:
    """The phone only reads the status. A settings write from it must not exist."""
    with TestClient(create_remote_app(handler=test_state, remote_token="pair-token")) as phone:
        response = phone.post(
            "/api/settings", headers=AUTH, json={"showNagStrengthControls": True}
        )
    assert response.status_code in (404, 405)
    assert test_state.state.app_settings.show_nag_strength_controls is False
