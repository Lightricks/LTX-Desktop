"""GET /api/ui-preferences: the setting that shows the Advanced section reaches the phone."""

from __future__ import annotations

from starlette.testclient import TestClient

from remote.app import create_remote_app

_AUTH = {"Authorization": "Bearer pair-token"}


def test_remote_reads_the_advanced_controls_setting_the_desktop_saved(client: TestClient, test_state) -> None:
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as remote:
        assert remote.get("/api/ui-preferences").status_code == 401
        assert remote.get("/api/ui-preferences", headers=_AUTH).json() == {"advancedControls": False}

        assert client.post("/api/settings", json={"advancedControls": True}).status_code == 200

        assert remote.get("/api/ui-preferences", headers=_AUTH).json() == {"advancedControls": True}
