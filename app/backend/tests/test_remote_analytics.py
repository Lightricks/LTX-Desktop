from __future__ import annotations

from starlette.testclient import TestClient

from remote.app import create_remote_app
from services.analytics import queued_generation_details
from services.sqlite_store import SqliteStore


def test_remote_generation_is_not_mislabeled_as_desktop(test_state) -> None:
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        response = client.post(
            "/api/generations/text-to-video",
            headers={"Authorization": "Bearer pair-token"},
            json={
                "params": {
                    "prompt": "a fox",
                    "model": "ltx-2.5-fast",
                }
            },
        )

    assert response.status_code == 200, response.text
    assert "_analytics" not in response.json()["spec"]
    stored = SqliteStore(test_state.config.app_data_dir).get_generation(
        response.json()["id"]
    )
    assert stored is not None
    assert stored.spec["_analytics"] == {"client": "remote"}
    details = queued_generation_details(stored)
    assert details is not None
    assert details["client"] == "remote"
    assert "prompt" not in details
