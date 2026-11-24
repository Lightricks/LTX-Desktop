"""GET/POST /api/generation-seed: the one seed shared by Settings, forms, and the remote app."""

from __future__ import annotations

from starlette.testclient import TestClient

from remote.app import create_remote_app

AUTH = {"Authorization": "Bearer pair-token"}


def test_post_seed_persists_to_the_settings_the_generators_read(client, test_state) -> None:
    saved = client.post("/api/generation-seed", json={"seed": 777, "locked": True})

    assert saved.status_code == 200
    assert saved.json() == {"seed": 777, "locked": True}
    assert client.get("/api/generation-seed").json() == {"seed": 777, "locked": True}
    assert test_state.state.app_settings.locked_seed == 777
    assert test_state.state.app_settings.seed_locked is True


def test_locking_alone_keeps_the_seed(client, test_state) -> None:
    client.post("/api/generation-seed", json={"seed": 555})

    locked = client.post("/api/generation-seed", json={"locked": True})

    assert locked.json() == {"seed": 555, "locked": True}


def test_seed_alone_keeps_the_lock(client, test_state) -> None:
    client.post("/api/generation-seed", json={"locked": True})

    client.post("/api/generation-seed", json={"seed": 12})

    assert test_state.state.app_settings.seed_locked is True


def test_out_of_range_seed_is_rejected_not_clamped(client, test_state) -> None:
    before = client.get("/api/generation-seed").json()

    too_big = client.post("/api/generation-seed", json={"seed": 2_147_483_648})
    negative = client.post("/api/generation-seed", json={"seed": -1})

    assert too_big.status_code == 422
    assert negative.status_code == 422
    assert client.get("/api/generation-seed").json() == before


def test_remote_can_read_and_change_the_seed_but_only_with_the_token(test_state) -> None:
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as remote:
        unauth = remote.post("/api/generation-seed", json={"seed": 9})
        saved = remote.post("/api/generation-seed", headers=AUTH, json={"seed": 9, "locked": True})
        loaded = remote.get("/api/generation-seed", headers=AUTH)
        settings = remote.get("/api/settings", headers=AUTH)

    assert unauth.status_code == 401
    assert saved.json() == {"seed": 9, "locked": True}
    assert loaded.json() == {"seed": 9, "locked": True}
    assert test_state.state.app_settings.locked_seed == 9
    assert settings.status_code == 404
