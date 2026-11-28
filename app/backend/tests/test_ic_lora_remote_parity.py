"""The phone (remote app) gets the same IC-LoRA tiling verdicts and refusals as the desktop app.

Both apps call one handler. These tests pin that no remote route changes a verdict, a refusal
or a stored choice of the adaptive chunk window.
"""

from __future__ import annotations

from pathlib import Path

from starlette.testclient import TestClient

from remote.app import create_remote_app
from tests.test_ic_lora_tile_size import (
    _REFINE,
    _body,
    _configure_windows,
    _ingest_sized,
    _install,
)

_AUTH = {"Authorization": "Bearer pair-token"}
_WINDOWS = [97, 73, 57, 41]
_PREFLIGHT = f"/api/generations/ic-lora-recipes/{_REFINE}/preflight"
_CREATE = f"/api/generations/ic-lora-recipes/{_REFINE}"


def _preflight_body(video_id: str, resolution: str) -> dict[str, object]:
    return {
        **_body(video_id, resolution),
        "tileSizes": ["standard", "large", "extra-large"],
        "maxWindows": _WINDOWS,
        "carries": [17, 25, 33],
    }


def test_remote_preflight_matches_the_desktop_for_a_4k_clip(
    client: TestClient, fake_services, test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files(model_id="ltx-2.5-22b-distilled")
    _install(fake_services, test_state, _REFINE)
    _configure_windows(test_state, vram_gib=48)
    video_id = _ingest_sized(client, test_state, tmp_path, width=3840, height=2160, frames=137)
    body = _preflight_body(video_id, "2160p")

    desktop = client.post(_PREFLIGHT, json=body)
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as remote:
        phone = remote.post(_PREFLIGHT, json=body, headers=_AUTH)

    assert desktop.status_code == phone.status_code == 200, phone.text
    assert phone.json() == desktop.json()
    cells = {cell["maxWindowFrames"]: cell for cell in phone.json()["maxWindowCells"]}
    assert cells[97]["ok"] is False
    assert all(cells[window]["ok"] for window in (73, 57, 41))


def test_remote_create_refuses_a_maximum_above_the_cap_like_the_desktop_and_keeps_a_valid_one(
    client: TestClient, fake_services, test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files(model_id="ltx-2.5-22b-distilled")
    _install(fake_services, test_state, _REFINE)
    _configure_windows(test_state, vram_gib=48)
    video_id = _ingest_sized(client, test_state, tmp_path, width=3840, height=2160, frames=137)

    desktop = client.post(_CREATE, json=_body(video_id, "2160p", maxWindowFrames=97))
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as remote:
        phone = remote.post(_CREATE, json=_body(video_id, "2160p", maxWindowFrames=97), headers=_AUTH)
        kept = remote.post(_CREATE, json=_body(video_id, "2160p", maxWindowFrames=73), headers=_AUTH)

    assert desktop.status_code == phone.status_code == 422
    assert phone.json()["message"] == desktop.json()["message"]
    # The phone shows the stored maximum, so Reuse restores the same run.
    assert kept.status_code == 200, kept.text
    assert kept.json()["spec"]["params"]["maxWindowFrames"] == 73


def test_remote_auto_carry_matches_the_desktop_for_a_chunk_length_the_default_carry_cannot_run(
    client: TestClient, fake_services, test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files(model_id="ltx-2.5-22b-distilled")
    _install(fake_services, test_state, _REFINE)
    _configure_windows(test_state, vram_gib=24)
    video_id = _ingest_sized(client, test_state, tmp_path, width=1920, height=1080, frames=137)
    body = {**_preflight_body(video_id, "1080p"), "params": {**_body(video_id, "1080p")["params"], "maxWindowFrames": 41}}

    desktop = client.post(_PREFLIGHT, json=body)
    desktop_create = client.post(_CREATE, json=_body(video_id, "1080p", maxWindowFrames=41))
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as remote:
        phone = remote.post(_PREFLIGHT, json=body, headers=_AUTH)
        phone_create = remote.post(_CREATE, json=_body(video_id, "1080p", maxWindowFrames=41), headers=_AUTH)

    assert desktop.status_code == phone.status_code == 200, phone.text
    assert phone.json() == desktop.json()
    # Auto carry runs 17, so the 41-frame chunk runs on both apps. Carry 25 would refuse it.
    assert phone.json()["defaults"]["carryFrames"] == 17
    assert desktop_create.status_code == phone_create.status_code == 200, phone_create.text
