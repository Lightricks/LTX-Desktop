from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from runtime_config.model_download_specs import is_ic_lora_downloaded, resolve_ic_lora_path
from state.app_state_types import HfNotAuthenticated


def test_ic_lora_path_layout(tmp_path: Path):
    p = resolve_ic_lora_path(tmp_path, "ingredients-v1", "i.safetensors")
    assert p == tmp_path / "ic-loras" / "ingredients-v1" / "i.safetensors"
    assert is_ic_lora_downloaded(tmp_path, "ingredients-v1", "i.safetensors") is False
    p.parent.mkdir(parents=True)
    p.write_bytes(b"x")
    assert is_ic_lora_downloaded(tmp_path, "ingredients-v1", "i.safetensors") is True


def test_ic_lora_id_cannot_traverse(tmp_path: Path):
    with pytest.raises(ValueError):
        resolve_ic_lora_path(tmp_path, "../escape", "i.safetensors")


def test_ic_lora_list_reports_downloaded_state(client: TestClient):
    # FakeLoraCatalogProvider (wired in conftest) returns one IC-LoRA "ingredients-v1".
    resp = client.get("/api/ic-loras")
    assert resp.status_code == 200
    items = resp.json()["ic_loras"]
    assert items[0]["ic_lora"]["id"] == "ingredients-v1"
    assert items[0]["downloaded"] is False
    assert items[0]["downloaded_variant_ids"] == []


def test_ic_lora_download_then_progress_complete(client: TestClient):
    start = client.post("/api/ic-loras/download", json={"ic_lora_id": "ingredients-v1"})
    assert start.status_code == 200
    session_id = start.json()["sessionId"]
    prog = client.get("/api/ic-loras/download/progress", params={"sessionId": session_id})
    assert prog.status_code == 200
    # FakeModelDownloader + FakeTaskRunner run synchronously, so it is already complete.
    assert prog.json()["status"] == "complete"


def test_ic_lora_download_gated_item_attaches_token_even_without_flag(client: TestClient, fake_services):
    # A gated IC-LoRA (requires_hf_login) attaches the in-app token at download start even
    # with global gating off and no use_hf_auth — otherwise it would 401 mid-download.
    # (Anonymous downloads are covered for public items in test_lora_download.py.)
    client.post("/api/ic-loras/download", json={"ic_lora_id": "ingredients-v1"})
    assert fake_services.model_downloader.calls[-1]["token"] == "fake-hf-token"


def test_ic_lora_download_attaches_in_app_token_when_flag_on(client: TestClient, fake_services):
    # conftest seeds an authenticated HF state; use_hf_auth opts the download into it.
    client.post("/api/ic-loras/download", json={"ic_lora_id": "ingredients-v1", "use_hf_auth": True})
    assert fake_services.model_downloader.calls[-1]["token"] == "fake-hf-token"


def test_ic_lora_download_public_item_stays_anonymous_when_signed_out(
    client: TestClient, test_state, fake_services
):
    # use_hf_auth is opt-in, not a hard gate: a public (non-gated) item still downloads
    # anonymously when the user isn't signed in — we never force sign-in for a public repo.
    original_auth = test_state.state.hf_auth_state
    test_state.state.hf_auth_state = HfNotAuthenticated()
    resp = client.post("/api/ic-loras/download", json={"ic_lora_id": "refimg-v1", "use_hf_auth": True})
    assert resp.status_code == 200
    assert fake_services.model_downloader.calls[-1]["token"] is None
    test_state.state.hf_auth_state = original_auth


def test_ic_lora_download_public_item_attaches_token_when_signed_in(client: TestClient, fake_services):
    # Signed in (conftest default) + use_hf_auth → the token rides along even for a public item.
    client.post("/api/ic-loras/download", json={"ic_lora_id": "refimg-v1", "use_hf_auth": True})
    assert fake_services.model_downloader.calls[-1]["token"] == "fake-hf-token"


def test_ic_lora_download_flag_requires_auth_and_leaves_no_session(client: TestClient, test_state):
    # Not signed into HF + a gated item → clean 403, and no dangling session blocks the next download.
    original_auth = test_state.state.hf_auth_state
    test_state.state.hf_auth_state = HfNotAuthenticated()
    resp = client.post("/api/ic-loras/download", json={"ic_lora_id": "ingredients-v1", "use_hf_auth": True})
    assert resp.status_code == 403
    assert test_state.state.ic_lora_download_session is None
    # Once signed in again, a subsequent download still starts (would 409 if the session had leaked).
    test_state.state.hf_auth_state = original_auth
    assert client.post("/api/ic-loras/download", json={"ic_lora_id": "ingredients-v1"}).status_code == 200
