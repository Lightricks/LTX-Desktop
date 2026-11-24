from pathlib import Path

from fastapi.testclient import TestClient
from runtime_config.model_download_specs import is_lora_downloaded, resolve_lora_path


def test_lora_path_layout(tmp_path: Path):
    p = resolve_lora_path(tmp_path, "cozy-felt-v1", "felt.safetensors")
    assert p == tmp_path / "loras" / "cozy-felt-v1" / "felt.safetensors"
    assert is_lora_downloaded(tmp_path, "cozy-felt-v1", "felt.safetensors") is False
    p.parent.mkdir(parents=True)
    p.write_bytes(b"x")
    assert is_lora_downloaded(tmp_path, "cozy-felt-v1", "felt.safetensors") is True


def test_lora_list_reports_downloaded_state(client: TestClient, create_fake_model_files):
    # Hide-when-unsupported-family needs a runnable LTX on disk; fake catalog rows default to both families.
    create_fake_model_files()
    # FakeLoraCatalogProvider (wired in conftest) returns one plain LoRA "cozy-felt-v1".
    resp = client.get("/api/loras")
    assert resp.status_code == 200
    items = resp.json()["loras"]
    assert items[0]["lora"]["id"] == "cozy-felt-v1"
    # Shared catalog metadata flows through for plain LoRAs too.
    assert items[0]["lora"]["trigger"] == "F3ltCut0u7"
    assert items[0]["lora"]["tags"] == ["style"]
    assert items[0]["downloaded"] is False
    assert items[0]["downloaded_variant_ids"] == []


def test_lora_download_then_progress_complete(client: TestClient):
    start = client.post("/api/loras/download", json={"lora_id": "cozy-felt-v1"})
    assert start.status_code == 200
    session_id = start.json()["sessionId"]
    prog = client.get("/api/loras/download/progress", params={"sessionId": session_id})
    assert prog.status_code == 200
    # FakeModelDownloader + FakeTaskRunner run synchronously, so it is already complete.
    assert prog.json()["status"] == "complete"


def test_lora_delete_installation(client: TestClient, create_fake_model_files):
    create_fake_model_files()
    client.post("/api/loras/download", json={"lora_id": "cozy-felt-v1"})
    listed = client.get("/api/loras")
    assert listed.json()["loras"][0]["downloaded"] is True

    delete = client.request(
        "DELETE",
        "/api/loras/installation",
        json={"lora_id": "cozy-felt-v1"},
    )
    assert delete.status_code == 200

    listed_after = client.get("/api/loras")
    assert listed_after.json()["loras"][0]["downloaded"] is False


def test_lora_download_anonymous_when_not_gated(client: TestClient, fake_services):
    # Plain LoRA is public (requires_hf_login=False) and gating is off → no token attached.
    client.post("/api/loras/download", json={"lora_id": "cozy-felt-v1"})
    assert fake_services.model_downloader.calls[-1]["token"] is None


def test_lora_download_unknown_id_404(client: TestClient):
    resp = client.post("/api/loras/download", json={"lora_id": "does-not-exist"})
    assert resp.status_code == 404


def test_lora_download_unknown_variant_id_404(client: TestClient):
    resp = client.post(
        "/api/loras/download", json={"lora_id": "cozy-felt-v1", "variant_id": "does-not-exist"}
    )
    assert resp.status_code == 404


def test_lora_active_download_null_when_idle(client: TestClient):
    resp = client.get("/api/loras/download/active")
    assert resp.status_code == 200
    assert resp.json()["session_id"] is None
    assert resp.json()["lora_id"] is None
    assert resp.json()["progress"] is None


def test_lora_active_download_reports_running_session(client: TestClient, test_state):
    from state.app_state_types import CatalogDownloadSession, DownloadSessionId

    # Simulate an in-flight download (FakeTaskRunner runs synchronously, so a real start
    # would already be complete) — the session a remounting screen needs to reattach to.
    test_state.state.lora_download_session = CatalogDownloadSession(
        id=DownloadSessionId("lora-dl-1"),
        item_id="cozy-felt-v1",
        downloaded_bytes=40,
        expected_bytes=100,
    )

    resp = client.get("/api/loras/download/active")
    assert resp.status_code == 200
    assert resp.json()["session_id"] == "lora-dl-1"
    assert resp.json()["lora_id"] == "cozy-felt-v1"
    assert resp.json()["progress"] == 40.0


def test_lora_list_empty_when_no_ltx_family_installed(client: TestClient):
    resp = client.get("/api/loras")
    assert resp.status_code == 200
    assert resp.json()["loras"] == []
    assert client.get("/api/loras", params={"fresh": False}).json()["loras"] == []

    fresh = client.get("/api/loras", params={"fresh": True})
    assert fresh.status_code == 200
    ids = [item["lora"]["id"] for item in fresh.json()["loras"]]
    assert "cozy-felt-v1" in ids
    assert "cozy-felt-style" in ids


def test_lora_list_hides_adapter_when_required_family_missing(
    client: TestClient, create_fake_model_files, fake_services
):
    create_fake_model_files(model_id="ltx-2.5-22b-distilled")
    catalog = fake_services.lora_catalog_provider._catalog
    fake_services.lora_catalog_provider._catalog = catalog.model_copy(
        update={
            "loras": [
                catalog.loras[0].model_copy(update={"supported_models": ["LTX-2.3"]}),
                catalog.loras[1],
            ]
        }
    )

    resp = client.get("/api/loras")
    assert resp.status_code == 200
    ids = [item["lora"]["id"] for item in resp.json()["loras"]]
    assert "cozy-felt-v1" not in ids
    assert "cozy-felt-style" in ids

    fresh_ids = [
        item["lora"]["id"]
        for item in client.get("/api/loras", params={"fresh": True}).json()["loras"]
    ]
    assert "cozy-felt-v1" in fresh_ids
    assert "cozy-felt-style" in fresh_ids


def test_lora_list_still_shows_undownloaded_adapter_when_family_installed(
    client: TestClient, create_fake_model_files
):
    create_fake_model_files(model_id="ltx-2.5-22b-distilled")
    resp = client.get("/api/loras")
    items = {item["lora"]["id"]: item for item in resp.json()["loras"]}
    assert items["cozy-felt-style"]["downloaded"] is False
