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


def test_lora_list_reports_downloaded_state(client: TestClient):
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
