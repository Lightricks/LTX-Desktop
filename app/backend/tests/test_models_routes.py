"""Integration tests for the GET /api/models listing endpoint."""

from __future__ import annotations

import json
import struct
from pathlib import Path

from starlette.testclient import TestClient


def _models_dir(test_state) -> Path:
    return test_state.config.default_models_dir


def _write_ic_lora(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps({"__metadata__": {"reference_downscale_factor": "1"}}).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)


def test_list_models_lora_only(client: TestClient, test_state) -> None:
    models_dir = _models_dir(test_state)
    (models_dir / "loras").mkdir(parents=True, exist_ok=True)
    (models_dir / "loras" / "style.safetensors").write_bytes(b"\0")
    (models_dir / "transformer.safetensors").write_bytes(b"\0")

    resp = client.get("/api/models", params={"type": "lora"})
    assert resp.status_code == 200
    names = [m["name"] for m in resp.json()["models"]]
    assert names == ["style.safetensors"]
    assert resp.json()["models"][0]["is_lora"] is True


def test_list_models_all(client: TestClient, test_state) -> None:
    models_dir = _models_dir(test_state)
    (models_dir / "loras").mkdir(parents=True, exist_ok=True)
    (models_dir / "loras" / "style.safetensors").write_bytes(b"\0")
    (models_dir / "transformer.safetensors").write_bytes(b"\0")

    resp = client.get("/api/models")
    assert resp.status_code == 200
    names = sorted(m["name"] for m in resp.json()["models"])
    assert names == ["style.safetensors", "transformer.safetensors"]


def test_lora_and_ic_lora_filters_split(client: TestClient, test_state) -> None:
    models_dir = _models_dir(test_state)
    (models_dir / "loras").mkdir(parents=True, exist_ok=True)
    (models_dir / "loras" / "style.safetensors").write_bytes(b"\0")  # regular
    _write_ic_lora(models_dir / "loras" / "control-ic-lora.safetensors")  # IC-LoRA

    lora_resp = client.get("/api/models", params={"type": "lora"})
    assert [m["name"] for m in lora_resp.json()["models"]] == ["style.safetensors"]

    ic_resp = client.get("/api/models", params={"type": "ic-lora"})
    ic_models = ic_resp.json()["models"]
    assert [m["name"] for m in ic_models] == ["control-ic-lora.safetensors"]
    assert ic_models[0]["is_ic_lora"] is True
