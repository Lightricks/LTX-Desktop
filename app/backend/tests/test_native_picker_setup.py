"""Radeon picker inventory, version-specific download, activation and hardware grid."""
import json
from pathlib import Path

import pytest
from runtime_config.model_download_specs import resolve_model_path
from tests.fakes.services import fixture_weights
from runtime_config.runtime_policy import decide_local_generation_mode
from services.features.video.text_to_video import fast_job_model_id
from api_types import GenerateVideoRequest

MODEL_23 = "ltx-2.3-22b-distilled-1.1"
MODEL_25 = "ltx-2.5-22b-distilled"
GEMMA = "gemma-3-12b-it-qat-q4_0-unquantized"


def versions(client):
    response = client.get("/api/models/ltx-versions")
    response.raise_for_status()
    return {row["model_id"]: row for row in response.json()["versions"]}


def write_gemma_snapshot(path):
    path.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "processor_config.json", "preprocessor_config.json"):
        (path / name).write_text("{}")
    shards = {"layer1": "model-00001-of-00002.safetensors", "layer2": "model-00002-of-00002.safetensors"}
    for name in shards.values():
        (path / name).write_bytes(fixture_weights())
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": shards, "metadata": {"total_size": 22}}))


def setup_download(client, model, cp_ids=()):
    response = client.post("/api/models/download", json={"type": "download", "activate_model_id": model, "cp_ids": cp_ids})
    response.raise_for_status()
    return response.json()["sessionId"]


def progress(client, session_id):
    return client.get("/api/models/download/progress", params={"sessionId": session_id}).json()


def use_family(client, model):
    return client.post("/api/models/active-ltx-model", json={"model_id": model})


def test_picker_download_repairs_23_and_preserves_25_then_switches_offline(
    client, test_state, create_fake_model_files, fake_services, monkeypatch,
):
    monkeypatch.setenv("LTX_REQUIRE_ROCM", "1")
    create_fake_model_files(model_id=MODEL_25)
    create_fake_model_files(model_id=MODEL_23)
    models_dir = test_state.config.default_models_dir
    gemma = resolve_model_path(models_dir, GEMMA)
    (gemma / "model.safetensors").unlink()
    write_gemma_snapshot(gemma)
    (gemma / "model-00002-of-00002.safetensors").unlink()
    assert versions(client)[MODEL_23]["cps_to_download"] == [GEMMA]
    assert versions(client)[MODEL_25]["active"]
    assert use_family(client, MODEL_23).status_code == 409
    assert not versions(client)[MODEL_23]["active"]

    requested_snapshots = []

    def download_snapshot(**kwargs):
        requested_snapshots.append(kwargs["repo_id"])
        write_gemma_snapshot(Path(kwargs["local_dir"]))

    monkeypatch.setattr(fake_services.model_downloader, "download_snapshot", download_snapshot)
    # A stale client supplies no files. The server computes the exact missing bundle.
    assert progress(client, setup_download(client, MODEL_23))["status"] == "complete"
    assert requested_snapshots == ["Lightricks/gemma-3-12b-it-qat-q4_0-unquantized"]
    assert fake_services.model_downloader.calls == []
    assert versions(client)[MODEL_25]["installed"]
    assert versions(client)[MODEL_23]["active"]
    assert json.loads(test_state.config.settings_file.read_text())["active_ltx_model_id"] == MODEL_23
    assert test_state.state.gpu_slot is None
    req = GenerateVideoRequest(prompt="a tree", amdFastEngine="native")
    assert fast_job_model_id(req, models_dir, "ltx-2.3-fast") == MODEL_23
    assert use_family(client, MODEL_25).status_code == 200
    assert fast_job_model_id(req, models_dir, "ltx-2.5-fast") == MODEL_25
    assert versions(client)[MODEL_25]["active"]
    assert versions(client)[MODEL_23]["installed"]


def test_picker_install_failure_keeps_previously_working_family_and_partial_gemma(
    client, test_state, create_fake_model_files, monkeypatch,
):
    monkeypatch.setenv("LTX_REQUIRE_ROCM", "1")
    create_fake_model_files(model_id=MODEL_25)
    gemma = resolve_model_path(test_state.config.default_models_dir, GEMMA)
    gemma.mkdir()
    (gemma / "README.md").write_text("partial copy")
    # The fake snapshot contains only a weight file, so staged validation must refuse it.
    result = progress(client, setup_download(client, MODEL_23))
    assert result["status"] == "error" and GEMMA in result["error"]
    assert (gemma / "README.md").exists()
    assert versions(client)[MODEL_23]["cps_to_download"] == [GEMMA]
    assert versions(client)[MODEL_25]["active"]


def test_newer_explicit_selection_wins_over_running_picker_download(
    client, test_state, fake_services, create_fake_model_files, monkeypatch,
):
    create_fake_model_files(model_id=MODEL_25)
    pending = []
    monkeypatch.setattr(fake_services.task_runner, "run_background", lambda target, **kwargs: pending.append(target))
    session = setup_download(client, MODEL_23)
    active = client.get("/api/models/download/active").json()
    assert active["activate_model_id"] == MODEL_23
    assert active["session_id"] == session
    assert versions(client)[MODEL_25]["active"]
    assert use_family(client, MODEL_25).status_code == 200
    pending.pop()()
    assert progress(client, session)["status"] == "complete"
    assert versions(client)[MODEL_23]["installed"]
    assert versions(client)[MODEL_25]["active"]


@pytest.mark.parametrize("kind,extra", [("upgrade", []), ("download", ["z-image-turbo"])])
def test_picker_rejects_foreign_download_targets(client, kind, extra):
    response = client.post("/api/models/download", json={"type": kind, "cp_ids": extra, "activate_model_id": MODEL_23})
    assert response.status_code == 400


def test_encoder_requirement_follows_explicit_prompt_setting(client, test_state, create_fake_model_files, monkeypatch):
    monkeypatch.setenv("LTX_REQUIRE_ROCM", "1")
    create_fake_model_files(model_id=MODEL_23)
    test_state.state.app_settings.ltx_api_key = "fixture"
    test_state.state.app_settings.use_local_text_encoder = False
    assert versions(client)[MODEL_23]["installed"]
    test_state.state.app_settings.use_local_text_encoder = True
    assert versions(client)[MODEL_23]["cps_to_download"] == [GEMMA]


def test_gfx1151_bf16_profiles_offer_real_native_cells_both_families(
    client, test_state, create_fake_model_files, monkeypatch,
):
    monkeypatch.setenv("LTX_REQUIRE_ROCM", "1")
    config = test_state.config
    config.system = "Windows"
    config.fp8_capable = False
    config.gpu_device_name = "AMD Radeon(TM) 8060S Graphics"
    config.vram_gb = 64
    config.vram_bytes = config.vram_gb * 1024**3
    config.total_ram_bytes = 128 * 1024**3
    config.local_generations_mode = decide_local_generation_mode(
        system="Windows", cuda_available=True, vram_gb=config.vram_gb, fp8_capable=False,
    )
    assert config.local_generations_mode == "streaming_models_loading"
    for model in (MODEL_25, MODEL_23):
        create_fake_model_files(model_id=model)
    write_gemma_snapshot(resolve_model_path(config.default_models_dir, GEMMA))
    assert all(versions(client)[model]["installed"] for model in (MODEL_25, MODEL_23))
    specs = client.get("/api/generate/models-specs").json()
    rows = {row["model"]: row["spec"] for row in specs["downloaded_local_models"]}
    for offering in ("ltx-2.5-fast", "ltx-2.3-fast"):
        assert 5 in rows[offering]["supported_resolutions_durations"]["540p"]["fps_to_durations"]["24"]
    assert specs["local_unavailable_reason"] is None


def test_partial_native_safetensors_rejected_and_rechecked_on_repair(client, test_state, monkeypatch, create_fake_model_files):
    monkeypatch.setenv("LTX_REQUIRE_ROCM", "1")
    create_fake_model_files(model_id=MODEL_25)
    path = resolve_model_path(test_state.config.default_models_dir, MODEL_25)
    payload = path.read_bytes()
    assert versions(client)[MODEL_25]["installed"]
    path.write_bytes(payload[:-1])
    assert versions(client)[MODEL_25]["cps_to_download"] == [MODEL_25]
    assert use_family(client, MODEL_25).status_code == 409
    assert progress(client, setup_download(client, MODEL_25))["status"] == "complete"
    assert versions(client)[MODEL_25]["active"]


def test_failed_settings_write_keeps_last_active_model(client, test_state, create_fake_model_files):
    for model in (MODEL_25, MODEL_23):
        create_fake_model_files(model_id=model)
    assert use_family(client, MODEL_25).status_code == 200
    test_state.config.settings_file = test_state.config.app_data_dir / "missing-directory" / "settings.json"
    assert use_family(client, MODEL_23).status_code == 500
    assert test_state.state.app_settings.active_ltx_model_id == MODEL_25
    assert versions(client)[MODEL_25]["active"]
