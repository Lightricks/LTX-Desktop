"""Comfy-only installations resolve/download weights without a native transformer."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from handlers.comfy_models_handler import ComfyFolderRegistry, ComfyModelsHandler
from services.amd_comfy_models import KANDINSKY_MODELS, LITE_MODEL, ComfyWeight
from tests.fakes.services import FakeModelDownloader, FakeTaskRunner, fixture_weights


def _registry(
    folders: dict[str, list[str]], *, nodes: bool = True
) -> ComfyFolderRegistry:
    return ComfyFolderRegistry(folders, "fixture", lambda: nodes)


def _weight(name: str, contents: bytes) -> ComfyWeight:
    return ComfyWeight("vae", name, len(contents), hashlib.sha256(contents).hexdigest())


def test_existing_nested_model_and_native_vae_are_reused_and_missing_file_published(
    tmp_path: Path,
) -> None:
    existing = b"already configured"
    vae = b"native vae"
    downloaded = fixture_weights()  # FakeModelDownloader's on-disk payload.
    model_dir = tmp_path / "comfy" / "models" / "vae"
    (model_dir / "nested").mkdir(parents=True)
    (model_dir / "nested" / "existing.safetensors").write_bytes(existing)
    native_dir = tmp_path / "native" / "ltx-2.5"
    native_dir.mkdir(parents=True)
    (native_dir / "native.safetensors").write_bytes(vae)
    weights = [
        _weight("existing.safetensors", existing),
        _weight("native.safetensors", vae),
        _weight("network.safetensors", downloaded),
    ]
    setup = ComfyModelsHandler(
        lambda: _registry({"vae": [str(model_dir)]}),
        lambda: "test-token",
        lambda: native_dir.parent,
        FakeModelDownloader(),
        FakeTaskRunner(),
        weights,
    )
    assert setup.status().completedFiles == 1
    result = setup.start()
    assert result.status == "ready"
    assert (
        result.completedBytes == result.totalBytes == len(existing + vae + downloaded)
    )
    assert (model_dir / "network.safetensors").read_bytes() == downloaded
    assert (model_dir / "native.safetensors").read_bytes() == vae
    assert not (model_dir / "existing.safetensors").exists()
    assert setup.start().status == "ready"


def test_invalid_download_preserves_other_files_and_can_retry(tmp_path: Path) -> None:
    target = tmp_path / "models" / "vae"
    target.mkdir(parents=True)
    weights = [
        _weight("valid.safetensors", fixture_weights()),
        _weight("broken.safetensors", b"other"),
    ]
    setup = ComfyModelsHandler(
        lambda: _registry({"vae": [str(target)]}),
        lambda: "test-token",
        lambda: tmp_path,
        FakeModelDownloader(),
        FakeTaskRunner(),
        weights,
    )
    error = setup.start()
    assert error.status == "error" and "SHA-256" in error.message
    assert not (target / "broken.safetensors").exists()
    assert (target / "valid.safetensors").read_bytes() == fixture_weights()
    (target / "broken.safetensors").write_bytes(b"other")
    assert setup.start().status == "ready"


def test_wrong_size_existing_file_is_preserved_and_auth_reports_gate(
    tmp_path: Path,
) -> None:
    target = tmp_path / "models" / "vae"
    target.mkdir(parents=True)
    model = target / "model.safetensors"
    model.write_bytes(b"user file")
    weight = _weight(model.name, fixture_weights())
    setup = ComfyModelsHandler(
        lambda: _registry({"vae": [str(target)]}),
        lambda: None,
        lambda: tmp_path,
        FakeModelDownloader(),
        FakeTaskRunner(),
        [weight],
    )
    assert setup.start().status == "error"
    assert model.read_bytes() == b"user file"
    model.unlink()
    error = setup.start()
    assert (
        "accept the model license" in error.message and "Hugging Face" in error.message
    )


def test_comfy_specs_exist_without_any_native_download(client, monkeypatch) -> None:
    monkeypatch.setenv("LTX_REQUIRE_ROCM", "1")
    response = client.get("/api/generate/models-specs")
    assert response.status_code == 200
    body = response.json()
    assert body["downloaded_local_models"] == []
    fast = body["comfy_local_model"]["spec"]
    assert not fast["capabilities"]["a2v"]
    assert set(fast["supported_resolutions_durations"]) == {
        "270p",
        "360p",
        "540p",
        "720p",
        "1080p",
    }
    assert (
        20
        not in fast["supported_resolutions_durations"]["540p"]["fps_to_durations"]["50"]
    )
    lite = body["kandinsky_local_model"]["spec"]["supported_resolutions_durations"][
        "540p"
    ]
    assert lite == {"fps_to_durations": {"24": [5]}, "aspect_ratios": ["16:9", "9:16"]}
    monkeypatch.delenv("LTX_REQUIRE_ROCM")
    assert client.get("/api/generate/models-specs").json()["comfy_local_model"] is None


def test_comfy_setup_routes_do_not_request_any_native_checkpoints(
    client, test_state, tmp_path: Path
) -> None:
    model_dir = tmp_path / "models" / "vae"
    weight = _weight("audio.safetensors", fixture_weights())
    test_state.comfy_models = ComfyModelsHandler(
        lambda: _registry({"vae": [str(model_dir)]}),
        lambda: "test-token",
        lambda: tmp_path / "ltx",
        FakeModelDownloader(),
        FakeTaskRunner(),
        [weight],
    )
    assert client.get("/api/comfy/models").json()["status"] == "missing"
    assert client.post("/api/comfy/models/download").json()["status"] == "ready"
    body = client.get("/api/comfy/models").json()
    assert body["models"][0]["path"] == str(model_dir / weight.filename)
    assert body["completedBytes"] == weight.size


def _test_kandinsky_weights() -> list[ComfyWeight]:
    payload = bytes(1024)
    return [
        replace(w, size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        for w in KANDINSKY_MODELS
    ]


def _comfy_dirs(tmp_path: Path) -> dict[str, list[str]]:
    return {
        category: [str(tmp_path / category)]
        for category in (
            "vae",
            "diffusion_models",
            "text_encoders",
            "latent_upscale_models",
        )
    }


def test_kandinsky_download_uses_public_pins_and_reuses_nested_encoder_before_node_install(
    client, test_state, tmp_path
):
    files = _test_kandinsky_weights()
    folders = _comfy_dirs(tmp_path)
    pro = (
        tmp_path
        / "diffusion_models"
        / LITE_MODEL.replace("Lite-distill", "Pro-distill")
    )
    pro.parent.mkdir(parents=True)
    pro.write_bytes(bytes(1024))
    existing = tmp_path / "text_encoders" / "my-library" / files[1].filename
    existing.parent.mkdir(parents=True)
    existing.write_bytes(bytes(1024))
    downloader = FakeModelDownloader()
    test_state.comfy_models = ComfyModelsHandler(
        lambda: _registry(folders),
        lambda: None,
        lambda: tmp_path / "native",
        downloader,
        FakeTaskRunner(),
        kandinsky_weights=files,
    )
    initial = client.get("/api/comfy/models").json()
    assert (
        initial["kandinskyStatus"] == "missing"
        and not initial["kandinskyNodesInstalled"]
    )
    assert (
        initial["kandinskyCompletedFiles"] == 1
        and "restart" in initial["kandinskyMessage"]
    )
    result = client.post("/api/comfy/models/kandinsky/download").json()
    assert result["kandinskyWeightsReady"] and not result["kandinskyReady"]
    assert result["kandinskyCompletedFiles"] == result["kandinskyTotalFiles"] == 7
    assert (
        result["kandinskyTotalBytes"]
        == result["kandinskyCompletedBytes"]
        == sum(w.size for w in files)
    )
    assert (
        result["status"] == "missing"
    )  # No native or Comfy LTX weights were downloaded.
    assert [
        (c["repo_id"], c["filename"], c["revision"], c["token"])
        for c in downloader.calls
    ] == [(w.repo, w.remote_path, w.revision, None) for w in files if w != files[1]]
    for w in files:
        target = existing if w == files[1] else tmp_path / w.category / w.filename
        assert hashlib.sha256(target.read_bytes()).hexdigest() == w.sha256
    assert (
        pro.exists() and not (tmp_path / "text_encoders" / files[1].filename).exists()
    )
    folders["kandinsky6_audio_vae"] = [str(tmp_path / "audio_vae")]
    ready = client.get("/api/comfy/models?fresh=true").json()
    assert (
        ready["kandinskyNodesInstalled"]
        and ready["kandinskyReady"]
        and ready["kandinskyStatus"] == "ready"
    )
    client.post("/api/comfy/models/kandinsky/download").raise_for_status()
    assert len(downloader.calls) == 6


def test_bad_kandinsky_transfer_errors_only_its_recipe_and_retries_without_overwrite(
    tmp_path,
    monkeypatch,
):
    dirs = _comfy_dirs(tmp_path)
    weight = replace(
        KANDINSKY_MODELS[-1], size=4, sha256=hashlib.sha256(b"good").hexdigest()
    )
    setup = ComfyModelsHandler(
        lambda: _registry(dirs),
        lambda: None,
        lambda: tmp_path,
        FakeModelDownloader(),
        FakeTaskRunner(),
        kandinsky_weights=[weight],
    )
    result = setup.start(kandinsky=True)
    assert result.kandinskyStatus == "error" and "SHA-256" in result.kandinskyMessage
    assert result.status == "missing"
    path = tmp_path / weight.category / weight.filename
    assert not path.exists()
    path.write_bytes(b"user's existing file")
    result = setup.start(kandinsky=True)
    assert result.kandinskyStatus == "error"
    assert (
        "Move it aside" in result.kandinskyMessage
        and path.read_bytes() == b"user's existing file"
    )
    from handlers import comfy_models_handler as handler

    monkeypatch.setattr(
        handler, "_BIGVGAN_CONFIG", hashlib.sha256(b'{"fixture":true}').hexdigest()
    )
    path.write_text('{ "fixture" : true }')
    result = setup.start(kandinsky=True)
    assert result.kandinskyWeightsReady and result.kandinskyStatus == "missing"


def test_recipe_jobs_have_separate_progress_and_deduplicate_repeated_requests(tmp_path):
    dirs = _comfy_dirs(tmp_path)
    weight = _weight("ltx.safetensors", fixture_weights())
    k = _test_kandinsky_weights()[0]
    pending = []

    class DeferredTasks:
        def run_background(self, target, **_kwargs):
            pending.append(target)

    setup = ComfyModelsHandler(
        lambda: _registry(dirs),
        lambda: "token",
        lambda: tmp_path,
        FakeModelDownloader(),
        DeferredTasks(),
        weights=[weight],
        kandinsky_weights=[k],
    )
    assert setup.start().status == "downloading"
    both = setup.start(kandinsky=True)
    assert both.status == both.kandinskyStatus == "downloading"
    setup.start(kandinsky=True)
    assert len(pending) == 2
    pending.pop(1)()
    midway = setup.status()
    assert midway.status == "downloading" and midway.kandinskyWeightsReady
    pending.pop()()
    assert setup.status().status == "ready"


def test_registered_audio_directory_with_missing_nodes_never_enables_picker(tmp_path):
    folders = _comfy_dirs(tmp_path)
    folders["kandinsky6_audio_vae"] = [str(tmp_path / "audio_vae")]
    loaded = False
    setup = ComfyModelsHandler(
        lambda: _registry(folders, nodes=loaded),
        lambda: None,
        lambda: tmp_path,
        FakeModelDownloader(),
        FakeTaskRunner(),
        kandinsky_weights=[],
    )
    incomplete = setup.status()
    assert incomplete.kandinskyWeightsReady and not incomplete.kandinskyNodesInstalled
    assert incomplete.kandinskyStatus == "missing" and not incomplete.kandinskyReady
    loaded = True
    assert setup.status(fresh=True).kandinskyReady


def test_reuses_equivalent_safetensor_layout_and_rejects_incomplete_export(
    tmp_path, monkeypatch
):
    import numpy as np
    from safetensors.numpy import save_file
    from handlers import comfy_models_handler as handler

    weight = KANDINSKY_MODELS[2]
    dirs = _comfy_dirs(tmp_path)
    file = tmp_path / weight.category / "shared" / weight.filename
    file.parent.mkdir(parents=True)
    tensors = {
        "output": np.zeros(3, dtype=np.float16),
        "bias": np.zeros(2, dtype=np.float16),
    }
    shape_hash = hashlib.sha256(b'[["bias",[2]],["output",[3]]]').hexdigest()
    monkeypatch.setitem(handler._KANDINSKY_LAYOUTS, weight.filename, shape_hash)
    save_file(tensors, str(file), metadata={"source": "repacked"})
    setup = ComfyModelsHandler(
        lambda: _registry(dirs),
        lambda: None,
        lambda: tmp_path,
        FakeModelDownloader(),
        FakeTaskRunner(),
        kandinsky_weights=[weight],
    )
    assert setup.status().kandinskyWeightsReady
    del tensors["output"]
    save_file(tensors, str(file))
    assert not setup.status(fresh=True).kandinskyWeightsReady
