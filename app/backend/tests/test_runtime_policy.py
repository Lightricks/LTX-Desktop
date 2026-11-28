"""Tests for /api/runtime-policy endpoint."""

from __future__ import annotations

import pytest


def test_runtime_policy_true(client, test_state):
    test_state.config.local_generations_mode = "unsupported"

    response = client.get("/api/runtime-policy")
    assert response.status_code == 200
    assert response.json() == {"amd_renderer_available": False, "force_api_generations": True, "local_viable": False}


def test_runtime_policy_false(client, test_state, windows_local_gpu):
    test_state.config.local_generations_mode = "full_models_loading"

    response = client.get("/api/runtime-policy")
    assert response.status_code == 200
    assert response.json() == {"amd_renderer_available": True, "force_api_generations": False, "local_viable": True}


def test_runtime_policy_stream_is_locally_viable(client, test_state):
    test_state.config.local_generations_mode = "streaming_models_loading"

    response = client.get("/api/runtime-policy")
    assert response.status_code == 200
    assert response.json() == {"amd_renderer_available": False, "force_api_generations": False, "local_viable": True}


def test_runtime_policy_dev_force_local_viable(client, test_state, monkeypatch):
    test_state.config.local_generations_mode = "unsupported"
    test_state.config.dev_mode = True
    monkeypatch.setenv("LTX_DEV_FORCE_LOCAL_VIABLE", "1")

    response = client.get("/api/runtime-policy")
    assert response.status_code == 200
    assert response.json() == {"amd_renderer_available": False, "force_api_generations": True, "local_viable": True}


def test_ineligible_runtime_cannot_set_up_or_submit_a_managed_model(client, test_state, windows_video_host, monkeypatch):
    from runtime_config import community_runtime

    # No native accelerator on the deliberately ineligible POSIX test machines.
    monkeypatch.setattr(community_runtime, "_native_backend", lambda _: None)
    for platform, runtime, mode in (
        ("win32", "api", "full_models_loading"),
        ("win32", "unknown", "full_models_loading"),
        ("win32", "cuda", "unsupported"),
        ("linux", "cuda", "full_models_loading"),
        ("darwin", "rocm", "full_models_loading"),
    ):
        monkeypatch.setattr(community_runtime, "HOST_PLATFORM", platform)
        monkeypatch.setenv("LTX_GPU_RUNTIME", runtime)
        monkeypatch.setenv("LTX_REQUIRE_ROCM", "1")  # The new receipt always takes precedence.
        test_state.config.local_generations_mode = mode
        assert not client.get("/api/runtime-policy").json()["amd_renderer_available"]
        specs = client.get("/api/generate/models-specs").json()
        assert all(specs[key] is None for key in (
            "comfy_local_model", "kandinsky_local_model", "h3_fl2va_local_model",
            "h3_fl2va_distill_local_model", "h3_ref2va_local_model",
        ))
        for suffix in ["download", "kandinsky/download", "h3/license?acceptedLicense=true",
                       "h3/download?variant=h3-fl2va&acceptedLicense=true"]:
            assert client.post("/api/comfy/models/" + suffix).status_code == 404
        assert client.post("/api/amd/comfy/kandinsky-node/install?managedOnly=true").status_code == 404
        req = {"prompt": "rain", "amdFastEngine": "h3-fl2va", "duration": 5, "fps": 24, "resolution": "540p"}
        assert client.post("/api/generate", json={**req, "model": "fast"}).status_code == 422
        assert client.post("/api/generations/text-to-video", json={
            "params": {**req, "model": "ltx-2.5-fast"},
        }).status_code == 422
    assert not (test_state.config.app_data_dir / "h3-license-v1.json").exists()


def test_h3_native_platforms_offer_only_their_supported_models(client, test_state, h3_gpu):
    from runtime_config import community_runtime

    test_state.config.local_generations_mode = "streaming_models_loading"
    assert client.get("/api/runtime-policy").json()["amd_renderer_available"]
    specs = client.get("/api/generate/models-specs").json()
    assert all(specs[k] is not None for k in ("h3_fl2va_local_model", "h3_fl2va_distill_local_model", "h3_ref2va_local_model"))
    if community_runtime.HOST_PLATFORM == "win32":
        assert specs["comfy_local_model"] and specs["kandinsky_local_model"]
        return
    assert specs["comfy_local_model"] is None and specs["kandinsky_local_model"] is None
    assert client.get("/api/settings").json()["amdFastEngine"] == "native"
    for path in ("/api/comfy/models/download", "/api/comfy/models/kandinsky/download", "/api/amd/comfy/kandinsky-node/install"):
        assert client.post(path).status_code == 404
    for engine in ("comfy", "kandinsky"):
        params = {"prompt": "rain", "amdFastEngine": engine, "duration": 5, "fps": 24, "resolution": "540p", "model": "ltx-2.5-fast"}
        assert client.post("/api/generations/text-to-video", json={"params": params}).status_code == 422


@pytest.mark.parametrize("host,chip,mps,cuda_build,cuda_ready,expected", [
    ("darwin", "arm64", True, None, False, "mps"),
    ("darwin", "arm64", False, None, False, None),
    ("darwin", "x86_64", True, None, False, None),
    ("linux", "x86_64", False, "12.8", True, "cuda"),
    ("linux", "x86_64", False, None, True, None),
    ("linux", "x86_64", False, "12.8", False, None),
])
def test_posix_selection_checks_native_torch_device(monkeypatch, host, chip, mps, cuda_build, cuda_ready, expected):
    import torch
    from runtime_config import community_runtime

    monkeypatch.setenv("LTX_GPU_RUNTIME", "rocm")
    monkeypatch.setattr(community_runtime, "HOST_PLATFORM", host)
    monkeypatch.setattr(community_runtime.platform, "machine", lambda: chip)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_ready)
    monkeypatch.setattr(torch.version, "cuda", cuda_build)
    community_runtime._native_backend.cache_clear()
    try:
        assert community_runtime.local_video_backend() == expected
    finally:
        community_runtime._native_backend.cache_clear()
