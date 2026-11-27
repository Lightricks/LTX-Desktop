"""Tests for /api/runtime-policy endpoint."""

from __future__ import annotations


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
