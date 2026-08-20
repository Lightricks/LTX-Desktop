"""Local Retake/Extend routing for LTX 2.3 and 2.5.

The fake pipeline never runs GPU code. These tests pin that each model id
loads the matching checkpoint/VAE paths and that a model switch rebuilds
the cached Retake pipeline.
"""

from __future__ import annotations

import uuid

import pytest

from api_types import LTXLocalModelId
from runtime_config.ltx_runtime_paths import resolve_ltx_runtime_paths
from runtime_config.model_download_specs import get_existing_cp_path, get_ltx_model_spec
from state.app_settings import resolved_use_conv_vae
from state.app_state_types import GpuSlot, RetakePipelineState

_LOCAL_2_3: LTXLocalModelId = "ltx-2.3-22b-distilled"
_LOCAL_2_5: LTXLocalModelId = "ltx-2.5-22b-distilled"


def _make_valid_video(
    test_state,
    *,
    frames: int = 73,
    width: int = 64,
    height: int = 64,
    fps: int = 24,
) -> str:
    import imageio.v2 as imageio
    import numpy as np

    video_file = test_state.config.outputs_dir / f"retake_extend_{uuid.uuid4().hex[:6]}.mp4"
    writer = imageio.get_writer(str(video_file), fps=fps, codec="libx264", macro_block_size=None)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(frames):
        writer.append_data(frame)
    writer.close()
    return str(video_file)


def _install_local(test_state, create_fake_model_files, model_id: LTXLocalModelId) -> None:
    create_fake_model_files(model_id=model_id)
    test_state.state.app_settings.active_ltx_model_id = model_id
    test_state.state.app_settings.use_local_text_encoder = True


def _expected_create_paths(test_state, model_id: LTXLocalModelId) -> dict[str, str | None]:
    spec = get_ltx_model_spec(model_id)
    gemma_root = str(get_existing_cp_path(test_state.config.default_models_dir, spec.text_encoder_cp))
    paths = resolve_ltx_runtime_paths(
        test_state.config.default_models_dir,
        model_id,
        gemma_root=gemma_root,
        use_conv_vae=resolved_use_conv_vae(test_state.state.app_settings),
    )
    return {
        "checkpoint_path": paths.checkpoint_path,
        "gemma_root": paths.gemma_root,
        "video_vae_path": paths.video_vae_path,
        "audio_vae_path": paths.audio_vae_path,
        "duration_head_path": paths.duration_head_path,
    }


def _assert_create_paths(fake_services, test_state, model_id: LTXLocalModelId) -> None:
    assert fake_services.retake_pipeline.create_calls[-1] == _expected_create_paths(test_state, model_id)


def _assert_retake_pipeline_model(test_state, model_id: LTXLocalModelId) -> None:
    slot = test_state.state.gpu_slot
    assert isinstance(slot, GpuSlot)
    assert isinstance(slot.active_pipeline, RetakePipelineState)
    assert slot.active_pipeline.ltx_model_id == model_id


@pytest.mark.parametrize("model_id", [_LOCAL_2_3, _LOCAL_2_5])
def test_local_retake_routes_to_model_paths(
    client, test_state, create_fake_model_files, fake_services, model_id: LTXLocalModelId
) -> None:
    _install_local(test_state, create_fake_model_files, model_id)
    video_path = _make_valid_video(test_state)

    r = client.post(
        "/api/retake",
        json={"video_path": video_path, "start_time": 1.0, "duration": 3.0, "prompt": "make it dramatic"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "complete"
    assert data["video_path"]
    _assert_create_paths(fake_services, test_state, model_id)
    _assert_retake_pipeline_model(test_state, model_id)


@pytest.mark.parametrize("model_id", [_LOCAL_2_3, _LOCAL_2_5])
@pytest.mark.parametrize("mode", ["start", "end"])
def test_local_extend_routes_to_model_paths(
    client, test_state, create_fake_model_files, fake_services, model_id: LTXLocalModelId, mode: str
) -> None:
    _install_local(test_state, create_fake_model_files, model_id)
    video_path = _make_valid_video(test_state, frames=9)

    r = client.post(
        "/api/extend",
        json={"video_path": video_path, "duration": 4.0, "prompt": "continue the motion", "mode": mode},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "complete"
    assert data["video_path"]
    _assert_create_paths(fake_services, test_state, model_id)
    _assert_retake_pipeline_model(test_state, model_id)
    assert fake_services.retake_pipeline.extend_calls[-1]["mode"] == mode


def test_retake_pipeline_rebuilds_when_switching_2_3_to_2_5(
    client, test_state, create_fake_model_files, fake_services
) -> None:
    create_fake_model_files(model_id=_LOCAL_2_3)
    create_fake_model_files(model_id=_LOCAL_2_5)
    test_state.state.app_settings.use_local_text_encoder = True
    test_state.state.app_settings.active_ltx_model_id = _LOCAL_2_3

    video_path = _make_valid_video(test_state)
    payload = {"video_path": video_path, "start_time": 1.0, "duration": 3.0, "prompt": "make it dramatic"}

    first = client.post("/api/retake", json=payload)
    assert first.status_code == 200
    _assert_create_paths(fake_services, test_state, _LOCAL_2_3)
    _assert_retake_pipeline_model(test_state, _LOCAL_2_3)

    test_state.state.app_settings.active_ltx_model_id = _LOCAL_2_5
    second = client.post("/api/retake", json=payload)
    assert second.status_code == 200
    _assert_create_paths(fake_services, test_state, _LOCAL_2_5)
    _assert_retake_pipeline_model(test_state, _LOCAL_2_5)
    assert len(fake_services.retake_pipeline.create_calls) == 2
    assert (
        fake_services.retake_pipeline.create_calls[0]["checkpoint_path"]
        != fake_services.retake_pipeline.create_calls[1]["checkpoint_path"]
    )
