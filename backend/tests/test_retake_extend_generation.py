"""Local Retake/Extend routing for LTX 2.3 and 2.5.

The fake pipeline never runs GPU code. These tests pin that each local model
loads the matching checkpoint/VAE paths.
"""

from __future__ import annotations

import logging
import uuid
from types import SimpleNamespace

import pytest

from api_types import ExtendParams, LTXLocalModelId, RetakeParams
from runtime_config.budget_admission import RiskAllowance
from runtime_config.ltx_runtime_paths import resolve_ltx_runtime_paths
from runtime_config.model_download_specs import get_existing_cp_path, get_ltx_model_spec
from runtime_config.video_job_budget import (
    LOCAL_GENERATION_UNSUPPORTED,
    LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
    VIDEO_JOB_TOO_LARGE,
)
from services.features.video.extend import ExtendExecutor
from services.features.video.retake import RetakeExecutor
from services.records import CapabilityFailedError
from state.app_settings import resolved_use_conv_vae
from state.app_state_types import GpuSlot, RetakePipelineState
from tests.http_error_assertions import assert_http_error
from tests.video_fixtures import write_mp4

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


def _expected_create_paths(test_state, model_id: LTXLocalModelId) -> dict[str, object]:
    from ltx_pipelines.utils.types import OffloadMode

    spec = get_ltx_model_spec(model_id)
    gemma_root = str(get_existing_cp_path(test_state.config.default_models_dir, spec.text_encoder_cp))
    paths = resolve_ltx_runtime_paths(
        test_state.config.default_models_dir,
        model_id,
        gemma_root=gemma_root,
        use_conv_vae=resolved_use_conv_vae(test_state.state.app_settings),
    )
    # Default config is full-resident cpu: nothing streams, so offload is NONE.
    return {
        "checkpoint_path": paths.checkpoint_path,
        "gemma_root": paths.gemma_root,
        "offload_mode": OffloadMode.NONE,
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


def test_local_retake_routes_to_model_paths(
    client, test_state, create_fake_model_files, fake_services
) -> None:
    _install_local(test_state, create_fake_model_files, _LOCAL_2_3)
    video_path = _make_valid_video(test_state)

    r = client.post(
        "/api/retake",
        json={"video_path": video_path, "start_time": 1.0, "duration": 3.0, "prompt": "make it dramatic"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "complete"
    assert data["video_path"]
    _assert_create_paths(fake_services, test_state, _LOCAL_2_3)
    _assert_retake_pipeline_model(test_state, _LOCAL_2_3)


def test_local_retake_emits_heartbeat_done(
    client, test_state, create_fake_model_files, fake_services, caplog
) -> None:
    _install_local(test_state, create_fake_model_files, _LOCAL_2_3)
    video_path = _make_valid_video(test_state)
    caplog.set_level(logging.INFO)

    r = client.post(
        "/api/retake",
        json={"video_path": video_path, "start_time": 1.0, "duration": 3.0, "prompt": "make it dramatic"},
    )
    assert r.status_code == 200
    start_records = [
        record
        for record in caplog.records
        if "Retake " in record.getMessage() and " started (" in record.getMessage()
    ]
    assert len(start_records) == 1
    done_records = [
        record
        for record in caplog.records
        if record.name == "server_utils.heartbeat"
        and "retake inference done" in record.getMessage()
    ]
    assert len(done_records) == 1


@pytest.mark.parametrize("mode", ["start", "end"])
def test_local_extend_routes_to_model_paths(
    client, test_state, create_fake_model_files, fake_services, mode: str
) -> None:
    _install_local(test_state, create_fake_model_files, _LOCAL_2_3)
    video_path = _make_valid_video(test_state, frames=9)

    r = client.post(
        "/api/extend",
        json={"video_path": video_path, "duration": 4.0, "prompt": "continue the motion", "mode": mode},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "complete"
    assert data["video_path"]
    _assert_create_paths(fake_services, test_state, _LOCAL_2_3)
    _assert_retake_pipeline_model(test_state, _LOCAL_2_3)
    assert fake_services.retake_pipeline.extend_calls[-1]["mode"] == mode


def test_local_extend_emits_heartbeat_done(
    client, test_state, create_fake_model_files, fake_services, caplog
) -> None:
    _install_local(test_state, create_fake_model_files, _LOCAL_2_3)
    video_path = _make_valid_video(test_state, frames=9)
    caplog.set_level(logging.INFO)

    r = client.post(
        "/api/extend",
        json={"video_path": video_path, "duration": 4.0, "prompt": "continue the motion", "mode": "end"},
    )
    assert r.status_code == 200
    start_records = [
        record
        for record in caplog.records
        if "Extend " in record.getMessage() and " started (" in record.getMessage()
    ]
    assert len(start_records) == 1
    done_records = [
        record
        for record in caplog.records
        if record.name == "server_utils.heartbeat"
        and "extend inference done" in record.getMessage()
    ]
    assert len(done_records) == 1


def test_local_retake_accepted_on_2_5(client, test_state, create_fake_model_files, fake_services) -> None:
    _install_local(test_state, create_fake_model_files, _LOCAL_2_5)
    video_path = _make_valid_video(test_state)

    r = client.post(
        "/api/retake",
        json={"video_path": video_path, "start_time": 1.0, "duration": 3.0, "prompt": "make it dramatic"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "complete"
    assert data["video_path"]
    _assert_create_paths(fake_services, test_state, _LOCAL_2_5)
    _assert_retake_pipeline_model(test_state, _LOCAL_2_5)


@pytest.mark.parametrize("mode", ["start", "end"])
def test_local_extend_accepted_on_2_5(
    client, test_state, create_fake_model_files, fake_services, mode: str
) -> None:
    _install_local(test_state, create_fake_model_files, _LOCAL_2_5)
    video_path = _make_valid_video(test_state, frames=9)

    r = client.post(
        "/api/extend",
        json={"video_path": video_path, "duration": 4.0, "prompt": "continue the motion", "mode": mode},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "complete"
    assert data["video_path"]
    _assert_create_paths(fake_services, test_state, _LOCAL_2_5)
    _assert_retake_pipeline_model(test_state, _LOCAL_2_5)
    assert fake_services.retake_pipeline.extend_calls[-1]["mode"] == mode


def test_retake_unsupported_hardware_without_a_key_is_not_an_api_key_prompt(
    client, test_state, create_fake_model_files, fake_services
) -> None:
    """Unsupported hardware must not be told to configure an API key."""
    _install_local(test_state, create_fake_model_files, _LOCAL_2_3)
    test_state.config.local_generations_mode = "unsupported"
    test_state.state.app_settings.ltx_api_key = ""
    video_path = _make_valid_video(test_state)

    r = client.post(
        "/api/retake",
        json={"video_path": video_path, "start_time": 1.0, "duration": 3.0, "prompt": "make it dramatic"},
    )
    assert_http_error(
        r,
        status_code=422,
        code=LOCAL_GENERATION_UNSUPPORTED,
        message=LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
    )
    body = r.json()["message"].lower()
    assert "api" not in body
    assert "key" not in body
    assert "settings" not in body
    assert fake_services.retake_pipeline.create_calls == []


def test_extend_unsupported_hardware_without_a_key_is_not_an_api_key_prompt(
    client, test_state, create_fake_model_files, fake_services
) -> None:
    _install_local(test_state, create_fake_model_files, _LOCAL_2_3)
    test_state.config.local_generations_mode = "unsupported"
    test_state.state.app_settings.ltx_api_key = ""
    video_path = _make_valid_video(test_state, frames=9)

    r = client.post(
        "/api/extend",
        json={"video_path": video_path, "duration": 4.0, "prompt": "continue the motion", "mode": "end"},
    )
    assert_http_error(
        r,
        status_code=422,
        code=LOCAL_GENERATION_UNSUPPORTED,
        message=LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
    )
    body = r.json()["message"].lower()
    assert "api" not in body
    assert "key" not in body
    assert "settings" not in body
    assert fake_services.retake_pipeline.create_calls == []


def test_retake_unsupported_hardware_with_a_key_still_routes_to_the_api(
    client, test_state, create_fake_model_files, fake_services
) -> None:
    """The guard only fires when no API route is left."""
    _install_local(test_state, create_fake_model_files, _LOCAL_2_3)
    test_state.config.local_generations_mode = "unsupported"
    test_state.state.app_settings.ltx_api_key = "ltx-key"
    video_path = _make_valid_video(test_state)

    r = client.post(
        "/api/retake",
        json={"video_path": video_path, "start_time": 1.0, "duration": 3.0, "prompt": "make it dramatic"},
    )
    assert r.json().get("code") != LOCAL_GENERATION_UNSUPPORTED
    assert fake_services.retake_pipeline.create_calls == []


def test_retake_on_2_5_rebuilds_pipeline_after_2_3(
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


def _prepare_extend(test_state, video_path: str, duration: float):
    """Drive ExtendExecutor.prepare directly: records are namespaces because
    prepare reads generation.id and video.path only (verified by reading)."""
    executor = ExtendExecutor(
        pipelines=test_state.pipelines,
        assets=None,  # type: ignore[arg-type]
        text=None,  # type: ignore[arg-type]
        models_dir=lambda: test_state.config.default_models_dir,
        default_negative_prompt="",
        generation=None,  # type: ignore[arg-type]
        prompt_enhancement=None,  # type: ignore[arg-type]
    )
    return executor.prepare(
        SimpleNamespace(id="gen-1"),
        ExtendParams(prompt="x", model="ltx-2.3-fast", duration=duration, mode="end"),
        SimpleNamespace(path=video_path),
        _LOCAL_2_3,
        42,
        test_state.config.outputs_dir / "out.mp4",
        str(test_state.config.outputs_dir / "windowed.mp4"),
        allowance=RiskAllowance.none(),
    )


def test_queued_extend_over_min_context_refuses_with_max_seconds(
    test_state, create_fake_model_files, fake_services
) -> None:
    """505 cap: 17s (408f) leaves <121 context → refuse naming 16s max."""
    _install_local(test_state, create_fake_model_files, _LOCAL_2_3)
    video_path = _make_valid_video(test_state, frames=600, fps=24)
    with pytest.raises(CapabilityFailedError) as exc_info:
        _prepare_extend(test_state, video_path, duration=17.0)
    assert exc_info.value.code == VIDEO_JOB_TOO_LARGE
    assert "(at most 16s)" in str(exc_info.value)
    assert "on this machine" not in str(exc_info.value)


def test_queued_extend_at_max_keeps_full_context(
    test_state, create_fake_model_files, fake_services
) -> None:
    """16s (384f) is exactly cap-121: allowed, context intact at 121."""
    _install_local(test_state, create_fake_model_files, _LOCAL_2_3)
    video_path = _make_valid_video(test_state, frames=600, fps=24)
    prepared = _prepare_extend(test_state, video_path, duration=16.0)
    assert prepared.window.context_frames == 121


@pytest.mark.parametrize("feature", ["retake", "extend"])
def test_queued_home_edit_loads_the_text_encoder_of_its_own_model(
    feature, test_state, create_fake_model_files, fake_services, tmp_path
) -> None:
    # Queued Home jobs carry their own checkpoint while Settings stays put. A 2.3 job under a
    # 2.5 Settings must get 2.3's Gemma 3 encoder, not 2.5's encode-only gemma4_unified one.
    create_fake_model_files(model_id=_LOCAL_2_3)
    create_fake_model_files(model_id=_LOCAL_2_5)
    test_state.state.app_settings.use_local_text_encoder = True
    test_state.state.app_settings.active_ltx_model_id = _LOCAL_2_5
    spec = get_ltx_model_spec(_LOCAL_2_3)
    expected_gemma_root = str(
        get_existing_cp_path(test_state.config.default_models_dir, spec.text_encoder_cp)
    )

    executor_class, params = {
        "retake": (RetakeExecutor, RetakeParams(model="ltx-2.3-fast", startTime=0, duration=2)),
        "extend": (ExtendExecutor, ExtendParams(model="ltx-2.3-fast", duration=4)),
    }[feature]
    executor = executor_class(
        test_state.pipelines,
        None,  # type: ignore[arg-type] - prepare() never touches the asset lookup
        test_state.text,
        lambda: test_state.video_generation.models_dir,
        "negative",
        None,  # type: ignore[arg-type] - prepare() never touches the generation slot
        None,  # type: ignore[arg-type] - prepare() never enhances the prompt
    )
    video = SimpleNamespace(path=str(write_mp4(tmp_path / "source.mp4", seconds=6)))

    executor.prepare(
        SimpleNamespace(id="gen-1"),  # type: ignore[arg-type]
        params,  # type: ignore[arg-type]
        video,  # type: ignore[arg-type]
        _LOCAL_2_3,
        1,
        tmp_path / "out.mp4",
        str(tmp_path / "out.window.mp4"),
        allowance=RiskAllowance.none(),
    )

    assert fake_services.retake_pipeline.create_calls[-1]["gemma_root"] == expected_gemma_root
