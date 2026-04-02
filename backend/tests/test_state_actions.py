"""State action and invariant tests for AppState."""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime_config.model_download_specs import resolve_model_path
from state.app_settings import UpdateSettingsRequest
from state.app_state_types import (
    CpuSlot,
    GpuSlot,
    ICLoraState,
    RetakePipelineState,
    StartupError,
    StartupLoading,
    StartupPending,
    StartupReady,
    VideoPipelineState,
    VideoPipelineWarmth,
)


def _model_path(test_state, model_type: str) -> Path:
    return resolve_model_path(
        test_state.config.default_models_dir,
        test_state.config.model_download_specs,
        model_type,
    )


def test_start_generation_requires_gpu(test_state):
    with pytest.raises(RuntimeError, match="No active GPU pipeline"):
        test_state.generation.start_generation("gen-1")


def test_generation_mutex_prevents_second_start(test_state):
    test_state.pipelines.load_gpu_pipeline("fast")
    test_state.generation.start_generation("gen-1")

    with pytest.raises(RuntimeError, match="Generation already in progress"):
        test_state.generation.start_generation("gen-2")


def test_cancel_marks_running_generation(test_state):
    test_state.pipelines.load_gpu_pipeline("fast")
    test_state.generation.start_generation("gen-1")

    out = test_state.generation.cancel_generation()
    assert out.status == "cancelling"
    assert out.id == "gen-1"


def test_zit_slot_invariant_enforced(test_state, fake_services):
    zit = fake_services.image_generation_pipeline
    test_state.state.gpu_slot = GpuSlot(active_pipeline=zit)
    test_state.state.cpu_slot = CpuSlot(active_pipeline=zit)

    with test_state._lock:  # noqa: SLF001 - explicit invariant check in tests
        with pytest.raises(RuntimeError, match="Invariant violation"):
            test_state.pipelines._assert_invariants()  # noqa: SLF001


def test_download_terminal_state_is_sticky_until_next_session(test_state):
    session_id = test_state.downloads.start_download({"checkpoint"})
    test_state.downloads.start_file("checkpoint", "checkpoint")
    test_state.downloads.finish_download()

    progress = test_state.downloads.get_download_progress(session_id)
    assert progress.status == "complete"


def test_generation_progress_resets_when_pipeline_unset(test_state):
    test_state.pipelines.load_gpu_pipeline("fast")
    test_state.generation.start_generation("gen-1")
    test_state.generation.complete_generation("/tmp/out.mp4")
    test_state.state.gpu_slot = None

    progress = test_state.generation.get_generation_progress()
    assert progress.status == "idle"


def test_api_generation_does_not_require_gpu(test_state):
    test_state.generation.start_api_generation("api-gen-1")
    test_state.generation.update_progress("inference", 25, 1, 4)

    progress = test_state.generation.get_generation_progress()
    assert progress.status == "running"
    assert progress.phase == "inference"
    assert progress.progress == 25


def test_cancel_marks_running_api_generation(test_state):
    test_state.generation.start_api_generation("api-gen-1")

    out = test_state.generation.cancel_generation()
    assert out.status == "cancelling"
    assert out.id == "api-gen-1"


def test_gpu_cancel_state_not_affected_by_stale_api_cancel_state(test_state):
    test_state.generation.start_api_generation("api-gen-1")
    test_state.generation.cancel_generation()

    test_state.pipelines.load_gpu_pipeline("fast")
    test_state.generation.start_generation("gpu-gen-1")

    assert test_state.generation.is_generation_cancelled() is False


def test_startup_state_transitions_are_tracked(test_state):
    test_state.health.set_startup_pending("waiting")
    assert isinstance(test_state.state.startup, StartupPending)

    test_state.health.set_startup_loading("warming", 60)
    assert isinstance(test_state.state.startup, StartupLoading)

    test_state.health.set_startup_ready()
    assert isinstance(test_state.state.startup, StartupReady)

    test_state.health.set_startup_error("boom")
    assert isinstance(test_state.state.startup, StartupError)


def test_handler_attributes_are_wired(test_state):
    assert test_state.settings is not None
    assert test_state.models is not None
    assert test_state.downloads is not None
    assert test_state.text is not None
    assert test_state.pipelines is not None
    assert test_state.generation is not None
    assert test_state.video_generation is not None
    assert test_state.image_generation is not None
    assert test_state.health is not None
    assert test_state.suggest_gap_prompt is not None
    assert test_state.retake is not None
    assert test_state.ic_lora is not None


def test_rlock_allows_nested_handler_calls(test_state):
    test_state.settings.update_settings(UpdateSettingsRequest(useTorchCompile=True))
    assert test_state.state.app_settings.use_torch_compile is True


def test_warmup_marks_pipeline_warm_and_leaves_no_temp_artifact(test_state):
    out = test_state.pipelines.load_gpu_pipeline("fast", should_warm=True)
    expected_path = test_state.config.outputs_dir / "_warmup_fast.mp4"

    assert out.warmth.value == "warm"
    assert not expected_path.exists()


def test_mps_skips_torch_compile(test_state, fake_services):
    test_state.state.app_settings.use_torch_compile = True
    test_state.pipelines._device = "mps"  # noqa: SLF001 - explicit platform behavior assertion
    test_state.pipelines._runtime_device = "mps"  # noqa: SLF001 - explicit platform behavior assertion

    pipeline_state = test_state.pipelines.load_gpu_pipeline("fast")
    assert fake_services.fast_video_pipeline.compile_calls == 0
    assert pipeline_state.is_compiled is False


def test_startup_warmup_keeps_fast_on_gpu_and_preloads_zit_on_cpu(test_state, fake_services, create_fake_model_files):
    create_fake_model_files(include_zit=True)
    test_state.state.app_settings.load_on_startup = True

    test_state.health.default_warmup()

    assert isinstance(test_state.state.gpu_slot, GpuSlot)
    active = test_state.state.gpu_slot.active_pipeline
    assert isinstance(active, VideoPipelineState)
    assert active.pipeline.pipeline_kind == "fast"
    assert active.warmth == VideoPipelineWarmth.WARM

    assert isinstance(test_state.state.cpu_slot, CpuSlot)
    assert test_state.state.cpu_slot.active_pipeline is fake_services.image_generation_pipeline
    assert fake_services.image_generation_pipeline.device is None


def test_forced_mode_warmup_skips_fast_pipeline(test_state):
    test_state.config.force_api_generations = True
    test_state.config.required_model_types = frozenset()
    test_state.state.app_settings.load_on_startup = True
    test_state.state.app_settings.ltx_api_key = "api-key"

    test_state.health.default_warmup()

    assert test_state.state.gpu_slot is None
    assert test_state.state.cpu_slot is None


def test_retake_pipeline_eviction(test_state):
    test_state.pipelines.load_gpu_pipeline("fast")

    retake_state = test_state.pipelines.load_retake_pipeline(distilled=True)
    assert isinstance(test_state.state.gpu_slot, GpuSlot)
    assert isinstance(test_state.state.gpu_slot.active_pipeline, RetakePipelineState)
    assert test_state.state.gpu_slot.active_pipeline is retake_state

    test_state.pipelines.load_gpu_pipeline("fast")
    assert isinstance(test_state.state.gpu_slot.active_pipeline, VideoPipelineState)


def test_ic_lora_load_includes_depth_resources(test_state, fake_services):
    lora_path = str(_model_path(test_state,"ic_lora"))
    depth_path = str(_model_path(test_state,"depth_processor"))

    ic_state = test_state.pipelines.load_ic_lora(lora_path, depth_path)

    assert isinstance(ic_state, ICLoraState)
    assert ic_state.pipeline is fake_services.ic_lora_pipeline
    assert ic_state.depth_pipeline is fake_services.depth_processor_pipeline
    assert ic_state.lora_path == lora_path
    assert ic_state.depth_model_path == depth_path


def test_ic_lora_unload_clears_preprocessing_resources(test_state):
    lora_path = str(_model_path(test_state,"ic_lora"))
    depth_path = str(_model_path(test_state,"depth_processor"))
    test_state.pipelines.load_ic_lora(lora_path, depth_path)

    assert isinstance(test_state.state.gpu_slot, GpuSlot)
    assert isinstance(test_state.state.gpu_slot.active_pipeline, ICLoraState)

    test_state.pipelines.unload_gpu_pipeline()

    assert test_state.state.gpu_slot is None
