"""State action and invariant tests for AppState."""

from __future__ import annotations

import threading
import time

import pytest

from _routes._errors import HTTPError
from handlers.generation_handler import _RESERVATION_TIMEOUT_S
from runtime_config.model_download_specs import (
    DEPTH_PROCESSOR_CP_ID,
    get_latest_ltx_model_id,
    get_ltx_model_spec,
    resolve_model_path,
)
from state.app_settings import UpdateSettingsRequest
from state.app_state_types import CpuSlot, GpuSlot, ICLoraState, RetakePipelineState, VideoPipelineState


def _current_model_spec():
    return get_ltx_model_spec(get_latest_ltx_model_id())


def test_start_generation_requires_gpu(test_state):
    with pytest.raises(RuntimeError, match="No active GPU pipeline"):
        test_state.generation.start_generation("gen-1")


def test_generation_mutex_prevents_second_start(test_state, create_fake_model_files):
    create_fake_model_files()
    test_state.pipelines.load_gpu_pipeline("fast")
    test_state.generation.start_generation("gen-1")

    with pytest.raises(RuntimeError, match="Generation already in progress"):
        test_state.generation.start_generation("gen-2")


def test_try_reserve_generation_start_blocks_second_reservation(test_state):
    assert test_state.generation.try_reserve_generation_start() is True
    assert test_state.generation.try_reserve_generation_start() is False


def test_try_reserve_generation_start_blocked_while_generation_running(
    test_state, create_fake_model_files,
):
    create_fake_model_files()
    test_state.pipelines.load_gpu_pipeline("fast")
    test_state.generation.start_generation("gen-1")

    assert test_state.generation.try_reserve_generation_start() is False


def test_try_reserve_generation_start_available_again_after_release(test_state):
    assert test_state.generation.try_reserve_generation_start() is True
    test_state.generation.release_generation_start_reservation()

    assert test_state.generation.try_reserve_generation_start() is True


def test_try_reserve_generation_start_stale_reservation_expires(test_state):
    # A path that raises before ever reaching start_generation()/fail_generation() (see
    # try_reserve_generation_start's own docstring) leaves the reservation stuck - simulate that
    # by backdating the timestamp past the timeout instead of actually sleeping in the test.
    assert test_state.generation.try_reserve_generation_start() is True
    test_state.state.generation_starting_since = time.monotonic() - _RESERVATION_TIMEOUT_S - 1

    assert test_state.generation.try_reserve_generation_start() is True


def test_reserved_generation_start_raises_409_while_reserved(test_state):
    with test_state.generation.reserved_generation_start():
        with pytest.raises(HTTPError) as exc_info:
            with test_state.generation.reserved_generation_start():
                pass
    assert exc_info.value.status_code == 409


def test_reserved_generation_start_releases_on_exception(test_state):
    with pytest.raises(ValueError):
        with test_state.generation.reserved_generation_start():
            raise ValueError("boom")

    # The reservation must be released even though the body never reached start_generation() -
    # a stuck reservation would 409 every future generation for _RESERVATION_TIMEOUT_S seconds.
    assert test_state.generation.try_reserve_generation_start() is True


def test_try_reserve_generation_start_only_one_thread_wins(test_state):
    # Mirrors the PR's own manual verification (30 truly-concurrent generate-image requests
    # split exactly 15 proceeded / 15 rejected) as a fast, deterministic regression test: fire N
    # threads at the reservation gate at the same instant via a barrier and assert exactly one
    # claims it, proving try_reserve_generation_start's lock genuinely serializes the check-and-set
    # instead of leaving a read-then-write gap two threads could both slip through.
    thread_count = 16
    barrier = threading.Barrier(thread_count)
    results: list[bool] = []
    results_lock = threading.Lock()

    def attempt() -> None:
        barrier.wait()
        reserved = test_state.generation.try_reserve_generation_start()
        with results_lock:
            results.append(reserved)

    threads = [threading.Thread(target=attempt) for _ in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results.count(True) == 1
    assert results.count(False) == thread_count - 1


def test_download_terminal_state_is_sticky_until_next_session(test_state):
    session_id = test_state.downloads.start_download({"ltx-2.3-22b-distilled"})
    test_state.downloads.start_file("ltx-2.3-22b-distilled", "ltx-2.3-22b-distilled.safetensors")
    test_state.downloads.finish_download()

    progress = test_state.downloads.get_download_progress(session_id)
    assert progress.status == "complete"


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


def test_mps_skips_torch_compile(test_state, fake_services, create_fake_model_files):
    create_fake_model_files()
    test_state.state.app_settings.use_torch_compile = True
    test_state.pipelines._runtime_device = "mps"  # noqa: SLF001 - explicit platform behavior assertion

    pipeline_state = test_state.pipelines.load_gpu_pipeline("fast")
    assert fake_services.fast_video_pipeline.compile_calls == 0
    assert pipeline_state.is_compiled is False


def test_retake_pipeline_eviction(test_state, create_fake_model_files):
    create_fake_model_files()
    test_state.pipelines.load_gpu_pipeline("fast")

    retake_state = test_state.pipelines.load_retake_pipeline(distilled=True)
    assert isinstance(test_state.state.gpu_slot, GpuSlot)
    assert isinstance(test_state.state.gpu_slot.active_pipeline, RetakePipelineState)
    assert test_state.state.gpu_slot.active_pipeline is retake_state

    test_state.pipelines.load_gpu_pipeline("fast")
    assert isinstance(test_state.state.gpu_slot.active_pipeline, VideoPipelineState)


def test_ic_lora_load_includes_depth_resources(test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files):
    create_fake_model_files()
    create_fake_ic_lora_files()
    model_spec = _current_model_spec()
    lora_path = str(resolve_model_path(test_state.config.default_models_dir, model_spec.ic_loras_spec.canny_cp))
    depth_path = str(resolve_model_path(test_state.config.default_models_dir, DEPTH_PROCESSOR_CP_ID))

    ic_state = test_state.pipelines.load_ic_lora(lora_path, depth_path)

    assert isinstance(ic_state, ICLoraState)
    assert ic_state.pipeline is fake_services.ic_lora_pipeline
    assert ic_state.depth_pipeline is fake_services.depth_processor_pipeline
    assert ic_state.lora_path == lora_path
    assert ic_state.depth_model_path == depth_path


def test_ic_lora_unload_clears_preprocessing_resources(test_state, create_fake_model_files, create_fake_ic_lora_files):
    create_fake_model_files()
    create_fake_ic_lora_files()
    model_spec = _current_model_spec()
    lora_path = str(resolve_model_path(test_state.config.default_models_dir, model_spec.ic_loras_spec.canny_cp))
    depth_path = str(resolve_model_path(test_state.config.default_models_dir, DEPTH_PROCESSOR_CP_ID))
    test_state.pipelines.load_ic_lora(lora_path, depth_path)

    assert isinstance(test_state.state.gpu_slot, GpuSlot)
    assert isinstance(test_state.state.gpu_slot.active_pipeline, ICLoraState)

    test_state.pipelines.unload_gpu_pipeline()

    assert test_state.state.gpu_slot is None
