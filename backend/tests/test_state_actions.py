"""State action and invariant tests for AppState."""

from __future__ import annotations

import threading
import time

import pytest

from _routes._errors import HTTPError
from services.patches import mps_conv3d_as_conv2d
from runtime_config.ic_lora_stage_mode import IcLoraStageMode
from runtime_config.video_job_budget import (
    HardwareProfile,
    LocalJobFilter,
    MemoryBudget,
    decide_edit_job,
)
from runtime_config.budget_decision import VideoJobLoad
from frame_math import compute_num_frames
from handlers.generation_handler import GenerationSlotWaitAborted, _RESERVATION_TIMEOUT_S
from services import generation_interrupt
from runtime_config.model_download_specs import (
    DEPTH_PROCESSOR_CP_ID,
    get_ltx_model_spec,
    resolve_model_path,
)
from state.app_settings import UpdateSettingsRequest
from state.app_state_types import CpuSlot, GpuSlot, ICLoraState, RetakePipelineState, VideoPipelineState
from tests.conftest import _IC_LORA_MODEL_ID


def _ic_lora_model_spec():
    return get_ltx_model_spec(_IC_LORA_MODEL_ID)


def test_start_generation_requires_gpu(test_state):
    with pytest.raises(RuntimeError, match="No active GPU pipeline"):
        test_state.generation.start_generation("gen-1")


def test_start_generation_pushes_mps_toggle_into_patch(test_state, create_fake_model_files, monkeypatch):
    monkeypatch.setattr(mps_conv3d_as_conv2d, "_enabled", False)
    create_fake_model_files()
    test_state.pipelines.load_gpu_pipeline("fast")
    test_state.settings.update_settings(UpdateSettingsRequest(mpsConv3DAsConv2DEnabled=True, useConvVae=True))

    test_state.generation.start_generation("gen-1")

    assert mps_conv3d_as_conv2d._enabled is True


def test_start_generation_keeps_mps_patch_off_without_fast_decode(test_state, create_fake_model_files, monkeypatch):
    monkeypatch.setattr(mps_conv3d_as_conv2d, "_enabled", True)
    create_fake_model_files()
    test_state.pipelines.load_gpu_pipeline("fast")
    test_state.settings.update_settings(UpdateSettingsRequest(mpsConv3DAsConv2DEnabled=True, useConvVae=False))

    test_state.generation.start_generation("gen-1")

    assert mps_conv3d_as_conv2d._enabled is False


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


def test_persistent_reservation_does_not_expire(test_state):
    with test_state.generation.reserved_generation_start(persistent=True):
        assert test_state.state.generation_starting_since is None
        assert test_state.generation.try_reserve_generation_start() is False
    assert test_state.generation.try_reserve_generation_start() is True


def test_queue_yields_holds_queue_waiters_but_not_direct_reservations(test_state):
    shutdown, claimed = threading.Event(), threading.Event()

    def queue_waiter():
        with test_state.generation.wait_for_generation_slot(shutdown):
            claimed.set()

    with test_state.generation.queue_yields():
        waiter = threading.Thread(target=queue_waiter)
        waiter.start()
        assert not claimed.wait(0.2)
        assert test_state.generation.try_reserve_generation_start(persistent=True) is True
        test_state.generation.release_generation_start_reservation()
        assert not claimed.is_set()
    assert claimed.wait(2)
    waiter.join(2)


def test_try_reserve_generation_start_stale_reservation_expires(test_state):
    # A path that raises before ever reaching start_generation()/fail_generation() (see
    # try_reserve_generation_start's own docstring) leaves the reservation stuck - simulate that
    # by backdating the timestamp past the timeout instead of actually sleeping in the test.
    assert test_state.generation.try_reserve_generation_start() is True
    test_state.state.generation_starting_since = time.monotonic() - _RESERVATION_TIMEOUT_S - 1

    assert test_state.generation.try_reserve_generation_start() is True


def test_cancel_during_reservation_blocks_second_start(test_state):
    with test_state.generation.reserved_generation_start():
        cancel = test_state.generation.cancel_generation()
        assert cancel.status == "cancelling"
        assert generation_interrupt.is_requested()
        with pytest.raises(HTTPError) as exc_info:
            with test_state.generation.reserved_generation_start():
                pass
        assert exc_info.value.status_code == 409

    assert test_state.generation.try_reserve_generation_start() is True
    test_state.generation.release_generation_start_reservation()


def test_cancel_after_start_keeps_slot_until_handler_exits(test_state, create_fake_model_files):
    create_fake_model_files()
    test_state.pipelines.load_gpu_pipeline("fast")
    with test_state.generation.reserved_generation_start():
        test_state.generation.start_generation("gen-1")
        cancel = test_state.generation.cancel_generation()
        assert cancel.status == "cancelling"
        assert test_state.generation.try_reserve_generation_start() is False
        progress = test_state.generation.get_generation_progress()
        assert progress.status == "running"
        assert progress.phase == "cancelled"
        assert progress.cancellable is True

    progress = test_state.generation.get_generation_progress()
    assert progress.status == "cancelled"
    assert progress.cancellable is False
    idle_cancel = test_state.generation.cancel_generation()
    assert idle_cancel.status == "no_active_generation"
    assert test_state.generation.try_reserve_generation_start() is True
    test_state.generation.release_generation_start_reservation()


def test_raise_if_cancelled_ignores_sticky_cancelled_from_previous_job(
    test_state, create_fake_model_files
):
    # After Stop, AppState stays GenerationCancelled until the next start_generation().
    # A new reservation must not treat that sticky terminal state as "this job was cancelled".
    create_fake_model_files()
    test_state.pipelines.load_gpu_pipeline("fast")
    with test_state.generation.reserved_generation_start():
        test_state.generation.start_generation("gen-1")
        test_state.generation.cancel_generation()
    assert test_state.generation.get_generation_progress().status == "cancelled"

    with test_state.generation.reserved_generation_start():
        test_state.generation.raise_if_cancelled()
        test_state.generation.start_generation("gen-2")

    assert test_state.generation.get_generation_progress().id == "gen-2"
    assert test_state.generation.get_generation_progress().status == "running"


def test_start_generation_honors_cancel_during_reservation(test_state, create_fake_model_files):
    from services.generation_interrupt import GenerationCancelledError

    create_fake_model_files()
    test_state.pipelines.load_gpu_pipeline("fast")
    with test_state.generation.reserved_generation_start():
        test_state.generation.cancel_generation()
        with pytest.raises(GenerationCancelledError):
            test_state.generation.start_generation("gen-1")
        assert test_state.generation.try_reserve_generation_start() is False

    assert test_state.generation.get_generation_progress().status == "cancelled"


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


def test_wait_for_generation_slot_acquires_after_existing_job_releases(test_state) -> None:
    assert test_state.generation.try_reserve_generation_start() is True
    shutdown = threading.Event()
    acquired = threading.Event()
    thread_error: BaseException | None = None

    def wait_for_slot() -> None:
        nonlocal thread_error
        try:
            with test_state.generation.wait_for_generation_slot(shutdown):
                acquired.set()
        except Exception as exc:
            thread_error = exc

    thread = threading.Thread(target=wait_for_slot)
    thread.start()
    try:
        assert acquired.wait(timeout=0.05) is False

        test_state.generation.release_generation_start_reservation()
        assert acquired.wait(timeout=1.0) is True
        thread.join(timeout=1.0)
        assert thread.is_alive() is False
        assert thread_error is None
        assert test_state.generation.try_reserve_generation_start() is True
        test_state.generation.release_generation_start_reservation()
    finally:
        shutdown.set()
        if thread.is_alive():
            test_state.generation.release_generation_start_reservation()
        thread.join(timeout=1.0)


def test_wait_for_generation_slot_exits_on_shutdown(test_state) -> None:
    assert test_state.generation.try_reserve_generation_start() is True
    shutdown = threading.Event()
    entered = threading.Event()
    thread_error: BaseException | None = None

    def wait_for_slot() -> None:
        nonlocal thread_error
        try:
            with test_state.generation.wait_for_generation_slot(shutdown):
                entered.set()
        except Exception as exc:
            thread_error = exc

    thread = threading.Thread(target=wait_for_slot)
    try:
        thread.start()
        shutdown.set()
        thread.join(timeout=1.0)
        assert thread.is_alive() is False
        assert entered.is_set() is False
        assert isinstance(thread_error, GenerationSlotWaitAborted)
        assert test_state.generation.try_reserve_generation_start() is False
    finally:
        test_state.generation.release_generation_start_reservation()
    assert test_state.generation.try_reserve_generation_start() is True
    test_state.generation.release_generation_start_reservation()


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
    assert test_state.queued_generations is not None
    assert test_state.assets is not None
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

    retake_state, _ = test_state.pipelines.load_retake_pipeline(distilled=True)
    assert isinstance(test_state.state.gpu_slot, GpuSlot)
    assert isinstance(test_state.state.gpu_slot.active_pipeline, RetakePipelineState)
    assert test_state.state.gpu_slot.active_pipeline is retake_state

    test_state.pipelines.load_gpu_pipeline("fast")
    assert isinstance(test_state.state.gpu_slot.active_pipeline, VideoPipelineState)


def test_retake_pipeline_uses_text_encoder_of_the_requested_model(
    test_state, fake_services, create_fake_model_files
):
    # Queued Home jobs pass the offering's checkpoint while Settings stays on 2.5. The 2.3
    # checkpoint must get its own Gemma 3 encoder, not the active 2.5 encode-only one.
    create_fake_model_files(model_id="ltx-2.5-22b-distilled")
    create_fake_model_files(model_id="ltx-2.3-22b-distilled-1.1")
    test_state.state.app_settings.use_local_text_encoder = True
    test_state.state.app_settings.active_ltx_model_id = "ltx-2.5-22b-distilled"
    spec_23 = get_ltx_model_spec("ltx-2.3-22b-distilled-1.1")
    expected_gemma_root = str(
        resolve_model_path(test_state.config.default_models_dir, spec_23.text_encoder_cp)
    )

    state, _ = test_state.pipelines.load_retake_pipeline(
        distilled=True, ltx_model_id="ltx-2.3-22b-distilled-1.1"
    )

    assert state.ltx_model_id == "ltx-2.3-22b-distilled-1.1"
    assert state.gemma_root == expected_gemma_root
    assert fake_services.retake_pipeline.create_calls[-1]["gemma_root"] == expected_gemma_root


def test_retake_1080p_streams_without_flipping_process_mode(
    test_state, fake_services, create_fake_model_files
):
    create_fake_model_files()
    assert test_state.config.local_generations_mode == "full_models_loading"
    frames = compute_num_frames(10, 24)

    state, _ = test_state.pipelines.load_retake_pipeline(
        distilled=True, width=1920, height=1088, frames=frames
    )

    assert state.loading_mode == "streaming_models_loading"
    assert fake_services.retake_pipeline.last_streaming_prefetch_count == 2
    assert test_state.config.local_generations_mode == "full_models_loading"


def test_long_1080p_edit_window_is_capped_to_what_31gb_loads(
    test_state, create_fake_model_files
):
    create_fake_model_files()
    cap = test_state.pipelines.edit_encode_frame_cap(1920, 1088)
    assert cap < 505
    assert test_state.pipelines.edit_encode_frame_cap(1280, 736) == 505
    decision = decide_edit_job(
        1920,
        1088,
        cap,
        job_filter=LocalJobFilter(
            profile=HardwareProfile(
                system="Linux",
                model_id=None,
                process_mode="full_models_loading",
            ),
            budget=MemoryBudget(
                total_vram_bytes=None,
                free_vram_bytes=None,
                total_ram_bytes=None,
                memory_gb=31,
                mps_pool_bytes=None,
            ),
        ),
    )
    assert isinstance(decision.priced.decision, VideoJobLoad)

    state, _ = test_state.pipelines.load_retake_pipeline(
        distilled=True, width=1920, height=1088, frames=cap
    )
    assert state.loading_mode == "streaming_models_loading"

    with pytest.raises(HTTPError) as gen_space:
        test_state.pipelines.load_retake_pipeline(
            distilled=True, width=1920, height=1088, frames=505
        )
    assert gen_space.value.status_code == 422
    assert gen_space.value.detail == (
        "This job needs 44.2 GB of GPU memory but this card has 31.0 GB. "
        "Lower the resolution, duration, frame rate, or aspect width."
    )

    with pytest.raises(HTTPError) as home:
        test_state.pipelines.load_retake_pipeline(
            distilled=True, width=1920, height=1088, frames=505, queued_home=True
        )
    assert home.value.detail == (
        "This job needs 44.2 GB of GPU memory but this card has 31.0 GB. "
        "Lower the resolution."
    )


def test_executor_frame_cap_ignores_resident_occupancy(
    test_state, fake_services, create_fake_model_files
):
    """Depressed free (resident pipeline still loaded) must not shorten the executor window."""
    create_fake_model_files()
    test_state.config.system = "Windows"
    test_state.config.vram_gb = 31
    test_state.config.vram_bytes = 31 * 1024**3
    test_state.config.total_ram_bytes = 64 * 1024**3
    test_state.config.local_generations_mode = "streaming_models_loading"
    bare = test_state.pipelines.edit_encode_frame_cap(1920, 1088)
    fake_services.gpu_info.free_vram_bytes = 20 * 1024**3
    assert test_state.pipelines.edit_encode_frame_cap(1920, 1088) < bare
    assert (
        test_state.pipelines.edit_encode_frame_cap(1920, 1088, total_only=True)
        == bare
    )


def _windows_low_ram(test_state) -> None:
    test_state.config.system = "Windows"
    test_state.config.vram_gb = 31
    test_state.config.vram_bytes = 31 * 1024**3
    test_state.config.total_ram_bytes = 16 * 1024**3
    test_state.config.checkpoint_drive_type = None
    test_state.config.local_generations_mode = "streaming_models_loading"


def test_edit_load_mode_maps_ram_shortfall(test_state) -> None:
    """Windows 16 GB RAM: streamed edit 422s with the RAM code, not TOO_LARGE."""
    _windows_low_ram(test_state)
    with pytest.raises(HTTPError) as exc_info:
        test_state.pipelines._edit_load_mode(1280, 704, 337)
    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == (
        "This job needs 64 GB of system RAM but this machine has 16 GB. "
        "Close heavy programs; on desktops, more RAM is the real fix."
    )


def test_ic_lora_load_mode_maps_ram_shortfall(test_state) -> None:
    """Windows 16 GB RAM: streamed IC-LoRA 422s through the shared table."""
    _windows_low_ram(test_state)
    with pytest.raises(HTTPError) as exc_info:
        test_state.ic_lora._ic_lora_job(
            1024,
            576,
            frame_count=121,
            stage_mode=IcLoraStageMode.SINGLE_STAGE,
            resolution_factor=1.5,
        )
    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == (
        "This job needs 64 GB of system RAM but this machine has 16 GB. "
        "Close heavy programs; on desktops, more RAM is the real fix."
    )


def test_retake_540p_5s_stays_full(test_state, fake_services, create_fake_model_files):
    create_fake_model_files()
    frames = compute_num_frames(5, 24)

    state, _ = test_state.pipelines.load_retake_pipeline(
        distilled=True, width=1024, height=576, frames=frames
    )

    assert state.loading_mode == "full_models_loading"
    assert fake_services.retake_pipeline.last_streaming_prefetch_count is None


def test_retake_full_pipeline_is_not_reused_for_a_stream_job(
    test_state, fake_services, create_fake_model_files
):
    create_fake_model_files()
    small, _ = test_state.pipelines.load_retake_pipeline(
        distilled=True, width=1024, height=576, frames=compute_num_frames(5, 24)
    )
    large, _ = test_state.pipelines.load_retake_pipeline(
        distilled=True, width=1920, height=1088, frames=compute_num_frames(10, 24)
    )

    assert small.loading_mode == "full_models_loading"
    assert large.loading_mode == "streaming_models_loading"
    assert large is not small
    assert len(fake_services.retake_pipeline.create_calls) == 2


def test_ic_lora_load_includes_depth_resources(test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files):
    create_fake_model_files(model_id=_IC_LORA_MODEL_ID)
    create_fake_ic_lora_files()
    model_spec = _ic_lora_model_spec()
    assert model_spec.ic_loras_spec is not None
    lora_path = str(resolve_model_path(test_state.config.default_models_dir, model_spec.ic_loras_spec.canny_cp))
    depth_path = str(resolve_model_path(test_state.config.default_models_dir, DEPTH_PROCESSOR_CP_ID))

    ic_state = test_state.pipelines.load_ic_lora(lora_path, depth_path)

    assert isinstance(ic_state, ICLoraState)
    assert ic_state.pipeline is fake_services.ic_lora_pipeline
    assert ic_state.depth_pipeline is fake_services.depth_processor_pipeline
    assert ic_state.lora_path == lora_path
    assert ic_state.depth_model_path == depth_path
    assert fake_services.ic_lora_pipeline.last_streaming_prefetch_count is None


def test_ic_lora_unload_clears_preprocessing_resources(test_state, create_fake_model_files, create_fake_ic_lora_files):
    create_fake_model_files(model_id=_IC_LORA_MODEL_ID)
    create_fake_ic_lora_files()
    model_spec = _ic_lora_model_spec()
    assert model_spec.ic_loras_spec is not None
    lora_path = str(resolve_model_path(test_state.config.default_models_dir, model_spec.ic_loras_spec.canny_cp))
    depth_path = str(resolve_model_path(test_state.config.default_models_dir, DEPTH_PROCESSOR_CP_ID))
    test_state.pipelines.load_ic_lora(lora_path, depth_path)

    assert isinstance(test_state.state.gpu_slot, GpuSlot)
    assert isinstance(test_state.state.gpu_slot.active_pipeline, ICLoraState)

    test_state.pipelines.unload_gpu_pipeline()

    assert test_state.state.gpu_slot is None


def test_pipeline_cache_rebuilds_when_active_model_changes(test_state, create_fake_model_files):
    # Two API-encodable versions both resolve gemma_root=None, so without an ltx_model_id cache
    # key the resident pipeline would keep serving the previous weights.
    create_fake_model_files(model_id="ltx-2.3-22b-distilled")
    create_fake_model_files(model_id="ltx-2.3-22b-distilled-1.1")
    test_state.state.app_settings.ltx_api_key = "test-key"
    test_state.state.app_settings.use_local_text_encoder = False
    test_state.state.app_settings.active_ltx_model_id = "ltx-2.3-22b-distilled-1.1"

    first = test_state.pipelines.load_gpu_pipeline("fast")
    assert first.ltx_model_id == "ltx-2.3-22b-distilled-1.1"
    assert first.gemma_root is None

    test_state.state.app_settings.active_ltx_model_id = "ltx-2.3-22b-distilled"
    second = test_state.pipelines.load_gpu_pipeline("fast")
    assert second.ltx_model_id == "ltx-2.3-22b-distilled"
    assert second.gemma_root is None
    assert second is not first


def test_pipeline_load_override_does_not_change_active_settings(
    test_state, create_fake_model_files
):
    create_fake_model_files()
    create_fake_model_files(model_id="ltx-2.3-22b-distilled-1.1")
    test_state.state.app_settings.ltx_api_key = "test-key"
    test_state.state.app_settings.use_local_text_encoder = False
    test_state.state.app_settings.active_ltx_model_id = "ltx-2.5-22b-distilled"

    loaded = test_state.pipelines.load_gpu_pipeline(
        "fast", ltx_model_id="ltx-2.3-22b-distilled-1.1"
    )
    assert loaded.ltx_model_id == "ltx-2.3-22b-distilled-1.1"
    assert test_state.state.app_settings.active_ltx_model_id == "ltx-2.5-22b-distilled"

    reloaded = test_state.pipelines.load_gpu_pipeline("fast")
    assert reloaded.ltx_model_id == "ltx-2.5-22b-distilled"
    assert test_state.state.app_settings.active_ltx_model_id == "ltx-2.5-22b-distilled"
