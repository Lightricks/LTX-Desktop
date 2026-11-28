"""A fal image job runs beside a local job, on its own lane.

Each test names the bug it catches.
"""

from __future__ import annotations

import io
import threading
import time

import pytest
from PIL import Image

_CREATE = "/api/generations/text-to-image"
_QUEUE = "/api/generation-queue"


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), color=(10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def _create(client, provider: str, **params: object) -> dict:
    response = client.post(_CREATE, json={"params": {"prompt": "a red fox", "provider": provider, **params}})
    assert response.status_code == 200, response.text
    return response.json()


def _wait_for(condition, what: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.02)


def _status(client, generation_id: str) -> str:
    return client.get(f"/api/generations/{generation_id}").json()["status"]


def _hold_fal(test_state, fake_services) -> threading.Event:
    """Keep every fal request open until the returned event is set."""
    test_state.state.app_settings.fal_api_key = "fal-key"
    fake_services.zit_api_client.text_to_image_result = _png_bytes()
    fake_services.zit_api_client.queued_release.clear()
    return fake_services.zit_api_client.queued_release


def _hold_local(fake_services, create_fake_model_files) -> None:
    create_fake_model_files(include_zit=True)
    pipeline = fake_services.image_generation_pipeline
    pipeline.inference_steps = 400
    pipeline.step_delay_s = 0.02


def _start_both(client, test_state, fake_services, create_fake_model_files) -> tuple[dict, dict]:
    _hold_fal(test_state, fake_services)
    _hold_local(fake_services, create_fake_model_files)
    local = _create(client, "local")
    assert fake_services.image_generation_pipeline.entered_inference.wait(timeout=5)
    fal = _create(client, "fal")
    assert fake_services.zit_api_client.queued_entered.wait(timeout=5)
    return local, fal


def test_a_local_job_and_a_fal_job_run_at_the_same_time(
    client, test_state, fake_services, create_fake_model_files
) -> None:
    # Bug caught: the fal job waits for the local job (one global queue).
    local, fal = _start_both(client, test_state, fake_services, create_fake_model_files)

    active = client.get(_QUEUE).json()["active"]

    assert {entry["generation"]["id"] for entry in active} == {local["id"], fal["id"]}
    assert {entry["generation"]["lane"] for entry in active} == {"local", "fal"}
    client.post(f"/api/generations/{local['id']}/cancel")
    fake_services.zit_api_client.queued_release.set()


def test_the_fal_job_reports_its_own_status_and_never_the_local_progress(
    client, test_state, fake_services, create_fake_model_files
) -> None:
    # Bug caught: the fal job writes the global generation state, so the local job's progress
    # bar jumps or the provider job shows the local job's percentage.
    local, fal = _start_both(client, test_state, fake_services, create_fake_model_files)

    _wait_for(
        lambda: any(
            entry["api_status"] == "pending" for entry in client.get(_QUEUE).json()["active"]
        ),
        "the fal status",
    )
    by_id = {entry["generation"]["id"]: entry for entry in client.get(_QUEUE).json()["active"]}
    global_progress = test_state.generation.get_generation_progress()

    assert by_id[fal["id"]]["api_status"] == "pending"
    assert by_id[local["id"]]["api_status"] is None
    assert global_progress.id == local["id"]
    assert global_progress.phase != "validating_request"
    client.post(f"/api/generations/{local['id']}/cancel")
    fake_services.zit_api_client.queued_release.set()


def test_cancelling_the_fal_job_leaves_the_local_job_running(
    client, test_state, fake_services, create_fake_model_files
) -> None:
    # Bug caught: a cancel is routed to the global interrupt and stops the wrong job.
    local, fal = _start_both(client, test_state, fake_services, create_fake_model_files)

    assert client.post(f"/api/generations/{fal['id']}/cancel").status_code == 200
    _wait_for(lambda: _status(client, fal["id"]) == "cancelled", "the fal job to cancel")

    assert _status(client, local["id"]) == "running"
    client.post(f"/api/generations/{local['id']}/cancel")


def test_cancelling_the_local_job_leaves_the_fal_job_running(
    client, test_state, fake_services, create_fake_model_files
) -> None:
    # Bug caught: the global interrupt flag set for the local job also stops the fal job.
    local, fal = _start_both(client, test_state, fake_services, create_fake_model_files)

    assert client.post(f"/api/generations/{local['id']}/cancel").status_code == 200
    _wait_for(lambda: _status(client, local["id"]) == "cancelled", "the local job to cancel")

    assert _status(client, fal["id"]) == "running"
    fake_services.zit_api_client.queued_release.set()
    _wait_for(lambda: _status(client, fal["id"]) == "succeeded", "the fal job to finish")


def test_a_cancelled_fal_job_frees_the_lane_for_the_next_fal_job(
    client, test_state, fake_services
) -> None:
    # Bug caught: a cancelled provider job keeps the lane until the provider answers.
    release = _hold_fal(test_state, fake_services)
    first = _create(client, "fal")
    assert fake_services.zit_api_client.queued_entered.wait(timeout=5)
    second = _create(client, "fal")

    assert _status(client, second["id"]) == "queued"
    client.post(f"/api/generations/{first['id']}/cancel")

    # The provider has not answered yet (the request is still held). The lane is free anyway.
    _wait_for(lambda: _status(client, first["id"]) == "cancelled", "the first job to cancel")
    _wait_for(lambda: _status(client, second["id"]) == "running", "the second job to start")
    release.set()
    _wait_for(lambda: _status(client, second["id"]) == "succeeded", "the second job to finish")


def test_two_fal_jobs_run_in_order_one_at_a_time(client, test_state, fake_services) -> None:
    # Bug caught: lane concurrency is not enforced and two fal requests overlap.
    release = _hold_fal(test_state, fake_services)
    first = _create(client, "fal")
    assert fake_services.zit_api_client.queued_entered.wait(timeout=5)
    second = _create(client, "fal")
    time.sleep(0.2)

    assert len(fake_services.zit_api_client.text_to_image_calls) == 1
    assert _status(client, second["id"]) == "queued"
    release.set()
    _wait_for(lambda: _status(client, second["id"]) == "succeeded", "the second job")
    assert _status(client, first["id"]) == "succeeded"


def test_a_fal_job_does_not_block_the_local_sync_generation_slot(
    client, test_state, fake_services
) -> None:
    # Bug caught: a fal job takes the global GPU slot, so Generate is locked while fal runs.
    _hold_fal(test_state, fake_services)
    _create(client, "fal")
    assert fake_services.zit_api_client.queued_entered.wait(timeout=5)

    # The global slot is free: a local job can reserve it while the fal request is open.
    assert test_state.generation.try_reserve_generation_start() is True
    test_state.generation.release_generation_start_reservation()
    fake_services.zit_api_client.queued_release.set()


@pytest.mark.parametrize("parallel_lanes", [False], indirect=True)
def test_with_parallel_lanes_off_a_fal_job_waits_behind_the_local_job(
    client, test_state, fake_services, create_fake_model_files
) -> None:
    # Bug caught: the kill switch leaves the fal lane on, so runs are not serial.
    _hold_fal(test_state, fake_services)
    fake_services.zit_api_client.queued_release.set()
    _hold_local(fake_services, create_fake_model_files)
    local = _create(client, "local")
    assert fake_services.image_generation_pipeline.entered_inference.wait(timeout=5)

    fal = _create(client, "fal")
    time.sleep(0.2)

    assert fal["lane"] == "local"
    assert _status(client, fal["id"]) == "queued"
    assert fake_services.zit_api_client.text_to_image_calls == []
    client.post(f"/api/generations/{local['id']}/cancel")
    _wait_for(lambda: _status(client, fal["id"]) == "succeeded", "the fal job to run after")


@pytest.mark.parametrize("parallel_lanes", [False], indirect=True)
def test_with_parallel_lanes_off_the_local_gemma_still_rewrites_a_fal_prompt(
    client, test_state, fake_services, create_fake_model_files
) -> None:
    # Bug caught: the kill switch makes the lease fail (the job holds the local slot itself), so
    # every fal job pays for fal expansion although Gemma is downloaded and the GPU is idle.
    _hold_fal(test_state, fake_services)
    fake_services.zit_api_client.queued_release.set()
    create_fake_model_files(include_prompt_enhancer=True)
    test_state.state.app_settings.explore_auto_enhance_prompts = True
    fake_services.prompt_enhancer_pipeline.enhanced_prompt = "a richly detailed red fox"

    fal = _create(client, "fal")
    _wait_for(lambda: _status(client, fal["id"]) == "succeeded", "the fal job")

    call = fake_services.zit_api_client.text_to_image_calls[-1]
    assert call["prompt"] == "a richly detailed red fox"
    assert call["enable_prompt_expansion"] is False


@pytest.mark.parametrize("parallel_lanes", [False], indirect=True)
def test_retrying_a_failed_fal_job_with_parallel_lanes_off_runs_it_on_the_local_lane(
    client, test_state, fake_services
) -> None:
    # Bug caught: the retry keeps the lane "fal", which has no runner now, and the job waits forever.
    _hold_fal(test_state, fake_services)
    fake_services.zit_api_client.queued_release.set()
    store = test_state.queued_generations._db
    spec = {"params": {"prompt": "a red fox", "provider": "fal", "width": 512, "height": 512}, "inputs": {}}
    job = store.insert_generation("text-to-image", spec, lane="fal")
    test_state.generation_queue.stop()
    claimed = store.claim_next_queued("fal")
    assert claimed is not None
    store.mark_failed(job.id, "OUTPUT_UNREADABLE", attempt_count=claimed.attempt_count)
    test_state.generation_queue.start()

    assert client.post(f"/api/generations/{job.id}/retry").status_code == 200

    _wait_for(lambda: _status(client, job.id) == "succeeded", "the retried job")
    assert client.get(f"/api/generations/{job.id}").json()["lane"] == "local"


def _store(test_state):
    return test_state.queued_generations._db


def test_the_fal_request_id_is_stored_before_the_first_status_check(
    client, test_state, fake_services
) -> None:
    # Bug caught: the id is stored late or never, so a stop during the wait leaves a request nobody can cancel.
    _hold_fal(test_state, fake_services)
    fal = _create(client, "fal")
    assert fake_services.zit_api_client.queued_entered.wait(timeout=5)

    body = client.get(f"/api/generations/{fal['id']}").json()
    assert body["provider_job_id"] == "fake-fal-1"
    fake_services.zit_api_client.queued_release.set()
    _wait_for(lambda: _status(client, fal["id"]) == "succeeded", "the fal job to finish")


def test_a_fal_request_id_that_cannot_be_stored_fails_the_job_and_cancels_the_request(
    client, test_state, fake_services, monkeypatch
) -> None:
    # Bug caught: the job runs on at fal with no id on record, so a boot cannot cancel it.
    _hold_fal(test_state, fake_services)
    monkeypatch.setattr(_store(test_state), "set_provider_job_id", lambda *args: False)

    fal = _create(client, "fal")

    _wait_for(lambda: _status(client, fal["id"]) == "failed", "the fal job to fail")
    assert fake_services.zit_api_client.cancelled_request_ids == ["fake-fal-1"]


def _two_running_fal_rows(test_state) -> list[str]:
    test_state.state.app_settings.fal_api_key = "fal-key"
    test_state.generation_queue.stop()
    store = _store(test_state)
    ids = []
    for request_id in ("req-a", "req-b"):
        store.insert_generation("text-to-image", {"params": {"prompt": "x"}, "inputs": {}}, lane="fal")
        claimed = store.claim_next_queued("fal")
        assert claimed is not None
        assert store.set_provider_job_id(claimed.id, claimed.attempt_count, request_id)
        ids.append(claimed.id)
    return ids


def _pending_provider_ids(test_state) -> list[str]:
    return sorted(orphan.provider_job_id for orphan in _store(test_state).list_provider_orphans())


def test_a_boot_cancels_every_fal_job_that_was_running_and_forgets_the_ids(
    test_state, fake_services
) -> None:
    # Bug caught: a job running at fal when the app stopped is failed locally and fal keeps billing it.
    _two_running_fal_rows(test_state)

    test_state.queued_generations.recover_on_boot()

    _wait_for(lambda: _pending_provider_ids(test_state) == [], "the boot cancel to finish")
    assert sorted(fake_services.zit_api_client.cancel_queued_calls) == ["req-a", "req-b"]


def test_a_failing_fal_cancel_at_boot_does_not_block_the_boot_and_keeps_the_ids(
    test_state, fake_services
) -> None:
    # Bug caught: fal is down at boot, the app fails to start, or the ids are dropped and the jobs run on.
    ids = _two_running_fal_rows(test_state)
    fake_services.zit_api_client.raise_on_cancel_queued = RuntimeError("fal down")

    test_state.queued_generations.recover_on_boot()

    _wait_for(
        lambda: len(fake_services.zit_api_client.cancel_queued_calls) == 2, "both cancels to run"
    )
    assert _pending_provider_ids(test_state) == ["req-a", "req-b"]
    assert all(_store(test_state).get_generation(i).status == "failed" for i in ids)
