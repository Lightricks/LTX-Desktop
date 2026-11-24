from __future__ import annotations

import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from threading import Event
from typing import get_args
from uuid import NAMESPACE_URL, uuid4, uuid5

import pytest
import httpx
from openai import APIConnectionError, APITimeoutError
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

from assistant_limits import ASSISTANT_LIMITS
from api_types import CreateTextToVideoRequest, DownloadSpec, DownloadVariant, IcLoraSettings, LTXLocalModelId, TextToVideoParams
from handlers.assistant_params import LocalVideoDuration, LocalVideoResolution
from services.analytics import queued_generation_details
from services.assistant_run import ASSISTANT_PAYLOAD_VERSION
from services.records import AssistantRunRecord
from services.sqlite_store import SqliteStore
from services.features.lora_recipes import get_lora_recipe, queued_lora_recipes
from runtime_config.accelerator import accelerator_backend
from runtime_config.model_download_specs import get_ltx_model_spec, resolve_ic_lora_path, resolve_lora_path
from tests.test_http import _png, _wav
from tests.test_remote_generations import _mp4
from tests.lora_catalog_helpers import add_ic_lora


def request_body(text="Generate a 270p, 2 second video of a rolling wave"):
    return {"request_id": str(uuid4()), "conversation_id": str(uuid4()), "text": text}


def call(name, args=None, call_id=None):
    if name == "generate_video" and args and "params" not in args:
        args = {"params": args}
    args = dict(args or {})
    if "request" in args:
        args.update(args.pop("request"))
    if name in ("generate_video", "image_to_video", "audio_to_video", "run_lora_recipe", "retake_video", "extend_video"):
        args.setdefault("job_number", 2 if name == "extend_video" else 1)
    return ModelResponse(parts=[ToolCallPart(name, args or {}, tool_call_id=call_id or str(uuid4()))])


def last_generation(response, client):
    body = response.json() if hasattr(response, "json") else response
    ids = body["generation_ids"]
    return client.get(f"/api/generations/{ids[-1]}").json() if ids else None


def say(text="The video is ready."):
    return ModelResponse(parts=[TextPart(text)])


def saved_run(test_state, run_id):
    record = SqliteStore(test_state.config.app_data_dir).get_assistant_run(run_id)
    return json.loads(record.payload) if record is not None else None


def run_record(run_id, *, stopped=False):
    return AssistantRunRecord(id=run_id, conversation_id="conversation", payload="{}", payload_version=ASSISTANT_PAYLOAD_VERSION, stopped=stopped)


def saved_history(test_state, run_id) -> str:
    run = saved_run(test_state, run_id)
    assert run is not None
    return run["history"]


@pytest.fixture(params=["gemini", "local"])
def setup_agent(test_state, fake_services, create_fake_model_files, request):
    create_fake_model_files()
    test_state.state.app_settings.gemini_api_key = "test-key"
    test_state.state.app_settings.assistant_provider = request.param
    test_state.generation_queue.stop()
    args = {"prompt": "A rolling wave", "model": "ltx-2.5-fast", "resolution": "270p",
            "duration": 2, "fps": 24, "aspectRatio": "16:9"}
    return fake_services.assistant_provider, args


def finish_job(test_state, generation_id):
    """Run the queue until the job succeeds; work the assistant queues in response stays queued."""
    queue, held = test_state.generation_queue, ExitStack()
    listener = queue._finished_listener

    def finished(done_id):
        if done_id == generation_id:
            # Runs on the queue thread before it claims another job.
            held.enter_context(test_state.generation.queue_yields())
        listener(done_id)

    queue.set_finished_listener(finished)
    queue.start()
    try:
        deadline = time.monotonic() + 5
        while test_state.queued_generations.get_generation(generation_id).status != "succeeded":
            assert time.monotonic() < deadline
            time.sleep(0.02)
    finally:
        queue.stop()
        queue.set_finished_listener(listener)
        held.close()


def ingest(client, path):
    response = client.post("/api/assets", json={"path": str(path)})
    assert response.status_code == 200, response.text
    return response.json()


def extension_call(source_id):
    return call("extend_video", {"generation_id": source_id, "params": {
        "model": "ltx-2.5-fast", "prompt": "The wave continues rolling", "added_seconds": 2, "mode": "end",
    }})


def queue_four(client, provider, args):
    provider.responses.extend([
        call("generate_video", {"params": args, "job_number": number})
        for number in range(1, 5)
    ])
    provider.responses.append(say("Queued four videos."))
    body = request_body("Generate four individual clips")
    response = client.post("/api/assistant/turn", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "waiting"
    ids = response.json()["generation_ids"]
    assert len(ids) == len(set(ids)) == 4
    return body, ids


def test_a_job_over_the_cap_is_refused_and_tells_the_model_to_report_what_was_skipped(client, test_state, setup_agent):
    provider, args = setup_agent
    provider.responses.extend([
        call("generate_video", {"params": args, "job_number": ASSISTANT_LIMITS.jobs + 1}),
        say(f"I can queue at most {ASSISTANT_LIMITS.jobs}."),
    ])
    body = request_body("Generate thirteen clips")
    response = client.post("/api/assistant/turn", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["generation_ids"] == []
    assert not SqliteStore(test_state.config.app_data_dir).has_queued()
    # The instructions state the cap too; this sentence is only in the refusal.
    assert "Do not queue more; tell the user how many were not queued" in saved_history(test_state, body["request_id"])


def test_missing_gemini_key_is_rejected_without_leaving_a_blocking_run(client, test_state):
    body = request_body()
    response = client.post("/api/assistant/turn", json=body)
    assert response.status_code == 400
    assert response.json()["code"] == "GEMINI_API_KEY_MISSING"
    assert saved_run(test_state, body["request_id"]) is None


def test_local_assistant_needs_no_gemini_key(client, test_state, fake_services):
    test_state.state.app_settings.assistant_provider = "local"
    fake_services.assistant_provider.responses.append(say("Hello from local Qwen."))
    body = request_body("Hello")
    response = client.post("/api/assistant/turn", json=body)
    assert response.status_code == 200, response.text
    turn = client.get(f"/api/assistant/conversations/{body['conversation_id']}").json()["turns"][0]
    assert (turn["provider"], turn["model"]) == ("local", "qwen3.5:9b")


def gemini_error(status):
    return ModelHTTPError(status, "gemini-3.5-flash-lite", {"error": {"code": status}})


@pytest.mark.parametrize("provider,error,code", [
    ("local", APIConnectionError(request=httpx.Request("POST", "http://localhost")), "ASSISTANT_LOCAL_UNAVAILABLE"),
    ("local", APITimeoutError(request=httpx.Request("POST", "http://localhost")), "ASSISTANT_LOCAL_TIMEOUT"),
    ("local", ModelHTTPError(404, "qwen3.5:9b", {"error": "not found"}), "ASSISTANT_LOCAL_MODEL_MISSING"),
    ("gemini", gemini_error(401), "GEMINI_AUTH_FAILED"),
    ("gemini", gemini_error(429), "GEMINI_RATE_LIMITED"),
    ("gemini", gemini_error(504), "GEMINI_TIMEOUT"),
    ("gemini", gemini_error(503), "GEMINI_UNAVAILABLE"),
    ("gemini", RuntimeError("secret-api-key"), None),
])
def test_provider_errors_map_to_codes_without_leaking_details(client, test_state, fake_services, provider, error, code):
    test_state.state.app_settings.assistant_provider = provider
    test_state.state.app_settings.gemini_api_key = "test-key"
    fake_services.assistant_provider.responses.append(error)
    response = client.post("/api/assistant/turn", json=request_body("Hello"))
    assert response.status_code >= 400
    assert "secret-api-key" not in response.text
    if code:
        assert response.json()["code"] == code


def test_resume_keeps_pinned_provider_and_model_after_settings_change(client, test_state, setup_agent):
    provider, args = setup_agent
    provider.responses.append(call("generate_video", args))
    body = request_body()
    first = client.post("/api/assistant/turn", json=body)
    assert first.status_code == 200, first.text
    pinned = saved_run(test_state, body["request_id"])
    settings = test_state.state.app_settings
    settings.assistant_provider = "local" if settings.assistant_provider == "gemini" else "gemini"
    settings.gemini_model = "gemini-2.5-flash"
    finish_job(test_state, last_generation(first, client)["id"])
    assert client.post(f"/api/assistant/runs/{body['request_id']}/continue").status_code == 200
    resumed = saved_run(test_state, body["request_id"])
    assert (resumed["provider"], resumed["model"]) == (pinned["provider"], pinned["model"])


def test_replayed_request_returns_the_same_reply(client, setup_agent):
    provider, _ = setup_agent
    provider.responses.extend([say("What would you like to make?"), say("A different reply")])
    body = request_body("Hello")
    first = client.post("/api/assistant/turn", json=body)
    second = client.post("/api/assistant/turn", json=body)
    assert first.status_code == 200, first.text
    assert first.json() == second.json()


def test_request_id_cannot_be_reused_for_different_input(client, setup_agent):
    provider, _ = setup_agent
    provider.responses.append(say())
    body = request_body("Hello")
    assert client.post("/api/assistant/turn", json=body).status_code == 200
    body["text"] = "Different request"
    assert client.post("/api/assistant/turn", json=body).status_code == 409


def test_read_only_tool_loop_pauses_at_the_request_limit(client, setup_agent):
    provider, _ = setup_agent
    provider.responses.extend(call("list_video_models") for _ in range(17))
    response = client.post("/api/assistant/turn", json=request_body("List available models"))
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "paused"
    resumed = client.post(f"/api/assistant/runs/{response.json()['run_id']}/continue")
    assert resumed.json()["status"] == "complete"


def test_resume_while_job_pending_reuses_it_after_store_rebind(client, test_state, setup_agent):
    provider, args = setup_agent
    provider.responses.append(call("generate_video", args))
    body = request_body()
    first = client.post("/api/assistant/turn", json=body)
    assert first.status_code == 200, first.text
    test_state._bind_store(SqliteStore(test_state.config.app_data_dir))
    test_state.generation_queue.stop()
    for _ in range(3):
        resumed = client.post(f"/api/assistant/runs/{body['request_id']}/continue")
        assert resumed.json()["status"] == "waiting"
        assert resumed.json()["generation_ids"] == first.json()["generation_ids"]
    assert len(client.get("/api/generations", params={"feature": "text-to-video"}).json()) == 1


def test_invalid_tool_params_can_be_corrected_in_the_same_run(client, setup_agent):
    provider, args = setup_agent
    provider.responses.extend([call("generate_video", {**args, "fps": 30}), call("generate_video", args)])
    response = client.post("/api/assistant/turn", json=request_body())
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "waiting"
    generation = last_generation(response, client)
    params = TextToVideoParams.model_validate(generation["spec"]["params"])
    assert params.fps == 24 and params.seed is None


def test_new_message_is_blocked_while_a_run_is_waiting(client, setup_agent):
    provider, args = setup_agent
    provider.responses.append(call("generate_video", args))
    first = request_body()
    assert client.post("/api/assistant/turn", json=first).status_code == 200
    refused = client.post(
        "/api/assistant/turn",
        json={**request_body("Another clip"), "conversation_id": first["conversation_id"]},
    )
    assert refused.status_code == 409
    assert refused.json()["code"] == "ASSISTANT_CONVERSATION_BUSY"
    assert client.post(f"/api/assistant/runs/{first['request_id']}/stop").json()["status"] == "stopped"


def test_stop_cancels_owned_jobs_but_not_unrelated_queue_and_prevents_resume(client, test_state, setup_agent):
    provider, args = setup_agent
    other = client.post("/api/generations/text-to-video", json={"params": args}).json()
    body, ids = queue_four(client, provider, args)
    url = f"/api/assistant/runs/{body['request_id']}"
    response = client.post(f"{url}/stop")
    assert response.json()["status"] == "stopped"
    assert all(test_state.queued_generations.get_generation(id).status == "cancelled" for id in ids)
    assert test_state.queued_generations.get_generation(other["id"]).status == "queued"
    assert client.post(f"{url}/continue").json()["status"] == "stopped"


@pytest.mark.parametrize("error_code", ["CAPABILITY_FAILED", "EXECUTOR_FAILED", "INTERRUPTED"])
def test_only_interrupted_jobs_retry_once(client, test_state, setup_agent, error_code):
    provider, args = setup_agent
    provider.responses.append(call("generate_video", args))
    body = request_body()
    first = client.post("/api/assistant/turn", json=body)
    assert first.status_code == 200, first.text
    db = SqliteStore(test_state.config.app_data_dir)
    generation = db.claim_next_queued()
    db.mark_failed(generation.id, error_code, attempt_count=generation.attempt_count)
    provider.responses.extend([call("retry_interrupted_generation"), say("Please check the failure before retrying.")])
    resumed = client.post(f"/api/assistant/runs/{body['request_id']}/continue")
    assert resumed.status_code == 200, resumed.text
    if error_code == "INTERRUPTED":
        assert resumed.json()["status"] == "waiting"
        generation = db.claim_next_queued()
        db.mark_failed(generation.id, "INTERRUPTED", attempt_count=generation.attempt_count)
        provider.responses.clear()
        provider.responses.extend([call("retry_interrupted_generation"), say("The retry failed; please check the app.")])
        resumed = client.post(f"/api/assistant/runs/{body['request_id']}/continue")
        assert resumed.status_code == 200, resumed.text
    assert resumed.json()["status"] == "complete"
    assert not db.has_queued()


def failed_job_in_chat(client, test_state, provider, args):
    """A chat whose first message queued a job that then failed; the first run is complete."""
    provider.responses.append(call("generate_video", args))
    body = request_body()
    assert client.post("/api/assistant/turn", json=body).status_code == 200
    db = SqliteStore(test_state.config.app_data_dir)
    generation = db.claim_next_queued()
    db.mark_failed(generation.id, "EXECUTOR_FAILED", attempt_count=generation.attempt_count)
    provider.responses.append(say("The job failed."))
    assert client.post(f"/api/assistant/runs/{body['request_id']}/continue").json()["status"] == "complete"
    return body["conversation_id"], generation.id, db


def retry_in(client, conversation_id):
    body = {**request_body("Retry that video"), "conversation_id": conversation_id}
    return body, client.post("/api/assistant/turn", json=body)


def test_user_can_retry_a_failed_job_from_an_earlier_message_in_place(client, test_state, setup_agent):
    provider, args = setup_agent
    conversation_id, generation_id, db = failed_job_in_chat(client, test_state, provider, args)
    provider.responses.append(call("retry_failed_job", {"generation_id": generation_id}))
    _, response = retry_in(client, conversation_id)
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "waiting"
    assert db.claim_next_queued().id == generation_id
    assert not db.has_queued()


def test_a_job_that_failed_in_the_request_that_queued_it_is_not_retried_unasked(client, test_state, setup_agent):
    provider, args = setup_agent
    provider.responses.append(call("generate_video", args))
    body = request_body()
    assert client.post("/api/assistant/turn", json=body).status_code == 200
    db = SqliteStore(test_state.config.app_data_dir)
    generation = db.claim_next_queued()
    db.mark_failed(generation.id, "EXECUTOR_FAILED", attempt_count=generation.attempt_count)
    provider.responses.extend([call("retry_failed_job", {"generation_id": generation.id}), say("The job failed. Retry it?")])
    assert client.post(f"/api/assistant/runs/{body['request_id']}/continue").json()["status"] == "complete"
    assert test_state.queued_generations.get_generation(generation.id).status == "failed"
    assert not db.has_queued()


def test_retry_failed_job_refuses_a_job_from_another_chat(client, test_state, setup_agent):
    provider, args = setup_agent
    _, generation_id, _ = failed_job_in_chat(client, test_state, provider, args)
    provider.responses.extend([call("retry_failed_job", {"generation_id": generation_id}), say("That job is not in this chat.")])
    _, response = retry_in(client, str(uuid4()))
    assert response.status_code == 200, response.text
    assert test_state.queued_generations.get_generation(generation_id).status == "failed"


def test_retry_failed_job_retries_a_job_at_most_once_per_request(client, test_state, setup_agent):
    provider, args = setup_agent
    conversation_id, generation_id, db = failed_job_in_chat(client, test_state, provider, args)
    provider.responses.append(call("retry_failed_job", {"generation_id": generation_id}))
    body, _ = retry_in(client, conversation_id)
    again = db.claim_next_queued()
    db.mark_failed(again.id, "EXECUTOR_FAILED", attempt_count=again.attempt_count)
    provider.responses.extend([call("retry_failed_job", {"generation_id": generation_id}), say("It failed again.")])
    resumed = client.post(f"/api/assistant/runs/{body['request_id']}/continue")
    assert resumed.json()["status"] == "complete"
    assert test_state.queued_generations.get_generation(generation_id).status == "failed"
    assert not db.has_queued()


def test_a_retried_job_shows_its_card_only_in_the_message_that_retried_it(client, test_state, setup_agent):
    provider, args = setup_agent
    conversation_id, generation_id, _ = failed_job_in_chat(client, test_state, provider, args)
    provider.responses.append(call("retry_failed_job", {"generation_id": generation_id}))
    _, response = retry_in(client, conversation_id)
    assert response.json()["generation_ids"] == [generation_id]
    turns = client.get(f"/api/assistant/conversations/{conversation_id}").json()["turns"]
    assert [turn["response"]["generation_ids"] for turn in turns] == [[], [generation_id]]


def test_stop_cancels_a_failed_job_the_assistant_retried(client, test_state, setup_agent):
    provider, args = setup_agent
    conversation_id, generation_id, _ = failed_job_in_chat(client, test_state, provider, args)
    provider.responses.append(call("retry_failed_job", {"generation_id": generation_id}))
    body, _ = retry_in(client, conversation_id)
    assert client.post(f"/api/assistant/runs/{body['request_id']}/stop").json()["status"] == "stopped"
    assert test_state.queued_generations.get_generation(generation_id).status == "cancelled"


def test_retry_cannot_target_an_unowned_job(client, test_state, setup_agent):
    provider, args = setup_agent
    external = client.post("/api/generations/text-to-video", json={"params": args}).json()
    db = SqliteStore(test_state.config.app_data_dir)
    claimed = db.claim_next_queued()
    db.mark_failed(claimed.id, "INTERRUPTED", attempt_count=claimed.attempt_count)
    provider.responses.extend([
        call("retry_interrupted_generation", {"generation_id": external["id"]}), say("Cannot retry that job automatically."),
    ])
    response = client.post("/api/assistant/turn", json=request_body("Check my video"))
    assert response.status_code == 200, response.text
    assert response.json()["generation_ids"] == []
    assert test_state.queued_generations.get_generation(external["id"]).status == "failed"


def test_run_storage_keeps_stop_monotonic(test_state):
    db = SqliteStore(test_state.config.app_data_dir)
    assert db.insert_assistant_run(run_record("run"), title="", after=None)
    db.save_assistant_run(run_record("run", stopped=True))
    db.save_assistant_run(run_record("run"))
    stored = db.get_assistant_run("run")
    assert stored is not None and stored.stopped


def test_generate_then_extend_uses_the_source_output_and_survives_replay(client, test_state, setup_agent):
    provider, args = setup_agent
    provider.responses.append(call("generate_video", args))
    body = request_body("Generate a wave then extend it by two seconds")
    first = client.post("/api/assistant/turn", json=body)
    assert first.status_code == 200, first.text
    source_id = last_generation(first, client)["id"]
    provider.responses.append(extension_call(source_id))
    finish_job(test_state, source_id)
    source = test_state.queued_generations.get_generation(source_id)
    url = f"/api/assistant/runs/{body['request_id']}/continue"
    second = client.post(url)
    assert second.json()["status"] == "waiting"
    extension = last_generation(second, client)
    assert extension["feature"] == "extend"
    assert extension["spec"]["inputs"]["video"]["assetId"] == source.outputs[0].id
    test_state._bind_store(SqliteStore(test_state.config.app_data_dir))
    test_state.generation_queue.stop()
    assert last_generation(client.post(url), client)["id"] == extension["id"]
    provider.responses.extend([extension_call(source_id), say("Generated and extended your clip.")])
    finish_job(test_state, extension["id"])
    final = client.post(url)
    assert final.json()["status"] == "complete"
    assert final.json()["generation_ids"] == [source_id, extension["id"]]
    assert len(test_state.queued_generations.list_generations("extend")) == 1


def test_extend_to_a_total_length_adds_the_total_instead_of_the_difference(client, test_state, setup_agent):
    provider, args = setup_agent
    provider.responses.append(call("generate_video", args))
    body = request_body("Generate a wave then extend it to a longer clip")
    source_id = last_generation(client.post("/api/assistant/turn", json=body), client)["id"]
    provider.responses.extend([call("extend_video", {"generation_id": source_id, "params": {
        "model": "ltx-2.5-fast", "prompt": "The wave continues rolling", "total_seconds": 8,
    }}), say("Extended.")])
    finish_job(test_state, source_id)
    source_ms = test_state.queued_generations.get_generation(source_id).outputs[0].metadata.metadata.durationMs
    extension = last_generation(client.post(f"/api/assistant/runs/{body['request_id']}/continue"), client)
    assert extension["feature"] == "extend"
    assert extension["spec"]["params"]["duration"] == round(8 - source_ms / 1000, 2)


def test_retake_past_the_source_end_is_refused_with_the_start_time_that_fits(client, test_state, tmp_path, setup_agent):
    provider, _args = setup_agent
    video = ingest(client, _mp4(tmp_path / "clip.mp4", duration_seconds=4))
    provider.responses.extend([call("retake_video", {"video_asset_id": video["id"], "params": {
        "model": "ltx-2.5-fast", "startTime": 3, "duration": 2,
    }}), say("That range runs past the end.")])
    body = request_body("Retake the last 2 seconds")
    response = client.post("/api/assistant/turn", json=body)
    assert response.status_code == 200, response.text
    assert last_generation(response, client) is None
    assert "use startTime 2.0" in saved_history(test_state, body["request_id"])


def test_a_retake_cannot_resize_the_source_video(client, tmp_path, setup_agent):
    provider, _args = setup_agent
    video = ingest(client, _mp4(tmp_path / "clip.mp4", duration_seconds=4))
    params = {"model": "ltx-2.5-fast", "startTime": 0, "duration": 2, "mode": "replace_audio"}
    provider.responses.extend([
        call("retake_video", {"video_asset_id": video["id"], "params": {**params, "resolution": {"width": 480, "height": 270}}}),
        call("retake_video", {"video_asset_id": video["id"], "params": params}),
        say("Replacing the audio."),
    ])
    response = client.post("/api/assistant/turn", json=request_body("New audio for the first 2 seconds, keep the video unchanged"))
    assert response.status_code == 200, response.text
    retake = last_generation(response, client)
    assert retake is not None
    assert retake["spec"]["params"]["resolution"] is None


@pytest.mark.parametrize("status", ["queued", "failed", "cancelled"])
def test_extend_rejects_unfinished_or_unsuccessful_source(client, test_state, setup_agent, status):
    provider, args = setup_agent
    source = client.post("/api/generations/text-to-video", json={"params": args}).json()
    db = SqliteStore(test_state.config.app_data_dir)
    if status == "failed":
        job = db.claim_next_queued()
        db.mark_failed(job.id, "EXECUTOR_FAILED", attempt_count=job.attempt_count)
    elif status == "cancelled":
        client.post(f"/api/generations/{source['id']}/cancel")
    provider.responses.extend([extension_call(source["id"]), say("The source is not ready to extend.")])
    response = client.post("/api/assistant/turn", json=request_body("Extend the existing clip"))
    assert response.status_code == 200, response.text
    assert test_state.queued_generations.list_generations("extend") == []


def test_asset_paths_never_reach_history_and_only_shown_assets_are_displayed(client, test_state, tmp_path, setup_agent):
    provider, _ = setup_agent
    image = ingest(client, _png(tmp_path / "input.png"))
    other = ingest(client, _png(tmp_path / "other.png"))
    provider.responses.extend([
        call("list_assets", {"media_kind": "image", "limit": 1}),
        call("get_asset", {"asset_id": other["id"]}),
        call("show_assets", {"asset_ids": [image["id"], image["id"]]}), say("Found your input image."),
    ])
    body = request_body("Find my input image")
    response = client.post("/api/assistant/turn", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["asset_ids"] == [image["id"]]
    history = saved_history(test_state, body["request_id"])
    assert other["id"] in history
    assert image["path"] not in history and str(tmp_path) not in history


def test_list_assets_without_a_sort_returns_the_newest_video_first(client, test_state, setup_agent):
    provider, args = setup_agent
    outputs = []
    for _ in range(2):
        job_id = client.post("/api/generations/text-to-video", json={"params": args}).json()["id"]
        finish_job(test_state, job_id)
        outputs.append(test_state.queued_generations.get_generation(job_id).outputs[0].id)
        time.sleep(0.01)
    provider.responses.extend([call("list_assets", {"media_kind": "video", "limit": 1}), say("Found it.")])
    body = request_body("Find my last video")
    client.post("/api/assistant/turn", json=body)
    assert [item["id"] for item in tool_return(test_state, body["request_id"], "list_assets")["items"]] == [outputs[-1]]


def test_a_job_tool_given_an_asset_id_says_it_is_an_asset_id(client, test_state, tmp_path, setup_agent):
    provider, _ = setup_agent
    video = ingest(client, _mp4(tmp_path / "turtle.mp4", duration_seconds=2))
    provider.responses.extend([call("get_generation_status", {"generation_id": video["id"]}), say("Checking.")])
    body = request_body("How did my turtle video turn out?")
    client.post("/api/assistant/turn", json=body)
    assert "That is an asset ID, not a job ID" in saved_history(test_state, body["request_id"])


def test_audio_to_video_derives_duration_from_existing_audio(client, tmp_path, setup_agent):
    provider, _ = setup_agent
    audio = ingest(client, _wav(tmp_path / "sound.wav"))
    params = {"prompt": "Waves moving in time", "model": "ltx-2.5-fast", "resolution": "540p", "fps": 24}
    provider.responses.append(call("audio_to_video", {"params": params, "inputs": {"audio": {"assetId": audio["id"]}}}))
    response = client.post("/api/assistant/turn", json=request_body("Make a video to match my audio"))
    assert response.status_code == 200, response.text
    generation = last_generation(response, client)
    assert generation["feature"] == "audio-to-video"
    assert generation["spec"]["params"]["numFrames"] > 0


@pytest.mark.parametrize("tool", ["image_to_video", "audio_to_video", "retake_video", "extend_video"])
def test_media_tools_reject_wrong_kind_without_enqueuing(client, test_state, tmp_path, setup_agent, tool):
    provider, args = setup_agent
    image = ingest(client, _png(tmp_path / "still.png"))
    audio = ingest(client, _wav(tmp_path / "audio.wav"))
    if tool == "image_to_video":
        arguments = {"params": args, "inputs": {"startFrame": {"assetId": audio["id"]}}}
    elif tool == "audio_to_video":
        arguments = {"params": {"prompt": "Test", "model": "ltx-2.5-fast"}, "inputs": {"audio": {"assetId": image["id"]}}}
    else:
        arguments = {"video_asset_id": image["id"], "params": {"model": "ltx-2.5-fast", "duration": 2}}
        if tool == "retake_video":
            arguments["params"]["startTime"] = 0
        else:
            arguments["params"] = {"model": "ltx-2.5-fast", "added_seconds": 2}
    provider.responses.extend([call(tool, arguments), say("Please select an appropriate input asset.")])
    response = client.post("/api/assistant/turn", json=request_body("Use my selected asset"))
    assert response.status_code == 200, response.text
    assert last_generation(response, client) is None
    assert not SqliteStore(test_state.config.app_data_dir).has_queued()


@pytest.mark.parametrize("recipe", queued_lora_recipes(), ids=lambda recipe: recipe.recipe_id)
def test_every_recipe_uses_existing_registry_and_device_gate(client, test_state, tmp_path, setup_agent, fake_services, recipe):
    provider, args = setup_agent
    item = fake_services.lora_catalog_provider.get_lora(recipe.catalog_id)
    assert item is not None
    path = resolve_lora_path(test_state.config.default_models_dir, recipe.catalog_id, item.download.default_variant().filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-weights")
    request = {"params": {**args, "scale": 0.8}}
    if recipe.mode == "i2v":
        image = ingest(client, _png(tmp_path / "frame.png"))
        request["inputs"] = {"startFrame": {"assetId": image["id"]}}
        if recipe.requires_end_frame:
            request["inputs"]["endFrame"] = {"assetId": image["id"]}
    provider.responses.extend([call("run_lora_recipe", {"recipe_id": recipe.recipe_id, "request": request}), say("This hardware cannot run the recipe.")])
    body = request_body(f"Run {recipe.recipe_id} with my input")
    response = client.post("/api/assistant/turn", json=body)
    assert response.status_code == 200, response.text
    if accelerator_backend() not in recipe.devices:
        # CPU-only CI still exercises the real device gate, not a patched gate.
        assert last_generation(response, client) is None
        assert "LORA_UNSUPPORTED_DEVICE" in saved_history(test_state, body["request_id"])
    else:
        generation = last_generation(response, client)
        assert generation["feature"] == recipe.recipe_id
        assert generation["spec"]["params"]["loras"] == [
            {
                "ref": "",
                "scale": 0.8,
                "catalogId": recipe.catalog_id,
                "displayName": item.name,
                "variantId": item.download.default_variant().id,
            }
        ]


def test_generate_video_without_resolution_is_sent_back_to_the_model_instead_of_queued_at_1080p(client, test_state, setup_agent):
    provider, args = setup_agent
    params = {key: value for key, value in args.items() if key != "resolution"}
    provider.responses.extend([call("generate_video", {"params": params, "job_number": 1}), say("Which resolution?")])
    body = request_body("Two 270p clips of a rolling wave")
    response = client.post("/api/assistant/turn", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["generation_ids"] == []
    retries = [part for message in json.loads(saved_history(test_state, body["request_id"]))
               for part in message.get("parts", []) if part.get("part_kind") == "retry-prompt"]
    assert retries and "resolution" in json.dumps(retries)


def test_unsupported_duration_tells_the_model_the_valid_durations_at_that_resolution(client, test_state, setup_agent):
    provider, args = setup_agent
    provider.responses.extend([
        call("generate_video", {"params": {**args, "resolution": "1080p", "duration": 20, "fps": 24}, "job_number": 1}),
        say("That length is not available."),
    ])
    body = request_body("A 20 second 1080p wave")
    client.post("/api/assistant/turn", json=body)
    history = saved_history(test_state, body["request_id"])
    assert "supports durations 2, 3, 4, 5, 6, 8, 10" in history
    assert "duration=20" in history


def rejected_clip(args, job_number):
    return call("generate_video", {"params": {**args, "resolution": "1080p", "duration": 20, "fps": 24}, "job_number": job_number})


def tool_returns(test_state, run_id):
    return [part for message in json.loads(saved_history(test_state, run_id))
            for part in message.get("parts", []) if part.get("part_kind") == "tool-return"]


def test_a_tool_that_runs_out_of_retries_ends_the_turn_with_its_last_error_not_a_502(client, test_state, setup_agent):
    provider, args = setup_agent
    provider.responses.extend(rejected_clip(args, 1) for _ in range(5))
    body = request_body("A 20 second 1080p wave")
    response = client.post("/api/assistant/turn", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "complete"
    assert "supports durations 2, 3, 4, 5, 6, 8, 10" in response.json()["text"]
    [closed] = tool_returns(test_state, body["request_id"])
    assert closed["content"]["status"] == "failed"
    assert "supports durations 2, 3, 4, 5, 6, 8, 10" in closed["content"]["message"]


def test_jobs_queued_before_retries_ran_out_are_kept(client, test_state, setup_agent):
    provider, args = setup_agent
    provider.responses.extend([call("generate_video", {"params": args, "job_number": 1}), *(rejected_clip(args, 2) for _ in range(5))])
    response = client.post("/api/assistant/turn", json=request_body("A wave, then a 6 second 1080p wave"))
    assert response.status_code == 200, response.text
    assert len(response.json()["generation_ids"]) == 1
    assert "Jobs already queued keep running." in response.json()["text"]


def test_generate_video_still_queues_after_four_rejected_attempts(client, test_state, setup_agent):
    provider, args = setup_agent
    provider.responses.extend([*(rejected_clip(args, 1) for _ in range(4)), call("generate_video", {"params": args, "job_number": 1}), say("Queued.")])
    response = client.post("/api/assistant/turn", json=request_body("A wave"))
    assert response.status_code == 200, response.text
    assert len(response.json()["generation_ids"]) == 1


def queued_provenance(client, test_state, provider, args, auto_enhance):
    test_state.state.app_settings.explore_auto_enhance_prompts = auto_enhance
    provider.responses.extend([call("generate_video", {"params": args, "job_number": 1}), say("Queued.")])
    generation = last_generation(client.post("/api/assistant/turn", json=request_body("A wave")), client)
    assert generation is not None
    return generation["spec"]["params"]["promptProvenance"]


def test_assistant_job_with_auto_enhance_on_is_left_for_the_enhancer_to_rewrite(client, test_state, setup_agent):
    provider, args = setup_agent
    assert queued_provenance(client, test_state, provider, args, auto_enhance=True) == "typed"


def test_assistant_job_with_auto_enhance_off_stays_enhanced_so_recipes_keep_their_wrap(client, test_state, setup_agent):
    provider, args = setup_agent
    assert queued_provenance(client, test_state, provider, args, auto_enhance=False) == "enhanced"


def install_variants(test_state, fake_services, catalog_id, base_models, *, ic=True):
    """Gives the catalog entry an LTX-2.3 and an LTX-2.5 variant and installs the weights for `base_models`."""
    download = DownloadSpec(repo_id="org/two", variants=[
        DownloadVariant(id=f"ltx-2.5__{catalog_id}", label="LTX-2.5", filename="new.safetensors", size_bytes=10, base_model="LTX-2.5"),
        DownloadVariant(id=f"ltx-2.3__{catalog_id}", label="LTX-2.3", filename="old.safetensors", size_bytes=10, base_model="LTX-2.3"),
    ])
    if ic:
        add_ic_lora(fake_services, id=catalog_id, name="Day to Night", download=download,
                    default_settings=IcLoraSettings(skip_stage_2=True, resolution_factor=1.5, audio_mode="source"))
    else:
        item = fake_services.lora_catalog_provider.get_lora(catalog_id)
        assert item is not None
        item.download = download
    resolve = resolve_ic_lora_path if ic else resolve_lora_path
    for variant in download.variants:
        if variant.base_model in base_models:
            path = resolve(test_state.config.default_models_dir, catalog_id, variant.filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake-weights")
    return {variant.base_model: variant.id for variant in download.variants}


def run_day_to_night(client, provider, tmp_path):
    video = ingest(client, _mp4(tmp_path / "street.mp4"))
    provider.responses.extend([
        call("run_video_recipe", {"recipe_id": "day-to-night", "job_number": 1, "video_asset_id": video["id"],
                                  "params": {"model": "ltx-2.5-fast", "prompt": "The street at night", "resolution": "540p"}}),
        say("Queued."),
    ])
    body = request_body("Make my street video night-time")
    response = client.post("/api/assistant/turn", json=body)
    assert response.status_code == 200, response.text
    return body, last_generation(response, client)


def test_video_recipe_with_both_variants_installed_runs_the_one_matching_the_model(client, test_state, tmp_path, fake_services, setup_agent):
    provider, _ = setup_agent
    variants = install_variants(test_state, fake_services, "day-to-night", ("LTX-2.3", "LTX-2.5"))
    body, generation = run_day_to_night(client, provider, tmp_path)
    assert generation is not None
    assert generation["spec"]["params"]["loras"][0]["variantId"] == variants["LTX-2.5"]


def test_video_recipe_with_only_the_other_familys_variant_runs_it_and_tells_the_model(client, test_state, tmp_path, fake_services, setup_agent):
    provider, _ = setup_agent
    variants = install_variants(test_state, fake_services, "day-to-night", ("LTX-2.3",))
    body, generation = run_day_to_night(client, provider, tmp_path)
    assert generation is not None
    assert generation["spec"]["params"]["loras"][0]["variantId"] == variants["LTX-2.3"]
    assert "Only the LTX-2.3 variant" in tool_return(test_state, body["request_id"], "run_video_recipe")["note"]


def test_assistant_queued_jobs_are_not_reported_as_requested_by_the_user(client, test_state, tmp_path, fake_services, setup_agent):
    # run_video_recipe queues through its own create wrapper, so it can drop the tag separately.
    provider, args = setup_agent
    install_variants(test_state, fake_services, "day-to-night", ("LTX-2.5",))
    video = ingest(client, _mp4(tmp_path / "street.mp4"))
    provider.responses.extend([
        call("generate_video", {"params": args, "job_number": 1}),
        call("run_video_recipe", {"recipe_id": "day-to-night", "job_number": 2, "video_asset_id": video["id"],
                                  "params": {"model": "ltx-2.5-fast", "prompt": "The street at night", "resolution": "540p"}}),
        say("Queued both."),
    ])
    response = client.post("/api/assistant/turn", json=request_body("Make a wave clip, and make my street video night-time"))
    assert response.status_code == 200, response.text
    ids = response.json()["generation_ids"]
    assert len(ids) == 2
    store = SqliteStore(test_state.config.app_data_dir)
    for generation_id in ids:
        record = store.get_generation(generation_id)
        assert record is not None
        details = queued_generation_details(record)
        assert details is not None
        assert details["requested_by"] == "assistant"


def test_lora_recipe_with_two_installed_variants_runs_the_matching_one(client, test_state, tmp_path, fake_services, setup_agent):
    recipe = get_lora_recipe("cinemagraph")
    assert recipe is not None
    if accelerator_backend() not in recipe.devices:
        pytest.skip("Cinemagraph does not run on this device")
    provider, args = setup_agent
    variants = install_variants(test_state, fake_services, "cinemagraph-motion", ("LTX-2.3", "LTX-2.5"), ic=False)
    image = ingest(client, _png(tmp_path / "frame.png"))
    provider.responses.extend([
        call("run_lora_recipe", {"recipe_id": "cinemagraph", "request": {"params": args, "inputs": {"startFrame": {"assetId": image["id"]}}}}),
        say("Queued."),
    ])
    response = client.post("/api/assistant/turn", json=request_body("Make this photo a cinemagraph"))
    assert response.status_code == 200, response.text
    generation = last_generation(response, client)
    assert generation is not None
    assert generation["spec"]["params"]["loras"][0]["variantId"] == variants["LTX-2.5"]


def test_a_rejected_recipe_reaches_the_model_with_its_message_and_no_job_number_advice(client, test_state, tmp_path, fake_services, setup_agent):
    provider, _ = setup_agent
    install_variants(test_state, fake_services, "day-to-night", ("LTX-2.5",))
    video = ingest(client, _mp4(tmp_path / "street.mp4"))
    provider.responses.extend([
        call("run_video_recipe", {"recipe_id": "day-to-night", "job_number": 1, "video_asset_id": video["id"],
                                  "params": {"model": "ltx-2.5-fast", "prompt": "", "resolution": "540p"}}),
        say("It needs a prompt."),
    ])
    body = request_body("Make my street video night-time")
    client.post("/api/assistant/turn", json=body)
    history = saved_history(test_state, body["request_id"])
    assert "Prompt is required." in history
    assert "Reused job numbers" not in history


def test_recipe_guidance_uses_catalog_id_not_recipe_id(client, test_state, setup_agent):
    provider, _ = setup_agent
    provider.responses.extend([call("get_recipe_guidance", {"recipe_id": "cozy-felt"}), say("Ready.")])
    body = request_body("How do I prompt Cozy Felt?")
    response = client.post("/api/assistant/turn", json=body)
    assert response.status_code == 200, response.text
    history = saved_history(test_state, body["request_id"])
    assert "Cozy Felt Style" in history
    assert "Unknown recipe" not in history


def test_named_recipe_sent_to_generate_video_is_redirected_to_run_lora_recipe(client, test_state, setup_agent):
    provider, args = setup_agent
    provider.responses.extend([call("generate_video", args), say("Use the recipe.")])
    body = request_body("Generate a claymation video of a turtle on a skateboard")
    response = client.post("/api/assistant/turn", json=body)
    assert response.status_code == 200, response.text
    assert last_generation(response, client) is None
    assert "run_lora_recipe, recipe_id=claymation" in saved_history(test_state, body["request_id"])


def test_named_recipe_guard_lets_a_repeated_plain_generate_video_through(client, test_state, setup_agent):
    provider, args = setup_agent
    provider.responses.extend([call("generate_video", args), call("generate_video", args), say("Queued without the recipe.")])
    response = client.post("/api/assistant/turn", json=request_body("A claymation-like wave, but no recipe please"))
    assert response.status_code == 200, response.text
    assert last_generation(response, client)["feature"] == "text-to-video"


def test_a_plain_clip_requested_alongside_a_recipe_clip_is_queued_without_the_recipe(client, test_state, fake_services, setup_agent):
    recipe = get_lora_recipe("cozy-felt")
    assert recipe is not None
    if accelerator_backend() not in recipe.devices:
        pytest.skip("Cozy Felt does not run on this device")
    provider, args = setup_agent
    item = fake_services.lora_catalog_provider.get_lora(recipe.catalog_id)
    assert item is not None
    path = resolve_lora_path(test_state.config.default_models_dir, recipe.catalog_id, item.download.default_variant().filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-weights")
    provider.responses.extend([
        call("run_lora_recipe", {"recipe_id": "cozy-felt", "job_number": 1, "request": {"params": args}}),
        call("generate_video", {"params": args, "job_number": 2}),
        say("Queued both."),
    ])
    response = client.post("/api/assistant/turn", json=request_body("One Cozy Felt video of a fox and one plain video of a cube"))
    assert response.status_code == 200, response.text
    features = [client.get(f"/api/generations/{id}").json()["feature"] for id in response.json()["generation_ids"]]
    assert features[1:] == ["text-to-video"] and len(features) == 2


def test_ordinary_words_like_transition_or_restore_do_not_redirect_a_text_to_video_request(client, setup_agent):
    provider, args = setup_agent
    provider.responses.extend([call("generate_video", args), say("Queued.")])
    body = request_body("A smooth transition over a harbour from day to night, as the king starts to restore his city")
    response = client.post("/api/assistant/turn", json=body)
    assert response.status_code == 200, response.text
    assert last_generation(response, client)["feature"] == "text-to-video"


def test_a_video_recipe_named_without_an_attached_video_does_not_redirect_generate_video(client, setup_agent):
    provider, args = setup_agent
    provider.responses.extend([call("generate_video", args), say("Queued.")])
    response = client.post("/api/assistant/turn", json=request_body("A racing car with a deblur effect"))
    assert response.status_code == 200, response.text
    assert last_generation(response, client)["feature"] == "text-to-video"


def test_recipe_nudge_fires_once_per_request_across_continuations(client, test_state, setup_agent):
    provider, args = setup_agent
    provider.responses.extend([call("generate_video", args), call("generate_video", args)])
    body = request_body("A claymation turtle, but no recipe please")
    first = client.post("/api/assistant/turn", json=body)
    assert first.status_code == 200, first.text
    job = last_generation(first, client)
    provider.responses.extend([call("generate_video", {"params": args, "job_number": 2}), say("Queued a second clip.")])
    finish_job(test_state, job["id"])
    second = client.post(f"/api/assistant/runs/{body['request_id']}/continue")
    assert second.status_code == 200, second.text
    assert len(second.json()["generation_ids"]) == 2


def test_recipe_nudge_does_not_use_up_the_retry_for_a_real_settings_error(client, setup_agent):
    provider, args = setup_agent
    unsupported = {**args, "aspectRatio": "auto"}
    provider.responses.extend([
        call("generate_video", args), call("generate_video", unsupported), call("generate_video", unsupported),
        call("generate_video", args), say("Queued without the recipe."),
    ])
    response = client.post("/api/assistant/turn", json=request_body("A claymation turtle, but no recipe please"))
    assert response.status_code == 200, response.text
    assert last_generation(response, client)["feature"] == "text-to-video"


def test_previous_settings_skip_a_clip_the_user_cancelled(client, test_state, setup_agent):
    provider, args = setup_agent
    provider.responses.append(call("generate_video", {**args, "aspectRatio": "1:1"}))
    body = request_body("A square wave")
    kept = last_generation(client.post("/api/assistant/turn", json=body), client)
    finish_job(test_state, kept["id"])
    client.post(f"/api/assistant/runs/{body['request_id']}/continue")
    provider.responses.append(call("generate_video", {**args, "aspectRatio": "9:16"}))
    tall = {**request_body("Now a tall one"), "conversation_id": body["conversation_id"]}
    cancelled = last_generation(client.post("/api/assistant/turn", json=tall), client)
    assert client.post(f"/api/generations/{cancelled['id']}/cancel").status_code == 200
    client.post(f"/api/assistant/runs/{tall['request_id']}/continue")
    provider.responses.append(say("Which settings?"))
    follow_up = {**request_body("Make another one"), "conversation_id": body["conversation_id"]}
    assert client.post("/api/assistant/turn", json=follow_up).status_code == 200
    assert '"aspectRatio":"1:1"' in provider.instructions[-1]


def test_previous_auto_aspect_ratio_becomes_the_finished_clips_ratio(client, test_state, tmp_path, setup_agent):
    provider, args = setup_agent
    image = ingest(client, _png(tmp_path / "frame.png"))
    provider.responses.append(call("image_to_video", {"params": {**args, "aspectRatio": "auto"}, "inputs": {"startFrame": {"assetId": image["id"]}}}))
    body = request_body("Animate this frame")
    clip = last_generation(client.post("/api/assistant/turn", json=body), client)
    assert clip["spec"]["params"]["aspectRatio"] == "auto"
    finish_job(test_state, clip["id"])
    client.post(f"/api/assistant/runs/{body['request_id']}/continue")
    provider.responses.extend([call("generate_video", {**args, "aspectRatio": "1:1"}), say("Queued.")])
    follow_up = {**request_body("Now a text-only clip with the same settings"), "conversation_id": body["conversation_id"]}
    response = client.post("/api/assistant/turn", json=follow_up)
    assert response.status_code == 200, response.text
    instructions = provider.instructions[-1]
    assert '"aspectRatio":"auto"' not in instructions
    assert '"aspectRatio":"1:1"' in instructions


def test_assistant_video_schema_offers_exactly_the_local_spec_values():
    resolutions: set[str] = set()
    durations: set[float] = set()
    for model_id in get_args(LTXLocalModelId):
        for _, spec in get_ltx_model_spec(model_id).supported_pipelines:
            for table in (spec.supported_resolutions_durations, spec.a2v_supported_resolutions_durations or {}):
                resolutions.update(table)
                for cell in table.values():
                    durations.update(d for per_fps in cell.fps_to_durations.values() for d in per_fps)
    assert set(get_args(LocalVideoResolution)) == resolutions
    assert set(get_args(LocalVideoDuration)) == durations


def test_new_source_tool_cannot_duplicate_existing_source(client, test_state, tmp_path, setup_agent):
    provider, args = setup_agent
    image = ingest(client, _png(tmp_path / "frame.png"))
    provider.responses.append(call("generate_video", args))
    body = request_body()
    generation = last_generation(client.post("/api/assistant/turn", json=body), client)
    provider.responses.extend([call("image_to_video", {"params": args, "inputs": {"startFrame": {"assetId": image["id"]}}}), say("The source clip is already generated.")])
    finish_job(test_state, generation["id"])
    response = client.post(f"/api/assistant/runs/{body['request_id']}/continue")
    assert response.status_code == 200, response.text
    assert test_state.queued_generations.list_generations("image-to-video") == []


def test_numbered_jobs_survive_model_failure_without_duplicates(client, test_state, setup_agent):
    provider, args = setup_agent
    body = request_body("Generate four clips")
    provider.responses.extend([
        call("generate_video", {"params": args, "job_number": 1}),
        call("generate_video", {"params": args, "job_number": 2}),
        RuntimeError("connection interrupted"),
    ])
    assert client.post("/api/assistant/turn", json=body).status_code == 502
    provider.responses.extend([call("generate_video", {"params": args, "job_number": n}) for n in range(1, 5)])
    provider.responses.append(say("Queued four."))
    response = client.post("/api/assistant/turn", json=body)
    assert response.status_code == 200, response.text
    assert len(response.json()["generation_ids"]) == 4
    assert len(test_state.queued_generations.list_generations("text-to-video")) == 4


def test_reused_job_number_rejects_changed_inputs(client, test_state, setup_agent):
    provider, args = setup_agent
    provider.responses.extend([
        call("generate_video", args),
        call("generate_video", {**args, "prompt": "Different video"}), say("One clip queued."),
    ])
    response = client.post("/api/assistant/turn", json=request_body())
    assert response.status_code == 200, response.text
    assert len(response.json()["generation_ids"]) == 1
    assert last_generation(response, client)["spec"]["params"]["prompt"] == args["prompt"]
    assert len(test_state.queued_generations.list_generations()) == 1


def test_removed_numbered_job_is_not_recreated(client, test_state, setup_agent):
    provider, args = setup_agent
    provider.responses.append(call("generate_video", args))
    body = request_body()
    generation_id = last_generation(client.post("/api/assistant/turn", json=body), client)["id"]
    client.post(f"/api/generations/{generation_id}/cancel")
    assert client.delete(f"/api/generations/{generation_id}").status_code == 200
    provider.responses.extend([call("generate_video", args), say("The job was removed.")])
    response = client.post(f"/api/assistant/runs/{body['request_id']}/continue")
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "complete"
    assert test_state.queued_generations.list_generations("text-to-video") == []


def assistant_job_id(body, job_number):
    return str(uuid5(NAMESPACE_URL, f"ltx-desktop-assistant:{body['request_id']}:{job_number}"))


def test_cancelling_the_only_job_in_the_queue_ends_the_run_without_calling_the_model(client, test_state, setup_agent):
    provider, args = setup_agent
    provider.responses.extend([call("generate_video", args), say("Queued.")])
    body = request_body()
    assert client.post("/api/assistant/turn", json=body).json()["status"] == "waiting"
    model_calls = len(provider.instructions)
    provider.responses.extend([call("generate_video", {"params": args, "job_number": 2}), say("Queued it again.")])
    client.post(f"/api/generations/{assistant_job_id(body, 1)}/cancel")
    response = client.post(f"/api/assistant/runs/{body['request_id']}/continue").json()
    assert (response["status"], response["text"]) == ("complete", "Job 1 was cancelled.")
    assert len(provider.instructions) == model_calls
    assert len(test_state.queued_generations.list_generations()) == 1


def test_cancelling_one_of_three_jobs_lets_the_others_finish_and_queues_nothing_new(client, test_state, setup_agent):
    provider, args = setup_agent
    body = request_body("Generate three clips")
    provider.responses.extend([
        *(call("generate_video", {"params": args, "job_number": number}) for number in (1, 2, 3)),
        call("wait_for_generation", {"generation_id": assistant_job_id(body, 2)}),
    ])
    assert client.post("/api/assistant/turn", json=body).json()["status"] == "waiting"
    model_calls = len(provider.instructions)
    provider.responses.append(call("generate_video", {"params": args, "job_number": 4}))
    client.post(f"/api/generations/{assistant_job_id(body, 2)}/cancel")
    url = f"/api/assistant/runs/{body['request_id']}/continue"
    assert client.post(url).json()["status"] == "waiting"
    finish_job(test_state, assistant_job_id(body, 3))
    response = client.post(url).json()
    assert (response["status"], response["text"]) == ("complete", "Job 2 was cancelled. Jobs 1 and 3 finished.")
    assert len(provider.instructions) == model_calls
    assert len(test_state.queued_generations.list_generations()) == 3
    # The closed wait call must leave a history the next message can build on.
    provider.responses.clear()
    provider.responses.append(say("Hello again."))
    follow_up = {**request_body("Hello"), "conversation_id": body["conversation_id"]}
    assert client.post("/api/assistant/turn", json=follow_up).json()["text"] == "Hello again."


def test_cancelling_one_job_keeps_the_finished_result_of_the_other(client, test_state, setup_agent):
    provider, args = setup_agent
    body = request_body("Generate two clips")
    first, second = assistant_job_id(body, 1), assistant_job_id(body, 2)
    waits = [call("wait_for_generation", {"generation_id": job_id}, call_id=f"wait-{job_id}") for job_id in (first, second)]
    provider.responses.extend([
        *(call("generate_video", {"params": args, "job_number": number}) for number in (1, 2)),
        ModelResponse(parts=[part for wait in waits for part in wait.parts]),
    ])
    assert client.post("/api/assistant/turn", json=body).json()["status"] == "waiting"
    finish_job(test_state, first)
    client.post(f"/api/generations/{second}/cancel")
    response = client.post(f"/api/assistant/runs/{body['request_id']}/continue").json()
    assert (response["status"], response["text"]) == ("complete", "Job 2 was cancelled. Job 1 finished.")
    returns = {part["tool_call_id"]: part["content"] for part in tool_returns(test_state, body["request_id"])}
    assert returns[f"wait-{first}"]["status"] == "succeeded"
    assert returns[f"wait-{second}"]["status"] == "cancelled"


def test_reusing_the_job_number_of_a_cancelled_job_is_refused_as_final(client, test_state, setup_agent):
    provider, args = setup_agent
    body = request_body()

    def cancel_then_requeue():
        test_state.queued_generations.cancel_generation(assistant_job_id(body, 1))
        return call("generate_video", args)

    provider.responses.extend([call("generate_video", args), cancel_then_requeue, say("Job 1 stays cancelled.")])
    assert client.post("/api/assistant/turn", json=body).status_code == 200
    assert "The user cancelled job 1. Do not queue it again" in saved_history(test_state, body["request_id"])
    assert [generation.status for generation in test_state.queued_generations.list_generations()] == ["cancelled"]


def tool_return(test_state, run_id, tool_name):
    """The last result the model received from `tool_name` in this run."""
    parts = [part for message in json.loads(saved_history(test_state, run_id)) for part in message["parts"]
             if part.get("part_kind") == "tool-return" and part.get("tool_name") == tool_name]
    return parts[-1]["content"]


EPOCH_KEYS = ("created_at", "queued_at", "started_at", "finished_at")


def test_job_result_reports_duration_in_seconds_and_no_epoch_timestamps(client, test_state, setup_agent):
    provider, args = setup_agent
    provider.responses.extend([call("generate_video", args), say("Queued.")])
    body = request_body()
    client.post("/api/assistant/turn", json=body)
    job_id = assistant_job_id(body, 1)
    finish_job(test_state, job_id)
    # The finished job wakes the first run; it must settle before the next turn's replies are scripted.
    deadline = time.monotonic() + 5
    while client.get(f"/api/assistant/runs/{body['request_id']}").json()["status"] != "complete":
        assert time.monotonic() < deadline
        time.sleep(0.02)
    conn = SqliteStore(test_state.config.app_data_dir)._connect()
    with conn:
        conn.execute("UPDATE generations SET started_at = ?, finished_at = ? WHERE id = ?", (1_791_803_483_029, 1_791_803_736_415, job_id))
    conn.close()
    provider.responses.clear()
    provider.responses.extend([call("get_generation_status", {"generation_id": job_id}), say("It took 4 minutes 13 seconds.")])
    status = request_body("How long did that take?")
    client.post("/api/assistant/turn", json=status)
    result = tool_return(test_state, status["request_id"], "get_generation_status")
    assert result["took_seconds"] == 253.4
    assert not any(key in json.dumps(result) for key in EPOCH_KEYS)


def finished_extension(client, test_state, tmp_path, provider):
    """An extension of an imported clip that finished; returns the clip and the extension's job ID."""
    video = ingest(client, _mp4(tmp_path / "turtle.mp4", duration_seconds=4))
    provider.responses.extend([
        call("extend_video", {"video_asset_id": video["id"], "params": {"model": "ltx-2.5-fast", "added_seconds": 2}}),
        say("Extending it."), say("It is done."),
    ])
    body = request_body("Extend my turtle video by 2 seconds")
    job_id = last_generation(client.post("/api/assistant/turn", json=body), client)["id"]
    finish_job(test_state, job_id)
    deadline = time.monotonic() + 5
    while client.get(f"/api/assistant/runs/{body['request_id']}").json()["status"] != "complete":
        assert time.monotonic() < deadline
        time.sleep(0.02)
    return video, job_id


def job_status_result(client, test_state, provider, job_id):
    provider.responses.extend([call("get_generation_status", {"generation_id": job_id}), say("Here it is.")])
    body = request_body("How did the extension turn out?")
    client.post("/api/assistant/turn", json=body)
    return tool_return(test_state, body["request_id"], "get_generation_status")


def test_extend_result_names_its_source_video_and_the_new_clip_length(client, test_state, tmp_path, setup_agent):
    provider, _ = setup_agent
    video, job_id = finished_extension(client, test_state, tmp_path, provider)
    meta = test_state.queued_generations.get_generation(job_id).outputs[0].metadata.metadata
    result = job_status_result(client, test_state, provider, job_id)
    assert result["source"] == {"id": video["id"], "name": video["name"]}
    assert result["clip_seconds"] == round(meta.durationMs / 1000, 2)
    assert result["size"] == f"{meta.width}x{meta.height}"


def test_a_job_whose_source_file_is_gone_reports_the_source_as_unavailable(client, test_state, tmp_path, setup_agent):
    # The app refuses to delete a job's source, but the file can still be removed outside it.
    provider, _ = setup_agent
    video, job_id = finished_extension(client, test_state, tmp_path, provider)
    record = SqliteStore(test_state.config.app_data_dir).get_asset(video["id"], tombstone_missing=False)
    assert record is not None
    Path(record.path).unlink()
    assert job_status_result(client, test_state, provider, job_id)["source"] == "unavailable"


def test_a_job_that_has_not_finished_has_no_duration(client, test_state, setup_agent):
    provider, args = setup_agent
    body = request_body()
    provider.responses.extend([
        call("generate_video", args), call("get_generation_status", {"generation_id": assistant_job_id(body, 1)}), say("Queued."),
    ])
    assert client.post("/api/assistant/turn", json=body).status_code == 200
    result = tool_return(test_state, body["request_id"], "get_generation_status")
    assert (result["status"], result["took_seconds"], result["finished"]) == ("queued", None, None)


def test_recover_enqueue_before_assistant_state_was_saved(client, test_state, setup_agent):
    provider, args = setup_agent
    body = request_body()
    generation_id = str(uuid5(NAMESPACE_URL, f"ltx-desktop-assistant:{body['request_id']}:1"))
    test_state.queued_generations.create_text_to_video(
        CreateTextToVideoRequest(params=TextToVideoParams.model_validate(args)), generation_id=generation_id,
    )
    provider.responses.append(call("generate_video", args))
    response = client.post("/api/assistant/turn", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["generation_ids"] == [generation_id]
    assert len(test_state.queued_generations.list_generations("text-to-video")) == 1


@pytest.mark.parametrize("job_number", [0, 13])
def test_job_number_is_bounded(client, test_state, setup_agent, job_number):
    provider, args = setup_agent
    provider.responses.extend([call("generate_video", {"params": args, "job_number": job_number}), say("Invalid job number.")])
    response = client.post("/api/assistant/turn", json=request_body())
    assert response.status_code == 200
    assert response.json()["generation_ids"] == []
    assert test_state.queued_generations.list_generations("text-to-video") == []


def test_all_jobs_are_waited_even_if_latest_is_already_terminal(client, test_state, setup_agent):
    provider, args = setup_agent
    body, ids = queue_four(client, provider, args)
    client.post(f"/api/generations/{ids[-1]}/cancel")
    url = f"/api/assistant/runs/{body['request_id']}/continue"
    assert client.post(url).json()["status"] == "waiting"
    db = SqliteStore(test_state.config.app_data_dir)
    failed = db.claim_next_queued()
    db.mark_failed(failed.id, "EXECUTOR_FAILED", attempt_count=failed.attempt_count)
    finish_job(test_state, ids[-2])
    assert client.post(url).json()["status"] == "complete"


def test_wait_tracks_external_source_without_owning_or_cancelling_it(client, test_state, setup_agent):
    provider, args = setup_agent
    external = client.post("/api/generations/text-to-video", json={"params": args}).json()
    body = request_body("Wait for the existing video")
    provider.responses.append(call("wait_for_generation", {"generation_id": external["id"]}))
    response = client.post("/api/assistant/turn", json=body)
    assert response.json()["generation_ids"] == []
    url = f"/api/assistant/runs/{body['request_id']}"
    assert client.post(f"{url}/continue").json()["status"] == "waiting"
    assert client.post(f"{url}/stop").json()["status"] == "stopped"
    assert test_state.queued_generations.get_generation(external["id"]).status == "queued"


def test_retrying_a_running_request_reattaches_instead_of_conflicting(client, setup_agent):
    provider, _ = setup_agent
    entered, release = Event(), Event()
    body = request_body("Hello")

    def delayed_reply():
        entered.set()
        assert release.wait(10)
        return say("Done.")

    provider.responses.append(delayed_reply)
    with ThreadPoolExecutor() as pool:
        first = pool.submit(client.post, "/api/assistant/turn", json=body)
        assert entered.wait(10)
        # The client lost the first response and sends the same request again.
        retry = pool.submit(client.post, "/api/assistant/turn", json=body)
        assert client.post("/api/assistant/turn", json={**body, "text": "Different"}).status_code == 409
        release.set()
        assert first.result(timeout=10).json()["text"] == "Done."
        assert retry.result(timeout=10).json()["text"] == "Done."


def test_stop_during_model_call_blocks_later_tools_and_concurrent_resume(client, test_state, setup_agent):
    provider, args = setup_agent
    entered, release = Event(), Event()
    body = request_body("Generate two videos")

    def delayed_call():
        entered.set()
        assert release.wait(10)
        return call("generate_video", {"params": args, "job_number": 2})

    provider.responses.extend([call("generate_video", args), delayed_call])
    with ThreadPoolExecutor() as pool:
        future = pool.submit(client.post, "/api/assistant/turn", json=body)
        try:
            assert entered.wait(10)
            url = f"/api/assistant/runs/{body['request_id']}"
            assert client.post(f"{url}/continue").status_code == 409
            stopped = client.post(f"{url}/stop")
            assert stopped.status_code == 200
            assert stopped.json()["status"] == "stopped"
        finally:
            release.set()
        assert future.result(timeout=10).json()["status"] == "stopped"
    assert client.post(f"{url}/continue").json()["status"] == "stopped"
    jobs = test_state.queued_generations.list_generations()
    assert len(jobs) == 1 and jobs[0].status == "cancelled"
