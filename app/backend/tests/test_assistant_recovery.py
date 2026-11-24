import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from uuid import uuid4

import pytest

from services.assistant_run import ASSISTANT_PAYLOAD_VERSION
from services.sqlite_store import SqliteStore
from tests.test_assistant import call, finish_job, ingest, last_generation, request_body, run_record, saved_history, saved_run, say, setup_agent
from tests.test_http import _png, _wav


def test_explicit_local_selection_overrides_stale_gemini_settings(client, fake_services, test_state):
    test_state.state.app_settings.assistant_provider = "gemini"
    body = {**request_body("Hello"), "provider": "local", "model": "qwen3.5:9b"}
    fake_services.assistant_provider.responses.append(say("Local reply"))
    response = client.post("/api/assistant/turn", json=body)
    assert response.status_code == 200, response.text
    run = saved_run(test_state, body["request_id"])
    assert (run["provider"], run["model"]) == ("local", "qwen3.5:9b")


def test_followup_loads_previous_conversation_and_attached_ids(client, test_state, setup_agent, tmp_path):
    provider, _ = setup_agent
    asset = ingest(client, _png(tmp_path / "subject.png"))
    first = {**request_body("Remember this image"), "asset_ids": [asset["id"]]}
    provider.responses.append(say("Ready"))
    assert client.post("/api/assistant/turn", json=first).status_code == 200
    provider.responses.append(say("Same image"))
    second = {**request_body("Use that image"), "conversation_id": first["conversation_id"]}
    assert client.post("/api/assistant/turn", json=second).status_code == 200
    assert asset["id"] in saved_history(test_state, second["request_id"])


def test_missing_attachment_has_no_uncancellable_run(client, setup_agent):
    body = {**request_body(), "asset_ids": [str(uuid4())]}
    assert client.post("/api/assistant/turn", json=body).status_code == 404
    response = client.post(f"/api/assistant/runs/{body['request_id']}/stop")
    assert response.status_code == 404
    assert response.json()["code"] == "ASSISTANT_RUN_NOT_FOUND"


def test_resume_adds_updates_without_repeating_the_user_request(client, test_state, setup_agent):
    provider, args = setup_agent
    body = request_body("Make this exact unique request")
    provider.responses.extend([call("generate_video", args), say("Queued")])
    first = client.post("/api/assistant/turn", json=body).json()
    client.post(f"/api/generations/{first['generation_ids'][0]}/cancel")
    provider.responses.append(say("Cancelled"))
    assert client.post(f"/api/assistant/runs/{body['request_id']}/continue").status_code == 200
    assert saved_history(test_state, body["request_id"]).count(body["text"]) == 1


def test_conversation_endpoint_restores_visible_turns_and_original_retry_payload(client, setup_agent):
    provider, _ = setup_agent
    first = request_body("First message")
    provider.responses.append(say("First reply"))
    assert client.post("/api/assistant/turn", json=first).status_code == 200
    second = {**request_body("Follow-up"), "conversation_id": first["conversation_id"]}
    provider.responses.append(RuntimeError("temporary failure"))
    assert client.post("/api/assistant/turn", json=second).status_code == 502
    conversation = client.get(f"/api/assistant/conversations/{first['conversation_id']}").json()
    assert len(conversation["turns"]) == 2
    assert conversation["turns"][0]["response"]["text"] == "First reply"
    assert conversation["turns"][1]["response"] is None
    retry = conversation["turns"][1]["request"]
    provider.responses.append(say("Recovered"))
    assert client.post("/api/assistant/turn", json=retry).json()["text"] == "Recovered"
    assert client.get("/api/assistant/conversations").json()[0]["conversation_id"] == first["conversation_id"]


def test_attached_image_bytes_are_never_saved(client, test_state, setup_agent, tmp_path):
    provider, _ = setup_agent
    image = ingest(client, _png(tmp_path / "subject.png"))
    audio = ingest(client, _wav(tmp_path / "voice.wav"))
    provider.responses.extend([call("view_asset", {"asset_id": image["id"]}), say("A small test image")])
    request = {**request_body("Describe this image"), "asset_ids": [image["id"], audio["id"]]}
    result = client.post("/api/assistant/turn", json=request)
    assert result.status_code == 200, result.text
    assert '"kind":"binary"' not in saved_history(test_state, request["request_id"])


def wait_for_run(client, run_id, status):
    deadline = time.monotonic() + 5
    while (run := client.get(f"/api/assistant/runs/{run_id}").json())["status"] != status:
        assert time.monotonic() < deadline, run
        time.sleep(0.02)
    return run


def test_finished_job_continues_the_run_without_a_client(client, test_state, setup_agent):
    provider, args = setup_agent
    provider.responses.append(call("generate_video", args))
    body = request_body()
    generation_id = last_generation(client.post("/api/assistant/turn", json=body), client)["id"]
    provider.responses.append(say("Your preview is ready."))
    finish_job(test_state, generation_id)
    assert wait_for_run(client, body["request_id"], "complete")["text"] == "Your preview is ready."


def test_continue_during_background_step_waits_for_its_result(client, test_state, setup_agent):
    provider, args = setup_agent
    provider.responses.append(call("generate_video", args))
    body = request_body()
    generation_id = last_generation(client.post("/api/assistant/turn", json=body), client)["id"]
    entered, release = Event(), Event()

    def slow_reply():
        entered.set()
        assert release.wait(10)
        return say("Finished in the background.")

    provider.responses.append(slow_reply)
    finish_job(test_state, generation_id)
    assert entered.wait(5)
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(client.post, f"/api/assistant/runs/{body['request_id']}/continue")
        time.sleep(0.1)
        assert not pending.done()
        release.set()
        response = pending.result(10)
    assert response.status_code == 200, response.text
    assert response.json()["text"] == "Finished in the background."


def test_progress_stream_ends_once_the_run_is_complete(client, setup_agent):
    provider, _ = setup_agent
    provider.responses.append(say("Hi"))
    body = request_body("Hello")
    assert client.post("/api/assistant/turn", json=body).status_code == 200
    with client.stream("GET", f"/api/assistant/runs/{body['request_id']}/events") as response:
        events = [json.loads(line.removeprefix("data: ")) for line in response.iter_lines() if line.startswith("data: ")]
    assert events[-1]["done"]


def test_progress_stream_ends_after_a_failed_step_instead_of_heartbeating_forever(client, test_state, setup_agent):
    provider, _ = setup_agent
    provider.responses.append(RuntimeError("model crashed"))
    body = request_body("Hello")
    assert client.post("/api/assistant/turn", json=body).status_code >= 400

    # The events route streams this; a test client cannot abandon a stream that never ends.
    async def ends_with_done() -> bool:
        async for progress in test_state.assistant.watch_progress(body["request_id"], heartbeat=0.05):
            if progress is not None and progress.done:
                return True
        return False

    assert asyncio.run(asyncio.wait_for(ends_with_done(), 2))


def test_event_stream_ends_when_the_assistant_shuts_down(test_state):
    async def stream_ends_on_shutdown() -> None:
        # Neither the heartbeat nor the unknown-run grace can end this stream within the timeout.
        updates = test_state.assistant.watch_progress("2b1c0d4e-1f2a-4b3c-8d4e-5f6a7b8c9d0e", heartbeat=60, unknown_grace=60)

        async def drain() -> None:
            async for _ in updates:
                pass

        watching = asyncio.ensure_future(drain())
        await asyncio.sleep(0.05)
        await test_state.assistant.shutdown()
        await asyncio.wait_for(watching, 2)

    asyncio.run(stream_ends_on_shutdown())


def test_chat_title_stays_the_first_message_after_follow_ups(client, setup_agent):
    provider, _ = setup_agent
    first = request_body("Make a  rolling\nwave")
    provider.responses.extend([say("Done"), say("Done again")])
    assert client.post("/api/assistant/turn", json=first).status_code == 200
    follow_up = {**request_body("Now make it longer"), "conversation_id": first["conversation_id"]}
    assert client.post("/api/assistant/turn", json=follow_up).status_code == 200
    [summary] = client.get("/api/assistant/conversations").json()
    assert summary["title"] == "Make a rolling wave"


def test_deleting_a_chat_with_live_work_is_refused_until_stopped(client, test_state, setup_agent):
    provider, args = setup_agent
    provider.responses.append(call("generate_video", args))
    body = request_body()
    generation_id = last_generation(client.post("/api/assistant/turn", json=body), client)["id"]
    url = f"/api/assistant/conversations/{body['conversation_id']}"
    refused = client.delete(url)
    assert refused.status_code == 409
    assert refused.json()["code"] == "ASSISTANT_CONVERSATION_BUSY"
    client.post(f"/api/assistant/runs/{body['request_id']}/stop")
    wait_for_run(client, body["request_id"], "stopped")
    deadline = time.monotonic() + 5
    while (response := client.delete(url)).status_code == 409:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    assert response.status_code == 200
    assert client.get("/api/assistant/conversations").json() == []
    assert saved_run(test_state, body["request_id"]) is None
    assert client.get(f"/api/generations/{generation_id}").status_code == 200


def test_a_new_run_needs_the_conversation_head_it_was_built_from(test_state):
    from services.sqlite_store import SqliteStore
    store = SqliteStore(test_state.config.app_data_dir)
    assert store.insert_assistant_run(run_record("first"), title="", after=None)
    assert not store.insert_assistant_run(run_record("second"), title="", after=None)
    assert store.insert_assistant_run(run_record("second"), title="", after="first")


def test_a_queued_run_is_not_inserted_once_its_queue_row_is_gone(test_state):
    # A deleted chat or removed message must not come back as a run.
    from services.sqlite_store import SqliteStore
    store = SqliteStore(test_state.config.app_data_dir)
    store.enqueue_assistant_message("removed", "conversation", "{}", payload_version=ASSISTANT_PAYLOAD_VERSION, title="")
    store.remove_assistant_queue_item("conversation", "removed")
    assert not store.insert_assistant_run(run_record("removed"), title="", after=None, queued=True)
    assert store.get_assistant_run("removed") is None
    store.enqueue_assistant_message("kept", "conversation", "{}", payload_version=ASSISTANT_PAYLOAD_VERSION, title="")
    assert store.insert_assistant_run(run_record("kept"), title="", after=None, queued=True)
    assert store.list_assistant_queue("conversation") == []


def test_a_late_save_cannot_recreate_a_run_in_a_deleted_chat(test_state):
    from services.sqlite_store import SqliteStore
    store = SqliteStore(test_state.config.app_data_dir)
    assert store.insert_assistant_run(run_record("run"), title="", after=None)
    store.delete_assistant_conversation("conversation")
    store.save_assistant_run(run_record("run"))
    assert store.get_assistant_run("run") is None
    assert store.list_assistant_runs("conversation") == []


def corrupt(test_state, sql, params):
    conn = SqliteStore(test_state.config.app_data_dir)._connect()
    with conn:
        conn.execute(sql, params)
    conn.close()


def chat_with_two_replies(client, provider):
    first = request_body("Hello")
    second = {**request_body("Again"), "conversation_id": first["conversation_id"]}
    for body, reply in ((first, "Hi"), (second, "Hi again")):
        provider.responses.append(say(reply))
        assert client.post("/api/assistant/turn", json=body).status_code == 200
    return first, second


@pytest.mark.parametrize("change", ["payload = '{not json'", "payload_version = 999"])
def test_an_unreadable_saved_run_closes_that_message_and_the_rest_of_the_chat_still_loads(client, test_state, setup_agent, change):
    provider, _ = setup_agent
    first, _ = chat_with_two_replies(client, provider)
    corrupt(test_state, f"UPDATE assistant_runs SET {change} WHERE id = ?", (first["request_id"],))
    response = client.get(f"/api/assistant/conversations/{first['conversation_id']}")
    assert response.status_code == 200, response.text
    assert [turn["response"]["text"] for turn in response.json()["turns"]] == ["This message can't be loaded.", "Hi again"]


def test_a_new_message_can_follow_an_unreadable_run(client, test_state, setup_agent):
    provider, _ = setup_agent
    _, second = chat_with_two_replies(client, provider)
    corrupt(test_state, "UPDATE assistant_runs SET payload = '{not json' WHERE id = ?", (second["request_id"],))
    provider.responses.append(say("Still here"))
    third = {**request_body("Are you there?"), "conversation_id": second["conversation_id"]}
    response = client.post("/api/assistant/turn", json=third)
    assert response.status_code == 200, response.text
    assert response.json()["text"] == "Still here"


def test_an_unreadable_queued_message_is_skipped_and_the_chat_still_loads(client, test_state, setup_agent):
    provider, _ = setup_agent
    first, _ = chat_with_two_replies(client, provider)
    conversation_id = first["conversation_id"]
    store = SqliteStore(test_state.config.app_data_dir)
    # Held, so the readable message stays queued for the assertion.
    store.set_assistant_queue_hold(conversation_id, True)
    store.enqueue_assistant_message("unreadable", conversation_id, "{}", payload_version=ASSISTANT_PAYLOAD_VERSION, title="")
    readable = {**request_body("Next"), "conversation_id": conversation_id}
    store.enqueue_assistant_message(readable["request_id"], conversation_id, json.dumps(readable), payload_version=ASSISTANT_PAYLOAD_VERSION, title="")
    response = client.get(f"/api/assistant/conversations/{conversation_id}")
    assert response.status_code == 200, response.text
    assert [item["text"] for item in response.json()["queue"]] == ["Next"]


def test_a_save_from_an_outdated_run_keeps_job_rows_it_did_not_release(test_state):
    store = SqliteStore(test_state.config.app_data_dir)
    assert store.insert_assistant_run(run_record("run"), title="", after=None)
    store.save_assistant_run(run_record("run").model_copy(update={"jobs": {"1": "job-1"}, "shown": ("job-1",)}))
    store.save_assistant_run(run_record("run"))
    stored = store.get_assistant_run("run")
    assert stored is not None and stored.jobs == {"1": "job-1"}


def test_stop_on_an_unreadable_run_does_not_error(client, test_state, setup_agent):
    provider, args = setup_agent
    provider.responses.extend([call("generate_video", args), say("Queued.")])
    body = request_body()
    assert client.post("/api/assistant/turn", json=body).json()["status"] == "waiting"
    corrupt(test_state, "UPDATE assistant_runs SET payload = '{not json' WHERE id = ?", (body["request_id"],))
    response = client.post(f"/api/assistant/runs/{body['request_id']}/stop")
    assert response.status_code == 200, response.text
