import json

import pytest

from services.assistant_run import ASSISTANT_PAYLOAD_VERSION
from services.sqlite_store import SqliteStore
from tests.fakes.assistant import InterruptedTextStream
from tests.test_assistant import (
    assistant_job_id, call, failed_job_in_chat, rejected_clip, request_body,
    retry_in, saved_history, say, setup_agent,
)
from tests.test_assistant_queue import chat_url, settled, wait_for
from tests.test_assistant_recovery import corrupt


def retry_failed_job(client, provider, conversation_id, generation_id):
    provider.responses.append(call("retry_failed_job", {"generation_id": generation_id}))
    body, response = retry_in(client, conversation_id)
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "waiting"
    return body


def test_a_stale_retry_save_cannot_move_a_job_card_back_from_a_later_retry(client, test_state, setup_agent):
    provider, args = setup_agent
    chat, job_id, store = failed_job_in_chat(client, test_state, provider, args)
    first_retry = retry_failed_job(client, provider, chat, job_id)
    job = store.claim_next_queued()
    store.mark_failed(job.id, "EXECUTOR_FAILED", attempt_count=job.attempt_count)
    provider.responses.append(say("The retry failed."))
    assert client.post(f"/api/assistant/runs/{first_retry['request_id']}/continue").json()["status"] == "complete"
    stale = store.get_assistant_run(first_retry["request_id"])
    assert stale is not None
    second_retry = retry_failed_job(client, provider, chat, job_id)

    store.save_assistant_run(stale)

    turns = client.get(chat_url(chat)).json()["turns"]
    assert [turn["response"]["generation_ids"] for turn in turns] == [[], [], [job_id]]
    assert store.get_assistant_run(second_retry["request_id"]).shown == (job_id,)


def test_stopping_an_older_message_does_not_cancel_a_job_moved_to_a_retry(client, test_state, setup_agent):
    provider, args = setup_agent
    chat, job_id, store = failed_job_in_chat(client, test_state, provider, args)
    original = store.get_latest_assistant_run(chat)
    retry = retry_failed_job(client, provider, chat, job_id)

    stopped = client.post(f"/api/assistant/runs/{original.id}/stop")

    assert stopped.status_code == 200, stopped.text
    assert stopped.json()["status"] == "complete"
    assert store.get_generation(job_id).status == "queued"
    assert client.get(chat_url(chat)).json()["queue_on_hold"] is False
    assert client.post(f"/api/assistant/runs/{retry['request_id']}/stop").json()["status"] == "stopped"
    assert store.get_generation(job_id).status == "cancelled"


@pytest.mark.parametrize("change", ["payload = '{not json'", "payload_version = 999"])
def test_an_unreadable_message_keeps_its_job_available_to_a_later_retry(client, test_state, setup_agent, change):
    provider, args = setup_agent
    chat, job_id, store = failed_job_in_chat(client, test_state, provider, args)
    original = store.get_latest_assistant_run(chat)
    corrupt(test_state, f"UPDATE assistant_runs SET {change} WHERE id = ?", (original.id,))

    response = client.get(chat_url(chat))
    assert response.status_code == 200, response.text
    [turn] = response.json()["turns"]
    assert turn["response"]["status"] == "complete"
    assert turn["response"]["generation_ids"] == [job_id]
    retry_failed_job(client, provider, chat, job_id)
    assert store.get_generation(job_id).status == "queued"
    assert [turn["response"]["generation_ids"] for turn in client.get(chat_url(chat)).json()["turns"]] == [[], [job_id]]


def test_an_unknown_queue_version_does_not_block_the_valid_message_behind_it(client, test_state, setup_agent):
    provider, _ = setup_agent
    first = request_body("First")
    provider.responses.append(say("Ready"))
    assert client.post("/api/assistant/turn", json=first).status_code == 200
    chat = first["conversation_id"]
    store = SqliteStore(test_state.config.app_data_dir)
    store.set_assistant_queue_hold(chat, True)
    unknown = {**request_body("Unknown shape"), "conversation_id": chat}
    valid = {**request_body("Next"), "conversation_id": chat}
    for body, version in ((unknown, 999), (valid, ASSISTANT_PAYLOAD_VERSION)):
        assert store.enqueue_assistant_message(body["request_id"], chat, json.dumps(body), payload_version=version, title=body["text"])
    provider.responses.append(say("Next finished"))

    released = client.post(f"{chat_url(chat)}/queue/release")

    assert released.status_code == 200, released.text
    conversation = wait_for(client, chat, lambda value: len(value["turns"]) == 2 and settled(value))
    assert [turn["request"]["request_id"] for turn in conversation["turns"]] == [first["request_id"], valid["request_id"]]
    assert conversation["turns"][-1]["response"]["text"] == "Next finished"
    assert conversation["queue"] == []
    assert store.get_assistant_run(unknown["request_id"]) is None
    assert store.list_assistant_queue(chat) == []


def test_a_rejected_submission_releases_ownership_and_its_number_can_be_reused_on_retry(client, test_state, setup_agent):
    provider, args = setup_agent
    body = request_body()
    provider.responses.extend([rejected_clip(args, 1), RuntimeError("Connection lost after validation feedback")])
    assert client.post("/api/assistant/turn", json=body).status_code == 502
    store = SqliteStore(test_state.config.app_data_dir)
    rejected = store.get_assistant_run(body["request_id"])
    assert rejected.jobs == {}
    assert rejected.shown == ()
    assert store.list_assistant_job_ids(body["conversation_id"]) == []
    assert store.get_generation(assistant_job_id(body, 1)) is None

    provider.responses.extend([call("generate_video", args), say("Queued.")])
    response = client.post("/api/assistant/turn", json=body)

    assert response.status_code == 200, response.text
    job_id = assistant_job_id(body, 1)
    assert response.json()["generation_ids"] == [job_id]
    saved = store.get_assistant_run(body["request_id"])
    assert saved.jobs == {"1": job_id}
    assert saved.shown == (job_id,)
    assert store.get_generation(job_id).status == "queued"
    assert len(store.list_generations()) == 1


@pytest.mark.parametrize("partial_text", ["", "Beginning an answer"])
def test_an_interrupted_text_reply_does_not_replace_the_users_request_with_job_instructions(client, test_state, setup_agent, partial_text):
    provider, _ = setup_agent
    body = request_body("Reply exactly ORIGINAL-REQUEST. Text only; do not use tools.")
    provider.responses.append(InterruptedTextStream(partial_text))
    failed = client.post("/api/assistant/turn", json=body)
    assert failed.status_code == 502, failed.text
    conversation = client.get(chat_url(body["conversation_id"])).json()
    [turn] = conversation["turns"]
    assert turn["running"] is False
    assert turn["response"] is None
    history = json.loads(saved_history(test_state, body["request_id"]))
    prompts = [part["content"] for message in history for part in message["parts"] if part["part_kind"] == "user-prompt"]
    assert prompts[-1].startswith(body["text"]), prompts[-1]

    provider.responses.append(say("ORIGINAL-REQUEST"))
    retried = client.post("/api/assistant/turn", json=turn["request"])
    assert retried.status_code == 200, retried.text
    assert retried.json()["text"] == "ORIGINAL-REQUEST"
    assert len(client.get(chat_url(body["conversation_id"])).json()["turns"]) == 1
