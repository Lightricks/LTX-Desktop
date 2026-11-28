from __future__ import annotations

import time
from threading import Event
from uuid import uuid4

import pytest

from tests.test_assistant import assistant_job_id, call, request_body, say, setup_agent  # noqa: F401  # pyright: ignore[reportUnusedImport]

# The queue does not depend on the provider.
one_provider = pytest.mark.parametrize("setup_agent", ["gemini"], indirect=True)


def chat_url(conversation_id: str) -> str:
    return f"/api/assistant/conversations/{conversation_id}"


def enqueue(client, conversation_id: str, text: str):
    return client.post(f"{chat_url(conversation_id)}/queue", json={"id": str(uuid4()), "text": text})


def wait_for(client, conversation_id: str, predicate):
    deadline = time.monotonic() + 10
    while True:
        chat = client.get(chat_url(conversation_id)).json()
        if predicate(chat):
            return chat
        assert time.monotonic() < deadline, chat
        time.sleep(0.02)


def settled(chat) -> bool:
    return all(not turn["running"] and turn["response"] for turn in chat["turns"])


def held_reply(text: str):
    entered, release = Event(), Event()

    def reply():
        entered.set()
        assert release.wait(10)
        return say(text)

    return reply, entered, release


@one_provider
def test_queued_messages_run_one_at_a_time_in_order(client, setup_agent):
    provider, _ = setup_agent
    reply, entered, release = held_reply("First done.")
    provider.responses.extend([reply, say("Second done.")])
    conversation_id = str(uuid4())
    try:
        assert enqueue(client, conversation_id, "First").status_code == 200
        assert entered.wait(10)
        queued = enqueue(client, conversation_id, "Second").json()
        assert [turn["request"]["text"] for turn in queued["turns"]] == ["First"]
        assert [item["text"] for item in queued["queue"]] == ["Second"]
    finally:
        release.set()
    chat = wait_for(client, conversation_id, lambda chat: len(chat["turns"]) == 2 and settled(chat))
    assert [turn["response"]["text"] for turn in chat["turns"]] == ["First done.", "Second done."]
    assert chat["queue"] == []


@one_provider
def test_stop_after_the_reply_completed_keeps_it_and_does_not_hold_the_queue(client, setup_agent):
    provider, _ = setup_agent
    provider.responses.append(say("Done."))
    conversation_id = str(uuid4())
    enqueue(client, conversation_id, "First")
    [turn] = wait_for(client, conversation_id, lambda chat: chat["turns"] and settled(chat))["turns"]
    stopped = client.post(f"/api/assistant/runs/{turn['request']['request_id']}/stop").json()
    assert (stopped["status"], stopped["text"]) == ("complete", "Done.")
    assert client.get(chat_url(conversation_id)).json()["queue_on_hold"] is False


def stop_with_one_queued(client, provider) -> str:
    reply, entered, release = held_reply("Never shown.")
    provider.responses.append(reply)
    conversation_id = str(uuid4())
    try:
        enqueue(client, conversation_id, "First")
        assert entered.wait(10)
        enqueue(client, conversation_id, "Second")
        first_id = client.get(chat_url(conversation_id)).json()["turns"][0]["request"]["request_id"]
        assert client.post(f"/api/assistant/runs/{first_id}/stop").json()["status"] == "stopped"
    finally:
        release.set()
    wait_for(client, conversation_id, settled)
    return conversation_id


@one_provider
def test_stop_holds_the_queue_until_released(client, setup_agent):
    provider, _ = setup_agent
    conversation_id = stop_with_one_queued(client, provider)
    provider.responses.append(say("Second done."))
    time.sleep(0.2)
    held = client.get(chat_url(conversation_id)).json()
    assert len(held["turns"]) == 1
    assert held["queue_on_hold"] is True
    assert [item["text"] for item in held["queue"]] == ["Second"]

    assert client.post(f"{chat_url(conversation_id)}/queue/release").status_code == 200
    chat = wait_for(client, conversation_id, lambda chat: len(chat["turns"]) == 2 and settled(chat))
    assert chat["turns"][1]["response"]["text"] == "Second done."
    assert chat["queue_on_hold"] is False


def test_queued_message_without_a_gemini_key_is_rejected_instead_of_blocking_the_chat(client, test_state):
    test_state.state.app_settings.assistant_provider = "gemini"
    test_state.state.app_settings.gemini_api_key = ""
    conversation_id = str(uuid4())
    response = enqueue(client, conversation_id, "Hello")
    assert response.status_code == 400
    assert response.json()["code"] == "GEMINI_API_KEY_MISSING"
    assert client.get(chat_url(conversation_id)).json()["queue"] == []


@one_provider
def test_deleting_a_chat_drops_its_queued_messages_and_hold(client, setup_agent):
    provider, _ = setup_agent
    conversation_id = stop_with_one_queued(client, provider)
    assert client.delete(chat_url(conversation_id)).status_code == 200
    chat = client.get(chat_url(conversation_id)).json()
    assert chat["queue"] == []
    assert chat["queue_on_hold"] is False


@one_provider
def test_chat_holding_only_a_never_started_message_can_be_deleted(client, setup_agent):
    conversation_id = str(uuid4())
    # Passes enqueue's model check, then cannot start: the attachment does not exist.
    item = {"id": str(uuid4()), "text": "Animate this", "asset_ids": [str(uuid4())]}
    assert client.post(f"{chat_url(conversation_id)}/queue", json=item).status_code == 200
    wait_for(client, conversation_id, lambda chat: chat["queue_on_hold"])
    assert client.delete(chat_url(conversation_id)).status_code == 200
    assert client.get(chat_url(conversation_id)).json()["queue"] == []


@one_provider
def test_chat_whose_last_turn_failed_can_be_deleted_without_stop(client, setup_agent):
    provider, _ = setup_agent
    provider.responses.append(RuntimeError("model crashed"))
    body = request_body("Hello")
    assert client.post("/api/assistant/turn", json=body).status_code >= 400
    assert client.delete(chat_url(body["conversation_id"])).status_code == 200
    assert client.get(chat_url(body["conversation_id"])).json()["turns"] == []


@one_provider
def test_running_clears_when_a_turn_fails(client, setup_agent):
    provider, _ = setup_agent
    provider.responses.append(RuntimeError("model crashed"))
    body = request_body("Hello")
    assert client.post("/api/assistant/turn", json=body).status_code >= 400
    [turn] = client.get(chat_url(body["conversation_id"])).json()["turns"]
    assert turn["response"] is None
    assert turn["running"] is False


def first_job_then_second_message(client, test_state, provider, args, *, wait: bool, parallel: bool = True):
    """Queues a job from a first message (its job stays queued), then queues a second message."""
    test_state.state.app_settings.assistant_parallel_gemini = parallel
    conversation_id, first_id = str(uuid4()), str(uuid4())
    provider.responses.append(call("generate_video", {"params": args, "job_number": 1}))
    if wait:
        provider.responses.append(call("wait_for_generation", {"generation_id": assistant_job_id({"request_id": first_id}, 1)}))
    provider.responses.extend([say("Queued."), say("Second done.")])
    assert client.post(f"{chat_url(conversation_id)}/queue", json={"id": first_id, "text": "A wave"}).status_code == 200
    wait_for(client, conversation_id, lambda chat: chat["turns"] and chat["turns"][0]["response"])
    enqueue(client, conversation_id, "How is it going?")
    return conversation_id


@pytest.mark.parametrize("setup_agent", ["gemini"], indirect=True)
def test_parallel_gemini_flag_lets_a_queued_message_start_while_a_job_runs(client, test_state, setup_agent):
    provider, args = setup_agent
    conversation_id = first_job_then_second_message(client, test_state, provider, args, wait=False)
    chat = wait_for(client, conversation_id, lambda chat: len(chat["turns"]) == 2 and settled(chat))
    assert [turn["response"]["text"] for turn in chat["turns"]] == ["Queued.", "Second done."]


def assert_second_message_held(client, conversation_id):
    time.sleep(0.3)
    chat = client.get(chat_url(conversation_id)).json()
    assert len(chat["turns"]) == 1
    assert chat["turns"][0]["response"]["status"] == "waiting"
    assert [item["text"] for item in chat["queue"]] == ["How is it going?"]


@pytest.mark.parametrize("setup_agent", ["local"], indirect=True)
def test_parallel_flag_does_not_release_a_local_run_while_its_job_runs(client, test_state, setup_agent):
    provider, args = setup_agent
    assert_second_message_held(client, first_job_then_second_message(client, test_state, provider, args, wait=False))


@pytest.mark.parametrize("setup_agent", ["gemini"], indirect=True)
def test_parallel_flag_does_not_release_a_run_waiting_on_wait_for_generation(client, test_state, setup_agent):
    provider, args = setup_agent
    assert_second_message_held(client, first_job_then_second_message(client, test_state, provider, args, wait=True))


@pytest.mark.parametrize("setup_agent", ["gemini"], indirect=True)
def test_without_the_flag_a_gemini_run_still_waits_for_its_jobs(client, test_state, setup_agent):
    provider, args = setup_agent
    assert_second_message_held(client, first_job_then_second_message(client, test_state, provider, args, wait=False, parallel=False))
