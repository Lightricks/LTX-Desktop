from pydantic_ai.messages import BinaryContent, ModelMessagesTypeAdapter, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart

from handlers.assistant_history import trim_conversation_history
from handlers.assistant_tools import WAIT_REMINDER
from services.assistant_media import ATTACHMENTS_HEADER, with_previews

CURRENT_PROMPT = "Make a puffin clip, then extend it"


def _queued(call_id):
    return ToolReturnPart("generate_video", {"id": call_id, "feature": "text-to-video", "status": "queued", "attempt_count": 0,
                                             "outputs": [], "next": WAIT_REMINDER}, tool_call_id=call_id)


def _compacted_queued_results():
    """An earlier and the current request each queued a job; two large later results force full compaction."""
    history = [
        ModelRequest(parts=[UserPromptPart("Make a fox clip")]),
        ModelResponse(parts=[ToolCallPart("generate_video", {}, tool_call_id="earlier")]),
        ModelRequest(parts=[_queued("earlier")]),
        ModelResponse(parts=[TextPart("Queued a fox clip.")]),
        ModelRequest(parts=[UserPromptPart(CURRENT_PROMPT)]),
        ModelResponse(parts=[ToolCallPart("generate_video", {}, tool_call_id="current")]),
        ModelRequest(parts=[_queued("current")]),
        ModelResponse(parts=[ToolCallPart("list_assets", {}, tool_call_id="first")]),
        ModelRequest(parts=[ToolReturnPart("list_assets", {"id": "clip", "metadata": "x" * 5000}, tool_call_id="first")]),
        ModelResponse(parts=[ToolCallPart("list_assets", {}, tool_call_id="second")]),
        ModelRequest(parts=[ToolReturnPart("list_assets", {"id": "clip", "metadata": "y" * 5000}, tool_call_id="second")]),
    ]
    result = trim_conversation_history(history, max_bytes=8000, keep_prompt=CURRENT_PROMPT)
    payload = ModelMessagesTypeAdapter.dump_json(result).decode()
    # Compacted, but no exchange dropped.
    assert "x" * 5000 not in payload and "Make a fox clip" in payload
    return {part.tool_call_id: part.content for message in result if isinstance(message, ModelRequest)
            for part in message.parts if isinstance(part, ToolReturnPart)}


def test_previews_are_added_to_a_copy_for_the_latest_prompt_and_viewed_results():
    prompt = "Animate this" + ATTACHMENTS_HEADER + '[{"id":"cat"}]'
    history = [
        ModelRequest(parts=[UserPromptPart(prompt)]),
        ModelResponse(parts=[TextPart("Done")]),
        ModelRequest(parts=[UserPromptPart(prompt)]),
        ModelResponse(parts=[ToolCallPart("view_asset", {"asset_id": "dog"}, tool_call_id="view")]),
        ModelRequest(parts=[ToolReturnPart("view_asset", {"asset_id": "dog"}, tool_call_id="view")]),
    ]
    saved = ModelMessagesTypeAdapter.dump_json(history)
    result = with_previews(history, prompt_prefix="Animate this" + ATTACHMENTS_HEADER,
                           attachments={"cat": b"cat"}, viewed={"view": ("dog", b"dog")})
    images = [[item.data for part in message.parts if isinstance(part, UserPromptPart) and not isinstance(part.content, str)
               for item in part.content if isinstance(item, BinaryContent)] for message in result]
    assert images == [[], [], [b"cat"], [], [b"dog"]]
    assert ModelMessagesTypeAdapter.dump_json(history) == saved


def test_compaction_keeps_final_request_and_call_pairs():
    history = [
        ModelRequest(parts=[UserPromptPart("Find my clip")]),
        ModelResponse(parts=[TextPart("Checking"), ToolCallPart("list_assets", {}, tool_call_id="call")]),
        ModelRequest(parts=[ToolReturnPart("list_assets", {"id": "clip", "metadata": "x" * 8000}, tool_call_id="call")]),
    ]
    result = trim_conversation_history(history, max_bytes=2000)
    assert isinstance(result[-1], ModelRequest)
    assert result[-1].parts[0].tool_call_id == result[-2].parts[-1].tool_call_id
    assert "clip" in ModelMessagesTypeAdapter.dump_json(result).decode()
    assert "x" * 8000 in ModelMessagesTypeAdapter.dump_json(history).decode()


def test_dropping_an_exchange_keeps_followup_subject():
    history = [
        ModelRequest(parts=[UserPromptPart("Generate a beach clip " + "x" * 5000)]),
        ModelResponse(parts=[ToolCallPart("generate_video", {"prompt": "x" * 5000}, tool_call_id="call")]),
        ModelRequest(parts=[ToolReturnPart("generate_video", {"id": "beach-clip", "status": "succeeded"}, tool_call_id="call")]),
        ModelResponse(parts=[TextPart("Done")]),
        ModelRequest(parts=[UserPromptPart("Extend that clip")]),
    ]
    result = trim_conversation_history(history, max_bytes=2000)
    payload = ModelMessagesTypeAdapter.dump_json(result).decode()
    assert "beach-clip" in payload
    assert "Extend that clip" in payload
    assert len(payload.encode()) <= 2000


def test_repeated_compaction_does_not_grow_result_wrappers():
    history = [
        ModelRequest(parts=[UserPromptPart("Find clips")]),
        ModelResponse(parts=[ToolCallPart("list_assets", {}, tool_call_id="call")]),
        ModelRequest(parts=[ToolReturnPart("list_assets", {"id": "clip", "metadata": "x" * 8000}, tool_call_id="call")]),
    ]
    first = trim_conversation_history(history, max_bytes=2000)
    second = trim_conversation_history(first, max_bytes=2000)
    assert ModelMessagesTypeAdapter.dump_json(first) == ModelMessagesTypeAdapter.dump_json(second)


def test_compaction_does_not_strip_the_wait_reminder_from_the_current_requests_queued_job():
    assert _compacted_queued_results()["current"]["next"] == WAIT_REMINDER


def test_compaction_strips_stale_wait_reminders_from_earlier_requests():
    assert "next" not in _compacted_queued_results()["earlier"]
