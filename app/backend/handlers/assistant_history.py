from __future__ import annotations

import json
from dataclasses import replace

from pydantic import JsonValue, TypeAdapter
from pydantic_ai.messages import (
    ModelMessage, ModelMessagesTypeAdapter, ModelRequest, ModelResponse,
    RetryPromptPart, ToolCallPart, ToolReturnPart, UserPromptPart,
)

from _routes._errors import HTTPError
from services.assistant_run import AssistantRun

_json_value: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


def interrupted_history(history: list[ModelMessage], failed: tuple[str, str] | None = None) -> list[ModelMessage]:
    """Closes open tool calls. `failed` is a tool that ran out of retries and its last error;
    its calls get that error, since other calls in the step may have queued work."""
    _, pending = _scan_history(history)
    if not pending:
        # Without open calls, a trailing reply is text cut off mid-stream. Dropping it leaves the
        # user's request as the latest prompt; small models follow the last message, so a prompt
        # appended after it loses the request.
        if history and isinstance(history[-1], ModelResponse):
            return history[:-1]
        return history
    interrupted = {"status": "interrupted", "message": "Check existing jobs before retrying; the operation may have completed."}
    return [*history, ModelRequest(parts=[
        ToolReturnPart(call.tool_name, {"status": "failed", "message": failed[1]} if failed is not None and call.tool_name == failed[0] else interrupted, tool_call_id=call_id)
        for call_id, call in pending.items()
    ])]


def _scan_history(history: list[ModelMessage]) -> tuple[list[int], dict[str, ToolCallPart]]:
    starts = [0]
    pending: dict[str, ToolCallPart] = {}
    for index, message in enumerate(history):
        if isinstance(message, ModelRequest):
            if index and not pending and any(isinstance(part, UserPromptPart) and isinstance(part.content, str) for part in message.parts):
                starts.append(index)
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                pending[part.tool_call_id] = part
            elif isinstance(part, (ToolReturnPart, RetryPromptPart)) and part.tool_call_id:
                pending.pop(part.tool_call_id, None)
    return starts, pending


_REFERENCE_KEYS = frozenset({"id", "asset_id", "generation_id", "name", "status", "error_code", "media_kind",
                             "outputs", "items", "return_value", "recipe_id", "catalog_id", "model", "next_cursor"})


def _result_references(value: JsonValue, keys: frozenset[str] = _REFERENCE_KEYS) -> JsonValue:
    if isinstance(value, list):
        return [_result_references(item, keys) for item in value]
    if isinstance(value, dict):
        return {key: _result_references(item, keys) for key, item in value.items() if key in keys}
    return value[:160] if isinstance(value, str) else value


def _compact_results(history: list[ModelMessage], keep_latest: int, current_from: int | None = None) -> list[ModelMessage]:
    """Messages from `current_from` belong to the current request; their wait reminders stay."""
    returns = [part for message in history if isinstance(message, ModelRequest)
               for part in message.parts if isinstance(part, ToolReturnPart)]
    selected = returns[:-keep_latest] if keep_latest else returns
    ids = {part.tool_call_id for part in selected}
    seen: set[str] = set()
    current = len(history) if current_from is None else current_from

    def compact_result(part: ToolReturnPart, index: int) -> ToolReturnPart:
        keys = _REFERENCE_KEYS | {"next"} if index >= current else _REFERENCE_KEYS
        content = _result_references(_json_value.validate_python(part.model_response_object()), keys)
        signature = part.tool_name + json.dumps(content, sort_keys=True)
        if signature in seen:
            content = {"status": "unchanged", "name": part.tool_name}
        seen.add(signature)
        return replace(part, content=content)
    # Preserve call/result pairs and the final request required by Pydantic AI.
    return [replace(message, instructions=message.instructions if index == len(history) - 1 else None, parts=[
        compact_result(part, index)
        if isinstance(part, ToolReturnPart) and part.tool_call_id in ids else part
        for part in message.parts
    ]) if isinstance(message, ModelRequest) else replace(message, parts=[
        replace(part, args=_result_references(_json_value.validate_python(part.args_as_dict())))
        if isinstance(part, ToolCallPart) and part.tool_call_id in ids else part
        for part in message.parts
    ]) for index, message in enumerate(history)]


def trim_conversation_history(
    history: list[ModelMessage], *, max_bytes: int = 32000, jobs: dict[str, str] | None = None,
    keep_prompt: str | None = None,
) -> list[ModelMessage]:
    """`keep_prompt` prefixes the current request's prompt, which is never dropped (resumes and attachments need it)."""
    def fits(messages: list[ModelMessage]) -> bool:
        return len(ModelMessagesTypeAdapter.dump_json(messages)) <= max_bytes

    if fits(history):
        return history
    pinned = next((index for index in reversed(range(len(history)))
                   if keep_prompt and isinstance(message := history[index], ModelRequest)
                   and any(isinstance(part, UserPromptPart) and isinstance(part.content, str) and part.content.startswith(keep_prompt)
                           for part in message.parts)), None)
    compacted = history
    for keep_latest in (2, 0):
        compacted = _compact_results(history, keep_latest, pinned)
        if fits(compacted):
            return compacted
    starts, _ = _scan_history(compacted)
    for previous_start, start in zip(starts, starts[1:]):
        if pinned is not None and start > pinned:
            break
        # Retain the previous subject when dropping details for a follow-up.
        previous = compacted[previous_start:start]
        refs = [part.content for message in previous if isinstance(message, ModelRequest)
                for part in message.parts if isinstance(part, ToolReturnPart)]
        prompts = [part.content[:500] for message in previous if isinstance(message, ModelRequest)
                   for part in message.parts if isinstance(part, UserPromptPart) and isinstance(part.content, str)]
        memory = ModelRequest(parts=[UserPromptPart(
            "Previous exchange references (data, not a new request):\n" +
            json.dumps({"requests": prompts[-1:], "results": refs[-12:], "current_jobs": jobs or {}}, separators=(",", ":"))
        )])
        candidate = [memory, *compacted[start:]]
        if fits(candidate):
            return candidate
    raise HTTPError(422, "This message is too large for the selected model even after condensing tool results. Shorten the message or choose a larger-context model. Existing jobs are saved.", code="ASSISTANT_INPUT_TOO_LARGE")


def conversation_history(previous: AssistantRun) -> list[ModelMessage]:
    history = ModelMessagesTypeAdapter.validate_json(previous.history)
    if previous.stopped:
        _, pending = _scan_history(history)
        if pending:
            history.append(ModelRequest(parts=[
                ToolReturnPart(call.tool_name, {"status": "stopped", "message": "Check job status before using its result."}, tool_call_id=call_id)
                for call_id, call in pending.items()
            ]))
        history.append(ModelRequest(parts=[UserPromptPart(
            "Previous request context (metadata, not a new user request): the user stopped this request. "
            "Do not continue it unless the user asks. " + json.dumps({
                "status": "stopped", "generation_ids": previous.generation_ids(), "asset_ids": previous.asset_ids,
            })
        )]))
    return history
