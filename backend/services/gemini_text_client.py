"""Shared helper for calling Gemini's generateContent REST API for text (not embedding) results."""

from __future__ import annotations

from typing import cast

from _routes._errors import HTTPError
from pydantic import BaseModel, Field, ValidationError
from services.interfaces import HTTPClient, HttpTransportError, JSONValue

GEMINI_GENERATE_CONTENT_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
)


class _GeminiPart(BaseModel):
    text: str


class _GeminiContent(BaseModel):
    parts: list[_GeminiPart] = Field(min_length=1)


class _GeminiCandidate(BaseModel):
    content: _GeminiContent


class _GeminiResponsePayload(BaseModel):
    candidates: list[_GeminiCandidate] = Field(min_length=1)


# Gemini returns these as HTTP 200 with no usable `content` — either the prompt was rejected
# before any candidate was generated (`promptFeedback.blockReason`, empty `candidates`) or a
# candidate was generated then withheld (`candidates[0].finishReason`, no `content` key at all).
# Both shapes fail the strict schema above with an opaque ValidationError unless caught first.
_BLOCKED_FINISH_REASONS = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}


def _raise_if_blocked(payload: object) -> None:
    if not isinstance(payload, dict):
        return
    payload = cast("dict[str, object]", payload)
    prompt_feedback = payload.get("promptFeedback")
    if isinstance(prompt_feedback, dict):
        prompt_feedback = cast("dict[str, object]", prompt_feedback)
        block_reason = prompt_feedback.get("blockReason")
        if block_reason:
            raise HTTPError(
                422,
                f"Prompt rejected by Gemini safety filters ({block_reason})",
                code="GEMINI_CONTENT_BLOCKED",
            )
    candidates = payload.get("candidates")
    if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
        first_candidate = cast("dict[str, object]", candidates[0])
        finish_reason = first_candidate.get("finishReason")
        if finish_reason in _BLOCKED_FINISH_REASONS:
            raise HTTPError(
                422,
                f"Response withheld by Gemini safety filters ({finish_reason})",
                code="GEMINI_CONTENT_BLOCKED",
            )


def extract_gemini_text(payload: object) -> str:
    _raise_if_blocked(payload)
    try:
        parsed = _GeminiResponsePayload.model_validate(payload)
    except ValidationError:
        raise HTTPError(500, "GEMINI_PARSE_ERROR")
    return parsed.candidates[0].content.parts[0].text


def call_gemini_generate_content(
    http: HTTPClient,
    *,
    api_key: str,
    contents: list[JSONValue],
    system_instruction: str | None = None,
    generation_config: dict[str, JSONValue] | None = None,
    timeout: int = 30,
) -> str:
    """POST to Gemini's generateContent and return the first candidate's text, stripped."""
    payload: dict[str, JSONValue] = {"contents": contents}
    if system_instruction is not None:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    if generation_config is not None:
        payload["generationConfig"] = generation_config

    try:
        response = http.post(
            GEMINI_GENERATE_CONTENT_URL,
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            json_payload=payload,
            timeout=timeout,
        )
    except HttpTransportError as exc:
        raise HTTPError(504, "Gemini API request timed out") from exc

    if response.status_code != 200:
        raise HTTPError(response.status_code, f"Gemini API error: {response.text}")

    text = extract_gemini_text(response.json()).strip()
    if not text:
        # Whitespace-only/empty text is a valid, non-blocked candidate as far as the schema is
        # concerned — but handing it back as `enhancedPrompt` would silently wipe the user's
        # prompt on what looks like a "success" response.
        raise HTTPError(500, "Gemini returned an empty response", code="GEMINI_EMPTY_RESPONSE")
    return text
