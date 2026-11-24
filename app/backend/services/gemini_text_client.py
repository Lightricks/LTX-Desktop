"""Shared helper for calling Gemini's generateContent REST API for text (not embedding) results."""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from typing import cast
from urllib.parse import urlencode

from _routes._errors import HTTPError
from api_types import GeminiModelOptionPayload
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from services.interfaces import HTTPClient, HttpResponseLike, HttpTransportError, JSONValue

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_MODELS_PREFIX = "models/"
_GEMINI_LIST_PAGE_SIZE = 1000
_GEMINI_LIST_MAX_PAGES = 5
_GEMINI_MODELS_CACHE_TTL_S = 3600.0


@dataclass(frozen=True, slots=True)
class _GeminiModelsListCache:
    api_key: str
    models: tuple[GeminiModelOptionPayload, ...]
    expires_at: float


_gemini_models_list_cache: _GeminiModelsListCache | None = None
_gemini_models_list_cache_lock = Lock()

# models.list returns every generateContent endpoint, including branded aliases
# (nano-banana-pro, lyria-*, deep-research-*) whose descriptions are often empty.
# Enhance / gap suggestions are Gemini-API *chat* models (text-out, optional image input).
# Hosted Gemma shares generateContent but returns empty text on this system prompt;
# local Enhance already runs Gemma on-device.
_TEXT_MODEL_ID_PREFIXES = ("gemini-",)
_NON_TEXT_OUTPUT_ID_MARKERS = (
    "embedding",
    "imagen",
    "veo",
    "tts",
    "image-generation",
    "-image",
    "native-audio",
    "-audio-",
    "live",
    "omni",
    "robotics",
    "computer-use",
)
_NON_TEXT_OUTPUT_LABEL_MARKERS = (
    "image generation",
    "generate images",
    "generates images",
    "image editing",
    "nano banana",
    "text-to-speech",
    "text to speech",
    "speech generation",
    "video generation",
    "generate video",
    "generates video",
    "audio generation",
    "music generation",
    "lyria",
    "deep research",
    "antigravity",
)


class _GeminiPart(BaseModel):
    model_config = ConfigDict(extra="ignore")
    text: str = ""
    thought: bool = False


class _GeminiContent(BaseModel):
    parts: list[_GeminiPart] = Field(min_length=1)


class _GeminiCandidate(BaseModel):
    content: _GeminiContent


class _GeminiResponsePayload(BaseModel):
    candidates: list[_GeminiCandidate] = Field(min_length=1)


class _GeminiListedModelPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = ""
    displayName: str = ""
    description: str = ""
    supportedGenerationMethods: list[str] = []
    # Not in the published v1beta Model schema (or google.genai.types.Model). Parsed
    # if Google starts sending them; list filtering does not depend on them.
    supportedInputModalities: list[str] = []
    supportedOutputModalities: list[str] = []


class _GeminiListModelsResponsePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")
    models: list[_GeminiListedModelPayload] = []
    nextPageToken: str | None = None


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
    # Skip thought parts — Enhance needs the rewritten prompt, not the reasoning trace.
    text_parts = [part.text for part in parsed.candidates[0].content.parts if not part.thought]
    return "".join(text_parts)


def normalize_gemini_model_id(model: str) -> str:
    stripped = model.strip()
    if stripped.startswith(_MODELS_PREFIX):
        stripped = stripped[len(_MODELS_PREFIX) :]
    return stripped


def resolve_gemini_model(stored: str) -> str:
    """Empty/missing stored value means the out-of-the-box default.

    A previously persisted image/audio/video generator (Nano Banana, Omni, …) is
    treated the same as empty — Enhance and gap suggestions are text-out only.
    """
    model_id = normalize_gemini_model_id(stored) or DEFAULT_GEMINI_MODEL
    if model_id != DEFAULT_GEMINI_MODEL and not is_text_to_text_gemini_model(model_id):
        return DEFAULT_GEMINI_MODEL
    return model_id


def _gemini_model_sort_key(model_id: str) -> tuple[int, str]:
    lowered = model_id.lower()
    flash_or_lite = 0 if ("flash" in lowered or "lite" in lowered) else 1
    return (flash_or_lite, model_id)


def _gemini_auth_headers(api_key: str) -> dict[str, str]:
    return {"x-goog-api-key": api_key}


def gemini_generate_content_url(model: str) -> str:
    model_id = resolve_gemini_model(model)
    return f"{GEMINI_API_BASE_URL}/models/{model_id}:generateContent"


def _list_models_url(*, page_token: str | None = None) -> str:
    params: dict[str, str] = {"pageSize": str(_GEMINI_LIST_PAGE_SIZE)}
    if page_token:
        params["pageToken"] = page_token
    return f"{GEMINI_API_BASE_URL}/models?{urlencode(params)}"


def _normalized_modalities(values: list[str]) -> list[str]:
    return [value.strip().upper() for value in values if value.strip()]


def is_text_to_text_gemini_model(
    model_id: str,
    description: str = "",
    display_name: str = "",
    input_modalities: list[str] | None = None,
    output_modalities: list[str] | None = None,
) -> bool:
    """True for Gemini-API chat models that emit text (optional image *input*).

    Hosted Gemma is excluded: generateContent comes back empty on Enhance's system
    prompt. Local Enhance already uses Gemma on-device.

    If supported_*_modalities are present, output must be text-only. Input may
    include IMAGE (Enhance i2v / gap frames). Requiring input==[TEXT] would drop
    Flash.
    """
    lowered_id = normalize_gemini_model_id(model_id).lower()
    if lowered_id.startswith("gemma-"):
        return False
    outputs = _normalized_modalities(output_modalities or [])
    inputs = _normalized_modalities(input_modalities or [])
    if outputs:
        if outputs != ["TEXT"]:
            return False
        return not inputs or "TEXT" in inputs

    if not lowered_id.startswith(_TEXT_MODEL_ID_PREFIXES):
        return False
    if any(marker in lowered_id for marker in _NON_TEXT_OUTPUT_ID_MARKERS):
        return False
    label = f"{display_name} {description}".lower()
    return not any(marker in label for marker in _NON_TEXT_OUTPUT_LABEL_MARKERS)


def _sorted_gemini_model_options(
    models_by_id: dict[str, GeminiModelOptionPayload],
) -> list[GeminiModelOptionPayload]:
    return [models_by_id[model_id] for model_id in sorted(models_by_id, key=_gemini_model_sort_key)]


def clear_gemini_models_cache() -> None:
    global _gemini_models_list_cache
    with _gemini_models_list_cache_lock:
        _gemini_models_list_cache = None


def _cached_gemini_models(api_key: str) -> list[GeminiModelOptionPayload] | None:
    with _gemini_models_list_cache_lock:
        entry = _gemini_models_list_cache
        if entry is None or entry.api_key != api_key or entry.expires_at <= time.monotonic():
            return None
        return [model.model_copy() for model in entry.models]


def _store_gemini_models_cache(api_key: str, models: list[GeminiModelOptionPayload]) -> None:
    global _gemini_models_list_cache
    with _gemini_models_list_cache_lock:
        _gemini_models_list_cache = _GeminiModelsListCache(
            api_key=api_key,
            models=tuple(model.model_copy() for model in models),
            expires_at=time.monotonic() + _GEMINI_MODELS_CACHE_TTL_S,
        )


def _with_included_model(
    models: list[GeminiModelOptionPayload],
    include_id: str | None,
) -> list[GeminiModelOptionPayload]:
    models_by_id = {model.id: model for model in models}
    included = normalize_gemini_model_id(include_id) if include_id else ""
    # Don't re-inject a stored image/audio/video generator the filter just dropped —
    # otherwise a leftover Nano Banana setting stays selectable and callable.
    if included and included not in models_by_id and is_text_to_text_gemini_model(included):
        models_by_id[included] = GeminiModelOptionPayload(
            id=included,
            displayName=included,
            description="",
        )
    return _sorted_gemini_model_options(models_by_id)


def list_gemini_generate_content_models(
    http: HTTPClient,
    *,
    api_key: str,
    include_id: str | None = None,
) -> list[GeminiModelOptionPayload]:
    """Page Gemini's models.list, keeping text-output generateContent models.

    `include_id` is appended when the resolved setting is not in the upstream list so a
    dropdown bound to that id is never blank. Successful lists are cached per API key for
    an hour — Settings reopens should not hit Google again.
    """
    cached = _cached_gemini_models(api_key)
    if cached is not None:
        return _with_included_model(cached, include_id)

    models_by_id: dict[str, GeminiModelOptionPayload] = {}
    page_token: str | None = None
    for _ in range(_GEMINI_LIST_MAX_PAGES):
        try:
            response = http.get(
                _list_models_url(page_token=page_token),
                headers=_gemini_auth_headers(api_key),
                timeout=30,
            )
        except HttpTransportError as exc:
            raise HTTPError(504, "Gemini API request timed out") from exc

        if response.status_code != 200:
            raise HTTPError(response.status_code, f"Gemini API error: {response.text}")

        try:
            parsed = _GeminiListModelsResponsePayload.model_validate(response.json())
        except ValidationError:
            raise HTTPError(500, "GEMINI_PARSE_ERROR")

        for item in parsed.models:
            if "generateContent" not in item.supportedGenerationMethods:
                continue
            model_id = normalize_gemini_model_id(item.name)
            description = item.description.strip()
            display_name = item.displayName.strip()
            if (
                not model_id
                or model_id in models_by_id
                or not is_text_to_text_gemini_model(
                    model_id,
                    description,
                    display_name=display_name,
                    input_modalities=item.supportedInputModalities,
                    output_modalities=item.supportedOutputModalities,
                )
            ):
                continue
            models_by_id[model_id] = GeminiModelOptionPayload(
                id=model_id,
                displayName=display_name or model_id,
                description=description,
            )

        page_token = (parsed.nextPageToken or "").strip() or None
        if not page_token:
            break

    fetched = _sorted_gemini_model_options(models_by_id)
    _store_gemini_models_cache(api_key, fetched)
    return _with_included_model(fetched, include_id)


# Thinking tokens count against maxOutputTokens. 512 is only enough when thinking is off;
# otherwise the rewrite is truncated (or missing parts entirely).
_THINKING_MODEL_MAX_OUTPUT_TOKENS = 2048
_2_5_PRO_MIN_THINKING_BUDGET = 128


def gemini_thinking_config_for_model(model: str) -> dict[str, JSONValue] | None:
    """Per-model thinking so Enhance gets a full rewritten prompt.

    2.5 Flash/Lite: thinkingBudget 0 (otherwise thinking eats the output budget).
    2.5 Pro: thinking cannot be 0; use the minimum allowed budget.
    Gemini 3: thinkingLevel LOW (MINIMAL 400s on some Flash/Pro variants).
    """
    lowered = normalize_gemini_model_id(model).lower()
    if lowered.startswith("gemini-3"):
        return {"thinkingLevel": "LOW"}
    if "2.5" not in lowered:
        return None
    if "flash" in lowered or "lite" in lowered:
        return {"thinkingBudget": 0}
    if "pro" in lowered:
        return {"thinkingBudget": _2_5_PRO_MIN_THINKING_BUDGET}
    return None


def apply_gemini_thinking_config(
    model: str, generation_config: dict[str, JSONValue]
) -> dict[str, JSONValue]:
    thinking_config = gemini_thinking_config_for_model(model)
    if thinking_config is None:
        return generation_config
    config: dict[str, JSONValue] = {**generation_config, "thinkingConfig": thinking_config}
    if thinking_config.get("thinkingBudget") == 0:
        return config
    current = config.get("maxOutputTokens")
    if isinstance(current, int) and current < _THINKING_MODEL_MAX_OUTPUT_TOKENS:
        config["maxOutputTokens"] = _THINKING_MODEL_MAX_OUTPUT_TOKENS
    return config


def _is_invalid_api_key_error(response: HttpResponseLike) -> bool:
    """True for Gemini 400s with error.status INVALID_ARGUMENT and details.reason API_KEY_INVALID."""
    try:
        body = response.json()
    except ValueError:
        return False
    if not isinstance(body, dict):
        return False
    error = cast(dict[str, object], body).get("error")
    if not isinstance(error, dict):
        return False
    error_fields = cast(dict[str, object], error)
    if error_fields.get("status") != "INVALID_ARGUMENT":
        return False
    details = error_fields.get("details")
    if not isinstance(details, list):
        return False
    for item in cast(list[object], details):
        if isinstance(item, dict) and cast(dict[str, object], item).get("reason") == "API_KEY_INVALID":
            return True
    return False


def call_gemini_generate_content(
    http: HTTPClient,
    *,
    api_key: str,
    model: str,
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
            gemini_generate_content_url(model),
            headers={"Content-Type": "application/json", **_gemini_auth_headers(api_key)},
            json_payload=payload,
            timeout=timeout,
        )
    except HttpTransportError as exc:
        raise HTTPError(504, "Gemini API request timed out") from exc

    if response.status_code != 200:
        if _is_invalid_api_key_error(response):
            raise HTTPError(
                response.status_code,
                "Gemini rejected the configured API key",
                code="GEMINI_INVALID_API_KEY",
            )
        raise HTTPError(response.status_code, f"Gemini API error: {response.text}")

    text = extract_gemini_text(response.json()).strip()
    if not text:
        # Whitespace-only/empty text is a valid, non-blocked candidate as far as the schema is
        # concerned — but handing it back as `enhancedPrompt` would silently wipe the user's
        # prompt on what looks like a "success" response.
        raise HTTPError(500, "Gemini returned an empty response", code="GEMINI_EMPTY_RESPONSE")
    return text
