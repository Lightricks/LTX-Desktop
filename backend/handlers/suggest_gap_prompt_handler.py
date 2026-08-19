"""Gap prompt suggestion handler (Gemini-powered)."""

from __future__ import annotations

import base64
import logging
from threading import RLock
from typing import TYPE_CHECKING

from api_types import (
    SuggestGapPromptRequest,
    SuggestGapPromptResponse,
)
from _routes._errors import HTTPError
from handlers.base import StateHandlerBase
from server_utils.media_validation import image_mime_type, normalize_optional_path, validate_image_file
from services.gemini_text_client import (
    apply_gemini_thinking_config,
    call_gemini_generate_content,
    resolve_gemini_model,
)
from services.interfaces import HTTPClient, JSONValue
from state.app_state_types import AppState

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)


def _read_image_file_as_base64(file_path: str | None) -> tuple[str, str] | None:
    """Read an image file from disk and return (base64 data, real MIME type) — the real format,
    not a hardcoded guess, since Gemini's inlineData needs the declared type to match the bytes.
    """
    normalized = normalize_optional_path(file_path)
    if not normalized:
        return None
    try:
        validated_path = validate_image_file(normalized)
    except HTTPError as exc:
        logger.warning("Ignoring invalid image file for gap prompt: %s (%s)", normalized, exc.detail)
        return None
    try:
        data = base64.b64encode(validated_path.read_bytes()).decode()
        return data, image_mime_type(str(validated_path))
    except Exception:
        logger.warning("Failed to read image file for gap prompt: %s", normalized, exc_info=True)
        return None


class SuggestGapPromptHandler(StateHandlerBase):
    def __init__(self, state: AppState, lock: RLock, config: RuntimeConfig, http: HTTPClient) -> None:
        super().__init__(state, lock, config)
        self._http = http

    def suggest_gap(self, req: SuggestGapPromptRequest) -> SuggestGapPromptResponse:
        before_frame = _read_image_file_as_base64(req.beforeFrame)
        after_frame = _read_image_file_as_base64(req.afterFrame)
        input_image = _read_image_file_as_base64(req.inputImage)
        before_prompt = req.beforePrompt
        after_prompt = req.afterPrompt
        gap_duration = req.gapDuration
        mode = req.mode

        if not before_frame and not after_frame and not before_prompt and not after_prompt:
            raise HTTPError(400, "At least one neighboring frame or prompt is required")

        gemini_api_key = self.state.app_settings.gemini_api_key
        if not gemini_api_key:
            raise HTTPError(400, "GEMINI_API_KEY_MISSING")

        is_image_gen = mode == "text-to-image"
        is_image_to_video = mode == "image-to-video"
        if not is_image_to_video:
            input_image = None

        system_text = (
            "You are a video production assistant. The user is editing a video timeline and has a gap "
            f"of {gap_duration:.1f} seconds between two shots. Your job is to suggest a detailed prompt "
            f"for generating {'an image' if is_image_gen else 'a video clip'} to fill this gap, so that it flows naturally between the "
            "preceding and following shots.\n\n"
            "Guidelines:\n"
            f"- Describe the scene, {'composition' if is_image_gen else 'action, camera movement'}, lighting, and mood\n"
            "- Match the visual style and tone of the surrounding shots\n"
            "- Create a smooth narrative or visual transition between the two shots\n"
            "- Keep the prompt concise (2-4 sentences max)\n"
            "- Write only the prompt text, no explanations or labels\n"
            "- If only one neighboring shot is available, suggest something that naturally leads into or out of it\n"
        )

        context_text = "Here is the context from the timeline:\n\n"
        if before_frame or before_prompt:
            context_text += "SHOT BEFORE THE GAP:\n"
            if before_prompt:
                context_text += f"  Prompt: {before_prompt}\n"
            if before_frame:
                context_text += "  Last frame (see image below):\n"

        if after_frame or after_prompt:
            context_text += "\nSHOT AFTER THE GAP:\n"
            if after_prompt:
                context_text += f"  Prompt: {after_prompt}\n"
            if after_frame:
                context_text += "  First frame (see image below):\n"

        context_text += f"\nGap duration: {gap_duration:.1f} seconds\n"
        mode_label = "image generation" if is_image_gen else ("image-to-video" if is_image_to_video else "text-to-video")
        context_text += f"Generation mode: {mode_label}\n"
        if input_image:
            context_text += "A reference image is provided to guide the start of the shot.\n"
        context_text += "\nPlease suggest a detailed prompt for generating " + ("an image" if is_image_gen else "a video clip") + " to fill this gap."

        user_parts: list[JSONValue] = [{"text": context_text}]

        if input_image:
            data, mime_type = input_image
            user_parts.append({"text": "Reference image for the start of the generated shot:"})
            user_parts.append({"inlineData": {"mimeType": mime_type, "data": data}})
        if before_frame:
            data, mime_type = before_frame
            user_parts.append({"text": "Last frame of the shot BEFORE the gap:"})
            user_parts.append({"inlineData": {"mimeType": mime_type, "data": data}})
        if after_frame:
            data, mime_type = after_frame
            user_parts.append({"text": "First frame of the shot AFTER the gap:"})
            user_parts.append({"inlineData": {"mimeType": mime_type, "data": data}})

        contents: list[JSONValue] = [{"role": "user", "parts": user_parts}]

        try:
            resolved_model = resolve_gemini_model(self.state.app_settings.gemini_model)
            logger.info("Suggesting gap prompt via Gemini API (%s)", resolved_model)
            suggested_prompt = call_gemini_generate_content(
                self._http,
                api_key=gemini_api_key,
                model=resolved_model,
                contents=contents,
                system_instruction=system_text,
                generation_config=apply_gemini_thinking_config(
                    resolved_model,
                    {"temperature": 0.7, "maxOutputTokens": 512},
                ),
                timeout=30,
            )
        except HTTPError as exc:
            logger.error("Gemini gap suggestion error: %s", exc.detail)
            raise
        except Exception as exc:
            raise HTTPError(500, str(exc)) from exc

        return SuggestGapPromptResponse(status="success", suggested_prompt=suggested_prompt)
