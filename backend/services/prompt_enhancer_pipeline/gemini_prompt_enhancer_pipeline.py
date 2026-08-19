"""Remote prompt enhancement backed by Gemini's hosted API — no local checkpoint required."""

from __future__ import annotations

import base64
from pathlib import Path

from PIL import Image

from _routes._errors import HTTPError
from services.gemini_text_client import apply_gemini_thinking_config, call_gemini_generate_content
from services.interfaces import HTTPClient, JSONValue
from services.prompt_enhancement import build_default_free_rewrite_system_prompt

# Gemini's inlineData only takes these formats (also HEIC/HEIF, which never reach here — this
# app's own image validation doesn't accept them as input in the first place).
_GEMINI_SUPPORTED_IMAGE_FORMATS = {"PNG", "JPEG", "WEBP"}
# Inline (non-Files-API) requests cap the *total* request — text + image bytes — at 20MB.
_GEMINI_MAX_INLINE_BYTES = 20 * 1024 * 1024


class GeminiPromptEnhancerPipeline:
    """Long-lived instance (unlike the local pipeline, nothing per-call needs loading/freeing).

    The API key is passed per call rather than baked in at construction — it can change at
    runtime via Settings, and this instance is constructed once at app startup.
    """

    def __init__(self, http: HTTPClient) -> None:
        self._http = http

    def enhance_t2v(self, prompt: str, system_prompt: str | None, seed: int, *, api_key: str, model: str) -> str:
        contents: list[JSONValue] = [{"role": "user", "parts": [{"text": prompt}]}]
        return self._call(contents, system_prompt, seed, api_key, model)

    def enhance_i2v(
        self, prompt: str, image_path: str, system_prompt: str | None, seed: int, *, api_key: str, model: str
    ) -> str:
        # Our own validate_image_file() allows more (GIF/BMP/TIFF, up to 50MB) than Gemini's
        # inlineData accepts — without this, those pass our gate and only fail once they bounce
        # off Gemini as an opaque upstream error.
        with Image.open(image_path) as img:
            fmt = str(img.format or "").upper()
        if fmt not in _GEMINI_SUPPORTED_IMAGE_FORMATS:
            raise HTTPError(
                400,
                f"Image format {fmt or 'unknown'} isn't supported by the Gemini API provider "
                "(use Local, or convert the image to PNG/JPEG/WEBP)",
                code="GEMINI_UNSUPPORTED_IMAGE_FORMAT",
            )
        size_bytes = Path(image_path).stat().st_size
        if size_bytes > _GEMINI_MAX_INLINE_BYTES:
            raise HTTPError(
                400,
                "Image is too large for the Gemini API provider (20MB request limit) — "
                "use Local, or a smaller image",
                code="GEMINI_IMAGE_TOO_LARGE",
            )

        image_data = base64.b64encode(Path(image_path).read_bytes()).decode()
        parts: list[JSONValue] = [
            {"text": prompt},
            {"inlineData": {"mimeType": f"image/{fmt.lower()}", "data": image_data}},
        ]
        contents: list[JSONValue] = [{"role": "user", "parts": parts}]
        return self._call(contents, system_prompt, seed, api_key, model)

    def _call(
        self, contents: list[JSONValue], system_prompt: str | None, seed: int, api_key: str, model: str
    ) -> str:
        # Unlike the local Gemma pipeline, Gemini has no implicit default system prompt of its
        # own — omitting systemInstruction entirely gets a chatty, markdown-formatted essay
        # instead of a rewritten prompt. Always resolve to something.
        resolved_system_prompt = system_prompt or build_default_free_rewrite_system_prompt()
        return call_gemini_generate_content(
            self._http,
            api_key=api_key,
            model=model,
            contents=contents,
            system_instruction=resolved_system_prompt,
            # maxOutputTokens 512 matches local Gemma when thinking is off. Thinking models
            # get a higher cap in apply_gemini_thinking_config so the rewrite is not truncated.
            generation_config=apply_gemini_thinking_config(
                model,
                {"seed": seed, "maxOutputTokens": 512},
            ),
            timeout=30,
        )
