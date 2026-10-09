"""Remote prompt enhancement backed by Gemini's hosted API — no local checkpoint required."""

from __future__ import annotations

import base64
import io

from PIL import Image

from _routes._errors import HTTPError
from server_utils.oriented_image import open_oriented_image
from services.gemini_text_client import apply_gemini_thinking_config, call_gemini_generate_content
from services.interfaces import HTTPClient, JSONValue
from services.prompt_enhancement import (
    build_default_free_rewrite_system_prompt,
    build_i2v_user_prompt_text,
    resolve_i2v_frames,
)
from services.prompt_enhancement.i2v_frames import KeyframeStill

# Gemini's inlineData only takes these formats (also HEIC/HEIF, which never reach here — this
# app's own image validation doesn't accept them as input in the first place).
_GEMINI_SUPPORTED_IMAGE_FORMATS = {"PNG", "JPEG", "WEBP"}
# Inline (non-Files-API) requests cap the *total* request — text + image bytes — at 20MB.
_GEMINI_MAX_INLINE_BYTES = 20 * 1024 * 1024
# Match local Gemma's vision long-edge. A 4K frame does not help the rewrite and burns tokens.
_ENHANCE_IMAGE_LONG_EDGE = 896
_JPEG_QUALITY = 85


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
        self,
        prompt: str,
        image_path: str,
        system_prompt: str | None,
        seed: int,
        *,
        api_key: str,
        model: str,
        last_image_path: str | None = None,
        keyframes: list[KeyframeStill] | None = None,
        duration: int | None = None,
        fps: int | None = None,
    ) -> str:
        frames = resolve_i2v_frames(image_path, last_image_path, keyframes, fps=fps)
        image_parts: list[tuple[str | None, JSONValue]] = []
        total_size = 0
        for path, label in frames:
            image_part, image_size = self._inline_image_part(path)
            total_size += image_size
            image_parts.append((label, image_part))
        if total_size > _GEMINI_MAX_INLINE_BYTES:
            raise HTTPError(
                400,
                "Image is too large for the Gemini API provider (20MB request limit) — "
                "use Local, or a smaller image",
                code="GEMINI_IMAGE_TOO_LARGE",
            )

        keyframe_count = len(keyframes) if keyframes else 0
        if keyframe_count > 0:
            user_text = build_i2v_user_prompt_text(
                prompt,
                has_last=False,
                keyframe_count=keyframe_count,
                duration=duration,
                fps=fps,
            )
        elif last_image_path:
            user_text = prompt.strip() or build_i2v_user_prompt_text(prompt, has_last=True)
        else:
            user_text = prompt if prompt.strip() else build_i2v_user_prompt_text(prompt, has_last=False)

        # Single unlabeled still keeps the original Gemini turn (caption, then image).
        # Labeled first/last and keyframe sequences interleave a label before each still.
        if len(image_parts) == 1 and image_parts[0][0] is None:
            parts: list[JSONValue] = [{"text": user_text}, image_parts[0][1]]
        else:
            parts = []
            for label, image_part in image_parts:
                if label is not None:
                    parts.append({"text": label})
                parts.append(image_part)
            parts.append({"text": user_text})
        contents: list[JSONValue] = [{"role": "user", "parts": parts}]
        return self._call(contents, system_prompt, seed, api_key, model)

    def _inline_image_part(self, image_path: str) -> tuple[JSONValue, int]:
        # Our own validate_image_file() allows more (GIF/BMP/TIFF, up to 50MB) than Gemini's
        # inlineData accepts — without this, those pass our gate and only fail once they bounce
        # off Gemini as an opaque upstream error.
        img = open_oriented_image(image_path)
        fmt = str(img.format or "").upper()
        if fmt not in _GEMINI_SUPPORTED_IMAGE_FORMATS:
            raise HTTPError(
                400,
                f"Image format {fmt or 'unknown'} isn't supported by the Gemini API provider "
                "(use Local, or convert the image to PNG/JPEG/WEBP)",
                code="GEMINI_UNSUPPORTED_IMAGE_FORMAT",
            )
        image = img.convert("RGB")
        image.thumbnail(
            (_ENHANCE_IMAGE_LONG_EDGE, _ENHANCE_IMAGE_LONG_EDGE),
            Image.Resampling.LANCZOS,
        )
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
        raw = buffer.getvalue()
        encoded = base64.b64encode(raw).decode()
        part: JSONValue = {"inlineData": {"mimeType": "image/jpeg", "data": encoded}}
        return part, len(raw)

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
