"""Z-Image Turbo API client implementation for FAL endpoints."""

from __future__ import annotations

import base64
import logging
import re
import time
from collections.abc import Callable
from typing import Any, cast
from urllib.parse import urlsplit

from api_types import ApiActivityStatus
from services.generation_interrupt import GenerationCancelledError
from services.http_client.http_client import (
    HTTPClient,
    HttpBodyTooLargeError,
    HttpRedirectRefusedError,
    HttpTransportError,
)
from services.zit_api_client.zit_api_client import ZitImageBlockedError, ZitQueuedImage
from services.services_utils import JSONValue

logger = logging.getLogger(__name__)

FAL_API_BASE_URL = "https://fal.run"
FAL_QUEUE_BASE_URL = "https://queue.fal.run"
FAL_TEXT_TO_IMAGE_ENDPOINT = "/fal-ai/z-image/turbo"
# The app id of the queued endpoint. fal puts it, not the sub-path, in the request URLs.
FAL_QUEUE_APP = "/".join(FAL_TEXT_TO_IMAGE_ENDPOINT.split("/")[:3])
FAL_IMAGE_TO_IMAGE_ENDPOINT = "/fal-ai/z-image/turbo/image-to-image"

DEFAULT_OUTPUT_FORMAT = "png"
DEFAULT_ACCELERATION = "regular"
DEFAULT_ENABLE_SAFETY_CHECKER = True

QUEUE_POLL_INTERVAL_SECONDS = 1.0
# A request that waits or runs longer than this fails the job.
QUEUE_TIMEOUT_SECONDS = 600.0
# Status checks that fail in a row before the job fails. One dropped poll is not an error.
_MAX_CONSECUTIVE_STATUS_FAILURES = 3
# Shorter than the fal lane stop grace, so a cancel at shutdown can finish inside it.
_PROVIDER_CANCEL_TIMEOUT_SECONDS = 4
# The largest image the app reads from fal. A Z-Image PNG at the top size is a few MB.
_MAX_IMAGE_DOWNLOAD_BYTES = 50 * 1024 * 1024

# A fal request id is a UUID. It goes into a URL path, so nothing else is accepted.
_REQUEST_ID = re.compile(r"[A-Za-z0-9-]{1,100}")

_STATUS_TO_API_STATUS: dict[str, ApiActivityStatus] = {
    "IN_QUEUE": "pending",
    "IN_PROGRESS": "processing",
}


class ZitAPIClientImpl:
    def __init__(
        self,
        http: HTTPClient,
        *,
        fal_api_base_url: str = FAL_API_BASE_URL,
        fal_queue_base_url: str = FAL_QUEUE_BASE_URL,
        queue_poll_interval_seconds: float = QUEUE_POLL_INTERVAL_SECONDS,
        queue_timeout_seconds: float = QUEUE_TIMEOUT_SECONDS,
    ) -> None:
        self._http = http
        self._base_url = fal_api_base_url.rstrip("/")
        self._queue_base_url = fal_queue_base_url.rstrip("/")
        self._poll_interval = queue_poll_interval_seconds
        self._queue_timeout = queue_timeout_seconds

    def generate_text_to_image(
        self,
        *,
        api_key: str,
        prompt: str,
        width: int,
        height: int,
        seed: int,
        num_inference_steps: int,
    ) -> bytes:
        payload = self._text_to_image_payload(
            prompt=prompt,
            width=width,
            height=height,
            seed=seed,
            num_inference_steps=num_inference_steps,
            # fal defaults this to false. The explicit value keeps GenSpace off the paid expansion.
            enable_prompt_expansion=False,
        )
        response_payload = self._post_blocking(
            endpoint=FAL_TEXT_TO_IMAGE_ENDPOINT, api_key=api_key, payload=payload
        )
        return self._download(self._extract_image_url(response_payload))

    def generate_text_to_image_queued(
        self,
        *,
        api_key: str,
        prompt: str,
        width: int,
        height: int,
        seed: int,
        num_inference_steps: int,
        enable_prompt_expansion: bool,
        on_status: Callable[[ApiActivityStatus], None],
        on_submitted: Callable[[str], bool],
        wait_cancelled: Callable[[float], bool],
    ) -> ZitQueuedImage:
        payload = self._text_to_image_payload(
            prompt=prompt,
            width=width,
            height=height,
            seed=seed,
            num_inference_steps=num_inference_steps,
            enable_prompt_expansion=enable_prompt_expansion,
        )
        submitted = self._submit_queued(api_key, payload)
        headers = self._json_headers(api_key)
        try:
            # The id must be stored before the first status request. A boot cannot cancel a
            # request it has no id for, so a request that is not recorded is dropped now.
            if not on_submitted(submitted["request_id"]):
                raise RuntimeError("FAL request id was not stored, so the request was cancelled")
            # A cancel that landed during the submit must not start a status request first.
            self._raise_if_cancelled(wait_cancelled)
            self._wait_until_completed(
                submitted, headers, on_status=on_status, wait_cancelled=wait_cancelled
            )
            self._raise_if_cancelled(wait_cancelled)
            result = self._fetch_result(submitted["response_url"], headers)
            self._raise_if_cancelled(wait_cancelled)
        except Exception:
            # Cancel, timeout and every failure free the provider side too. Best effort.
            self._cancel_at_provider(submitted["cancel_url"], headers)
            raise
        if self._is_blocked(result):
            raise ZitImageBlockedError("FAL safety checker blocked the image")
        image = self._download(self._extract_image_url(result))
        used = result.get("prompt")
        return ZitQueuedImage(
            image_bytes=image,
            prompt=used if isinstance(used, str) and used.strip() else None,
        )

    def generate_image_to_image(
        self,
        *,
        api_key: str,
        prompt: str,
        image_bytes: bytes,
        strength: float,
        seed: int,
        num_inference_steps: int,
    ) -> bytes:
        data_uri = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
        payload: dict[str, JSONValue] = {
            "prompt": prompt,
            "image_url": data_uri,
            "strength": strength,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "num_images": 1,
            "output_format": DEFAULT_OUTPUT_FORMAT,
            "acceleration": DEFAULT_ACCELERATION,
            "enable_safety_checker": DEFAULT_ENABLE_SAFETY_CHECKER,
        }
        response_payload = self._post_blocking(
            endpoint=FAL_IMAGE_TO_IMAGE_ENDPOINT, api_key=api_key, payload=payload
        )
        return self._download(self._extract_image_url(response_payload))

    @staticmethod
    def _text_to_image_payload(
        *,
        prompt: str,
        width: int,
        height: int,
        seed: int,
        num_inference_steps: int,
        enable_prompt_expansion: bool,
    ) -> dict[str, JSONValue]:
        # One builder for the blocking and the queued request, so the two cannot drift.
        return {
            "prompt": prompt,
            "image_size": {"width": width, "height": height},
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "num_images": 1,
            "output_format": DEFAULT_OUTPUT_FORMAT,
            "acceleration": DEFAULT_ACCELERATION,
            "enable_safety_checker": DEFAULT_ENABLE_SAFETY_CHECKER,
            "enable_prompt_expansion": enable_prompt_expansion,
        }

    def _post_blocking(
        self, *, endpoint: str, api_key: str, payload: dict[str, JSONValue]
    ) -> dict[str, Any]:
        response = self._http.post(
            f"{self._base_url}{endpoint}",
            headers=self._json_headers(api_key),
            json_payload=payload,
            timeout=180,
        )
        if response.status_code != 200:
            detail = response.text[:500] if response.text else "Unknown error"
            raise RuntimeError(f"FAL submit failed ({response.status_code}): {detail}")
        return self._json_object(response.json(), context="submit")

    def _submit_queued(self, api_key: str, payload: dict[str, JSONValue]) -> dict[str, str]:
        response = self._http.post(
            f"{self._queue_base_url}{FAL_TEXT_TO_IMAGE_ENDPOINT}",
            headers=self._json_headers(api_key),
            json_payload=payload,
            timeout=60,
        )
        if response.status_code != 200:
            detail = response.text[:500] if response.text else "Unknown error"
            raise RuntimeError(f"FAL submit failed ({response.status_code}): {detail}")
        body = self._json_object(response.json(), context="submit")
        # Use the URLs fal returns. For an endpoint with a sub-path they differ from a built URL.
        # Every later request sends the API key, so a URL on another host is refused.
        queue_origin = urlsplit(self._queue_base_url)[:2]
        urls: dict[str, str] = {}
        for key in ("status_url", "response_url", "cancel_url"):
            value = body.get(key)
            if not isinstance(value, str) or not value:
                raise RuntimeError(f"FAL queue response missing {key}")
            if urlsplit(value)[:2] != queue_origin:
                raise RuntimeError(f"FAL queue response has a {key} on an unexpected host")
            urls[key] = value
        request_id = body.get("request_id")
        if not isinstance(request_id, str) or not _REQUEST_ID.fullmatch(request_id):
            raise RuntimeError("FAL queue response has no usable request_id")
        urls["request_id"] = request_id
        return urls

    def _wait_until_completed(
        self,
        submitted: dict[str, str],
        headers: dict[str, str],
        *,
        on_status: Callable[[ApiActivityStatus], None],
        wait_cancelled: Callable[[float], bool],
    ) -> None:
        deadline = time.monotonic() + self._queue_timeout
        failures = 0
        while True:
            try:
                status = self._poll_status(submitted["status_url"], headers)
            except (HttpTransportError, RuntimeError) as exc:
                failures += 1
                if failures >= _MAX_CONSECUTIVE_STATUS_FAILURES:
                    raise RuntimeError(f"FAL status check failed: {exc}") from exc
            else:
                failures = 0
                if status.get("status") == "COMPLETED":
                    error = status.get("error")
                    if error:
                        raise RuntimeError(f"FAL request failed: {str(error)[:500]}")
                    return
                on_status(_STATUS_TO_API_STATUS.get(str(status.get("status")), "pending"))
            if time.monotonic() >= deadline:
                raise RuntimeError("FAL request timed out")
            if wait_cancelled(self._poll_interval):
                raise GenerationCancelledError()

    @staticmethod
    def _raise_if_cancelled(wait_cancelled: Callable[[float], bool]) -> None:
        """A cancel that lands during a request must not start the next, longer one."""
        if wait_cancelled(0):
            raise GenerationCancelledError()

    def _poll_status(self, status_url: str, headers: dict[str, str]) -> dict[str, Any]:
        response = self._http.get(status_url, headers=headers, timeout=30)
        if response.status_code not in (200, 202):
            detail = response.text[:200] if response.text else "Unknown error"
            raise RuntimeError(f"({response.status_code}) {detail}")
        try:
            body = response.json()
        except ValueError as exc:
            # A proxy page with status 200 is a transient failure, like a bad status code.
            raise RuntimeError("FAL status response is not JSON") from exc
        return self._json_object(body, context="status")

    def _fetch_result(self, response_url: str, headers: dict[str, str]) -> dict[str, Any]:
        response = self._http.get(response_url, headers=headers, timeout=60)
        if response.status_code != 200:
            detail = response.text[:500] if response.text else "Unknown error"
            raise RuntimeError(f"FAL result failed ({response.status_code}): {detail}")
        return self._json_object(response.json(), context="result")

    def cancel_queued(self, *, api_key: str, request_id: str) -> None:
        """Drop a request that a stopped app left at fal. Used at boot, with only the id.

        Only the text-to-image endpoint is queued. fal builds its request URLs from the app id
        (the first two path segments), not from the sub-path, so the URL is derived from that.
        202 (cancelling), 400 (already completed) and 404 (unknown or expired) all mean there
        is nothing left to drop.
        """
        if not _REQUEST_ID.fullmatch(request_id):
            raise ValueError("not a fal request id")
        response = self._http.put(
            f"{self._queue_base_url}{FAL_QUEUE_APP}/requests/{request_id}/cancel",
            headers=self._json_headers(api_key),
            timeout=_PROVIDER_CANCEL_TIMEOUT_SECONDS,
        )
        if response.status_code not in (200, 202, 400, 404):
            raise RuntimeError(f"FAL cancel failed ({response.status_code})")

    def _cancel_at_provider(self, cancel_url: str, headers: dict[str, str]) -> None:
        """Ask fal to drop the request. 202 and 400 ALREADY_COMPLETED are both fine."""
        try:
            self._http.put(cancel_url, headers=headers, timeout=_PROVIDER_CANCEL_TIMEOUT_SECONDS)
        except Exception as exc:
            logger.warning("FAL cancel request failed: %s", exc)

    def _download(self, image_url: str) -> bytes:
        # The URL comes from the fal response. Refuse a plain-http URL (CWE-918) and cap the
        # body (CWE-770). The error text has the status only, not the body (CWE-209).
        if urlsplit(image_url).scheme != "https":
            raise RuntimeError("FAL image URL is not https")
        try:
            download = self._http.get(
                image_url, timeout=120, max_bytes=_MAX_IMAGE_DOWNLOAD_BYTES
            )
        except HttpBodyTooLargeError as exc:
            raise RuntimeError("FAL image is too large") from exc
        except HttpRedirectRefusedError as exc:
            raise RuntimeError("FAL image redirect refused") from exc
        if download.status_code != 200:
            raise RuntimeError(f"FAL image download failed ({download.status_code})")
        if not download.content:
            raise RuntimeError("FAL image download returned empty body")
        return download.content

    @staticmethod
    def _json_headers(api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Key {api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _is_blocked(payload: dict[str, Any]) -> bool:
        flags = payload.get("has_nsfw_concepts")
        if isinstance(flags, list) and any(flag is True for flag in cast(list[object], flags)):
            return True
        images = payload.get("images")
        return isinstance(images, list) and not images

    @staticmethod
    def _extract_image_url(payload: dict[str, Any]) -> str:
        images = payload.get("images")
        if isinstance(images, list) and images:
            images_list = cast(list[object], images)
            first = images_list[0]
            if isinstance(first, dict):
                first_payload = cast(dict[str, Any], first)
                url = first_payload.get("url")
                if isinstance(url, str) and url:
                    return url
            if isinstance(first, str) and first:
                return first

        for key in ("image_url", "imageUrl", "url"):
            url = payload.get(key)
            if isinstance(url, str) and url:
                return url

        raise RuntimeError("FAL response missing image url")

    @staticmethod
    def _json_object(payload: object, *, context: str) -> dict[str, Any]:
        if isinstance(payload, dict):
            return cast(dict[str, Any], payload)
        raise RuntimeError(f"Unexpected FAL {context} response format")
