"""Replicate API client implementation for cloud image generation."""

from __future__ import annotations

import time
from typing import Any, cast

from services.http_client.http_client import HTTPClient
from services.services_utils import JSONValue

REPLICATE_API_BASE_URL = "https://api.replicate.com/v1"

_MODEL_ROUTES: dict[str, str] = {
    "z-image-turbo": "prunaai/z-image-turbo",
    "nano-banana-2": "google/nano-banana-2",
}

_NANO_BANANA_ASPECT_RATIOS = [
    (1, 1),
    (2, 3),
    (3, 2),
    (3, 4),
    (4, 3),
    (4, 5),
    (5, 4),
    (9, 16),
    (16, 9),
]

_POLL_INTERVAL_SECONDS = 2
_POLL_TIMEOUT_SECONDS = 120


def _closest_aspect_ratio(width: int, height: int) -> str:
    target = width / height
    best: tuple[int, int] | None = None
    best_diff = float("inf")
    for w, h in _NANO_BANANA_ASPECT_RATIOS:
        diff = abs(target - w / h)
        if diff < best_diff:
            best_diff = diff
            best = (w, h)
    assert best is not None
    return f"{best[0]}:{best[1]}"


def _resolution_bucket(width: int, height: int) -> str:
    largest = max(width, height)
    if largest <= 512:
        return "512px"
    if largest <= 1024:
        return "1K"
    if largest <= 2048:
        return "2K"
    return "4K"


class ReplicateImageClientImpl:
    def __init__(self, http: HTTPClient, *, api_base_url: str = REPLICATE_API_BASE_URL) -> None:
        self._http = http
        self._base_url = api_base_url.rstrip("/")

    def generate_text_to_image(
        self,
        *,
        api_key: str,
        model: str,
        prompt: str,
        width: int,
        height: int,
        seed: int,
        num_inference_steps: int,
    ) -> bytes:
        replicate_model = _MODEL_ROUTES.get(model)
        if replicate_model is None:
            raise RuntimeError(f"Unknown image model: {model}")

        input_payload = self._build_input(
            model=model,
            prompt=prompt,
            width=width,
            height=height,
            seed=seed,
            num_inference_steps=num_inference_steps,
        )

        prediction = self._create_prediction(
            api_key=api_key,
            replicate_model=replicate_model,
            input_payload=input_payload,
        )

        output_url = self._wait_for_output(api_key, prediction)
        return self._download_image(output_url)

    def _build_input(
        self,
        *,
        model: str,
        prompt: str,
        width: int,
        height: int,
        seed: int,
        num_inference_steps: int,
    ) -> dict[str, JSONValue]:
        if model == "nano-banana-2":
            return {
                "prompt": prompt,
                "aspect_ratio": _closest_aspect_ratio(width, height),
                "resolution": _resolution_bucket(width, height),
                "output_format": "png",
                "seed": seed,
            }
        return {
            "prompt": prompt,
            "width": width,
            "height": height,
            "seed": seed,
            "num_inference_steps": num_inference_steps,
        }

    def _create_prediction(
        self,
        *,
        api_key: str,
        replicate_model: str,
        input_payload: dict[str, JSONValue],
    ) -> dict[str, Any]:
        url = f"{self._base_url}/models/{replicate_model}/predictions"
        payload: dict[str, JSONValue] = {"input": input_payload}

        response = self._http.post(
            url,
            headers=self._headers(api_key, prefer_wait=True),
            json_payload=payload,
            timeout=180,
        )
        if response.status_code not in (200, 201):
            detail = response.text[:500] if response.text else "Unknown error"
            raise RuntimeError(f"Replicate prediction failed ({response.status_code}): {detail}")

        return self._json_object(response.json(), context="create prediction")

    def _wait_for_output(self, api_key: str, prediction: dict[str, Any]) -> str:
        status = prediction.get("status", "")
        if status == "succeeded":
            return self._extract_output_url(prediction)

        if status in ("failed", "canceled"):
            error = prediction.get("error", "Unknown error")
            raise RuntimeError(f"Replicate prediction {status}: {error}")

        poll_url = prediction.get("urls", {}).get("get")
        if not isinstance(poll_url, str) or not poll_url:
            prediction_id = prediction.get("id", "")
            poll_url = f"{self._base_url}/predictions/{prediction_id}"

        deadline = time.monotonic() + _POLL_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            time.sleep(_POLL_INTERVAL_SECONDS)
            resp = self._http.get(poll_url, headers=self._headers(api_key), timeout=30)
            if resp.status_code != 200:
                detail = resp.text[:500] if resp.text else "Unknown error"
                raise RuntimeError(f"Replicate poll failed ({resp.status_code}): {detail}")

            data = self._json_object(resp.json(), context="poll")
            poll_status = data.get("status", "")
            if poll_status == "succeeded":
                return self._extract_output_url(data)
            if poll_status in ("failed", "canceled"):
                error = data.get("error", "Unknown error")
                raise RuntimeError(f"Replicate prediction {poll_status}: {error}")

        raise RuntimeError("Replicate prediction timed out")

    def _download_image(self, url: str) -> bytes:
        download = self._http.get(url, timeout=120)
        if download.status_code != 200:
            detail = download.text[:500] if download.text else "Unknown error"
            raise RuntimeError(f"Replicate image download failed ({download.status_code}): {detail}")
        if not download.content:
            raise RuntimeError("Replicate image download returned empty body")
        return download.content

    @staticmethod
    def _headers(api_key: str, *, prefer_wait: bool = False) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if prefer_wait:
            headers["Prefer"] = "wait"
        return headers

    @staticmethod
    def _extract_output_url(prediction: dict[str, Any]) -> str:
        output = prediction.get("output")
        if isinstance(output, list) and output:
            output_list = cast(list[object], output)
            first = output_list[0]
            if isinstance(first, str) and first:
                return first

        if isinstance(output, str) and output:
            return output

        raise RuntimeError("Replicate response missing output URL")

    @staticmethod
    def _json_object(payload: object, *, context: str) -> dict[str, Any]:
        if isinstance(payload, dict):
            return cast(dict[str, Any], payload)
        raise RuntimeError(f"Unexpected Replicate {context} response format")
