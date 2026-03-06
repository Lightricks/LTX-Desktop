"""Replicate API client implementation for cloud video generation."""

from __future__ import annotations

import time
from typing import Any, cast

from services.http_client.http_client import HTTPClient
from services.services_utils import JSONValue

REPLICATE_API_BASE_URL = "https://api.replicate.com/v1"

_MODEL_ROUTES: dict[str, str] = {
    "seedance-1.5-pro": "bytedance/seedance-1.5-pro",
}

_POLL_INTERVAL_SECONDS = 2
_POLL_TIMEOUT_SECONDS = 300


class ReplicateVideoClientImpl:
    def __init__(self, http: HTTPClient, *, api_base_url: str = REPLICATE_API_BASE_URL) -> None:
        self._http = http
        self._base_url = api_base_url.rstrip("/")

    def generate_text_to_video(
        self,
        *,
        api_key: str,
        model: str,
        prompt: str,
        duration: int,
        resolution: str,
        aspect_ratio: str,
        generate_audio: bool,
    ) -> bytes:
        replicate_model = _MODEL_ROUTES.get(model)
        if replicate_model is None:
            raise RuntimeError(f"Unknown video model: {model}")

        seed = int(time.time()) % 2_147_483_647

        input_payload: dict[str, JSONValue] = {
            "prompt": prompt,
            "duration": duration,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "generate_audio": generate_audio,
            "seed": seed,
        }

        prediction = self._create_prediction(
            api_key=api_key,
            replicate_model=replicate_model,
            input_payload=input_payload,
        )

        output_url = self._wait_for_output(api_key, prediction)
        return self._download_video(api_key, output_url)

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
            timeout=300,
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

    def _download_video(self, api_key: str, url: str) -> bytes:
        download = self._http.get(url, headers=self._headers(api_key), timeout=300)
        if download.status_code != 200:
            detail = download.text[:500] if download.text else "Unknown error"
            raise RuntimeError(f"Replicate video download failed ({download.status_code}): {detail}")
        if not download.content:
            raise RuntimeError("Replicate video download returned empty body")
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
