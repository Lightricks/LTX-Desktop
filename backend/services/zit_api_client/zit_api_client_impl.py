"""Z-Image Turbo API client implementation for FAL endpoints."""

from __future__ import annotations

import base64
from typing import Any, cast

from services.http_client.http_client import HTTPClient
from services.services_utils import JSONValue

FAL_API_BASE_URL = "https://fal.run"
FAL_TEXT_TO_IMAGE_ENDPOINT = "/fal-ai/z-image/turbo"
FAL_IMAGE_TO_IMAGE_ENDPOINT = "/fal-ai/z-image/turbo/image-to-image"

DEFAULT_OUTPUT_FORMAT = "png"
DEFAULT_ACCELERATION = "regular"
DEFAULT_ENABLE_SAFETY_CHECKER = True


class ZitAPIClientImpl:
    def __init__(self, http: HTTPClient, *, fal_api_base_url: str = FAL_API_BASE_URL) -> None:
        self._http = http
        self._base_url = fal_api_base_url.rstrip("/")

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
        payload: dict[str, JSONValue] = {
            "prompt": prompt,
            "image_size": {"width": width, "height": height},
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "num_images": 1,
            "output_format": DEFAULT_OUTPUT_FORMAT,
            "acceleration": DEFAULT_ACCELERATION,
            "enable_safety_checker": DEFAULT_ENABLE_SAFETY_CHECKER,
        }
        return self._submit_and_download(
            endpoint=FAL_TEXT_TO_IMAGE_ENDPOINT,
            api_key=api_key,
            payload=payload,
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
        return self._submit_and_download(
            endpoint=FAL_IMAGE_TO_IMAGE_ENDPOINT,
            api_key=api_key,
            payload=payload,
        )

    def _submit_and_download(
        self,
        *,
        endpoint: str,
        api_key: str,
        payload: dict[str, JSONValue],
    ) -> bytes:
        response = self._http.post(
            f"{self._base_url}{endpoint}",
            headers=self._json_headers(api_key),
            json_payload=payload,
            timeout=180,
        )
        if response.status_code != 200:
            detail = response.text[:500] if response.text else "Unknown error"
            raise RuntimeError(f"FAL submit failed ({response.status_code}): {detail}")

        response_payload = self._json_object(response.json(), context="submit")
        image_url = self._extract_image_url(response_payload)

        download = self._http.get(image_url, timeout=120)
        if download.status_code != 200:
            detail = download.text[:500] if download.text else "Unknown error"
            raise RuntimeError(f"FAL image download failed ({download.status_code}): {detail}")
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
