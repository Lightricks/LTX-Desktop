"""LTX API client implementation."""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any, Literal, cast

from api_types import RetakeMode, VideoCameraMotion
from pydantic import BaseModel, ConfigDict, ValidationError
from services.ltx_api_client.ltx_api_client import LTXAPIClientError, LTXRetakeResult
from services.http_client.http_client import HTTPClient
from services.services_utils import JSONValue

LTXCameraMotion = Literal[
    "dolly_in",
    "dolly_out",
    "dolly_left",
    "dolly_right",
    "jib_up",
    "jib_down",
    "static",
    "focus_shift",
]

_CAMERA_MOTION_TO_LTX: dict[VideoCameraMotion, LTXCameraMotion | None] = {
    "none": None,
    "dolly_in": "dolly_in",
    "dolly_out": "dolly_out",
    "dolly_left": "dolly_left",
    "dolly_right": "dolly_right",
    "jib_up": "jib_up",
    "jib_down": "jib_down",
    "static": "static",
    "focus_shift": "focus_shift",
}


class _RetakeNestedPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")
    video_url: str | None = None


class _RetakeResponsePayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    video_url: str | None = None
    output_video: str | None = None
    result: _RetakeNestedPayload | None = None

    def extract_video_url(self) -> str | None:
        return self.video_url or self.output_video or (self.result.video_url if self.result is not None else None)


class LTXAPIClientImpl:
    def __init__(self, http: HTTPClient, ltx_api_base_url: str) -> None:
        self._http = http
        self._base_url = ltx_api_base_url.rstrip("/")

    def generate_text_to_video(
        self,
        *,
        api_key: str,
        prompt: str,
        model: str,
        resolution: str,
        duration: float,
        fps: float,
        generate_audio: bool,
        camera_motion: VideoCameraMotion = "none",
    ) -> bytes:
        payload: dict[str, JSONValue] = {
            "prompt": prompt,
            "model": model,
            "resolution": resolution,
            "duration": duration,
            "fps": fps,
            "generate_audio": generate_audio,
        }
        mapped_camera_motion = self._map_camera_motion(camera_motion)
        if mapped_camera_motion is not None:
            payload["camera_motion"] = mapped_camera_motion
        response = self._http.post(
            f"{self._base_url}/v1/text-to-video",
            headers=self._json_headers(api_key),
            json_payload=payload,
            timeout=1200,
        )
        return self._extract_video_bytes(response, api_key)

    def generate_image_to_video(
        self,
        *,
        api_key: str,
        prompt: str,
        image_uri: str,
        model: str,
        resolution: str,
        duration: float,
        fps: float,
        generate_audio: bool,
        camera_motion: VideoCameraMotion = "none",
    ) -> bytes:
        payload: dict[str, JSONValue] = {
            "prompt": prompt,
            "image_uri": image_uri,
            "model": model,
            "resolution": resolution,
            "duration": duration,
            "fps": fps,
            "generate_audio": generate_audio,
        }
        mapped_camera_motion = self._map_camera_motion(camera_motion)
        if mapped_camera_motion is not None:
            payload["camera_motion"] = mapped_camera_motion
        response = self._http.post(
            f"{self._base_url}/v1/image-to-video",
            headers=self._json_headers(api_key),
            json_payload=payload,
            timeout=1200,
        )
        return self._extract_video_bytes(response, api_key)

    def generate_audio_to_video(
        self,
        *,
        api_key: str,
        prompt: str,
        audio_uri: str,
        image_uri: str | None,
        model: str,
        resolution: str,
    ) -> bytes:
        payload: dict[str, JSONValue] = {
            "prompt": prompt,
            "audio_uri": audio_uri,
            "model": model,
            "resolution": resolution,
        }
        if image_uri is not None:
            payload["image_uri"] = image_uri
        response = self._http.post(
            f"{self._base_url}/v1/audio-to-video",
            headers=self._json_headers(api_key),
            json_payload=payload,
            timeout=1200,
        )
        return self._extract_video_bytes(response, api_key)

    def retake(
        self,
        *,
        api_key: str,
        video_path: str,
        start_time: float,
        duration: float,
        prompt: str,
        mode: RetakeMode,
    ) -> LTXRetakeResult:
        try:
            storage_uri = self.upload_file(api_key=api_key, file_path=video_path)
        except LTXAPIClientError as exc:
            if exc.stage == "upload_init":
                err_text = self._extract_error_detail(exc.detail)
                raise LTXAPIClientError(exc.status_code, f"Failed to get upload URL: {err_text}") from exc
            if exc.stage == "upload_parse":
                raise LTXAPIClientError(500, "Unexpected upload response format") from exc
            if exc.stage == "upload_put":
                err_text = self._extract_error_detail(exc.detail)
                raise LTXAPIClientError(500, f"Video upload failed: {err_text}") from exc
            raise

        payload: dict[str, JSONValue] = {
            "video_uri": storage_uri,
            "start_time": float(start_time),
            "duration": float(duration),
            "mode": mode,
        }
        if prompt:
            payload["prompt"] = prompt

        response = self._http.post(
            f"{self._base_url}/v1/retake",
            headers=self._json_headers(api_key),
            json_payload=payload,
            timeout=600,
        )

        rid = self._fmt_request_id(response)
        if response.status_code == 200:
            content_type = str(response.headers.get("Content-Type", "")).lower()
            if "video" in content_type or "octet-stream" in content_type:
                return LTXRetakeResult(video_bytes=response.content, result_payload=None)

            try:
                payload_obj = response.json()
            except json.JSONDecodeError as exc:
                raise LTXAPIClientError(500, f"Unexpected response format: {response.text[:200]}{rid}") from exc

            try:
                parsed_payload = _RetakeResponsePayload.model_validate(payload_obj)
            except ValidationError as exc:
                raise LTXAPIClientError(500, f"Unexpected response format{rid}") from exc

            video_url = parsed_payload.extract_video_url()
            if video_url:
                dl_resp = self._http.get(video_url, timeout=120)
                if dl_resp.status_code == 200:
                    return LTXRetakeResult(video_bytes=dl_resp.content, result_payload=None)
                raise LTXAPIClientError(500, f"Failed to download retake video: {dl_resp.status_code}{rid}")

            response_payload = parsed_payload.model_dump(mode="python")
            return LTXRetakeResult(video_bytes=None, result_payload=response_payload)

        if response.status_code == 422:
            raise LTXAPIClientError(422, f"Content rejected by safety filters{rid}")

        error_text = response.text[:500] if response.text else "Unknown error"
        raise LTXAPIClientError(response.status_code, f"Retake API error: {error_text}{rid}")

    def upload_file(self, *, file_path: str, api_key: str) -> str:
        upload_resp = self._http.post(
            f"{self._base_url}/v1/upload",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        if upload_resp.status_code != 200:
            err = upload_resp.text[:500]
            rid = self._fmt_request_id(upload_resp)
            raise LTXAPIClientError(
                upload_resp.status_code,
                f"LTX upload init failed ({upload_resp.status_code}): {err}{rid}",
                stage="upload_init",
            )

        try:
            payload = cast(dict[str, Any], upload_resp.json())
            upload_url = str(payload["upload_url"])
            storage_uri = str(payload["storage_uri"])
            required_headers = cast(dict[str, str], payload.get("required_headers", {}))
        except Exception as exc:
            rid = self._fmt_request_id(upload_resp)
            raise LTXAPIClientError(500, f"Unexpected LTX upload response format{rid}", stage="upload_parse") from exc

        path_obj = Path(file_path)
        mime = mimetypes.guess_type(path_obj.name)[0] or "application/octet-stream"
        with open(path_obj, "rb") as media_file:
            put_resp = self._http.put(
                upload_url,
                data=media_file,
                headers={"Content-Type": mime, **required_headers},
                timeout=300,
            )
        if put_resp.status_code not in (200, 201):
            err = put_resp.text[:500]
            rid = self._fmt_request_id(upload_resp)
            raise LTXAPIClientError(500, f"LTX upload failed ({put_resp.status_code}): {err}{rid}", stage="upload_put")

        return storage_uri

    def _extract_video_bytes(self, response: Any, api_key: str) -> bytes:
        rid = self._fmt_request_id(response)
        if response.status_code != 200:
            err = response.text[:500] if response.text else "Unknown error"
            raise RuntimeError(f"LTX API generation failed ({response.status_code}): {err}{rid}")

        content_type = str(response.headers.get("Content-Type", "")).lower()
        if "video" in content_type or "octet-stream" in content_type:
            if not response.content:
                raise RuntimeError(f"LTX API returned empty video body{rid}")
            return response.content

        try:
            payload = cast(dict[str, Any], response.json())
        except Exception as exc:
            raise RuntimeError(f"Unexpected LTX API response format{rid}") from exc

        video_url = self._extract_video_url(payload)
        if video_url is not None:
            dl_resp = self._http.get(
                video_url,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=120,
            )
            if dl_resp.status_code != 200:
                raise RuntimeError(f"Failed to download generated video ({dl_resp.status_code}){rid}")
            if not dl_resp.content:
                raise RuntimeError(f"Downloaded generated video is empty{rid}")
            return dl_resp.content

        error_text = payload.get("error") or payload.get("message") or payload.get("detail")
        if isinstance(error_text, str) and error_text:
            raise RuntimeError(f"LTX API returned an error payload: {error_text}{rid}")
        raise RuntimeError(f"LTX API response did not include a video payload{rid}")

    @staticmethod
    def _request_id(response: Any) -> str | None:
        rid = response.headers.get("x-request-id")
        return str(rid) if rid else None

    @staticmethod
    def _fmt_request_id(response: Any) -> str:
        rid = response.headers.get("x-request-id")
        return f" [request_id={rid}]" if rid else ""

    @staticmethod
    def _extract_error_detail(detail: str) -> str:
        if ":" not in detail:
            return detail
        return detail.split(":", 1)[1].strip()

    @staticmethod
    def _extract_video_url(payload: dict[str, Any]) -> str | None:
        direct_keys = ("video_url", "output_video", "output_video_url", "output_url", "url")
        for key in direct_keys:
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value

        nested = payload.get("result")
        if isinstance(nested, dict):
            nested_payload = cast(dict[str, Any], nested)
            for key in direct_keys:
                value = nested_payload.get(key)
                if isinstance(value, str) and value:
                    return value
        return None

    @staticmethod
    def _json_headers(api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _map_camera_motion(camera_motion: VideoCameraMotion) -> LTXCameraMotion | None:
        return _CAMERA_MOTION_TO_LTX[camera_motion]
