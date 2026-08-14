"""LTX API client implementation."""

from __future__ import annotations

import json
import logging
import mimetypes
import time
from pathlib import Path
from typing import Any, Literal, cast

from api_types import ExtendMode, RetakeMode, VideoCameraMotion
from pydantic import BaseModel, ConfigDict, ValidationError
from services.ltx_api_client.ltx_api_client import LTXAPIClientError, LTXRetakeResult
from services.http_client.http_client import HTTPClient, HttpResponseLike, HttpTransportError
from services.services_utils import JSONValue

logger = logging.getLogger(__name__)

# Payload fields that are large and/or sensitive (signed URLs, base64 data, the user's
# prompt) — logged as "<set>"/"<unset>" instead of their value so request logs stay
# readable and don't leak private content.
_REDACTED_PAYLOAD_KEYS = frozenset(
    {"video_uri", "image_uri", "audio_uri", "last_frame_uri", "prompt"}
)

# A transient transport blip on a single poll/download shouldn't discard a minutes-long
# async job; retry the individual GET a few times before surfacing the failure.
_GET_RETRIES = 3
# Non-terminal async job statuses. Anything else that isn't "completed" (failed, error,
# canceled, rejected, or an unknown string) is treated as terminal failure and surfaced,
# rather than polled until the timeout hides the real cause behind a 504.
_IN_PROGRESS_STATUSES = frozenset(
    {"queued", "pending", "processing", "running", "in_progress", "starting", "started"}
)


def _loggable_payload(payload: dict[str, JSONValue]) -> dict[str, object]:
    return {
        key: ("<set>" if value else "<unset>") if key in _REDACTED_PAYLOAD_KEYS else value
        for key, value in payload.items()
    }

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


class _LTXErrorDetailPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: str | None = None
    message: str | None = None


class _LTXErrorPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: str | None = None
    error: _LTXErrorDetailPayload | None = None
    message: str | None = None
    detail: str | None = None


class _AsyncSubmitPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str | None = None


class _AsyncJobErrorPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")
    message: str | None = None


class _AsyncJobStatusPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str
    result: dict[str, Any] | None = None
    error: _AsyncJobErrorPayload | None = None


class LTXAPIClientImpl:
    def __init__(
        self,
        http: HTTPClient,
        ltx_api_base_url: str,
        *,
        poll_interval_s: float = 3.0,
        async_max_wait_s: float = 600.0,
    ) -> None:
        self._http = http
        self._base_url = ltx_api_base_url.rstrip("/")
        # Async (v2) polling cadence/ceiling; small values injected in tests.
        self._poll_interval_s = poll_interval_s
        self._async_max_wait_s = async_max_wait_s

    def _post_json(
        self,
        endpoint: str,
        *,
        api_key: str,
        payload: dict[str, JSONValue],
        timeout: int,
    ) -> HttpResponseLike:
        """POST JSON to an LTX endpoint, logging the request (safe fields), the response
        (status + request id + elapsed), and transport failures."""
        logger.info("LTXV → POST %s %s (timeout=%ss)", endpoint, _loggable_payload(payload), timeout)
        started = time.monotonic()
        try:
            response = self._http.post(
                f"{self._base_url}{endpoint}",
                headers=self._json_headers(api_key),
                json_payload=payload,
                timeout=timeout,
            )
        except HttpTransportError:
            logger.warning("LTXV ✗ POST %s failed (transport) after %.1fs", endpoint, time.monotonic() - started)
            raise
        logger.info(
            "LTXV ← POST %s %s in %.1fs%s",
            endpoint,
            response.status_code,
            time.monotonic() - started,
            self._fmt_request_id(response),
        )
        return response

    def generate_text_to_video(
        self,
        *,
        api_key: str,
        prompt: str,
        model: str,
        resolution: str,
        duration: float | None,
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
        response = self._post_json("/v1/text-to-video", api_key=api_key, payload=payload, timeout=1200)
        return self._extract_video_bytes(response, api_key)

    def generate_image_to_video(
        self,
        *,
        api_key: str,
        prompt: str,
        image_uri: str,
        model: str,
        resolution: str,
        duration: float | None,
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
        response = self._post_json("/v1/image-to-video", api_key=api_key, payload=payload, timeout=1200)
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
        response = self._post_json("/v1/audio-to-video", api_key=api_key, payload=payload, timeout=1200)
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
        model: str,
    ) -> LTXRetakeResult:
        return self._run_video_edit(
            api_key=api_key,
            video_path=video_path,
            endpoint="/v1/retake",
            label="Retake",
            edit_payload={
                "start_time": float(start_time),
                "duration": float(duration),
                "mode": mode,
                "model": model,
            },
            prompt=prompt,
        )

    def extend(
        self,
        *,
        api_key: str,
        video_path: str,
        duration: float,
        prompt: str,
        mode: ExtendMode,
        model: str,
    ) -> LTXRetakeResult:
        # Extend uses the async v2 endpoint (submit → poll → download): a 12s Pro extend
        # takes minutes, which the sync v1 endpoint can't hold open without the connection
        # being reset mid-response. Mirrors ltx-studio's extend path.
        return self._run_async_video_edit(
            api_key=api_key,
            video_path=video_path,
            endpoint="/v2/extend",
            label="Extend",
            edit_payload={
                "duration": float(duration),
                "mode": mode,
                "model": model,
            },
            prompt=prompt,
        )

    def _upload_source_video(self, *, api_key: str, video_path: str) -> str:
        """Upload the source clip, mapping upload-stage failures to friendly messages."""
        try:
            return self.upload_file(api_key=api_key, file_path=video_path)
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

    def _run_video_edit(
        self,
        *,
        api_key: str,
        video_path: str,
        endpoint: str,
        label: str,
        edit_payload: dict[str, JSONValue],
        prompt: str,
    ) -> LTXRetakeResult:
        """Synchronous upload → POST → parse flow for the v1 /retake endpoint.

        Gets back either raw video bytes, a JSON payload with a downloadable URL, or a 422
        safety reject.
        """
        storage_uri = self._upload_source_video(api_key=api_key, video_path=video_path)

        payload: dict[str, JSONValue] = {"video_uri": storage_uri, **edit_payload}
        if prompt:
            payload["prompt"] = prompt

        response = self._post_json(endpoint, api_key=api_key, payload=payload, timeout=600)

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
                raise LTXAPIClientError(500, f"Failed to download {label.lower()} video: {dl_resp.status_code}{rid}")

            response_payload = parsed_payload.model_dump(mode="python")
            return LTXRetakeResult(video_bytes=None, result_payload=response_payload)

        if response.status_code == 422:
            raise LTXAPIClientError(422, f"Content rejected by safety filters{rid}")

        error_text = response.text[:500] if response.text else "Unknown error"
        raise LTXAPIClientError(response.status_code, f"{label} API error: {error_text}{rid}")

    def _run_async_video_edit(
        self,
        *,
        api_key: str,
        video_path: str,
        endpoint: str,
        label: str,
        edit_payload: dict[str, JSONValue],
        prompt: str,
    ) -> LTXRetakeResult:
        """Async (v2) flow: upload → submit job → poll until done → download the result.

        The v2 endpoints respond immediately with a job id, so no connection is held open
        for the whole (minutes-long) generation. ``endpoint`` is the submit path
        (e.g. ``/v2/extend``); polling is ``GET {endpoint}/{id}``.
        """
        try:
            storage_uri = self._upload_source_video(api_key=api_key, video_path=video_path)

            payload: dict[str, JSONValue] = {"video_uri": storage_uri, **edit_payload}
            if prompt:
                payload["prompt"] = prompt

            submit = self._post_json(endpoint, api_key=api_key, payload=payload, timeout=120)
            rid = self._fmt_request_id(submit)
            if submit.status_code == 422:
                raise LTXAPIClientError(422, f"Content rejected by safety filters{rid}")
            if submit.status_code not in (200, 202):
                error_text = submit.text[:500] if submit.text else "Unknown error"
                raise LTXAPIClientError(submit.status_code, f"{label} API error: {error_text}{rid}")

            try:
                job = _AsyncSubmitPayload.model_validate(submit.json())
            except (json.JSONDecodeError, ValidationError) as exc:
                raise LTXAPIClientError(500, f"Unexpected {label.lower()} submit response{rid}") from exc
            if not job.id:
                raise LTXAPIClientError(500, f"{label} API returned no job id{rid}")
            logger.info("LTXV %s job accepted: id=%s%s", label.lower(), job.id, rid)

            video_url = self._poll_job(api_key=api_key, endpoint=endpoint, job_id=job.id, label=label)
            logger.info("LTXV %s downloading result for job %s", label.lower(), job.id)
            # video_url comes straight from the (trusted) LTX API; we fetch it whole with no
            # scheme/host allowlist or size cap. Acceptable while we fully trust the API —
            # revisit (allowlist + streamed cap) if that trust ever weakens.
            dl_resp = self._get_with_retries(video_url, headers=None, timeout=300, label=label)
            if dl_resp.status_code != 200:
                raise LTXAPIClientError(500, f"Failed to download {label.lower()} video: {dl_resp.status_code}")
            logger.info("LTXV %s complete: %d bytes (job %s)", label.lower(), len(dl_resp.content), job.id)
            return LTXRetakeResult(video_bytes=dl_resp.content, result_payload=None)
        except HttpTransportError as exc:
            # Connection failed/reset/timed out somewhere in the flow — surface as a clean
            # retryable error instead of a raw 500.
            raise LTXAPIClientError(504, f"{label} request to the LTX API failed; please retry.") from exc

    def _get_with_retries(
        self, url: str, *, headers: dict[str, str] | None, timeout: int, label: str
    ) -> HttpResponseLike:
        """GET ``url``, retrying transport failures a few times so a single blip doesn't
        discard the whole async job. Re-raises HttpTransportError once retries are spent."""
        for attempt in range(1, _GET_RETRIES + 1):
            try:
                return self._http.get(url, headers=headers, timeout=timeout)
            except HttpTransportError:
                if attempt == _GET_RETRIES:
                    raise
                logger.warning(
                    "LTXV %s GET transport error (attempt %d/%d); retrying", label.lower(), attempt, _GET_RETRIES
                )
                time.sleep(self._poll_interval_s)
        raise AssertionError("unreachable")  # the loop either returns or re-raises

    def _poll_job(self, *, api_key: str, endpoint: str, job_id: str, label: str) -> str:
        """Poll ``GET {endpoint}/{job_id}`` until ``completed`` and return the video_url."""
        started = time.monotonic()
        deadline = started + self._async_max_wait_s
        poll_url = f"{self._base_url}{endpoint}/{job_id}"
        polls = 0
        while time.monotonic() < deadline:
            resp = self._get_with_retries(
                poll_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=60, label=label
            )
            rid = self._fmt_request_id(resp)
            if resp.status_code != 200:
                error_text = resp.text[:500] if resp.text else "Unknown error"
                raise LTXAPIClientError(resp.status_code, f"{label} status check failed: {error_text}{rid}")
            try:
                status = _AsyncJobStatusPayload.model_validate(resp.json())
            except (json.JSONDecodeError, ValidationError) as exc:
                raise LTXAPIClientError(500, f"Unexpected {label.lower()} status response{rid}") from exc

            polls += 1
            elapsed = time.monotonic() - started
            logger.debug("LTXV %s job %s status=%s (poll %d, %.0fs)", label.lower(), job_id, status.status, polls, elapsed)

            if status.status == "completed":
                video_url = (status.result or {}).get("video_url")
                if not isinstance(video_url, str) or not video_url:
                    raise LTXAPIClientError(500, f"{label} job completed without a video_url{rid}")
                logger.info("LTXV %s job %s completed after %.0fs (%d polls)", label.lower(), job_id, elapsed, polls)
                return video_url
            if status.status not in _IN_PROGRESS_STATUSES:
                # failed / error / canceled / rejected / unknown — surface it instead of
                # polling until the timeout masks the cause behind a 504.
                detail = (
                    status.error.message
                    if status.error and status.error.message
                    else f"{label} job ended with status '{status.status}'"
                )
                logger.warning(
                    "LTXV %s job %s ended status=%s after %.0fs: %s", label.lower(), job_id, status.status, elapsed, detail
                )
                raise LTXAPIClientError(500, detail)

            time.sleep(self._poll_interval_s)

        raise LTXAPIClientError(504, f"{label} job timed out after {int(self._async_max_wait_s)}s")

    def upload_file(self, *, file_path: str, api_key: str) -> str:
        logger.info("LTXV → POST /v1/upload (requesting signed upload URL)")
        started = time.monotonic()
        # 60s (not 30): the signed-URL request has been observed to take >30s under API
        # load, and a tight ceiling here fails the whole generation before it starts.
        upload_resp = self._http.post(
            f"{self._base_url}/v1/upload",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
        )
        logger.info(
            "LTXV ← POST /v1/upload %s in %.1fs%s",
            upload_resp.status_code,
            time.monotonic() - started,
            self._fmt_request_id(upload_resp),
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
        put_started = time.monotonic()
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

        logger.info("LTXV source video uploaded in %.1fs", time.monotonic() - put_started)
        return storage_uri

    def _extract_video_bytes(self, response: Any, api_key: str) -> bytes:
        rid = self._fmt_request_id(response)
        if response.status_code != 200:
            provider_error_type, provider_message, err = self._extract_generation_error(response)
            raise LTXAPIClientError(
                response.status_code,
                f"LTX API generation failed ({response.status_code}): {err}{rid}",
                stage="generation",
                provider_error_type=provider_error_type,
                provider_message=provider_message,
                request_id=self._request_id(response),
            )

        content_type = str(response.headers.get("Content-Type", "")).lower()
        if "video" in content_type or "octet-stream" in content_type:
            if not response.content:
                raise LTXAPIClientError(500, f"LTX API returned empty video body{rid}", stage="generation")
            return response.content

        try:
            payload = cast(dict[str, Any], response.json())
        except Exception as exc:
            raise LTXAPIClientError(500, f"Unexpected LTX API response format{rid}", stage="generation") from exc

        video_url = self._extract_video_url(payload)
        if video_url is not None:
            dl_resp = self._http.get(
                video_url,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=120,
            )
            if dl_resp.status_code != 200:
                raise LTXAPIClientError(500, f"Failed to download generated video ({dl_resp.status_code}){rid}", stage="generation")
            if not dl_resp.content:
                raise LTXAPIClientError(500, f"Downloaded generated video is empty{rid}", stage="generation")
            return dl_resp.content

        error_text = payload.get("error") or payload.get("message") or payload.get("detail")
        if isinstance(error_text, str) and error_text:
            raise LTXAPIClientError(500, f"LTX API returned an error payload: {error_text}{rid}", stage="generation")
        raise LTXAPIClientError(500, f"LTX API response did not include a video payload{rid}", stage="generation")

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
    def _extract_generation_error(response: Any) -> tuple[str | None, str | None, str]:
        if response.text:
            error_text = response.text[:500]
        else:
            error_text = "Unknown error"

        try:
            payload = _LTXErrorPayload.model_validate(response.json())
        except (ValidationError, TypeError, ValueError):
            return None, None, error_text

        provider_error_type = payload.error.type if payload.error is not None else None
        provider_message = (
            payload.error.message
            if payload.error is not None and payload.error.message
            else payload.message or payload.detail
        )
        if not response.text:
            serialized_payload = payload.model_dump(mode="json", exclude_none=True)
            if serialized_payload:
                error_text = json.dumps(serialized_payload, separators=(",", ":"))[:500]
            elif provider_message:
                error_text = provider_message
        return provider_error_type, provider_message, error_text

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
