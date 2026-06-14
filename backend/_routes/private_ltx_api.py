"""Private LTX-compatible API used by self-hosted RunPod backends."""

from __future__ import annotations

import mimetypes
import os
import uuid
from pathlib import Path
from typing import Literal, cast

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict

from _routes._errors import HTTPError
from api_types import (
    GenerateVideoRequest,
    LTXLocalModelId,
    LTXVideoGenPipeline,
    LTXVideoGenDuration,
    LTXVideoGenFps,
    LTXVideoGenResolution,
    ModelCheckpointID,
    RetakeRequest,
    RetakeVideoResponse,
    VideoCameraMotion,
)
from app_handler import AppHandler
from runtime_config.model_download_specs import get_ltx_model_id_for_pipeline, get_ltx_model_spec, is_cp_downloaded
from state import get_state_service

router = APIRouter(prefix="/v1", tags=["private-ltx-api"])


class PrivateUploadInitResponse(BaseModel):
    upload_url: str
    storage_uri: str
    required_headers: dict[str, str] = {}


class PrivateTextToVideoRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt: str
    model: str = "fast"
    resolution: str = "1080p"
    duration: float = 5
    fps: float = 24
    generate_audio: bool = False
    camera_motion: str = "none"
    aspect_ratio: str = "16:9"
    enhance_prompt: bool = False


class PrivateImageToVideoRequest(PrivateTextToVideoRequest):
    image_uri: str


class PrivateAudioToVideoRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt: str
    audio_uri: str
    image_uri: str | None = None
    model: str = "fast"
    resolution: str = "1080p"
    duration: float = 5
    fps: float = 24
    aspect_ratio: str = "16:9"
    enhance_prompt: bool = False


class PrivateRetakeRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    video_uri: str
    start_time: float
    duration: float
    prompt: str = ""
    mode: Literal["replace_audio_and_video", "replace_video", "replace_audio"] = "replace_audio_and_video"


def _uploads_dir(handler: AppHandler) -> Path:
    path = handler.config.app_data_dir / "private-api-uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _storage_uri(upload_id: str) -> str:
    return f"private-upload://{upload_id}"


def _public_base_url(request: Request) -> str:
    configured = os.environ.get("RUNPOD_PUBLIC_BASE_URL", "").strip().rstrip("/")
    if configured:
        return configured

    proto = request.headers.get("x-forwarded-proto")
    host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    if proto and host:
        return f"{proto}://{host}".rstrip("/")

    return str(request.base_url).rstrip("/")


def _public_upload_url(request: Request, upload_id: str) -> str:
    return f"{_public_base_url(request)}/v1/upload/{upload_id}"


def _upload_id_from_uri(storage_uri: str) -> str:
    prefix = "private-upload://"
    if not storage_uri.startswith(prefix):
        raise HTTPError(400, "INVALID_PRIVATE_STORAGE_URI")
    upload_id = storage_uri[len(prefix):]
    if not upload_id or "/" in upload_id or "\\" in upload_id:
        raise HTTPError(400, "INVALID_PRIVATE_STORAGE_URI")
    return upload_id


def _resolve_uploaded_path(handler: AppHandler, storage_uri: str) -> Path:
    upload_id = _upload_id_from_uri(storage_uri)
    path = _uploads_dir(handler) / upload_id
    if not path.exists():
        raise HTTPError(400, "PRIVATE_UPLOAD_NOT_FOUND")
    return path


def _extension_for_content_type(content_type: str) -> str:
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type == "video/mp4":
        return ".mp4"
    if media_type in ("image/jpeg", "image/jpg"):
        return ".jpg"
    if media_type == "image/png":
        return ".png"
    if media_type in ("audio/wav", "audio/x-wav"):
        return ".wav"
    if media_type == "audio/mpeg":
        return ".mp3"
    return mimetypes.guess_extension(media_type) or ".bin"


def _safe_int(value: float, allowed: set[int], default: int) -> int:
    candidate = int(round(value))
    return candidate if candidate in allowed else default


def _local_model(model: str) -> LTXVideoGenPipeline:
    if model in ("fast", "ltx-2-3-fast", "ltx-2.3-22b-distilled-1.1"):
        return "fast"
    if model in ("fast_legacy", "ltx-2.3-22b-distilled"):
        return "fast_legacy"
    if model in ("pro", "ltx-2-3-pro"):
        raise HTTPError(422, "RUNPOD_PRIVATE_API_PRO_MODEL_NOT_AVAILABLE")
    raise HTTPError(422, "RUNPOD_PRIVATE_API_MODEL_NOT_AVAILABLE")


def _local_model_id(pipeline: LTXVideoGenPipeline) -> LTXLocalModelId:
    model_id = get_ltx_model_id_for_pipeline(pipeline)
    if model_id is None:
        raise HTTPError(422, "RUNPOD_PRIVATE_API_MODEL_NOT_AVAILABLE")
    return model_id


def _required_private_generation_cp_ids(model_id: LTXLocalModelId) -> set[ModelCheckpointID]:
    spec = get_ltx_model_spec(model_id)
    return {spec.model_cp, spec.upscale_cp, spec.text_encoder_cp}


def _ensure_private_model_downloaded(handler: AppHandler, model_id: LTXLocalModelId) -> None:
    missing: set[ModelCheckpointID] = {
        cp_id
        for cp_id in _required_private_generation_cp_ids(model_id)
        if not is_cp_downloaded(handler.config.default_models_dir, cp_id)
    }
    if not missing:
        return
    try:
        handler.downloads.download_missing_sync(missing)
    except HTTPError:
        raise
    except Exception as exc:
        raise HTTPError(500, f"RUNPOD_PRIVATE_MODEL_DOWNLOAD_FAILED: {exc}") from exc


def _duration(value: float) -> LTXVideoGenDuration:
    return cast(LTXVideoGenDuration, _safe_int(value, {5, 6, 8, 10, 12, 14, 16, 18, 20}, 5))


def _fps(value: float) -> LTXVideoGenFps:
    return cast(LTXVideoGenFps, _safe_int(value, {24, 25, 48, 50}, 24))


def _camera_motion(value: str) -> VideoCameraMotion:
    allowed: set[str] = {
        "none",
        "dolly_in",
        "dolly_out",
        "dolly_left",
        "dolly_right",
        "jib_up",
        "jib_down",
        "static",
        "focus_shift",
    }
    if value not in allowed:
        raise HTTPError(422, "RUNPOD_PRIVATE_API_CAMERA_MOTION_NOT_AVAILABLE")
    return cast(VideoCameraMotion, value)


def _local_resolution(resolution: str) -> LTXVideoGenResolution:
    if resolution in ("540p", "720p", "1080p"):
        return cast(LTXVideoGenResolution, resolution)
    pixel_map = {
        "960x544": "540p",
        "544x960": "540p",
        "1280x704": "720p",
        "704x1280": "720p",
        "1920x1088": "1080p",
        "1088x1920": "1080p",
        "1920x1080": "1080p",
        "1080x1920": "1080p",
    }
    mapped = pixel_map.get(resolution)
    if mapped is None:
        raise HTTPError(422, "RUNPOD_PRIVATE_API_RESOLUTION_NOT_AVAILABLE")
    return cast(LTXVideoGenResolution, mapped)


def _aspect_ratio(value: str) -> Literal["16:9", "9:16"]:
    if value in ("16:9", "9:16"):
        return value  # type: ignore[return-value]
    raise HTTPError(422, "RUNPOD_PRIVATE_API_ASPECT_RATIO_NOT_AVAILABLE")


def _video_response(video_path: str) -> Response:
    data = Path(video_path).read_bytes()
    return Response(content=data, media_type="video/mp4")


def _generate_video(handler: AppHandler, req: GenerateVideoRequest) -> Response:
    result = handler.video_generation.generate(req)
    if result.status == "cancelled":
        raise HTTPError(499, "Generation was cancelled")
    return _video_response(result.video_path)


@router.post("/upload", response_model=PrivateUploadInitResponse)
def route_private_upload_init(
    request: Request,
    handler: AppHandler = Depends(get_state_service),
) -> PrivateUploadInitResponse:
    upload_id = f"{uuid.uuid4().hex}.bin"
    upload_url = _public_upload_url(request, upload_id)
    required_headers: dict[str, str] = {}
    auth_header = request.headers.get("authorization")
    if auth_header:
        required_headers["Authorization"] = auth_header
    return PrivateUploadInitResponse(
        upload_url=upload_url,
        storage_uri=_storage_uri(upload_id),
        required_headers=required_headers,
    )


@router.put("/upload/{upload_id}", name="route_private_upload_put")
async def route_private_upload_put(
    upload_id: str,
    request: Request,
    handler: AppHandler = Depends(get_state_service),
) -> dict[str, str]:
    if not upload_id.endswith(".bin") or "/" in upload_id or "\\" in upload_id:
        raise HTTPError(400, "INVALID_PRIVATE_UPLOAD_ID")

    content_type = request.headers.get("content-type", "application/octet-stream")
    final_id = f"{upload_id[:-4]}{_extension_for_content_type(content_type)}"
    body = await request.body()
    if not body:
        raise HTTPError(400, "EMPTY_PRIVATE_UPLOAD")

    path = _uploads_dir(handler) / final_id
    path.write_bytes(body)
    return {"storage_uri": _storage_uri(final_id)}


@router.post("/text-to-video")
def route_private_text_to_video(
    payload: PrivateTextToVideoRequest,
    handler: AppHandler = Depends(get_state_service),
) -> Response:
    model = _local_model(payload.model)
    _ensure_private_model_downloaded(handler, _local_model_id(model))
    req = GenerateVideoRequest(
        prompt=payload.prompt,
        model=model,
        resolution=_local_resolution(payload.resolution),
        duration=_duration(payload.duration),
        fps=_fps(payload.fps),
        audio=payload.generate_audio,
        cameraMotion=_camera_motion(payload.camera_motion),
        aspectRatio=_aspect_ratio(payload.aspect_ratio),
        enhancePrompt=payload.enhance_prompt,
    )
    return _generate_video(handler, req)


@router.post("/image-to-video")
def route_private_image_to_video(
    payload: PrivateImageToVideoRequest,
    handler: AppHandler = Depends(get_state_service),
) -> Response:
    model = _local_model(payload.model)
    _ensure_private_model_downloaded(handler, _local_model_id(model))
    image_path = _resolve_uploaded_path(handler, payload.image_uri)
    req = GenerateVideoRequest(
        prompt=payload.prompt,
        model=model,
        resolution=_local_resolution(payload.resolution),
        duration=_duration(payload.duration),
        fps=_fps(payload.fps),
        audio=payload.generate_audio,
        imagePath=str(image_path),
        cameraMotion=_camera_motion(payload.camera_motion),
        aspectRatio=_aspect_ratio(payload.aspect_ratio),
        enhancePrompt=payload.enhance_prompt,
    )
    return _generate_video(handler, req)


@router.post("/audio-to-video")
def route_private_audio_to_video(
    payload: PrivateAudioToVideoRequest,
    handler: AppHandler = Depends(get_state_service),
) -> Response:
    model = _local_model(payload.model)
    _ensure_private_model_downloaded(handler, _local_model_id(model))
    audio_path = _resolve_uploaded_path(handler, payload.audio_uri)
    image_path = _resolve_uploaded_path(handler, payload.image_uri) if payload.image_uri else None
    req = GenerateVideoRequest(
        prompt=payload.prompt,
        model=model,
        resolution=_local_resolution(payload.resolution),
        duration=_duration(payload.duration),
        fps=_fps(payload.fps),
        audioPath=str(audio_path),
        imagePath=str(image_path) if image_path else None,
        aspectRatio=_aspect_ratio(payload.aspect_ratio),
        enhancePrompt=payload.enhance_prompt,
    )
    return _generate_video(handler, req)


@router.post("/retake")
def route_private_retake(
    payload: PrivateRetakeRequest,
    handler: AppHandler = Depends(get_state_service),
) -> Response:
    video_path = _resolve_uploaded_path(handler, payload.video_uri)
    result = handler.retake.run(
        RetakeRequest(
            video_path=str(video_path),
            start_time=payload.start_time,
            duration=payload.duration,
            prompt=payload.prompt,
            mode=payload.mode,
        )
    )
    if result.status == "cancelled":
        raise HTTPError(499, "Retake was cancelled")
    if isinstance(result, RetakeVideoResponse):
        return _video_response(result.video_path)
    raise HTTPError(500, "Private retake returned no video output")
