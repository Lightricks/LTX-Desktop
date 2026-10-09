from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Annotated, Any, BinaryIO

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response
from starlette.types import Message, Receive

from _routes._errors import HTTPError, build_http_error_response
from api_types import (
    Asset,
    AssetListQuery,
    AssetListResponse,
    IngestAssetRequest,
    MediaKind,
    StatusResponse,
    TrimMediaRequest,
)
from app_handler import AppHandler
from remote.deps import RemoteContext, get_remote_context
from remote.dto import (
    RemoteAsset,
    RemoteAssetListResponse,
    to_remote_list_item,
)
from services.media_probe import (
    ALLOWED_AUDIO_SUFFIXES,
    ALLOWED_IMAGE_SUFFIXES,
    ALLOWED_VIDEO_SUFFIXES,
    MAX_AUDIO_BYTES,
    MAX_IMAGE_BYTES,
    MAX_VIDEO_BYTES,
)
from state import get_state_service

router = APIRouter(prefix="/api")

_UPLOAD_READ_CHUNK_BYTES = 1024 * 1024
_UPLOAD_TEMP_PREFIX = "ltx-remote-upload-"
# Multipart bodies are larger than the file (boundaries, disposition, MIME).
# This slack is only for the HTTP Content-Length early abort so a file of
# the per-kind max is not rejected. UploadFile.size and the copied-byte cap
# still use that max with no slack.
_MULTIPART_OVERHEAD_BYTES = 64 * 1024
# Remote-only ASGI body cap: largest permitted file plus multipart framing.
# Enforced on actual receive bytes before Starlette parses/spools multipart.
MAX_REMOTE_UPLOAD_BODY_BYTES = (
    max(MAX_IMAGE_BYTES, MAX_AUDIO_BYTES, MAX_VIDEO_BYTES)
    + _MULTIPART_OVERHEAD_BYTES
)


class _UploadBodyTooLarge(StarletteHTTPException):
    """Raised when ASGI receive bytes exceed MAX_REMOTE_UPLOAD_BODY_BYTES.

    Subclasses Starlette HTTPException so FastAPI's body-parsing wrapper
    (which converts generic Exceptions to "There was an error parsing the
    body") re-raises it unchanged to the route handler.
    """

    def __init__(self) -> None:
        super().__init__(status_code=400, detail="FILE_TOO_LARGE")


def _declared_upload_body_too_large(content_length: str | None) -> bool:
    if content_length is None:
        return False
    try:
        declared = int(content_length.strip())
    except ValueError:
        return False
    return declared > MAX_REMOTE_UPLOAD_BODY_BYTES


def _upload_too_large_response() -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content=build_http_error_response(400, "FILE_TOO_LARGE").model_dump(),
    )


def _bounded_upload_receive(base_receive: Receive) -> Receive:
    """Wrap ASGI receive to bound total body bytes and per-event chunks.

    Counts actual receive bytes so chunked/missing/lying Content-Length cannot
    spool unbounded temp disk in Starlette's multipart parser. Large ASGI
    events are split into _UPLOAD_READ_CHUNK_BYTES pieces; an event that would
    push the total over the cap raises before any of it is forwarded.
    """
    total = 0
    pending = b""
    pending_more_body = False

    async def bounded_receive() -> Message:
        nonlocal total, pending, pending_more_body
        if pending:
            chunk = pending[:_UPLOAD_READ_CHUNK_BYTES]
            pending = pending[_UPLOAD_READ_CHUNK_BYTES:]
            if total + len(chunk) > MAX_REMOTE_UPLOAD_BODY_BYTES:
                raise _UploadBodyTooLarge()
            total += len(chunk)
            more_body = bool(pending) or pending_more_body
            if not pending:
                pending_more_body = False
            return {"type": "http.request", "body": chunk, "more_body": more_body}
        message = await base_receive()
        if message.get("type") != "http.request":
            return message
        raw_body: Any = message.get("body", b"")
        body = raw_body if isinstance(raw_body, bytes) else b""
        more_body = bool(message.get("more_body", False))
        if len(body) == 0:
            return {"type": "http.request", "body": b"", "more_body": more_body}
        if total + len(body) > MAX_REMOTE_UPLOAD_BODY_BYTES:
            raise _UploadBodyTooLarge()
        if len(body) <= _UPLOAD_READ_CHUNK_BYTES:
            total += len(body)
            return {"type": "http.request", "body": body, "more_body": more_body}
        chunk = body[:_UPLOAD_READ_CHUNK_BYTES]
        pending = body[_UPLOAD_READ_CHUNK_BYTES:]
        pending_more_body = more_body
        total += len(chunk)
        return {"type": "http.request", "body": chunk, "more_body": True}

    return bounded_receive


class BoundedRemoteUploadRoute(APIRoute):
    """Bound POST /api/assets/upload body before multipart parsing (remote only)."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        original = super().get_route_handler()
        if self.path != "/api/assets/upload":
            return original
        if self.methods is None or "POST" not in self.methods:
            return original

        async def bounded_upload_handler(request: Request) -> Response:
            if _declared_upload_body_too_large(
                request.headers.get("content-length")
            ):
                return _upload_too_large_response()
            bounded_request = Request(
                request.scope, _bounded_upload_receive(request.receive)
            )
            try:
                return await original(bounded_request)
            except _UploadBodyTooLarge:
                return _upload_too_large_response()

        return bounded_upload_handler


phone_assets_router = APIRouter(prefix="/api", route_class=BoundedRemoteUploadRoute)


def asset_list_query(
    media_kind: MediaKind | None = None,
    # Document the allowed values in OpenAPI without FastAPI 422; store/handler map invalid sort to 400.
    sort: Annotated[
        str,
        Query(
            json_schema_extra={"enum": ["created_at-desc", "created_at-asc"]},
        ),
    ] = "created_at-desc",
    q: str | None = None,
    cursor: str | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
) -> AssetListQuery:
    return AssetListQuery(
        media_kind=media_kind,
        sort=sort,
        q=q,
        cursor=cursor,
        limit=limit,
    )


@router.post("/assets", response_model=Asset, tags=["assets"])
def route_ingest_asset(
    req: IngestAssetRequest, handler: AppHandler = Depends(get_state_service)
) -> Asset:
    return handler.assets.ingest_asset(req)


@router.get(
    "/assets",
    response_model=AssetListResponse,
    tags=["assets"],
    operation_id="route_list_assets",
)
def route_list_assets(
    query: AssetListQuery = Depends(asset_list_query),
    handler: AppHandler = Depends(get_state_service),
) -> AssetListResponse:
    return handler.assets.list_assets(query)


@router.delete(
    "/assets/{asset_id}",
    response_model=StatusResponse,
    tags=["assets"],
    operation_id="route_delete_asset",
)
def route_delete_asset(
    asset_id: str, handler: AppHandler = Depends(get_state_service)
) -> StatusResponse:
    handler.assets.delete_asset(asset_id)
    return StatusResponse(status="ok")


@router.get("/assets/{asset_id}", response_model=Asset, tags=["assets"])
def route_get_asset(
    asset_id: str, handler: AppHandler = Depends(get_state_service)
) -> Asset:
    return handler.assets.get_asset(asset_id)


@router.post(
    "/assets/{asset_id}/trim-audio", response_model=Asset, tags=["assets"]
)
def route_trim_audio(
    asset_id: str,
    req: TrimMediaRequest,
    handler: AppHandler = Depends(get_state_service),
) -> Asset:
    return handler.assets.trim_audio(asset_id, req)


@router.post(
    "/assets/{asset_id}/trim-video", response_model=Asset, tags=["assets"]
)
def route_trim_video(
    asset_id: str,
    req: TrimMediaRequest,
    handler: AppHandler = Depends(get_state_service),
) -> Asset:
    return handler.assets.trim_video(asset_id, req)


@router.post(
    "/assets/{asset_id}/extract-audio", response_model=Asset, tags=["assets"]
)
def route_extract_audio(
    asset_id: str, handler: AppHandler = Depends(get_state_service)
) -> Asset:
    return handler.assets.extract_audio(asset_id)


@phone_assets_router.get(
    "/assets",
    response_model=RemoteAssetListResponse,
    tags=["assets"],
    operation_id="route_list_phone_assets",
)
def route_list_phone_assets(
    query: AssetListQuery = Depends(asset_list_query),
    handler: AppHandler = Depends(get_state_service),
    remote: RemoteContext = Depends(get_remote_context),
) -> RemoteAssetListResponse:
    page = handler.assets.list_assets(query, tombstone_missing=False)
    return RemoteAssetListResponse(
        items=[
            to_remote_list_item(
                item,
                in_use=item.in_use,
                signer=remote.signer,
                device_id=remote.device.id,
            )
            for item in page.items
        ],
        next_cursor=page.next_cursor,
    )


@phone_assets_router.get("/assets/{asset_id}", response_model=RemoteAsset, tags=["assets"])
def route_get_phone_asset(
    asset_id: str,
    handler: AppHandler = Depends(get_state_service),
    remote: RemoteContext = Depends(get_remote_context),
) -> RemoteAsset:
    return remote.asset(
        handler.assets.get_asset(asset_id, tombstone_missing=False)
    )


@phone_assets_router.post(
    "/assets/{asset_id}/trim-audio",
    response_model=RemoteAsset,
    tags=["assets"],
)
def route_phone_trim_audio(
    asset_id: str,
    req: TrimMediaRequest,
    handler: AppHandler = Depends(get_state_service),
    remote: RemoteContext = Depends(get_remote_context),
) -> RemoteAsset:
    return remote.asset(handler.assets.trim_audio(asset_id, req))


@phone_assets_router.post(
    "/assets/{asset_id}/trim-video",
    response_model=RemoteAsset,
    tags=["assets"],
)
def route_phone_trim_video(
    asset_id: str,
    req: TrimMediaRequest,
    handler: AppHandler = Depends(get_state_service),
    remote: RemoteContext = Depends(get_remote_context),
) -> RemoteAsset:
    return remote.asset(handler.assets.trim_video(asset_id, req))


@phone_assets_router.post(
    "/assets/{asset_id}/extract-audio",
    response_model=RemoteAsset,
    tags=["assets"],
)
def route_phone_extract_audio(
    asset_id: str,
    handler: AppHandler = Depends(get_state_service),
    remote: RemoteContext = Depends(get_remote_context),
) -> RemoteAsset:
    return remote.asset(handler.assets.extract_audio(asset_id))


def _upload_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in (
        ALLOWED_IMAGE_SUFFIXES | ALLOWED_AUDIO_SUFFIXES | ALLOWED_VIDEO_SUFFIXES
    ):
        raise HTTPError(400, "UNSUPPORTED_MEDIA")
    return suffix


def _upload_max_bytes(suffix: str) -> int:
    if suffix in ALLOWED_AUDIO_SUFFIXES:
        return MAX_AUDIO_BYTES
    if suffix in ALLOWED_VIDEO_SUFFIXES:
        return MAX_VIDEO_BYTES
    return MAX_IMAGE_BYTES


def _upload_body_exceeds_limit(
    *, file_size: int | None, content_length: str | None, max_bytes: int
) -> bool:
    if file_size is not None and file_size > max_bytes:
        return True
    if content_length is None:
        return False
    try:
        declared = int(content_length)
    except ValueError:
        return False
    return declared > max_bytes + _MULTIPART_OVERHEAD_BYTES


def _reject_declared_oversize(
    file: UploadFile, request: Request, *, max_bytes: int
) -> None:
    if _upload_body_exceeds_limit(
        file_size=file.size,
        content_length=request.headers.get("content-length"),
        max_bytes=max_bytes,
    ):
        raise HTTPError(400, "FILE_TOO_LARGE")


async def _write_bounded_upload(
    file: UploadFile, handle: BinaryIO, max_bytes: int
) -> None:
    received = 0
    while True:
        chunk = await file.read(_UPLOAD_READ_CHUNK_BYTES)
        if not chunk:
            return
        if received + len(chunk) > max_bytes:
            raise HTTPError(400, "FILE_TOO_LARGE")
        handle.write(chunk)
        received += len(chunk)


@phone_assets_router.post(
    "/assets/upload", response_model=RemoteAsset, tags=["assets"]
)
async def route_upload_asset(
    request: Request,
    file: UploadFile = File(),
    handler: AppHandler = Depends(get_state_service),
    remote: RemoteContext = Depends(get_remote_context),
) -> RemoteAsset:
    suffix = _upload_suffix(file.filename)
    max_bytes = _upload_max_bytes(suffix)
    _reject_declared_oversize(file, request, max_bytes=max_bytes)
    fd, tmp_name = tempfile.mkstemp(prefix=_UPLOAD_TEMP_PREFIX, suffix=suffix)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            await _write_bounded_upload(file, handle, max_bytes)
        return remote.asset(
            handler.assets.ingest_asset(IngestAssetRequest(path=str(tmp_path))),
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def _file_response_for(
    path_value: str, *, mime_type: str, filename: str
) -> FileResponse:
    path = Path(path_value)
    if not path.is_file():
        raise HTTPError(404, "ASSET_NOT_FOUND")
    return FileResponse(
        path,
        media_type=mime_type,
        filename=filename,
        content_disposition_type="inline",
    )


# Also on the Desktop router: the renderer streams asset bytes over HTTP for
# waveform decoding instead of marshalling a base64 copy through IPC.
@router.get(
    "/assets/{asset_id}/bytes",
    tags=["assets"],
    operation_id="route_desktop_asset_bytes",
)
@router.head("/assets/{asset_id}/bytes", tags=["assets"])
def route_desktop_asset_bytes(
    asset_id: str, handler: AppHandler = Depends(get_state_service)
) -> FileResponse:
    asset = handler.assets.get_asset(asset_id, tombstone_missing=False)
    return _file_response_for(
        asset.path, mime_type=asset.mime_type, filename=asset.name
    )


@router.get(
    "/assets/{asset_id}/thumbnail/bytes",
    tags=["assets"],
    operation_id="route_desktop_asset_thumbnail_bytes",
)
@router.head("/assets/{asset_id}/thumbnail/bytes", tags=["assets"])
def route_desktop_asset_thumbnail_bytes(
    asset_id: str, handler: AppHandler = Depends(get_state_service)
) -> FileResponse:
    asset = handler.assets.get_asset(asset_id, tombstone_missing=False)
    if asset.thumbnail_path is None:
        raise HTTPError(404, "THUMBNAIL_NOT_FOUND")
    return _file_response_for(
        asset.thumbnail_path,
        mime_type="image/jpeg",
        filename=f"{asset.id}-thumb.jpg",
    )


@phone_assets_router.get("/assets/{asset_id}/bytes", tags=["assets"])
@phone_assets_router.head("/assets/{asset_id}/bytes", tags=["assets"])
def route_asset_bytes(
    asset_id: str, handler: AppHandler = Depends(get_state_service)
) -> FileResponse:
    asset = handler.assets.get_asset(asset_id, tombstone_missing=False)
    return _file_response_for(
        asset.path, mime_type=asset.mime_type, filename=asset.name
    )


@phone_assets_router.get("/assets/{asset_id}/thumbnail/bytes", tags=["assets"])
@phone_assets_router.head("/assets/{asset_id}/thumbnail/bytes", tags=["assets"])
def route_asset_thumbnail_bytes(
    asset_id: str, handler: AppHandler = Depends(get_state_service)
) -> FileResponse:
    asset = handler.assets.get_asset(asset_id, tombstone_missing=False)
    if asset.thumbnail_path is None:
        raise HTTPError(404, "THUMBNAIL_NOT_FOUND")
    return _file_response_for(
        asset.thumbnail_path,
        mime_type="image/jpeg",
        filename=f"{asset.id}-thumb.jpg",
    )
