"""Desktop-only remote exposure status and paired-device control.

Part of the desktop OpenAPI contract so Settings can share a typed status payload.
Not mounted on the remote phone app.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from _routes._errors import HTTPError
from api_types import StatusResponse
from remote.controller import remote_controller_from_app
from remote.pairing import PairedDevicePublic
from remote.status import RemoteStatusResponse
from services.records import UnavailableError

router = APIRouter(prefix="/api", tags=["remote"])


@router.get("/remote/status", response_model=RemoteStatusResponse)
def route_remote_status(request: Request) -> RemoteStatusResponse:
    controller = remote_controller_from_app(request.app)
    if controller is None:
        return RemoteStatusResponse(
            permitted=False,
            serving=False,
            reason="remote controller is not initialized",
        )
    return controller.status()


@router.get("/remote/devices", response_model=list[PairedDevicePublic])
def route_list_remote_devices(request: Request) -> list[PairedDevicePublic]:
    controller = remote_controller_from_app(request.app)
    if controller is None:
        return []
    try:
        return controller.list_devices()
    except UnavailableError as exc:
        raise HTTPError(503, "STORE_UNAVAILABLE") from exc


@router.post("/remote/devices/{device_id}/revoke", response_model=StatusResponse)
def route_revoke_remote_device(device_id: str, request: Request) -> StatusResponse:
    controller = remote_controller_from_app(request.app)
    if controller is None:
        raise HTTPError(404, "DEVICE_NOT_FOUND")
    try:
        revoked = controller.revoke_device(device_id)
    except UnavailableError as exc:
        raise HTTPError(503, "STORE_UNAVAILABLE") from exc
    if not revoked:
        raise HTTPError(404, "DEVICE_NOT_FOUND")
    return StatusResponse(status="ok")
