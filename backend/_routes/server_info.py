"""Remote-backend capability handshake."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import ServerCapabilities, ServerInfoResponse
from app_handler import AppHandler
from state import get_state_service

router = APIRouter(prefix="/api", tags=["server"])


@router.get("/server-info", response_model=ServerInfoResponse)
def route_server_info(handler: AppHandler = Depends(get_state_service)) -> ServerInfoResponse:
    config = handler.config
    return ServerInfoResponse(
        deployment_mode=config.deployment_mode,
        capabilities=ServerCapabilities(
            legacy_path_inputs=config.allow_legacy_path_inputs,
            models_dir_editable=config.models_dir_editable,
        ),
    )
