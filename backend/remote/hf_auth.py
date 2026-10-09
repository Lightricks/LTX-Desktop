"""Remote HuggingFace auth surface: status / login / logout.

Mirrors ``_routes.hf_auth`` minus the OAuth ``/callback`` (the provider redirect
is handled by the desktop backend). Exposed so gated LoRA recipes can check auth
state from a paired session. Note: the shipped ``cozy-felt`` recipe is ungated
(``requires_hf_login: false``), so this surface is dormant for it today.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import (
    HuggingFaceAuthStatusResponse,
    HuggingFaceLoginResponse,
    HuggingFaceLogoutResponse,
)
from app_handler import AppHandler
from state import get_state_service

router = APIRouter(prefix="/api/auth/huggingface", tags=["hf_auth"])


@router.post("/login", response_model=HuggingFaceLoginResponse)
def route_hf_login(
    handler: AppHandler = Depends(get_state_service),
) -> HuggingFaceLoginResponse:
    return handler.hf_auth.start_login()


@router.get("/status", response_model=HuggingFaceAuthStatusResponse)
def route_hf_auth_status(
    handler: AppHandler = Depends(get_state_service),
) -> HuggingFaceAuthStatusResponse:
    return handler.hf_auth.get_auth_status()


@router.post("/logout", response_model=HuggingFaceLogoutResponse)
def route_hf_logout(
    handler: AppHandler = Depends(get_state_service),
) -> HuggingFaceLogoutResponse:
    return handler.hf_auth.logout()
