"""HuggingFace OAuth routes — login, callback, status, logout."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse

from api_types import HuggingFaceAuthStatusResponse, HuggingFaceLoginResponse, HuggingFaceLogoutResponse
from app_handler import AppHandler
from state import get_state_service

router = APIRouter(prefix="/api/auth/huggingface", tags=["hf_auth"])


@router.post("/login", response_model=HuggingFaceLoginResponse)
def route_hf_login(
    handler: AppHandler = Depends(get_state_service),
) -> HuggingFaceLoginResponse:
    return handler.hf_auth.start_login()


@router.get("/callback", response_class=HTMLResponse)
def route_hf_callback(
    code: str = Query(default=""),
    state: str = Query(default=""),
    error: str = Query(default=""),
    handler: AppHandler = Depends(get_state_service),
) -> HTMLResponse:
    return HTMLResponse(handler.hf_auth.handle_callback(code, state, error))


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
