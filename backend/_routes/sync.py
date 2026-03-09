"""Palette sync routes."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app_handler import AppHandler
from state import get_state_service

router = APIRouter(prefix="/api/sync", tags=["sync"])


class ConnectRequest(BaseModel):
    token: str


class LoginRequest(BaseModel):
    email: str
    password: str


class EnhancePromptRequest(BaseModel):
    prompt: str
    level: str = "2x"


@router.get("/status")
def sync_status(handler: AppHandler = Depends(get_state_service)) -> dict[str, Any]:
    return handler.sync.get_status()


@router.post("/connect")
def sync_connect(body: ConnectRequest, handler: AppHandler = Depends(get_state_service)) -> dict[str, Any]:
    result = handler.sync.connect(body.token)
    if result.get("connected"):
        handler.settings.save_settings()
    return result


@router.post("/login")
def sync_login(body: LoginRequest, handler: AppHandler = Depends(get_state_service)) -> dict[str, Any]:
    result = handler.sync.login(body.email, body.password)
    if result.get("connected"):
        handler.settings.save_settings()
    return result


@router.post("/disconnect")
def sync_disconnect(handler: AppHandler = Depends(get_state_service)) -> dict[str, Any]:
    result = handler.sync.disconnect()
    handler.settings.save_settings()
    return result


class CheckCreditsRequest(BaseModel):
    generation_type: str
    count: int = 1


class DeductCreditsRequest(BaseModel):
    generation_type: str
    count: int = 1
    metadata: dict[str, Any] | None = None


@router.get("/credits")
def sync_credits(handler: AppHandler = Depends(get_state_service)) -> dict[str, Any]:
    return handler.sync.get_credits()


@router.post("/credits/check")
def sync_check_credits(
    body: CheckCreditsRequest,
    handler: AppHandler = Depends(get_state_service),
) -> dict[str, Any]:
    return handler.sync.check_credits(body.generation_type, body.count)


@router.post("/credits/deduct")
def sync_deduct_credits(
    body: DeductCreditsRequest,
    handler: AppHandler = Depends(get_state_service),
) -> dict[str, Any]:
    return handler.sync.deduct_credits(body.generation_type, body.count, body.metadata)


@router.get("/gallery")
def sync_gallery(
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=50, ge=1, le=100),
    type: str = Query(default="all"),
    handler: AppHandler = Depends(get_state_service),
) -> dict[str, Any]:
    return handler.sync.list_gallery(page=page, per_page=per_page, asset_type=type)


@router.get("/library/characters")
def sync_characters(handler: AppHandler = Depends(get_state_service)) -> dict[str, Any]:
    return handler.sync.list_characters()


@router.get("/library/styles")
def sync_styles(handler: AppHandler = Depends(get_state_service)) -> dict[str, Any]:
    return handler.sync.list_styles()


@router.get("/library/references")
def sync_references(
    category: str | None = Query(default=None),
    handler: AppHandler = Depends(get_state_service),
) -> dict[str, Any]:
    return handler.sync.list_references(category=category)


@router.post("/prompt/enhance")
def sync_enhance_prompt(
    body: EnhancePromptRequest,
    handler: AppHandler = Depends(get_state_service),
) -> dict[str, Any]:
    return handler.sync.enhance_prompt(prompt=body.prompt, level=body.level)
