"""Route handler for POST /api/extend."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import ExtendRequest, ExtendResponse
from state import get_state_service
from app_handler import AppHandler

router = APIRouter(prefix="/api", tags=["extend"])


@router.post("/extend", response_model=ExtendResponse)
def route_extend(req: ExtendRequest, handler: AppHandler = Depends(get_state_service)) -> ExtendResponse:
    return handler.extend.run(req)
