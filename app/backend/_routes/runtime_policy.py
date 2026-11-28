"""Route handlers for /api/runtime-policy."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import RuntimePolicyResponse
from app_handler import AppHandler
from state import get_state_service

router = APIRouter(prefix="/api", tags=["runtime-policy"])


@router.get("/runtime-policy", response_model=RuntimePolicyResponse)
def route_runtime_policy(handler: AppHandler = Depends(get_state_service)) -> RuntimePolicyResponse:
    return handler.runtime_policy.get_runtime_policy()
