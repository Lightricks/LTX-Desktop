"""Route handlers for prompt library and wildcard endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from _routes._errors import HTTPError
from api_types import (
    CreateWildcardRequest,
    ExpandWildcardsRequest,
    ExpandWildcardsResponse,
    IncrementUsageResponse,
    PromptListResponse,
    SavedPromptResponse,
    SavePromptRequest,
    StatusResponse,
    UpdateWildcardRequest,
    WildcardListResponse,
    WildcardResponse,
)
from app_handler import AppHandler
from state import get_state_service

router = APIRouter(prefix="/api", tags=["prompts"])


# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------


@router.get("/prompts", response_model=PromptListResponse)
def route_list_prompts(
    search: str | None = None,
    tag: str | None = None,
    sort_by: str | None = None,
    handler: AppHandler = Depends(get_state_service),
) -> PromptListResponse:
    prompts = handler.prompts.list_prompts(search=search, tag=tag, sort_by=sort_by)
    return PromptListResponse(
        prompts=[
            SavedPromptResponse(
                id=p.id,
                text=p.text,
                tags=p.tags,
                category=p.category,
                used_count=p.used_count,
                created_at=p.created_at,
                last_used_at=p.last_used_at,
            )
            for p in prompts
        ]
    )


@router.post("/prompts", response_model=SavedPromptResponse)
def route_save_prompt(
    req: SavePromptRequest,
    handler: AppHandler = Depends(get_state_service),
) -> SavedPromptResponse:
    p = handler.prompts.save_prompt(text=req.text, tags=req.tags, category=req.category)
    return SavedPromptResponse(
        id=p.id,
        text=p.text,
        tags=p.tags,
        category=p.category,
        used_count=p.used_count,
        created_at=p.created_at,
        last_used_at=p.last_used_at,
    )


@router.delete("/prompts/{prompt_id}", response_model=StatusResponse)
def route_delete_prompt(
    prompt_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> StatusResponse:
    deleted = handler.prompts.delete_prompt(prompt_id)
    if not deleted:
        raise HTTPError(404, f"Prompt {prompt_id} not found")
    return StatusResponse(status="ok")


@router.post("/prompts/{prompt_id}/usage", response_model=IncrementUsageResponse)
def route_increment_usage(
    prompt_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> IncrementUsageResponse:
    result = handler.prompts.increment_usage(prompt_id)
    if result is None:
        raise HTTPError(404, f"Prompt {prompt_id} not found")
    return IncrementUsageResponse(status="ok", used_count=result.used_count)


# ------------------------------------------------------------------
# Wildcards
# ------------------------------------------------------------------


@router.get("/wildcards", response_model=WildcardListResponse)
def route_list_wildcards(
    handler: AppHandler = Depends(get_state_service),
) -> WildcardListResponse:
    wildcards = handler.prompts.list_wildcards()
    return WildcardListResponse(
        wildcards=[
            WildcardResponse(
                id=w.id, name=w.name, values=w.values, created_at=w.created_at,
            )
            for w in wildcards
        ]
    )


@router.post("/wildcards", response_model=WildcardResponse)
def route_create_wildcard(
    req: CreateWildcardRequest,
    handler: AppHandler = Depends(get_state_service),
) -> WildcardResponse:
    w = handler.prompts.create_wildcard(name=req.name, values=req.values)
    return WildcardResponse(id=w.id, name=w.name, values=w.values, created_at=w.created_at)


@router.put("/wildcards/{wildcard_id}", response_model=WildcardResponse)
def route_update_wildcard(
    wildcard_id: str,
    req: UpdateWildcardRequest,
    handler: AppHandler = Depends(get_state_service),
) -> WildcardResponse:
    w = handler.prompts.update_wildcard(wildcard_id, values=req.values)
    if w is None:
        raise HTTPError(404, f"Wildcard {wildcard_id} not found")
    return WildcardResponse(id=w.id, name=w.name, values=w.values, created_at=w.created_at)


@router.delete("/wildcards/{wildcard_id}", response_model=StatusResponse)
def route_delete_wildcard(
    wildcard_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> StatusResponse:
    deleted = handler.prompts.delete_wildcard(wildcard_id)
    if not deleted:
        raise HTTPError(404, f"Wildcard {wildcard_id} not found")
    return StatusResponse(status="ok")


@router.post("/wildcards/expand", response_model=ExpandWildcardsResponse)
def route_expand_wildcards(
    req: ExpandWildcardsRequest,
    handler: AppHandler = Depends(get_state_service),
) -> ExpandWildcardsResponse:
    expanded = handler.prompts.expand_wildcards(
        prompt=req.prompt, mode=req.mode, count=req.count,
    )
    return ExpandWildcardsResponse(expanded=expanded)
