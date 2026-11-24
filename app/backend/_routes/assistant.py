import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import aclosing
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from api_types import (
    AssistantConversation, AssistantConversationSummary, AssistantModelDownloadRequest, AssistantModelsResponse, AssistantQueueRequest,
    AssistantSelection, AssistantSelectionRequest, AssistantTurnRequest, AssistantTurnResponse, StatusResponse,
)
from app_handler import AppHandler
from remote.pairing import AuthenticatedDevice, RemotePairing
from state import get_state_service

router = APIRouter(prefix="/api/assistant", tags=["assistant"])
# Remote gets the chat, never model downloads or deletes.
phone_assistant_router = APIRouter(prefix="/api/assistant", tags=["assistant"])


@router.get("/conversations", response_model=list[AssistantConversationSummary])
@phone_assistant_router.get("/conversations", response_model=list[AssistantConversationSummary])
def list_conversations(handler: AppHandler = Depends(get_state_service)) -> list[AssistantConversationSummary]:
    return handler.assistant.list_conversations()


@router.get("/conversations/{conversation_id}", response_model=AssistantConversation)
@phone_assistant_router.get("/conversations/{conversation_id}", response_model=AssistantConversation)
def get_conversation(conversation_id: str, handler: AppHandler = Depends(get_state_service)) -> AssistantConversation:
    return handler.assistant.get_conversation(conversation_id)


@router.delete("/conversations/{conversation_id}", response_model=StatusResponse)
@phone_assistant_router.delete("/conversations/{conversation_id}", response_model=StatusResponse)
def delete_conversation(conversation_id: str, handler: AppHandler = Depends(get_state_service)) -> StatusResponse:
    handler.assistant.delete_conversation(conversation_id)
    return StatusResponse(status="ok")


@router.post("/conversations/{conversation_id}/queue", response_model=AssistantConversation)
def enqueue_message(
    conversation_id: UUID, req: AssistantQueueRequest, handler: AppHandler = Depends(get_state_service),
) -> AssistantConversation:
    return handler.assistant.enqueue(str(conversation_id), req, origin="desktop")


@phone_assistant_router.post("/conversations/{conversation_id}/queue", response_model=AssistantConversation)
def enqueue_phone_message(
    conversation_id: UUID, req: AssistantQueueRequest, handler: AppHandler = Depends(get_state_service),
) -> AssistantConversation:
    return handler.assistant.enqueue(str(conversation_id), req, origin="remote")


@router.delete("/conversations/{conversation_id}/queue/{item_id}", response_model=AssistantConversation)
@phone_assistant_router.delete("/conversations/{conversation_id}/queue/{item_id}", response_model=AssistantConversation)
def remove_queued_message(conversation_id: UUID, item_id: UUID, handler: AppHandler = Depends(get_state_service)) -> AssistantConversation:
    return handler.assistant.remove_queued(str(conversation_id), str(item_id))


@router.post("/conversations/{conversation_id}/queue/release", response_model=AssistantConversation)
@phone_assistant_router.post("/conversations/{conversation_id}/queue/release", response_model=AssistantConversation)
def release_queue(conversation_id: UUID, handler: AppHandler = Depends(get_state_service)) -> AssistantConversation:
    return handler.assistant.release_queue(str(conversation_id))


@router.get("/selection", response_model=AssistantSelection)
@phone_assistant_router.get("/selection", response_model=AssistantSelection)
def assistant_selection(handler: AppHandler = Depends(get_state_service)) -> AssistantSelection:
    return handler.assistant.selection()


@router.put("/selection", response_model=AssistantSelection)
@phone_assistant_router.put("/selection", response_model=AssistantSelection)
def set_assistant_selection(req: AssistantSelectionRequest, handler: AppHandler = Depends(get_state_service)) -> AssistantSelection:
    return handler.assistant.set_selection(req)


@router.get("/models", response_model=AssistantModelsResponse)
@phone_assistant_router.get("/models", response_model=AssistantModelsResponse)
def assistant_models(handler: AppHandler = Depends(get_state_service)) -> AssistantModelsResponse:
    return handler.assistant_models.list_models()


@router.post("/models/download", response_model=StatusResponse)
def download_assistant_model(req: AssistantModelDownloadRequest, handler: AppHandler = Depends(get_state_service)) -> StatusResponse:
    handler.assistant_models.start_download(req.model)
    return StatusResponse(status="started")


@router.post("/models/delete", response_model=StatusResponse)
def delete_assistant_model(req: AssistantModelDownloadRequest, handler: AppHandler = Depends(get_state_service)) -> StatusResponse:
    handler.assistant_models.delete_model(req.model)
    return StatusResponse(status="ok")


@router.post("/turn", response_model=AssistantTurnResponse)
async def assistant_turn(
    req: AssistantTurnRequest, handler: AppHandler = Depends(get_state_service),
) -> AssistantTurnResponse:
    return await handler.assistant.turn(req, origin="desktop")


@phone_assistant_router.post("/turn", response_model=AssistantTurnResponse)
async def assistant_phone_turn(
    req: AssistantTurnRequest, handler: AppHandler = Depends(get_state_service),
) -> AssistantTurnResponse:
    return await handler.assistant.turn(req, origin="remote")


@router.post("/runs/{run_id}/continue", response_model=AssistantTurnResponse)
@phone_assistant_router.post("/runs/{run_id}/continue", response_model=AssistantTurnResponse)
async def continue_assistant(run_id: str, handler: AppHandler = Depends(get_state_service)) -> AssistantTurnResponse:
    return await handler.assistant.continue_run(run_id)


@router.get("/runs/{run_id}", response_model=AssistantTurnResponse)
@phone_assistant_router.get("/runs/{run_id}", response_model=AssistantTurnResponse)
def assistant_run(run_id: str, handler: AppHandler = Depends(get_state_service)) -> AssistantTurnResponse:
    return handler.assistant.get_run(run_id)


@router.get("/runs/{run_id}/events")
@phone_assistant_router.get("/runs/{run_id}/events")
async def assistant_events(run_id: str, request: Request, handler: AppHandler = Depends(get_state_service)) -> StreamingResponse:
    device = getattr(request.state, "paired_device", None)
    pairing = getattr(request.app.state, "remote_pairing", None)

    async def events() -> AsyncIterator[str]:
        checked = time.monotonic()
        # aclosing: an early return must remove the watcher now, not when the generator is collected.
        async with aclosing(handler.assistant.watch_progress(run_id)) as updates:
            async for progress in updates:
                if isinstance(device, AuthenticatedDevice) and isinstance(pairing, RemotePairing) and time.monotonic() - checked > 5:
                    # Remote auth runs when the stream opens; a revoked phone must stop receiving the reply.
                    if not await asyncio.to_thread(pairing.is_device_active, device.id):
                        return
                    checked = time.monotonic()
                yield ": heartbeat\n\n" if progress is None else f"data: {progress.model_dump_json()}\n\n"
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@router.post("/runs/{run_id}/stop", response_model=AssistantTurnResponse)
@phone_assistant_router.post("/runs/{run_id}/stop", response_model=AssistantTurnResponse)
def stop_assistant(run_id: str, handler: AppHandler = Depends(get_state_service)) -> AssistantTurnResponse:
    return handler.assistant.stop(run_id)
