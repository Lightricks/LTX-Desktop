"""Route for contact sheet generation."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import GenerateContactSheetRequest, GenerateContactSheetResponse
from app_handler import AppHandler
from state import get_state_service

router = APIRouter(prefix="/api/contact-sheet", tags=["contact-sheet"])


@router.post("/generate", response_model=GenerateContactSheetResponse)
def route_generate_contact_sheet(
    req: GenerateContactSheetRequest,
    handler: AppHandler = Depends(get_state_service),
) -> GenerateContactSheetResponse:
    return handler.contact_sheet.generate(req)
