"""Handler for contact sheet generation (3x3 cinematic angle grid)."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from api_types import GenerateContactSheetRequest, GenerateContactSheetResponse

if TYPE_CHECKING:
    from state.job_queue import JobQueue

logger = logging.getLogger(__name__)

CAMERA_ANGLES: list[str] = [
    "Close-up portrait, tight framing on the face",
    "Medium shot, waist-up framing",
    "Full body shot, head to toe framing",
    "Over-the-shoulder shot, looking past one subject",
    "Low angle hero shot, looking up dramatically",
    "High angle bird's eye view, looking down from above",
    "Profile side view, lateral perspective",
    "Three-quarter view, angled between front and side",
    "Wide establishing shot, showing full environment",
]


class ContactSheetHandler:
    def __init__(self, job_queue: "JobQueue") -> None:
        self._job_queue = job_queue

    def generate(self, req: GenerateContactSheetRequest) -> GenerateContactSheetResponse:
        job_ids: list[str] = []
        style_suffix = f", {req.style}" if req.style else ""

        for angle in CAMERA_ANGLES:
            prompt = f"{req.subject_description}, {angle}{style_suffix}"
            job = self._job_queue.submit(
                job_type="image",
                model="z-image-turbo",
                params={
                    "prompt": prompt,
                    "width": 1024,
                    "height": 1024,
                    "reference_image_path": req.reference_image_path,
                },
                slot="api",
            )
            job_ids.append(job.id)

        return GenerateContactSheetResponse(job_ids=job_ids)
