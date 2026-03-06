"""Handler for style guide grid generation (3x3 style-across-subjects grid)."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from api_types import GenerateStyleGuideRequest, GenerateStyleGuideResponse

if TYPE_CHECKING:
    from state.job_queue import JobQueue

logger = logging.getLogger(__name__)

STYLE_SUBJECTS: list[str] = [
    "Portrait of a person",
    "Cityscape",
    "Nature landscape",
    "Interior room",
    "Food still life",
    "Vehicle on a road",
    "Animal in its habitat",
    "Architecture detail",
    "Abstract pattern",
]


class StyleGuideHandler:
    def __init__(self, job_queue: "JobQueue") -> None:
        self._job_queue = job_queue

    def generate(self, req: GenerateStyleGuideRequest) -> GenerateStyleGuideResponse:
        job_ids: list[str] = []
        description_suffix = f": {req.style_description}" if req.style_description else ""

        for subject in STYLE_SUBJECTS:
            prompt = f"{subject}, in the style of {req.style_name}{description_suffix}"
            params: dict[str, object] = {
                "prompt": prompt,
                "width": 1024,
                "height": 1024,
            }
            if req.reference_image_path is not None:
                params["reference_image_path"] = req.reference_image_path

            job = self._job_queue.submit(
                job_type="image",
                model="z-image-turbo",
                params=params,
                slot="api",
            )
            job_ids.append(job.id)

        return GenerateStyleGuideResponse(job_ids=job_ids)
