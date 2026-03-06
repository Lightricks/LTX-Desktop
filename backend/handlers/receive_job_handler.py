"""Handler for receiving generation jobs from Director's Palette."""
from __future__ import annotations

import logging
import tempfile
from typing import TYPE_CHECKING

from _routes._errors import HTTPError
from api_types import ReceiveJobRequest, ReceiveJobResponse

if TYPE_CHECKING:
    from services.interfaces import HTTPClient
    from state.app_state_types import AppState
    from state.job_queue import JobQueue

logger = logging.getLogger(__name__)


class ReceiveJobHandler:
    def __init__(
        self,
        state: AppState,
        http: "HTTPClient",
        job_queue: "JobQueue",
    ) -> None:
        self._state = state
        self._http = http
        self._job_queue = job_queue

    def receive_job(self, req: ReceiveJobRequest) -> ReceiveJobResponse:
        api_key = self._state.app_settings.palette_api_key
        if not api_key:
            raise HTTPError(403, "Not connected to Director's Palette. Set a palette API key first.")

        params: dict[str, object] = {
            "prompt": req.prompt,
            "resolution": req.settings.resolution,
            "duration": req.settings.duration,
            "fps": req.settings.fps,
            "aspectRatio": req.settings.aspect_ratio,
        }

        if req.character_id is not None:
            params["character_id"] = req.character_id

        if req.first_frame_url is not None:
            local_path = self._download_remote_image(req.first_frame_url, "first_frame")
            params["imagePath"] = local_path

        if req.last_frame_url is not None:
            local_path = self._download_remote_image(req.last_frame_url, "last_frame")
            params["lastFramePath"] = local_path

        job = self._job_queue.submit(
            job_type="video",
            model=req.model,
            params=params,
            slot="api",
        )

        return ReceiveJobResponse(id=job.id, status=job.status)

    def _download_remote_image(self, url: str, prefix: str) -> str:
        try:
            resp = self._http.get(url, timeout=30)
            if resp.status_code != 200:
                raise HTTPError(400, f"Failed to download image from {url}: HTTP {resp.status_code}")
            suffix = ".png"
            if ".jpg" in url or ".jpeg" in url:
                suffix = ".jpg"
            tmp = tempfile.NamedTemporaryFile(prefix=f"{prefix}_", suffix=suffix, delete=False)
            tmp.write(resp.content)
            tmp.close()
            return tmp.name
        except HTTPError:
            raise
        except Exception as exc:
            raise HTTPError(400, f"Failed to download image from {url}") from exc
