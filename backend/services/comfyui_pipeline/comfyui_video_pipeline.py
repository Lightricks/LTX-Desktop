"""ComfyUI workflow integration.

This module is the single place to add your ComfyUI API code.
The function ``call_comfyui_workflow`` is called by the video generation
handler in place of the local LTX GPU pipeline.  Replace the stub body
with your real implementation and every generation request (T2V and I2V)
will automatically use it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ComfyUIGenerationParams:
    """All parameters required to run a ComfyUI video-generation workflow."""

    prompt: str
    negative_prompt: str
    width: int
    height: int
    num_frames: int
    fps: int
    seed: int
    duration: int

    # Optional conditioning inputs — None when not provided by the user.
    image_path: str | None = None   # path to a local PNG/JPG for image-to-video
    audio_path: str | None = None   # path to a local audio file for audio-to-video


# ==============================================================================
# TODO: ADD YOUR COMFYUI WORKFLOW CODE HERE
# ==============================================================================
#
# Replace the body of ``call_comfyui_workflow`` with your implementation.
#
# What this function must do:
#   1. Connect to your ComfyUI instance (potentially via a tunnel URL).
#   2. Build / populate the workflow JSON with the values in ``params``.
#   3. Submit the workflow to the ComfyUI /prompt endpoint.
#   4. Poll the /history or websocket endpoint until the job is complete.
#   5. Download the output video from ComfyUI's /view endpoint.
#   6. Return the raw video bytes.  The caller writes them to disk and
#      serves the file path back to the Electron frontend.
#
# The returned bytes are expected to be a valid MP4 (or any container that
# the frontend's video player can display).
#
# Minimal sketch (fill in the blanks):
#
#   import requests, time
#
#   COMFYUI_URL = "http://<your-tunnel-hostname>"   # ← set this
#
#   def call_comfyui_workflow(params: ComfyUIGenerationParams) -> bytes:
#       workflow = _build_workflow(params)           # ← build your JSON
#       r = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow})
#       r.raise_for_status()
#       prompt_id = r.json()["prompt_id"]
#
#       while True:
#           hist = requests.get(f"{COMFYUI_URL}/history/{prompt_id}").json()
#           if prompt_id in hist:
#               output = hist[prompt_id]["outputs"]   # ← parse your node output
#               filename = output[<node_id>]["videos"][0]["filename"]
#               video = requests.get(f"{COMFYUI_URL}/view",
#                                    params={"filename": filename})
#               return video.content
#           time.sleep(1)
#
# ==============================================================================


def call_comfyui_workflow(params: ComfyUIGenerationParams) -> bytes:
    """Submit a request to the ComfyUI workflow and return the video as bytes.

    **This is a stub — replace the body with your ComfyUI API integration.**

    Parameters
    ----------
    params:
        All generation parameters collected from the frontend request.

    Returns
    -------
    bytes
        Raw video file bytes (MP4 expected).  The caller writes these bytes
        to the output path and returns the path to the frontend.
    """
    # ──────────────────────────────────────────────────────────────────────────
    # TODO: implement your ComfyUI workflow call here.
    # ──────────────────────────────────────────────────────────────────────────
    logger.error(
        "call_comfyui_workflow is a stub and has not been implemented yet. "
        "Edit backend/services/comfyui_pipeline/comfyui_video_pipeline.py."
    )
    raise NotImplementedError(
        "ComfyUI workflow stub: add your implementation to "
        "backend/services/comfyui_pipeline/comfyui_video_pipeline.py"
    )
