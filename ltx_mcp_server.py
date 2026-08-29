#!/usr/bin/env python3
"""LTX-Desktop MCP Server.

Exposes LTX-Desktop's video generation backend (FastAPI on :8000) and
Electron project bridge (HTTP on :8100) as MCP tools so AI assistants
can automate video generation and timeline editing.

Usage:
    1. Start LTX-Desktop normally (pnpm dev)
    2. Run this server:  python ltx_mcp_server.py
    3. Register in your AI assistant's MCP configuration.
"""
from __future__ import annotations

import os
os.environ["PYTHONUNBUFFERED"] = "1"

import json
import logging
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LTX_BACKEND = os.getenv("LTX_BACKEND_URL", "http://127.0.0.1:8000")
LTX_BRIDGE  = os.getenv("LTX_BRIDGE_URL",  "http://127.0.0.1:8100")
REQUEST_TIMEOUT = float(os.getenv("LTX_TIMEOUT", "600"))  # seconds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ltx-mcp")

mcp = FastMCP(
    "ltx-desktop",
    description="MCP server for LTX-Desktop: AI video generation + timeline editing",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def _backend_get(path: str, params: dict | None = None) -> dict[str, Any]:
    """GET request to the LTX FastAPI backend."""
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.get(f"{LTX_BACKEND}{path}", params=params)
        resp.raise_for_status()
        return resp.json()


async def _backend_post(path: str, body: dict | None = None) -> dict[str, Any]:
    """POST request to the LTX FastAPI backend."""
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.post(f"{LTX_BACKEND}{path}", json=body or {})
        resp.raise_for_status()
        return resp.json()


async def _bridge_get(path: str) -> Any:
    """GET request to the Electron project bridge."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{LTX_BRIDGE}{path}")
        resp.raise_for_status()
        return resp.json()


async def _bridge_post(path: str, body: dict | None = None) -> Any:
    """POST / PUT request to the Electron project bridge."""
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(f"{LTX_BRIDGE}{path}", json=body or {})
        resp.raise_for_status()
        return resp.json()


async def _bridge_put(path: str, body: Any = None) -> Any:
    """PUT request to the Electron project bridge."""
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.put(f"{LTX_BRIDGE}{path}", json=body)
        resp.raise_for_status()
        return resp.json()


# ===================================================================
# BACKEND API TOOLS  (connect to FastAPI on :8000)
# ===================================================================

@mcp.tool()
async def get_health() -> str:
    """Check LTX-Desktop backend health: GPU status, loaded models, etc."""
    result = await _backend_get("/health")
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def get_models_status() -> str:
    """Get the download status of all required model files (checkpoint, upsampler, text encoder, etc.)."""
    result = await _backend_get("/api/models/status")
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def download_models(model_types: list[str] | None = None) -> str:
    """Start downloading required model files.

    Args:
        model_types: Optional list of model types to download.
                     Valid values: checkpoint, upsampler, distilled_lora, ic_lora,
                     depth_processor, person_detector, pose_processor, text_encoder, zit.
                     If empty, downloads all required models.
    """
    body: dict[str, Any] = {}
    if model_types:
        body["modelTypes"] = model_types
    result = await _backend_post("/api/models/download", body)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def generate_video(
    prompt: str,
    resolution: str = "512p",
    model: str = "fast",
    duration: str = "2",
    fps: str = "24",
    camera_motion: str = "none",
    negative_prompt: str = "",
    audio: str = "false",
    image_path: str | None = None,
    audio_path: str | None = None,
    aspect_ratio: str = "16:9",
) -> str:
    """Generate a video from a text prompt (text-to-video), an image (image-to-video), or audio (audio-to-video).

    Args:
        prompt: The text description of the video to generate. Required.
        resolution: Video resolution. Default "512p".
        model: Model variant. "fast" (default) or "pro".
        duration: Video duration in seconds. Default "2".
        fps: Frames per second. Default "24".
        camera_motion: Camera movement type. Options: none, dolly_in, dolly_out,
                       dolly_left, dolly_right, jib_up, jib_down, static, focus_shift.
        negative_prompt: Things to avoid in the generated video.
        audio: Whether to generate audio. "true" or "false" (default).
        image_path: Local file path of an input image for image-to-video generation.
        audio_path: Local file path of an input audio for audio-to-video generation.
        aspect_ratio: "16:9" (default) or "9:16".
    """
    body: dict[str, Any] = {
        "prompt": prompt,
        "resolution": resolution,
        "model": model,
        "duration": duration,
        "fps": fps,
        "cameraMotion": camera_motion,
        "negativePrompt": negative_prompt,
        "audio": audio,
        "aspectRatio": aspect_ratio,
    }
    if image_path:
        body["imagePath"] = image_path
    if audio_path:
        body["audioPath"] = audio_path

    result = await _backend_post("/api/generate", body)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def generate_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    num_steps: int = 4,
    num_images: int = 1,
) -> str:
    """Generate images from a text prompt.

    Args:
        prompt: The text description of the image to generate. Required.
        width: Image width in pixels. Default 1024.
        height: Image height in pixels. Default 1024.
        num_steps: Number of diffusion steps. Default 4.
        num_images: Number of images to generate. Default 1.
    """
    body = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "numSteps": num_steps,
        "numImages": num_images,
    }
    result = await _backend_post("/api/generate-image", body)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def retake_video(
    video_path: str,
    start_time: float,
    duration: float,
    prompt: str = "",
    mode: str = "replace_audio_and_video",
) -> str:
    """Re-generate a portion of an existing video (retake / inpainting).

    Args:
        video_path: Local path to the source video file.
        start_time: Start time in seconds for the retake region.
        duration: Duration in seconds for the retake region.
        prompt: Optional prompt to guide the retake.
        mode: Retake mode. Default "replace_audio_and_video".
    """
    body = {
        "video_path": video_path,
        "start_time": start_time,
        "duration": duration,
        "prompt": prompt,
        "mode": mode,
    }
    result = await _backend_post("/api/retake", body)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def suggest_gap_prompt(
    before_prompt: str = "",
    after_prompt: str = "",
    gap_duration: float = 5,
    mode: str = "t2v",
    before_frame: str | None = None,
    after_frame: str | None = None,
    input_image: str | None = None,
) -> str:
    """Ask the AI to suggest a prompt for filling a gap in the timeline.

    Args:
        before_prompt: Prompt of the clip before the gap.
        after_prompt: Prompt of the clip after the gap.
        gap_duration: Duration of the gap in seconds.
        mode: Generation mode. "t2v" (text-to-video) or others.
        before_frame: Optional path to the last frame image before the gap.
        after_frame: Optional path to the first frame image after the gap.
        input_image: Optional input image path.
    """
    body: dict[str, Any] = {
        "beforePrompt": before_prompt,
        "afterPrompt": after_prompt,
        "gapDuration": gap_duration,
        "mode": mode,
    }
    if before_frame:
        body["beforeFrame"] = before_frame
    if after_frame:
        body["afterFrame"] = after_frame
    if input_image:
        body["inputImage"] = input_image

    result = await _backend_post("/api/suggest-gap-prompt", body)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def get_generation_progress() -> str:
    """Get the current video generation progress (status, phase, step count)."""
    result = await _backend_get("/api/generation/progress")
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def cancel_generation() -> str:
    """Cancel the currently running video generation."""
    result = await _backend_post("/api/generate/cancel")
    return json.dumps(result, ensure_ascii=False, indent=2)


# ===================================================================
# ELECTRON BRIDGE TOOLS  (connect to project bridge on :8100)
# ===================================================================

@mcp.tool()
async def list_projects() -> str:
    """List all projects in LTX-Desktop. Each project contains assets and timelines.

    Returns a JSON array of project objects with their IDs, names, and metadata.
    """
    result = await _bridge_get("/api/projects")
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def get_project(project_id: str) -> str:
    """Get a specific project's full data including assets, timelines, clips, and settings.

    Args:
        project_id: The project's unique ID (e.g. "project-1710000000000-abc123def").
    """
    result = await _bridge_get(f"/api/projects/{project_id}")
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def update_project(project_id: str, project_json: str) -> str:
    """Update a project's data (assets, timelines, clips, etc.).

    This tool edits the entire project JSON. Use get_project first to read
    the current state, modify the needed fields, then send back the full
    project JSON.

    Args:
        project_id: The project's unique ID.
        project_json: The complete project JSON string. Must be valid JSON
                      conforming to the Project type schema.
    """
    project_data = json.loads(project_json)
    result = await _bridge_put(f"/api/projects/{project_id}", project_data)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def export_video(
    project_id: str,
    output_path: str,
    width: int = 1920,
    height: int = 1080,
    fps: int = 24,
    codec: str = "h264",
    quality: int = 80,
) -> str:
    """Export the current timeline of a project as a video file using FFmpeg.

    Args:
        project_id: The project's ID to export from.
        output_path: Local filesystem path for the output video (e.g. "/home/user/output.mp4").
        width: Output video width. Default 1920.
        height: Output video height. Default 1080.
        fps: Output frame rate. Default 24.
        codec: Video codec. Default "h264".
        quality: Output quality (0-100). Default 80.
    """
    body = {
        "projectId": project_id,
        "outputPath": output_path,
        "width": width,
        "height": height,
        "fps": fps,
        "codec": codec,
        "quality": quality,
    }
    result = await _bridge_post("/api/export", body)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def import_asset(
    project_id: str,
    file_path: str,
) -> str:
    """Import a local file (video, image, or audio) into a project's asset library.

    The file will be copied into the project's assets directory and be
    available for use on the timeline.

    Args:
        project_id: The project's ID to import into.
        file_path: Absolute local path to the file to import.
    """
    body = {
        "projectId": project_id,
        "filePath": file_path,
    }
    result = await _bridge_post("/api/import-asset", body)
    return json.dumps(result, ensure_ascii=False, indent=2)


# ===================================================================
# Entry point
# ===================================================================
if __name__ == "__main__":
    logger.info("Starting LTX-Desktop MCP Server...")
    logger.info("  Backend URL: %s", LTX_BACKEND)
    logger.info("  Bridge URL:  %s", LTX_BRIDGE)
    mcp.run()
