"""Validate and select an independent local ComfyUI video workflow."""
from pathlib import Path
from typing import Literal, TypeGuard

from _routes._errors import HTTPError
from api_types import AmdVideoEngine, GenerateVideoRequest, ImageToVideoParams, LTXVideoGenAspectRatio, TextToVideoParams
from inference.frame_math import compute_num_frames

H3Engine = Literal["h3-fl2va", "h3-fl2va-distill", "h3-ref2va"]
ExternalEngine = Literal["comfy", "kandinsky"] | H3Engine
MODEL_ID = "ltx-2.5-22b-distilled"
H3_ASPECTS: tuple[LTXVideoGenAspectRatio, ...] = ("16:9", "9:16", "1:1")
H3_NAMES: dict[H3Engine, str] = {
    "h3-fl2va": "MiniMax H3 FL2VA",
    "h3-fl2va-distill": "MiniMax H3 FL2VA Wuli Distill",
    "h3-ref2va": "MiniMax H3 Ref2VA",
}


def is_h3_engine(engine: object) -> TypeGuard[H3Engine]:
    return isinstance(engine, str) and engine in H3_NAMES


def h3_canvas(resolution: str, aspect: str) -> tuple[int, int]:
    # These two wire labels also exist in older clients. H3 uses its own pixel grid.
    width, height = {"540p": (864, 480), "720p": (1344, 768)}[resolution]
    return {"16:9": (width, height), "9:16": (height, width), "1:1": (height, height)}[aspect]


def h3_frames(duration: int, fps: int) -> int:
    # H3's audio/video VAE accepts 17n+5 frames. Round up to include the requested duration.
    return ((duration * fps - 5 + 16) // 17) * 17 + 5


def require_h3_license(engine: ExternalEngine | None, data_dir: Path | None) -> None:
    if is_h3_engine(engine):
        from services.amd_h3_models import h3_license_acknowledged
        if data_dir is None or not h3_license_acknowledged(data_dir):
            raise HTTPError(403, "Review and accept the MiniMax H3 model license in Settings → Models before generating.",
                            code="H3_LICENSE_ACK_REQUIRED")


def external_engine(
    req: GenerateVideoRequest | TextToVideoParams | ImageToVideoParams,
    *,
    default: AmdVideoEngine,
    model_25: bool,
    available: bool,
    windows_video_available: bool = True,
    complex_inputs: bool = False,
    has_start: bool = False,
    has_end: bool = False,
    reference_count: int = 0,
) -> ExternalEngine | None:
    choice = req.amdFastEngine or default
    if choice == "native":
        return None
    if isinstance(req, GenerateVideoRequest):
        has_start = bool((req.imagePath or "").strip())
        has_end = bool((req.lastImagePath or "").strip())
        reference_count = len(req.referenceImagePaths)
    h3 = is_h3_engine(choice)
    if not available:
        reason = "This engine requires a supported local GPU runtime."
    elif not h3 and not windows_video_available:
        reason = "The LTX INT8 and Kandinsky renderers require the Windows GPU edition."
    elif not model_25:
        reason = "Select a local LTX 2.5 Fast offering for this engine."
    elif complex_inputs or getattr(req, "loras", []) or req.cameraMotion != "none":
        reason = "Use Native for LoRAs, camera controls, timeline keyframes, audio input or recipe jobs."
    elif getattr(req, "passShape", "auto") == "parts":
        reason = "Select Native to split a generation into parts."
    elif has_end and choice not in ("h3-fl2va", "h3-fl2va-distill"):
        reason = "Select Native or MiniMax H3 FL2VA to use an end frame."
    elif choice == "h3-ref2va" and not 1 <= int(has_start) + reference_count <= 9:
        reason = "MiniMax H3 Ref2VA requires one to nine ordered reference images."
    elif reference_count and choice != "h3-ref2va":
        reason = "Additional reference images require MiniMax H3 Ref2VA."
    elif h3 and (req.duration not in (5, 10, 15) or req.fps != 24 or req.resolution not in ("540p", "720p")):
        reason = "MiniMax H3 supports 480p or 768p, 5, 10 or 15 seconds, and 24 fps."
    elif h3 and req.seed is not None and not 0 <= req.seed < 2**64:
        reason = "MiniMax H3 requires an unsigned 64-bit seed."
    elif h3 and req.aspectRatio not in (*H3_ASPECTS, "auto"):
        reason = "MiniMax H3 supports landscape 16:9, portrait 9:16 and square 1:1."
    elif req.duration is None or req.resolution not in ("270p", "360p", "540p", "720p", "1080p"):
        reason = "Choose a fixed duration and a resolution up to 1080p, or select Native."
    elif choice == "comfy" and compute_num_frames(req.duration, req.fps) > 993:
        reason = "Choose a shorter duration or lower fps; this Comfy workflow supports up to 993 frames."
    elif req.nagScale not in (None, 0) or req.nagAlpha is not None or req.nagTau is not None:
        reason = "Select Native to apply custom negative-attention guidance."
    elif req.negativePrompt.strip():
        reason = "These Comfy workflows do not apply negative prompts. Select Native to apply NAG guidance."
    elif choice == "kandinsky" and (req.duration != 5 or req.fps != 24 or req.resolution != "540p"):
        reason = "Kandinsky Lite requires the 480p SD preset, 5 seconds and 24 fps."
    elif choice == "kandinsky" and req.aspectRatio not in ("auto", "16:9", "9:16"):
        reason = "Kandinsky Lite supports landscape 864x480 and portrait 480x864."
    else:
        return choice
    if req.amdFastEngine is not None:
        raise HTTPError(422, reason, code="INVALID_VIDEO_GENERATION_SPEC")
    return None
