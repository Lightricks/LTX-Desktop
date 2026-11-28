"""Executable Comfy API prompts for LTX 2.5 Fast with jointly generated audio.

Derived from Comfy-Org/workflow_templates v0.11.54 (b0bcdb274878ea3c5cf46fed3819faa05b920707),
templates/video_ltx2_5_{i2v,t2v}.json. Node contracts verified against ComfyUI
v0.34.3 (87465b8f1f64a27a46f16f22b13b410494dca66d). Prompt enhancement is off.
"""

from __future__ import annotations

import math
from pathlib import PureWindowsPath
from typing import Any

from services.amd_comfy_models import AUDIO_VAE, TEXT_ENCODER, TRANSFORMER, UPSCALER, VIDEO_VAE

BASE_SAMPLER_ID = "40"
REFINE_SAMPLER_ID = "60"
SAVE_VIDEO_ID = "90"

_BASE_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
_REFINE_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"

Graph = dict[str, dict[str, Any]]


def mp4_save_inputs(object_info: dict[str, Any] | None = None) -> dict[str, str]:
    """Choose the MP4 container with the codec selector used by this Comfy version."""
    if object_info is not None:
        required = object_info.get("SaveVideo", {}).get("input", {}).get("required", {})
        format_spec = required.get("format", [])
        if format_spec and format_spec[0] != "COMFY_DYNAMICCOMBO_V3":
            return {"format": "mp4", "codec": "auto"}
    return {"format": "mp4", "format.codec": "auto"}


def resolve_model_name(object_info: dict[str, Any], node_type: str, field: str, filename: str) -> str:
    """Use precisely the model name this server accepts, including any subfolders."""
    if node_type not in object_info:
        raise ValueError(f"ComfyUI is missing {node_type}. Update ComfyUI or install its required extension.")
    spec = object_info[node_type].get("input", {}).get("required", {}).get(field, [])
    choices: list[object] = []
    if spec and isinstance(spec[0], list):  # Legacy Comfy combo
        choices = spec[0]
    elif len(spec) >= 2 and isinstance(spec[1], dict):  # Current COMBO contract
        choices = spec[1].get("options", [])

    # Some families store several variants under identical basenames. Match the
    # whole requested suffix; tolerate a user's extra model-directory prefix.
    wanted = PureWindowsPath(filename.casefold()).parts
    paths = [
        name for name in choices
        if isinstance(name, str) and PureWindowsPath(name.casefold()).parts[-len(wanted):] == wanted
    ]
    if not paths:
        raise ValueError(
            f"ComfyUI cannot find {filename} in {node_type}. "
            "Download the Comfy model files in Settings, then refresh or restart ComfyUI."
        )
    return min(paths, key=lambda name: (name != filename, len(PureWindowsPath(name).parts), name))


def build_workflow(
    *,
    prompt: str,
    width: int,
    height: int,
    fps: float,
    frames: int,
    seed: int,
    negative_prompt: str = "",
    image_name: str | None = None,
    single_stage: bool = False,
    object_info: dict[str, Any] | None = None,
) -> Graph:
    """Build a Comfy /prompt graph. Width and height describe the saved video.

    Set image_name to the input-relative name returned by /upload/image for I2V.
    Supply the running server's /object_info to validate nodes and resolve weights.
    The single-stage route samples the requested canvas directly.
    """
    spatial_factor = 1 if single_stage else 2
    multiple = 32 * spatial_factor
    minimum = 64 * spatial_factor
    if any(n < minimum or n > 16384 or n % multiple for n in (width, height)):
        raise ValueError(f"Comfy LTX output dimensions must both be multiples of {multiple}, {minimum}–16384.")
    if not 1 <= frames <= 993 or (frames - 1) % 8:
        raise ValueError("Comfy LTX requires 8n+1 frames between 1 and 993.")
    if not math.isfinite(fps) or not 1 <= fps <= 120:
        raise ValueError("Comfy LTX requires a frame rate between 1 and 120.")
    if not 0 <= seed < 2**64:
        raise ValueError("Comfy LTX requires an unsigned 64-bit seed.")
    if image_name is not None:
        image_path = PureWindowsPath(image_name)
        if not image_path.name or image_path.anchor or ".." in image_path.parts:
            raise ValueError("Expected the input-relative image name returned by ComfyUI.")

    graph: Graph = {}

    def node(node_id: str, node_type: str, **inputs: Any) -> list[str | int]:
        if object_info is not None and node_type not in object_info:
            raise ValueError(f"ComfyUI is missing {node_type}. Update ComfyUI Desktop for LTX 2.5 support.")
        graph[node_id] = {"class_type": node_type, "inputs": inputs}
        return [node_id, 0]

    def loader(node_id: str, node_type: str, field: str, filename: str, **inputs: Any) -> list[str | int]:
        link = node(node_id, node_type, **inputs)
        graph[node_id]["inputs"][field] = (
            filename if object_info is None else resolve_model_name(object_info, node_type, field, filename)
        )
        return link

    model = loader("1", "UNETLoader", "unet_name", TRANSFORMER, weight_dtype="default")
    clip = loader("2", "CLIPLoader", "clip_name", TEXT_ENCODER, type="ltxv", device="default")
    video_vae = loader("3", "VAELoader", "vae_name", VIDEO_VAE)
    audio_vae = loader("4", "VAELoader", "vae_name", AUDIO_VAE)
    positive = node("10", "CLIPTextEncode", clip=clip, text=prompt)
    negative = node("11", "CLIPTextEncode", clip=clip, text=negative_prompt)
    conditioning = node("12", "LTXVConditioning", positive=positive, negative=negative, frame_rate=fps)
    guider = node(
        "13", "LTXVDualCFGGuider", model=model, positive=conditioning, negative=["12", 1],
        video_cfg=1.0, audio_cfg=1.0,
    )
    sampler = node("14", "KSamplerSelect", sampler_name="euler_ancestral")
    video = node(
        "20", "EmptyLTXVLatentVideo", width=width // spatial_factor, height=height // spatial_factor,
        length=frames, batch_size=1,
    )
    audio = node("21", "LTXVEmptyLatentAudio", frames_number=frames, frame_rate=fps, batch_size=1, audio_vae=audio_vae)

    image = None
    if image_name is not None:
        source = node("25", "LoadImage", image=image_name)
        scaled = node(
            "26", "ResizeImageMaskNode", input=source, resize_type="scale longer dimension",
            scale_method="lanczos", **{"resize_type.longer_size": 1536},
        )
        image = node("27", "LTXVPreprocess", image=scaled, img_compression=18)
        video = node(
            "28", "LTXVImgToVideoInplace", vae=video_vae, image=image, latent=video, strength=0.7, bypass=False,
        )

    latent = node("30", "LTXVConcatAVLatent", video_latent=video, audio_latent=audio)
    noise = node("31", "RandomNoise", noise_seed=seed)
    sigmas = node("32", "ManualSigmas", sigmas=_BASE_SIGMAS)
    sampled = node(
        BASE_SAMPLER_ID, "SamplerCustomAdvanced", noise=noise, guider=guider, sampler=sampler,
        sigmas=sigmas, latent_image=latent,
    )
    separated = node("41", "LTXVSeparateAVLatent", av_latent=sampled)
    if not single_stage:
        upscaler = loader("50", "LatentUpscaleModelLoader", "model_name", UPSCALER)
        video = node("51", "LTXVLatentUpsampler", samples=separated, upscale_model=upscaler, vae=video_vae)
        if image is not None:
            video = node(
                "52", "LTXVImgToVideoInplace", vae=video_vae, image=image, latent=video, strength=1.0, bypass=False,
            )
        latent = node("53", "LTXVConcatAVLatent", video_latent=video, audio_latent=["41", 1])
        # The stock template uses the user's seed for the base and fixed seed 42 for refinement.
        noise = node("54", "RandomNoise", noise_seed=42)
        sigmas = node("55", "ManualSigmas", sigmas=_REFINE_SIGMAS)
        sampled = node(
            REFINE_SAMPLER_ID, "SamplerCustomAdvanced", noise=noise, guider=guider, sampler=sampler,
            sigmas=sigmas, latent_image=latent,
        )
        separated = node("61", "LTXVSeparateAVLatent", av_latent=sampled)

    pixels = node(
        "70", "VAEDecodeTiled", samples=separated, vae=video_vae,
        tile_size=512, overlap=64, temporal_size=64, temporal_overlap=16,
    )
    sound = node("71", "LTXVAudioVAEDecode", samples=[separated[0], 1], audio_vae=audio_vae)
    movie = node("75", "CreateVideo", images=pixels, audio=sound, fps=fps)
    node(
        SAVE_VIDEO_ID, "SaveVideo", video=movie, filename_prefix="video/LTXDesktopAMD",
        **mp4_save_inputs(object_info),
    )
    return graph
