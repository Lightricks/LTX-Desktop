"""Local MiniMax H3 audiovisual prompts for ComfyUI 0.38.0.

Sampling and joint-latent decoding follow Comfy-Org/workflow_templates at
8a665db5f05cea82ab8e77a8be70e8c71d536837, benchmarks/video_minimax_h3_t2v.json
and templates/video_minimax_h3_{i2v,r2v}.json. The Radeon bundle uses the INT8
Qwen encoder in place of the native template's NVFP4 encoder.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import PureWindowsPath
from typing import Any

from services.amd_comfy_workflow import Graph, mp4_save_inputs, resolve_model_name
from services.amd_h3_models import (
    H3_AUDIO_VAE,
    H3_FL2VA_MODEL,
    H3_REF2VA_MODEL,
    H3_TEXT_ENCODER,
    H3_TURBO_LORA,
    H3_VIDEO_VAE,
)

_CANVASES = {(864, 480), (480, 864), (480, 480), (1344, 768), (768, 1344), (768, 768)}


def build_h3_workflow(
    *,
    engine: str,
    prompt: str,
    negative_prompt: str = "",
    width: int,
    height: int,
    frames: int,
    fps: float,
    seed: int,
    first_image: str | None = None,
    last_image: str | None = None,
    reference_images: Sequence[str] = (),
    object_info: dict[str, Any],
) -> tuple[Graph, str]:
    """Build a local H3 prompt using uploaded image names and the worker's actual loader paths."""
    if engine not in ("h3-fl2va", "h3-fl2va-distill", "h3-ref2va"):
        raise ValueError(f"Unknown local MiniMax H3 engine: {engine}")
    reference = engine == "h3-ref2va"
    turbo = engine == "h3-fl2va-distill"
    if (width, height) not in _CANVASES or fps != 24 or frames not in (124, 243, 362):
        raise ValueError("Choose a supported H3 480p or 768p canvas, 24 fps and 5, 10 or 15 seconds.")
    if not 0 <= seed < 2**64:
        raise ValueError("MiniMax H3 requires an unsigned 64-bit seed.")
    if negative_prompt.strip():
        raise ValueError("Local MiniMax H3 uses one audiovisual prompt and does not accept a negative prompt.")
    if isinstance(reference_images, str):
        raise ValueError("Pass reference images as an ordered list of ComfyUI image names.")
    if reference and (first_image is not None or last_image is not None or not 1 <= len(reference_images) <= 9):
        raise ValueError("MiniMax H3 Reference requires 1–9 ordered reference images and no keyframes.")
    if not reference and reference_images:
        raise ValueError("Select MiniMax H3 Reference to use reference images.")

    graph: Graph = {}

    def node(node_id: str, kind: str, **inputs: Any) -> list[str | int]:
        if kind not in object_info:
            raise ValueError(f"The ComfyUI worker is missing {kind}. Use the private engine or update ComfyUI for MiniMax H3.")
        graph[node_id] = {"class_type": kind, "inputs": inputs}
        return [node_id, 0]

    def loader(node_id: str, kind: str, field: str, filename: str, **inputs: Any) -> list[str | int]:
        inputs[field] = resolve_model_name(object_info, kind, field, filename)
        return node(node_id, kind, **inputs)

    def image(node_id: str, name: str) -> list[str | int]:
        path = PureWindowsPath(name)
        if not path.name or path.anchor or ".." in path.parts:
            raise ValueError("Expected an input-relative image name returned by ComfyUI.")
        return node(node_id, "LoadImage", image=name)

    model = loader("transformer", "UNETLoader", "unet_name", H3_REF2VA_MODEL if reference else H3_FL2VA_MODEL,
                   weight_dtype="default")
    if turbo:
        model = loader("turbo", "LoraLoaderModelOnly", "lora_name", H3_TURBO_LORA, model=model, strength_model=1.0)
    encoder = loader("encoder", "CLIPLoader", "clip_name", H3_TEXT_ENCODER, type="minimax", device="default")
    video_vae = loader("video_vae", "VAELoader", "vae_name", H3_VIDEO_VAE)
    audio_vae = loader("audio_vae", "VAELoader", "vae_name", H3_AUDIO_VAE)

    conditions: dict[str, Any] = dict(clip=encoder, vae=video_vae, prompt=prompt, width=width, height=height, length=frames)
    if reference:
        for idx, name in enumerate(reference_images):
            conditions[f"ref_images.ref_image_{idx}"] = image(f"reference_{idx}", name)
        positive = node("condition", "MiniMaxH3ReferenceToVideo", **conditions, audio_vae=audio_vae, ref_image_size="match")
    else:
        if first_image is not None:
            conditions["first_frame"] = image("first", first_image)
        if last_image is not None:
            conditions["last_frame"] = image("last", last_image)
        positive = node("condition", "MiniMaxH3ImageToVideo", **conditions)

    guider = node("guider", "BasicGuider", model=model, conditioning=positive)
    sampler = node("sampler", "KSamplerSelect", sampler_name="res_multistep")
    sigmas = node("schedule", "BasicScheduler", model=model, scheduler="simple", steps=8 if turbo else 20, denoise=1.0)
    noise = node("noise", "RandomNoise", noise_seed=seed)
    joint = node("sample", "SamplerCustomAdvanced", noise=noise, guider=guider, sampler=sampler,
                 sigmas=sigmas, latent_image=["condition", 1])
    # These core decoders select the video and audio branches of the same nested latent.
    video = node("frames", "VAEDecode", samples=joint, vae=video_vae)
    audio = node("audio", "VAEDecodeAudio", samples=joint, vae=audio_vae)
    clip = node("movie", "CreateVideo", images=video, audio=audio, fps=fps)
    node("output", "SaveVideo", video=clip, filename_prefix="video/LTXDesktopAMD-H3", **mp4_save_inputs(object_info))
    return graph, "output"
