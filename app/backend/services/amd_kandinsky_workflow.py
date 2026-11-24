"""Kandinsky 6.0 Lite Distill SD image/text to synchronized video and audio.

The SD graphs follow kandinskylab/kandinsky-6 at
73b892b730885e02ab58fbf7fc5cb10b78e1e3da, comfyui/example_workflows.
"""

from __future__ import annotations

from pathlib import PureWindowsPath
from typing import Any

from services.amd_comfy_models import LITE_MODEL
from services.amd_comfy_workflow import Graph, mp4_save_inputs, resolve_model_name

SETUP_HELP = (
    "Open Kandinsky setup in the LTX Model menu to install its seven SD model files "
    "and the kandinsky6 extension from kandinskylab. External ComfyUI servers require core 0.38.0 or newer."
)


def build_kandinsky_workflow(
    *, prompt: str, negative_prompt: str = "", width: int, height: int, fps: float,
    frames: int, seed: int, image_name: str | None = None, object_info: dict[str, Any],
) -> tuple[Graph, str]:
    """Build the SD graph using the live server's loader paths and node contracts."""
    if (width, height) not in ((864, 480), (480, 864)) or fps != 24 or frames != 121:
        raise ValueError("Kandinsky Lite uses 5 seconds at 24 fps in 864×480 or 480×864 SD.")
    if not 0 <= seed < 2**64:
        raise ValueError("Kandinsky Lite requires an unsigned 64-bit seed.")
    if image_name is not None:
        path = PureWindowsPath(image_name)
        if not path.name or path.anchor or ".." in path.parts:
            raise ValueError("Expected the input-relative image name returned by ComfyUI.")

    graph: Graph = {}

    def node(node_id: str, node_type: str, **inputs: Any) -> list[str | int]:
        if node_type not in object_info:
            raise ValueError(f"Missing ComfyUI node {node_type}. {SETUP_HELP}")
        graph[node_id] = {"class_type": node_type, "inputs": inputs}
        return [node_id, 0]

    if "Kandinsky6Sampler" not in object_info:
        raise ValueError(SETUP_HELP)

    def model(node_type: str, field: str, filename: str) -> str:
        if node_type not in object_info:
            raise ValueError(f"Missing ComfyUI node {node_type}. {SETUP_HELP}")
        try:
            return resolve_model_name(object_info, node_type, field, filename)
        except ValueError as exc:
            raise ValueError(f"Missing Kandinsky model {filename}. {SETUP_HELP}") from exc

    network = node("denoiser", "UNETLoader", weight_dtype="default",
                   unet_name=model("UNETLoader", "unet_name", LITE_MODEL))
    encoder = node("encoder", "DualCLIPLoader", type="kandinsky5", device="default",
                   clip_name1=model("DualCLIPLoader", "clip_name1", "qwen_2.5_vl_7b.safetensors"),
                   clip_name2=model("DualCLIPLoader", "clip_name2", "clip_l.safetensors"))
    video_vae = node("video_vae", "VAELoader",
                     vae_name=model("VAELoader", "vae_name", "hunyuan_video_vae_bf16.safetensors"))
    audio_vae = node("audio_vae", "Kandinsky6AudioVAELoader",
                     tod_vae=model("Kandinsky6AudioVAELoader", "tod_vae", "v1-44.pth"),
                     bigvgan_dir=model("Kandinsky6AudioVAELoader", "bigvgan_dir", "bigvgan_vocoder"),
                     mode="44k", scaling_factor=0.417)
    # Desktop provides one audiovisual caption. Feed it to both conditioning streams.
    positive = node("positive", "Kandinsky6TextEncode", clip=encoder, video_caption=prompt, audio_caption=prompt)
    negative = node("negative", "Kandinsky6TextEncode", clip=encoder,
                    video_caption=negative_prompt, audio_caption="")
    latent = node("latent", "Kandinsky6EmptyLatent", width=width, height=height,
                  length=frames, fps=fps, batch_size=1)
    if image_name is not None:
        source = node("source", "LoadImage", image=image_name)
        scaled = node("crop", "ImageScale", image=source, upscale_method="bilinear",
                      width=width, height=height, crop="center")
        reference = node("reference", "VAEEncode", pixels=scaled, vae=video_vae)
        positive = node("condition", "Kandinsky6ImageToVideoAudio", positive=positive, negative=negative,
                        empty_latent=latent, reference_latent=reference)
        negative = ["condition", 1]
        latent = ["condition", 2]

    sampled = node("k_sampler", "Kandinsky6Sampler", model=network, positive=positive, negative=negative,
                   latent_image=latent, seed=seed, steps=10, cfg=1.0, sampler_name="euler",
                   scheduler="simple", denoise=1.0)
    if image_name is not None:
        sampled = node("trim_reference", "Kandinsky6RemoveReferenceLatent", joint_latent=sampled)
    video = node("decode", "VAEDecode", samples=sampled, vae=video_vae)
    sound = node("sound", "Kandinsky6AudioVAEDecode", joint_latent=sampled, audio_vae=audio_vae)
    movie = node("video", "CreateVideo", images=video, audio=sound, fps=fps)
    node("output", "SaveVideo", video=movie, filename_prefix="video/LTXDesktopAMD-K6", **mp4_save_inputs(object_info))
    return graph, "output"
