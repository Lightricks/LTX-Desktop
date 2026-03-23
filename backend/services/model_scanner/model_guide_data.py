"""Static format metadata for the video model guide."""

from __future__ import annotations

from api_types import DistilledLoraInfo, ModelFormatInfo

MODEL_FORMATS: list[ModelFormatInfo] = [
    ModelFormatInfo(
        id="bf16",
        name="BF16 (Full Precision)",
        size_gb=27.5,
        min_vram_gb=32,
        quality_tier="best",
        needs_distilled_lora=False,
        download_url="https://huggingface.co/Lightricks/LTX-Video-0.9.7/resolve/main/ltx-video-2b-v0.9.7.safetensors",
        description="Full precision model. Best quality, requires high-end GPU (32GB+ VRAM).",
    ),
    ModelFormatInfo(
        id="fp8",
        name="FP8 (8-bit Float)",
        size_gb=14.0,
        min_vram_gb=16,
        quality_tier="high",
        needs_distilled_lora=False,
        download_url="https://huggingface.co/Lightricks/LTX-Video-0.9.7-FP8/resolve/main/ltx-video-2b-v0.9.7-fp8.safetensors",
        description="8-bit float quantization. Excellent quality with half the VRAM of BF16.",
    ),
    ModelFormatInfo(
        id="gguf_q8",
        name="GGUF Q8 (8-bit Integer)",
        size_gb=14.2,
        min_vram_gb=16,
        quality_tier="high",
        needs_distilled_lora=True,
        download_url="https://huggingface.co/city96/LTX-Video-gguf/resolve/main/ltx-video-2b-v0.9.5-Q8_0.gguf",
        description="GGUF Q8_0 quantization. Near-lossless quality, GGUF-format inference.",
    ),
    ModelFormatInfo(
        id="gguf_q5k",
        name="GGUF Q5_K_M (5-bit)",
        size_gb=9.1,
        min_vram_gb=12,
        quality_tier="good",
        needs_distilled_lora=True,
        download_url="https://huggingface.co/city96/LTX-Video-gguf/resolve/main/ltx-video-2b-v0.9.5-Q5_K_M.gguf",
        description="GGUF Q5_K_M quantization. Good quality at ~9GB. Suitable for 12GB GPUs.",
    ),
    ModelFormatInfo(
        id="gguf_q4k",
        name="GGUF Q4_K_M (4-bit)",
        size_gb=7.4,
        min_vram_gb=10,
        quality_tier="acceptable",
        needs_distilled_lora=True,
        download_url="https://huggingface.co/city96/LTX-Video-gguf/resolve/main/ltx-video-2b-v0.9.5-Q4_K_M.gguf",
        description="GGUF Q4_K_M quantization. Acceptable quality at ~7GB. For 10GB GPUs.",
    ),
    ModelFormatInfo(
        id="nf4",
        name="NF4 (4-bit Normal Float)",
        size_gb=8.0,
        min_vram_gb=10,
        quality_tier="good",
        needs_distilled_lora=True,
        download_url="https://huggingface.co/Lightricks/LTX-Video-0.9.7-NF4",
        description="NF4 bitsandbytes quantization. Good quality at 4-bit precision.",
    ),
]

DISTILLED_LORA_INFO = DistilledLoraInfo(
    name="LTX-Video Distilled LoRA",
    size_gb=0.4,
    download_url="https://huggingface.co/Lightricks/LTX-Video-0.9.7-distilled/resolve/main/ltxv-2b-0.9.7-distilled-04steps-lora-rank32.safetensors",
    description="Required for GGUF and NF4 models. Enables fast 4-step inference.",
)


def recommend_format(vram_gb: int | None) -> str:
    """Return the recommended format ID for the given VRAM amount."""
    if vram_gb is None:
        return "gguf_q5k"
    if vram_gb >= 32:
        return "bf16"
    if vram_gb >= 16:
        return "fp8"
    if vram_gb >= 12:
        return "gguf_q5k"
    if vram_gb >= 10:
        return "gguf_q4k"
    return "gguf_q4k"
