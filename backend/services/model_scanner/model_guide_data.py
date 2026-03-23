"""Static metadata about available video model formats and download URLs."""

from __future__ import annotations

from api_types import DistilledLoraInfo, ModelFormatInfo

MODEL_FORMATS: list[ModelFormatInfo] = [
    ModelFormatInfo(
        id="bf16",
        name="BF16 (Full Precision)",
        size_gb=43,
        min_vram_gb=32,
        quality_tier="Best",
        needs_distilled_lora=False,
        download_url="https://huggingface.co/Lightricks/LTX-Video-2.3-22b-distilled",
        description="Best quality. Requires 32GB+ VRAM. Auto-downloaded by default.",
    ),
    ModelFormatInfo(
        id="fp8",
        name="FP8 Distilled Checkpoint",
        size_gb=22,
        min_vram_gb=20,
        quality_tier="Excellent",
        needs_distilled_lora=False,
        download_url="https://huggingface.co/Lightricks/LTX-Video-2.3-22b-distilled",
        description="Excellent quality, smaller file. Good for 20-31GB VRAM GPUs.",
    ),
    ModelFormatInfo(
        id="gguf_q8",
        name="GGUF Q8",
        size_gb=22,
        min_vram_gb=18,
        quality_tier="Excellent",
        needs_distilled_lora=True,
        download_url="https://huggingface.co/city96/LTX-Video-2.3-22b-0.9.7-dev-gguf",
        description="Excellent quality quantized model. Needs distilled LoRA.",
    ),
    ModelFormatInfo(
        id="gguf_q5k",
        name="GGUF Q5_K",
        size_gb=15,
        min_vram_gb=13,
        quality_tier="Very Good",
        needs_distilled_lora=True,
        download_url="https://huggingface.co/city96/LTX-Video-2.3-22b-0.9.7-dev-gguf",
        description="Very good quality, balanced size. Good for 16-19GB VRAM GPUs.",
    ),
    ModelFormatInfo(
        id="gguf_q4k",
        name="GGUF Q4_K",
        size_gb=12,
        min_vram_gb=10,
        quality_tier="Good",
        needs_distilled_lora=True,
        download_url="https://huggingface.co/city96/LTX-Video-2.3-22b-0.9.7-dev-gguf",
        description="Good quality, smallest file. Good for 10-15GB VRAM GPUs.",
    ),
    ModelFormatInfo(
        id="nf4",
        name="NF4 (4-bit BitsAndBytes)",
        size_gb=12,
        min_vram_gb=10,
        quality_tier="Good",
        needs_distilled_lora=True,
        download_url="https://huggingface.co/Lightricks/LTX-Video-2.3-22b-distilled",
        description="4-bit quantization via BitsAndBytes. Good for 10-15GB VRAM GPUs.",
    ),
]

DISTILLED_LORA_INFO = DistilledLoraInfo(
    name="LTX 2.3 Distilled LoRA",
    size_gb=0.5,
    download_url="https://huggingface.co/Lightricks/LTX-Video-2.3-22b-distilled",
    description="Required for GGUF and NF4 models to enable fast distilled generation.",
)


def recommend_format(vram_gb: int | None) -> str:
    """Return the recommended format ID based on available VRAM."""
    if vram_gb is None:
        return "bf16"
    if vram_gb >= 32:
        return "bf16"
    if vram_gb >= 20:
        return "fp8"
    if vram_gb >= 16:
        return "gguf_q5k"
    if vram_gb >= 10:
        return "gguf_q4k"
    return "api_only"
