"""Static metadata about available video model formats and download URLs."""

from __future__ import annotations

from api_types import DistilledLoraInfo, ModelFormatInfo

MODEL_FORMATS: list[ModelFormatInfo] = [
    ModelFormatInfo(
        id="bf16",
        name="Full Quality (BF16)",
        size_gb=43,
        min_vram_gb=32,
        quality_tier="Best",
        needs_distilled_lora=False,
        download_url="https://huggingface.co/Lightricks/LTX-Video-2.3-22b-distilled",
        description=(
            "The original, uncompressed model. Best possible video quality but "
            "needs a high-end GPU with 32 GB+ of video memory (e.g. RTX 4090, A100). "
            "This is the default — it downloads automatically on first run."
        ),
    ),
    ModelFormatInfo(
        id="fp8",
        name="Half-Size (FP8)",
        size_gb=22,
        min_vram_gb=20,
        quality_tier="Excellent",
        needs_distilled_lora=False,
        download_url="https://huggingface.co/Lightricks/LTX-Video-2.3-22b-distilled",
        description=(
            "Same model, compressed to half the file size with almost no quality loss. "
            "Great for GPUs with 20–31 GB of video memory (e.g. RTX 3090, RTX 4080). "
            "Drop-in replacement — just swap the file."
        ),
    ),
    ModelFormatInfo(
        id="gguf_q8",
        name="Compressed Q8 (GGUF)",
        size_gb=22,
        min_vram_gb=18,
        quality_tier="Excellent",
        needs_distilled_lora=True,
        download_url="https://huggingface.co/city96/LTX-Video-2.3-22b-0.9.7-dev-gguf",
        description=(
            "High-quality compressed model. Very close to the original but uses "
            "less video memory. Needs the Speed Boost LoRA file (see below). "
            "Good for 20–24 GB GPUs."
        ),
    ),
    ModelFormatInfo(
        id="gguf_q5k",
        name="Compressed Q5 (GGUF)",
        size_gb=15,
        min_vram_gb=13,
        quality_tier="Very Good",
        needs_distilled_lora=True,
        download_url="https://huggingface.co/city96/LTX-Video-2.3-22b-0.9.7-dev-gguf",
        description=(
            "Nicely balanced — smaller file, still looks great. Best pick for "
            "16 GB GPUs like the RTX 4060 Ti 16GB or RTX 3080. "
            "Needs the Speed Boost LoRA file (see below)."
        ),
    ),
    ModelFormatInfo(
        id="gguf_q4k",
        name="Compressed Q4 (GGUF)",
        size_gb=12,
        min_vram_gb=10,
        quality_tier="Good",
        needs_distilled_lora=True,
        download_url="https://huggingface.co/city96/LTX-Video-2.3-22b-0.9.7-dev-gguf",
        description=(
            "Smallest file, runs on GPUs with as little as 10 GB of video memory "
            "(e.g. RTX 3060 12GB). Some quality loss but still very usable. "
            "Needs the Speed Boost LoRA file (see below)."
        ),
    ),
    ModelFormatInfo(
        id="nf4",
        name="4-Bit Compressed (NF4)",
        size_gb=12,
        min_vram_gb=10,
        quality_tier="Good",
        needs_distilled_lora=True,
        download_url="https://huggingface.co/Lightricks/LTX-Video-2.3-22b-distilled",
        description=(
            "Another way to run on 10–15 GB GPUs. Uses a different compression method "
            "that needs extra software (bitsandbytes). Try the Q4 GGUF option first — "
            "it's simpler to set up. Needs the Speed Boost LoRA file (see below)."
        ),
    ),
]

DISTILLED_LORA_INFO = DistilledLoraInfo(
    name="Speed Boost LoRA (Required for Compressed Models)",
    size_gb=0.5,
    download_url="https://huggingface.co/Lightricks/LTX-Video-2.3-22b-distilled",
    description=(
        "A small add-on file that makes compressed models generate videos fast. "
        "You MUST download this if you're using any of the compressed (GGUF or NF4) models. "
        "Just put it in the same folder as your model file."
    ),
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
