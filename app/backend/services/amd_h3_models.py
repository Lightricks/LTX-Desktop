"""The native ComfyUI H3-Base recipes and the matching MiniMax license revision."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal, cast

from services.amd_comfy_models import ComfyWeight

H3Variant = Literal["h3-fl2va", "h3-fl2va-distill", "h3-ref2va"]
REPO = "Comfy-Org/MiniMax-H3"
REVISION = "4b53a35191437de05370da548e59aeabe2a5a09a"
H3_FL2VA_MODEL = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
H3_REF2VA_MODEL = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
H3_TEXT_ENCODER = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
H3_VIDEO_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
H3_MPS_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
H3_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
H3_TURBO_LORA = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"

# The license blob is distinct from the weight revision. A later model release
# with the same license can reuse the acknowledgement.
_LICENSE_BLOB = "c389b45855337ccec8ddeb389f4fe902abcd0b19"
_LICENSE_RECEIPT = "h3-license-v1.json"


def h3_license_acknowledged(app_data_dir: Path) -> bool:
    try:
        receipt = json.loads(
            (app_data_dir / _LICENSE_RECEIPT).read_text(encoding="utf-8")
        )
    except (OSError, ValueError, UnicodeError):
        return False
    if not isinstance(receipt, dict):
        return False
    record = cast(dict[str, object], receipt)
    return record.get("licenseBlob") == _LICENSE_BLOB and record.get("accepted") is True


def acknowledge_h3_license(app_data_dir: Path) -> None:
    app_data_dir.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=app_data_dir, delete=False
    ) as out:
        pending = Path(out.name)
        try:
            json.dump({"accepted": True, "licenseBlob": _LICENSE_BLOB}, out)
            out.flush()
            os.fsync(out.fileno())
        except OSError:
            out.close()
            pending.unlink(missing_ok=True)
            raise
    try:
        pending.replace(app_data_dir / _LICENSE_RECEIPT)
    finally:
        pending.unlink(missing_ok=True)


def _weight(
    category: str, name: str, size: int, sha256: str, sample_sha256: str
) -> ComfyWeight:
    return ComfyWeight(
        category,
        name,
        size,
        sha256,
        repo=REPO,
        revision=REVISION,
        sample_sha256=sample_sha256,
    )


_SHARED = (
    _weight(
        "text_encoders",
        H3_TEXT_ENCODER,
        27141342152,
        "bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923",
        "b20443758cc88bcd574344b83784f6738025ffa861fcd8921715b775a9aeea4b",
    ),
    _weight(
        "vae",
        H3_VIDEO_VAE,
        2811065184,
        "52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e",
        "e116899ddf041f835271dde278ffa7472d41514122bae523ecfba22239b34844",
    ),
    _weight(
        "vae",
        H3_AUDIO_VAE,
        605254808,
        "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
        "0f59b3889a8de7427c1b55a829a7eefff8d447bee28c5b12b31933c01296ecf3",
    ),
)
_FL2VA = (
    _weight(
        "diffusion_models",
        H3_FL2VA_MODEL,
        20970379616,
        "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a",
        "3beee54a87acccd2f177b7132366210c4bf17697439136bd8a87fdad07697350",
    ),
    *_SHARED,
)
H3_MODELS: dict[H3Variant, tuple[ComfyWeight, ...]] = {
    "h3-fl2va": _FL2VA,
    "h3-fl2va-distill": (
        *_FL2VA,
        _weight(
            "loras",
            H3_TURBO_LORA,
            1956193000,
            "2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e",
            "50c98f1166536676f2b69d8683430a0f6ea65553db3fa11b475f2cc2caeb98d5",
        ),
    ),
    "h3-ref2va": (
        _weight(
            "diffusion_models",
            H3_REF2VA_MODEL,
            20970379616,
            "9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779",
            "797419d6e3c5846e9f74e410aa1035af25a89d9c455b6d677830eb9146f98171",
        ),
        *_SHARED,
    ),
}


# The FP16 video VAE uses portable convolution and attention. The pinned core
# dequantizes INT8 ConvRot transformer/encoder weights on devices without int8 matmul.
_METAL_VAE = ComfyWeight(
    "vae", H3_MPS_VIDEO_VAE, 5207808496,
    "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522",
    repo=REPO, revision=REVISION,
)


def h3_model_recipes(*, metal: bool) -> dict[H3Variant, tuple[ComfyWeight, ...]]:
    if not metal:
        return H3_MODELS
    return {
        variant: tuple(_METAL_VAE if item.filename == H3_VIDEO_VAE else item for item in weights)
        for variant, weights in H3_MODELS.items()
    }
