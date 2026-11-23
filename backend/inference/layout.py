"""Optional Desktop ``models/`` path table. The image must not call this module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, assert_never

# 2.5-native weights live under this family dir. 2.3 and shared tooling stay at models_root.
# 2.5.1 patches use new filenames in the same dir — not a nested ltx-2.5.1/.
LTX_2_5_FAMILY_DIR = Path("ltx-2.5")

# Same ids as Desktop ``api_types.ModelCheckpointID``. Duplicated so this module
# never imports ``api_types``.
CheckpointId: TypeAlias = Literal[
    "ltx-2.3-22b-distilled",
    "ltx-2.3-22b-distilled-1.1",
    "ltx-2.3-spatial-upscaler-x2-1.0",
    "ltx-2.3-spatial-upscaler-x2-1.1",
    "ltx-2.3-22b-ic-lora-union-control-ref0.5",
    "ltx-2.5-22b-distilled",
    "ltx-2.5-spatial-upscaler-x2-1.0",
    "ltx-2.5-video-vae",
    "ltx-2.5-video-vae-conv",
    "ltx-2.5-audio-vae",
    "ltx-2.5-duration-head",
    "dpt-hybrid-midas",
    "yolox-l-torchscript",
    "dw-ll-ucoco-384-bs5",
    "gemma-3-12b-it-qat-q4_0-unquantized",
    "gemma4-12b-with-proj-ltx-2.5",
    "gemma-4-e2b-it",
    "z-image-turbo",
]


@dataclass(frozen=True, slots=True)
class CheckpointLayout:
    """Relative location of one checkpoint under a Desktop ``models/`` folder."""

    relative_path: Path
    is_folder: bool


@dataclass(frozen=True, slots=True)
class ResolvedLtxModelPaths:
    """Filesystem paths for the active local LTX model bundle."""

    checkpoint_path: str
    upsampler_path: str
    gemma_root: str | None
    video_vae_path: str | None
    audio_vae_path: str | None
    duration_head_path: str | None


def get_checkpoint_layout(cp_id: CheckpointId) -> CheckpointLayout:
    """Desktop ``models/`` relative path and folder-vs-file for ``cp_id``."""
    match cp_id:
        case "ltx-2.3-22b-distilled":
            return CheckpointLayout(Path("ltx-2.3-22b-distilled.safetensors"), False)
        case "ltx-2.3-22b-distilled-1.1":
            return CheckpointLayout(Path("ltx-2.3-22b-distilled-1.1.safetensors"), False)
        case "ltx-2.3-spatial-upscaler-x2-1.0":
            return CheckpointLayout(Path("ltx-2.3-spatial-upscaler-x2-1.0.safetensors"), False)
        case "ltx-2.3-spatial-upscaler-x2-1.1":
            return CheckpointLayout(Path("ltx-2.3-spatial-upscaler-x2-1.1.safetensors"), False)
        case "ltx-2.3-22b-ic-lora-union-control-ref0.5":
            return CheckpointLayout(
                Path("ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"), False
            )
        case "ltx-2.5-22b-distilled":
            return CheckpointLayout(
                LTX_2_5_FAMILY_DIR / "ltx-2.5-22b-distilled-transformer-bf16.safetensors",
                False,
            )
        case "ltx-2.5-spatial-upscaler-x2-1.0":
            return CheckpointLayout(
                LTX_2_5_FAMILY_DIR / "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
                False,
            )
        case "ltx-2.5-video-vae":
            return CheckpointLayout(
                LTX_2_5_FAMILY_DIR / "ltx-2.5-video-vae-bf16.safetensors", False
            )
        case "ltx-2.5-video-vae-conv":
            return CheckpointLayout(
                LTX_2_5_FAMILY_DIR / "ltx-2.5-video-vae-conv-bf16.safetensors", False
            )
        case "ltx-2.5-audio-vae":
            return CheckpointLayout(
                LTX_2_5_FAMILY_DIR / "ltx-2.5-audio-vae-bf16.safetensors", False
            )
        case "ltx-2.5-duration-head":
            return CheckpointLayout(
                LTX_2_5_FAMILY_DIR / "ltx-2.5-duration-head-bf16.safetensors", False
            )
        case "dpt-hybrid-midas":
            return CheckpointLayout(Path("dpt-hybrid-midas"), True)
        case "yolox-l-torchscript":
            return CheckpointLayout(Path("yolox_l.torchscript.pt"), False)
        case "dw-ll-ucoco-384-bs5":
            return CheckpointLayout(Path("dw-ll_ucoco_384_bs5.torchscript.pt"), False)
        case "gemma-3-12b-it-qat-q4_0-unquantized":
            return CheckpointLayout(Path("gemma-3-12b-it-qat-q4_0-unquantized"), True)
        case "gemma4-12b-with-proj-ltx-2.5":
            return CheckpointLayout(
                LTX_2_5_FAMILY_DIR / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
                False,
            )
        case "gemma-4-e2b-it":
            return CheckpointLayout(Path("gemma-4-E2B-it"), True)
        case "z-image-turbo":
            return CheckpointLayout(Path("Z-Image-Turbo"), True)
        case _:
            assert_never(cp_id)
