"""Canonical checkpoint specs and LTX model relationships."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never, cast, get_args

from api_types import (
    LTXLocalModelId,
    LTXVideoGenDuration,
    LTXVideoGenFps,
    LTXVideoGenPipeline,
    LTXVideoGenerationResolutionSpec,
    LTXVideoGenerationSpec,
    ModelCheckpointID,
)

logger = logging.getLogger(__name__)


ALL_MODEL_CP_IDS = cast(tuple[ModelCheckpointID, ...], get_args(ModelCheckpointID))
ALL_LTX_LOCAL_MODEL_IDS = cast(tuple[LTXLocalModelId, ...], get_args(LTXLocalModelId))


@dataclass(frozen=True, slots=True)
class ModelCheckpointSpec:
    relative_path: Path
    expected_size_bytes: int
    is_folder: bool
    repo_id: str
    description: str

    @property
    def name(self) -> str:
        return self.relative_path.name


@dataclass(frozen=True, slots=True)
class LTXLocalModelDeprecated:
    pass


@dataclass(frozen=True, slots=True)
class LTXLocalModelRelevant:
    upgrade_messages: dict[LTXLocalModelId, str]


LTXLocalModelRelevance = LTXLocalModelDeprecated | LTXLocalModelRelevant


@dataclass(frozen=True, slots=True)
class LtxIcLorasSpec:
    depth_cp: ModelCheckpointID
    canny_cp: ModelCheckpointID
    pose_cp: ModelCheckpointID


@dataclass(frozen=True, slots=True)
class LTXLocalModelSpec:
    model_cp: ModelCheckpointID
    upscale_cp: ModelCheckpointID
    text_encoder_cp: ModelCheckpointID
    ic_loras_spec: LtxIcLorasSpec
    relevance: LTXLocalModelRelevance
    supported_pipelines: tuple[tuple[LTXVideoGenPipeline, LTXVideoGenerationSpec], ...]
    version_label: str
    # The single newest model the app should recommend/upgrade to. Exactly one spec sets this
    # True (enforced in _validate_ltx_specs) so "latest" is explicit, not tuple-order-dependent.
    is_latest: bool = False


def _local_resolution_spec(
    *,
    fps_to_durations: dict[LTXVideoGenFps, tuple[LTXVideoGenDuration, ...]],
) -> LTXVideoGenerationResolutionSpec:
    return LTXVideoGenerationResolutionSpec(
        fps_to_durations={
            fps: list(durations)
            for fps, durations in fps_to_durations.items()
        },
    )


IMG_GEN_MODEL_CP_ID: ModelCheckpointID = "z-image-turbo"
DEPTH_PROCESSOR_CP_ID: ModelCheckpointID = "dpt-hybrid-midas"
PERSON_DETECTOR_CP_ID: ModelCheckpointID = "yolox-l-torchscript"
POSE_PROCESSOR_CP_ID: ModelCheckpointID = "dw-ll-ucoco-384-bs5"

_DISTILLED_PIPELINES: tuple[tuple[LTXVideoGenPipeline, LTXVideoGenerationSpec], ...] = (
    (
        "fast",
        LTXVideoGenerationSpec(
            display_name="LTX 2.3 Fast",
            supported_resolutions_durations={
                "540p": _local_resolution_spec(
                    fps_to_durations={
                        24: (5, 6, 8, 10, 20),
                    },
                ),
                "720p": _local_resolution_spec(
                    fps_to_durations={
                        24: (5, 6, 8, 10),
                    },
                ),
                "1080p": _local_resolution_spec(
                    fps_to_durations={
                        24: (5,),
                    },
                ),
            },
        ),
    ),
)


def get_model_cp_spec(cp_id: ModelCheckpointID) -> ModelCheckpointSpec:
    match cp_id:
        case "ltx-2.3-22b-distilled":
            return ModelCheckpointSpec(
                relative_path=Path("ltx-2.3-22b-distilled.safetensors"),
                expected_size_bytes=43_000_000_000,
                is_folder=False,
                repo_id="Lightricks/LTX-2.3",
                description="Main transformer model",
            )
        case "ltx-2.3-22b-distilled-1.1":
            return ModelCheckpointSpec(
                relative_path=Path("ltx-2.3-22b-distilled-1.1.safetensors"),
                expected_size_bytes=46_149_345_334,
                is_folder=False,
                repo_id="Lightricks/LTX-2.3",
                description="Main transformer model",
            )
        case "ltx-2.3-spatial-upscaler-x2-1.0":
            # Superseded by 1.1, but kept as a known checkpoint so persisted settings /
            # in-flight sessions referencing it still validate, and an orphaned on-disk
            # copy can be listed/deleted rather than failing enum validation.
            return ModelCheckpointSpec(
                relative_path=Path("ltx-2.3-spatial-upscaler-x2-1.0.safetensors"),
                expected_size_bytes=995_743_504,
                is_folder=False,
                repo_id="Lightricks/LTX-2.3",
                description="2x upscaler (legacy)",
            )
        case "ltx-2.3-spatial-upscaler-x2-1.1":
            return ModelCheckpointSpec(
                relative_path=Path("ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
                expected_size_bytes=995_743_560,
                is_folder=False,
                repo_id="Lightricks/LTX-2.3",
                description="2x upscaler",
            )
        case "ltx-2.3-22b-ic-lora-union-control-ref0.5":
            return ModelCheckpointSpec(
                relative_path=Path("ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"),
                expected_size_bytes=654_465_352,
                is_folder=False,
                repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
                description="Union IC-LoRA control model",
            )
        case "dpt-hybrid-midas":
            return ModelCheckpointSpec(
                relative_path=Path("dpt-hybrid-midas"),
                expected_size_bytes=500_000_000,
                is_folder=True,
                repo_id="Intel/dpt-hybrid-midas",
                description="DPT-Hybrid MiDaS depth processor",
            )
        case "yolox-l-torchscript":
            return ModelCheckpointSpec(
                relative_path=Path("yolox_l.torchscript.pt"),
                expected_size_bytes=217_697_649,
                is_folder=False,
                repo_id="hr16/yolox-onnx",
                description="YOLOX person detector for pose preprocessing",
            )
        case "dw-ll-ucoco-384-bs5":
            return ModelCheckpointSpec(
                relative_path=Path("dw-ll_ucoco_384_bs5.torchscript.pt"),
                expected_size_bytes=135_059_124,
                is_folder=False,
                repo_id="hr16/DWPose-TorchScript-BatchSize5",
                description="DW Pose TorchScript processor",
            )
        case "gemma-3-12b-it-qat-q4_0-unquantized":
            return ModelCheckpointSpec(
                relative_path=Path("gemma-3-12b-it-qat-q4_0-unquantized"),
                expected_size_bytes=25_000_000_000,
                is_folder=True,
                repo_id="Lightricks/gemma-3-12b-it-qat-q4_0-unquantized",
                description="Gemma text encoder (bfloat16)",
            )
        case "z-image-turbo":
            return ModelCheckpointSpec(
                relative_path=Path("Z-Image-Turbo"),
                expected_size_bytes=31_000_000_000,
                is_folder=True,
                repo_id="Tongyi-MAI/Z-Image-Turbo",
                description="Z-Image-Turbo model for text-to-image generation",
            )
        case _:
            assert_never(cp_id)


_DISTILLED_IC_LORAS = LtxIcLorasSpec(
    depth_cp="ltx-2.3-22b-ic-lora-union-control-ref0.5",
    canny_cp="ltx-2.3-22b-ic-lora-union-control-ref0.5",
    pose_cp="ltx-2.3-22b-ic-lora-union-control-ref0.5",
)

# What's-new for the distilled 1.1 upgrade, authored here and surfaced in the upgrade prompt
# (one improvement per line; the UI renders them as a bulleted "What's new" list).
_DISTILLED_1_1_WHATS_NEW = (
    "Fixes glitches and quality degradation in the final frames of longer clips (15s+).\n"
    "Removes stray text, logos, and watermark-like overlays that could appear near the end of long videos.\n"
    "Keeps fine detail consistent through to the last frame."
)


def get_ltx_model_spec(model_id: LTXLocalModelId) -> LTXLocalModelSpec:
    match model_id:
        case "ltx-2.3-22b-distilled-1.1":
            return LTXLocalModelSpec(
                model_cp="ltx-2.3-22b-distilled-1.1",
                upscale_cp="ltx-2.3-spatial-upscaler-x2-1.1",
                text_encoder_cp="gemma-3-12b-it-qat-q4_0-unquantized",
                ic_loras_spec=_DISTILLED_IC_LORAS,
                relevance=LTXLocalModelRelevant(
                    upgrade_messages={"ltx-2.3-22b-distilled": _DISTILLED_1_1_WHATS_NEW},
                ),
                supported_pipelines=_DISTILLED_PIPELINES,
                version_label="1.1",
                is_latest=True,
            )
        case "ltx-2.3-22b-distilled":
            return LTXLocalModelSpec(
                model_cp="ltx-2.3-22b-distilled",
                upscale_cp="ltx-2.3-spatial-upscaler-x2-1.1",
                text_encoder_cp="gemma-3-12b-it-qat-q4_0-unquantized",
                ic_loras_spec=_DISTILLED_IC_LORAS,
                relevance=LTXLocalModelRelevant(upgrade_messages={}),
                supported_pipelines=_DISTILLED_PIPELINES,
                version_label="1.0",
            )
        case _:
            assert_never(model_id)


def get_ltx_cps() -> set[ModelCheckpointID]:
    cp_ids: set[ModelCheckpointID] = set()
    for model_id in ALL_LTX_LOCAL_MODEL_IDS:
        cp_ids.add(get_ltx_model_spec(model_id).model_cp)
    return cp_ids


def get_latest_ltx_model_id() -> LTXLocalModelId:
    latest: list[LTXLocalModelId] = [m for m in ALL_LTX_LOCAL_MODEL_IDS if get_ltx_model_spec(m).is_latest]
    if len(latest) != 1:
        raise RuntimeError(f"Exactly one LTX model must set is_latest=True, found {len(latest)}")
    return latest[0]


def get_ltx_model_id_for_cp(cp_id: ModelCheckpointID) -> LTXLocalModelId | None:
    for model_id in ALL_LTX_LOCAL_MODEL_IDS:
        if get_ltx_model_spec(model_id).model_cp == cp_id:
            return model_id
    return None


def get_ic_loras_cp_ids(ic_loras_spec: LtxIcLorasSpec) -> tuple[ModelCheckpointID, ...]:
    return tuple(dict.fromkeys((ic_loras_spec.depth_cp, ic_loras_spec.canny_cp, ic_loras_spec.pose_cp)))


def get_ltx_model_cp_ids(model_id: LTXLocalModelId) -> tuple[ModelCheckpointID, ...]:
    spec = get_ltx_model_spec(model_id)
    return (
        spec.model_cp,
        spec.upscale_cp,
        spec.text_encoder_cp,
        *get_ic_loras_cp_ids(spec.ic_loras_spec),
    )


def _normalized_relative_path(cp_id: ModelCheckpointID) -> Path:
    relative_path = get_model_cp_spec(cp_id).relative_path
    if relative_path.is_absolute():
        raise ValueError(f"Model path for {cp_id} must be relative: {relative_path}")

    normalized_parts = [part for part in relative_path.parts if part not in ("", ".")]
    if not normalized_parts:
        raise ValueError(f"Model path for {cp_id} cannot be empty: {relative_path}")
    if ".." in normalized_parts:
        raise ValueError(f"Model path for {cp_id} cannot traverse parents: {relative_path}")

    return Path(*normalized_parts)


def resolve_model_path(models_dir: Path, cp_id: ModelCheckpointID) -> Path:
    return models_dir / _normalized_relative_path(cp_id)


def resolve_downloading_dir(models_dir: Path) -> Path:
    return models_dir / ".downloading"


def resolve_downloading_target_path(models_dir: Path, cp_id: ModelCheckpointID) -> Path:
    return resolve_downloading_dir(models_dir) / _normalized_relative_path(cp_id)


def resolve_downloading_path(models_dir: Path, cp_id: ModelCheckpointID) -> Path:
    spec = get_model_cp_spec(cp_id)
    relative_path = _normalized_relative_path(cp_id)
    downloading_dir = resolve_downloading_dir(models_dir)
    if spec.is_folder:
        return downloading_dir / relative_path
    parent = relative_path.parent
    if parent == Path("."):
        return downloading_dir
    return downloading_dir / parent


def is_cp_downloaded(models_dir: Path, cp_id: ModelCheckpointID) -> bool:
    path = resolve_model_path(models_dir, cp_id)
    spec = get_model_cp_spec(cp_id)
    if spec.is_folder:
        return path.exists() and any(path.iterdir())
    return path.exists()


def get_existing_cp_path(models_dir: Path, cp_id: ModelCheckpointID) -> Path:
    path = resolve_model_path(models_dir, cp_id)
    if not is_cp_downloaded(models_dir, cp_id):
        raise FileNotFoundError(f"Checkpoint not found: {cp_id} at {path}")
    return path


def delete_cp_path(models_dir: Path, cp_id: ModelCheckpointID) -> None:
    path = resolve_model_path(models_dir, cp_id)
    spec = get_model_cp_spec(cp_id)
    if spec.is_folder:
        if path.exists():
            import shutil

            shutil.rmtree(path)
        return
    path.unlink(missing_ok=True)


def get_downloaded_ltx_model_id(models_dir: Path) -> LTXLocalModelId | None:
    downloaded: list[LTXLocalModelId] = []
    for model_id in ALL_LTX_LOCAL_MODEL_IDS:
        if is_cp_downloaded(models_dir, get_ltx_model_spec(model_id).model_cp):
            downloaded.append(model_id)
    if not downloaded:
        return None
    if len(downloaded) == 1:
        return downloaded[0]

    logger.warning("Multiple LTX model checkpoints detected: %s", ", ".join(downloaded))
    relevant: list[LTXLocalModelId] = []
    for model_id in downloaded:
        if isinstance(get_ltx_model_spec(model_id).relevance, LTXLocalModelRelevant):
            relevant.append(model_id)
    if len(relevant) == 1:
        return relevant[0]
    if len(relevant) > 1:
        logger.warning("Multiple relevant LTX models detected; selecting the first available: %s", relevant[0])
        return relevant[0]
    logger.warning("Multiple deprecated LTX models detected; selecting the first available: %s", downloaded[0])
    return downloaded[0]


def _ltx_generation_bundle_on_disk(models_dir: Path, model_id: LTXLocalModelId) -> bool:
    """True when the always-required generation checkpoints for ``model_id`` are present.

    Transformer + upscaler are required regardless of settings; the text encoder is
    optional (an LTX API key encodes prompts instead), so it isn't checked here.
    """
    spec = get_ltx_model_spec(model_id)
    return is_cp_downloaded(models_dir, spec.model_cp) and is_cp_downloaded(models_dir, spec.upscale_cp)


def resolve_active_ltx_model_id(
    models_dir: Path, preferred: LTXLocalModelId | None
) -> LTXLocalModelId | None:
    # Only honour ``preferred`` if its full generation bundle is on disk — otherwise it would
    # be picked and then die in get_existing_cp_path on a missing companion (e.g. upscaler).
    if preferred is not None and _ltx_generation_bundle_on_disk(models_dir, preferred):
        return preferred
    # Prefer a version that can actually generate (newest first); fall back to whatever
    # transformer is on disk as a last resort so a partial install still resolves to something.
    for model_id in ALL_LTX_LOCAL_MODEL_IDS:
        if _ltx_generation_bundle_on_disk(models_dir, model_id):
            return model_id
    return get_downloaded_ltx_model_id(models_dir)


def _validate_model_cp_specs() -> None:
    relative_paths: dict[Path, ModelCheckpointID] = {}
    for cp_id in ALL_MODEL_CP_IDS:
        normalized = _normalized_relative_path(cp_id)
        existing = relative_paths.get(normalized)
        if existing is not None:
            raise RuntimeError(f"Duplicate checkpoint path mapping: {existing} and {cp_id} -> {normalized}")
        relative_paths[normalized] = cp_id


def _validate_ltx_specs() -> None:
    ltx_cps = get_ltx_cps()
    if len(ltx_cps) != len(ALL_LTX_LOCAL_MODEL_IDS):
        raise RuntimeError("LTX model primary checkpoints must map 1:1 with LTX model ids")
    _ = get_latest_ltx_model_id()


_validate_model_cp_specs()
_validate_ltx_specs()


# --- IC-LoRA weights ---
IC_LORA_SUBDIR = "ic-loras"


def _safe_segment(value: str, label: str) -> str:
    if not value or "/" in value or "\\" in value or ".." in value:
        raise ValueError(f"Unsafe {label}: {value!r}")
    return value


def resolve_ic_lora_path(models_dir: Path, ic_lora_id: str, filename: str) -> Path:
    rid = _safe_segment(ic_lora_id, "ic_lora_id")
    name = _safe_segment(filename, "filename")
    return models_dir / IC_LORA_SUBDIR / rid / name


def _find_installed(
    models_dir: Path,
    item_id: str,
    filenames: list[str],
    resolve: Callable[[Path, str, str], Path],
) -> Path | None:
    for filename in filenames:
        path = resolve(models_dir, item_id, filename)
        if path.exists():
            return path
    return None


def _downloaded_variant_ids(
    models_dir: Path,
    item_id: str,
    variants: list[tuple[str, str]],
    resolve: Callable[[Path, str, str], Path],
) -> list[str]:
    return [
        variant_id
        for variant_id, filename in variants
        if resolve(models_dir, item_id, filename).exists()
    ]


def is_ic_lora_downloaded(models_dir: Path, ic_lora_id: str, filename: str) -> bool:
    return resolve_ic_lora_path(models_dir, ic_lora_id, filename).exists()


def find_installed_ic_lora_path(models_dir: Path, ic_lora_id: str, filenames: list[str]) -> Path | None:
    """Return the first existing weights path among ``filenames`` (preferred order)."""
    return _find_installed(models_dir, ic_lora_id, filenames, resolve_ic_lora_path)


def is_any_ic_lora_variant_downloaded(models_dir: Path, ic_lora_id: str, filenames: list[str]) -> bool:
    return find_installed_ic_lora_path(models_dir, ic_lora_id, filenames) is not None


def downloaded_ic_lora_variant_ids(
    models_dir: Path, ic_lora_id: str, variants: list[tuple[str, str]]
) -> list[str]:
    """Return variant ids whose weights file exists. ``variants`` is ``(id, filename)``."""
    return _downloaded_variant_ids(models_dir, ic_lora_id, variants, resolve_ic_lora_path)


# --- Plain LoRA weights (catalog + manually-placed). Files land under "loras/<id>/" so the
#     model scanner classifies them as is_lora && !is_ic_lora. ---
LORA_SUBDIR = "loras"


def resolve_lora_path(models_dir: Path, lora_id: str, filename: str) -> Path:
    lid = _safe_segment(lora_id, "lora_id")
    name = _safe_segment(filename, "filename")
    return models_dir / LORA_SUBDIR / lid / name


def is_lora_downloaded(models_dir: Path, lora_id: str, filename: str) -> bool:
    return resolve_lora_path(models_dir, lora_id, filename).exists()


def find_installed_lora_path(models_dir: Path, lora_id: str, filenames: list[str]) -> Path | None:
    return _find_installed(models_dir, lora_id, filenames, resolve_lora_path)


def is_any_lora_variant_downloaded(models_dir: Path, lora_id: str, filenames: list[str]) -> bool:
    return find_installed_lora_path(models_dir, lora_id, filenames) is not None


def downloaded_lora_variant_ids(
    models_dir: Path, lora_id: str, variants: list[tuple[str, str]]
) -> list[str]:
    """Return variant ids whose weights file exists. ``variants`` is ``(id, filename)``."""
    return _downloaded_variant_ids(models_dir, lora_id, variants, resolve_lora_path)
