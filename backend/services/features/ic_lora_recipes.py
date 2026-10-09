"""Home IC-LoRA recipes. One row per video-in catalog entry that shares this form.

The path id is the recipe. The body does not repeat the catalog id. A row that
is not a Home recipe is rejected here so it cannot silently run on this
generic path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from api_types import (
    CreateIcLoraRecipeRequest,
    IcLoraCatalogItem,
    IcLoraRecipeStoredParams,
    LoraEntry,
)
from runtime_config.model_download_specs import (
    downloaded_ic_lora_variant_ids,
    is_ic_lora_downloaded,
)
from runtime_config.offerings import resolve_offering_local_model_id
from services.features.lora_recipes import RecipeCreateError, require_catalog_runnable
from services.lora_catalog import LoraCatalogProvider


@dataclass(frozen=True, slots=True)
class IcLoraRecipe:
    recipe_id: str
    catalog_id: str
    # The model writes a matte. The recipe also bakes a WebM with alpha from it
    # and stores both. Backend only: the shared table pins ids, not outputs.
    cutout: bool = False


_CAP_PATH = Path(__file__).resolve().parents[3] / "shared" / "ic-lora-input-cap.json"
_CAP = json.loads(_CAP_PATH.read_text(encoding="utf-8"))
# Shared with the Home form via shared/ic-lora-input-cap.json.
MAX_IC_LORA_INPUT_VIDEO_SECONDS = int(_CAP["maxInputVideoSeconds"])
MAX_IC_LORA_INPUT_FRAMES = int(_CAP["maxInputFrames"])
_IC_LORA_DURATION_TOLERANCE_SECONDS = float(_CAP["durationToleranceSeconds"])
_IC_LORA_FPS_MATCH_TOLERANCE = float(_CAP["fpsMatchTolerance"])
IC_LORA_SUPPORTED_FPS: tuple[int, ...] = tuple(int(rate) for rate in _CAP["supportedFps"])


def ic_lora_max_input_seconds(fps: float) -> float:
    """Longest clip the selected output rate can run. 240 frames, and never past 10s."""
    return min(MAX_IC_LORA_INPUT_VIDEO_SECONDS, MAX_IC_LORA_INPUT_FRAMES / fps)


def _format_seconds(seconds: float) -> str:
    rounded = round(seconds, 1)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:.1f}"


def ic_lora_input_cap_message(duration_ms: int, fps: float) -> str | None:
    """Why this clip cannot run at ``fps``, or None when it fits."""
    limit = ic_lora_max_input_seconds(fps)
    if duration_ms / 1000 > limit + _IC_LORA_DURATION_TOLERANCE_SECONDS:
        return (
            f"This clip is longer than {_format_seconds(limit)}s at {fps:g}fps. "
            "Trim it to continue."
        )
    return None


def ic_lora_original_fps(source_fps: float) -> float:
    """Supported rate closest to the source, never more than a half frame above it.

    A source slower than every supported rate keeps its own rate. An equal
    distance picks the lower rate, so Original never prefers a faster rate.
    """
    eligible = [
        rate
        for rate in IC_LORA_SUPPORTED_FPS
        if rate <= source_fps + _IC_LORA_FPS_MATCH_TOLERANCE
    ]
    if not eligible:
        return source_fps
    return min(eligible, key=lambda rate: (abs(rate - source_fps), rate))


def ic_lora_fps_choices(source_fps: float) -> tuple[float, ...]:
    """Original, then every supported rate below it, highest first."""
    original = ic_lora_original_fps(source_fps)
    below = tuple(
        sorted((rate for rate in IC_LORA_SUPPORTED_FPS if rate < original), reverse=True)
    )
    return (original, *below)


def resolve_ic_lora_output_fps(source_fps: float, selected: float | None) -> float:
    """The rate create stores. Omitted means Original. A faster selection is refused."""
    if selected is None:
        return ic_lora_original_fps(source_fps)
    if selected > source_fps + _IC_LORA_FPS_MATCH_TOLERANCE:
        raise RecipeCreateError(
            "Frame rate cannot be higher than the source clip.",
            code="IC_LORA_FPS_ABOVE_SOURCE",
        )
    return selected


# Order matches shared/ic-lora-recipes.json.
IC_LORA_RECIPES: dict[str, IcLoraRecipe] = {
    "day-to-night": IcLoraRecipe(recipe_id="day-to-night", catalog_id="day-to-night"),
    "alpha-gen": IcLoraRecipe(
        recipe_id="alpha-gen", catalog_id="alpha-gen", cutout=True
    ),
    "deblur": IcLoraRecipe(recipe_id="deblur", catalog_id="deblur"),
    "colorization": IcLoraRecipe(recipe_id="colorization", catalog_id="colorization"),
    "clean-plate": IcLoraRecipe(recipe_id="clean-plate", catalog_id="clean-plate"),
    "decompression": IcLoraRecipe(recipe_id="decompression", catalog_id="decompression"),
    "water-simulation": IcLoraRecipe(recipe_id="water-simulation", catalog_id="water-simulation"),
    "layout-to-render": IcLoraRecipe(
        recipe_id="layout-to-render", catalog_id="layout-to-render"
    ),
    "restore": IcLoraRecipe(recipe_id="restore", catalog_id="restore"),
}


def get_ic_lora_recipe(recipe_id: str) -> IcLoraRecipe | None:
    return IC_LORA_RECIPES.get(recipe_id)


def queued_ic_lora_recipes() -> tuple[IcLoraRecipe, ...]:
    return tuple(IC_LORA_RECIPES.values())


def resolve_ic_lora_recipe_create(
    recipe_id: str,
    req: CreateIcLoraRecipeRequest,
    *,
    catalog: LoraCatalogProvider,
    models_dir: Path,
    hf_authenticated: bool,
    source_fps: float,
) -> tuple[IcLoraRecipe, IcLoraCatalogItem, IcLoraRecipeStoredParams]:
    """Validate a recipe create and return the stored params, including ``loras``."""
    recipe = get_ic_lora_recipe(recipe_id)
    if recipe is None:
        raise RecipeCreateError(f"Unknown IC-LoRA recipe: {recipe_id}", code="IC_LORA_UNKNOWN")
    item = catalog.get_ic_lora(recipe.catalog_id)
    if item is None:
        raise RecipeCreateError(
            f"Catalog id '{recipe.catalog_id}' for recipe '{recipe_id}' not found",
            code="IC_LORA_UNKNOWN",
        )
    if not item.is_home_recipe():
        raise RecipeCreateError(
            f"'{item.id}' needs its own form",
            code="IC_LORA_UNSUPPORTED_RECIPE",
        )
    if req.params.prompt.strip() == "" and not item.allows_empty_prompt:
        raise RecipeCreateError("Prompt is required.", code="INVALID_GENERATION_SPEC")
    if req.inputs.image is None and item.reference_image_required:
        raise RecipeCreateError(
            "Reference image is required.", code="INVALID_GENERATION_SPEC"
        )
    if req.inputs.image is not None and not item.allows_reference_image:
        raise RecipeCreateError(
            f"'{item.id}' takes no reference image.", code="INVALID_GENERATION_SPEC"
        )
    require_catalog_runnable(
        item,
        models_dir=models_dir,
        hf_authenticated=hf_authenticated,
        hf_code="IC_LORA_HF_AUTH_REQUIRED",
        model_code="IC_LORA_UNSUPPORTED_MODEL",
        local_model_id=resolve_offering_local_model_id(models_dir, req.params.model),
    )
    installed = downloaded_ic_lora_variant_ids(
        models_dir,
        item.id,
        [(variant.id, variant.filename) for variant in item.download.variants],
    )
    variant = item.download.variant_for_run(req.params.variantId, installed)
    if variant is None:
        raise RecipeCreateError(
            "UNKNOWN_DOWNLOAD_VARIANT", code="UNKNOWN_DOWNLOAD_VARIANT"
        )
    if not is_ic_lora_downloaded(models_dir, item.id, variant.filename):
        raise RecipeCreateError(
            "IC_LORA_NOT_DOWNLOADED", code="IC_LORA_NOT_DOWNLOADED"
        )
    fps = resolve_ic_lora_output_fps(source_fps, req.params.fps)
    audio_mode = (
        req.params.audioMode
        if req.params.audioMode is not None
        else item.default_settings.audio_mode
    )
    scale = (
        req.params.scale
        if req.params.scale is not None
        else item.default_settings.lora_strength
    )
    stored = IcLoraRecipeStoredParams.model_validate(
        {
            **req.params.model_dump(exclude={"scale", "variantId"}),
            "fps": fps,
            "audioMode": audio_mode,
            "loras": [
                LoraEntry(
                    ref="",
                    scale=scale,
                    catalogId=item.id,
                    displayName=item.name,
                    variantId=variant.id,
                )
            ],
        }
    )
    return recipe, item, stored
