"""Home IC-LoRA recipes. One row per video-in catalog entry that shares this form.

The path id is the recipe. The body does not repeat the catalog id. A row that
is not a Home recipe is rejected here so it cannot silently run on this
generic path.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast, get_args

from api_types import (
    CreateIcLoraRecipeRequest,
    CreateIcLoraSheetRequest,
    CreateIcLoraTrackRequest,
    IcLoraAudioMode,
    IcLoraCatalogItem,
    IcLoraControlType,
    IcLoraRecipeParams,
    IcLoraSynthesizedStoredParams,
    IcLoraVideoRecipeStoredParams,
    LoraEntry,
    LTXVideoGenAspectRatio,
)
from runtime_config.ic_lora_job_budget import catalog_output_fps
from runtime_config.model_download_specs import (
    downloaded_ic_lora_variant_ids,
    is_ic_lora_downloaded,
)
from runtime_config.offerings import resolve_offering_local_model_id
from services.aspect_ratio import find_closest_aspect_ratio_option
from services.features.lora_recipes import RecipeCreateError, require_catalog_runnable
from services.lora_catalog import LoraCatalogProvider


SynthesizedIcLoraKind = Literal["sheet", "track"]
IcLoraRecipeKind = Literal["video", "sheet", "track"]
_KIND_INPUT: dict[IcLoraRecipeKind, str] = {
    "video": "a video",
    "sheet": "images",
    "track": "an image and a path",
}

# The ltx.io wrap. The user types the scene only. The control clip fixes the motion, so the
# prompt locks the camera and names what the model must not add.
CAMERA_LOCK_PROMPT_TEMPLATE = (
    "Static locked camera. {prompt} No camera movement, no zoom, no reframing, no cuts.\n"
    "Negative Prompt: camera movement, zoom, reframing, panning, cuts"
)


@dataclass(frozen=True, slots=True)
class IcLoraRecipe:
    recipe_id: str
    catalog_id: str
    # The model writes a matte. The recipe also bakes a WebM with alpha from it
    # and stores both. Backend only: the shared table pins ids, not outputs.
    cutout: bool = False
    # "sheet" is an image list (Ingredients). The catalog declares it too (`input.max_count`).
    # Both exist because app_handler picks the executor at start-up, before the catalog
    # loads. `_recipe_item` rejects a row and an entry that disagree. "track" is one image and
    # a motion path (Motion Track). Its catalog entry is a plain video recipe: the job draws
    # the control clip from the path, so only this row says where the clip comes from.
    kind: IcLoraRecipeKind = "video"
    # The model refines a clip that is scaled up to the output canvas (Refine Details).
    # The recipe scales the clip with Lanczos to the canvas, as the model card says, and
    # a 1440p or 2160p cell may be larger than the source. Backend only, like ``cutout``.
    upscales_to_cell: bool = False
    # Fixed scaffold (with a ``{prompt}`` placeholder) for the synthesized kinds only. A video
    # recipe never reads it: a strict video prompt uses the catalog ``prompt_template``.
    # The job applies this scaffold before enhancement, so it needs ``skips_prompt_enhancement``.
    # A style recipe applies its scaffold after the rewrite instead.
    prompt_template: str | None = None
    # The job sends the prompt as it stands, with no automatic rewrite. Motion Track wraps
    # the prompt in a camera lock that a rewrite would change. Manual Enhance still works.
    skips_prompt_enhancement: bool = False

    def __post_init__(self) -> None:
        if self.prompt_template is None:
            return
        if self.kind == "video":
            raise ValueError(
                f"IC-LoRA recipe '{self.recipe_id}': prompt_template needs a synthesized kind"
            )
        if not self.skips_prompt_enhancement:
            raise ValueError(
                f"IC-LoRA recipe '{self.recipe_id}': prompt_template needs skips_prompt_enhancement"
            )


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
    "ingredients": IcLoraRecipe(recipe_id="ingredients", catalog_id="ingredients", kind="sheet"),
    "refine-details": IcLoraRecipe(
        recipe_id="refine-details", catalog_id="refine-details", upscales_to_cell=True
    ),
    "motion-track": IcLoraRecipe(
        recipe_id="motion-track",
        catalog_id="motion-track",
        kind="track",
        prompt_template=CAMERA_LOCK_PROMPT_TEMPLATE,
        skips_prompt_enhancement=True,
    ),
    "video-guide": IcLoraRecipe(recipe_id="video-guide", catalog_id="video-guide"),
}


def get_ic_lora_recipe(recipe_id: str) -> IcLoraRecipe | None:
    return IC_LORA_RECIPES.get(recipe_id)


def ic_lora_skips_prompt_enhancement(catalog_id: str) -> bool:
    return any(
        recipe.skips_prompt_enhancement
        for recipe in IC_LORA_RECIPES.values()
        if recipe.catalog_id == catalog_id
    )


def queued_ic_lora_recipes() -> tuple[IcLoraRecipe, ...]:
    return tuple(IC_LORA_RECIPES.values())


def _recipe_item(
    recipe_id: str, catalog: LoraCatalogProvider, *, kind: IcLoraRecipeKind
) -> tuple[IcLoraRecipe, IcLoraCatalogItem]:
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
    if recipe.kind != kind or item.is_sheet_recipe() != (kind == "sheet"):
        raise RecipeCreateError(
            f"'{item.id}' takes {_KIND_INPUT[recipe.kind]}.",
            code="INVALID_GENERATION_SPEC",
        )
    return recipe, item


def _resolve_checkpoint(
    item: IcLoraCatalogItem,
    params: IcLoraRecipeParams,
    *,
    models_dir: Path,
    hf_authenticated: bool,
    require_prompt: bool = True,
) -> tuple[IcLoraAudioMode, list[LoraEntry]]:
    """The checks both bodies share. Returns the audio mode and the one LoRA this run uses.

    ``require_prompt`` is false only for the tile preflight. That route judges the budget,
    and the budget does not read the prompt. Create keeps the check.
    """
    if require_prompt and params.prompt.strip() == "" and not item.allows_empty_prompt:
        raise RecipeCreateError("Prompt is required.", code="INVALID_GENERATION_SPEC")
    require_catalog_runnable(
        item,
        models_dir=models_dir,
        hf_authenticated=hf_authenticated,
        hf_code="IC_LORA_HF_AUTH_REQUIRED",
        model_code="IC_LORA_UNSUPPORTED_MODEL",
        local_model_id=resolve_offering_local_model_id(models_dir, params.model),
    )
    installed = downloaded_ic_lora_variant_ids(
        models_dir,
        item.id,
        [(variant.id, variant.filename) for variant in item.download.variants],
    )
    variant = item.download.variant_for_run(params.variantId, installed)
    if variant is None:
        raise RecipeCreateError(
            "UNKNOWN_DOWNLOAD_VARIANT", code="UNKNOWN_DOWNLOAD_VARIANT"
        )
    if not is_ic_lora_downloaded(
        models_dir, item.id, variant.filename, item.requires_checkpoints
    ):
        raise RecipeCreateError(
            "IC_LORA_NOT_DOWNLOADED", code="IC_LORA_NOT_DOWNLOADED"
        )
    audio_mode = (
        params.audioMode
        if params.audioMode is not None
        else item.default_settings.audio_mode
    )
    scale = (
        params.scale
        if params.scale is not None
        else item.default_settings.lora_strength
    )
    lora = LoraEntry(
        ref="",
        scale=scale,
        catalogId=item.id,
        displayName=item.name,
        variantId=variant.id,
    )
    return audio_mode, [lora]


def _catalog_fps(item: IcLoraCatalogItem, selected: float | None) -> float:
    """The rate a run without a source clip makes: the selected one, else the catalog rate."""
    fps = catalog_output_fps(item) if selected is None else selected
    if fps not in IC_LORA_SUPPORTED_FPS:
        raise RecipeCreateError(
            f"Frame rate must be one of {IC_LORA_SUPPORTED_FPS}.",
            code="INVALID_GENERATION_SPEC",
        )
    return fps


def _resolve_control_type(
    item: IcLoraCatalogItem, req: CreateIcLoraRecipeRequest
) -> IcLoraControlType | None:
    """The control type a run stores. Only a video-control entry takes a control type."""
    if not item.is_video_control_recipe():
        if req.params.controlType is not None:
            raise RecipeCreateError(
                f"'{item.id}' takes no control type.", code="INVALID_GENERATION_SPEC"
            )
        return None
    return req.params.controlType or cast(IcLoraControlType, item.controls[0].default)


def resolve_ic_lora_recipe_create(
    recipe_id: str,
    req: CreateIcLoraRecipeRequest,
    *,
    catalog: LoraCatalogProvider,
    models_dir: Path,
    hf_authenticated: bool,
    source_fps: float,
    require_prompt: bool = True,
) -> tuple[IcLoraRecipe, IcLoraCatalogItem, IcLoraVideoRecipeStoredParams]:
    """Validate a recipe create and return the stored params, including ``loras``."""
    recipe, item = _recipe_item(recipe_id, catalog, kind="video")
    if req.inputs.image is None and item.reference_image_required:
        raise RecipeCreateError(
            "Reference image is required.", code="INVALID_GENERATION_SPEC"
        )
    if req.inputs.image is not None and not item.allows_reference_image:
        raise RecipeCreateError(
            f"'{item.id}' takes no reference image.", code="INVALID_GENERATION_SPEC"
        )
    control_type = _resolve_control_type(item, req)
    audio_mode, loras = _resolve_checkpoint(
        item,
        req.params,
        models_dir=models_dir,
        hf_authenticated=hf_authenticated,
        require_prompt=require_prompt,
    )
    stored = IcLoraVideoRecipeStoredParams.model_validate(
        {
            **req.params.model_dump(exclude={"scale", "variantId"}),
            "fps": resolve_ic_lora_output_fps(source_fps, req.params.fps),
            "controlType": control_type,
            "audioMode": audio_mode,
            "loras": loras,
        }
    )
    return recipe, item, stored


def resolve_ic_lora_sheet_create(
    recipe_id: str,
    req: CreateIcLoraSheetRequest,
    *,
    catalog: LoraCatalogProvider,
    models_dir: Path,
    hf_authenticated: bool,
) -> tuple[IcLoraRecipe, IcLoraCatalogItem, IcLoraSynthesizedStoredParams]:
    """Validate an image-list recipe create. The sheet loops at the selected rate, the catalog rate when none."""
    recipe, item = _recipe_item(recipe_id, catalog, kind="sheet")
    assert item.input is not None and item.input.max_count is not None
    if len(req.inputs.images) > item.input.max_count:
        raise RecipeCreateError(
            f"'{item.id}' takes at most {item.input.max_count} images.",
            code="INVALID_GENERATION_SPEC",
        )
    audio_mode, loras = _resolve_checkpoint(
        item, req.params, models_dir=models_dir, hf_authenticated=hf_authenticated
    )
    stored = IcLoraSynthesizedStoredParams.model_validate(
        {
            **req.params.model_dump(exclude={"scale", "variantId"}),
            "fps": _catalog_fps(item, req.params.fps),
            "audioMode": audio_mode,
            "loras": loras,
        }
    )
    return recipe, item, stored


def resolve_ic_lora_track_create(
    recipe_id: str,
    req: CreateIcLoraTrackRequest,
    *,
    catalog: LoraCatalogProvider,
    models_dir: Path,
    hf_authenticated: bool,
    image_size: tuple[int, int],
) -> tuple[IcLoraRecipe, IcLoraCatalogItem, IcLoraSynthesizedStoredParams]:
    """Validate a Motion Track create. The run stores the ratio closest to the image."""
    recipe, item = _recipe_item(recipe_id, catalog, kind="track")
    if any(
        path.startHoldSec + path.endHoldSec >= req.params.duration
        for path in req.inputs.paths
    ):
        raise RecipeCreateError(
            "The pauses must leave time for the path.", code="INVALID_GENERATION_SPEC"
        )
    if image_size[1] <= 0:
        raise RecipeCreateError(
            "The start frame has no height.", code="INVALID_GENERATION_SPEC"
        )
    audio_mode, loras = _resolve_checkpoint(
        item, req.params, models_dir=models_dir, hf_authenticated=hf_authenticated
    )
    stored = IcLoraSynthesizedStoredParams.model_validate(
        {
            **req.params.model_dump(exclude={"scale", "variantId"}),
            "fps": _catalog_fps(item, req.params.fps),
            "aspectRatio": cast(
                LTXVideoGenAspectRatio,
                find_closest_aspect_ratio_option(*image_size, get_args(LTXVideoGenAspectRatio)),
            ),
            "audioMode": audio_mode,
            "loras": loras,
        }
    )
    return recipe, item, stored


def resolve_synthesized_ic_lora_create(
    recipe_id: str,
    req: CreateIcLoraSheetRequest | CreateIcLoraTrackRequest,
    *,
    catalog: LoraCatalogProvider,
    models_dir: Path,
    hf_authenticated: bool,
    require_image: Callable[[str], tuple[int, int]],
) -> tuple[IcLoraRecipe, IcLoraCatalogItem, IcLoraSynthesizedStoredParams]:
    """Validate a synthesized IC-LoRA create (a sheet or a track) and its image inputs."""
    if isinstance(req, CreateIcLoraSheetRequest):
        for ref in req.inputs.images:
            require_image(ref.assetId)
        return resolve_ic_lora_sheet_create(
            recipe_id,
            req,
            catalog=catalog,
            models_dir=models_dir,
            hf_authenticated=hf_authenticated,
        )
    image_size = require_image(req.inputs.image.assetId)
    return resolve_ic_lora_track_create(
        recipe_id,
        req,
        catalog=catalog,
        models_dir=models_dir,
        hf_authenticated=hf_authenticated,
        image_size=image_size,
    )
