"""LoRA recipe registry — the backend half of the io LoRA product config.

One table keyed by the **web Explore id** (also the io route id and the enqueued
``feature``). Each row maps that id to a catalog adapter plus the metadata the
create path and the queued executor need: the generation ``mode``, the GPU
``devices`` the adapter is known to run on, and the fixed prompt scaffold applied
after enhancement. (Whether a recipe is surfaced on Home is a frontend concern —
see ``listed`` in ``shared/lora-recipes.json`` / ``frontend/lib/lora-recipes.ts``.)

Unmapped ids resolve to ``None`` — the create path turns that into a typed
``LORA_UNKNOWN`` miss and never guesses a filesystem path. The frontend keeps a
mirrored registry (``frontend/lib/lora-recipes.ts``) for Home/router/Explore; the
shared id/catalogId/mode/devices contract is pinned on both sides against
``shared/lora-recipes.json`` by tests. Adding a template is a row here plus the
matching frontend row.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from _routes._errors import HTTPError
from api_types import (
    CreateLoraRecipeRequest,
    ImageToVideoInputs,
    ImageToVideoRecipeParams,
    LTXLocalModelId,
    LoraCatalogItem,
    LoraEntry,
    TextToVideoParams,
)
from runtime_config.model_download_specs import (
    catalog_item_visible_for_installed_ltx,
    downloaded_lora_variant_ids,
    is_lora_downloaded,
    ltx_catalog_family_for_model,
    resolve_lora_path,
)
from services.lora_catalog import LoraCatalogProvider

LoraRecipeMode = Literal["t2v", "i2v", "a2v"]
LoraRecipeDevice = Literal["cuda", "mps"]


@dataclass(frozen=True, slots=True)
class LoraRecipe:
    """One io LoRA product, keyed by its web Explore id.

    ``prompt_template`` is the fixed scaffold (with a ``{prompt}`` placeholder)
    applied to the generation prompt after enhancement. It is deliberately *not*
    the catalog ``prompt_template`` (that is the IC-LoRA LLM template-fill path).
    The wrap is idempotent — skipped when the scaffold prefix is already present
    (see ``apply_recipe_prompt_template``). Explore auto-enhance off also skips
    the wrap for typed prompts so Generate stays as-typed; a manual Enhance
    still wraps. The trigger phrase lives on the catalog item and is enforced
    by prompt enhancement, not stored here.
    """

    recipe_id: str
    catalog_id: str
    mode: LoraRecipeMode
    devices: tuple[LoraRecipeDevice, ...]
    prompt_template: str | None = None
    requires_end_frame: bool = False


# Web Explore id -> recipe. Order matches Home LoRAs row / shared/lora-recipes.json.
LORA_RECIPES: dict[str, LoraRecipe] = {
    "cozy-felt": LoraRecipe(
        recipe_id="cozy-felt",
        catalog_id="cozy-felt-style",
        mode="t2v",
        devices=("cuda", "mps"),
        prompt_template=(
            "F3ltCut0u7 handcrafted felt cutout cinematic diorama style, "
            "everything constructed from soft layered felt with visible stitched "
            "seams, fuzzy fibers, plush rounded edges, and embroidered details. "
            "{prompt}"
        ),
    ),
    "claymation": LoraRecipe(
        recipe_id="claymation",
        catalog_id="claymation-style",
        mode="t2v",
        devices=("cuda", "mps"),
        prompt_template="Clay animation, claymation {prompt}",
    ),
    "fantasy-painterly": LoraRecipe(
        recipe_id="fantasy-painterly",
        catalog_id="fantasy-painterly-style",
        mode="t2v",
        devices=("cuda", "mps"),
        prompt_template="D4rkP41nt3r, fantasy painterly style {prompt}",
    ),
    "paper-cut-out-style": LoraRecipe(
        recipe_id="paper-cut-out-style",
        catalog_id="paper-cutout-style",
        mode="t2v",
        devices=("cuda", "mps"),
        prompt_template=(
            "Pap3rCut0u7 {prompt}.\n Everything is handcrafted from layered "
            "colored paper with visible cut edges, stacked cardstock, subtle "
            "paper curl, textured paper fibers, hand-cut imperfections, soft "
            "layered shadows, and a miniature paper diorama aesthetic"
        ),
    ),
    "cinemagraph": LoraRecipe(
        recipe_id="cinemagraph",
        catalog_id="cinemagraph-motion",
        mode="i2v",
        devices=("cuda", "mps"),
        prompt_template=(
            "CINEMAGRAPH_MOTION, tripod locked-off static camera, zero camera "
            "movement, {prompt}, seamless natural loop."
        ),
    ),
    "jib-up": LoraRecipe(
        recipe_id="jib-up",
        catalog_id="jib-up",
        mode="i2v",
        devices=("cuda", "mps"),
        prompt_template=(
            "A smooth jib-up camera move rising vertically, revealing more of "
            "what sits above the initial frame. {prompt}"
        ),
    ),
    "jib-down": LoraRecipe(
        recipe_id="jib-down",
        catalog_id="jib-down",
        mode="i2v",
        devices=("cuda", "mps"),
        prompt_template=(
            "A smooth jib-down camera move descending vertically, revealing more "
            "of what sits lower in the scene. {prompt}"
        ),
    ),
    "dolly-in": LoraRecipe(
        recipe_id="dolly-in",
        catalog_id="dolly-in",
        mode="i2v",
        devices=("cuda", "mps"),
        prompt_template=(
            "A smooth dolly-in camera move pushing forward toward the subject, "
            "increasing subject scale and reducing background prominence. {prompt}"
        ),
    ),
    "dolly-out": LoraRecipe(
        recipe_id="dolly-out",
        catalog_id="dolly-out",
        mode="i2v",
        devices=("cuda", "mps"),
        prompt_template=(
            "A smooth dolly-out camera move pulling backward away from the "
            "subject, revealing more of the environment. {prompt}"
        ),
    ),
    "fpv-motion": LoraRecipe(
        recipe_id="fpv-motion",
        catalog_id="fpv-motion",
        mode="t2v",
        devices=("cuda", "mps"),
        prompt_template="FPV footage, {prompt},",
    ),
    "openwheel-t-cam": LoraRecipe(
        recipe_id="openwheel-t-cam",
        catalog_id="openwheel-tcam-style",
        mode="t2v",
        devices=("cuda", "mps"),
        prompt_template="T-cam onboard view {prompt}",
    ),
    "transition": LoraRecipe(
        recipe_id="transition",
        catalog_id="transition",
        mode="i2v",
        devices=("cuda", "mps"),
        prompt_template="{prompt} zhuanchang",
        requires_end_frame=True,
    ),
    "vbvr": LoraRecipe(
        recipe_id="vbvr",
        catalog_id="vbvr",
        mode="i2v",
        devices=("cuda", "mps"),
    ),
}


def get_lora_recipe(recipe_id: str) -> LoraRecipe | None:
    """The recipe for a web Explore id, or ``None`` for an unmapped id."""
    return LORA_RECIPES.get(recipe_id)


def t2v_lora_recipes() -> list[LoraRecipe]:
    """Recipes that run on the queued text-to-video executor."""
    return [recipe for recipe in LORA_RECIPES.values() if recipe.mode == "t2v"]


def i2v_lora_recipes() -> list[LoraRecipe]:
    """Recipes that run on the queued image-to-video executor."""
    return [recipe for recipe in LORA_RECIPES.values() if recipe.mode == "i2v"]


def queued_lora_recipes() -> list[LoraRecipe]:
    """Recipes that have a queued executor (t2v or i2v). ``a2v`` is still unsupported."""
    return [recipe for recipe in LORA_RECIPES.values() if recipe.mode in ("t2v", "i2v")]


def apply_recipe_prompt_template(recipe: LoraRecipe, prompt: str) -> str:
    """Wrap ``prompt`` in the recipe's scaffold, after enhancement.

    Skipped when the recipe has no template, when the template is malformed (no
    ``{prompt}`` placeholder — fail closed and return the prompt untouched rather
    than drop the user's scene), or when the prompt already *starts with* the
    scaffold prefix (an idempotent re-wrap).

    The guard checks ``startswith``, not a substring: generation enhancement is
    catalog-aware, and the adapter's prompting example embeds this same scaffold
    sentence, so an enhanced rewrite can contain it *mid-prompt*. A substring
    check would then skip the wrap and the style lock would never lead the
    prompt — so the scaffold is prepended unless it is already at the front.
    """
    template = recipe.prompt_template
    if not template or "{prompt}" not in template:
        return prompt
    prefix, _, suffix = template.partition("{prompt}")
    scaffold_prefix = prefix.strip()
    scaffold_suffix = suffix.strip()
    lowered = prompt.lower()
    if scaffold_prefix and lowered.startswith(scaffold_prefix.lower()):
        return prompt
    if not scaffold_prefix and scaffold_suffix and lowered.endswith(scaffold_suffix.lower()):
        return prompt
    return template.replace("{prompt}", prompt, 1)


def make_recipe_prompt_wrap(recipe: LoraRecipe) -> Callable[[str], str]:
    """A ``(prompt) -> prompt`` wrap the queued executor applies post-enhance.

    The executor always passes this wrap; ``generate_local_reserved`` decides
    whether to apply it (skipped for typed Explore jobs when auto-enhance is off).
    """
    return lambda prompt: apply_recipe_prompt_template(recipe, prompt)


def require_catalog_runnable(
    item: LoraCatalogItem,
    *,
    models_dir: Path,
    hf_authenticated: bool,
    hf_code: str,
    model_code: str,
    local_model_id: LTXLocalModelId | None = None,
) -> None:
    """The catalog gates shared by a style LoRA and a Home IC-LoRA.

    A missing Hugging Face login or an unsupported installed family fails before enqueue.
    When ``local_model_id`` is set, the job's own family must be one this entry supports.
    The caller still checks that the weights file itself is on disk.
    """
    if item.requires_hf_login and not hf_authenticated:
        raise RecipeCreateError("HuggingFace sign-in required", code=hf_code)
    if not catalog_item_visible_for_installed_ltx(item.supported_models, models_dir):
        raise RecipeCreateError(
            f"'{item.id}' is not available with the installed LTX models",
            code=model_code,
        )
    if local_model_id is not None and not item.supports_family(
        ltx_catalog_family_for_model(local_model_id)
    ):
        raise RecipeCreateError(
            f"'{item.id}' does not support the selected LTX model",
            code=model_code,
        )


class RecipeCreateError(Exception):
    """A typed create-time miss for a LoRA recipe.

    Carries the wire ``code`` (``LORA_UNKNOWN``/``LORA_UNSUPPORTED_MODE``/…) so
    the create route can map it to a 422 without the recipe-resolution logic
    knowing about HTTP.
    """

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


def resolve_recipe_create(
    recipe_id: str,
    req: CreateLoraRecipeRequest,
    *,
    catalog: LoraCatalogProvider,
    models_dir: Path,
    device: str | None,
    hf_authenticated: bool,
) -> tuple[str, TextToVideoParams | ImageToVideoRecipeParams, ImageToVideoInputs | None]:
    """Validate a recipe create and hydrate its queued params.

    Returns ``(feature, params, inputs)`` — the Explore id to enqueue under, the
    resolved params with the ``loras`` entry injected (empty ``ref`` until
    execute), and ``inputs`` (a start frame for i2v recipes, else ``None``).
    Every miss raises :class:`RecipeCreateError` *before* the caller enqueues
    anything, so nothing is inserted and no GPU slot is taken on a bad request.
    ``device`` is ``None`` when the caller has no runtime config to gate against
    (the check is skipped, matching the plain t2v/i2v queued behavior on
    unsupported hardware).
    """
    recipe = get_lora_recipe(recipe_id)
    if recipe is None:
        raise RecipeCreateError(f"Unknown LoRA recipe: {recipe_id}", code="LORA_UNKNOWN")
    if recipe.mode not in ("t2v", "i2v"):
        raise RecipeCreateError(
            f"LoRA recipe '{recipe_id}' mode '{recipe.mode}' is not supported yet",
            code="LORA_UNSUPPORTED_MODE",
        )
    if device is not None and device not in recipe.devices:
        raise RecipeCreateError(
            f"LoRA recipe '{recipe_id}' does not support device '{device}'",
            code="LORA_UNSUPPORTED_DEVICE",
        )

    params = req.params
    catalog_item = catalog.get_lora(recipe.catalog_id)
    if catalog_item is None:
        # A mapped recipe whose catalog id has no entry is a config error, but it is
        # still "no adapter to resolve" from the client's side.
        raise RecipeCreateError(
            f"Catalog id '{recipe.catalog_id}' for recipe '{recipe_id}' not found",
            code="LORA_UNKNOWN",
        )
    if params.catalogId != recipe.catalog_id:
        raise RecipeCreateError(
            f"catalogId '{params.catalogId}' does not match recipe '{recipe_id}'",
            code="LORA_UNKNOWN",
        )
    require_catalog_runnable(
        catalog_item,
        models_dir=models_dir,
        hf_authenticated=hf_authenticated,
        hf_code="LORA_HF_AUTH_REQUIRED",
        model_code="LORA_UNSUPPORTED_MODEL",
    )

    installed = downloaded_lora_variant_ids(
        models_dir,
        recipe.catalog_id,
        [(variant.id, variant.filename) for variant in catalog_item.download.variants],
    )
    variant = catalog_item.download.variant_for_run(params.variantId, installed)
    if variant is None:
        raise RecipeCreateError(
            f"Unknown variant '{params.variantId}'", code="UNKNOWN_DOWNLOAD_VARIANT"
        )
    if not is_lora_downloaded(models_dir, recipe.catalog_id, variant.filename):
        raise RecipeCreateError(
            f"LoRA '{recipe.catalog_id}' is not downloaded", code="LORA_NOT_DOWNLOADED"
        )

    loras = [
        LoraEntry(
            ref="",
            scale=params.scale,
            catalogId=recipe.catalog_id,
            displayName=catalog_item.name,
            variantId=variant.id,
        )
    ]
    # Store catalogId + scale only; the weights path is resolved at execution by
    # make_recipe_lora_resolver (no server path persisted, and a retry re-resolves).
    if recipe.mode == "i2v":
        if req.inputs is None:
            raise RecipeCreateError("Start Frame is required.", code="INVALID_GENERATION_SPEC")
        if recipe.requires_end_frame and req.inputs.endFrame is None:
            raise RecipeCreateError("End Frame is required.", code="INVALID_GENERATION_SPEC")
        if req.inputs.endFrame is not None and not recipe.requires_end_frame:
            raise RecipeCreateError(
                f"LoRA recipe '{recipe_id}' does not take an end frame",
                code="INVALID_GENERATION_SPEC",
            )
        i2v_params = ImageToVideoRecipeParams(
            prompt=params.prompt,
            resolution=params.resolution,
            model=params.model,
            cameraMotion=params.cameraMotion,
            negativePrompt=params.negativePrompt,
            duration=params.duration,
            fps=params.fps,
            seed=params.seed,
            aspectRatio=params.aspectRatio,
            promptProvenance=params.promptProvenance,
            loras=loras,
        )
        return (
            recipe.recipe_id,
            i2v_params,
            ImageToVideoInputs(
                startFrame=req.inputs.startFrame,
                endFrame=req.inputs.endFrame,
            ),
        )

    if req.inputs is not None:
        raise RecipeCreateError(
            f"LoRA recipe '{recipe_id}' does not take a start frame",
            code="INVALID_GENERATION_SPEC",
        )
    if params.aspectRatio == "auto":
        raise RecipeCreateError(
            f"LoRA recipe '{recipe_id}' does not support aspectRatio 'auto'",
            code="INVALID_GENERATION_SPEC",
        )
    tt_params = TextToVideoParams(
        prompt=params.prompt,
        resolution=params.resolution,
        model=params.model,
        cameraMotion=params.cameraMotion,
        negativePrompt=params.negativePrompt,
        duration=params.duration,
        fps=params.fps,
        seed=params.seed,
        aspectRatio=params.aspectRatio,
        promptProvenance=params.promptProvenance,
        loras=loras,
    )
    return recipe.recipe_id, tt_params, None


def make_recipe_lora_resolver(
    recipe: LoraRecipe,
    *,
    catalog: LoraCatalogProvider,
    models_dir: Callable[[], Path],
) -> Callable[[list[LoraEntry]], list[LoraEntry]]:
    """Hydrate a recipe LoRA's ``ref`` from its catalog id at execution time.

    The stored spec carries only ``catalogId`` + ``scale`` (empty ``ref``); this resolver
    fills the absolute weights path against the *current* models dir just before the job
    runs, re-checking that the LoRA is still downloaded. It runs inside the recipe's
    executor, so a retry re-resolves rather than replaying a path captured at create time.
    Entries whose ``catalogId`` isn't this recipe's are passed through untouched.

    Raises :class:`HTTPError` (mapped to a capability failure by the executor) when the
    catalog entry vanished or the weights are no longer on disk.
    """

    def resolve(loras: list[LoraEntry]) -> list[LoraEntry]:
        resolved: list[LoraEntry] = []
        for entry in loras:
            if entry.catalogId != recipe.catalog_id:
                resolved.append(entry)
                continue
            item = catalog.get_lora(recipe.catalog_id)
            if item is None:
                raise HTTPError(
                    409, f"Catalog id '{recipe.catalog_id}' not found", code="LORA_UNKNOWN"
                )
            variant = (
                item.download.resolve_variant(entry.variantId)
                if entry.variantId is not None
                else item.download.legacy_variant()
            )
            if variant is None:
                raise HTTPError(
                    404,
                    "UNKNOWN_DOWNLOAD_VARIANT",
                    code="UNKNOWN_DOWNLOAD_VARIANT",
                )
            base = models_dir()
            if not is_lora_downloaded(base, recipe.catalog_id, variant.filename):
                raise HTTPError(
                    409,
                    f"LoRA '{recipe.catalog_id}' is not downloaded",
                    code="LORA_NOT_DOWNLOADED",
                )
            ref = str(resolve_lora_path(base, recipe.catalog_id, variant.filename))
            resolved.append(entry.model_copy(update={"ref": ref}))
        return resolved

    return resolve
