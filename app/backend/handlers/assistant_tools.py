"""Typed Explore tools. Existing handlers own validation and generation logic."""

import asyncio
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Annotated, Any, Literal, NoReturn, Protocol, Self
from uuid import NAMESPACE_URL, uuid5

from pydantic import AfterValidator, BaseModel, Field, JsonValue
from pydantic_ai import CallDeferred, ModelRetry, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.capabilities.abstract import ValidatedToolArgs, WrapToolExecuteHandler
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import FunctionToolset

from _routes._errors import HTTPError
from assistant_limits import ASSISTANT_LIMITS
from api_types import (
    Asset, AssetCore, AssetListQuery, AssetMetadata, AudioAssetMetadata, AudioToVideoInputs, CreateAudioToVideoParams, CreateAudioToVideoRequest, CreateExtendRequest,
    CreateIcLoraRecipeRequest, CreateIcLoraSheetRequest, CreateImageToVideoRequest, CreateLoraRecipeRequest, CreateRetakeRequest,
    CreateTextToVideoRequest, DownloadedLocalVideoGenerationModelSpecItem,
    DownloadSpec, ExtendInputs, ExtendParams, Generation, GenerationErrorCode, GenerationStatus, IC_LORA_SHEET_MAX_IMAGES,
    IcLoraCatalogItem, IcLoraListResponse, IcLoraRecipeInputs, IcLoraVideoRecipeParams, IcLoraSheetInputs, IcLoraSheetParams,
    ImageToVideoInputs, ImageToVideoParams,
    InputAssetRef, LoraRecipeInputs, LoraRecipeParams, LoraListResponse, LTXVideoGenerationResolutionSpec,
    LTXVideoGenResolution, MediaKind, OfferingId, Origin, PromptProvenance,
    RetakeInputs, RetakeRequestParams, TextToVideoParams, VideoMeta,
)
from handlers.asset_handler import AssetHandler
from handlers.assistant_params import (
    AudioParams, ExtensionParams, ImageParams, IngredientsParams, RecipeInputs, RecipeParams, ReplacementParams,
    TextParams, VideoParams, VideoRecipeParams, api_params,
)
from handlers.queued_generation_handler import AnalyticsClient, AnalyticsRequestedBy, QueuedGenerationHandler
from services.assistant_run import AssistantRun
from services.features.ic_lora_recipes import get_ic_lora_recipe, queued_ic_lora_recipes
from services.features.lora_recipes import LoraRecipe, LoraRecipeMode, get_lora_recipe, queued_lora_recipes
from services.prompt_enhancement.system_prompt import (
    enforce_trigger_placement, fill_prompt_template, render_catalog_item_block,
)


def _local_time(epoch_ms: int) -> str:
    return datetime.fromtimestamp(epoch_ms / 1000).astimezone().strftime("%Y-%m-%d %H:%M")


# What the model reads. Times are given ready to use: models misread and mis-subtract epoch milliseconds.
class AssistantAssetCore(BaseModel):
    id: str
    media_kind: MediaKind
    origin: Origin
    mime_type: str
    name: str
    metadata: AssetMetadata
    created: str

    @classmethod
    def of(cls, asset: AssetCore) -> Self:
        return cls.model_validate({**asset.model_dump(exclude={"created_at"}), "created": _local_time(asset.created_at)})


class AssistantJobSource(BaseModel):
    id: str
    name: str


class AssistantGeneration(BaseModel):
    id: str
    feature: str
    status: GenerationStatus
    error_code: GenerationErrorCode | None = None
    attempt_count: int
    # Generation time only, not the wait in the queue. None until the job finishes.
    took_seconds: float | None = None
    finished: str | None = None
    # The asset the job worked from (the video it extended, the start frame); "unavailable" once deleted.
    source: AssistantJobSource | Literal["unavailable"] | None = None
    # The first output's length and frame size, flattened so replies need not dig into outputs.
    clip_seconds: float | None = None
    size: str | None = None
    outputs: list[AssistantAssetCore]
    # Something about how the job was queued that the user should hear.
    note: str | None = None
    # On a just-queued job: the model often replies here and drops a later step that needs this clip.
    next: str | None = None


WAIT_REMINDER = "If the user asked for a later step on this clip, call wait_for_generation with this generation_id now."


class AssistantAssetSource(BaseModel):
    """The job that generated an asset: reuse these to recreate or restyle it."""

    generation_id: str
    feature: str
    params: dict[str, JsonValue]


class AssistantAsset(AssistantAssetCore):
    source: AssistantAssetSource | None = None


# Settings only; inputs and LoRA selections reference other assets and stay out.
_SOURCE_PARAMS = ("prompt", "negativePrompt", "model", "resolution", "duration", "fps", "aspectRatio", "cameraMotion")


def describe_asset(assets: AssetHandler, generations: QueuedGenerationHandler, asset_id: str) -> AssistantAsset:
    described = AssistantAsset.model_validate(AssistantAssetCore.of(AssetCore.model_validate(assets.get_asset(asset_id).model_dump())).model_dump())
    producer_id = assets.get_producer_id(asset_id)
    if producer_id is None:
        return described
    try:
        generation = generations.get_generation(producer_id)
    except HTTPError as exc:
        if exc.status_code != 404:
            raise
        return described
    params = generation.spec.get("params")
    if isinstance(params, dict):
        described.source = AssistantAssetSource(
            generation_id=generation.id, feature=generation.feature,
            params={key: params[key] for key in _SOURCE_PARAMS if key in params},
        )
    return described


class AssistantAssetPage(BaseModel):
    items: list[AssistantAssetCore]
    next_cursor: str | None


class AssistantAssetQuery(BaseModel):
    media_kind: MediaKind | None = None
    sort: Literal["newest", "oldest"] = "newest"
    q: str | None = None
    cursor: str | None = None
    limit: int = Field(default=100, ge=1, le=200)

    def api_query(self, limit: int) -> AssetListQuery:
        return AssetListQuery(
            media_kind=self.media_kind, sort="created_at-desc" if self.sort == "newest" else "created_at-asc",
            q=self.q, cursor=self.cursor, limit=min(self.limit, limit),
        )


class AssistantRecipe(BaseModel):
    recipe_id: str
    mode: LoraRecipeMode
    requires_end_frame: bool
    name: str
    description: str
    downloaded: bool
    requires_hf_login: bool


@dataclass
class AssistantDeps:
    run: AssistantRun
    generations: QueuedGenerationHandler
    assets: AssetHandler
    models: Callable[[], list[DownloadedLocalVideoGenerationModelSpecItem]]
    default_model: Callable[[], OfferingId | None]
    loras: Callable[[], LoraListResponse]
    ic_loras: Callable[[], IcLoraListResponse]
    # False when the run was stopped or deleted, so nothing was saved.
    save: Callable[[], bool]
    check_stopped: Callable[[], None]
    view: Callable[[str, str], None]
    progress: Callable[[str], None] = lambda _: None
    auto_enhance: Callable[[], bool] = lambda: False
    chat_generation_ids: Callable[[], set[str]] = lambda: set()


tools = FunctionToolset[AssistantDeps](max_retries=2, sequential=True)
# Submit tools run in parallel and share one budget per tool name.
SUBMIT_RETRIES = 3
RECIPE_GUARDED_RETRIES = SUBMIT_RETRIES + 1
tool_hooks = Hooks[AssistantDeps]()
_JOB_CAP = f"A request can queue at most {ASSISTANT_LIMITS.jobs} jobs."


def _within_job_cap(job_number: int) -> int:
    # Not `le`: its generic message does not tell the model to report the skipped jobs.
    if job_number > ASSISTANT_LIMITS.jobs:
        raise ValueError(f"{_JOB_CAP} Do not queue more; tell the user how many were not queued.")
    return job_number


JobNumber = Annotated[int, Field(ge=1, strict=True, json_schema_extra={"maximum": ASSISTANT_LIMITS.jobs},
                                 description=f"One 1..N sequence across requested jobs. {_JOB_CAP} Reuse unchanged on retries."),
                      AfterValidator(_within_job_cap)]


class GenerationCreator[T](Protocol):
    def __call__(self, request: T, /, *, generation_id: str, analytics_client: AnalyticsClient, analytics_requested_by: AnalyticsRequestedBy) -> Generation: ...


@tool_hooks.on.tool_execute
async def execute_tool(ctx: RunContext[AssistantDeps], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs, handler: WrapToolExecuteHandler) -> Any:
    await asyncio.to_thread(ctx.deps.check_stopped)
    labels = {
        "list_video_models": "Checking installed models", "list_lora_recipes": "Checking available styles",
        "get_recipe_guidance": "Reading style guidance", "list_video_recipes": "Checking video effects",
        "run_video_recipe": "Queueing a video effect",
        "list_assets": "Finding assets", "get_asset": "Reading asset details", "show_assets": "Showing assets",
        "list_generations": "Checking recent videos", "get_generation_status": "Checking video progress",
        "wait_for_generation": "Waiting for the video", "view_asset": "Looking at an image",
        "generate_video": "Queueing a video", "image_to_video": "Queueing a video from an image",
        "audio_to_video": "Queueing a video from audio", "run_lora_recipe": "Queueing a styled video",
        "extend_video": "Queueing an extension", "retake_video": "Queueing a retake",
        "retry_interrupted_generation": "Retrying the video",
        "retry_failed_job": "Retrying the video",
        "extract_frame": "Extracting a frame", "run_ingredients": "Queueing an Ingredients video",
    }
    ctx.deps.progress(labels.get(call.tool_name, "Working"))
    worker = asyncio.ensure_future(handler(args))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        # A synchronous submission may already be running in the thread pool.
        # Let its post-enqueue stop check finish before releasing local weights.
        await asyncio.gather(worker, return_exceptions=True)
        raise


_SOURCE_INPUTS = ("video", "startFrame", "audio")


def _job_source(generation: Generation, assets: AssetHandler) -> AssistantJobSource | Literal["unavailable"] | None:
    inputs = generation.spec.get("inputs")
    ref = next((inputs[key] for key in _SOURCE_INPUTS if isinstance(inputs.get(key), dict)), None) if isinstance(inputs, dict) else None
    asset_id = ref.get("assetId") if isinstance(ref, dict) else None
    if not isinstance(asset_id, str):
        return None
    try:
        asset = assets.get_asset(asset_id)
    except HTTPError as exc:
        if exc.status_code != 404:
            raise
        return "unavailable"
    return AssistantJobSource(id=asset.id, name=asset.name)


def generation_result(generation: Generation, assets: AssetHandler) -> AssistantGeneration:
    # Only metadata and asset IDs, never filesystem paths; view_asset supplies pixels.
    started, finished = generation.started_at, generation.finished_at
    outputs = [AssistantAssetCore.of(AssetCore.model_validate(output.model_dump())) for output in generation.outputs]
    meta = next((output.metadata.metadata for output in outputs if not isinstance(output.metadata, AudioAssetMetadata)), None)
    return AssistantGeneration(
        id=generation.id, feature=generation.feature, status=generation.status, error_code=generation.error_code,
        attempt_count=generation.attempt_count,
        took_seconds=round((finished - started) / 1000, 1) if started is not None and finished is not None else None,
        finished=_local_time(finished) if finished is not None else None,
        source=_job_source(generation, assets),
        clip_seconds=round(meta.durationMs / 1000, 2) if isinstance(meta, VideoMeta) else None,
        size=f"{meta.width}x{meta.height}" if meta is not None else None,
        outputs=outputs,
    )


def _read_generation(deps: AssistantDeps, generation_id: str) -> Generation:
    try:
        return deps.generations.get_generation(generation_id)
    except HTTPError as exc:
        if exc.status_code != 404:
            raise
        try:
            deps.assets.get_asset(generation_id, tombstone_missing=False)
        except HTTPError:
            raise ModelRetry("Generation unavailable. Use list_generations to find a current ID.") from exc
        raise ModelRetry("That is an asset ID, not a job ID. Use get_asset, or list_generations for jobs.") from exc


def _model(deps: AssistantDeps, model: OfferingId, capability: Literal["t2v", "i2v", "a2v", "ic_lora", "retake", "extend"]) -> DownloadedLocalVideoGenerationModelSpecItem:
    selected = next((item for item in deps.models() if item.model == model), None)
    if selected is None or selected.spec.capabilities is None or not getattr(selected.spec.capabilities, capability):
        raise ModelRetry("That model is not installed or cannot run this feature locally. Check list_video_models.")
    return selected


def _provenance(deps: AssistantDeps) -> PromptProvenance:
    # Off: the scene goes as written, but a recipe keeps its wrap, which "typed" would drop.
    return "typed" if deps.auto_enhance() else "enhanced"


def _listed(values: Iterable[object]) -> str:
    return ", ".join(str(value) for value in values)


def _video_settings(deps: AssistantDeps, params: VideoParams, capability: Literal["t2v", "i2v"], aspect_ratio: str) -> None:
    selected = _model(deps, params.model, capability)
    supported = selected.spec.supported_resolutions_durations
    resolution = supported.get(params.resolution)
    if resolution is None:
        raise ModelRetry(f"{params.model} does not support {params.resolution}. Supported: {_listed(supported)}.")
    durations = resolution.fps_to_durations.get(params.fps)
    if not durations:
        raise ModelRetry(f"{params.model} at {params.resolution} does not support {params.fps} fps. Supported: {_listed(resolution.fps_to_durations)}.")
    at = f"At {params.resolution}/{params.fps} fps it supports durations {_listed(durations)}"
    if params.duration is None:
        if selected.spec.capabilities is None or not selected.spec.capabilities.auto_duration:
            raise ModelRetry(f"{params.model} needs a duration. {at}.")
    elif params.duration not in durations:
        raise ModelRetry(f"{params.model} cannot make {params.duration} s. {at}. You sent duration={params.duration}.")
    if params.cameraMotion != "none" and selected.spec.capabilities is not None and not selected.spec.capabilities.camera_motion:
        raise ModelRetry(f"{params.model} has no camera motion. Leave cameraMotion as none, or describe the move in the prompt.")
    if aspect_ratio == "auto" and capability == "t2v":
        raise ModelRetry("auto follows a start frame; a text-only clip needs an explicit aspectRatio.")
    if aspect_ratio != "auto" and aspect_ratio not in resolution.aspect_ratios:
        raise ModelRetry(f"aspectRatio {aspect_ratio} is not available at {params.resolution}. Available: {', '.join(resolution.aspect_ratios)}.")


def _audio_settings(deps: AssistantDeps, params: AudioParams, audio_seconds: float) -> None:
    selected = _model(deps, params.model, "a2v")
    cells = selected.spec.a2v_supported_resolutions_durations or {}
    longest = {name: max(cell.fps_to_durations.get(params.fps) or [0]) for name, cell in cells.items()}
    resolution = cells.get(params.resolution)
    if resolution is None or not longest[params.resolution]:
        raise ModelRetry(f"Audio-to-video is not available at {params.resolution}. Available: {', '.join(name for name, top in longest.items() if top)}.")
    if params.cameraMotion != "none" and selected.spec.capabilities is not None and not selected.spec.capabilities.camera_motion:
        raise ModelRetry(f"{params.model} has no camera motion. Leave cameraMotion as none, or describe the move in the prompt.")
    if params.aspectRatio != "auto" and params.aspectRatio not in resolution.aspect_ratios:
        raise ModelRetry(f"aspectRatio {params.aspectRatio} is not available at {params.resolution}. Available: {', '.join(resolution.aspect_ratios)}.")
    if audio_seconds < 2:
        raise ModelRetry(f"The audio is {audio_seconds:.1f} s; audio-to-video needs at least 2 s. Tell the user.")
    # Same 0.1 s tolerance as the queue, which would cut audio within another resolution's limit to this one's.
    if audio_seconds < longest[params.resolution] + 0.1:
        return
    covering = [name for name, top in longest.items() if audio_seconds < top + 0.1]
    raise ModelRetry(
        f"The audio is {audio_seconds:.1f} s, longer than {params.resolution} allows ({longest[params.resolution]:g} s). "
        + (f"Resolutions that cover all of it: {', '.join(covering)}. Ask the user which to use, or to trim the audio."
           if covering else "No resolution covers it. Ask the user to trim the audio.")
    )


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _mentions(words: list[str], label: str) -> bool:
    """Whether the words contain the label, ignoring spacing and punctuation ("paper cutout", "Paper Cut-Out")."""
    phrase = _words(label)
    while phrase and phrase[-1] in ("style", "motion"):
        phrase.pop()
    target = "".join(phrase)
    for start in range(len(words) if target else 0):
        joined = ""
        for word in words[start:]:
            joined += word
            if joined == target:
                return True
            if len(joined) >= len(target):
                break
    return False


# Recipe names that are also ordinary prompt words ("a smooth transition", "from day to night", "cooking ingredients").
_COMMON_WORD_RECIPES = frozenset({"transition", "restore", "day-to-night", "water-simulation", "ingredients"})
_RECIPE_CUES = frozenset({"recipe", "effect", "style", "lora", "filter"})


def _attaches_video(deps: AssistantDeps) -> bool:
    for asset_id in deps.run.request.asset_ids:
        try:
            if deps.assets.get_asset(asset_id).metadata.mediaType == "video":
                return True
        except HTTPError:
            continue
    return False


def _named_recipe(deps: AssistantDeps, *, video_only: bool) -> tuple[str, str] | None:
    """The tool and recipe id of the first recipe the user's message names."""
    words = _words(deps.run.request.text)
    cued = not _RECIPE_CUES.isdisjoint(words)
    candidates: list[tuple[str, str, str]] = []
    if not video_only:
        lora_names = {item.lora.id: item.lora.name for item in deps.loras().loras}
        candidates += [("run_lora_recipe", recipe.recipe_id, lora_names.get(recipe.catalog_id, "")) for recipe in queued_lora_recipes()]
    ic_names = {item.ic_lora.id: item.ic_lora.name for item in deps.ic_loras().ic_loras}
    if not video_only:
        candidates += [("run_ingredients", recipe.recipe_id, ic_names.get(recipe.catalog_id, "")) for recipe in queued_ic_lora_recipes() if recipe.kind == "sheet"]
    # A video recipe needs a source video, so a text or image request without one cannot run it.
    if video_only or _attaches_video(deps):
        candidates += [("run_video_recipe", recipe.recipe_id, ic_names.get(recipe.catalog_id, "")) for recipe in queued_ic_lora_recipes() if recipe.kind == "video"]
    for tool, recipe_id, name in candidates:
        if recipe_id in _COMMON_WORD_RECIPES and not cued:
            continue
        if _mentions(words, recipe_id) or _mentions(words, name):
            return tool, recipe_id
    return None


def _recipe_guard(deps: AssistantDeps, *, video_only: bool = False) -> None:
    # Text rules alone did not stop small models writing a recipe's style into a plain prompt.
    # Once per request, so a user who wants no recipe gets the plain job on the next call.
    # Extend and retake name the clip's style ("extend the claymation clip"), so only video recipes count.
    # The tools that call this get one extra retry, so the nudge leaves the budget for real errors.
    if deps.run.recipe_nudged or (named := _named_recipe(deps, video_only=video_only)) is None:
        return
    deps.run.recipe_nudged = True
    tool, recipe_id = named
    call = tool if tool == "run_ingredients" else f"{tool}, recipe_id={recipe_id}"
    raise ModelRetry(
        f"The user named the {recipe_id} recipe. Queue it with {call}. "
        "If the user wants no recipe, or this is a job they asked for without it, call this tool again unchanged."
    )


def _recipe_queued(deps: AssistantDeps) -> None:
    # In a mixed request ("one Cozy Felt clip and one plain clip") the recipe is already handled.
    deps.run.recipe_nudged = True
    deps.save()


def _video_recipe_lora(deps: AssistantDeps, recipe_id: str) -> IcLoraCatalogItem:
    recipe = get_ic_lora_recipe(recipe_id)
    ic_lora = None if recipe is None else next(
        (item.ic_lora for item in deps.ic_loras().ic_loras if item.ic_lora.id == recipe.catalog_id), None)
    if ic_lora is None:
        raise ModelRetry("Unknown video recipe. Use list_video_recipes.")
    return ic_lora


def _model_family(model: str) -> str:
    match = re.match(r"ltx-(\d+(?:\.\d+)?)", model, re.IGNORECASE)
    return f"LTX-{match.group(1)}" if match else model


def _variant_for(download: DownloadSpec, installed_ids: list[str], model: str, name: str) -> tuple[str | None, str | None]:
    """Home's choice (`resolveVariantChoice`): the installed variant trained on the model's family, else any installed one.

    Returns the variant id, or None to let the server decide (one variant, or none installed), and a note for the user.
    """
    installed = [variant for variant in download.variants if variant.id in installed_ids]
    if len(download.variants) < 2 or not installed:
        return None, None
    family = _model_family(model)
    # LTX-2 weights run on every current family.
    matching = next((variant for variant in installed if variant.base_model in (family, "LTX-2")), None)
    if matching is not None:
        return matching.id, None
    return installed[0].id, f"Only the {installed[0].label} variant of {name} is installed, so results with {family} may not be optimal. Tell the user."


def _with_trigger(deps: AssistantDeps, recipe: LoraRecipe, prompt: str) -> str:
    """Add the LoRA's trigger when its scaffold lacks it; the assistant skips the enhancer that would."""
    lora = next((item.lora for item in deps.loras().loras if item.lora.id == recipe.catalog_id), None)
    if lora is None or lora.trigger is None or lora.trigger_placement is None:
        return prompt
    if recipe.prompt_template and lora.trigger.lower() in recipe.prompt_template.lower():
        return prompt
    return enforce_trigger_placement(prompt, lora.trigger, lora.trigger_placement)


_RECIPE_BLOCKERS = {
    "LORA_NOT_DOWNLOADED": "This recipe's weights are not downloaded. Tell the user to download them from the recipe in Home or in Settings > Models.",
    "LORA_HF_AUTH_REQUIRED": "This recipe needs a Hugging Face sign-in. Tell the user to sign in under Settings > Models.",
    "LORA_UNSUPPORTED_MODEL": "This recipe does not run on the installed LTX model. Tell the user, and ask whether to use another recipe.",
}


def _submit[T](deps: AssistantDeps, job_number: int, create: GenerationCreator[T], request: T) -> AssistantGeneration:
    deps.check_stopped()
    run = deps.run
    job_key = str(job_number)
    job_id = str(uuid5(NAMESPACE_URL, f"ltx-desktop-assistant:{run.request.request_id}:{job_number}"))
    # Persist ownership before enqueueing. The store enforces identity and inputs,
    # including crash recovery and tombstones for deleted jobs.
    already_owned = job_key in run.jobs
    if already_owned:
        try:
            existing = deps.generations.get_generation(job_id)
        except HTTPError:
            existing = None
        # The store would hand back the cancelled job as if it were queued.
        if existing is not None and existing.status == "cancelled":
            raise ModelRetry(f"The user cancelled job {job_number}. Do not queue it again unless the user asks.")
    run.add_job(job_key, job_id)
    if not deps.save():
        # A job created now would have no row, so no message would show or stop it.
        raise HTTPError(409, "Assistant run stopped.")
    try:
        generation = create(request, generation_id=job_id, analytics_client=run.request.origin, analytics_requested_by="assistant")
    except HTTPError as exc:
        if not already_owned:
            # A failing enqueue can still have committed. Release ownership only
            # when the store positively confirms that no generation exists.
            try:
                deps.generations.get_generation(job_id)
            except HTTPError as lookup_error:
                if lookup_error.status_code == 404:
                    run.release_job(job_key)
                    deps.save()
        if exc.status_code not in (400, 404, 409, 422):
            raise
        if (blocker := _RECIPE_BLOCKERS.get((exc.code or "").removeprefix("IC_"))) is not None:
            raise ModelRetry(f"{blocker} Do not retry, and do not switch to another recipe or a plain video unless they ask.") from exc
        if "already been used" in exc.detail:
            raise ModelRetry("Explore rejected this step: this job number already has different inputs. Reused job numbers must have identical inputs; use a new number only for new requested work.") from exc
        # A coded error carries a message written for users. Uncoded ones (HTTP_400) may quote internals, such as paths.
        reason = exc.code if exc.code.startswith("HTTP_") or exc.detail == exc.code else f"{exc.detail} ({exc.code})"
        raise ModelRetry(f"Explore rejected this step: {reason}. Fix the arguments, or tell the user what is missing.") from exc
    _cancel_if_stopped(deps, generation)
    result = generation_result(generation, deps.assets)
    if generation.status in ("queued", "running"):
        result.next = WAIT_REMINDER
    return result


def _cancel_if_stopped(deps: AssistantDeps, generation: Generation) -> None:
    try:
        deps.check_stopped()
    except HTTPError:
        if generation.id in deps.run.generation_ids() and generation.status in ("queued", "running"):
            deps.generations.cancel_generation(generation.id)
        raise


def _defer(generation: Generation) -> NoReturn:
    raise CallDeferred(metadata={"generation_id": generation.id})


def _video_input(deps: AssistantDeps, generation_id: str | None, video_asset_id: str | None) -> InputAssetRef:
    if (generation_id is None) == (video_asset_id is None):
        raise ModelRetry("Specify exactly one generation_id or video_asset_id from tool results.")
    if generation_id is not None:
        source = _read_generation(deps, generation_id)
        if source.status != "succeeded":
            raise ModelRetry("The source must succeed before editing. Use wait_for_generation for queued/running sources. Never edit failed or cancelled jobs.")
        video = next((asset for asset in source.outputs if asset.media_kind == "video"), None)
        if video is None:
            raise ModelRetry("The source has no video output.")
        return InputAssetRef(assetId=video.id)
    assert video_asset_id is not None
    return InputAssetRef(assetId=video_asset_id)


def _source_seconds(deps: AssistantDeps, source: InputAssetRef, kind: Literal["video", "audio"] = "video") -> float:
    try:
        metadata = deps.assets.get_asset(source.assetId).metadata
    except HTTPError as exc:
        if exc.status_code != 404:
            raise
        raise ModelRetry("Asset unavailable. Ask the user to select/import an existing asset.") from exc
    if metadata.mediaType == "image" or metadata.mediaType != kind:
        raise ModelRetry(f"The source must be {'an audio' if kind == 'audio' else 'a video'} asset.")
    return metadata.metadata.durationMs / 1000


class AssistantVideoModelCapabilities(BaseModel):
    """Only what a tool can set; user LoRAs and multi-keyframe stay out."""

    t2v: bool
    i2v: bool
    a2v: bool
    ic_lora: bool
    retake: bool
    extend: bool
    camera_motion: bool
    auto_duration: bool


class AssistantVideoModel(BaseModel):
    model: OfferingId
    display_name: str
    default: bool
    supported_resolutions_durations: dict[LTXVideoGenResolution, LTXVideoGenerationResolutionSpec]
    a2v_supported_resolutions_durations: dict[LTXVideoGenResolution, LTXVideoGenerationResolutionSpec] | None
    capabilities: AssistantVideoModelCapabilities | None


@tools.tool
def list_video_models(ctx: RunContext[AssistantDeps]) -> list[AssistantVideoModel]:
    """Read installed local model capabilities and valid resolution, duration, FPS and aspect ratio combinations."""
    default = ctx.deps.default_model()
    return [AssistantVideoModel(
        model=item.model, display_name=item.spec.display_name, default=item.model == default,
        supported_resolutions_durations=item.spec.supported_resolutions_durations,
        a2v_supported_resolutions_durations=item.spec.a2v_supported_resolutions_durations,
        capabilities=None if item.spec.capabilities is None
        else AssistantVideoModelCapabilities.model_validate(item.spec.capabilities, from_attributes=True),
    ) for item in ctx.deps.models()]


@tools.tool
def list_lora_recipes(ctx: RunContext[AssistantDeps]) -> list[AssistantRecipe]:
    """Discover Explore style/motion recipes, required frames and download status.

    Missing weights or required HuggingFace sign-in must be handled by the user in the app.
    """
    catalog = {item.lora.id: item for item in ctx.deps.loras().loras}
    return [AssistantRecipe(
        recipe_id=recipe.recipe_id, mode=recipe.mode,
        requires_end_frame=recipe.requires_end_frame,
        name=item.lora.name, description=item.lora.description,
        downloaded=item.lora.download.default_variant().id in item.downloaded_variant_ids,
        requires_hf_login=item.lora.requires_hf_login,
    ) for recipe in queued_lora_recipes() if (item := catalog.get(recipe.catalog_id)) is not None]


class AssistantRecipeGuidance(BaseModel):
    next_step: str
    guidance: str


@tools.tool
def get_recipe_guidance(ctx: RunContext[AssistantDeps], recipe_id: str) -> AssistantRecipeGuidance:
    """How to write a prompt for one style, motion or video recipe, and which tool queues it."""
    if (ic_recipe := get_ic_lora_recipe(recipe_id)) is not None and ic_recipe.kind != "track":
        ic_lora = _video_recipe_lora(ctx.deps, recipe_id)
        if ic_recipe.kind == "sheet":
            return AssistantRecipeGuidance(
                next_step="Queue it with run_ingredients, passing the image asset IDs. Write params.prompt following the guidance below.",
                guidance=render_catalog_item_block(ic_lora),
            )
        next_step = f"Queue it with run_video_recipe, recipe_id={recipe_id}, on the source video."
        if (template := ic_lora.prompt_template) is None:
            return AssistantRecipeGuidance(
                next_step=f"{next_step} Write any prompt in params.prompt, following the guidance below; this recipe takes no template_values.",
                guidance=render_catalog_item_block(ic_lora),
            )
        fields = [
            f"{name}: one of {placeholder.choices}" if placeholder.choices is not None
            else f"{name}: a short phrase from the user's request"
            for name, placeholder in template.placeholders.items()
        ]
        return AssistantRecipeGuidance(
            next_step=(
                f"{next_step} Leave prompt empty and pass template_values with these keys; "
                f"the app builds the prompt from the recipe's template. {'; '.join(fields)}."
            ),
            guidance=f"{render_catalog_item_block(ic_lora)}\nTemplate: {template.template}",
        )
    recipe = get_lora_recipe(recipe_id)
    catalog = {item.lora.id: item.lora for item in ctx.deps.loras().loras}
    lora = catalog.get(recipe.catalog_id) if recipe is not None else None
    if recipe is None or lora is None:
        raise ModelRetry("Unknown recipe. Use list_lora_recipes.")
    # A small model copies a trigger or template into generate_video instead of using the recipe.
    return AssistantRecipeGuidance(
        next_step=(
            f"Queue it with run_lora_recipe, recipe_id={recipe_id}, not generate_video or image_to_video. "
            "The recipe adds its own trigger and style words; describe the scene only."
        ),
        guidance=render_catalog_item_block(lora, include_trigger=False),
    )


class AssistantVideoRecipe(BaseModel):
    recipe_id: str
    name: str
    description: str
    prompt_required: bool
    reference_image: Literal["none", "optional", "required"]
    downloaded: bool
    requires_hf_login: bool


@tools.tool
def list_video_recipes(ctx: RunContext[AssistantDeps]) -> list[AssistantVideoRecipe]:
    """Discover recipes that transform an existing video: day to night, restore, deblur, colourise, clean plate, alpha cut-out and more.

    Sources must be short clips; a longer one is rejected and the user must trim it.
    Missing weights or required HuggingFace sign-in must be handled by the user in the app.
    """
    catalog = {item.ic_lora.id: item for item in ctx.deps.ic_loras().ic_loras}
    return [AssistantVideoRecipe(
        recipe_id=recipe.recipe_id, name=item.ic_lora.name, description=item.ic_lora.description,
        prompt_required=not item.ic_lora.allows_empty_prompt,
        reference_image="required" if item.ic_lora.reference_image_required else "optional" if item.ic_lora.allows_reference_image else "none",
        downloaded=item.downloaded, requires_hf_login=item.ic_lora.requires_hf_login,
    ) for recipe in queued_ic_lora_recipes() if recipe.kind == "video" and (item := catalog.get(recipe.catalog_id)) is not None]


@tools.tool
def list_assets(ctx: RunContext[AssistantDeps], query: AssistantAssetQuery) -> AssistantAssetPage:
    """Find assets by media_kind/name, newest first by creation time. Use show_assets to display selected results.

    For the user's last or latest video, call with media_kind="video" and limit=1.
    Returns asset metadata and IDs for input references, with a cursor for the next page.
    """
    try:
        page = ctx.deps.assets.list_assets(query.api_query(ASSISTANT_LIMITS.discovery_assets))
    except HTTPError as exc:
        if exc.status_code != 400:
            raise
        raise ModelRetry("Invalid asset query or cursor. Use a valid media kind and returned cursor.") from exc
    return AssistantAssetPage(items=[AssistantAssetCore.of(AssetCore.model_validate(item.model_dump())) for item in page.items], next_cursor=page.next_cursor)


@tools.tool
def get_asset(ctx: RunContext[AssistantDeps], asset_id: str) -> AssistantAsset:
    """Read an existing asset's metadata by ID. Use show_assets to display its preview.

    A generated asset includes `source`: the prompt and settings that made it.
    """
    try:
        asset = describe_asset(ctx.deps.assets, ctx.deps.generations, asset_id)
    except HTTPError as exc:
        if exc.status_code != 404:
            raise
        raise ModelRetry("Asset unavailable. Ask the user to select/import an existing asset.") from exc
    return asset


@tools.tool
def view_asset(ctx: RunContext[AssistantDeps], asset_id: str) -> dict[str, str]:
    """Look at an image, or a video's thumbnail frame, from earlier messages or results.

    Images attached to the current message are already visible. A video thumbnail is one
    frame, not a review of the full video. Use get_asset for audio metadata.
    """
    assert ctx.tool_call_id is not None
    try:
        ctx.deps.view(ctx.tool_call_id, asset_id)
    except HTTPError as exc:
        raise ModelRetry(exc.detail) from exc
    return {"asset_id": asset_id, "preview": "shown after this result"}


@tools.tool
def show_assets(ctx: RunContext[AssistantDeps], asset_ids: Annotated[list[str], Field(max_length=ASSISTANT_LIMITS.shown_assets)]) -> list[AssistantAsset]:
    """Show the selected assets in this reply, replacing any earlier selection. Use [] to clear.

    Search first, then pass only IDs matching the user's request. The app renders previews.
    """
    assets = [get_asset(ctx, asset_id) for asset_id in dict.fromkeys(asset_ids)]
    ctx.deps.check_stopped()
    ctx.deps.run.asset_ids = [asset.id for asset in assets]
    ctx.deps.save()
    return assets


@tools.tool
def list_generations(ctx: RunContext[AssistantDeps]) -> list[AssistantGeneration]:
    """Read the ten most recent Explore jobs. Use show_assets for selected output previews."""
    return [generation_result(item, ctx.deps.assets) for item in ctx.deps.generations.list_generations(limit=10)]


@tools.tool
def get_generation_status(ctx: RunContext[AssistantDeps], generation_id: str) -> AssistantGeneration:
    """Read status, error code and output asset metadata for an existing generation."""
    return generation_result(_read_generation(ctx.deps, generation_id), ctx.deps.assets)


@tools.tool
def wait_for_generation(ctx: RunContext[AssistantDeps], generation_id: str) -> AssistantGeneration:
    """Wait for a job needed by a dependent step. Queue independent clips before waiting."""
    generation = _read_generation(ctx.deps, generation_id)
    if generation.status in ("queued", "running", "cancelling"):
        _defer(generation)
    return generation_result(generation, ctx.deps.assets)


@tools.tool(retries=RECIPE_GUARDED_RETRIES)
def generate_video(ctx: RunContext[AssistantDeps], params: TextParams, job_number: JobNumber) -> AssistantGeneration:
    """Queue one local text-to-video clip and return immediately with its job ID.

    For a named style or recipe, use list_lora_recipes and run_lora_recipe instead.
    """
    _recipe_guard(ctx.deps)
    _video_settings(ctx.deps, params, "t2v", params.aspectRatio)
    request = CreateTextToVideoRequest(params=api_params(params, TextToVideoParams, _provenance(ctx.deps)))
    return _submit(ctx.deps, job_number, ctx.deps.generations.create_text_to_video, request)


@tools.tool(retries=RECIPE_GUARDED_RETRIES)
def image_to_video(ctx: RunContext[AssistantDeps], params: ImageParams, inputs: ImageToVideoInputs, job_number: JobNumber) -> AssistantGeneration:
    """Queue image animation and return its job ID.

    For a named style, camera move or recipe, use list_lora_recipes and run_lora_recipe instead.
    """
    _recipe_guard(ctx.deps)
    _video_settings(ctx.deps, params, "i2v", params.aspectRatio)
    request = CreateImageToVideoRequest(params=api_params(params, ImageToVideoParams, _provenance(ctx.deps)), inputs=inputs)
    return _submit(ctx.deps, job_number, ctx.deps.generations.create_image_to_video, request)


@tools.tool(retries=SUBMIT_RETRIES)
def audio_to_video(ctx: RunContext[AssistantDeps], params: AudioParams, inputs: AudioToVideoInputs, job_number: JobNumber) -> AssistantGeneration:
    """Generate local video synchronized to an audio asset, optionally with a start image.

    Duration comes from the audio; use the advertised a2v settings.
    Returns the queued job immediately.
    """
    _audio_settings(ctx.deps, params, _source_seconds(ctx.deps, inputs.audio, "audio"))
    request = CreateAudioToVideoRequest(params=api_params(params, CreateAudioToVideoParams, _provenance(ctx.deps)), inputs=inputs)
    return _submit(ctx.deps, job_number, ctx.deps.generations.create_audio_to_video, request)


@tools.tool(retries=SUBMIT_RETRIES)
def run_lora_recipe(ctx: RunContext[AssistantDeps], recipe_id: str, params: RecipeParams, job_number: JobNumber, inputs: RecipeInputs | None = None) -> AssistantGeneration:
    """Queue an installed Explore recipe and return its job ID.

    Write the prompt following get_recipe_guidance for this recipe.
    Image recipes need startFrame; Transition also needs endFrame. Text recipes omit inputs.
    Use existing asset IDs from discovery or conversation context.
    """
    recipe = get_lora_recipe(recipe_id)
    if recipe is None or recipe.mode not in ("t2v", "i2v"):
        raise ModelRetry("Unknown or unsupported recipe. Use list_lora_recipes.")
    _video_settings(ctx.deps, params, recipe.mode, params.aspectRatio)
    params = params.model_copy(update={"prompt": _with_trigger(ctx.deps, recipe, params.prompt)})
    listed = next((item for item in ctx.deps.loras().loras if item.lora.id == recipe.catalog_id), None)
    variant, note = (None, None) if listed is None else _variant_for(
        listed.lora.download, listed.downloaded_variant_ids, params.model, listed.lora.name)
    request = CreateLoraRecipeRequest(
        params=api_params(params, LoraRecipeParams, _provenance(ctx.deps), catalogId=recipe.catalog_id, **({"variantId": variant} if variant else {})),
        inputs=None if inputs is None else LoraRecipeInputs(startFrame=inputs.startFrame, endFrame=inputs.endFrame),
    )
    result = _submit(ctx.deps, job_number, partial(ctx.deps.generations.create_lora_recipe, recipe_id), request)
    _recipe_queued(ctx.deps)
    return result.model_copy(update={"note": note})


@tools.tool(retries=SUBMIT_RETRIES)
def run_video_recipe(
    ctx: RunContext[AssistantDeps], recipe_id: str, params: VideoRecipeParams, job_number: JobNumber,
    generation_id: str | None = None, video_asset_id: str | None = None, reference_image_asset_id: str | None = None,
    template_values: dict[str, str] | None = None,
) -> AssistantGeneration:
    """Queue a recipe from list_video_recipes on an existing video and return its job ID.

    Supply either a succeeded generation_id or an existing video_asset_id. The output keeps the
    source's framing and length. Video Guide also takes params.controlType (pose, depth or canny). Write the prompt, or template_values for a templated recipe,
    following get_recipe_guidance for this recipe. It runs at 720p unless params.resolution is set.
    """
    if (ic_recipe := get_ic_lora_recipe(recipe_id)) is None or ic_recipe.kind == "track":
        raise ModelRetry("Unknown video recipe. Use list_video_recipes.")
    if ic_recipe.kind == "sheet":
        raise ModelRetry(f"{recipe_id} takes images, not a video. Queue it with run_ingredients.")
    _model(ctx.deps, params.model, "ic_lora")
    ic_lora = _video_recipe_lora(ctx.deps, recipe_id)
    listed = next(item for item in ctx.deps.ic_loras().ic_loras if item.ic_lora.id == ic_lora.id)
    variant, note = _variant_for(ic_lora.download, listed.downloaded_variant_ids, params.model, ic_lora.name)
    if (template := ic_lora.prompt_template) is not None:
        # With auto-enhance off nothing else fills the template; with it on, the enhancer fills it again.
        try:
            params = params.model_copy(update={"prompt": fill_prompt_template(template, template_values or {})})
        except ValueError as exc:
            raise ModelRetry(f"{exc}. See get_recipe_guidance for {recipe_id}.") from exc
    elif template_values:
        raise ModelRetry(f"{recipe_id} has no template. Put the prompt text in params.prompt and leave template_values out.")
    image = None if reference_image_asset_id is None else InputAssetRef(assetId=reference_image_asset_id)
    request = CreateIcLoraRecipeRequest(
        params=api_params(params, IcLoraVideoRecipeParams, _provenance(ctx.deps), **({"variantId": variant} if variant else {})),
        inputs=IcLoraRecipeInputs(video=_video_input(ctx.deps, generation_id, video_asset_id), image=image),
    )

    def create(request: CreateIcLoraRecipeRequest, /, *, generation_id: str, analytics_client: AnalyticsClient, analytics_requested_by: AnalyticsRequestedBy) -> Generation:
        return ctx.deps.generations.create_ic_lora_recipe(
            recipe_id, request, generation_id=generation_id, analytics_client=analytics_client, analytics_requested_by=analytics_requested_by)

    result = _submit(ctx.deps, job_number, create, request)
    _recipe_queued(ctx.deps)
    return result.model_copy(update={"note": note})


_INGREDIENTS = "ingredients"
# Home's strength. A create without one gets the catalog's 1, which GenSpace keeps.
_INGREDIENTS_SCALE = 1.35


@tools.tool(retries=SUBMIT_RETRIES)
def run_ingredients(
    ctx: RunContext[AssistantDeps],
    image_asset_ids: Annotated[list[str], Field(min_length=1, max_length=IC_LORA_SHEET_MAX_IMAGES)],
    params: IngredientsParams, job_number: JobNumber,
) -> AssistantGeneration:
    """Queue Ingredients: a new video that keeps the characters, props and places in 1 to 10 images consistent.

    The images are joined into one reference sheet; the video shows nothing that is not in an image.
    Write the prompt following get_recipe_guidance for ingredients. To use a video's frame, extract_frame it first.
    Returns the queued job immediately.
    """
    _model(ctx.deps, params.model, "ic_lora")
    ic_lora = _video_recipe_lora(ctx.deps, _INGREDIENTS)
    listed = next(item for item in ctx.deps.ic_loras().ic_loras if item.ic_lora.id == ic_lora.id)
    variant, note = _variant_for(ic_lora.download, listed.downloaded_variant_ids, params.model, ic_lora.name)
    request = CreateIcLoraSheetRequest(
        params=IcLoraSheetParams.model_validate({
            **params.model_dump(), "scale": _INGREDIENTS_SCALE, "promptProvenance": _provenance(ctx.deps),
            **({"variantId": variant} if variant else {}),
        }),
        inputs=IcLoraSheetInputs(images=[InputAssetRef(assetId=asset_id) for asset_id in image_asset_ids]),
    )

    def create(request: CreateIcLoraSheetRequest, /, *, generation_id: str, analytics_client: AnalyticsClient, analytics_requested_by: AnalyticsRequestedBy) -> Generation:
        return ctx.deps.generations.create_ic_lora_recipe(
            _INGREDIENTS, request, generation_id=generation_id, analytics_client=analytics_client, analytics_requested_by=analytics_requested_by)

    result = _submit(ctx.deps, job_number, create, request)
    _recipe_queued(ctx.deps)
    return result.model_copy(update={"note": note})


@tools.tool(retries=RECIPE_GUARDED_RETRIES)
def extend_video(ctx: RunContext[AssistantDeps], params: ExtensionParams, job_number: JobNumber, generation_id: str | None = None, video_asset_id: str | None = None) -> AssistantGeneration:
    """Queue a local video extension; keeps the source size unless the user asks for a resolution, and inherits FPS.

    Pass exactly one of added_seconds ("add 3 seconds") or total_seconds ("extend it to
    10 seconds"). Supply either a succeeded generation_id or an existing video_asset_id.
    For chained jobs use the intended source's returned ID. Returns immediately.
    """
    _recipe_guard(ctx.deps, video_only=True)
    _model(ctx.deps, params.model, "extend")
    if (params.added_seconds is None) == (params.total_seconds is None):
        raise ModelRetry("Pass exactly one of added_seconds or total_seconds.")
    video = _video_input(ctx.deps, generation_id, video_asset_id)
    added = params.added_seconds
    if added is None:
        assert params.total_seconds is not None
        source = _source_seconds(ctx.deps, video)
        added = round(params.total_seconds - source, 2)
        if not 2 <= added <= 20:
            raise ModelRetry(
                f"The source is {source:.1f} s, so a total of {params.total_seconds:g} s adds {added:g} s. "
                "An extension adds 2 to 20 s. Tell the user and ask what they want."
            )
    request = CreateExtendRequest(
        params=ExtendParams.model_validate({
            **params.model_dump(exclude={"added_seconds", "total_seconds"}),
            "duration": added, "promptProvenance": _provenance(ctx.deps),
        }),
        inputs=ExtendInputs(video=video),
    )
    return _submit(ctx.deps, job_number, ctx.deps.generations.create_extend, request)


@tools.tool(retries=RECIPE_GUARDED_RETRIES)
def retake_video(ctx: RunContext[AssistantDeps], params: ReplacementParams, job_number: JobNumber, generation_id: str | None = None, video_asset_id: str | None = None) -> AssistantGeneration:
    """Replace a time range's audio/video locally. startTime and duration are seconds, at most 10.

    Supply either a succeeded generation_id or an existing video_asset_id. Preserve
    the user's requested mode and time range. Returns immediately.
    """
    _recipe_guard(ctx.deps, video_only=True)
    _model(ctx.deps, params.model, "retake")
    video = _video_input(ctx.deps, generation_id, video_asset_id)
    source = _source_seconds(ctx.deps, video)
    # Same 0.1 s tolerance as the queue's own range check.
    if params.startTime + params.duration > source + 0.1:
        latest = max(source - params.duration, 0)
        raise ModelRetry(
            f"The source is {source:.1f} s, so startTime + duration must end by then. "
            f"For the last {params.duration:g} s use startTime {latest:.1f}."
        )
    request = CreateRetakeRequest(
        params=api_params(params, RetakeRequestParams, _provenance(ctx.deps)),
        inputs=RetakeInputs(video=video),
    )
    return _submit(ctx.deps, job_number, ctx.deps.generations.create_retake, request)


def _edit(deps: AssistantDeps, produce: Callable[[], Asset]) -> AssistantAssetCore:
    deps.check_stopped()
    try:
        asset = produce()
    except HTTPError as exc:
        if exc.status_code == 404:
            raise ModelRetry("Asset unavailable. Use list_assets to find a current ID.") from exc
        if exc.status_code not in (400, 422):
            raise
        raise ModelRetry(f"Explore rejected this edit: {exc.detail} ({exc.code}). Fix the arguments, or tell the user.") from exc
    return AssistantAssetCore.of(AssetCore.model_validate(asset.model_dump()))


@tools.tool
def extract_frame(
    ctx: RunContext[AssistantDeps], at_seconds: Annotated[float, Field(ge=0)] | None = None,
    generation_id: str | None = None, video_asset_id: str | None = None,
) -> AssistantAssetCore:
    """Save one frame of a video as a new image asset. at_seconds=0 is the first frame; omit it for the last frame.

    The image can start a new clip with image_to_video, or be inspected with view_asset.
    Supply either a succeeded generation_id or an existing video_asset_id.
    """
    video = _video_input(ctx.deps, generation_id, video_asset_id)
    return _edit(ctx.deps, lambda: ctx.deps.assets.extract_frame(video.assetId, at_seconds))


@tools.tool
def retry_interrupted_generation(ctx: RunContext[AssistantDeps], generation_id: str | None = None) -> AssistantGeneration:
    """Retry an owned interrupted job, unchanged (one retry per request). Defaults to the latest job."""
    deps = ctx.deps
    generation_id = generation_id or next(reversed(deps.run.jobs.values()), None)
    if generation_id not in deps.run.generation_ids() or deps.run.retry_count >= 1:
        raise ModelRetry("No retry is available. Explain the result and ask the user how to proceed.")
    assert generation_id is not None
    generation = _read_generation(deps, generation_id)
    if generation.status != "failed" or generation.error_code != "INTERRUPTED":
        raise ModelRetry("Only interrupted jobs can be retried automatically. Never retry cancellations.")
    deps.run.retry_count += 1
    deps.save()
    generation = deps.generations.retry_generation(generation.id)
    _cancel_if_stopped(deps, generation)
    _defer(generation)


@tools.tool
def retry_failed_job(ctx: RunContext[AssistantDeps], generation_id: str) -> AssistantGeneration:
    """Rerun a failed job from this chat, unchanged, when the user asks to retry it. Keeps its job card."""
    deps = ctx.deps
    if generation_id not in deps.chat_generation_ids():
        raise ModelRetry("That job is not from this chat. Use list_generations to find the user's job.")
    if generation_id in deps.run.user_retried:
        raise ModelRetry("This job was already retried in this request. Explain the result and ask the user how to proceed.")
    generation = _read_generation(deps, generation_id)
    if generation.status == "cancelled":
        raise ModelRetry("Cancelled jobs cannot be retried. Queue it again only if the user asks.")
    if generation.status != "failed":
        raise ModelRetry(f"Only failed jobs can be retried; this one is {generation.status}.")
    if generation_id in deps.run.jobs.values():
        # The user cannot have asked yet: they have not seen this request's job fail.
        raise ModelRetry("This job failed in this request. Explain the failure and ask whether to retry.")
    deps.run.show_retried(generation_id)
    deps.save()
    generation = deps.generations.retry_generation(generation.id)
    _cancel_if_stopped(deps, generation)
    _defer(generation)
