from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Literal, Protocol, cast

from pydantic import BaseModel, JsonValue, ValidationError

from api_types import (
    CreateCustomLoraRequest,
    CreateIcLoraRecipeRequest,
    CreateIcLoraSheetRequest,
    CreateIcLoraTrackRequest,
    AudioToVideoParams,
    Asset,
    CreateAudioToVideoParams,
    CreateAudioToVideoRequest,
    CreateImageToVideoRequest,
    CreateKeyframesRequest,
    CreateLoraRecipeRequest,
    CreateExtendRequest,
    CreateRetakeRequest,
    CreateTextToImageRequest,
    ImageGenerationStatusResponse,
    CreateTextToVideoRequest,
    EditPreflightRequest,
    EditPreflightResponse,
    IcLoraCatalogItem,
    IcLoraTilePreflightRequest,
    IcLoraTilePreflightResponse,
    IcLoraVideoRecipeStoredParams,
    ExtendParams,
    Generation,
    ApiActivityStatus,
    GenerationRisk,
    IcLoraRecipeStoredParams,
    ImageToVideoInputs,
    ImageToVideoRecipeParams,
    LTXQueuedAspectRatio,
    LTXVideoGenFps,
    LTXVideoGenResolution,
    OfferingId,
    QueueEntry,
    QueueProgress,
    QueueSnapshot,
    RetakeParams,
    TextToVideoParams,
)
from _routes._errors import HTTPError
from handlers.base import resolve_models_dir
from handlers.hf_auth_utils import optional_hf_token
from runtime_config.accelerator import accelerator_backend
from runtime_config.ic_lora_high_res import (
    HIGH_RES_NOT_ENABLED,
    high_res_cell,
    high_res_flag_satisfied,
    high_res_source_large_enough,
)
from runtime_config.ic_lora_stage_mode import IcLoraStageMode
from runtime_config.reject_reasons import PassShape
from runtime_config.video_job_budget import (
    LOCAL_GENERATION_UNSUPPORTED,
    LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
    resolve_pass_shape,
)
from runtime_config.ltx_capabilities import LtxCapabilityFeature
from runtime_config.model_download_specs import local_aspect_ratios
from services.features.image.feature_id import FEATURE as TEXT_TO_IMAGE_FEATURE
from services.features.image.text_to_image import resolve_text_to_image_params
from services.features.video.job_common import (
    PASS_SHAPE_SPEC_KEY,
    high_res_unsupported_error,
    require_capable_local_model,
    stored_generation_risk,
)
from services.aspect_ratio import find_closest_aspect_ratio_option
from services.fast_video_pipeline.nag_config import (
    DEFAULT_SCALE as DEFAULT_NAG_SCALE,
    decide_nag,
)
from services.media_probe import MAX_VIDEO_BYTES, video_fps
from services.features.ic_lora_recipes import (
    IC_LORA_RECIPES,
    IcLoraRecipe,
    ic_lora_input_cap_message,
    resolve_ic_lora_recipe_create,
    resolve_synthesized_ic_lora_create,
)
from services.features.custom_loras import (
    CustomLoraCreateError,
    resolve_custom_lora_create,
)
from services.features.lora_recipes import RecipeCreateError, resolve_recipe_create
from services.lora_catalog import LoraCatalogProvider
from services.store import Store
from services.sqlite_store import collect_asset_ids
from services.records import (
    AttemptError,
    AssetRecord,
    CapabilityFailedError,
    GenerationRecord,
    MediaError,
    QueueGenerationRecord,
    StatusError,
    UnavailableError,
)
from services.generation_queue.control import (
    QueueControl,
    NoOpQueueControl,
    cancel_queued_generation,
)
from services.generation_queue.finished_tray import FinishedTray
from services.generation_queue.orphans import (
    cancel_provider_orphans_in_background,
    forget_in,
)
from services.generation_queue.registry import ExecutorRegistry
from services.features.video import (
    AUDIO_TO_VIDEO_FEATURE,
    EXTEND_FEATURE,
    FEATURE as TEXT_TO_VIDEO_FEATURE,
    IMAGE_TO_VIDEO_FEATURE,
    KEYFRAMES_FEATURE,
    RETAKE_FEATURE,
)
from state.app_state_types import AppState

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig


AnalyticsClient = Literal["desktop", "remote"]
# Set only when the user did not ask for the job directly; analytics reports those as "user".
AnalyticsRequestedBy = Literal["assistant"]


logger = logging.getLogger(__name__)

# Error code of a Keyframes create while the `keyframes` feature flag is off.
KEYFRAMES_NOT_ENABLED = "KEYFRAMES_NOT_ENABLED"


def _generation_to_api(record: GenerationRecord) -> Generation:
    payload = record.model_dump(mode="json")
    spec = dict(record.spec)
    spec.pop("_analytics", None)
    spec.pop("_risk", None)
    spec.pop(PASS_SHAPE_SPEC_KEY, None)
    payload["spec"] = spec
    # Junk fails closed on read the same way execute does. A strict validate
    # of the raw dict raises on list, get, queue, retry, and cancel.
    risk = stored_generation_risk(record.spec)
    payload["risk"] = None if risk is None else risk.model_dump(mode="json")
    return Generation.model_validate(payload)


# Create-time Fast budget gate. Raises HTTPError; returns the bet annotation.
QueuedFastPreflight = Callable[..., GenerationRisk | None]


# Create-time A2V budget gate over derived frames. Raises HTTPError; returns the bet annotation.
QueuedA2VPreflight = Callable[..., GenerationRisk | None]


class LocalA2VDeriver(Protocol):
    """Frames from the audio, sized for the resolved shape. Raises HTTPError."""

    def __call__(
        self,
        params: CreateAudioToVideoParams,
        audio_duration_ms: int,
        *,
        pass_shape: PassShape,
    ) -> AudioToVideoParams: ...


# Create-time edit window + budget gate. Raises HTTPError; returns the bet annotation.
QueuedEditPreflight = Callable[..., GenerationRisk | None]


# Create-time IC-LoRA recipe budget gate. Raises HTTPError; returns the bet annotation.
QueuedIcLoraRecipePreflight = Callable[..., GenerationRisk | None]


# Advisory verdict per tile size of a recipe create. Per-size failures are returned, never raised.
IcLoraTileSizePreflight = Callable[..., IcLoraTilePreflightResponse]


@dataclass(frozen=True)
class _PreparedIcLoraRecipe:
    """A recipe create that passed every check before the budget."""

    recipe: IcLoraRecipe
    item: IcLoraCatalogItem
    stored: IcLoraVideoRecipeStoredParams
    video: AssetRecord
    source_fps: float


# Advisory per-cell verdicts for edit pickers. Per-cell failures are returned, never raised.
EditMatrixPreflight = Callable[..., EditPreflightResponse]


def _ic_lora_create_error(exc: RecipeCreateError) -> HTTPError:
    if exc.code == "IC_LORA_NOT_DOWNLOADED":
        status = 409
    elif exc.code == "UNKNOWN_DOWNLOAD_VARIANT":
        status = 404
    else:
        status = 422
    return HTTPError(status, str(exc), code=exc.code)


class QueuedGenerationHandler:
    def __init__(
        self,
        store: Store,
        executor_registry: ExecutorRegistry,
        derive_local_a2v: LocalA2VDeriver,
        queue_control: QueueControl | None = None,
        *,
        config: "RuntimeConfig | None" = None,
        lora_catalog_provider: LoraCatalogProvider | None = None,
        state: AppState | None = None,
        lock: RLock | None = None,
        preflight_queued_fast: QueuedFastPreflight,
        preflight_queued_a2v: QueuedA2VPreflight,
        preflight_queued_edit: QueuedEditPreflight,
        preflight_edit_matrix: EditMatrixPreflight,
        preflight_queued_ic_lora_recipe: QueuedIcLoraRecipePreflight,
        preflight_queued_synthesized_ic_lora: QueuedIcLoraRecipePreflight,
        preflight_ic_lora_tile_sizes: IcLoraTileSizePreflight,
        high_res_enabled: Callable[[], bool] | None = None,
        keyframes_enabled: Callable[[], bool] | None = None,
        should_use_local_encoding: Callable[[], bool] | None = None,
        image_generation_status: Callable[[], ImageGenerationStatusResponse] | None = None,
    ) -> None:
        self._db = store
        self._executor_registry = executor_registry
        self._derive_local_a2v = derive_local_a2v
        self._preflight_queued_fast = preflight_queued_fast
        self._preflight_queued_a2v = preflight_queued_a2v
        self._preflight_queued_edit = preflight_queued_edit
        self._preflight_edit_matrix = preflight_edit_matrix
        self._preflight_queued_ic_lora_recipe = preflight_queued_ic_lora_recipe
        self._preflight_queued_synthesized_ic_lora = preflight_queued_synthesized_ic_lora
        self._preflight_ic_lora_tile_sizes = preflight_ic_lora_tile_sizes
        self._queue_control = (
            NoOpQueueControl() if queue_control is None else queue_control
        )
        # LoRA-recipe create needs the models dir, the catalog, and HF-auth state to
        # resolve a catalog id to an installed path and to gate typed misses before
        # enqueue. Optional so tests that only exercise the plain t2v/i2v create paths
        # (or construct the handler directly) don't have to wire them.
        self._config = config
        self._lora_catalog_provider = lora_catalog_provider
        self._state = state
        self._lock = lock
        self._high_res_enabled = high_res_enabled or (lambda: False)
        self._keyframes_enabled = keyframes_enabled or (lambda: False)
        self._should_use_local_encoding = should_use_local_encoding
        self._image_generation_status = image_generation_status
        self._finished = FinishedTray()

    def recover_on_boot(self) -> None:
        lanes = self._executor_registry.lanes
        self._db.fail_running_on_boot()
        self._db.reassign_unknown_lanes(lanes.ids)
        # A row keeps its provider job id until a cancel succeeds. A failure here, or a stop
        # before the cancel ends, leaves the id for the next boot.
        orphans = self._db.list_provider_orphans()
        cancel_provider_orphans_in_background(orphans, lanes.spec, forget_in(self._db))

    def _dump_fast_params(
        self,
        params: BaseModel,
        *,
        exclude: set[str] | None = None,
        uses_torch_compile: bool = True,
    ) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = params.model_dump(mode="json", exclude=exclude)
        default_scale = DEFAULT_NAG_SCALE
        compile_on = False
        local_encoding_on = True
        if self._state is not None and self._lock is not None:
            with self._lock:
                settings = self._state.app_settings
                default_scale = settings.nag_scale
                compile_on = (
                    uses_torch_compile
                    and settings.use_torch_compile
                    and not (
                        self._config is not None and self._config.use_sage_attention
                    )
                )
                if self._should_use_local_encoding is not None:
                    local_encoding_on = self._should_use_local_encoding()
                else:
                    local_encoding_on = not (
                        bool(settings.ltx_api_key.strip())
                        and not settings.use_local_text_encoder
                    )
        negative_prompt = getattr(params, "negativePrompt", "")
        explicit_scale = getattr(params, "nagScale", None)
        decision = decide_nag(
            negative_prompt if isinstance(negative_prompt, str) else "",
            os.environ,
            scale=explicit_scale,
            alpha=getattr(params, "nagAlpha", None),
            tau=getattr(params, "nagTau", None),
            default_scale=default_scale,
            local_encoding=local_encoding_on,
            compiled=compile_on,
        )
        if decision.config is not None:
            payload["nagScale"] = decision.config.scale
            payload["nagAlpha"] = decision.config.alpha
            payload["nagTau"] = decision.config.tau
        else:
            for key in ("nagScale", "nagAlpha", "nagTau"):
                payload.pop(key, None)
            if isinstance(explicit_scale, (int, float)) and explicit_scale <= 0.0:
                payload["nagScale"] = 0.0
        return payload

    def create_text_to_video(
        self,
        req: CreateTextToVideoRequest,
        *,
        analytics_client: AnalyticsClient | None = None,
        analytics_requested_by: AnalyticsRequestedBy | None = None,
        generation_id: str | None = None,
    ) -> Generation:
        spec: dict[str, JsonValue] = {
            "params": self._dump_fast_params(req.params),
            "inputs": {},
        }
        pass_shape = self._resolve_pass_shape(req.params.passShape)
        risk = self._preflight_fast_params(
            offering=req.params.model,
            resolution=req.params.resolution,
            aspect_ratio=req.params.aspectRatio,
            duration=req.params.duration,
            fps=req.params.fps,
            pass_shape=pass_shape,
        )
        return self._enqueue_generation(
            TEXT_TO_VIDEO_FEATURE,
            spec,
            contract_version=req.contract_version,
            analytics_client=analytics_client,
            analytics_requested_by=analytics_requested_by,
            generation_id=generation_id,
            risk=risk,
            pass_shape=pass_shape,
        )

    def create_image_to_video(
        self,
        req: CreateImageToVideoRequest,
        *,
        analytics_client: AnalyticsClient | None = None,
        analytics_requested_by: AnalyticsRequestedBy | None = None,
        generation_id: str | None = None,
    ) -> Generation:
        start_frame = self._require_image_asset(req.inputs.startFrame.assetId, slot="startFrame")
        if req.inputs.endFrame is not None:
            self._require_image_asset(req.inputs.endFrame.assetId, slot="endFrame")
        spec: dict[str, JsonValue] = {
            "params": self._dump_fast_params(req.params),
            "inputs": req.inputs.model_dump(mode="json", exclude_none=True),
        }
        pass_shape = self._resolve_pass_shape(req.params.passShape)
        risk = self._preflight_fast_params(
            offering=req.params.model,
            resolution=req.params.resolution,
            aspect_ratio=self._resolve_queued_preflight_aspect(
                req.params.aspectRatio, req.params.resolution, start_frame
            ),
            duration=req.params.duration,
            fps=req.params.fps,
            pass_shape=pass_shape,
        )
        return self._enqueue_generation(
            IMAGE_TO_VIDEO_FEATURE,
            spec,
            contract_version=req.contract_version,
            analytics_client=analytics_client,
            analytics_requested_by=analytics_requested_by,
            generation_id=generation_id,
            risk=risk,
            pass_shape=pass_shape,
        )

    def create_keyframes(
        self,
        req: CreateKeyframesRequest,
        *,
        analytics_client: AnalyticsClient | None = None,
        analytics_requested_by: AnalyticsRequestedBy | None = None,
        generation_id: str | None = None,
    ) -> Generation:
        # The flag is the backend gate. Hiding the feature on Home does not stop a direct request.
        if not self._keyframes_enabled():
            raise HTTPError(422, KEYFRAMES_NOT_ENABLED, code=KEYFRAMES_NOT_ENABLED)
        stills = [
            (keyframe, self._require_image_asset(keyframe.assetId, slot="keyframe"))
            for keyframe in req.inputs.keyframes
        ]
        self._require_offering_capability(
            req.params.model,
            "multi_keyframe",
            code="UNSUPPORTED_MULTI_KEYFRAME",
            message="Keyframes are not supported for the selected LTX model.",
        )
        # Auto follows the first still on the timeline. Execute picks the same one.
        _, first_still = min(stills, key=lambda pair: pair[0].frameIndex)
        spec: dict[str, JsonValue] = {
            "params": self._dump_fast_params(req.params),
            "inputs": req.inputs.model_dump(mode="json"),
        }
        pass_shape = self._resolve_pass_shape(req.params.passShape)
        risk = self._preflight_fast_params(
            offering=req.params.model,
            resolution=req.params.resolution,
            aspect_ratio=self._resolve_queued_preflight_aspect(
                req.params.aspectRatio, req.params.resolution, first_still
            ),
            duration=req.params.duration,
            fps=req.params.fps,
            pass_shape=pass_shape,
        )
        return self._enqueue_generation(
            KEYFRAMES_FEATURE,
            spec,
            contract_version=req.contract_version,
            analytics_client=analytics_client,
            analytics_requested_by=analytics_requested_by,
            generation_id=generation_id,
            risk=risk,
            pass_shape=pass_shape,
        )

    def create_text_to_image(
        self,
        req: CreateTextToImageRequest,
        *,
        analytics_client: AnalyticsClient | None = None,
        generation_id: str | None = None,
    ) -> Generation:
        if self._image_generation_status is None:
            raise HTTPError(500, "Image generation status is not configured")
        params = resolve_text_to_image_params(req.params, self._image_generation_status())
        spec: dict[str, JsonValue] = {
            "params": params.model_dump(mode="json"),
            "inputs": {},
        }
        return self._enqueue_generation(
            TEXT_TO_IMAGE_FEATURE,
            spec,
            contract_version=req.contract_version,
            analytics_client=analytics_client,
            generation_id=generation_id,
        )

    def create_lora_recipe(
        self,
        recipe_id: str,
        req: CreateLoraRecipeRequest,
        *,
        analytics_client: AnalyticsClient | None = None,
        analytics_requested_by: AnalyticsRequestedBy | None = None,
        generation_id: str | None = None,
    ) -> Generation:
        """Create a queued generation for a LoRA recipe.

        The client sends the catalog id + scale, never a filesystem path.
        :func:`resolve_recipe_create` validates the recipe (unknown / mode / device /
        HF / visibility / not-downloaded / missing start frame) and stores
        ``loras: [{ ref: "", scale, catalogId }]``. Execute hydrates ``ref`` via
        ``make_recipe_lora_resolver``. Anything invalid raises a typed miss *before*
        this handler enqueues — no row, no GPU slot. The row is stored under
        ``feature == recipe_id`` (the web Explore id). i2v recipes also persist
        ``inputs.startFrame``.
        """
        self._require_local_generation_possible()
        try:
            feature, params, inputs = resolve_recipe_create(
                recipe_id,
                req,
                catalog=self._require_catalog(),
                models_dir=self._models_dir(),
                device=self._recipe_device(),
                hf_authenticated=self._hf_authenticated(),
            )
        except RecipeCreateError as exc:
            raise HTTPError(422, str(exc), code=exc.code) from exc

        return self._enqueue_resolved_lora_run(
            feature,
            params,
            inputs,
            contract_version=req.contract_version,
            analytics_client=analytics_client,
            analytics_requested_by=analytics_requested_by,
            generation_id=generation_id,
        )

    def create_custom_lora(
        self,
        req: CreateCustomLoraRequest,
        *,
        analytics_client: AnalyticsClient | None = None,
        analytics_requested_by: AnalyticsRequestedBy | None = None,
        generation_id: str | None = None,
    ) -> Generation:
        """Create a queued generation for a saved custom LoRA.

        The client sends the custom LoRA id + scale, never a filesystem path.
        :func:`resolve_custom_lora_create` reads the LoRA's saved mode and weights path, and
        validates the Start Frame rules, before anything is queued. The row is stored under the
        ``custom-lora`` feature.
        """
        self._require_local_generation_possible()
        try:
            feature, params, inputs = resolve_custom_lora_create(
                req, models_dir=self._models_dir()
            )
        except CustomLoraCreateError as exc:
            raise HTTPError(exc.status, str(exc), code=exc.code) from exc
        return self._enqueue_resolved_lora_run(
            feature,
            params,
            inputs,
            contract_version=req.contract_version,
            analytics_client=analytics_client,
            analytics_requested_by=analytics_requested_by,
            generation_id=generation_id,
        )

    def _enqueue_resolved_lora_run(
        self,
        feature: str,
        params: TextToVideoParams | ImageToVideoRecipeParams,
        inputs: ImageToVideoInputs | None,
        *,
        contract_version: int,
        analytics_client: AnalyticsClient | None,
        analytics_requested_by: AnalyticsRequestedBy | None,
        generation_id: str | None,
    ) -> Generation:
        """Check the input frames, run the Fast budget gate, and enqueue a resolved LoRA run."""
        if inputs is not None:
            start_frame = self._require_image_asset(
                inputs.startFrame.assetId, slot="startFrame"
            )
            if inputs.endFrame is not None:
                self._require_image_asset(inputs.endFrame.assetId, slot="endFrame")
        else:
            start_frame = None
        # Same Fast budget gate as the plain t2v/i2v creates: the run's params
        # carry the client's offering, and execute budgets via that same field,
        # so preflighting here admits exactly what the run budgets. i2v auto
        # resolves from the start frame; t2v never sends auto.
        pass_shape = self._resolve_pass_shape(params.passShape)
        risk = self._preflight_fast_params(
            offering=params.model,
            resolution=params.resolution,
            aspect_ratio=self._resolve_queued_preflight_aspect(
                params.aspectRatio, params.resolution, start_frame
            ),
            duration=params.duration,
            fps=params.fps,
            pass_shape=pass_shape,
        )

        spec: dict[str, JsonValue] = {
            "params": self._dump_fast_params(params),
            "inputs": (
                inputs.model_dump(mode="json", exclude_none=True) if inputs is not None else {}
            ),
        }
        return self._enqueue_generation(
            feature,
            spec,
            contract_version=contract_version,
            analytics_client=analytics_client,
            analytics_requested_by=analytics_requested_by,
            generation_id=generation_id,
            risk=risk,
            pass_shape=pass_shape,
        )

    def _require_local_generation_possible(self) -> None:
        # The Explore queue is local-only (no API fallback), so a machine that can't
        # generate locally must fail the create rather than enqueue a job that can
        # never run. Mirrors StateHandlerBase._require_local_generation_possible.
        if (
            self._config is not None
            and self._config.local_generations_mode == "unsupported"
        ):
            raise HTTPError(
                422,
                LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
                code=LOCAL_GENERATION_UNSUPPORTED,
            )

    def _recipe_device(self) -> str | None:
        # The real accelerator backend (cuda / rocm / mps / cpu), not the Darwin flag:
        # an Intel Mac is not MPS and a CPU/ROCm Linux box is not CUDA. None when there
        # is no config to gate against (tests that construct the handler directly).
        if self._config is None:
            return None
        return accelerator_backend()

    def _hf_authenticated(self) -> bool:
        if self._state is None or self._lock is None:
            return False
        return optional_hf_token(self._state, self._lock) is not None

    def _require_catalog(self) -> LoraCatalogProvider:
        if self._lora_catalog_provider is None:
            raise HTTPError(500, "LoRA catalog is not configured")
        return self._lora_catalog_provider

    def _models_dir(self) -> Path:
        if self._config is None:
            raise HTTPError(500, "Runtime config is not configured")
        return resolve_models_dir(self._state, self._config)

    def create_audio_to_video(
        self,
        req: CreateAudioToVideoRequest,
        *,
        analytics_client: AnalyticsClient | None = None,
        analytics_requested_by: AnalyticsRequestedBy | None = None,
        generation_id: str | None = None,
    ) -> Generation:
        if req.params.prompt.strip() == "" and req.inputs.startFrame is None:
            raise HTTPError(
                422,
                "Connect a prompt or a start image before running this node.",
                code="INVALID_GENERATION_SPEC",
            )
        audio = self._require_audio_asset(req.inputs.audio.assetId, slot="audio")
        if audio.metadata.mediaType != "audio":
            raise HTTPError(422, "INVALID_AUDIO_ASSET", code="INVALID_GENERATION_SPEC")
        # Resolved before the budget runs: auto needs the start frame's shape,
        # and the budget needs the resolved aspect.
        start_frame = (
            None
            if req.inputs.startFrame is None
            else self._require_image_asset(
                req.inputs.startFrame.assetId, slot="startFrame"
            )
        )
        pass_shape = self._resolve_pass_shape(req.params.passShape)
        params = self._derive_local_a2v(
            req.params,
            audio.metadata.metadata.durationMs,
            pass_shape=pass_shape,
        )
        # budget_shape, the same canvas as a fully dropped row. Auto resolves
        # from the start frame first; a missing frame falls back to 16:9.
        risk = self._preflight_queued_a2v(
            offering=params.model,
            resolution=params.resolution,
            aspect_ratio=self._resolve_queued_preflight_aspect(
                params.aspectRatio, params.resolution, start_frame, is_a2v=True
            ),
            num_frames=params.numFrames,
            pass_shape=pass_shape,
        )
        spec: dict[str, JsonValue] = {
            "params": self._dump_fast_params(params, uses_torch_compile=False),
            "inputs": req.inputs.model_dump(mode="json", exclude_none=True),
        }
        return self._enqueue_generation(
            AUDIO_TO_VIDEO_FEATURE,
            spec,
            contract_version=req.contract_version,
            analytics_client=analytics_client,
            analytics_requested_by=analytics_requested_by,
            generation_id=generation_id,
            risk=risk,
            pass_shape=pass_shape,
        )

    def create_retake(
        self,
        req: CreateRetakeRequest,
        *,
        analytics_client: AnalyticsClient | None = None,
        analytics_requested_by: AnalyticsRequestedBy | None = None,
        generation_id: str | None = None,
    ) -> Generation:
        return self._enqueue_video_edit(
            RETAKE_FEATURE,
            req,
            "retake",
            code="UNSUPPORTED_RETAKE",
            message="Retake is not supported for the selected LTX model.",
            analytics_client=analytics_client,
            analytics_requested_by=analytics_requested_by,
            generation_id=generation_id,
            extra=lambda duration_ms: self._validate_retake_selection(
                req.params.startTime,
                req.params.duration,
                duration_ms,
            ),
        )

    def create_ic_lora_recipe(
        self,
        recipe_id: str,
        req: CreateIcLoraRecipeRequest | CreateIcLoraSheetRequest | CreateIcLoraTrackRequest,
        *,
        analytics_client: AnalyticsClient | None = None,
        analytics_requested_by: AnalyticsRequestedBy | None = None,
        generation_id: str | None = None,
    ) -> Generation:
        if not isinstance(req, CreateIcLoraRecipeRequest):
            return self._create_synthesized_ic_lora(
                recipe_id, req, analytics_client=analytics_client, analytics_requested_by=analytics_requested_by,
                generation_id=generation_id,
            )
        prepared = self._prepare_ic_lora_recipe(recipe_id, req)
        risk = self._preflight_queued_ic_lora_recipe(
            item=prepared.item,
            video=prepared.video,
            stored=prepared.stored,
            source_fps=prepared.source_fps,
            upscale=prepared.recipe.upscales_to_cell,
            cutout=prepared.recipe.cutout,
        )
        return self._enqueue_ic_lora_recipe(
            prepared.recipe,
            prepared.stored,
            req,
            risk=risk,
            analytics_client=analytics_client,
            analytics_requested_by=analytics_requested_by,
            generation_id=generation_id,
        )

    def preflight_ic_lora_tile_sizes(
        self, recipe_id: str, req: IcLoraTilePreflightRequest
    ) -> IcLoraTilePreflightResponse:
        """Advisory verdict per tile size and window. Every check before the budget is the create's own."""
        prepared = self._prepare_ic_lora_recipe(
            recipe_id,
            CreateIcLoraRecipeRequest(params=req.params, inputs=req.inputs),
            require_prompt=False,
        )
        return self._preflight_ic_lora_tile_sizes(
            item=prepared.item,
            video=prepared.video,
            stored=prepared.stored,
            source_fps=prepared.source_fps,
            upscale=prepared.recipe.upscales_to_cell,
            cutout=prepared.recipe.cutout,
            tile_sizes=req.tileSizes,
            max_windows=req.maxWindows,
            carries=req.carries,
        )

    def _prepare_ic_lora_recipe(
        self, recipe_id: str, req: CreateIcLoraRecipeRequest, *, require_prompt: bool = True
    ) -> _PreparedIcLoraRecipe:
        """Every create check that runs before the budget. Raises the create's own 422.

        The tile preflight sets ``require_prompt`` false. An empty prompt must not turn every
        verdict into "Prompt is required." Create still refuses an empty prompt.
        """
        self._require_local_generation_possible()
        high_res = high_res_cell(req.params.resolution)
        if high_res is not None:
            self._require_high_res_cell(recipe_id)
        video = self._require_asset(
            req.inputs.video.assetId,
            slot="video",
            kind="video",
            max_bytes=MAX_VIDEO_BYTES,
        )
        if video.metadata.mediaType != "video":
            raise HTTPError(422, "INVALID_VIDEO_ASSET", code="INVALID_GENERATION_SPEC")
        if req.inputs.image is not None:
            self._require_image_asset(req.inputs.image.assetId, slot="image")
        source_fps = video.metadata.metadata.fps
        if source_fps is None:
            try:
                source_fps = video_fps(Path(video.path))
            except MediaError as exc:
                raise HTTPError(
                    422, "INVALID_VIDEO_ASSET", code="INVALID_GENERATION_SPEC"
                ) from exc
        try:
            recipe, item, stored = resolve_ic_lora_recipe_create(
                recipe_id,
                req,
                catalog=self._require_catalog(),
                models_dir=self._models_dir(),
                hf_authenticated=self._hf_authenticated(),
                source_fps=source_fps,
                require_prompt=require_prompt,
            )
        except RecipeCreateError as exc:
            raise _ic_lora_create_error(exc) from exc
        # A high cell runs one tiled stage. An IC-LoRA that keeps its LoRA on stage 2 has no
        # such run, and its look still cannot go through the tiled path.
        if high_res is not None and item.default_settings.stage_mode is IcLoraStageMode.STAGE_2_LORA:
            raise high_res_unsupported_error(item.name)
        too_long = ic_lora_input_cap_message(
            video.metadata.metadata.durationMs, stored.fps
        )
        if too_long is not None:
            raise HTTPError(422, too_long, code="IC_LORA_INPUT_TOO_LONG")
        self._require_offering_capability(
            req.params.model,
            "ic_lora",
            code="UNSUPPORTED_IC_LORA",
            message=f"{item.name} is not supported for the selected LTX model.",
        )
        # An upscaling recipe (Refine Details) scales a smaller clip up to the cell.
        if (
            high_res is not None
            and not recipe.upscales_to_cell
            and not high_res_source_large_enough(
                video.metadata.metadata.width, video.metadata.metadata.height, high_res
            )
        ):
            raise HTTPError(
                422,
                f"The clip is smaller than {req.params.resolution}. Pick a lower resolution "
                "or a larger clip.",
                code="IC_LORA_SOURCE_TOO_SMALL",
            )
        return _PreparedIcLoraRecipe(recipe, item, stored, video, source_fps)

    def _require_high_res_cell(self, recipe_id: str) -> None:
        """422 for a 1440p or 2160p cell of a recipe the highResVideoToVideo flag still gates.

        An unknown id is not treated as a catalog id. The create path reports that id itself.
        """
        recipe = IC_LORA_RECIPES.get(recipe_id)
        if recipe is None:
            return
        if not high_res_flag_satisfied(recipe.catalog_id, flag_enabled=self._high_res_enabled()):
            raise HTTPError(422, HIGH_RES_NOT_ENABLED, code=HIGH_RES_NOT_ENABLED)

    def _create_synthesized_ic_lora(
        self,
        recipe_id: str,
        req: CreateIcLoraSheetRequest | CreateIcLoraTrackRequest,
        *,
        analytics_client: AnalyticsClient | None,
        analytics_requested_by: AnalyticsRequestedBy | None = None,
        generation_id: str | None = None,
    ) -> Generation:
        """Queue a run whose control clip the job makes. Its size and length come from the stored params."""
        self._require_local_generation_possible()
        try:
            recipe, item, stored = resolve_synthesized_ic_lora_create(
                recipe_id,
                req,
                catalog=self._require_catalog(),
                models_dir=self._models_dir(),
                hf_authenticated=self._hf_authenticated(),
                require_image=self._require_image_size,
            )
        except RecipeCreateError as exc:
            raise _ic_lora_create_error(exc) from exc
        self._require_offering_capability(
            req.params.model,
            "ic_lora",
            code="UNSUPPORTED_IC_LORA",
            message=f"{item.name} is not supported for the selected LTX model.",
        )
        return self._enqueue_ic_lora_recipe(
            recipe,
            stored,
            req,
            risk=self._preflight_queued_synthesized_ic_lora(recipe=recipe, item=item, stored=stored),
            analytics_client=analytics_client,
            analytics_requested_by=analytics_requested_by,
            generation_id=generation_id,
        )

    def _enqueue_ic_lora_recipe(
        self,
        recipe: IcLoraRecipe,
        stored: IcLoraRecipeStoredParams,
        req: CreateIcLoraRecipeRequest | CreateIcLoraSheetRequest | CreateIcLoraTrackRequest,
        *,
        risk: GenerationRisk | None,
        analytics_client: AnalyticsClient | None,
        analytics_requested_by: AnalyticsRequestedBy | None = None,
        generation_id: str | None = None,
    ) -> Generation:
        spec: dict[str, JsonValue] = {
            "params": self._dump_fast_params(
                stored, exclude={"scale", "variantId"}, uses_torch_compile=False
            ),
            "inputs": req.inputs.model_dump(mode="json", exclude_none=True),
        }
        return self._enqueue_generation(
            recipe.recipe_id,
            spec,
            contract_version=req.contract_version,
            analytics_client=analytics_client,
            analytics_requested_by=analytics_requested_by,
            generation_id=generation_id,
            risk=risk,
        )

    def create_extend(
        self,
        req: CreateExtendRequest,
        *,
        analytics_client: AnalyticsClient | None = None,
        analytics_requested_by: AnalyticsRequestedBy | None = None,
        generation_id: str | None = None,
    ) -> Generation:
        return self._enqueue_video_edit(
            EXTEND_FEATURE,
            req,
            "extend",
            code="UNSUPPORTED_EXTEND",
            message="Extend is not supported for the selected LTX model.",
            analytics_client=analytics_client,
            analytics_requested_by=analytics_requested_by,
            generation_id=generation_id,
        )

    def preflight_edit(self, req: EditPreflightRequest) -> EditPreflightResponse:
        """Advisory picker verdicts. Same asset checks as create (missing asset,
        wrong kind, oversized file fail identically); per-cell failures are
        returned, never raised, so the UI can disable with tooltips."""
        video = self._require_asset(
            req.assetId,
            slot="video",
            kind="video",
            max_bytes=MAX_VIDEO_BYTES,
        )
        if video.metadata.mediaType != "video":
            raise HTTPError(422, "INVALID_VIDEO_ASSET", code="INVALID_GENERATION_SPEC")
        return self._preflight_edit_matrix(
            feature=req.feature,
            asset_path=video.path,
            start_time=req.startTime,
            mode=req.mode,
            candidates=req.candidates,
            durations=req.durations,
        )

    def _enqueue_video_edit(
        self,
        feature: str,
        req: CreateRetakeRequest | CreateExtendRequest,
        capability: LtxCapabilityFeature,
        *,
        code: str,
        message: str,
        analytics_client: AnalyticsClient | None,
        analytics_requested_by: AnalyticsRequestedBy | None = None,
        extra: Callable[[int], None] | None = None,
        generation_id: str | None = None,
    ) -> Generation:
        video = self._require_asset(
            req.inputs.video.assetId,
            slot="video",
            kind="video",
            max_bytes=MAX_VIDEO_BYTES,
        )
        if video.metadata.mediaType != "video":
            raise HTTPError(422, "INVALID_VIDEO_ASSET", code="INVALID_GENERATION_SPEC")
        if extra is not None:
            extra(video.metadata.metadata.durationMs)
        self._require_offering_capability(
            req.params.model, capability, code=code, message=message
        )
        risk = self._preflight_edit_params(
            feature, asset_path=video.path, params=req.params
        )
        params_payload = self._dump_fast_params(req.params, uses_torch_compile=False)
        spec: dict[str, JsonValue] = {
            "params": params_payload,
            "inputs": req.inputs.model_dump(mode="json", exclude_none=True),
        }
        return self._enqueue_generation(
            feature,
            spec,
            contract_version=req.contract_version,
            analytics_client=analytics_client,
            analytics_requested_by=analytics_requested_by,
            generation_id=generation_id,
            risk=risk,
        )

    def _preflight_fast_params(
        self,
        *,
        offering: OfferingId,
        resolution: LTXVideoGenResolution,
        aspect_ratio: LTXQueuedAspectRatio,
        duration: int | None,
        fps: LTXVideoGenFps,
        pass_shape: PassShape,
    ) -> GenerationRisk | None:
        return self._preflight_queued_fast(
            offering=offering,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            duration=duration,
            fps=fps,
            pass_shape=pass_shape,
        )

    def _resolve_pass_shape(self, requested: PassShape) -> PassShape:
        never_split = self._state is not None and self._state.app_settings.never_split_into_parts
        return resolve_pass_shape(requested, never_split=never_split)

    @staticmethod
    def _resolve_queued_preflight_aspect(
        aspect_ratio: LTXQueuedAspectRatio,
        resolution: LTXVideoGenResolution,
        start_frame: AssetRecord | None,
        *,
        is_a2v: bool = False,
    ) -> LTXQueuedAspectRatio:
        """Explicit aspect passes through; auto resolves from the start frame.

        Same closest-match execute uses, so create admits exactly what the run
        budgets. Corrupt image metadata 422s like its audio sibling. A missing
        frame falls back to the anchor rather than refusing the queue insert —
        the A2V no-start default; unreachable for t2v/i2v (i2v creates require
        one, t2v rejects auto).
        """
        if aspect_ratio != "auto":
            return aspect_ratio
        if start_frame is None:
            return "16:9"
        metadata = start_frame.metadata
        if metadata.mediaType != "image":
            raise HTTPError(422, "INVALID_IMAGE_ASSET", code="INVALID_GENERATION_SPEC")
        # find_closest returns an element of options, so the cast is exact.
        return cast(
            LTXQueuedAspectRatio,
            find_closest_aspect_ratio_option(
                metadata.metadata.width,
                metadata.metadata.height,
                local_aspect_ratios(resolution, is_a2v=is_a2v),
            ),
        )

    def _preflight_edit_params(
        self,
        feature: str,
        *,
        asset_path: str,
        params: RetakeParams | ExtendParams,
    ) -> GenerationRisk | None:
        return self._preflight_queued_edit(
            feature=feature,
            asset_path=asset_path,
            params=params,
        )

    def _require_asset(
        self,
        asset_id: str,
        *,
        slot: str,
        kind: Literal["image", "audio", "video"],
        max_bytes: int | None = None,
    ) -> AssetRecord:
        try:
            asset = self._db.get_asset(asset_id)
        except UnavailableError as exc:
            raise HTTPError(503, "STORE_UNAVAILABLE") from exc
        missing = (
            f"{slot} image is no longer available"
            if kind == "image"
            else f"{slot} is no longer available"
        )
        if asset is None:
            raise HTTPError(
                422,
                missing,
                code="INPUT_ASSET_NOT_FOUND",
            )
        if asset.media_kind != kind:
            article = "an" if kind[0] in "aeiou" else "a"
            raise HTTPError(
                422,
                f"{slot} must reference {article} {kind} asset",
                code="INVALID_GENERATION_SPEC",
            )
        if max_bytes is not None:
            try:
                if Path(asset.path).stat().st_size > max_bytes:
                    raise HTTPError(400, "FILE_TOO_LARGE", code="FILE_TOO_LARGE")
            except OSError as exc:
                raise HTTPError(
                    422,
                    missing,
                    code="INPUT_ASSET_NOT_FOUND",
                ) from exc
        return asset

    def _require_image_asset(self, asset_id: str, *, slot: str) -> AssetRecord:
        return self._require_asset(asset_id, slot=slot, kind="image")

    def _require_image_size(self, asset_id: str) -> tuple[int, int]:
        image = self._require_image_asset(asset_id, slot="image")
        assert image.metadata.mediaType == "image"
        return image.metadata.metadata.width, image.metadata.metadata.height

    def _require_audio_asset(self, asset_id: str, *, slot: str) -> AssetRecord:
        return self._require_asset(asset_id, slot=slot, kind="audio")

    def _validate_retake_selection(
        self, start_time: float, duration: float, duration_ms: int
    ) -> None:
        source_seconds = duration_ms / 1000
        if source_seconds + 0.1 < 2.0:
            raise HTTPError(
                422,
                "This video is shorter than 2s. Replace it to continue.",
                code="INVALID_GENERATION_SPEC",
            )
        if start_time + duration > source_seconds + 0.1:
            raise HTTPError(
                422,
                "Selection is outside the usable video range",
                code="INVALID_GENERATION_SPEC",
            )

    def _require_offering_capability(
        self,
        offering: OfferingId,
        capability: LtxCapabilityFeature,
        *,
        code: str,
        message: str,
    ) -> None:
        try:
            require_capable_local_model(
                self._models_dir(),
                offering,
                capability,
                unsupported_code=code,
                unsupported_message=message,
            )
        except CapabilityFailedError as exc:
            raise HTTPError(409, exc.detail, code=exc.code) from exc

    def _enqueue_generation(
        self,
        feature: str,
        spec: dict[str, JsonValue],
        *,
        contract_version: int,
        analytics_client: AnalyticsClient | None = None,
        analytics_requested_by: AnalyticsRequestedBy | None = None,
        generation_id: str | None = None,
        risk: GenerationRisk | None = None,
        pass_shape: PassShape | None = None,
    ) -> Generation:
        try:
            self._executor_registry.validate_params(
                feature, spec, contract_version=contract_version
            )
        except (ValidationError, ValueError, CapabilityFailedError) as exc:
            if isinstance(exc, CapabilityFailedError):
                raise HTTPError(
                    422,
                    exc.detail,
                    code=exc.code or "INVALID_GENERATION_SPEC",
                ) from exc
            raise HTTPError(
                422,
                str(exc),
                code="INVALID_GENERATION_SPEC",
            ) from exc
        if risk is not None:
            spec = {**spec, "_risk": risk.model_dump(mode="json")}
        if pass_shape is not None:
            spec = {**spec, PASS_SHAPE_SPEC_KEY: pass_shape}
        if analytics_client is not None:
            analytics: dict[str, JsonValue] = {"client": analytics_client}
            if analytics_requested_by is not None:
                analytics["requested_by"] = analytics_requested_by
            spec = {**spec, "_analytics": analytics}
        try:
            record = self._db.insert_generation(
                feature,
                spec,
                contract_version=contract_version,
                generation_id=generation_id,
                lane=self._executor_registry.lane_for(feature, spec),
            )
        except UnavailableError as exc:
            raise HTTPError(503, "STORE_UNAVAILABLE") from exc
        except ValueError as exc:
            raise HTTPError(400, str(exc)) from exc
        except sqlite3.IntegrityError as exc:
            raise HTTPError(400, str(exc)) from exc
        self._queue_control.wake()
        return _generation_to_api(record)

    def retry_generation(self, generation_id: str) -> Generation:
        try:
            record = self._db.retry_generation(
                generation_id, lane=self._lane_for_stored(generation_id)
            )
        except UnavailableError as exc:
            raise HTTPError(503, "STORE_UNAVAILABLE") from exc
        except KeyError as exc:
            raise HTTPError(404, "GENERATION_NOT_FOUND") from exc
        except StatusError as exc:
            raise HTTPError(409, "INVALID_GENERATION_STATUS") from exc
        self._finished.release(generation_id)
        self._queue_control.wake()
        return _generation_to_api(record)

    def _lane_for_stored(self, generation_id: str) -> str | None:
        """The lane a stored job runs on now. None keeps the stored lane.

        The registry can differ from the one at insert (the kill switch, a removed provider).
        """
        try:
            stored = self._db.get_generation(generation_id)
        except UnavailableError:
            return None
        if stored is None:
            return None
        try:
            return self._executor_registry.lane_for(stored.feature, stored.spec)
        except ValueError:
            # The spec or the feature is no longer valid. The job fails when it runs.
            # It still needs a lane that has a runner.
            return self._executor_registry.lanes.resolve(stored.lane)

    def cancel_generation(self, generation_id: str) -> Generation:
        try:
            record = cancel_queued_generation(
                self._db,
                generation_id,
                self._queue_control.interrupt,
                uncancellable_lanes=self._executor_registry.lanes.uncancellable_ids,
            )
        except UnavailableError as exc:
            raise HTTPError(503, "STORE_UNAVAILABLE") from exc
        except KeyError as exc:
            raise HTTPError(404, "GENERATION_NOT_FOUND") from exc
        except (AttemptError, StatusError) as exc:
            raise HTTPError(409, "INVALID_GENERATION_STATUS") from exc
        self._queue_control.wake()
        return _generation_to_api(record)

    def get_queue_snapshot(self) -> QueueSnapshot:
        try:
            entries = self._db.list_queue_generations()
            self._finished.observe(entries, self._db.get_generation)
        except UnavailableError as exc:
            raise HTTPError(503, "STORE_UNAVAILABLE") from exc

        active = [
            self._active_entry(entry)
            for entry in entries
            if entry.generation.status in {"running", "cancelling"}
        ]
        queued = [
            self._queue_entry(entry, None)
            for entry in entries
            if entry.generation.status == "queued"
        ]
        done, failed, unseen_ids = self._finished.lists()
        return QueueSnapshot(
            active=active,
            queued=queued,
            done=[self._queue_entry(entry, None) for entry in done],
            failed=[self._queue_entry(entry, None) for entry in failed],
            unseen_ids=unseen_ids,
        )

    def remember_finished(self, generation_id: str) -> None:
        try:
            record = self._db.get_generation(generation_id)
            if record is None:
                return
            assets = tuple(
                asset
                for asset_id in collect_asset_ids(record.spec)
                if (asset := self._db.get_asset(asset_id)) is not None
            )
        except UnavailableError:
            return
        self._finished.remember(
            QueueGenerationRecord(generation=record, input_assets=assets)
        )

    def mark_done_seen(self, generation_id: str) -> QueueSnapshot:
        if not self._finished.mark_seen(generation_id):
            raise HTTPError(404, "GENERATION_NOT_FOUND")
        return self.get_queue_snapshot()

    def dismiss_done(self, generation_id: str) -> QueueSnapshot:
        if not self._finished.dismiss_done(generation_id):
            raise HTTPError(404, "GENERATION_NOT_FOUND")
        return self.get_queue_snapshot()

    def clear_done(
        self, *, feature_is_allowed: Callable[[str], bool] | None = None
    ) -> QueueSnapshot:
        self._finished.clear_done(feature_is_allowed)
        return self.get_queue_snapshot()

    def clear_failed(
        self, *, feature_is_allowed: Callable[[str], bool] | None = None
    ) -> QueueSnapshot:
        self._finished.clear_failed(feature_is_allowed)
        return self.get_queue_snapshot()

    def reorder_queue(
        self, generation_id: str, before_generation_id: str | None
    ) -> QueueSnapshot:
        try:
            self._db.reorder_queued_generation(generation_id, before_generation_id)
        except UnavailableError as exc:
            raise HTTPError(503, "STORE_UNAVAILABLE") from exc
        except KeyError as exc:
            raise HTTPError(404, "GENERATION_NOT_FOUND") from exc
        except StatusError as exc:
            raise HTTPError(409, "INVALID_GENERATION_STATUS") from exc
        self._queue_control.wake()
        return self.get_queue_snapshot()

    def _active_entry(self, entry: QueueGenerationRecord) -> QueueEntry:
        live = self._queue_control.live_state(entry.generation.id)
        if live is None:
            return self._queue_entry(entry, None)
        return self._queue_entry(
            entry,
            live.progress,
            cancellable=live.cancellable,
            api_status=live.api_status,
        )

    @staticmethod
    def _queue_entry(
        entry: QueueGenerationRecord,
        progress: QueueProgress | None,
        *,
        cancellable: bool = True,
        api_status: ApiActivityStatus | None = None,
    ) -> QueueEntry:
        return QueueEntry(
            generation=_generation_to_api(entry.generation),
            input_assets=[
                Asset.model_validate(asset.model_dump(mode="json"))
                for asset in entry.input_assets
            ],
            progress=progress,
            cancellable=cancellable,
            api_status=api_status,
        )

    def list_generations(self, feature: str | None = None, *, limit: int | None = None) -> list[Generation]:
        try:
            records = self._db.list_generations(feature, limit=limit)
        except UnavailableError as exc:
            raise HTTPError(503, "STORE_UNAVAILABLE") from exc
        return [_generation_to_api(record) for record in records]

    def list_recent_features(
        self, *, limit: int, allowed: Collection[str] | None = None
    ) -> list[str]:
        if allowed is not None and not self._keyframes_enabled():
            # Drop the hidden feature before the store applies the limit, so it does
            # not take a slot from an older feature that is offered.
            allowed = [name for name in allowed if name != KEYFRAMES_FEATURE]
        try:
            return self._db.list_recent_features(limit=limit, allowed=allowed)
        except UnavailableError as exc:
            raise HTTPError(503, "STORE_UNAVAILABLE") from exc

    def get_generation(self, generation_id: str) -> Generation:
        try:
            record = self._db.get_generation(generation_id)
        except UnavailableError as exc:
            raise HTTPError(503, "STORE_UNAVAILABLE") from exc
        if record is None:
            raise HTTPError(404, "GENERATION_NOT_FOUND")
        return _generation_to_api(record)

    def delete_generation(self, generation_id: str) -> None:
        try:
            deleted = self._db.delete_generation(generation_id)
        except UnavailableError as exc:
            raise HTTPError(503, "STORE_UNAVAILABLE") from exc
        except StatusError as exc:
            raise HTTPError(409, "INVALID_GENERATION_STATUS") from exc
        if not deleted:
            raise HTTPError(404, "GENERATION_NOT_FOUND")
        self._finished.forget(generation_id)
