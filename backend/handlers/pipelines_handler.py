"""Pipeline lifecycle handler."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING, TypeVar

from _routes._errors import HTTPError
from api_types import (
    EditPreflightCandidate,
    EditPreflightCell,
    EditPreflightResolution,
    EditPreflightResponse,
    EditPreflightSource,
    ExtendMode,
    ExtendParams,
    GenerationRisk,
    LTXLocalModelId,
    RetakeParams,
)
from handlers.base import StateHandlerBase
from handlers.text_handler import TextHandler
from handlers.video_resolution import (
    correct_frame_count,
    generation_letterbox,
    read_source_metadata,
    resolve_home_edit_size,
    resolve_requested_home_edit_size,
)
from runtime_config.ic_lora_tiling import IcLoraTiling
from runtime_config.model_download_specs import (
    IMG_GEN_MODEL_CP_ID,
    ltx_generation_bundle_on_disk,
    get_existing_cp_path,
    resolve_active_ltx_model_id,
)
from runtime_config.evict_retry import evict_retry_filter
from runtime_config.runtime_policy import LocalGenerationMode, streaming_prefetch_count_for_mode
from runtime_config.budget_admission import Admission, PricedJob, Refused, RiskAllowance, admit
from runtime_config.device_calibration import JobEnvelope
from runtime_config.eta import estimate_envelope_wall_time
from runtime_config.memory_meter import job_filter_machine_factor
from runtime_config.video_job_budget import (
    WINDOWS_ICLORA_CPU_STREAMING_RAM_GB,
    LocalJobFilter,
    max_edit_frames_that_load,
    price_edit_job,
    reject_message_and_code,
    video_job_reject,
    vram_snapshot_suffix,
)
from services.retake_pipeline.window import (
    MAX_INPUT_FRAMES,
    EditRefusal,
    prepare_extend_window,
    prepare_retake_window,
)
from services.interfaces import (
    A2VPipeline,
    DepthProcessorPipeline,
    FastVideoPipeline,
    ImageGenerationPipeline,
    GpuCleaner,
    GpuInfo,
    IcLoraPipeline,
    PoseProcessorPipeline,
    RetakePipeline,
    VideoPipelineModelType,
)
from server_utils.generation_liveness import liveness_hold
from services.services_utils import device_supports_fp8, get_device_type
from state.app_state_types import (
    A2VPipelineState,
    AppState,
    CpuSlot,
    GpuGeneration,
    GenerationRunning,
    GpuSlot,
    GpuSlotContent,
    ICLoraState,
    RetakePipelineState,
    VideoPipelineState,
)

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)

_StateT = TypeVar("_StateT", bound=GpuSlotContent)

# Resolution-only shrink advice for queued Home edit rejects. Home pads
# context back up to the cap, so a shorter duration does not shrink that job.
_EDIT_HOME_ACTION = "Lower the resolution."


@dataclass(frozen=True)
class _EditVerdict:
    """One edit cell verdict under the live mode. ``cause`` chains the window
    ValueError so create-time raises keep ``from exc`` chaining; the matrix
    drops it. ``ok`` means create would enqueue it. ``risk`` is the bet: the
    admitted one on ok, the one the mode refused otherwise (None for a hard
    refusal). Failing verdicts always set ``message``/``code``; an ok verdict
    with a bet carries the bet line as ``message``."""

    ok: bool
    message: str | None = None
    code: str | None = None
    cause: BaseException | None = None
    risk: GenerationRisk | None = None
    estimated_seconds: int | None = None


def _budget_verdict(
    admission: Admission,
    refusal: tuple[str, str] | None,
    *,
    job_filter: LocalJobFilter | None = None,
) -> _EditVerdict:
    if isinstance(admission, Refused):
        assert refusal is not None  # _decide_edit_budget pairs every refusal
        message, code = refusal
        return _EditVerdict(ok=False, message=message, code=code, risk=admission.bet)
    risk = admission.risk
    estimated_seconds: int | None = None
    if job_filter is not None:
        system = (
            job_filter.profile.system
            if job_filter.profile.system in ("Windows", "Linux", "Darwin")
            else "Windows"
        )
        est = estimate_envelope_wall_time(
            admission.envelope,
            system=system,
            machine_factor=job_filter_machine_factor(job_filter),
            conv_vae=job_filter.conv_vae,
        )
        estimated_seconds = None if est is None else est.seconds
    return _EditVerdict(
        ok=True,
        message=None if risk is None else risk.message,
        risk=risk,
        estimated_seconds=estimated_seconds,
    )


class PipelinesHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        text_handler: TextHandler,
        gpu_cleaner: GpuCleaner,
        fast_video_pipeline_class: type[FastVideoPipeline],
        image_generation_pipeline_class: type[ImageGenerationPipeline],
        ic_lora_pipeline_class: type[IcLoraPipeline],
        depth_processor_pipeline_class: type[DepthProcessorPipeline],
        pose_processor_pipeline_class: type[PoseProcessorPipeline],
        a2v_pipeline_class: type[A2VPipeline],
        retake_pipeline_class: type[RetakePipeline],
        config: RuntimeConfig,
        gpu_info: GpuInfo,
        high_res_enabled: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(state, lock, config, gpu_info=gpu_info)
        self._high_res_enabled = high_res_enabled or (lambda: False)
        self._text_handler = text_handler
        self._gpu_cleaner = gpu_cleaner
        self._fast_video_pipeline_class = fast_video_pipeline_class
        self._image_generation_pipeline_class = image_generation_pipeline_class
        self._ic_lora_pipeline_class = ic_lora_pipeline_class
        self._depth_processor_pipeline_class = depth_processor_pipeline_class
        self._pose_processor_pipeline_class = pose_processor_pipeline_class
        self._a2v_pipeline_class = a2v_pipeline_class
        self._retake_pipeline_class = retake_pipeline_class
        self._runtime_device = get_device_type(self.config.device)

    def _resolve_ltx_paths(self, model_id: LTXLocalModelId, gemma_root: str | None):
        from runtime_config.ltx_runtime_paths import ResolvedLtxModelPaths, resolve_ltx_runtime_paths
        from state.app_settings import resolved_use_conv_vae

        paths: ResolvedLtxModelPaths = resolve_ltx_runtime_paths(
            self.models_dir,
            model_id,
            gemma_root=gemma_root,
            use_conv_vae=resolved_use_conv_vae(self.state.app_settings),
        )
        return paths

    def _ensure_no_running_generation(self) -> None:
        match self.state.active_generation:
            case GpuGeneration(state=GenerationRunning()) if self.state.gpu_slot is not None:
                raise RuntimeError("Generation already running; cannot swap pipelines")
            case _:
                return

    def _assert_invariants(self) -> None:
        match self.state.gpu_slot:
            case GpuSlot(active_pipeline=active_pipeline):
                gpu_has_image_generation_pipeline = isinstance(active_pipeline, ImageGenerationPipeline)
            case _:
                gpu_has_image_generation_pipeline = False

        if gpu_has_image_generation_pipeline and self.state.cpu_slot is not None:
            raise RuntimeError("Invariant violation: image generation pipeline cannot be in both GPU and CPU slots")

    def _install_text_patches_if_needed(self) -> None:
        te = self.state.text_encoder
        if te is None:
            return
        te.service.install_patches(lambda: self.state)

    def _require_downloaded_ltx_model_id(
        self, model_id: LTXLocalModelId | None = None
    ) -> LTXLocalModelId:
        if model_id is not None:
            if not ltx_generation_bundle_on_disk(self.models_dir, model_id):
                raise HTTPError(409, "LTX_MODEL_NOT_INSTALLED")
            return model_id
        resolved = resolve_active_ltx_model_id(
            self.models_dir, self.state.app_settings.active_ltx_model_id
        )
        if resolved is None:
            raise HTTPError(409, "NO_DOWNLOADED_LTX_MODEL")
        return resolved

    def _compile_if_enabled(self, state: VideoPipelineState) -> VideoPipelineState:
        if not self.state.app_settings.use_torch_compile:
            return state
        if state.is_compiled:
            return state
        if self._runtime_device == "mps":
            logger.info("Skipping torch.compile() for %s - not supported on MPS", state.pipeline.pipeline_kind)
            return state
        if self.config.use_sage_attention:
            # SageAttention's Python-level kernel forces a Dynamo graph break at every
            # attention call. The resume graph Dynamo builds for the code after each
            # break doesn't inherit the video/audio tensors' pre-existing mark_dynamic
            # annotations, so it specializes their sequence-length dim to whatever
            # shape it first sees — then hard-fails (ConstraintViolationError) on any
            # later call with a different frame count or resolution. The two features
            # conflict; SageAttention already provides its own speedup, so skip
            # compilation rather than compiling a graph that can only serve one shape.
            logger.info(
                "Skipping torch.compile() for %s - conflicts with SageAttention's graph breaks",
                state.pipeline.pipeline_kind,
            )
            return state

        try:
            state.pipeline.compile_transformer()
            state.is_compiled = True
        except Exception as exc:
            logger.warning("Failed to compile transformer: %s", exc, exc_info=True)
        return state

    def _create_video_pipeline(
        self,
        model_type: VideoPipelineModelType,
        loras: list[tuple[str, float]] | None = None,
        *,
        load_mode: LocalGenerationMode,
        ltx_model_id: LTXLocalModelId | None = None,
    ) -> VideoPipelineState:
        model_id = self._require_downloaded_ltx_model_id(ltx_model_id)
        gemma_root = self._text_handler.resolve_gemma_root(model_id=model_id)
        paths = self._resolve_ltx_paths(model_id, gemma_root)

        pipeline = self._fast_video_pipeline_class.create(
            paths.checkpoint_path,
            paths.gemma_root,
            paths.upsampler_path,
            self.config.device,
            streaming_prefetch_count_for_mode(load_mode),
            loras=loras or [],
            offload_mode=self._offload_mode(load_mode),
            video_vae_path=paths.video_vae_path,
            audio_vae_path=paths.audio_vae_path,
            duration_head_path=paths.duration_head_path,
        )

        state = VideoPipelineState(
            pipeline=pipeline,
            is_compiled=False,
            ltx_model_id=model_id,
            loading_mode=load_mode,
            loras=tuple(loras) if loras else (),
            gemma_root=gemma_root,
            video_vae_path=paths.video_vae_path,
        )
        return self._compile_if_enabled(state)

    def unload_gpu_pipeline(self) -> None:
        with self._lock:
            self._ensure_no_running_generation()
            self.state.gpu_slot = None
            self._assert_invariants()
        self._gpu_cleaner.cleanup()

    def park_image_generation_pipeline_on_cpu(self) -> None:
        image_generation_pipeline: ImageGenerationPipeline | None = None

        with self._lock:
            if self.state.gpu_slot is None:
                return

            active = self.state.gpu_slot.active_pipeline
            if not isinstance(active, ImageGenerationPipeline):
                return

            if isinstance(self.state.active_generation, GpuGeneration) and isinstance(
                self.state.active_generation.state, GenerationRunning
            ):
                raise RuntimeError("Cannot park image generation pipeline while generation is running")

            image_generation_pipeline = active
            self.state.gpu_slot = None

        assert image_generation_pipeline is not None
        image_generation_pipeline.to("cpu")
        self._gpu_cleaner.cleanup()

        with self._lock:
            self.state.cpu_slot = CpuSlot(active_pipeline=image_generation_pipeline)
            self._assert_invariants()

    def load_image_generation_pipeline_to_gpu(self) -> ImageGenerationPipeline:
        resident = self._resident_matching(ImageGenerationPipeline, lambda _: True)
        if resident is not None:
            return resident

        image_generation_pipeline: ImageGenerationPipeline | None = None

        # Cold zit loads read weights off disk: same proof-of-life hold as
        # the GPU swaps so a slow load reads as busy, not hung.
        with liveness_hold():
            # A resident video pipeline is released before zit allocates, as in
            # _swap_gpu_pipeline: both models must not be resident at once on a
            # tight card. A queued image job often runs right after a video job.
            self._evict_gpu_pipeline_for_swap()
            with self._lock:
                match self.state.cpu_slot:
                    case CpuSlot(active_pipeline=stored):
                        image_generation_pipeline = stored
                        self.state.cpu_slot = None
                    case _:
                        image_generation_pipeline = None

            if image_generation_pipeline is None:
                zit_path = get_existing_cp_path(self.models_dir, IMG_GEN_MODEL_CP_ID)
                image_generation_pipeline = self._image_generation_pipeline_class.create(str(zit_path), self._runtime_device)
            else:
                image_generation_pipeline.to(self._runtime_device)

            self._gpu_cleaner.cleanup()

            with self._lock:
                self.state.gpu_slot = GpuSlot(active_pipeline=image_generation_pipeline)
                self._assert_invariants()

        return image_generation_pipeline

    def _evict_gpu_pipeline_for_swap(self) -> None:
        should_park_image_generation_pipeline = False
        should_cleanup = False

        with self._lock:
            self._ensure_no_running_generation()
            if self.state.gpu_slot is None:
                return

            active = self.state.gpu_slot.active_pipeline
            if isinstance(active, ImageGenerationPipeline):
                should_park_image_generation_pipeline = True
            else:
                self.state.gpu_slot = None
                self._assert_invariants()
                should_cleanup = True
            # Drop the local ref before cleanup runs below: in the drop arm
            # this frame is the last owner, and gc.collect() cannot free the
            # weights while it still names the pipeline. The evict-retry
            # re-samples free VRAM right after this returns.
            del active

        if should_park_image_generation_pipeline:
            self.park_image_generation_pipeline_on_cpu()
        elif should_cleanup:
            self._gpu_cleaner.cleanup()

    def evict_resident_pipeline(self) -> None:
        """Drop the resident pipeline (if any), freeing the VRAM it holds.

        Same mechanics as a pipeline swap: image pipelines park to CPU (preserved),
        video pipelines are dropped and rebuilt by the subsequent load on admit. No-op
        when the slot is empty. Callers must hold the GPU slot pre-start (no Running
        generation); the swap guard refuses otherwise. Shared by prompt enhancement
        (needs the VRAM for Gemma) and the budget evict-retry (needs a bare card to
        re-sample).
        """
        self._evict_gpu_pipeline_for_swap()

    def _resident_matching(
        self, cls: type[_StateT], matches: Callable[[_StateT], bool]
    ) -> _StateT | None:
        """Resident state when it is a ``cls`` serving the requested key, else None.

        Owns the reuse comparison so the loader frame never names a rejected
        pipeline: match-case bindings survive a failed guard, and a leaked
        name would keep the old weights alive through the swap's evict and
        re-create (transient double residency on a tight card). Plain
        ``isinstance`` + predicate here, so the hazard is absent by
        construction rather than contained.
        """
        with self._lock:
            slot = self.state.gpu_slot
            if not isinstance(slot, GpuSlot):
                return None
            active = slot.active_pipeline
            if isinstance(active, cls) and matches(active):
                return active
            return None

    def _swap_gpu_pipeline(self, create: Callable[[], _StateT]) -> _StateT:
        """Cold-load path shared by every GPU pipeline loader.

        Holds proof-of-life across evict + create (a slow weights read / MPS
        graph build can stall the GIL past the kill threshold — without the
        hold a slow load reads as hung, not busy) and installs the result.
        """
        with liveness_hold():
            self._evict_gpu_pipeline_for_swap()
            state = create()
            with self._lock:
                self.state.gpu_slot = GpuSlot(active_pipeline=state)
                self._assert_invariants()
            return state

    def load_gpu_pipeline(
        self,
        model_type: VideoPipelineModelType,
        loras: list[tuple[str, float]] | None = None,
        mode: LocalGenerationMode | None = None,
        *,
        ltx_model_id: LTXLocalModelId | None = None,
    ) -> VideoPipelineState:
        self._install_text_patches_if_needed()

        load_mode = mode or self.config.local_generations_mode
        requested_loras = tuple(loras) if loras else ()
        requested_model_id = self._require_downloaded_ltx_model_id(ltx_model_id)
        requested_gemma_root = self._text_handler.resolve_gemma_root(model_id=requested_model_id)
        requested_video_vae_path = self._resolve_ltx_paths(
            requested_model_id, requested_gemma_root
        ).video_vae_path
        state = self._resident_matching(
            VideoPipelineState,
            lambda s: (
                s.pipeline.pipeline_kind == model_type
                and s.ltx_model_id == requested_model_id
                and s.loading_mode == load_mode
                and s.loras == requested_loras
                and s.gemma_root == requested_gemma_root
                and s.video_vae_path == requested_video_vae_path
            ),
        )

        if state is None:
            state = self._swap_gpu_pipeline(
                lambda: self._create_video_pipeline(
                    model_type, loras=loras, load_mode=load_mode, ltx_model_id=requested_model_id
                )
            )

        return state

    def create_depth_processor(self, model_path: str) -> DepthProcessorPipeline:
        return self._depth_processor_pipeline_class.create(model_path, self.config.device)

    def create_pose_processor(self, pose_model_path: str, person_detector_model_path: str) -> PoseProcessorPipeline:
        return self._pose_processor_pipeline_class.create(
            pose_model_path, person_detector_model_path, self.config.device
        )

    def free_unreferenced_gpu_memory(self) -> None:
        self._gpu_cleaner.cleanup()

    def load_ic_lora(
        self,
        lora_path: str,
        depth_model_path: str | None = None,
        lora_strength: float = 1.0,
        mode: LocalGenerationMode | None = None,
        *,
        ltx_model_id: LTXLocalModelId | None = None,
        stage_2_ic_lora: bool = False,
        tiling: IcLoraTiling | None = None,
    ) -> ICLoraState:
        self._install_text_patches_if_needed()

        load_mode = mode or self.config.local_generations_mode
        prefetch = streaming_prefetch_count_for_mode(load_mode)
        # GenSpace omits ltx_model_id and follows Settings. A queued Home job passes
        # the offering's local checkpoint so Settings stays put. The text encoder
        # follows that same checkpoint: 2.3 and 2.5 do not share one.
        model_id = self._require_downloaded_ltx_model_id(ltx_model_id)
        gemma_root = self._text_handler.resolve_gemma_root(model_id=model_id)
        paths = self._resolve_ltx_paths(model_id, gemma_root)
        state = self._resident_matching(
            ICLoraState,
            lambda s: (
                s.lora_path == lora_path
                and s.stage_2_ic_lora == stage_2_ic_lora
                and s.tiling == tiling
                and s.depth_model_path == depth_model_path
                and s.lora_strength == lora_strength
                and s.loading_mode == load_mode
                and s.gemma_root == gemma_root
                and s.ltx_model_id == model_id
                and s.video_vae_path == paths.video_vae_path
            ),
        )

        if state is None:
            def _create() -> ICLoraState:
                pipeline = self._ic_lora_pipeline_class.create(
                    paths.checkpoint_path,
                    paths.gemma_root,
                    paths.upsampler_path,
                    lora_path,
                    self.config.device,
                    prefetch,
                    lora_strength,
                    offload_mode=self._offload_mode(
                        load_mode,
                        cpu_ram_floor_gb=WINDOWS_ICLORA_CPU_STREAMING_RAM_GB,
                    ),
                    video_vae_path=paths.video_vae_path,
                    audio_vae_path=paths.audio_vae_path,
                    duration_head_path=paths.duration_head_path,
                    stage_2_ic_lora=stage_2_ic_lora,
                    tiling=tiling,
                )
                depth_pipeline = (
                    self.create_depth_processor(depth_model_path)
                    if depth_model_path is not None
                    else None
                )
                return ICLoraState(
                    pipeline=pipeline,
                    lora_path=lora_path,
                    depth_pipeline=depth_pipeline,
                    depth_model_path=depth_model_path,
                    ltx_model_id=model_id,
                    loading_mode=load_mode,
                    lora_strength=lora_strength,
                    gemma_root=gemma_root,
                    video_vae_path=paths.video_vae_path,
                    stage_2_ic_lora=stage_2_ic_lora,
                    tiling=tiling,
                )

            state = self._swap_gpu_pipeline(_create)
        return state

    def load_a2v_pipeline(
        self,
        loras: list[tuple[str, float]] | None = None,
        mode: LocalGenerationMode | None = None,
        *,
        ltx_model_id: LTXLocalModelId | None = None,
    ) -> A2VPipelineState:
        self._install_text_patches_if_needed()

        load_mode = mode or self.config.local_generations_mode
        requested_loras = tuple(loras) if loras else ()
        model_id = self._require_downloaded_ltx_model_id(ltx_model_id)
        gemma_root = self._text_handler.resolve_gemma_root(model_id=model_id)
        paths = self._resolve_ltx_paths(model_id, gemma_root)
        state = self._resident_matching(
            A2VPipelineState,
            lambda s: (
                s.ltx_model_id == model_id
                and s.loading_mode == load_mode
                and s.loras == requested_loras
                and s.gemma_root == gemma_root
                and s.video_vae_path == paths.video_vae_path
            ),
        )

        if state is None:
            def _create() -> A2VPipelineState:
                pipeline = self._a2v_pipeline_class.create(
                    paths.checkpoint_path,
                    paths.gemma_root,
                    paths.upsampler_path,
                    self.config.device,
                    streaming_prefetch_count_for_mode(load_mode),
                    loras=loras or [],
                    offload_mode=self._offload_mode(load_mode),
                    video_vae_path=paths.video_vae_path,
                    audio_vae_path=paths.audio_vae_path,
                    duration_head_path=paths.duration_head_path,
                )
                return A2VPipelineState(
                    pipeline=pipeline,
                    ltx_model_id=model_id,
                    loading_mode=load_mode,
                    loras=requested_loras,
                    gemma_root=gemma_root,
                    video_vae_path=paths.video_vae_path,
                )

            state = self._swap_gpu_pipeline(_create)
        return state

    def _admit_edit_budget(
        self,
        priced: PricedJob,
        *,
        job_filter: LocalJobFilter,
        queued_home: bool = False,
        allowance: RiskAllowance | None = None,
    ) -> tuple[Admission, tuple[str, str] | None]:
        admission = admit(
            priced,
            allowance if allowance is not None else RiskAllowance.for_mode(job_filter.budget_mode),
        )
        if not isinstance(admission, Refused):
            return admission, None
        reject = video_job_reject(admission)
        if queued_home:
            return admission, reject_message_and_code(
                reject,
                job_filter,
                action=_EDIT_HOME_ACTION,
                free_short_action=_EDIT_HOME_ACTION,
            )
        return admission, reject_message_and_code(reject, job_filter)

    def _decide_edit_budget(
        self,
        width: int,
        height: int,
        frames: int,
        *,
        job_filter: LocalJobFilter,
        queued_home: bool = False,
        allowance: RiskAllowance | None = None,
    ) -> tuple[Admission, tuple[str, str] | None]:
        """Edit admission plus the refusal pair, if it refuses.

        Single decide + message mapping shared by the raise path
        (``_edit_load_mode``) and the verdict path (create-time preflight,
        preflight matrix): one decide call, one message builder, no drift.
        """
        return self._admit_edit_budget(
            price_edit_job(width, height, frames, job_filter=job_filter),
            job_filter=job_filter,
            queued_home=queued_home,
            allowance=allowance,
        )

    def _edit_load_mode(
        self,
        width: int | None,
        height: int | None,
        frames: int | None,
        *,
        queued_home: bool = False,
        allow_evict_retry: bool = False,
        allowance: RiskAllowance | None = None,
    ) -> tuple[LocalGenerationMode, JobEnvelope | None]:
        """Per-job full vs stream, and the priced envelope. Missing shape keeps the process mode.

        Does not change ``local_generations_mode``. Uses the edit headroom, so a
        31 GB card streams a long 720p retake/extend as well as 1080p. Real
        loads pass ``allow_evict_retry``: a total-possible CUDA reject whose
        shortfall live weights cover gets one evict-and-resample before admission.
        ``allowance=None`` follows the filter's live mode (every request-scoped
        caller); executors pass the stored allowance so a mode flip never
        re-gates a queued job. An over bet streams, including when the
        shortfall later grows. An unmeasured bet does not cover a later
        memory miss. Hard rejects raise in every mode.
        """
        if width is None or height is None or frames is None:
            return self.config.local_generations_mode, None
        job_filter = self._local_job_filter()
        priced = price_edit_job(width, height, frames, job_filter=job_filter)
        if allow_evict_retry:
            rebuilt = evict_retry_filter(
                priced=priced,
                job_filter=job_filter,
                live_allocated_bytes=self._gpu.get_cuda_live_allocated_bytes(),
                reserved_bytes=self._gpu.get_cuda_reserved_bytes(),
                log_prefix="video-edit",
                evict=self.evict_resident_pipeline,
                rebuild=self._local_job_filter,
            )
            if rebuilt is not None:
                job_filter = rebuilt
                priced = price_edit_job(width, height, frames, job_filter=job_filter)
        admission, refusal = self._admit_edit_budget(
            priced,
            job_filter=job_filter,
            queued_home=queued_home,
            allowance=allowance,
        )
        snapshot = vram_snapshot_suffix(
            allocated_bytes=self._gpu.get_cuda_live_allocated_bytes(),
            reserved_bytes=self._gpu.get_cuda_reserved_bytes(),
            free_bytes=job_filter.budget.free_vram_bytes,
        )
        if isinstance(admission, Refused):
            assert refusal is not None  # _decide_edit_budget pairs every refusal
            logger.info(
                "[video-edit] job budget seq=%.0f full=%.1fGiB stream=%.1fGiB -> reject:%s%s",
                priced.seq,
                priced.full_gib,
                priced.stream_gib,
                admission.reject.reason,
                snapshot,
            )
            message, code = refusal
            raise HTTPError(422, message, code=code)
        risk = admission.risk
        logger.info(
            "[video-edit] job budget seq=%.0f full=%.1fGiB stream=%.1fGiB -> %s%s%s",
            priced.seq,
            priced.full_gib,
            priced.stream_gib,
            admission.load.mode,
            (
                f" risk:{risk.tier} over_gib={risk.over_gib} over_ram_gb={risk.over_ram_gb}"
                if risk is not None
                else ""
            ),
            snapshot,
        )
        return admission.load.mode, admission.envelope

    def high_res_video_enabled(self) -> bool:
        return self._high_res_enabled()

    def edit_encode_frame_cap(
        self,
        width: int,
        height: int,
        *,
        job_filter: LocalJobFilter | None = None,
        total_only: bool = False,
    ) -> int:
        """Longest Home encode window this machine will load at ``width``×``height``.

        ``width``×``height`` is the encode (canvas) size, not the picture size:
        the budget prices what the model encodes. Callers letterbox first.
        Does not change GenSpace, which never asks for this cap. A 31 GB card
        keeps a long 720p window and shortens 1080p below 505 frames. A 1440p
        or 4K window shortens the same way. When no window loads, the cap is
        returned and the load gate refuses with the figures.
        ``job_filter`` shares one free-VRAM sample across a preflight; callers
        that omit it (executors) sample fresh. Executors pass ``total_only``:
        the resident pipeline is still loaded when they cap, so a free sample
        would shorten the window before the load gate's evict-retry can
        reclaim those weights — the gate stays the free check.
        """
        return max_edit_frames_that_load(
            width,
            height,
            frame_cap=MAX_INPUT_FRAMES,
            job_filter=(
                job_filter
                if job_filter is not None
                else self._local_job_filter(total_only=total_only)
            ),
        )

    def preflight_queued_edit_job(
        self,
        *,
        feature: str,
        asset_path: str,
        params: RetakeParams | ExtendParams,
    ) -> GenerationRisk | None:
        """Create-time mirror of the executors' window + budget checks.

        Runs the same verdicts execute would — the same window math on the
        same file metadata, the same frame cap, the same load gate — so an
        over-budget create 422s here instead of enqueueing a job that fails
        in the runner with a bare code. Raises HTTPError; returns the bet
        annotation for an admitted risky job, else None.
        ``feature`` must agree with the params type (wiring check only —
        the params type is authoritative for dispatch).
        """
        if isinstance(params, RetakeParams):
            if feature != "retake":
                raise AssertionError(
                    f"preflight feature/params mismatch: {feature!r} with RetakeParams"
                )
            return self._preflight_retake(asset_path, params)
        if feature != "extend":
            raise AssertionError(
                f"preflight feature/params mismatch: {feature!r} with ExtendParams"
            )
        return self._preflight_extend(asset_path, params)

    def _retake_verdict(
        self,
        *,
        fps: float,
        target_frames: int,
        target_width: int,
        target_height: int,
        frame_cap: int,
        job_filter: LocalJobFilter,
        start_time: float,
        duration: float,
    ) -> _EditVerdict:
        """Retake window + budget verdict. Window math is shared with the executor."""
        prepared = prepare_retake_window(
            fps=fps,
            target_frames=target_frames,
            target_width=target_width,
            target_height=target_height,
            frame_cap=frame_cap,
            start_time=start_time,
            duration=duration,
        )
        if isinstance(prepared, EditRefusal):
            return _EditVerdict(
                ok=False,
                message=prepared.message,
                code=prepared.code,
                cause=prepared.cause,
            )
        # Window math and refusal copy read the picture size; the budget reads
        # the letterboxed canvas the model actually encodes (never smaller).
        canvas = generation_letterbox(target_width, target_height)
        return _budget_verdict(
            *self._decide_edit_budget(
                canvas.canvas_width,
                canvas.canvas_height,
                prepared.window.encode_frames,
                job_filter=job_filter,
                queued_home=True,
            ),
            job_filter=job_filter,
        )

    def _extend_verdict(
        self,
        *,
        fps: float,
        source_frames: int,
        target_width: int,
        target_height: int,
        frame_cap: int,
        job_filter: LocalJobFilter,
        duration: float,
        mode: ExtendMode,
    ) -> _EditVerdict:
        """Extend window + budget verdict. Window math is shared with the executor."""
        prepared = prepare_extend_window(
            fps=fps,
            source_frames=source_frames,
            target_width=target_width,
            target_height=target_height,
            frame_cap=frame_cap,
            duration=duration,
            mode=mode,
        )
        if isinstance(prepared, EditRefusal):
            return _EditVerdict(
                ok=False,
                message=prepared.message,
                code=prepared.code,
                cause=prepared.cause,
            )
        # Same picture/canvas split as the retake verdict: messages name the
        # picture, the budget prices the canvas.
        canvas = generation_letterbox(target_width, target_height)
        return _budget_verdict(
            *self._decide_edit_budget(
                canvas.canvas_width,
                canvas.canvas_height,
                prepared.window.context_frames + prepared.extend_frames,
                job_filter=job_filter,
                queued_home=True,
            ),
            job_filter=job_filter,
        )

    @staticmethod
    def _raise_verdict_or_risk(verdict: _EditVerdict) -> GenerationRisk | None:
        """Raise a failing verdict as the create-time 422, else hand back the
        admitted bet for the annotation."""
        if verdict.ok:
            return verdict.risk
        assert verdict.message is not None and verdict.code is not None
        if verdict.cause is not None:
            raise HTTPError(422, verdict.message, code=verdict.code) from verdict.cause
        raise HTTPError(422, verdict.message, code=verdict.code)

    def _preflight_retake(self, asset_path: str, params: RetakeParams) -> GenerationRisk | None:
        """Create-time window + budget checks for a queued retake.

        Same verdicts execute would reach (shared window math, same frame
        cap, same load gate), raised as the create 422 instead of failing
        in the runner. An admitted risky job returns its annotation.
        """
        fps, source_width, source_height, source_frames = read_source_metadata(
            asset_path
        )
        target_frames = correct_frame_count(source_frames)
        target_width, target_height = resolve_requested_home_edit_size(
            params.resolution,
            source_width,
            source_height,
            high_res=self.high_res_video_enabled(),
        )
        # Queued admission gates on total ("will it ever fit"): a running job
        # depresses free, and the queue exists to wait for a free slot.
        job_filter = self._local_job_filter(total_only=True)
        canvas = generation_letterbox(target_width, target_height)
        frame_cap = self.edit_encode_frame_cap(
            canvas.canvas_width, canvas.canvas_height, job_filter=job_filter
        )
        return self._raise_verdict_or_risk(
            self._retake_verdict(
                fps=fps,
                target_frames=target_frames,
                target_width=target_width,
                target_height=target_height,
                frame_cap=frame_cap,
                job_filter=job_filter,
                start_time=params.startTime,
                duration=params.duration,
            ),
        )

    def _preflight_extend(self, asset_path: str, params: ExtendParams) -> GenerationRisk | None:
        """Create-time window + budget checks for a queued extend.

        Same verdicts execute would reach (shared window math, same frame
        cap, same load gate), raised as the create 422 instead of failing
        in the runner. An admitted risky job returns its annotation.
        """
        fps, source_width, source_height, source_frames = read_source_metadata(
            asset_path
        )
        # Min-length only, like the executor: window math uses the raw count.
        correct_frame_count(source_frames)
        target_width, target_height = resolve_requested_home_edit_size(
            params.resolution,
            source_width,
            source_height,
            high_res=self.high_res_video_enabled(),
        )
        # Queued admission gates on total ("will it ever fit"): a running job
        # depresses free, and the queue exists to wait for a free slot.
        job_filter = self._local_job_filter(total_only=True)
        canvas = generation_letterbox(target_width, target_height)
        frame_cap = self.edit_encode_frame_cap(
            canvas.canvas_width, canvas.canvas_height, job_filter=job_filter
        )
        return self._raise_verdict_or_risk(
            self._extend_verdict(
                fps=fps,
                source_frames=source_frames,
                target_width=target_width,
                target_height=target_height,
                frame_cap=frame_cap,
                job_filter=job_filter,
                duration=params.duration,
                mode=params.mode,
            ),
        )

    @staticmethod
    def _max_passing_duration(
        verdict_ok: Callable[[float], bool], lo: float, hi: float
    ) -> float:
        """Bisect the pass/fail boundary. ``verdict_ok(lo)`` must pass; ``hi``
        is the feature cap. Verdicts are monotone decreasing in duration (a
        longer selection/extension never fits where a shorter one fails —
        windows only grow, budgets only rise), so the boundary is unique."""
        if verdict_ok(hi):
            return hi
        for _ in range(8):
            mid = (lo + hi) / 2
            if mid == lo or mid == hi:
                break
            if verdict_ok(mid):
                lo = mid
            else:
                hi = mid
        return lo

    def preflight_edit_matrix(
        self,
        *,
        feature: str,
        asset_path: str,
        start_time: float,
        mode: ExtendMode,
        candidates: list[EditPreflightCandidate],
        durations: list[float],
    ) -> EditPreflightResponse:
        """Advisory per-cell verdicts for Home retake/extend pickers.

        One metadata read, one free-VRAM sample; every cell runs the same
        window + budget verdicts the create path raises with, so a failing
        cell reads exactly like its create 422. Raises like create on
        unreadable/too-short sources; per-cell failures never raise.
        """
        fps, source_width, source_height, source_frames = read_source_metadata(
            asset_path
        )
        target_frames = source_frames
        if feature == "retake":
            target_frames = correct_frame_count(source_frames)
            hi_cap = target_frames / fps - start_time
        elif feature == "extend":
            # Min-length only, like the executor: window math uses the raw count.
            correct_frame_count(source_frames)
            hi_cap = 20.0
        else:
            raise AssertionError(f"unknown edit feature: {feature!r}")
        # Display on total: advisory cells must not grey total-fitting shapes
        # just because free is momentarily depressed; create stays authoritative.
        job_filter = self._local_job_filter(total_only=True)
        min_duration = min(durations)

        resolutions: list[EditPreflightResolution] = []
        cells: list[EditPreflightCell] = []
        for candidate in candidates:
            target_width, target_height = resolve_home_edit_size(
                candidate.width,
                candidate.height,
                source_width,
                source_height,
                high_res=self.high_res_video_enabled(),
            )
            canvas = generation_letterbox(target_width, target_height)
            frame_cap = self.edit_encode_frame_cap(
                canvas.canvas_width, canvas.canvas_height, job_filter=job_filter
            )

            # One branch per candidate, not per cell: the verdict closure below
            # is called per duration plus ~8 bisection probes.
            if feature == "retake":

                def verdict_at(d: float) -> _EditVerdict:
                    return self._retake_verdict(
                        fps=fps,
                        target_frames=target_frames,
                        target_width=target_width,
                        target_height=target_height,
                        frame_cap=frame_cap,
                        job_filter=job_filter,
                        start_time=start_time,
                        duration=d,
                    )

            else:

                def verdict_at(d: float) -> _EditVerdict:
                    return self._extend_verdict(
                        fps=fps,
                        source_frames=source_frames,
                        target_width=target_width,
                        target_height=target_height,
                        frame_cap=frame_cap,
                        job_filter=job_filter,
                        duration=d,
                        mode=mode,
                    )

            if not verdict_at(min_duration).ok or hi_cap <= 0:
                max_duration = 0.0
            elif hi_cap <= min_duration:
                max_duration = hi_cap
            else:
                max_duration = self._max_passing_duration(
                    lambda d: verdict_at(d).ok,
                    min_duration,
                    hi_cap,
                )
            # Floor to the 0.5s slider grid; the floor of a passing point passes.
            max_duration = max(0.0, math.floor(max_duration * 2) / 2)
            resolutions.append(
                EditPreflightResolution(
                    key=candidate.key,
                    width=target_width,
                    height=target_height,
                    maxDurationSeconds=max_duration,
                )
            )
            for duration in durations:
                verdict = verdict_at(duration)
                cells.append(
                    EditPreflightCell(
                        key=candidate.key,
                        duration=duration,
                        ok=verdict.ok,
                        message=verdict.message,
                        code=verdict.code,
                        tier=None if verdict.risk is None else verdict.risk.tier,
                        over_gib=None if verdict.risk is None else verdict.risk.over_gib,
                        over_ram_gb=None if verdict.risk is None else verdict.risk.over_ram_gb,
                        shortfall_kind=(
                            None if verdict.risk is None else verdict.risk.shortfall_kind
                        ),
                        readout=None if verdict.risk is None else verdict.risk.readout,
                        estimated_seconds=verdict.estimated_seconds,
                    )
                )
        return EditPreflightResponse(
            source=EditPreflightSource(
                width=source_width,
                height=source_height,
                fps=fps,
                frames=source_frames,
                durationSeconds=source_frames / fps,
            ),
            resolutions=resolutions,
            cells=cells,
            budgetMode=job_filter.budget_mode,
        )

    def load_retake_pipeline(
        self,
        *,
        distilled: bool = True,
        ltx_model_id: LTXLocalModelId | None = None,
        width: int | None = None,
        height: int | None = None,
        frames: int | None = None,
        queued_home: bool = False,
        allowance: RiskAllowance | None = None,
    ) -> tuple[RetakePipelineState, JobEnvelope | None]:
        """The pipeline for this edit job, and the envelope a clean run of it records."""
        self._install_text_patches_if_needed()

        quantized = device_supports_fp8(self.config.device)
        model_id = self._require_downloaded_ltx_model_id(ltx_model_id)
        # The text encoder follows the checkpoint being loaded, not Settings: a queued Home
        # job can load 2.3 while Settings stays on 2.5, and the two do not share an encoder.
        gemma_root = self._text_handler.resolve_gemma_root(model_id=model_id)
        paths = self._resolve_ltx_paths(model_id, gemma_root)
        load_mode, envelope = self._edit_load_mode(
            width,
            height,
            frames,
            queued_home=queued_home,
            allow_evict_retry=True,
            allowance=allowance,
        )

        state = self._resident_matching(
            RetakePipelineState,
            lambda s: (
                s.distilled == distilled
                and s.quantized == quantized
                and s.gemma_root == gemma_root
                and s.ltx_model_id == model_id
                and s.video_vae_path == paths.video_vae_path
                and s.loading_mode == load_mode
            ),
        )

        if state is None:
            def _create() -> RetakePipelineState:
                from ltx_core.quantization.fp8_cast import build_policy as build_fp8_cast_policy

                quantization = build_fp8_cast_policy(paths.checkpoint_path) if quantized else None
                pipeline = self._retake_pipeline_class.create(
                    checkpoint_path=paths.checkpoint_path,
                    gemma_root=paths.gemma_root,
                    device=self.config.device,
                    streaming_prefetch_count=streaming_prefetch_count_for_mode(load_mode),
                    loras=[],
                    quantization=quantization,
                    offload_mode=self._offload_mode(
                        load_mode,
                        cpu_ram_floor_gb=None,
                    ),
                    video_vae_path=paths.video_vae_path,
                    audio_vae_path=paths.audio_vae_path,
                    duration_head_path=paths.duration_head_path,
                )
                return RetakePipelineState(
                    pipeline=pipeline,
                    distilled=distilled,
                    quantized=quantized,
                    ltx_model_id=model_id,
                    loading_mode=load_mode,
                    gemma_root=gemma_root,
                    video_vae_path=paths.video_vae_path,
                )

            state = self._swap_gpu_pipeline(_create)
        return state, envelope
