"""Pipeline lifecycle handler."""

from __future__ import annotations

import logging
from threading import RLock
from typing import TYPE_CHECKING

from _routes._errors import HTTPError
from api_types import LTXLocalModelId
from handlers.base import StateHandlerBase
from handlers.text_handler import TextHandler
from runtime_config.ic_lora_tiling import IcLoraTiling
from runtime_config.model_download_specs import (
    IMG_GEN_MODEL_CP_ID,
    ltx_generation_bundle_on_disk,
    get_existing_cp_path,
    resolve_active_ltx_model_id,
)
from runtime_config.runtime_policy import LocalGenerationMode, streaming_prefetch_count_for_mode
from runtime_config.video_job_budget import (
    EDIT_JOB_TOO_LARGE_MESSAGE,
    LOCAL_GENERATION_UNSUPPORTED,
    LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
    VIDEO_JOB_TOO_LARGE,
    VIDEO_JOB_TOO_LARGE_MESSAGE,
    VideoJobReject,
    decide_edit_job,
    max_edit_frames_that_load,
    memory_gb_for_job,
)
from services.retake_pipeline.window import MAX_INPUT_FRAMES
from services.interfaces import (
    A2VPipeline,
    DepthProcessorPipeline,
    FastVideoPipeline,
    ImageGenerationPipeline,
    GpuCleaner,
    IcLoraPipeline,
    PoseProcessorPipeline,
    RetakePipeline,
    VideoPipelineModelType,
)
from services.services_utils import device_supports_fp8, get_device_type
from state.app_state_types import (
    A2VPipelineState,
    AppState,
    CpuSlot,
    GpuGeneration,
    GenerationRunning,
    GpuSlot,
    ICLoraState,
    RetakePipelineState,
    VideoPipelineState,
)

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)


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
    ) -> None:
        super().__init__(state, lock, config)
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

    def _pipeline_matches_model_type(self, model_type: VideoPipelineModelType) -> bool:
        match self.state.gpu_slot:
            case GpuSlot(active_pipeline=VideoPipelineState(pipeline=pipeline)):
                return pipeline.pipeline_kind == model_type
            case _:
                return False

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
        with self._lock:
            if self.state.gpu_slot is not None:
                active = self.state.gpu_slot.active_pipeline
                if isinstance(active, ImageGenerationPipeline):
                    return active
                self._ensure_no_running_generation()

        image_generation_pipeline: ImageGenerationPipeline | None = None

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

        if should_park_image_generation_pipeline:
            self.park_image_generation_pipeline_on_cpu()
        elif should_cleanup:
            self._gpu_cleaner.cleanup()

    def evict_gpu_pipeline_for_prompt_enhancement(self) -> None:
        """Free whatever big pipeline is resident before a standalone Gemma enhance call.

        The prompt enhancer never claims the GPU slot itself — it loads/frees Gemma inside a
        single call, same as every other local Gemma use in this codebase — but it still needs
        the VRAM a resident video/image pipeline is holding.
        """
        self._evict_gpu_pipeline_for_swap()

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
        state: VideoPipelineState | None = None
        with self._lock:
            if self._pipeline_matches_model_type(model_type):
                match self.state.gpu_slot:
                    case GpuSlot(
                        active_pipeline=VideoPipelineState() as existing_state
                    ) if (
                        existing_state.ltx_model_id == requested_model_id
                        and existing_state.loading_mode == load_mode
                        and existing_state.loras == requested_loras
                        and existing_state.gemma_root == requested_gemma_root
                        and existing_state.video_vae_path == requested_video_vae_path
                    ):
                        state = existing_state
                    case _:
                        pass

        if state is None:
            self._evict_gpu_pipeline_for_swap()
            state = self._create_video_pipeline(
                model_type, loras=loras, load_mode=load_mode, ltx_model_id=requested_model_id
            )
            with self._lock:
                self.state.gpu_slot = GpuSlot(active_pipeline=state)
                self._assert_invariants()

        return state

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
        with self._lock:
            match self.state.gpu_slot:
                case GpuSlot(
                    active_pipeline=ICLoraState(
                        lora_path=current_lora_path,
                        depth_model_path=current_depth_model_path,
                        lora_strength=current_lora_strength,
                        loading_mode=current_loading_mode,
                        gemma_root=current_gemma_root,
                        ltx_model_id=current_model_id,
                        video_vae_path=current_video_vae_path,
                        stage_2_ic_lora=current_stage_2_ic_lora,
                        tiling=current_tiling,
                    ) as state
                ) if (
                    current_lora_path == lora_path
                    and current_stage_2_ic_lora == stage_2_ic_lora
                    and current_tiling == tiling
                    and current_depth_model_path == depth_model_path
                    and current_lora_strength == lora_strength
                    and current_loading_mode == load_mode
                    and current_gemma_root == gemma_root
                    and current_model_id == model_id
                    and current_video_vae_path == paths.video_vae_path
                ):
                    return state
                case _:
                    pass

        self._evict_gpu_pipeline_for_swap()

        pipeline = self._ic_lora_pipeline_class.create(
            paths.checkpoint_path,
            paths.gemma_root,
            paths.upsampler_path,
            lora_path,
            self.config.device,
            prefetch,
            lora_strength,
            video_vae_path=paths.video_vae_path,
            audio_vae_path=paths.audio_vae_path,
            duration_head_path=paths.duration_head_path,
            stage_2_ic_lora=stage_2_ic_lora,
            tiling=tiling,
        )
        depth_pipeline = (
            self._depth_processor_pipeline_class.create(depth_model_path, self.config.device)
            if depth_model_path is not None
            else None
        )
        state = ICLoraState(
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

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=state)
            self._assert_invariants()
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
        with self._lock:
            match self.state.gpu_slot:
                case GpuSlot(active_pipeline=A2VPipelineState() as state) if (
                    state.ltx_model_id == model_id
                    and state.loading_mode == load_mode
                    and state.loras == requested_loras
                    and state.gemma_root == gemma_root
                    and state.video_vae_path == paths.video_vae_path
                ):
                    return state
                case _:
                    pass

        self._evict_gpu_pipeline_for_swap()

        pipeline = self._a2v_pipeline_class.create(
            paths.checkpoint_path,
            paths.gemma_root,
            paths.upsampler_path,
            self.config.device,
            streaming_prefetch_count_for_mode(load_mode),
            loras=loras or [],
            video_vae_path=paths.video_vae_path,
            audio_vae_path=paths.audio_vae_path,
            duration_head_path=paths.duration_head_path,
        )
        state = A2VPipelineState(
            pipeline=pipeline,
            ltx_model_id=model_id,
            loading_mode=load_mode,
            loras=requested_loras,
            gemma_root=gemma_root,
            video_vae_path=paths.video_vae_path,
        )

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=state)
            self._assert_invariants()
        return state

    def _edit_load_mode(
        self,
        width: int | None,
        height: int | None,
        frames: int | None,
        *,
        queued_home: bool = False,
    ) -> LocalGenerationMode:
        """Per-job full vs stream. Missing shape keeps the process mode.

        Does not change ``local_generations_mode``. Uses the edit headroom, so a
        31 GB card streams a long 720p retake/extend as well as 1080p.
        """
        if width is None or height is None or frames is None:
            return self.config.local_generations_mode
        darwin = self.config.darwin_unified_memory
        memory_gb = memory_gb_for_job(
            vram_gb=self.config.vram_gb,
            available_ram_gb=self.config.available_ram_gb,
            darwin=darwin,
        )
        decision = decide_edit_job(
            width,
            height,
            frames,
            memory_gb=memory_gb,
            process_mode=self.config.local_generations_mode,
            darwin=darwin,
        )
        outcome = "reject" if isinstance(decision, VideoJobReject) else decision.mode
        logger.info(
            "[video-edit] job budget seq=%.0f full=%.1fGiB stream=%.1fGiB -> %s",
            decision.seq,
            decision.full_gib,
            decision.stream_gib,
            outcome,
        )
        if isinstance(decision, VideoJobReject):
            if decision.reason == "unsupported":
                raise HTTPError(
                    422,
                    LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
                    code=LOCAL_GENERATION_UNSUPPORTED,
                )
            # GenSpace still encodes the whole source, so a shorter duration
            # shrinks that job. Queued Home pads context back up to the cap.
            message = (
                EDIT_JOB_TOO_LARGE_MESSAGE
                if queued_home
                else VIDEO_JOB_TOO_LARGE_MESSAGE
            )
            raise HTTPError(422, message, code=VIDEO_JOB_TOO_LARGE)
        return decision.mode

    def edit_encode_frame_cap(self, width: int, height: int) -> int:
        """Longest Home encode window this machine will load at ``width``×``height``.

        Does not change GenSpace, which never asks for this cap. A 31 GB card
        keeps a long 720p window and shortens 1080p below 505 frames.
        """
        darwin = self.config.darwin_unified_memory
        memory_gb = memory_gb_for_job(
            vram_gb=self.config.vram_gb,
            available_ram_gb=self.config.available_ram_gb,
            darwin=darwin,
        )
        return max_edit_frames_that_load(
            width,
            height,
            frame_cap=MAX_INPUT_FRAMES,
            memory_gb=memory_gb,
            process_mode=self.config.local_generations_mode,
            darwin=darwin,
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
    ) -> RetakePipelineState:
        self._install_text_patches_if_needed()

        quantized = device_supports_fp8(self.config.device)
        gemma_root = self._text_handler.resolve_gemma_root()
        model_id = self._require_downloaded_ltx_model_id(ltx_model_id)
        paths = self._resolve_ltx_paths(model_id, gemma_root)
        load_mode = self._edit_load_mode(
            width, height, frames, queued_home=queued_home
        )

        with self._lock:
            match self.state.gpu_slot:
                case GpuSlot(
                    active_pipeline=RetakePipelineState(
                        distilled=current_distilled,
                        quantized=current_quantized,
                        gemma_root=current_gemma_root,
                        ltx_model_id=current_model_id,
                        video_vae_path=current_video_vae_path,
                        loading_mode=current_loading_mode,
                    ) as state
                ) if (
                    current_distilled == distilled
                    and current_quantized == quantized
                    and current_gemma_root == gemma_root
                    and current_model_id == model_id
                    and current_video_vae_path == paths.video_vae_path
                    and current_loading_mode == load_mode
                ):
                    return state
                case _:
                    pass

        self._evict_gpu_pipeline_for_swap()

        from ltx_core.quantization.fp8_cast import build_policy as build_fp8_cast_policy

        quantization = build_fp8_cast_policy(paths.checkpoint_path) if quantized else None
        pipeline = self._retake_pipeline_class.create(
            checkpoint_path=paths.checkpoint_path,
            gemma_root=paths.gemma_root,
            device=self.config.device,
            streaming_prefetch_count=streaming_prefetch_count_for_mode(load_mode),
            loras=[],
            quantization=quantization,
            video_vae_path=paths.video_vae_path,
            audio_vae_path=paths.audio_vae_path,
            duration_head_path=paths.duration_head_path,
        )
        state = RetakePipelineState(
            pipeline=pipeline,
            distilled=distilled,
            quantized=quantized,
            ltx_model_id=model_id,
            loading_mode=load_mode,
            gemma_root=gemma_root,
            video_vae_path=paths.video_vae_path,
        )

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=state)
            self._assert_invariants()
        return state
