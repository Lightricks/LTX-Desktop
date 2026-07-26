"""Pipeline lifecycle handler."""

from __future__ import annotations

import logging
from threading import RLock
from typing import TYPE_CHECKING, Literal

from _routes._errors import HTTPError
from api_types import LTXLocalModelId
from handlers.base import StateHandlerBase
from handlers.text_handler import TextHandler
from runtime_config.model_download_specs import (
    IMG_GEN_MODEL_CP_ID,
    get_existing_cp_path,
    get_ltx_model_spec,
    resolve_active_ltx_model_id,
)
from runtime_config.runtime_policy import (
    decide_fast_video_execution_mode,
    streaming_prefetch_count_for_mode,
)
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
        mlx_fast_video_pipeline_class: type[FastVideoPipeline],
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
        self._mlx_fast_video_pipeline_class = mlx_fast_video_pipeline_class
        self._image_generation_pipeline_class = image_generation_pipeline_class
        self._ic_lora_pipeline_class = ic_lora_pipeline_class
        self._depth_processor_pipeline_class = depth_processor_pipeline_class
        self._pose_processor_pipeline_class = pose_processor_pipeline_class
        self._a2v_pipeline_class = a2v_pipeline_class
        self._retake_pipeline_class = retake_pipeline_class
        self._runtime_device = get_device_type(self.config.device)

    def _ensure_no_running_generation(self) -> None:
        match self.state.active_generation:
            case GpuGeneration(state=GenerationRunning()) if self.state.gpu_slot is not None:
                raise RuntimeError("Generation already running; cannot swap pipelines")
            case _:
                return

    def _pipeline_matches_model_type(
        self,
        model_type: VideoPipelineModelType,
        runtime_engine: Literal["torch", "mlx"],
    ) -> bool:
        match self.state.gpu_slot:
            case GpuSlot(active_pipeline=VideoPipelineState(pipeline=pipeline, runtime_engine=active_engine)):
                return pipeline.pipeline_kind == model_type and active_engine == runtime_engine
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

    def _require_downloaded_ltx_model_id(self) -> LTXLocalModelId:
        model_id = resolve_active_ltx_model_id(
            self.models_dir, self.state.app_settings.active_ltx_model_id
        )
        if model_id is None:
            raise HTTPError(409, "NO_DOWNLOADED_LTX_MODEL")
        return model_id

    def _compile_if_enabled(self, state: VideoPipelineState) -> VideoPipelineState:
        if state.runtime_engine == "mlx":
            return state
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
        runtime_engine: Literal["torch", "mlx"] = "torch",
    ) -> VideoPipelineState:
        gemma_root = self._text_handler.resolve_gemma_root()
        model_id = self._require_downloaded_ltx_model_id()
        spec = get_ltx_model_spec(model_id)
        checkpoint_path = str(get_existing_cp_path(self.models_dir, spec.model_cp))
        upsampler_path = str(get_existing_cp_path(self.models_dir, spec.upscale_cp))

        pipeline_class = (
            self._mlx_fast_video_pipeline_class
            if runtime_engine == "mlx"
            else self._fast_video_pipeline_class
        )
        if runtime_engine == "mlx":
            execution_mode = decide_fast_video_execution_mode(
                "mlx",
                self.config.local_generations_mode,
                self.config.available_ram_gb,
            )
            streaming_prefetch_count = 2 if execution_mode == "low_ram" else None
        else:
            streaming_prefetch_count = streaming_prefetch_count_for_mode(
                self.config.local_generations_mode
            )

        pipeline = pipeline_class.create(
            checkpoint_path,
            gemma_root,
            upsampler_path,
            self.config.device,
            streaming_prefetch_count,
            loras=loras or [],
        )

        state = VideoPipelineState(
            pipeline=pipeline,
            is_compiled=False,
            runtime_engine=runtime_engine,
            loras=tuple(loras) if loras else (),
            gemma_root=gemma_root,
        )
        return self._compile_if_enabled(state)

    def unload_gpu_pipeline(self) -> None:
        with self._lock:
            self._ensure_no_running_generation()
            self.state.gpu_slot = None
            self._assert_invariants()
        self._gpu_cleaner.cleanup()

    def cleanup_runtime_caches(self) -> None:
        """Release allocator/cache memory while preserving a reusable pipeline."""
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
        runtime_engine: Literal["torch", "mlx"] = "torch",
    ) -> VideoPipelineState:
        self._install_text_patches_if_needed()

        requested_loras = tuple(loras) if loras else ()
        requested_gemma_root = self._text_handler.resolve_gemma_root()
        state: VideoPipelineState | None = None
        with self._lock:
            if self._pipeline_matches_model_type(model_type, runtime_engine):
                match self.state.gpu_slot:
                    case GpuSlot(
                        active_pipeline=VideoPipelineState() as existing_state
                    ) if (
                        existing_state.loras == requested_loras
                        and existing_state.gemma_root == requested_gemma_root
                    ):
                        state = existing_state
                    case _:
                        pass

        if state is None:
            self._evict_gpu_pipeline_for_swap()
            state = self._create_video_pipeline(
                model_type,
                loras=loras,
                runtime_engine=runtime_engine,
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
    ) -> ICLoraState:
        self._install_text_patches_if_needed()

        gemma_root = self._text_handler.resolve_gemma_root()
        with self._lock:
            match self.state.gpu_slot:
                case GpuSlot(
                    active_pipeline=ICLoraState(
                        lora_path=current_lora_path,
                        depth_model_path=current_depth_model_path,
                        lora_strength=current_lora_strength,
                        gemma_root=current_gemma_root,
                    ) as state
                ) if (
                    current_lora_path == lora_path
                    and current_depth_model_path == depth_model_path
                    and current_lora_strength == lora_strength
                    and current_gemma_root == gemma_root
                ):
                    return state
                case _:
                    pass

        self._evict_gpu_pipeline_for_swap()
        model_id = self._require_downloaded_ltx_model_id()
        model_spec = get_ltx_model_spec(model_id)

        pipeline = self._ic_lora_pipeline_class.create(
            str(get_existing_cp_path(self.models_dir, model_spec.model_cp)),
            gemma_root,
            str(get_existing_cp_path(self.models_dir, model_spec.upscale_cp)),
            lora_path,
            self.config.device,
            streaming_prefetch_count_for_mode(self.config.local_generations_mode),
            lora_strength,
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
            lora_strength=lora_strength,
            gemma_root=gemma_root,
        )

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=state)
            self._assert_invariants()
        return state

    def load_a2v_pipeline(self, loras: list[tuple[str, float]] | None = None) -> A2VPipelineState:
        self._install_text_patches_if_needed()

        requested_loras = tuple(loras) if loras else ()
        gemma_root = self._text_handler.resolve_gemma_root()
        with self._lock:
            match self.state.gpu_slot:
                case GpuSlot(active_pipeline=A2VPipelineState() as state) if (
                    state.loras == requested_loras and state.gemma_root == gemma_root
                ):
                    return state
                case _:
                    pass

        self._evict_gpu_pipeline_for_swap()
        model_id = self._require_downloaded_ltx_model_id()
        model_spec = get_ltx_model_spec(model_id)

        pipeline = self._a2v_pipeline_class.create(
            str(get_existing_cp_path(self.models_dir, model_spec.model_cp)),
            gemma_root,
            str(get_existing_cp_path(self.models_dir, model_spec.upscale_cp)),
            self.config.device,
            streaming_prefetch_count_for_mode(self.config.local_generations_mode),
            loras=loras or [],
        )
        state = A2VPipelineState(pipeline=pipeline, loras=requested_loras, gemma_root=gemma_root)

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=state)
            self._assert_invariants()
        return state

    def load_retake_pipeline(self, *, distilled: bool = True) -> RetakePipelineState:
        self._install_text_patches_if_needed()

        quantized = device_supports_fp8(self.config.device)
        gemma_root = self._text_handler.resolve_gemma_root()

        with self._lock:
            match self.state.gpu_slot:
                case GpuSlot(
                    active_pipeline=RetakePipelineState(
                        distilled=current_distilled, quantized=current_quantized, gemma_root=current_gemma_root
                    ) as state
                ) if (
                    current_distilled == distilled
                    and current_quantized == quantized
                    and current_gemma_root == gemma_root
                ):
                    return state
                case _:
                    pass

        self._evict_gpu_pipeline_for_swap()

        from ltx_core.quantization.fp8_cast import build_policy as build_fp8_cast_policy

        model_id = self._require_downloaded_ltx_model_id()
        model_spec = get_ltx_model_spec(model_id)
        checkpoint_path = str(get_existing_cp_path(self.models_dir, model_spec.model_cp))
        quantization = build_fp8_cast_policy(checkpoint_path) if quantized else None
        pipeline = self._retake_pipeline_class.create(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            device=self.config.device,
            streaming_prefetch_count=streaming_prefetch_count_for_mode(self.config.local_generations_mode),
            loras=[],
            quantization=quantization,
        )
        state = RetakePipelineState(
            pipeline=pipeline, distilled=distilled, quantized=quantized, gemma_root=gemma_root
        )

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=state)
            self._assert_invariants()
        return state
