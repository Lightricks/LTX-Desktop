"""Pipeline lifecycle and warmup handler."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import RLock
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from handlers.base import StateHandlerBase
from handlers.text_handler import TextHandler
from services.interfaces import (
    A2VPipeline,
    FastVideoPipeline,
    ImageGenerationPipeline,
    GpuCleaner,
    IcLoraPipeline,
    RetakePipeline,
    VideoPipelineModelType,
)
from services.gpu_optimizations.ffn_chunking import patch_ffn_chunking
from services.gpu_optimizations.tea_cache import install_tea_cache_patch
from services.services_utils import device_supports_fp8, get_device_type
from state.app_state_types import (
    A2VPipelineState,
    AppState,
    CpuSlot,
    GenerationRunning,
    GpuSlot,
    ICLoraState,
    RetakePipelineState,
    VideoPipelineState,
    VideoPipelineWarmth,
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
        gguf_video_pipeline_class: type[FastVideoPipeline] | None,
        nf4_video_pipeline_class: type[FastVideoPipeline] | None,
        image_generation_pipeline_class: type[ImageGenerationPipeline],
        flux_klein_pipeline_class: type[ImageGenerationPipeline] | None,
        ic_lora_pipeline_class: type[IcLoraPipeline],
        a2v_pipeline_class: type[A2VPipeline],
        retake_pipeline_class: type[RetakePipeline],
        config: RuntimeConfig,
        outputs_dir: Path,
        device: torch.device,
    ) -> None:
        super().__init__(state, lock)
        self._text_handler = text_handler
        self._gpu_cleaner = gpu_cleaner
        self._fast_video_pipeline_class = fast_video_pipeline_class
        self._gguf_video_pipeline_class = gguf_video_pipeline_class
        self._nf4_video_pipeline_class = nf4_video_pipeline_class
        self._image_generation_pipeline_class = image_generation_pipeline_class
        self._flux_klein_pipeline_class = flux_klein_pipeline_class
        self._ic_lora_pipeline_class = ic_lora_pipeline_class
        self._a2v_pipeline_class = a2v_pipeline_class
        self._retake_pipeline_class = retake_pipeline_class
        self._config = config
        self._outputs_dir = outputs_dir
        self._device = device
        self._runtime_device = get_device_type(device)
        # Track which image model is currently loaded (zit or flux_klein)
        self._loaded_image_model: str | None = None

    def _ensure_no_running_generation(self) -> None:
        match self.state.gpu_slot:
            case GpuSlot(generation=GenerationRunning()):
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
        gpu_is_zit = False
        match self.state.gpu_slot:
            case GpuSlot(active_pipeline=VideoPipelineState() | ICLoraState() | A2VPipelineState() | RetakePipelineState()):
                gpu_is_zit = False
            case GpuSlot():
                gpu_is_zit = True
            case _:
                gpu_is_zit = False

        if gpu_is_zit and self.state.cpu_slot is not None:
            raise RuntimeError("Invariant violation: ZIT cannot be in both GPU and CPU slots")

    def _install_text_patches_if_needed(self) -> None:
        te = self.state.text_encoder
        if te is None:
            return
        te.service.install_patches(lambda: self.state)

    def _compile_if_enabled(self, state: VideoPipelineState) -> VideoPipelineState:
        if not self.state.app_settings.use_torch_compile:
            return state
        if state.is_compiled:
            return state
        if self._runtime_device == "mps":
            logger.info("Skipping torch.compile() for %s - not supported on MPS", state.pipeline.pipeline_kind)
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
        lora_path: str | None = None,
        lora_weight: float = 1.0,
    ) -> VideoPipelineState:
        gemma_root = self._text_handler.resolve_gemma_root()

        # Determine checkpoint path and pipeline class based on selected model
        selected = self.state.app_settings.selected_video_model
        custom_dir = self.state.app_settings.custom_video_model_path
        pipeline_class = self._fast_video_pipeline_class

        if selected:
            base_dir = Path(custom_dir) if custom_dir else self._config.models_dir
            model_path = base_dir / selected
            checkpoint_path = str(model_path)

            # Validate model file/folder still exists on disk
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Selected model not found: {checkpoint_path}. "
                    "Go to Settings → Models to select a different model."
                )

            if selected.endswith(".gguf") and self._gguf_video_pipeline_class is not None:
                pipeline_class = self._gguf_video_pipeline_class
            elif model_path.is_dir() and self._nf4_video_pipeline_class is not None:
                # NF4 models are folders
                pipeline_class = self._nf4_video_pipeline_class
            # else: safetensors files use default pipeline
        else:
            checkpoint_path = str(self._config.model_path("checkpoint"))

        upsampler_path = str(self._config.model_path("upsampler"))

        pipeline = pipeline_class.create(
            checkpoint_path,
            gemma_root,
            upsampler_path,
            self._device,
            lora_path=lora_path,
            lora_weight=lora_weight,
        )

        state = VideoPipelineState(
            pipeline=pipeline,
            warmth=VideoPipelineWarmth.COLD,
            is_compiled=False,
            lora_path=lora_path,
        )
        state = self._compile_if_enabled(state)

        # Apply FFN chunking if enabled and torch.compile is not active
        chunk_count = self.state.app_settings.ffn_chunk_count
        if chunk_count > 0 and not state.is_compiled:
            try:
                transformer: torch.nn.Module = state.pipeline.pipeline.model_ledger.transformer()  # type: ignore[union-attr]
                patch_ffn_chunking(transformer, num_chunks=chunk_count)  # pyright: ignore[reportUnknownArgumentType]
            except AttributeError:
                logger.debug("FFN chunking skipped — pipeline has no model_ledger")

        # Install TeaCache denoising loop patch
        tea_threshold = self.state.app_settings.tea_cache_threshold
        try:
            install_tea_cache_patch(tea_threshold)
        except (ImportError, AttributeError):
            logger.debug("TeaCache skipped — ltx_pipelines not available")

        return state

    def unload_gpu_pipeline(self) -> None:
        with self._lock:
            self._ensure_no_running_generation()
            self.state.gpu_slot = None
            self._assert_invariants()
        self._gpu_cleaner.cleanup()

    def park_zit_on_cpu(self) -> None:
        zit: ImageGenerationPipeline | None = None

        with self._lock:
            if self.state.gpu_slot is None:
                return

            active = self.state.gpu_slot.active_pipeline
            if isinstance(active, (VideoPipelineState, ICLoraState, A2VPipelineState, RetakePipelineState)):
                return

            generation = self.state.gpu_slot.generation
            if isinstance(generation, GenerationRunning):
                raise RuntimeError("Cannot park ZIT while generation is running")

            zit = active
            self.state.gpu_slot = None

        assert zit is not None
        zit.to("cpu")
        self._gpu_cleaner.cleanup()

        with self._lock:
            self.state.cpu_slot = CpuSlot(active_pipeline=zit)
            self._assert_invariants()

    def load_zit_to_gpu(
        self,
        on_phase: "Callable[[str], None] | None" = None,
    ) -> ImageGenerationPipeline:
        def _report(phase: str) -> None:
            if on_phase is not None:
                on_phase(phase)

        with self._lock:
            if self.state.gpu_slot is not None:
                active = self.state.gpu_slot.active_pipeline
                if not isinstance(active, (VideoPipelineState, ICLoraState, A2VPipelineState, RetakePipelineState)):
                    return active
                self._ensure_no_running_generation()
                # Unload the video/other pipeline from GPU before loading ZIT
                _report("unloading_video_model")
                self.state.gpu_slot = None
                self._assert_invariants()
        _report("cleaning_gpu")
        self._gpu_cleaner.cleanup()

        zit_service: ImageGenerationPipeline | None = None

        with self._lock:
            match self.state.cpu_slot:
                case CpuSlot(active_pipeline=stored):
                    zit_service = stored
                    self.state.cpu_slot = None
                case _:
                    zit_service = None

        _report("loading_image_model")
        if zit_service is None:
            zit_path = self._config.model_path("zit")
            if not (zit_path.exists() and any(zit_path.iterdir())):
                raise RuntimeError("Z-Image-Turbo model not downloaded. Please download the AI models first using the Model Status menu.")
            zit_service = self._image_generation_pipeline_class.create(str(zit_path), self._runtime_device)
        else:
            zit_service.to(self._runtime_device)

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=zit_service, generation=None)
            self._loaded_image_model = "zit"
            self._assert_invariants()

        return zit_service

    def load_image_model_to_gpu(
        self,
        model_name: str = "flux-klein-9b",
        on_phase: "Callable[[str], None] | None" = None,
    ) -> ImageGenerationPipeline:
        """Load the requested image model to GPU.

        Supports 'zit' (Z-Image-Turbo) and 'flux_klein' (FLUX.2 Klein 9B).
        If the requested model is already loaded, returns it immediately.
        """
        if model_name == "flux-klein-9b" or model_name == "flux_klein":
            return self._load_flux_klein_to_gpu(on_phase=on_phase)
        # Default: load ZIT
        return self.load_zit_to_gpu(on_phase=on_phase)

    def _load_flux_klein_to_gpu(
        self,
        on_phase: "Callable[[str], None] | None" = None,
    ) -> ImageGenerationPipeline:
        """Load FLUX.2 Klein 9B to GPU."""
        def _report(phase: str) -> None:
            if on_phase is not None:
                on_phase(phase)

        # If FLUX Klein is already on GPU, return it (unless pipeline was
        # destroyed after previous generation to avoid Windows VAE segfault)
        with self._lock:
            if self.state.gpu_slot is not None and self._loaded_image_model == "flux_klein":
                active = self.state.gpu_slot.active_pipeline
                if not isinstance(active, (VideoPipelineState, ICLoraState, A2VPipelineState, RetakePipelineState)):
                    if hasattr(active, "pipeline"):
                        return active
                    # Pipeline was destroyed after last gen; fall through to reload

            # Evict whatever is on GPU
            if self.state.gpu_slot is not None:
                self._ensure_no_running_generation()
                _report("unloading_image_model")
                self.state.gpu_slot = None
                self.state.cpu_slot = None  # Don't cache ZIT when switching to FLUX
                self._assert_invariants()

        _report("cleaning_gpu")
        self._gpu_cleaner.cleanup()

        _report("loading_image_model")
        if self._flux_klein_pipeline_class is None:
            raise RuntimeError("FLUX.2 Klein pipeline class not configured")
        flux_path = self._config.model_path("flux_klein")
        if not (flux_path.exists() and any(flux_path.iterdir())):
            raise RuntimeError("FLUX.2 Klein 9B model not downloaded. Please download it from the Model Status menu.")
        flux_service = self._flux_klein_pipeline_class.create(str(flux_path), self._runtime_device)

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=flux_service, generation=None)
            self._loaded_image_model = "flux_klein"
            self._assert_invariants()

        return flux_service

    def preload_zit_to_cpu(self) -> ImageGenerationPipeline:
        with self._lock:
            match self.state.cpu_slot:
                case CpuSlot(active_pipeline=existing):
                    return existing
                case _:
                    pass

        zit_path = self._config.model_path("zit")
        if not (zit_path.exists() and any(zit_path.iterdir())):
            raise RuntimeError("Z-Image-Turbo model not downloaded. Please download the AI models first using the Model Status menu.")

        zit_service = self._image_generation_pipeline_class.create(str(zit_path), None)
        with self._lock:
            if self.state.cpu_slot is None:
                self.state.cpu_slot = CpuSlot(active_pipeline=zit_service)
                self._assert_invariants()
                return zit_service
            return self.state.cpu_slot.active_pipeline

    def _evict_gpu_pipeline_for_swap(self) -> None:
        should_park_zit = False
        should_cleanup = False

        with self._lock:
            self._ensure_no_running_generation()
            if self.state.gpu_slot is None:
                return

            active = self.state.gpu_slot.active_pipeline
            if isinstance(active, (VideoPipelineState, ICLoraState, A2VPipelineState, RetakePipelineState)):
                self.state.gpu_slot = None
                self._assert_invariants()
                should_cleanup = True
            else:
                should_park_zit = True

        if should_park_zit:
            self.park_zit_on_cpu()
        elif should_cleanup:
            self._gpu_cleaner.cleanup()

    def load_gpu_pipeline(
        self,
        model_type: VideoPipelineModelType,
        should_warm: bool = False,
        on_phase: Callable[[str], None] | None = None,
        lora_path: str | None = None,
        lora_weight: float = 1.0,
    ) -> VideoPipelineState:
        self._install_text_patches_if_needed()

        def _report(phase: str) -> None:
            if on_phase is not None:
                on_phase(phase)

        state: VideoPipelineState | None = None
        with self._lock:
            if self._pipeline_matches_model_type(model_type):
                match self.state.gpu_slot:
                    case GpuSlot(active_pipeline=VideoPipelineState() as existing_state):
                        # Reload if the LoRA changed
                        if existing_state.lora_path == lora_path:
                            state = existing_state
                    case _:
                        pass

        if state is None:
            _report("unloading_image_model")
            self._evict_gpu_pipeline_for_swap()
            if lora_path:
                _report("loading_lora")
            _report("loading_video_model")
            state = self._create_video_pipeline(model_type, lora_path=lora_path, lora_weight=lora_weight)
            with self._lock:
                self.state.gpu_slot = GpuSlot(active_pipeline=state, generation=None)
                self._assert_invariants()

        if should_warm and state.warmth == VideoPipelineWarmth.COLD:
            with self._lock:
                state.warmth = VideoPipelineWarmth.WARMING

            self.warmup_pipeline(model_type)
            with self._lock:
                if state.warmth == VideoPipelineWarmth.WARMING:
                    state.warmth = VideoPipelineWarmth.WARM

        return state

    def load_ic_lora(self, lora_path: str) -> ICLoraState:
        self._install_text_patches_if_needed()

        with self._lock:
            match self.state.gpu_slot:
                case GpuSlot(active_pipeline=ICLoraState(lora_path=current_path) as state) if current_path == lora_path:
                    return state
                case _:
                    pass

        self._evict_gpu_pipeline_for_swap()

        pipeline = self._ic_lora_pipeline_class.create(
            str(self._config.model_path("checkpoint")),
            self._text_handler.resolve_gemma_root(),
            str(self._config.model_path("upsampler")),
            lora_path,
            self._device,
        )
        state = ICLoraState(pipeline=pipeline, lora_path=lora_path)

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=state, generation=None)
            self._assert_invariants()
        return state

    def load_a2v_pipeline(self) -> A2VPipelineState:
        self._install_text_patches_if_needed()

        with self._lock:
            match self.state.gpu_slot:
                case GpuSlot(active_pipeline=A2VPipelineState() as state):
                    return state
                case _:
                    pass

        self._evict_gpu_pipeline_for_swap()

        pipeline = self._a2v_pipeline_class.create(
            str(self._config.model_path("checkpoint")),
            self._text_handler.resolve_gemma_root(),
            str(self._config.model_path("upsampler")),
            self._device,
        )
        state = A2VPipelineState(pipeline=pipeline)

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=state, generation=None)
            self._assert_invariants()
        return state

    def load_retake_pipeline(self, *, distilled: bool = True) -> RetakePipelineState:
        self._install_text_patches_if_needed()

        quantized = device_supports_fp8(self._device)

        with self._lock:
            match self.state.gpu_slot:
                case GpuSlot(
                    active_pipeline=RetakePipelineState(distilled=current_distilled, quantized=current_quantized) as state
                ) if current_distilled == distilled and current_quantized == quantized:
                    return state
                case _:
                    pass

        self._evict_gpu_pipeline_for_swap()

        from ltx_core.quantization import QuantizationPolicy

        quantization = QuantizationPolicy.fp8_cast() if quantized else None
        pipeline = self._retake_pipeline_class.create(
            checkpoint_path=str(self._config.model_path("checkpoint")),
            gemma_root=self._text_handler.resolve_gemma_root(),
            device=self._device,
            loras=[],
            quantization=quantization,
        )
        state = RetakePipelineState(pipeline=pipeline, distilled=distilled, quantized=quantized)

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=state, generation=None)
            self._assert_invariants()
        return state

    def warmup_pipeline(self, model_type: VideoPipelineModelType) -> None:
        state = self.load_gpu_pipeline(model_type, should_warm=False)
        warmup_path = self._outputs_dir / f"_warmup_{model_type}.mp4"
        state.pipeline.warmup(output_path=str(warmup_path))
