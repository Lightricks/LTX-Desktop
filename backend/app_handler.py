"""Application state composition root and dependency wiring."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from api_types import LTXLocalModelId
from state.app_settings import AppSettings
from handlers import (
    AssetHandler,
    DownloadHandler,
    GenerationHandler,
    HealthHandler,
    HuggingFaceAuthHandler,
    IcLoraHandler,
    ImageGenerationHandler,
    ModelsHandler,
    PipelinesHandler,
    LoraCatalogHandler,
    PromptEnhancementHandler,
    SuggestGapPromptHandler,
    RetakeHandler,
    ExtendHandler,
    FeatureFlagsHandler,
    RuntimePolicyHandler,
    SettingsHandler,
    TextHandler,
    VideoGenerationHandler,
)
from handlers.dashboard_handler import DashboardHandler
from handlers.queued_generation_handler import QueuedGenerationHandler
from runtime_config.offerings import OFFERING_IDS, resolve_offering_local_model_id
from runtime_config.runtime_config import RuntimeConfig
from services.interfaces import (
    A2VPipeline,
    DepthProcessorPipeline,
    FastVideoPipeline,
    ZitAPIClient,
    ImageGenerationPipeline,
    GpuCleaner,
    GpuInfo,
    HTTPClient,
    IcLoraPipeline,
    LTXAPIClient,
    ModelDownloader,
    PoseProcessorPipeline,
    PromptEnhancerPipeline,
    RetakePipeline,
    TaskRunner,
    TextEncoder,
    VideoProcessor,
)
from services.analytics import AnalyticsService, QueuedEnhancement
from services.lora_catalog import LoraCatalogProvider
from services.store import Store
from services.records import GenerationRecord, UnavailableError
from services.unavailable_store import UnavailableStore
from services.generation_queue.registry import ExecutorRegistry
from services.generation_queue.runner import QueueRunner
from services.features.video import (
    AUDIO_TO_VIDEO_FEATURE,
    EXTEND_FEATURE,
    FEATURE as TEXT_TO_VIDEO_FEATURE,
    IMAGE_TO_VIDEO_FEATURE,
    RETAKE_FEATURE,
    AudioToVideoExecutor,
    IcLoraRecipeExecutor,
    ExtendExecutor,
    ImageToVideoExecutor,
    RebindableAssetResolver,
    RetakeExecutor,
    TextToVideoExecutor,
)
from services.features.ic_lora_recipes import queued_ic_lora_recipes
from services.features.lora_recipes import (
    i2v_lora_recipes,
    make_recipe_lora_resolver,
    make_recipe_prompt_wrap,
    t2v_lora_recipes,
)
from services.generation_queue.types import GenerationExecutor
from services.prompt_enhancer_pipeline.gemini_prompt_enhancer_pipeline import GeminiPromptEnhancerPipeline
from state.app_state_types import AppState, TextEncoderState

logger = logging.getLogger(__name__)


class AppHandler:
    """Composition-only state service exposing typed domain handlers."""

    def __init__(
        self,
        config: RuntimeConfig,
        default_settings: AppSettings,
        http: HTTPClient,
        gpu_cleaner: GpuCleaner,
        model_downloader: ModelDownloader,
        lora_catalog_provider: LoraCatalogProvider,
        gpu_info: GpuInfo,
        video_processor: VideoProcessor,
        text_encoder: TextEncoder,
        task_runner: TaskRunner,
        ltx_api_client: LTXAPIClient,
        zit_api_client: ZitAPIClient,
        fast_video_pipeline_class: type[FastVideoPipeline],
        image_generation_pipeline_class: type[ImageGenerationPipeline],
        ic_lora_pipeline_class: type[IcLoraPipeline],
        depth_processor_pipeline_class: type[DepthProcessorPipeline],
        pose_processor_pipeline_class: type[PoseProcessorPipeline],
        a2v_pipeline_class: type[A2VPipeline],
        retake_pipeline_class: type[RetakePipeline],
        prompt_enhancer_pipeline_class: type[PromptEnhancerPipeline],
        store: Store,
        generation_executors: Mapping[str, GenerationExecutor] | None = None,
    ) -> None:
        self.config = config

        # Exposed for tests and diagnostics.
        self.http = http
        self.gpu_cleaner = gpu_cleaner
        self.model_downloader = model_downloader
        self.gpu_info = gpu_info
        self.video_processor = video_processor
        self.task_runner = task_runner
        self.ltx_api_client = ltx_api_client
        self.zit_api_client = zit_api_client
        self.fast_video_pipeline_class = fast_video_pipeline_class
        self.image_generation_pipeline_class = image_generation_pipeline_class
        self.ic_lora_pipeline_class = ic_lora_pipeline_class
        self.depth_processor_pipeline_class = depth_processor_pipeline_class
        self.pose_processor_pipeline_class = pose_processor_pipeline_class
        self.a2v_pipeline_class = a2v_pipeline_class
        self.retake_pipeline_class = retake_pipeline_class
        self.prompt_enhancer_pipeline_class = prompt_enhancer_pipeline_class

        self._lock = threading.RLock()
        self.analytics = AnalyticsService(
            http=http,
            task_runner=task_runner,
            sink_url=os.environ.get("LTX_ANALYTICS_SINK_URL", ""),
            token=os.environ.get("LTX_ANALYTICS_TOKEN", ""),
        )

        self.state = AppState(
            downloading_session=None,
            gpu_slot=None,
            active_generation=None,
            cpu_slot=None,
            text_encoder=TextEncoderState(service=text_encoder),
            app_settings=default_settings.model_copy(deep=True),
        )

        # ============================================================
        # Handlers (wired in dependency order)
        # ============================================================

        self.settings = SettingsHandler(
            state=self.state,
            lock=self._lock,
            config=config,
            http=http,
        )

        self.models = ModelsHandler(
            state=self.state,
            lock=self._lock,
            config=config,
            settings_handler=self.settings,
        )

        self.hf_auth = HuggingFaceAuthHandler(
            state=self.state,
            lock=self._lock,
            config=config,
        )

        self.downloads = DownloadHandler(
            state=self.state,
            lock=self._lock,
            models_handler=self.models,
            model_downloader=model_downloader,
            task_runner=task_runner,
            config=config,
        )

        self.catalog = LoraCatalogHandler(
            state=self.state,
            lock=self._lock,
            catalog=lora_catalog_provider,
            model_downloader=model_downloader,
            task_runner=task_runner,
            config=config,
        )

        self.text = TextHandler(
            state=self.state,
            lock=self._lock,
            config=config,
        )

        self.pipelines = PipelinesHandler(
            state=self.state,
            lock=self._lock,
            text_handler=self.text,
            gpu_cleaner=gpu_cleaner,
            fast_video_pipeline_class=fast_video_pipeline_class,
            image_generation_pipeline_class=image_generation_pipeline_class,
            ic_lora_pipeline_class=ic_lora_pipeline_class,
            depth_processor_pipeline_class=depth_processor_pipeline_class,
            pose_processor_pipeline_class=pose_processor_pipeline_class,
            a2v_pipeline_class=a2v_pipeline_class,
            retake_pipeline_class=retake_pipeline_class,
            config=config,
        )

        self.generation = GenerationHandler(state=self.state, lock=self._lock, config=config)

        # Before video generation: local text encoding has no server-side rewrite step, so the
        # generation path runs this enhancer itself.
        self.prompt_enhancement = PromptEnhancementHandler(
            state=self.state,
            lock=self._lock,
            generation_handler=self.generation,
            pipelines_handler=self.pipelines,
            text_handler=self.text,
            lora_catalog_provider=lora_catalog_provider,
            prompt_enhancer_pipeline_class=prompt_enhancer_pipeline_class,
            gemini_pipeline=GeminiPromptEnhancerPipeline(http),
            config=config,
        )

        self.video_generation = VideoGenerationHandler(
            state=self.state,
            lock=self._lock,
            generation_handler=self.generation,
            pipelines_handler=self.pipelines,
            text_handler=self.text,
            prompt_enhancement_handler=self.prompt_enhancement,
            ltx_api_client=ltx_api_client,
            config=config,
        )

        self._lora_catalog_provider = lora_catalog_provider
        self.ic_lora = IcLoraHandler(
            state=self.state,
            lock=self._lock,
            generation_handler=self.generation,
            pipelines_handler=self.pipelines,
            text_handler=self.text,
            prompt_enhancement_handler=self.prompt_enhancement,
            video_processor=video_processor,
            lora_catalog=lora_catalog_provider,
            config=config,
        )
        self._queued_asset_resolver: RebindableAssetResolver | None = None
        if generation_executors is None:
            self._queued_asset_resolver = RebindableAssetResolver(store)
            executors: dict[str, GenerationExecutor] = {
                TEXT_TO_VIDEO_FEATURE: TextToVideoExecutor(self.video_generation),
                IMAGE_TO_VIDEO_FEATURE: ImageToVideoExecutor(
                    self.video_generation, self._queued_asset_resolver
                ),
            }
            # Each t2v LoRA recipe reuses the text-to-video executor, keyed on its
            # Explore id, with its scaffold applied post-enhance. i2v recipes reuse
            # the image-to-video executor the same way (start frame + LoRA).
            for recipe in t2v_lora_recipes():
                executors[recipe.recipe_id] = TextToVideoExecutor(
                    self.video_generation,
                    prompt_wrap=make_recipe_prompt_wrap(recipe),
                    lora_resolver=make_recipe_lora_resolver(
                        recipe,
                        catalog=self._lora_catalog_provider,
                        models_dir=lambda: self.video_generation.models_dir,
                    ),
                )
            for recipe in i2v_lora_recipes():
                executors[recipe.recipe_id] = ImageToVideoExecutor(
                    self.video_generation,
                    self._queued_asset_resolver,
                    prompt_wrap=make_recipe_prompt_wrap(recipe),
                    lora_resolver=make_recipe_lora_resolver(
                        recipe,
                        catalog=self._lora_catalog_provider,
                        models_dir=lambda: self.video_generation.models_dir,
                    ),
                )
            executors[AUDIO_TO_VIDEO_FEATURE] = AudioToVideoExecutor(
                self.video_generation, self._queued_asset_resolver
            )
            executors[RETAKE_FEATURE] = RetakeExecutor(
                self.pipelines,
                self._queued_asset_resolver,
                self.text,
                lambda: self.video_generation.models_dir,
                config.default_negative_prompt,
                self.generation,
            )
            executors[EXTEND_FEATURE] = ExtendExecutor(
                self.pipelines,
                self._queued_asset_resolver,
                self.text,
                lambda: self.video_generation.models_dir,
                config.default_negative_prompt,
                self.generation,
            )
            for recipe in queued_ic_lora_recipes():
                executors[recipe.recipe_id] = IcLoraRecipeExecutor(
                    self.ic_lora,
                    self._queued_asset_resolver,
                    lambda: self.video_generation.models_dir,
                    catalog_id=recipe.catalog_id,
                    cutout=recipe.cutout,
                )
            self.executor_registry = ExecutorRegistry(executors)
        else:
            self.executor_registry = ExecutorRegistry(generation_executors)
        self._bind_store(store)

        self.image_generation = ImageGenerationHandler(
            state=self.state,
            lock=self._lock,
            generation_handler=self.generation,
            pipelines_handler=self.pipelines,
            config=config,
            zit_api_client=zit_api_client,
        )

        self.health = HealthHandler(
            state=self.state,
            lock=self._lock,
            models_handler=self.models,
            gpu_info=gpu_info,
            config=config,
        )

        self.runtime_policy = RuntimePolicyHandler(config=config)
        self.feature_flags = FeatureFlagsHandler(lock=self._lock, config=config)

        self.suggest_gap_prompt = SuggestGapPromptHandler(
            state=self.state,
            lock=self._lock,
            config=config,
            http=http,
        )

        self.retake = RetakeHandler(
            state=self.state,
            lock=self._lock,
            ltx_api_client=ltx_api_client,
            config=config,
            generation_handler=self.generation,
            pipelines_handler=self.pipelines,
            text_handler=self.text,
        )

        self.extend = ExtendHandler(
            state=self.state,
            lock=self._lock,
            ltx_api_client=ltx_api_client,
            config=config,
            generation_handler=self.generation,
            pipelines_handler=self.pipelines,
            text_handler=self.text,
        )

        self.downloads.cleanup_downloading_dir()

        self.load_persistent_state(default_settings)

    def load_persistent_state(self, default_settings: AppSettings) -> None:
        """Load persisted state from disk (settings, HF auth token, etc.)."""
        self.settings.load_settings(default_settings)
        self.hf_auth.load_token()
        try:
            self.queued_generations.recover_on_boot()
        except UnavailableError:
            logger.warning(
                "Store boot recovery failed; store unavailable at %s",
                self.config.app_data_dir / "store.sqlite3",
                exc_info=True,
            )
            self._bind_store(UnavailableStore())

    def _queued_local_model_id(self, generation: GenerationRecord) -> LTXLocalModelId | None:
        """The checkpoint this queued job will encode with, not the active model."""
        params = generation.spec.get("params")
        if not isinstance(params, dict):
            return None
        model = params.get("model")
        if not isinstance(model, str) or model not in OFFERING_IDS:
            return None
        return resolve_offering_local_model_id(self.video_generation.models_dir, model)

    def _queued_enhancement_policy(self, generation: GenerationRecord) -> QueuedEnhancement:
        settings = self.state.app_settings
        model_id = self._queued_local_model_id(generation)
        return QueuedEnhancement(
            explore_auto_enhance_prompts=settings.explore_auto_enhance_prompts,
            prompt_enhancer_enabled=settings.prompt_enhancer_enabled,
            use_local_encoding=self.text.should_use_local_encoding(model_id),
            local_enhancer_available=(
                self.text.resolve_prompt_enhancer_root_if_downloaded(model_id) is not None
            ),
        )

    def _bind_store(self, store: Store) -> None:
        if self._queued_asset_resolver is not None:
            self._queued_asset_resolver.bind(store)
        self.generation_queue = QueueRunner(
            db=store,
            generation=self.generation,
            executors=self.executor_registry,
            analytics=self.analytics,
            enhancement_policy=self._queued_enhancement_policy,
        )
        self.assets = AssetHandler(
            store=store,
        )
        self.queued_generations = QueuedGenerationHandler(
            store=store,
            executor_registry=self.executor_registry,
            derive_local_a2v=self.video_generation.derive_local_a2v_params,
            queue_control=self.generation_queue,
            progress_reader=self.generation.get_generation_progress,
            config=self.config,
            lora_catalog_provider=self._lora_catalog_provider,
            state=self.state,
            lock=self._lock,
        )
        self.generation_queue.set_finished_listener(
            self.queued_generations.remember_finished
        )
        self.dashboard = DashboardHandler(store, self._lora_catalog_provider)


@dataclass
class ServiceBundle:
    http: HTTPClient
    gpu_cleaner: GpuCleaner
    model_downloader: ModelDownloader
    lora_catalog_provider: LoraCatalogProvider
    gpu_info: GpuInfo
    video_processor: VideoProcessor
    text_encoder: TextEncoder
    task_runner: TaskRunner
    ltx_api_client: LTXAPIClient
    zit_api_client: ZitAPIClient
    fast_video_pipeline_class: type[FastVideoPipeline]
    image_generation_pipeline_class: type[ImageGenerationPipeline]
    ic_lora_pipeline_class: type[IcLoraPipeline]
    depth_processor_pipeline_class: type[DepthProcessorPipeline]
    pose_processor_pipeline_class: type[PoseProcessorPipeline]
    a2v_pipeline_class: type[A2VPipeline]
    retake_pipeline_class: type[RetakePipeline]
    prompt_enhancer_pipeline_class: type[PromptEnhancerPipeline]
    store: Store
    generation_executors: Mapping[str, GenerationExecutor] | None = None


def build_default_service_bundle(config: RuntimeConfig) -> ServiceBundle:
    """Build real runtime services with lazy heavy imports isolated from tests."""
    from services.fast_video_pipeline.ltx_fast_video_pipeline import LTXFastVideoPipeline
    from services.zit_api_client.zit_api_client_impl import ZitAPIClientImpl
    from services.gpu_cleaner.torch_cleaner import TorchCleaner
    from services.gpu_info.gpu_info_impl import GpuInfoImpl
    from services.http_client.http_client_impl import HTTPClientImpl
    from services.a2v_pipeline.ltx_a2v_pipeline import LTXa2vPipeline
    from services.depth_processor_pipeline.midas_dpt_pipeline import MidasDPTPipeline
    from services.ic_lora_pipeline.ltx_ic_lora_pipeline import LTXIcLoraPipeline
    from services.image_generation_pipeline.zit_image_generation_pipeline import ZitImageGenerationPipeline
    from services.ltx_api_client.ltx_api_client_impl import LTXAPIClientImpl
    from services.model_downloader.hugging_face_downloader import HuggingFaceDownloader
    from services.retake_pipeline.ltx_retake_pipeline import LTXRetakePipeline
    from services.prompt_enhancer_pipeline.ltx_prompt_enhancer_pipeline import LtxPromptEnhancerPipeline
    from services.pose_processor_pipeline.dw_pose_pipeline import DWPosePipeline
    from services.task_runner.threading_runner import ThreadingRunner
    from services.text_encoder.ltx_text_encoder import LTXTextEncoder
    from services.video_processor.video_processor_impl import VideoProcessorImpl
    from services.lora_catalog import FileLoraCatalogProvider

    http = HTTPClientImpl()

    return ServiceBundle(
        http=http,
        gpu_cleaner=TorchCleaner(device=config.device),
        model_downloader=HuggingFaceDownloader(),
        lora_catalog_provider=FileLoraCatalogProvider(
            config.lora_catalog_source, config.lora_catalog_fallback_path or None
        ),
        gpu_info=GpuInfoImpl(),
        video_processor=VideoProcessorImpl(),
        text_encoder=LTXTextEncoder(
            device=config.device,
            http=http,
            ltx_api_base_url=config.ltx_api_base_url,
        ),
        task_runner=ThreadingRunner(),
        ltx_api_client=LTXAPIClientImpl(http=http, ltx_api_base_url=config.ltx_api_base_url),
        zit_api_client=ZitAPIClientImpl(http=http),
        fast_video_pipeline_class=LTXFastVideoPipeline,
        image_generation_pipeline_class=ZitImageGenerationPipeline,
        ic_lora_pipeline_class=LTXIcLoraPipeline,
        depth_processor_pipeline_class=MidasDPTPipeline,
        pose_processor_pipeline_class=DWPosePipeline,
        a2v_pipeline_class=LTXa2vPipeline,
        retake_pipeline_class=LTXRetakePipeline,
        prompt_enhancer_pipeline_class=LtxPromptEnhancerPipeline,
        store=open_store(config.app_data_dir),
    )


def open_store(app_data_dir: Path) -> Store:
    from services.sqlite_store import SqliteStore

    try:
        return SqliteStore(app_data_dir)
    except UnavailableError:
        logger.warning(
            "Store unavailable at %s", app_data_dir / "store.sqlite3", exc_info=True
        )
        return UnavailableStore()


def build_initial_state(
    config: RuntimeConfig,
    default_settings: AppSettings,
    service_bundle: ServiceBundle | None = None,
) -> AppHandler:
    bundle = service_bundle or build_default_service_bundle(config)

    return AppHandler(
        config=config,
        default_settings=default_settings,
        http=bundle.http,
        gpu_cleaner=bundle.gpu_cleaner,
        model_downloader=bundle.model_downloader,
        lora_catalog_provider=bundle.lora_catalog_provider,
        gpu_info=bundle.gpu_info,
        video_processor=bundle.video_processor,
        text_encoder=bundle.text_encoder,
        task_runner=bundle.task_runner,
        ltx_api_client=bundle.ltx_api_client,
        zit_api_client=bundle.zit_api_client,
        fast_video_pipeline_class=bundle.fast_video_pipeline_class,
        image_generation_pipeline_class=bundle.image_generation_pipeline_class,
        ic_lora_pipeline_class=bundle.ic_lora_pipeline_class,
        depth_processor_pipeline_class=bundle.depth_processor_pipeline_class,
        pose_processor_pipeline_class=bundle.pose_processor_pipeline_class,
        a2v_pipeline_class=bundle.a2v_pipeline_class,
        retake_pipeline_class=bundle.retake_pipeline_class,
        prompt_enhancer_pipeline_class=bundle.prompt_enhancer_pipeline_class,
        store=bundle.store,
        generation_executors=bundle.generation_executors,
    )
