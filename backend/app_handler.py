"""Application state composition root and dependency wiring."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from state.app_settings import AppSettings
from handlers import (
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
    RuntimePolicyHandler,
    SettingsHandler,
    TextHandler,
    VideoGenerationHandler,
)
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
from services.lora_catalog import LoraCatalogProvider
from services.prompt_enhancer_pipeline.gemini_prompt_enhancer_pipeline import GeminiPromptEnhancerPipeline
from state.app_state_types import AppState, TextEncoderState


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
        mlx_fast_video_pipeline_class: type[FastVideoPipeline],
        image_generation_pipeline_class: type[ImageGenerationPipeline],
        ic_lora_pipeline_class: type[IcLoraPipeline],
        depth_processor_pipeline_class: type[DepthProcessorPipeline],
        pose_processor_pipeline_class: type[PoseProcessorPipeline],
        a2v_pipeline_class: type[A2VPipeline],
        retake_pipeline_class: type[RetakePipeline],
        prompt_enhancer_pipeline_class: type[PromptEnhancerPipeline],
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
        self.mlx_fast_video_pipeline_class = mlx_fast_video_pipeline_class
        self.image_generation_pipeline_class = image_generation_pipeline_class
        self.ic_lora_pipeline_class = ic_lora_pipeline_class
        self.depth_processor_pipeline_class = depth_processor_pipeline_class
        self.pose_processor_pipeline_class = pose_processor_pipeline_class
        self.a2v_pipeline_class = a2v_pipeline_class
        self.retake_pipeline_class = retake_pipeline_class
        self.prompt_enhancer_pipeline_class = prompt_enhancer_pipeline_class

        self._lock = threading.RLock()

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
            mlx_fast_video_pipeline_class=mlx_fast_video_pipeline_class,
            image_generation_pipeline_class=image_generation_pipeline_class,
            ic_lora_pipeline_class=ic_lora_pipeline_class,
            depth_processor_pipeline_class=depth_processor_pipeline_class,
            pose_processor_pipeline_class=pose_processor_pipeline_class,
            a2v_pipeline_class=a2v_pipeline_class,
            retake_pipeline_class=retake_pipeline_class,
            config=config,
        )

        self.generation = GenerationHandler(state=self.state, lock=self._lock, config=config)

        self.video_generation = VideoGenerationHandler(
            state=self.state,
            lock=self._lock,
            generation_handler=self.generation,
            pipelines_handler=self.pipelines,
            text_handler=self.text,
            ltx_api_client=ltx_api_client,
            config=config,
        )

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

        self.ic_lora = IcLoraHandler(
            state=self.state,
            lock=self._lock,
            generation_handler=self.generation,
            pipelines_handler=self.pipelines,
            text_handler=self.text,
            video_processor=video_processor,
            lora_catalog=lora_catalog_provider,
            config=config,
        )

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

        self.downloads.cleanup_downloading_dir()

        self.load_persistent_state(default_settings)

    def load_persistent_state(self, default_settings: AppSettings) -> None:
        """Load persisted state from disk (settings, HF auth token, etc.)."""
        self.settings.load_settings(default_settings)
        self.hf_auth.load_token()


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
    mlx_fast_video_pipeline_class: type[FastVideoPipeline]
    image_generation_pipeline_class: type[ImageGenerationPipeline]
    ic_lora_pipeline_class: type[IcLoraPipeline]
    depth_processor_pipeline_class: type[DepthProcessorPipeline]
    pose_processor_pipeline_class: type[PoseProcessorPipeline]
    a2v_pipeline_class: type[A2VPipeline]
    retake_pipeline_class: type[RetakePipeline]
    prompt_enhancer_pipeline_class: type[PromptEnhancerPipeline]


def build_default_service_bundle(config: RuntimeConfig) -> ServiceBundle:
    """Build real runtime services with lazy heavy imports isolated from tests."""
    from services.fast_video_pipeline.ltx_fast_video_pipeline import LTXFastVideoPipeline
    from services.fast_video_pipeline.mlx_fast_video_pipeline import MLXFastVideoPipeline
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
        mlx_fast_video_pipeline_class=MLXFastVideoPipeline,
        image_generation_pipeline_class=ZitImageGenerationPipeline,
        ic_lora_pipeline_class=LTXIcLoraPipeline,
        depth_processor_pipeline_class=MidasDPTPipeline,
        pose_processor_pipeline_class=DWPosePipeline,
        a2v_pipeline_class=LTXa2vPipeline,
        retake_pipeline_class=LTXRetakePipeline,
        prompt_enhancer_pipeline_class=LtxPromptEnhancerPipeline,
    )


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
        mlx_fast_video_pipeline_class=bundle.mlx_fast_video_pipeline_class,
        image_generation_pipeline_class=bundle.image_generation_pipeline_class,
        ic_lora_pipeline_class=bundle.ic_lora_pipeline_class,
        depth_processor_pipeline_class=bundle.depth_processor_pipeline_class,
        pose_processor_pipeline_class=bundle.pose_processor_pipeline_class,
        a2v_pipeline_class=bundle.a2v_pipeline_class,
        retake_pipeline_class=bundle.retake_pipeline_class,
        prompt_enhancer_pipeline_class=bundle.prompt_enhancer_pipeline_class,
    )
