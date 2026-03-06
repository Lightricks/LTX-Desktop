"""Application state composition root and dependency wiring."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from state.app_settings import AppSettings
from handlers import (
    DownloadHandler,
    GenerationHandler,
    HealthHandler,
    IcLoraHandler,
    ImageGenerationHandler,
    ModelsHandler,
    PipelinesHandler,
    SuggestGapPromptHandler,
    RetakeHandler,
    RuntimePolicyHandler,
    SettingsHandler,
    TextHandler,
    VideoGenerationHandler,
)
from handlers.gallery_handler import GalleryHandler
from handlers.library_handler import LibraryHandler
from runtime_config.runtime_config import RuntimeConfig
from handlers.contact_sheet_handler import ContactSheetHandler
from handlers.enhance_prompt_handler import EnhancePromptHandler
from handlers.prompt_handler import PromptHandler
from handlers.receive_job_handler import ReceiveJobHandler
from handlers.style_guide_handler import StyleGuideHandler
from handlers.sync_handler import SyncHandler
from services.interfaces import (
    A2VPipeline,
    FastVideoPipeline,
    ImageAPIClient,
    ImageGenerationPipeline,
    PaletteSyncClient,
    VideoAPIClient,
    GpuCleaner,
    GpuInfo,
    HTTPClient,
    IcLoraModelDownloader,
    IcLoraPipeline,
    LTXAPIClient,
    ModelDownloader,
    RetakePipeline,
    TaskRunner,
    TextEncoder,
    VideoProcessor,
)
from state.app_state_types import AppState, StartupPending, TextEncoderState


class AppHandler:
    """Composition-only state service exposing typed domain handlers."""

    def __init__(
        self,
        config: RuntimeConfig,
        default_settings: AppSettings,
        http: HTTPClient,
        gpu_cleaner: GpuCleaner,
        model_downloader: ModelDownloader,
        gpu_info: GpuInfo,
        video_processor: VideoProcessor,
        text_encoder: TextEncoder,
        task_runner: TaskRunner,
        ltx_api_client: LTXAPIClient,
        image_api_client: ImageAPIClient,
        video_api_client: VideoAPIClient,
        palette_sync_client: PaletteSyncClient,
        fast_video_pipeline_class: type[FastVideoPipeline],
        image_generation_pipeline_class: type[ImageGenerationPipeline],
        ic_lora_pipeline_class: type[IcLoraPipeline],
        a2v_pipeline_class: type[A2VPipeline],
        retake_pipeline_class: type[RetakePipeline],
        ic_lora_model_downloader: IcLoraModelDownloader,
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
        self.image_api_client = image_api_client
        self.video_api_client = video_api_client
        self.palette_sync_client = palette_sync_client
        self.fast_video_pipeline_class = fast_video_pipeline_class
        self.image_generation_pipeline_class = image_generation_pipeline_class
        self.ic_lora_pipeline_class = ic_lora_pipeline_class
        self.a2v_pipeline_class = a2v_pipeline_class
        self.retake_pipeline_class = retake_pipeline_class
        self.ic_lora_model_downloader = ic_lora_model_downloader

        self._lock = threading.RLock()

        self.state = AppState(
            available_files={
                "checkpoint": None,
                "upsampler": None,
                "text_encoder": None,
                "zit": None,
            },
            downloading_session=None,
            gpu_slot=None,
            api_generation=None,
            cpu_slot=None,
            text_encoder=TextEncoderState(service=text_encoder),
            startup=StartupPending(message="Not started"),
            app_settings=default_settings.model_copy(deep=True),
        )

        # ============================================================
        # Handlers (wired in dependency order)
        # ============================================================

        self.settings = SettingsHandler(
            state=self.state,
            lock=self._lock,
            settings_file=config.settings_file,
        )
        self.settings.load_settings(default_settings)

        self.models = ModelsHandler(
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
            a2v_pipeline_class=a2v_pipeline_class,
            retake_pipeline_class=retake_pipeline_class,
            config=config,
            outputs_dir=config.outputs_dir,
            device=config.device,
        )

        self.generation = GenerationHandler(state=self.state, lock=self._lock)

        self.video_generation = VideoGenerationHandler(
            state=self.state,
            lock=self._lock,
            generation_handler=self.generation,
            pipelines_handler=self.pipelines,
            text_handler=self.text,
            ltx_api_client=ltx_api_client,
            outputs_dir=config.outputs_dir,
            config=config,
            camera_motion_prompts=config.camera_motion_prompts,
            default_negative_prompt=config.default_negative_prompt,
        )

        self.image_generation = ImageGenerationHandler(
            state=self.state,
            lock=self._lock,
            generation_handler=self.generation,
            pipelines_handler=self.pipelines,
            outputs_dir=config.outputs_dir,
            config=config,
            image_api_client=image_api_client,
        )

        self.health = HealthHandler(
            state=self.state,
            lock=self._lock,
            models_handler=self.models,
            pipelines_handler=self.pipelines,
            gpu_info=gpu_info,
            config=config,
            use_sage_attention=config.use_sage_attention,
        )

        self.runtime_policy = RuntimePolicyHandler(config=config)

        self.suggest_gap_prompt = SuggestGapPromptHandler(
            state=self.state,
            lock=self._lock,
            http=http,
        )

        self.enhance_prompt = EnhancePromptHandler(
            state=self.state,
            lock=self._lock,
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
            outputs_dir=config.outputs_dir,
        )

        self.ic_lora = IcLoraHandler(
            state=self.state,
            lock=self._lock,
            generation_handler=self.generation,
            pipelines_handler=self.pipelines,
            text_handler=self.text,
            video_processor=video_processor,
            ic_lora_model_downloader=ic_lora_model_downloader,
            ic_lora_dir=config.ic_lora_dir,
            outputs_dir=config.outputs_dir,
        )

        self.sync = SyncHandler(
            state=self.state,
            palette_sync_client=palette_sync_client,
        )

        self.gallery = GalleryHandler(outputs_dir=config.outputs_dir)

        self.downloads.cleanup_downloading_dir()

        from state.job_queue import JobQueue
        self.job_queue = JobQueue(persistence_path=config.settings_file.parent / "job_queue.json")

        from state.library_store import LibraryStore
        library_store = LibraryStore(config.settings_file.parent / "library")
        self.library = LibraryHandler(store=library_store)

        self.prompts = PromptHandler(
            state=self.state,
            lock=self._lock,
            store_path=config.settings_file.parent / "prompt_store.json",
        )

        self.receive_job_handler = ReceiveJobHandler(
            state=self.state,
            http=http,
            job_queue=self.job_queue,
        )

        self.contact_sheet = ContactSheetHandler(job_queue=self.job_queue)
        self.style_guide = StyleGuideHandler(job_queue=self.job_queue)

        self.models.refresh_available_files()

    def determine_slot(self, model: str) -> str:
        """Determine whether a job should use the gpu or api slot."""
        always_api_models = {"seedance-1.5-pro", "nano-banana-2"}
        if model in always_api_models:
            return "api"
        if self.config.force_api_generations:
            return "api"
        return "gpu"


@dataclass
class ServiceBundle:
    http: HTTPClient
    gpu_cleaner: GpuCleaner
    model_downloader: ModelDownloader
    gpu_info: GpuInfo
    video_processor: VideoProcessor
    text_encoder: TextEncoder
    task_runner: TaskRunner
    ltx_api_client: LTXAPIClient
    image_api_client: ImageAPIClient
    video_api_client: VideoAPIClient
    palette_sync_client: PaletteSyncClient
    fast_video_pipeline_class: type[FastVideoPipeline]
    image_generation_pipeline_class: type[ImageGenerationPipeline]
    ic_lora_pipeline_class: type[IcLoraPipeline]
    a2v_pipeline_class: type[A2VPipeline]
    retake_pipeline_class: type[RetakePipeline]
    ic_lora_model_downloader: IcLoraModelDownloader


def build_default_service_bundle(config: RuntimeConfig) -> ServiceBundle:
    """Build real runtime services with lazy heavy imports isolated from tests."""
    from services.fast_video_pipeline.ltx_fast_video_pipeline import LTXFastVideoPipeline
    from services.image_api_client.replicate_client_impl import ReplicateImageClientImpl
    from services.video_api_client.replicate_video_client_impl import ReplicateVideoClientImpl
    from services.gpu_cleaner.torch_cleaner import TorchCleaner
    from services.gpu_info.gpu_info_impl import GpuInfoImpl
    from services.http_client.http_client_impl import HTTPClientImpl
    from services.ic_lora_model_downloader.ic_lora_model_downloader_impl import IcLoraModelDownloaderImpl
    from services.a2v_pipeline.ltx_a2v_pipeline import LTXa2vPipeline
    from services.ic_lora_pipeline.ltx_ic_lora_pipeline import LTXIcLoraPipeline
    from services.image_generation_pipeline.zit_image_generation_pipeline import ZitImageGenerationPipeline
    from services.ltx_api_client.ltx_api_client_impl import LTXAPIClientImpl
    from services.model_downloader.hugging_face_downloader import HuggingFaceDownloader
    from services.retake_pipeline.ltx_retake_pipeline import LTXRetakePipeline
    from services.task_runner.threading_runner import ThreadingRunner
    from services.text_encoder.ltx_text_encoder import LTXTextEncoder
    from services.palette_sync_client.palette_sync_client_impl import PaletteSyncClientImpl
    from services.video_processor.video_processor_impl import VideoProcessorImpl

    http = HTTPClientImpl()

    return ServiceBundle(
        http=http,
        gpu_cleaner=TorchCleaner(device=config.device),
        model_downloader=HuggingFaceDownloader(),
        gpu_info=GpuInfoImpl(),
        video_processor=VideoProcessorImpl(),
        text_encoder=LTXTextEncoder(
            device=config.device,
            http=http,
            ltx_api_base_url=config.ltx_api_base_url,
        ),
        task_runner=ThreadingRunner(),
        ltx_api_client=LTXAPIClientImpl(http=http, ltx_api_base_url=config.ltx_api_base_url),
        image_api_client=ReplicateImageClientImpl(http=http),
        video_api_client=ReplicateVideoClientImpl(http=http),
        palette_sync_client=PaletteSyncClientImpl(http=http),
        fast_video_pipeline_class=LTXFastVideoPipeline,
        image_generation_pipeline_class=ZitImageGenerationPipeline,
        ic_lora_pipeline_class=LTXIcLoraPipeline,
        a2v_pipeline_class=LTXa2vPipeline,
        retake_pipeline_class=LTXRetakePipeline,
        ic_lora_model_downloader=IcLoraModelDownloaderImpl(),
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
        gpu_info=bundle.gpu_info,
        video_processor=bundle.video_processor,
        text_encoder=bundle.text_encoder,
        task_runner=bundle.task_runner,
        ltx_api_client=bundle.ltx_api_client,
        image_api_client=bundle.image_api_client,
        video_api_client=bundle.video_api_client,
        palette_sync_client=bundle.palette_sync_client,
        fast_video_pipeline_class=bundle.fast_video_pipeline_class,
        image_generation_pipeline_class=bundle.image_generation_pipeline_class,
        ic_lora_pipeline_class=bundle.ic_lora_pipeline_class,
        a2v_pipeline_class=bundle.a2v_pipeline_class,
        retake_pipeline_class=bundle.retake_pipeline_class,
        ic_lora_model_downloader=bundle.ic_lora_model_downloader,
    )
