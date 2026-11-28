"""Desktop service protocols. GPU pipeline protocols live in ``inference``."""

from __future__ import annotations

from typing import Literal

from services.depth_processor_pipeline.depth_processor_pipeline import DepthProcessorPipeline
from services.zit_api_client.zit_api_client import ZitAPIClient
from services.gpu_cleaner.gpu_cleaner import GpuCleaner
from services.gpu_info.gpu_info import GpuInfo, GpuTelemetryPayload
from services.http_client.http_client import HTTPClient, HttpResponseLike, HttpTransportError
from services.image_generation_pipeline.image_generation_pipeline import ImageGenerationPipeline
from services.ltx_api_client.ltx_api_client import LTXAPIClient
from services.model_downloader.model_downloader import ModelDownloader
from services.pose_processor_pipeline.pose_processor_pipeline import PoseProcessorPipeline
from services.prompt_enhancer_pipeline.prompt_enhancer_pipeline import PromptEnhancerPipeline
from services.services_utils import JSONScalar, JSONValue
from services.task_runner.task_runner import TaskRunner
from services.text_encoder.text_encoder import TextEncoder
from services.video_processor.video_processor import VideoInfoPayload, VideoProcessor

VideoPipelineModelType = Literal["fast"]

__all__ = [
    "JSONScalar",
    "JSONValue",
    "GpuTelemetryPayload",
    "VideoInfoPayload",
    "HttpTransportError",
    "HttpResponseLike",
    "HTTPClient",
    "ModelDownloader",
    "GpuCleaner",
    "GpuInfo",
    "VideoProcessor",
    "DepthProcessorPipeline",
    "PoseProcessorPipeline",
    "TaskRunner",
    "VideoPipelineModelType",
    "ZitAPIClient",
    "ImageGenerationPipeline",
    "LTXAPIClient",
    "TextEncoder",
    "PromptEnhancerPipeline",
]
