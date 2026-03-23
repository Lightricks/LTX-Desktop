"""Compatibility re-exports for service interfaces."""

from __future__ import annotations

from typing import Literal

from services.a2v_pipeline.a2v_pipeline import A2VPipeline
from services.fast_video_pipeline.fast_video_pipeline import FastVideoPipeline
from services.image_api_client.image_api_client import ImageAPIClient
from services.gpu_cleaner.gpu_cleaner import GpuCleaner
from services.gpu_info.gpu_info import GpuInfo, GpuTelemetryPayload
from services.http_client.http_client import HTTPClient, HttpResponseLike, HttpTimeoutError
from services.ic_lora_model_downloader.ic_lora_model_downloader import (
    IcLoraDownloadPayload,
    IcLoraModelDownloader,
    IcLoraModelPayload,
)
from services.ic_lora_pipeline.ic_lora_pipeline import IcLoraPipeline
from services.image_generation_pipeline.image_generation_pipeline import ImageGenerationPipeline
from services.ltx_api_client.ltx_api_client import LTXAPIClient
from services.retake_pipeline.retake_pipeline import RetakePipeline
from services.model_downloader.model_downloader import ModelDownloader
from services.services_utils import JSONScalar, JSONValue
from services.task_runner.task_runner import TaskRunner
from services.text_encoder.text_encoder import TextEncoder
from services.model_scanner.model_scanner import ModelScanner
from services.palette_sync_client.palette_sync_client import PaletteSyncClient
from services.video_api_client.video_api_client import VideoAPIClient
from services.video_processor.video_processor import VideoInfoPayload, VideoProcessor

VideoPipelineModelType = Literal["fast"]

__all__ = [
    "A2VPipeline",
    "JSONScalar",
    "JSONValue",
    "GpuTelemetryPayload",
    "VideoInfoPayload",
    "IcLoraModelPayload",
    "IcLoraDownloadPayload",
    "HttpTimeoutError",
    "HttpResponseLike",
    "HTTPClient",
    "ModelDownloader",
    "GpuCleaner",
    "GpuInfo",
    "VideoProcessor",
    "TaskRunner",
    "VideoPipelineModelType",
    "FastVideoPipeline",
    "ImageAPIClient",
    "ImageGenerationPipeline",
    "IcLoraPipeline",
    "IcLoraModelDownloader",
    "LTXAPIClient",
    "ModelScanner",
    "PaletteSyncClient",
    "RetakePipeline",
    "TextEncoder",
    "VideoAPIClient",
]
