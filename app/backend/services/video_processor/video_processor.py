"""Video processor service protocol definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypedDict

from services.services_utils import FrameArray, VideoCaptureLike, VideoWriterLike

if TYPE_CHECKING:
    from services.depth_processor_pipeline.depth_processor_pipeline import DepthProcessorPipeline
    from services.pose_processor_pipeline.pose_processor_pipeline import PoseProcessorPipeline


class VideoInfoPayload(TypedDict):
    fps: float
    frame_count: int
    width: int
    height: int


class VideoProcessor(Protocol):
    def open_video(self, path: str) -> VideoCaptureLike:
        ...

    def get_video_info(self, cap: VideoCaptureLike) -> VideoInfoPayload:
        ...

    def read_frame(self, cap: VideoCaptureLike, frame_idx: int | None = None) -> FrameArray | None:
        ...

    def read_image(self, path: str) -> FrameArray:
        ...

    def apply_canny(self, frame: FrameArray) -> FrameArray:
        ...

    def apply_depth(self, frame: FrameArray, depth_pipeline: DepthProcessorPipeline) -> FrameArray:
        ...

    def apply_pose(self, frame: FrameArray, pose_pipeline: PoseProcessorPipeline) -> FrameArray:
        ...

    def encode_frame_jpeg(self, frame: FrameArray, quality: int = 85) -> bytes:
        ...

    def create_writer(self, path: str, fourcc: str, fps: float, size: tuple[int, int]) -> VideoWriterLike:
        ...

    def release(self, cap_or_writer: VideoCaptureLike | VideoWriterLike) -> None:
        ...
