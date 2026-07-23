"""Test doubles for backend side-effect services."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from PIL import Image
from api_types import (
    ExtendMode,
    ImageConditioningInput,
    InstructionSection,
    IcLoraCatalogItem,
    IcLoraControl,
    LoraCatalogFile,
    LoraCatalogItem,
    DownloadSpec,
    DownloadVariant,
    InputSpec,
    PreprocessingStep,
    IcLoraSettings,
    RetakeMode,
    VideoCameraMotion,
)
from services.interfaces import VideoInfoPayload
from services.ltx_api_client.ltx_api_client import LTXRetakeResult
from tests.fakes.fake_gpu_info import FakeGpuInfo


def _dl(repo_id: str, filename: str, size_bytes: int = 10) -> DownloadSpec:
    """Single-checkpoint DownloadSpec (first/only variant = default)."""
    return DownloadSpec(
        repo_id=repo_id,
        variants=[DownloadVariant(id="default", label="Default", filename=filename, size_bytes=size_bytes)],
    )


@dataclass
class FakeResponse:
    status_code: int = 200
    text: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    content: bytes = b""
    json_payload: Any = field(default_factory=dict)

    def json(self) -> Any:
        return self.json_payload


@dataclass
class HttpCall:
    method: str
    url: str
    headers: dict[str, str] | None
    json_payload: dict[str, Any] | None
    data: Any
    timeout: int


class FakeHTTPClient:
    def __init__(self) -> None:
        self.calls: list[HttpCall] = []
        self._queues: dict[str, list[FakeResponse | Exception]] = {
            "post": [],
            "get": [],
            "put": [],
        }

    def queue(self, method: str, *items: FakeResponse | Exception) -> None:
        self._queues[method].extend(items)

    def _dequeue(self, method: str) -> FakeResponse:
        queue = self._queues[method]
        if not queue:
            raise RuntimeError(f"No queued {method.upper()} response")
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def post(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        json_payload: dict[str, Any] | None = None,
        data: Any = None,
        timeout: int = 30,
    ) -> FakeResponse:
        self.calls.append(HttpCall("post", url, headers, json_payload, data, timeout))
        return self._dequeue("post")

    def get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: int = 30,
    ) -> FakeResponse:
        self.calls.append(HttpCall("get", url, headers, None, None, timeout))
        return self._dequeue("get")

    def put(
        self,
        url: str,
        data: Any = None,
        headers: dict[str, str] | None = None,
        timeout: int = 300,
    ) -> FakeResponse:
        self.calls.append(HttpCall("put", url, headers, None, data, timeout))
        return self._dequeue("put")


class FakeTaskRunner:
    def __init__(self) -> None:
        self.jobs_run = 0
        self.last_task_name: str | None = None
        self.errors: list[Exception] = []

    def run_background(
        self,
        target,
        *,
        task_name: str,
        on_error=None,
        daemon: bool = True,
    ) -> None:  # noqa: ARG002
        self.jobs_run += 1
        self.last_task_name = task_name
        try:
            target()
        except Exception as exc:
            self.errors.append(exc)
            if on_error is not None:
                on_error(exc)


class FakeLTXAPIClient:
    def __init__(self) -> None:
        self.upload_file_calls: list[dict[str, Any]] = []
        self.text_to_video_calls: list[dict[str, Any]] = []
        self.image_to_video_calls: list[dict[str, Any]] = []
        self.audio_to_video_calls: list[dict[str, Any]] = []
        self.retake_calls: list[dict[str, Any]] = []
        self.extend_calls: list[dict[str, Any]] = []
        self.raise_on_upload_file: Exception | None = None
        self.raise_on_text_to_video: Exception | None = None
        self.raise_on_image_to_video: Exception | None = None
        self.raise_on_audio_to_video: Exception | None = None
        self.raise_on_retake: Exception | None = None
        self.raise_on_extend: Exception | None = None
        self.text_to_video_result = b"fake-ltx-api-t2v-video"
        self.image_to_video_result = b"fake-ltx-api-i2v-video"
        self.audio_to_video_result = b"fake-ltx-api-a2v-video"
        self.retake_result = LTXRetakeResult(video_bytes=b"fake-ltx-api-retake-video", result_payload=None)
        self.extend_result = LTXRetakeResult(video_bytes=b"fake-ltx-api-extend-video", result_payload=None)
        self.upload_file_results: dict[str, str] = {}

    def upload_file(
        self,
        *,
        api_key: str,
        file_path: str,
    ) -> str:
        self.upload_file_calls.append(
            {
                "api_key": api_key,
                "file_path": file_path,
            }
        )
        if self.raise_on_upload_file is not None:
            raise self.raise_on_upload_file
        default_uri = f"storage://uploaded/{Path(file_path).name}"
        return self.upload_file_results.get(file_path, default_uri)

    def generate_text_to_video(
        self,
        *,
        api_key: str,
        prompt: str,
        model: str,
        resolution: str,
        duration: float,
        fps: float,
        generate_audio: bool,
        camera_motion: VideoCameraMotion = "none",
    ) -> bytes:
        self.text_to_video_calls.append(
            {
                "api_key": api_key,
                "prompt": prompt,
                "model": model,
                "resolution": resolution,
                "duration": duration,
                "fps": fps,
                "generate_audio": generate_audio,
                "camera_motion": camera_motion,
            }
        )
        if self.raise_on_text_to_video is not None:
            raise self.raise_on_text_to_video
        return self.text_to_video_result

    def generate_image_to_video(
        self,
        *,
        api_key: str,
        prompt: str,
        image_uri: str,
        model: str,
        resolution: str,
        duration: float,
        fps: float,
        generate_audio: bool,
        camera_motion: VideoCameraMotion = "none",
    ) -> bytes:
        self.image_to_video_calls.append(
            {
                "api_key": api_key,
                "prompt": prompt,
                "image_uri": image_uri,
                "model": model,
                "resolution": resolution,
                "duration": duration,
                "fps": fps,
                "generate_audio": generate_audio,
                "camera_motion": camera_motion,
            }
        )
        if self.raise_on_image_to_video is not None:
            raise self.raise_on_image_to_video
        return self.image_to_video_result

    def generate_audio_to_video(
        self,
        *,
        api_key: str,
        prompt: str,
        audio_uri: str,
        image_uri: str | None,
        model: str,
        resolution: str,
    ) -> bytes:
        self.audio_to_video_calls.append(
            {
                "api_key": api_key,
                "prompt": prompt,
                "audio_uri": audio_uri,
                "image_uri": image_uri,
                "model": model,
                "resolution": resolution,
            }
        )
        if self.raise_on_audio_to_video is not None:
            raise self.raise_on_audio_to_video
        return self.audio_to_video_result

    def retake(
        self,
        *,
        api_key: str,
        video_path: str,
        start_time: float,
        duration: float,
        prompt: str,
        mode: RetakeMode,
    ) -> LTXRetakeResult:
        self.retake_calls.append(
            {
                "api_key": api_key,
                "video_path": video_path,
                "start_time": start_time,
                "duration": duration,
                "prompt": prompt,
                "mode": mode,
            }
        )
        if self.raise_on_retake is not None:
            raise self.raise_on_retake
        return self.retake_result

    def extend(
        self,
        *,
        api_key: str,
        video_path: str,
        duration: float,
        prompt: str,
        mode: ExtendMode,
    ) -> LTXRetakeResult:
        self.extend_calls.append(
            {
                "api_key": api_key,
                "video_path": video_path,
                "duration": duration,
                "prompt": prompt,
                "mode": mode,
            }
        )
        if self.raise_on_extend is not None:
            raise self.raise_on_extend
        return self.extend_result


class FakeZitAPIClient:
    def __init__(self) -> None:
        self.configured = True
        self.text_to_image_calls: list[dict[str, Any]] = []
        self.raise_on_text_to_image: Exception | None = None
        self.fail_text_to_image_after: int | None = None  # raise once len(calls) exceeds this
        self.text_to_image_result = b"fake-zit-api-image"
        self.image_to_image_calls: list[dict[str, Any]] = []
        self.raise_on_image_to_image: Exception | None = None
        self.fail_image_to_image_after: int | None = None  # raise once len(calls) exceeds this
        self.image_to_image_result = b"fake-zit-api-edit"

    def is_configured(self) -> bool:
        return self.configured

    def generate_text_to_image(
        self,
        *,
        api_key: str,
        prompt: str,
        width: int,
        height: int,
        seed: int,
        num_inference_steps: int,
    ) -> bytes:
        self.text_to_image_calls.append(
            {
                "api_key": api_key,
                "prompt": prompt,
                "width": width,
                "height": height,
                "seed": seed,
                "num_inference_steps": num_inference_steps,
            }
        )
        if self.raise_on_text_to_image is not None:
            if self.fail_text_to_image_after is None or len(self.text_to_image_calls) > self.fail_text_to_image_after:
                raise self.raise_on_text_to_image
        return self.text_to_image_result

    def generate_image_to_image(
        self,
        *,
        api_key: str,
        prompt: str,
        image_bytes: bytes,
        strength: float,
        seed: int,
        num_inference_steps: int,
    ) -> bytes:
        self.image_to_image_calls.append(
            {
                "api_key": api_key,
                "prompt": prompt,
                "image_bytes": image_bytes,
                "image_bytes_len": len(image_bytes),
                "strength": strength,
                "seed": seed,
                "num_inference_steps": num_inference_steps,
            }
        )
        if self.raise_on_image_to_image is not None:
            if self.fail_image_to_image_after is None or len(self.image_to_image_calls) > self.fail_image_to_image_after:
                raise self.raise_on_image_to_image
        return self.image_to_image_result


class FakeModelDownloader:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_next: Exception | None = None

    def _raise_if_needed(self) -> None:
        if self.fail_next is None:
            return
        error = self.fail_next
        self.fail_next = None
        raise error

    def download_file(
        self,
        repo_id: str,
        filename: str,
        local_dir: str,
        token: str | None,
        on_progress: Callable[[int], None] | None = None,
    ) -> Path:
        self._raise_if_needed()
        self.calls.append({"kind": "file", "repo_id": repo_id, "filename": filename, "local_dir": local_dir, "token": token, "on_progress": on_progress})

        if on_progress is not None:
            on_progress(512)
            on_progress(1024)

        destination = Path(local_dir) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"\x00" * 1024)
        return destination

    def download_snapshot(
        self,
        repo_id: str,
        local_dir: str,
        token: str | None,
        on_progress: Callable[[int], None] | None = None,
    ) -> Path:
        self._raise_if_needed()
        self.calls.append(
            {
                "kind": "snapshot",
                "repo_id": repo_id,
                "local_dir": local_dir,
                "on_progress": on_progress,
            }
        )

        if on_progress is not None:
            on_progress(512)
            on_progress(1024)

        root = Path(local_dir)
        root.mkdir(parents=True, exist_ok=True)

        (root / "model.safetensors").write_bytes(b"\x00" * 1024)

        return root


class FakeLoraCatalogProvider:
    def __init__(self, catalog: LoraCatalogFile | None = None) -> None:
        self._catalog = catalog or LoraCatalogFile(
            schema_version=1,
            ic_loras=[
                IcLoraCatalogItem(
                    id="ingredients-v1",
                    name="Ingredients",
                    description="d",
                    download=_dl("Lightricks/x", "i.safetensors"),
                    requires_hf_login=True,
                    input=InputSpec(kind="image"),
                    preprocessing=[PreprocessingStep(utility="image_to_frames", params={})],
                    controls=[IcLoraControl(id="duration", label="Duration", default=5, options=[5, 8])],
                    instructions=[InstructionSection(kind="summary", title="What it does", body="use it")],
                    default_settings=IcLoraSettings(skip_stage_2=True),
                ),
                IcLoraCatalogItem(
                    id="colorize-v1",
                    name="Colorize",
                    description="d",
                    download=_dl("Lightricks/y", "c.safetensors"),
                    requires_hf_login=True,
                    input=InputSpec(kind="video"),
                    preprocessing=[],
                    instructions=[InstructionSection(kind="summary", title="What it does", body="use it")],
                    default_settings=IcLoraSettings(skip_stage_2=True),
                ),
                IcLoraCatalogItem(
                    id="outpaint-v1",
                    name="Outpainting",
                    description="d",
                    download=_dl("Lightricks/op", "op.safetensors"),
                    requires_hf_login=True,
                    allows_empty_prompt=True,
                    input=InputSpec(kind="video"),
                    preprocessing=[PreprocessingStep(utility="outpaint_canvas", params={})],
                    controls=[IcLoraControl(id="outpaint_pads", kind="position_canvas", label="Canvas")],
                    instructions=[InstructionSection(kind="summary", title="What it does", body="use it")],
                    default_settings=IcLoraSettings(skip_stage_2=True),
                ),
                IcLoraCatalogItem(
                    id="refimg-v1",
                    name="Render to Real",
                    description="d",
                    download=_dl("Lightricks/rr", "rr.safetensors"),
                    requires_hf_login=False,
                    allows_reference_image=True,
                    input=InputSpec(kind="video"),
                    preprocessing=[],
                    instructions=[InstructionSection(kind="summary", title="What it does", body="use it")],
                    default_settings=IcLoraSettings(skip_stage_2=True),
                ),
                IcLoraCatalogItem(
                    id="multi-v1",
                    name="Multi Variant",
                    description="d",
                    download=DownloadSpec(
                        repo_id="org/multi",
                        variants=[
                            DownloadVariant(id="strong", label="Strong", filename="strong.safetensors", size_bytes=10),
                            DownloadVariant(id="light", label="Light", filename="light.safetensors", size_bytes=5),
                        ],
                    ),
                    requires_hf_login=False,
                    input=InputSpec(kind="video"),
                    preprocessing=[],
                    instructions=[InstructionSection(kind="summary", title="What it does", body="use it")],
                    default_settings=IcLoraSettings(skip_stage_2=True),
                ),
            ],
            loras=[
                LoraCatalogItem(
                    id="cozy-felt-v1",
                    name="Cozy Felt",
                    description="d",
                    download=_dl("vrg/felt", "felt.safetensors", size_bytes=20),
                    requires_hf_login=False,
                    trigger="F3ltCut0u7",
                    trigger_placement="anywhere",
                    tags=["style"],
                    recommended_strength=1.0,
                ),
            ],
        )

    def get_catalog(self) -> LoraCatalogFile:
        return self._catalog

    def get_ic_lora(self, ic_lora_id: str) -> IcLoraCatalogItem | None:
        return next((r for r in self._catalog.ic_loras if r.id == ic_lora_id), None)

    def get_lora(self, lora_id: str) -> LoraCatalogItem | None:
        return next((r for r in self._catalog.loras if r.id == lora_id), None)


class FakeGpuCleaner:
    def __init__(self) -> None:
        self.cleanup_calls = 0

    def cleanup(self) -> None:
        self.cleanup_calls += 1


class FakeCapture:
    def __init__(
        self,
        frames: list[Any] | None = None,
        *,
        fps: float = 24,
        width: int = 64,
        height: int = 64,
        opened: bool = True,
    ) -> None:
        self.frames = list(frames) if frames is not None else ["frame-0", "frame-1", "frame-2"]
        self.fps = fps
        self.width = width
        self.height = height
        self.opened = opened
        self.position = 0
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802
        return self.opened

    def release(self) -> None:
        self.released = True


class FakeWriter:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.frames: list[Any] = []
        self.released = False

    def write(self, frame: Any) -> None:
        self.frames.append(frame)

    def release(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(b"writer-output")
        self.released = True


class FakeVideoProcessor:
    def __init__(self) -> None:
        self.videos: dict[str, FakeCapture] = {}
        self.writers: list[FakeWriter] = []
        self.open_video_calls: list[str] = []

    def register_video(self, path: str, capture: FakeCapture) -> None:
        self.videos[path] = capture

    def open_video(self, path: str) -> FakeCapture:
        self.open_video_calls.append(path)
        return self.videos.setdefault(path, FakeCapture())

    def get_video_info(self, cap: FakeCapture) -> VideoInfoPayload:
        return {
            "fps": cap.fps,
            "frame_count": len(cap.frames),
            "width": cap.width,
            "height": cap.height,
        }

    def read_frame(self, cap: FakeCapture, frame_idx: int | None = None) -> Any | None:
        if frame_idx is not None:
            cap.position = frame_idx
        if cap.position >= len(cap.frames):
            return None
        frame = cap.frames[cap.position]
        cap.position += 1
        return frame

    def read_image(self, path: str) -> Any:  # noqa: ARG002
        import numpy as np

        return np.zeros((64, 64, 3), dtype=np.uint8)

    def apply_canny(self, frame: Any) -> Any:
        return f"canny:{frame}"

    def apply_depth(self, frame: Any, depth_pipeline: Any) -> Any:
        return depth_pipeline.apply(frame)

    def apply_pose(self, frame: Any, pose_pipeline: Any) -> Any:
        return pose_pipeline.apply(frame)

    def encode_frame_jpeg(self, frame: Any, quality: int = 85) -> bytes:  # noqa: ARG002
        return f"jpeg:{frame}".encode("utf-8")

    def create_writer(self, path: str, fourcc: str, fps: float, size: tuple[int, int]) -> FakeWriter:  # noqa: ARG002
        writer = FakeWriter(path)
        self.writers.append(writer)
        return writer

    def release(self, cap_or_writer: FakeCapture | FakeWriter) -> None:
        cap_or_writer.release()


class _FakeVideoPipelineBase:
    pipeline_kind: str

    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []
        self.warmup_calls: list[dict[str, Any]] = []
        self.compile_calls = 0
        self.create_loras: list[list[tuple[str, float]]] = []
        self.raise_on_generate: Exception | None = None

    def _record_generate(self, payload: dict[str, Any]) -> None:
        self.generate_calls.append(payload)
        if self.raise_on_generate is not None:
            raise self.raise_on_generate

        output_path = Path(payload["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-video")

    def warmup(self, output_path: str) -> None:
        self.warmup_calls.append({"output_path": output_path})
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"warmup")
        path.unlink(missing_ok=True)

    def compile_transformer(self) -> None:
        self.compile_calls += 1


class FakeFastVideoPipeline(_FakeVideoPipelineBase):
    pipeline_kind = "fast"
    _singleton: ClassVar["FakeFastVideoPipeline | None"] = None

    @classmethod
    def bind_singleton(cls, pipeline: "FakeFastVideoPipeline") -> None:
        cls._singleton = pipeline

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: str | object,
        streaming_prefetch_count: int | None,
        loras: list[tuple[str, float]] | None = None,
    ) -> "FakeFastVideoPipeline":
        del checkpoint_path, gemma_root, upsampler_path, device, streaming_prefetch_count
        pipeline = FakeFastVideoPipeline._singleton
        if pipeline is None:
            raise RuntimeError("FakeFastVideoPipeline singleton is not bound")
        pipeline.create_loras.append(list(loras or []))
        return pipeline

    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        output_path: str,
    ) -> None:
        self._record_generate(
            {
                "prompt": prompt,
                "seed": seed,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "frame_rate": frame_rate,
                "images": images,
                "output_path": output_path,
            }
        )


class FakeZitOutput:
    def __init__(self, color: str = "red") -> None:
        self.images = [Image.new("RGB", (32, 32), color)]


class FakeImageGenerationPipeline:
    _singleton: ClassVar["FakeImageGenerationPipeline | None"] = None

    @classmethod
    def bind_singleton(cls, pipeline: "FakeImageGenerationPipeline") -> None:
        cls._singleton = pipeline

    @staticmethod
    def create(
        model_path: str,
        device: str | None = None,
    ) -> "FakeImageGenerationPipeline":
        del model_path
        pipeline = FakeImageGenerationPipeline._singleton
        if pipeline is None:
            raise RuntimeError("FakeImageGenerationPipeline singleton is not bound")
        if device is not None:
            pipeline.to(device)
        return pipeline

    def __init__(self) -> None:
        self.device: str | None = None
        self.generate_calls: list[dict[str, Any]] = []
        self.raise_on_generate: Exception | None = None
        self.fail_generate_after: int | None = None  # raise once len(generate_calls) exceeds this
        self.edit_calls: list[dict[str, Any]] = []
        self.raise_on_edit: Exception | None = None
        self.fail_edit_after: int | None = None  # raise once len(edit_calls) exceeds this

    def generate(self, **kwargs: Any) -> FakeZitOutput:
        self.generate_calls.append(kwargs)
        if self.raise_on_generate is not None:
            if self.fail_generate_after is None or len(self.generate_calls) > self.fail_generate_after:
                raise self.raise_on_generate
        return FakeZitOutput(color="blue")

    def edit(self, **kwargs: Any) -> FakeZitOutput:
        self.edit_calls.append(kwargs)
        if self.raise_on_edit is not None:
            if self.fail_edit_after is None or len(self.edit_calls) > self.fail_edit_after:
                raise self.raise_on_edit
        return FakeZitOutput(color="green")

    def to(self, device: str) -> None:
        self.device = device


class FakePromptEnhancerPipeline:
    _singleton: ClassVar["FakePromptEnhancerPipeline | None"] = None

    @classmethod
    def bind_singleton(cls, pipeline: "FakePromptEnhancerPipeline") -> None:
        cls._singleton = pipeline

    @staticmethod
    def create(gemma_root: str, device: str) -> "FakePromptEnhancerPipeline":
        del gemma_root, device
        pipeline = FakePromptEnhancerPipeline._singleton
        if pipeline is None:
            raise RuntimeError("FakePromptEnhancerPipeline singleton is not bound")
        return pipeline

    def __init__(self) -> None:
        self.enhance_t2v_calls: list[dict[str, Any]] = []
        self.enhance_i2v_calls: list[dict[str, Any]] = []
        self.raise_on_enhance: Exception | None = None
        self.enhanced_prompt: str = "enhanced prompt"

    def enhance_t2v(self, prompt: str, system_prompt: str | None, seed: int) -> str:
        self.enhance_t2v_calls.append({"prompt": prompt, "system_prompt": system_prompt, "seed": seed})
        if self.raise_on_enhance is not None:
            raise self.raise_on_enhance
        return self.enhanced_prompt

    def enhance_i2v(self, prompt: str, image_path: str, system_prompt: str | None, seed: int) -> str:
        self.enhance_i2v_calls.append(
            {"prompt": prompt, "image_path": image_path, "system_prompt": system_prompt, "seed": seed}
        )
        if self.raise_on_enhance is not None:
            raise self.raise_on_enhance
        return self.enhanced_prompt


class FakeIcLoraPipeline:
    _singleton: ClassVar["FakeIcLoraPipeline | None"] = None

    @classmethod
    def bind_singleton(cls, pipeline: "FakeIcLoraPipeline") -> None:
        cls._singleton = pipeline

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        lora_path: str,
        device: str | object,
        streaming_prefetch_count: int | None,
        lora_strength: float = 1.0,
    ) -> "FakeIcLoraPipeline":
        del checkpoint_path, gemma_root, upsampler_path, lora_path, device, streaming_prefetch_count, lora_strength
        pipeline = FakeIcLoraPipeline._singleton
        if pipeline is None:
            raise RuntimeError("FakeIcLoraPipeline singleton is not bound")
        return pipeline

    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []
        self.raise_on_generate: Exception | None = None

    def generate(self, **kwargs: Any) -> None:
        self.generate_calls.append(kwargs)
        if self.raise_on_generate is not None:
            raise self.raise_on_generate

        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-ic-lora-video")


class FakeDepthProcessorPipeline:
    _singleton: ClassVar["FakeDepthProcessorPipeline | None"] = None

    @classmethod
    def bind_singleton(cls, pipeline: "FakeDepthProcessorPipeline") -> None:
        cls._singleton = pipeline

    @staticmethod
    def create(
        model_path: str,
        device: str | object,
    ) -> "FakeDepthProcessorPipeline":
        del model_path, device
        pipeline = FakeDepthProcessorPipeline._singleton
        if pipeline is None:
            raise RuntimeError("FakeDepthProcessorPipeline singleton is not bound")
        return pipeline

    def __init__(self) -> None:
        self.apply_calls: list[Any] = []

    def apply(self, frame: Any) -> Any:
        self.apply_calls.append(frame)
        return f"depth:{frame}"


class FakePoseProcessorPipeline:
    _singleton: ClassVar["FakePoseProcessorPipeline | None"] = None

    @classmethod
    def bind_singleton(cls, pipeline: "FakePoseProcessorPipeline") -> None:
        cls._singleton = pipeline

    @staticmethod
    def create(
        pose_model_path: str,
        person_detector_model_path: str,
        device: str | object,
    ) -> "FakePoseProcessorPipeline":
        del pose_model_path, person_detector_model_path, device
        pipeline = FakePoseProcessorPipeline._singleton
        if pipeline is None:
            raise RuntimeError("FakePoseProcessorPipeline singleton is not bound")
        return pipeline

    def __init__(self) -> None:
        self.apply_calls: list[Any] = []

    def apply(self, frame: Any) -> Any:
        self.apply_calls.append(frame)
        return f"pose:{frame}"


class FakeA2VPipeline:
    _singleton: ClassVar["FakeA2VPipeline | None"] = None

    @classmethod
    def bind_singleton(cls, pipeline: "FakeA2VPipeline") -> None:
        cls._singleton = pipeline

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: str | object,
        streaming_prefetch_count: int | None,
        loras: list[tuple[str, float]] | None = None,
    ) -> "FakeA2VPipeline":
        del checkpoint_path, gemma_root, upsampler_path, device, streaming_prefetch_count
        pipeline = FakeA2VPipeline._singleton
        if pipeline is None:
            raise RuntimeError("FakeA2VPipeline singleton is not bound")
        pipeline.create_loras.append(list(loras or []))
        return pipeline

    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []
        self.create_loras: list[list[tuple[str, float]]] = []
        self.raise_on_generate: Exception | None = None

    def generate(self, **kwargs: Any) -> None:
        self.generate_calls.append(kwargs)
        if self.raise_on_generate is not None:
            raise self.raise_on_generate

        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-a2v-video")


class FakeRetakePipeline:
    _singleton: ClassVar["FakeRetakePipeline | None"] = None

    @classmethod
    def bind_singleton(cls, pipeline: "FakeRetakePipeline") -> None:
        cls._singleton = pipeline

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        device: str | object,
        streaming_prefetch_count: int | None,
        *,
        loras: list[object] | None = None,
        quantization: object | None = None,
    ) -> "FakeRetakePipeline":
        del checkpoint_path, gemma_root, device, streaming_prefetch_count, loras, quantization
        pipeline = FakeRetakePipeline._singleton
        if pipeline is None:
            raise RuntimeError("FakeRetakePipeline singleton is not bound")
        return pipeline

    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []
        self.extend_calls: list[dict[str, Any]] = []
        self.raise_on_generate: Exception | None = None
        self.raise_on_extend: Exception | None = None

    def generate(self, **kwargs: Any) -> None:
        self.generate_calls.append(kwargs)
        if self.raise_on_generate is not None:
            raise self.raise_on_generate

        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-retake-video")

    def extend(self, **kwargs: Any) -> None:
        self.extend_calls.append(kwargs)
        if self.raise_on_extend is not None:
            raise self.raise_on_extend

        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-extend-video")


class FakeTextEncoder:
    def __init__(self) -> None:
        self.install_calls = 0
        self.encode_calls: list[dict[str, Any]] = []
        self.encode_responses: list[Any] = []

    def install_patches(self, state_getter) -> None:  # noqa: ARG002
        self.install_calls += 1

    def encode_via_api(self, prompt: str, api_key: str, checkpoint_path: str, enhance_prompt: bool) -> Any | None:
        self.encode_calls.append(
            {
                "prompt": prompt,
                "api_key": api_key,
                "checkpoint_path": checkpoint_path,
                "enhance_prompt": enhance_prompt,
            }
        )
        if self.encode_responses:
            return self.encode_responses.pop(0)
        return None


@dataclass
class FakeServices:
    http: FakeHTTPClient = field(default_factory=FakeHTTPClient)
    gpu_cleaner: FakeGpuCleaner = field(default_factory=FakeGpuCleaner)
    model_downloader: FakeModelDownloader = field(default_factory=FakeModelDownloader)
    lora_catalog_provider: FakeLoraCatalogProvider = field(default_factory=FakeLoraCatalogProvider)
    gpu_info: FakeGpuInfo = field(default_factory=FakeGpuInfo)
    video_processor: FakeVideoProcessor = field(default_factory=FakeVideoProcessor)
    text_encoder: FakeTextEncoder = field(default_factory=FakeTextEncoder)
    task_runner: FakeTaskRunner = field(default_factory=FakeTaskRunner)
    ltx_api_client: FakeLTXAPIClient = field(default_factory=FakeLTXAPIClient)
    zit_api_client: FakeZitAPIClient = field(default_factory=FakeZitAPIClient)
    fast_video_pipeline: FakeFastVideoPipeline = field(default_factory=FakeFastVideoPipeline)
    image_generation_pipeline: FakeImageGenerationPipeline = field(default_factory=FakeImageGenerationPipeline)
    ic_lora_pipeline: FakeIcLoraPipeline = field(default_factory=FakeIcLoraPipeline)
    depth_processor_pipeline: FakeDepthProcessorPipeline = field(default_factory=FakeDepthProcessorPipeline)
    pose_processor_pipeline: FakePoseProcessorPipeline = field(default_factory=FakePoseProcessorPipeline)
    a2v_pipeline: FakeA2VPipeline = field(default_factory=FakeA2VPipeline)
    retake_pipeline: FakeRetakePipeline = field(default_factory=FakeRetakePipeline)
    prompt_enhancer_pipeline: FakePromptEnhancerPipeline = field(default_factory=FakePromptEnhancerPipeline)

    def __post_init__(self) -> None:
        FakeFastVideoPipeline.bind_singleton(self.fast_video_pipeline)
        FakeImageGenerationPipeline.bind_singleton(self.image_generation_pipeline)
        FakeIcLoraPipeline.bind_singleton(self.ic_lora_pipeline)
        FakeDepthProcessorPipeline.bind_singleton(self.depth_processor_pipeline)
        FakePoseProcessorPipeline.bind_singleton(self.pose_processor_pipeline)
        FakeA2VPipeline.bind_singleton(self.a2v_pipeline)
        FakeRetakePipeline.bind_singleton(self.retake_pipeline)
        FakePromptEnhancerPipeline.bind_singleton(self.prompt_enhancer_pipeline)
