"""Test doubles for backend side-effect services."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from PIL import Image
from api_types import DetectedModel, ImageConditioningInput, VideoCameraMotion
from services.interfaces import IcLoraDownloadPayload, IcLoraModelPayload, VideoInfoPayload
from services.ltx_api_client.ltx_api_client import LTXRetakeResult
from tests.fakes.fake_gpu_info import FakeGpuInfo


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
        self.raise_on_upload_file: Exception | None = None
        self.raise_on_text_to_video: Exception | None = None
        self.raise_on_image_to_video: Exception | None = None
        self.raise_on_audio_to_video: Exception | None = None
        self.raise_on_retake: Exception | None = None
        self.text_to_video_result = b"fake-ltx-api-t2v-video"
        self.image_to_video_result = b"fake-ltx-api-i2v-video"
        self.audio_to_video_result = b"fake-ltx-api-a2v-video"
        self.retake_result = LTXRetakeResult(video_bytes=b"fake-ltx-api-retake-video", result_payload=None)
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
        last_frame_uri: str | None = None,
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
                "last_frame_uri": last_frame_uri,
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
        last_frame_uri: str | None = None,
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
                "last_frame_uri": last_frame_uri,
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
        mode: str,
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


class FakeImageAPIClient:
    def __init__(self) -> None:
        self.configured = True
        self.text_to_image_calls: list[dict[str, Any]] = []
        self.raise_on_text_to_image: Exception | None = None
        self.text_to_image_result = b"fake-api-image"

    def is_configured(self) -> bool:
        return self.configured

    def generate_text_to_image(
        self,
        *,
        api_key: str,
        model: str,
        prompt: str,
        width: int,
        height: int,
        seed: int,
        num_inference_steps: int,
    ) -> bytes:
        self.text_to_image_calls.append(
            {
                "api_key": api_key,
                "model": model,
                "prompt": prompt,
                "width": width,
                "height": height,
                "seed": seed,
                "num_inference_steps": num_inference_steps,
            }
        )
        if self.raise_on_text_to_image is not None:
            raise self.raise_on_text_to_image
        return self.text_to_image_result


class FakeVideoAPIClient:
    def __init__(self) -> None:
        self.text_to_video_calls: list[dict[str, Any]] = []
        self.raise_on_text_to_video: Exception | None = None
        self.text_to_video_result = b"fake-seedance-video"

    def generate_text_to_video(
        self,
        *,
        api_key: str,
        model: str,
        prompt: str,
        duration: int,
        resolution: str,
        aspect_ratio: str,
        generate_audio: bool,
        last_frame: str | None = None,
    ) -> bytes:
        self.text_to_video_calls.append({
            "api_key": api_key,
            "model": model,
            "prompt": prompt,
            "duration": duration,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "generate_audio": generate_audio,
            "last_frame": last_frame,
        })
        if self.raise_on_text_to_video is not None:
            raise self.raise_on_text_to_video
        return self.text_to_video_result


class FakePaletteSyncClient:
    def __init__(self) -> None:
        self.validate_calls: list[str] = []
        self.credits_calls: list[str] = []
        self.login_calls: list[tuple[str, str]] = []
        self.raise_on_validate: Exception | None = None
        self.raise_on_login: Exception | None = None
        self.user_info: dict[str, Any] = {"id": "user-123", "email": "test@example.com", "name": "Test User"}
        self.credits_info: dict[str, Any] = {
            "balance_cents": 5000,
            "lifetime_purchased_cents": 10000,
            "lifetime_used_cents": 5000,
            "pricing": {
                "video_t2v": 40, "video_i2v": 40, "video_seedance": 80,
                "image": 20, "image_edit": 20, "audio": 15, "text_enhance": 3,
            },
        }

    def validate_connection(self, *, api_key: str) -> dict[str, Any]:
        self.validate_calls.append(api_key)
        if self.raise_on_validate is not None:
            raise self.raise_on_validate
        return self.user_info

    def sign_in_with_email(self, *, email: str, password: str) -> dict[str, Any]:
        self.login_calls.append((email, password))
        if self.raise_on_login is not None:
            raise self.raise_on_login
        return {
            "access_token": "fake-jwt-token",
            "refresh_token": "fake-refresh-token",
            "user": self.user_info,
        }

    def refresh_access_token(self, *, refresh_token: str) -> dict[str, Any]:
        return {
            "access_token": "refreshed-jwt-token",
            "refresh_token": "refreshed-refresh-token",
            "user": self.user_info,
        }

    def get_credits(self, *, api_key: str) -> dict[str, Any]:
        self.credits_calls.append(api_key)
        return self.credits_info

    def check_credits(
        self, *, api_key: str, generation_type: str, count: int,
    ) -> dict[str, Any]:
        cost = {"video_t2v": 40, "video_i2v": 40, "video_seedance": 80, "image": 20, "text_enhance": 3}.get(generation_type, 40)
        total = cost * count
        balance = self.credits_info.get("balance_cents", 5000)
        if not isinstance(balance, int):
            balance = 5000
        return {
            "can_afford": balance >= total,
            "cost_cents": total,
            "balance_cents": balance,
            "balance_after_cents": balance - total,
        }

    def deduct_credits(
        self, *, api_key: str, generation_type: str, count: int,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        cost = {"video_t2v": 40, "video_i2v": 40, "video_seedance": 80, "image": 20, "text_enhance": 3}.get(generation_type, 40)
        total = cost * count
        balance = self.credits_info.get("balance_cents", 5000)
        if not isinstance(balance, int):
            balance = 5000
        new_balance = balance - total
        self.credits_info["balance_cents"] = new_balance
        return {"deducted_cents": total, "balance_cents": new_balance}

    def list_gallery(
        self, *, api_key: str, page: int, per_page: int, asset_type: str,
    ) -> dict[str, Any]:
        return {"items": [], "total": 0, "page": page, "per_page": per_page, "total_pages": 0}

    def list_characters(self, *, api_key: str) -> dict[str, Any]:
        return {"characters": []}

    def list_styles(self, *, api_key: str) -> dict[str, Any]:
        return {"styles": [], "brands": []}

    def list_references(self, *, api_key: str, category: str | None) -> dict[str, Any]:
        return {"references": []}

    def list_loras(self, *, api_key: str) -> dict[str, Any]:
        return {"loras": []}

    def enhance_prompt(self, *, api_key: str, prompt: str, level: str) -> dict[str, Any]:
        return {"enhanced_prompt": f"Enhanced ({level}): {prompt}"}


class FakeModelScanner:
    def __init__(self) -> None:
        self._models: list[DetectedModel] = []

    def set_models(self, models: list[DetectedModel]) -> None:
        self._models = list(models)

    def scan_video_models(self, folder: Path) -> list[DetectedModel]:  # noqa: ARG002
        return list(self._models)


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
        on_progress: Callable[[int, int], None] | None = None,
    ) -> Path:
        self._raise_if_needed()
        self.calls.append({"kind": "file", "repo_id": repo_id, "filename": filename, "local_dir": local_dir, "on_progress": on_progress})

        if on_progress is not None:
            on_progress(512, 1024)
            on_progress(1024, 1024)

        destination = Path(local_dir) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"\x00" * 1024)
        return destination

    def download_snapshot(
        self,
        repo_id: str,
        local_dir: str,
        on_progress: Callable[[int, int], None] | None = None,
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
            on_progress(512, 1024)
            on_progress(1024, 1024)

        root = Path(local_dir)
        root.mkdir(parents=True, exist_ok=True)

        (root / "model.safetensors").write_bytes(b"\x00" * 1024)

        return root


class FakeGpuCleaner:
    def __init__(self) -> None:
        self.cleanup_calls = 0

    def cleanup(self) -> None:
        self.cleanup_calls += 1

    def deep_cleanup(self) -> None:
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

    def register_video(self, path: str, capture: FakeCapture) -> None:
        self.videos[path] = capture

    def open_video(self, path: str) -> FakeCapture:
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

    def apply_canny(self, frame: Any) -> Any:
        return f"canny:{frame}"

    def apply_depth(self, frame: Any) -> Any:
        return f"depth:{frame}"

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
        lora_path: str | None = None,
        lora_weight: float = 1.0,
    ) -> "FakeFastVideoPipeline":
        del checkpoint_path, gemma_root, upsampler_path, device, lora_path, lora_weight
        pipeline = FakeFastVideoPipeline._singleton
        if pipeline is None:
            raise RuntimeError("FakeFastVideoPipeline singleton is not bound")
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
        self.img2img_calls: list[dict[str, Any]] = []
        self.raise_on_generate: Exception | None = None

    def generate(self, **kwargs: Any) -> FakeZitOutput:
        self.generate_calls.append(kwargs)
        if self.raise_on_generate is not None:
            raise self.raise_on_generate
        return FakeZitOutput(color="blue")

    def img2img(self, **kwargs: Any) -> FakeZitOutput:
        self.img2img_calls.append(kwargs)
        if self.raise_on_generate is not None:
            raise self.raise_on_generate
        return FakeZitOutput(color="green")

    def to(self, device: str) -> None:
        self.device = device

    def load_lora(self, lora_path: str, weight: float = 1.0) -> None:
        pass

    def unload_lora(self) -> None:
        pass


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
    ) -> "FakeIcLoraPipeline":
        del checkpoint_path, gemma_root, upsampler_path, lora_path, device
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
    ) -> "FakeA2VPipeline":
        del checkpoint_path, gemma_root, upsampler_path, device
        pipeline = FakeA2VPipeline._singleton
        if pipeline is None:
            raise RuntimeError("FakeA2VPipeline singleton is not bound")
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
        *,
        loras: list[object] | None = None,
        quantization: object | None = None,
    ) -> "FakeRetakePipeline":
        del checkpoint_path, gemma_root, device, loras, quantization
        pipeline = FakeRetakePipeline._singleton
        if pipeline is None:
            raise RuntimeError("FakeRetakePipeline singleton is not bound")
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
        output_path.write_bytes(b"fake-retake-video")


class FakeIcLoraModelDownloader:
    _MODEL_FILES = {
        "canny": "ltx-2-19b-ic-lora-canny-control.safetensors",
        "depth": "ltx-2-19b-ic-lora-depth-control.safetensors",
        "pose": "ltx-2-19b-ic-lora-pose-control.safetensors",
        "detailer": "ltx-2-19b-ic-lora-detailer.safetensors",
    }

    def __init__(self) -> None:
        self.download_calls: list[str] = []
        self.fail_next: Exception | None = None

    def list_models(self, directory: Path) -> list[IcLoraModelPayload]:
        if not directory.exists():
            return []

        models: list[IcLoraModelPayload] = []
        for file_path in sorted(directory.iterdir()):
            if file_path.suffix != ".safetensors" or not file_path.is_file():
                continue
            models.append(
                {
                    "name": file_path.stem,
                    "path": str(file_path),
                    "conditioning_type": "unknown",
                    "reference_downscale_factor": 1,
                }
            )
        return models

    def download_model(self, model_name: str, directory: Path) -> IcLoraDownloadPayload:
        self.download_calls.append(model_name)
        if self.fail_next is not None:
            error = self.fail_next
            self.fail_next = None
            raise error

        filename = self._MODEL_FILES.get(model_name)
        if filename is None:
            raise ValueError(f"Unknown model: {model_name}. Must be one of: {list(self._MODEL_FILES)}")

        destination = directory / filename
        if destination.exists() and destination.stat().st_size > 1_000_000:
            return {
                "status": "complete",
                "path": str(destination),
                "already_existed": True,
            }

        directory.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"\x00" * 2_000_000)
        return {
            "status": "complete",
            "path": str(destination),
            "already_existed": False,
        }


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
    gpu_info: FakeGpuInfo = field(default_factory=FakeGpuInfo)
    video_processor: FakeVideoProcessor = field(default_factory=FakeVideoProcessor)
    text_encoder: FakeTextEncoder = field(default_factory=FakeTextEncoder)
    task_runner: FakeTaskRunner = field(default_factory=FakeTaskRunner)
    ltx_api_client: FakeLTXAPIClient = field(default_factory=FakeLTXAPIClient)
    image_api_client: FakeImageAPIClient = field(default_factory=FakeImageAPIClient)
    video_api_client: FakeVideoAPIClient = field(default_factory=FakeVideoAPIClient)
    palette_sync_client: FakePaletteSyncClient = field(default_factory=FakePaletteSyncClient)
    fast_video_pipeline: FakeFastVideoPipeline = field(default_factory=FakeFastVideoPipeline)
    image_generation_pipeline: FakeImageGenerationPipeline = field(default_factory=FakeImageGenerationPipeline)
    ic_lora_pipeline: FakeIcLoraPipeline = field(default_factory=FakeIcLoraPipeline)
    a2v_pipeline: FakeA2VPipeline = field(default_factory=FakeA2VPipeline)
    retake_pipeline: FakeRetakePipeline = field(default_factory=FakeRetakePipeline)
    ic_lora_model_downloader: FakeIcLoraModelDownloader = field(default_factory=FakeIcLoraModelDownloader)
    model_scanner: FakeModelScanner = field(default_factory=FakeModelScanner)

    def __post_init__(self) -> None:
        FakeFastVideoPipeline.bind_singleton(self.fast_video_pipeline)
        FakeImageGenerationPipeline.bind_singleton(self.image_generation_pipeline)
        FakeIcLoraPipeline.bind_singleton(self.ic_lora_pipeline)
        FakeA2VPipeline.bind_singleton(self.a2v_pipeline)
        FakeRetakePipeline.bind_singleton(self.retake_pipeline)
