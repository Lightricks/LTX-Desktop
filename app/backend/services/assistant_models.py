"""Verified Qwen weights and vision projectors, using the shared downloader."""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from api_types import LocalAssistantModel
from services.llama_runtime import AssistantModelError, LlamaRuntime, RuntimeAsset
from services.model_downloader.model_downloader import ModelDownloader


@dataclass(frozen=True)
class AssistantModelSpec:
    name: str
    repo: str
    weights: RuntimeAsset
    projector: RuntimeAsset

    @property
    def size(self) -> int:
        return self.weights.size + self.projector.size


LOCAL_ASSISTANT_MODELS: dict[LocalAssistantModel, AssistantModelSpec] = {
    "qwen3.5:9b": AssistantModelSpec(
        "Qwen 3.5 9B", "unsloth/Qwen3.5-9B-GGUF",
        RuntimeAsset("Qwen3.5-9B-Q4_K_M.gguf", 5_680_522_464,
                     "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8"),
        RuntimeAsset("mmproj-F16.gguf", 918_166_080,
                     "f70dc3509053962b0d0d3ee8a7eacebf5d60aa560cad78254ae8698516ae029f"),
    ),
}


def assistant_model_spec(model: str) -> AssistantModelSpec:
    spec = LOCAL_ASSISTANT_MODELS.get(cast(LocalAssistantModel, model))
    if spec is None:
        raise AssistantModelError("Unknown local assistant model. Choose Qwen in Settings > Assistant.")
    return spec


class LocalAssistantModels(Protocol):
    def path(self, model: str) -> Path: ...
    def projector_path(self, model: str) -> Path: ...
    def installed(self) -> dict[str, int]: ...
    def pull(self, model: LocalAssistantModel, on_progress: Callable[[int, int], None]) -> None: ...
    def delete(self, model: LocalAssistantModel) -> None: ...


class FileAssistantModels:
    def __init__(self, models_dir: Callable[[], Path], downloader: ModelDownloader, runtime: LlamaRuntime) -> None:
        self._models_dir, self._downloader, self._runtime = models_dir, downloader, runtime

    def path(self, model: str) -> Path:
        return self._models_dir() / "assistant" / assistant_model_spec(model).weights.filename

    def projector_path(self, model: str) -> Path:
        spec = assistant_model_spec(model)
        # Qwen repositories share the projector filename, so each projector sits in its repository's folder.
        return self.path(model).parent / spec.repo.split("/")[-1] / spec.projector.filename

    def delete(self, model: LocalAssistantModel) -> None:
        try:
            for path in (self.path(model), self.projector_path(model)):
                path.unlink(missing_ok=True)
                path.with_suffix(".verified").unlink(missing_ok=True)
        except OSError as exc:
            raise AssistantModelError("Could not delete the assistant model. Check file permissions and retry.") from exc

    @staticmethod
    def _receipt(path: Path, spec: RuntimeAsset) -> str:
        stat = path.stat()
        return f"{spec.sha256}:{stat.st_size}:{stat.st_mtime_ns}"

    def _verified(self, path: Path, spec: RuntimeAsset) -> bool:
        try:
            return path.stat().st_size == spec.size and path.with_suffix(".verified").read_text() == self._receipt(path, spec)
        except OSError:
            return False

    def installed(self) -> dict[str, int]:
        self._runtime.check_platform()
        if not self._runtime.installed():
            return {}
        return {model: spec.size for model, spec in LOCAL_ASSISTANT_MODELS.items()
                if self._verified(self.path(model), spec.weights)
                and self._verified(self.projector_path(model), spec.projector)}

    def pull(self, model: LocalAssistantModel, on_progress: Callable[[int, int], None]) -> None:
        spec = LOCAL_ASSISTANT_MODELS[model]
        runtime_size = self._runtime.download_size
        total = spec.size + runtime_size
        try:
            self._runtime.install(lambda done: on_progress(done, total))
            completed = runtime_size
            for path, asset in ((self.path(model), spec.weights), (self.projector_path(model), spec.projector)):
                if not self._verified(path, asset):
                    self._downloader.download_file(
                        repo_id=spec.repo, filename=asset.filename, local_dir=str(path.parent), token=None,
                        on_progress=lambda done: on_progress(completed + done, total),
                    )
                    with path.open("rb") as file:
                        digest = hashlib.file_digest(file, "sha256").hexdigest()
                    if path.stat().st_size != asset.size or digest != asset.sha256:
                        path.unlink(missing_ok=True)
                        raise AssistantModelError("Model verification failed. Retry the download.")
                    receipt = path.with_suffix(".verified.tmp")
                    receipt.write_text(self._receipt(path, asset))
                    receipt.replace(path.with_suffix(".verified"))
                completed += asset.size
                on_progress(completed, total)
        except AssistantModelError:
            raise
        except Exception as exc:
            raise AssistantModelError("Could not download the assistant model. Check your connection and disk space, then retry.") from exc
