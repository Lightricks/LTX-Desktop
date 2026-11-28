from threading import RLock

from _routes._errors import HTTPError
from api_types import AssistantModelDownloadProgress, AssistantModelItem, AssistantModelsResponse, LocalAssistantModel
from handlers.generation_handler import GenerationHandler
from services.assistant_models import AssistantModelError, LOCAL_ASSISTANT_MODELS, LocalAssistantModels
from services.interfaces import TaskRunner
from state.app_state_types import AppState


class AssistantModelsHandler:
    def __init__(self, *, state: AppState, lock: RLock, models: LocalAssistantModels, task_runner: TaskRunner, generation: GenerationHandler) -> None:
        self._state, self._lock, self._models, self._tasks = state, lock, models, task_runner
        self._generation = generation
        self._deleting = False

    def _download_progress(self) -> AssistantModelDownloadProgress | None:
        download = self._state.assistant_model_download
        return download.model_copy() if download else None

    def list_models(self) -> AssistantModelsResponse:
        error = None
        try:
            installed = self._models.installed()
        except AssistantModelError as exc:
            installed = {}
            error = str(exc)
        with self._lock:
            snapshot = self._download_progress()
        return AssistantModelsResponse(
            available=error is None, error=error, download=snapshot,
            models=[AssistantModelItem(model=model, name=spec.name, size_bytes=installed.get(model, spec.size), installed=model in installed)
                    for model, spec in LOCAL_ASSISTANT_MODELS.items()],
        )

    def require_installed(self, model: str) -> None:
        try:
            installed = self._models.installed()
        except AssistantModelError as exc:
            raise HTTPError(503, str(exc), code="ASSISTANT_LOCAL_UNAVAILABLE") from exc
        if model not in installed:
            raise HTTPError(409, "Download this assistant model in Settings > Models before selecting it.", code="ASSISTANT_LOCAL_MODEL_MISSING")

    def delete_model(self, model: LocalAssistantModel) -> None:
        if not self._generation.try_reserve_generation_start():
            raise HTTPError(409, "Wait for the current video or assistant step to finish before deleting a model.")
        try:
            with self._lock:
                progress = self._download_progress()
                if self._deleting or (progress and progress.status == "downloading"):
                    raise HTTPError(409, "Wait for model management to finish before deleting a model.")
                self._deleting = True
            try:
                self._models.delete(model)
            except AssistantModelError as exc:
                raise HTTPError(500, str(exc)) from exc
            finally:
                with self._lock:
                    self._deleting = False
                    if self._state.assistant_model_download and self._state.assistant_model_download.model == model:
                        self._state.assistant_model_download = None
        finally:
            self._generation.release_generation_start_reservation()

    def start_download(self, model: LocalAssistantModel) -> None:
        try:
            installed = self._models.installed()
        except AssistantModelError as exc:
            raise HTTPError(503, str(exc), code="ASSISTANT_LOCAL_UNAVAILABLE") from exc
        with self._lock:
            current = self._download_progress()
            if self._deleting or (current and current.status == "downloading"):
                raise HTTPError(409, "DOWNLOAD_ALREADY_RUNNING")
            if model in installed:
                return
            self._state.assistant_model_download = AssistantModelDownloadProgress(
                model=model, status="downloading",
            )

        def progress(downloaded: int, total: int) -> None:
            with self._lock:
                current = self._state.assistant_model_download
                if current and current.status == "downloading":
                    current.downloaded_bytes = downloaded
                    current.expected_bytes = total
                    current.progress = min(99, downloaded / total * 100) if total else 0

        def failed(exc: Exception) -> None:
            with self._lock:
                self._state.assistant_model_download = AssistantModelDownloadProgress(
                    model=model, status="error",
                    error=str(exc) if isinstance(exc, AssistantModelError) else "Download failed. Retry to resume it.",
                )

        def worker() -> None:
            self._models.pull(model, progress)
            self.require_installed(model)
            with self._lock:
                self._state.assistant_model_download = AssistantModelDownloadProgress(model=model, status="complete", progress=100)

        try:
            self._tasks.run_background(worker, task_name="assistant-model-download", on_error=failed, daemon=True)
        except HTTPError as exc:
            failed(exc)
            raise
        except Exception as exc:
            failed(exc)
            raise HTTPError(500, "Could not start the model download.") from exc
